class CloudSeg(object):
	def __init__(self):
		self.ENCODER = 'resnet50'
		self.ENCODER_WEIGHTS = 'imagenet'
		self.DEVICE = 'cuda'

		self.ACTIVATION = None
		
		self.model = smp.Unet(
		    encoder_name=ENCODER, 
		    encoder_weights=ENCODER_WEIGHTS, 
		    classes=4, 
		    activation=ACTIVATION,
		)
		self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
		
		self.num_workers = 0
		self.bs = 16
		self.train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
		self.valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

		self.train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
		self.valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

		self.loaders = {
		    "train": train_loader,
		    "valid": valid_loader
		}
		
		pass

	def train(self, plot_training_metrics = False):
		num_epochs = 19
		self.logdir = "./logs/segmentation"

		# model, criterion, optimizer
		optimizer = torch.optim.Adam([
		    {'params': self.model.decoder.parameters(), 'lr': 1e-2}, 
		    {'params': self.model.encoder.parameters(), 'lr': 1e-3},  
		])
		scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
		criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
		self.runner = SupervisedRunner()
		
		##################################
		# Model training: 
		##################################
		self.runner.train(
		    model=self.model,
		    criterion=criterion,
		    optimizer=optimizer,
		    scheduler=scheduler,
		    loaders=self.loaders,
		    callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
		    logdir=logdir,
		    num_epochs=num_epochs,
		    verbose=True
		)
		
		if plot_training_metrics:
			utils.plot_metrics(
			    logdir=self.logdir, 
			    # specify which metrics we want to plot
			    metrics=["loss", "dice", 'lr', '_base/lr']
			)
		
		pass

	def select_param(self, plot_samples=False):
		##################################
		# Get the optimal parameter for each cloud class:
		#
		# INPUTS: runner.
		# OUTPUTS: class_params
		##################################
		valid_dataset = self.valid_dataset
		valid_loader = self.valid_loader
		loader = self.loader
		runner = self.runner
		model = self.model
		logdir = self.logdir

		# Step 1 of 3: Get the probability 
		# NOTE: What does the "probability" do?
		encoded_pixels = []
		loaders = {"infer": valid_loader}
		runner.infer(
		    model=model,
		    loaders=loaders,
		    callbacks=[
			CheckpointCallback(
			    resume=f"{logdir}/checkpoints/best.pth"),
			InferCallback()
		    ],
		)

		valid_masks = []
		probabilities = np.zeros((2220, 350, 525))
		for i, (batch, output) in enumerate(tqdm.tqdm(zip(
			valid_dataset, runner.callbacks[0].predictions["logits"]))):
		    image, mask = batch
		    for m in mask:
			if m.shape != (350, 525):
			    m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
			valid_masks.append(m)

		    for j, probability in enumerate(output):
			if probability.shape != (350, 525):
			    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
			probabilities[i * 4 + j, :, :] = probability

		# Step 2 of 3: Optimal threshold in "class_params": 
		class_params = {}
		for class_id in range(4):
		    print(class_id)
		    attempts = []
		    for t in range(0, 100, 5):
			t /= 100
			for ms in [0, 100, 1200, 5000, 10000]:
			    masks = []
			    for i in range(class_id, len(probabilities), 4):
				probability = probabilities[i]
				predict, num_predict = post_process(sigmoid(probability), t, ms)
				masks.append(predict)

			    d = []
			    for i, j in zip(masks, valid_masks[class_id::4]):
				if (i.sum() == 0) & (j.sum() == 0):
				    d.append(1)
				else:
				    d.append(dice(i, j))

			    attempts.append((t, ms, np.mean(d)))

		    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
		    attempts_df = attempts_df.sort_values('dice', ascending=False)
		    print(attempts_df.head())

		    best_threshold = attempts_df['threshold'].values[0]
		    best_size = attempts_df['size'].values[0]
		    class_params[class_id] = (best_threshold, best_size)
			
		self.class_param = class_param


		# Step 3 of 3: Visualize some masks:
		if plot_samples: 
			for i, (input, output) in enumerate(zip(
				valid_dataset, runner.callbacks[0].predictions["logits"])):
			    image, mask = input

			    image_vis = image.transpose(1, 2, 0)
			    mask = mask.astype('uint8').transpose(1, 2, 0)
			    pr_mask = np.zeros((350, 525, 4))
			    for j in range(4):
				probability = cv2.resize(output.transpose(1, 2, 0)[:, :, j], dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
				pr_mask[:, :, j], _ = post_process(sigmoid(probability), class_params[j][0], class_params[j][1])
			    #pr_mask = (sigmoid(output) > best_threshold).astype('uint8').transpose(1, 2, 0)


			    visualize_with_raw(image=image_vis, mask=pr_mask, original_image=image_vis, original_mask=mask, raw_image=image_vis, raw_mask=output.transpose(1, 2, 0))

			    if i >= 2:
				break
		pass

	def predict(self, plot_samples=False):
		##################################
		# Make predictions:
		#
		# INPUTS: runner, class_params
		##################################
		import gc
		torch.cuda.empty_cache()
		gc.collect()
		
		runner = self.runner

		# Load testing data: 
		test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
		test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

		loaders = {"test": test_loader}

		# Visualize some predictions: 
		encoded_pixels = []
		image_id = 0
		for i, test_batch in enumerate(tqdm.tqdm(loaders['test'])):
		    runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
		    for i, batch in enumerate(runner_out):
			for probability in batch:

			    probability = probability.cpu().detach().numpy()
			    if probability.shape != (350, 525):
				probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
			    predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
			    if num_predict == 0:
				encoded_pixels.append('')
			    else:
				r = mask2rle(predict)
				encoded_pixels.append(r)
			    image_id += 1
		pass


if __name__ == '__main__':
	pass
