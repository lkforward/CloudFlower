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
		train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
		valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

		train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
		valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

		self.loaders = {
		    "train": train_loader,
		    "valid": valid_loader
		}
		
		pass

	def train(self, plot_training_metrics = False):
		num_epochs = 19
		logdir = "./logs/segmentation"

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
			    logdir=logdir, 
			    # specify which metrics we want to plot
			    metrics=["loss", "dice", 'lr', '_base/lr']
			)
		
		pass

	def select_param(self, plot_samples=False):
		pass

	def predict(self, plot_samples=False):
		pass


if __name__ == '__main__':
	pass
