# CloudFlower

## 1. Problem Description

## 2. Prerequisites
This project utilizes some computer vision libraries and utilities. In order to run the full work-flow notebook, you should have python 3 and install the following libraries before digging into the notebook:

`!pip install catalyst`

`!pip install pretrainedmodels`

`!pip install git+https://github.com/qubvel/segmentation_models.pytorch`

`!pip install pytorch_toolbelt`

`!pip install torchvision==0.4`

`!pip install albumentations==0.3.2`


## 3. Code structure: 
-- a. download_kaggle_data.ipynb: A notebook to download the dataset from Kaggle to google storage. 
  
  Since I am training my model on google colab, it is much more convenient to have the dataset reside on google storage.

-- b. helper_functions.py: To get image, translate rle coding and the masks, and visualization. 

They form the basic utilities to analyze the cloud image dataset. 

-- c. run_cloud_model.ipynb: A full work-flow from loading the data, training to making inference.

-- d. Convex_Hull_postprocess.ipynb: post-process the segmentation masks (from the model) into more 
"regular" shape. 

-- e. resnet.py: An implementation of resNet (including resNet34 and resNet 50) in pytorch. 

-- f. unet_for_cloud.py: python script corresponding to run_cloud_model.ipynb. 

-- g. CloudSeg.py: Integrading the notebook (run_cloud_model.ipynb) into a compact class for cloud 
segmentation. 

-- h. unet.ipynb: An working progress to make a customized UNet structure where we can add dilation
to "upgrade" the existing UNet model. 

Users are suggested to check the code in the order: **a --> b --> c --> d**. 

**e and h** are part of the efforts for customized extension of UNet. 

**f and g** are working towards a better coding strcuture based on the notebook workflow.

# References
