# RSNA-MICCAI Brain Tumor Radiogenomic Classificaiton (Kaggle) competition Winning Submission
References
* https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification
* https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347
* https://github.com/FirasBaba/rsna-resnet10

## Model Details
* He uses simple model (Resnet10-50) because biggerm models give noisy results
* https://monai.io/ Monai library could be useful for preprocessing stuff
* Ensembling didn’t improve score
* Train 100 models (train model 20 times, 5-fold cross validation, take the 5-fold average). Select top 5 ideas+models and 
* Final model
  * 3D CNN
  * Resnet10
  * BCE loss
  * Adam optimizer
  * 15 epochs
  * LR: epoch 1->10; lr = 0.0001 | epoch 10 to 15 lr=0.00005
  * Image size: 256x256
  * Batch size: 8 (the bigger bs I use the worse CV I get, I was alternating between bs=4 and bs=8)
  * No mixed-precision is used.
  * central image trick
  * One epoch takes around 1minute and 20 seconds using an RTX 3090.
## Central Image Trick
* Small trick to build the 3D images.
* Using the biggest image as a central image (the image that contains the largest brain cutaway view) will slightly improve my local CV (improvements were between 0.01 and 0.02). I think that this was the only 100% successful experiment I did in this competition.
## What did not work
* 2d CNNs
* 4 backbones 3D CNNs (each for one structural multi-parametric MRI folder)
* Ensembling
* Pretrained on brain images
* Using the metadata from the DCIM images
* Stacking the output from the different CNNs using based tree models.
* Deep CNNs
* Some tricks I did to normalize the voxels in some consistent ways.
