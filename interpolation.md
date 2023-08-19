# Data Availability Statement:
Bangladesh train and test data and indian dataset of brick kilns are provided in `data folder` and all the saved models are in `models folder`.

Data folder consist of `final_train.hdf5`, `final_val.hdf5` and `Indian_dataset.h5` files.
Each file contains two keys, `image` and `label`. 
* `image`: data has the shape `(x, 256, 256, 3)`, where `x` is the number of datapoints/images, 3 is the number of channels (red, green, blue, respectively), and 256 x 256 is the image size. 
* `label`: key has shape `(x,)`, i.e. the label for each datapoint (0 for `nokiln`, 1 for `yeskiln`).

## Bangladesh Data:
We provide HDF5 files for the train, and test dataset: `final_train.hdf5`, and `final_val.hdf5`. 

Data composition: 
* `final_train.hdf5` : Contains 2186 images. 269 of them are positive (contains kilns) and 1917 are negative (does not contain any kiln).

* `final_val.hdf5`: Contains 618 images. 73 of them are positive, and 545 are negative.

We converted all the images to (224,224,3) before passing to models.

## Indian Data:
We have provided `Indian_dataset.h5`.
* `Indian_dataset.h5` : Contains 762 images are positive(contains Brick kilns-labels_1) and 2000 images are negative (contains no Brick kilns-label_0) and shape of each images is (256,256,3).

We converted all this image to (224,224,3) before to passing to models.

--------------
* We use `final_train.hdf5` for train and validation using stratify split of 80:20. 


* Train set contains 1748 images: 215 of them are positive (contains kilns) and 1533 are negative (does not contain any kiln) 
* Validation set contains 438 images: 384 of them are positive (contains kilns) and 54 are negative (does not contain any kiln).  

# Overview of ML Pipeline
We have two experimental setup:
1. Fine tune on Bangladesh dataset and test on Bangladesh test data and test on Indian test data.
    1. Valnila (standard)
    2. Augmentation setting

2. Fine tune on Indian dataset and test on Indian dataset. Here we have standard setting.

## 1. Fine tune on Bangladesh dataset and test on Bangladesh test data and test on Indian data.
1. Training on bangladesh `final_train.hdf5` using train-validation using stratify split to keep the ratio of class 0 and class 1 same across train and validation set. The we train the model performing early stop on validation loss to prevent from overfitting.
2. Testing on bangladesh test data `final_val.hdf5` 
3. Testing on Indian test dataset `Indian_dataset.h5`.

We do this experiment in two setting:
1. Vanilla (standard)
2. Augmentation: Here we apply rotation, horizontal flip, vertical flip, width shift, height shift, augmentation on train data to generate 5 times training data and only rescale on validation and test data.

## 2. Fine tune on Indian dataset and test on Indian dataset. 
Here we have Vanilla(standard) setting.

1. Training: We use 80% Indian data in `Indian_dataset.h5` and make train and validation set using stratify split to keep the ratio of class 0 and class 1 same across train and validation set. Then we train the model performing early stop on validation loss to prevent from overfitting.
2. Testing: We test on remaining 20% Indian data.


# Models:
We have used four models:
1. VGG16
2. ResNet50
3. DenseNet121
4. EfficientNetB0

Each model takes input of shape `x_train` of shape (x,224,224,3) and `y_train` as of shape (x,) and output of shape (x,1). We have used binary cross entropy loss and Adam optimizer with default learning rate 0.00002. We have used early stopping on validation loss to prevent from overfitting. Initial weights for model are of `imagenet` excluding the fully connected top layers of each model. We fine tune the model on our training dataset. Then we apply AveragePooling2D, Flatten layer and then fully connected layer with sigmoid activation function at last for binary classification to get the output of shape (x,1). We have used `sklearn` to calculate the metrics. 




