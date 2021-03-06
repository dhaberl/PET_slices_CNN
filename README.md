# 2D Convolutional Neural Network in Keras/Tensorflow
Classification CNN to predict colorectal cancer on PET slices in the transverse plane using a custom Data Generator allowing real-time data feeding and data augmentation. This repo intends to evaluate whether image data augmentation can improve the prediction of colorectal cancer using CNNs.

## Prerequisites
- Linux or Windows 
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3
- env_PET_slices_CNN.yml

## Getting Started
### Branches
- master - First, a CNN is trained and tested on the original images only acting as a baseline. Then, a new model is trained using image augmentations with handcrafted (empirical) augmentation parameters. 
- BayesianOptimization - Same setup as master-branch but with a Bayesian optimization in order to evaluate if better parameters can be found compared to the empirical approach.


### Installation
- Clone or download this repo
- Install dependencies (see env_PET_slices_CNN.yml) and set up your environment

### Dataset
A clinical dataset of 54 positive and 50 negative PET scans were used. 2D images were obtained by slicing the PET volumes into axial slices (512x512x1). 

The scans were stored as npy-files. The dataset also contained a csv-file with the ID and the corresponding ground truth label (binary).

The dataset can not be provided.

folder/
- main.py
- DataGenerator.py
- data/
	- img_0.npy
	- ...
	- labels.csv

where labels.csv contains for instance:

ID; Label \
img_0; 1 \
img_1; 1 \
img_2; 0 \
...

### Data Generator
The Data Generator generates the dataset in batches on multiple cores for real-time data feeding to the machine learning model. 

The generator can be used by importing it in the main file:

```
from DataGenerator import DataGenerator
```

Input parameters are:

- data_dir: path to the data directory (string)
- list_ids: list of IDs as shown above (list)
- labels: list of labels as shown above (list)
- batch_size: number of samples that will be propagated through the network (integer)
- dim: dimensions of the data (tuple with intergers). E.g., image with 512x512 pixels => (512, 512)
- n_channels: number of channels (integer). E.g., RGB = 3 channels
- n_classes: number of classes (integer)
- shuffle: whether to shuffle at generation or not (boolean) 
- **da_parameters

### Train and test
Define hyperparameters, e.g.:

```
- num_epochs_no_da = 30
- num_epochs_da = 55
- train_ratio = 0.7
- validation_ratio = 0.15
- test_ratio = 0.15
- batch_size = 32
```

Define augmentation parameters, e.g.:

```
da_parameters = {"width_shift": 20.,
                 "height_shift": 20.,
                 "rotation_range": 20.,
                 "horizontal_flip": 1.,
                 "vertical_flip": 1.,
                 "min_zoom": 0.5,
                 "max_zoom": 1.5,
                 "random_crop_size": 0.85,
                 "random_crop_rate": 1.,
                 "center_crop_size": 0.85,
                 "center_crop_rate": 1.,
                 "gaussian_filter_std": 2.,
                 "gaussian_filter_rate": 1.
                 }
```

Augmentations:
- width_shift: Shifts are randomly sampled from [-width_shift, +width_shift].
- height_shift: Shifts are randomly sampled from [-height_shift, +height_shift].
- rotation_range: Degree range for random rotations. Randomly sampled from [-rotation_range, +rotation_range].
- horizontal_flip: Probability rate for horizontal flips.
- vertical_flip: Probability rate for vertical flips.
- min_zoom: Lower limit for a random zoom.
- max_zoom: Upper limit for a random zoom. The zoom factor is randomly sampled from [min_zoom, max_zoom].
- random_crop_size: Fraction of the total width/height. The final crop is performed by randomly sampling a section from the original image.
- random_crop_rate: Probability rate for random cropping.
- center_crop_size: Fraction of the total width/height. The final crop is based on the center of the image.
- center_crop_rate: Probability rate for centered cropping.
- gaussian_filter_std: Images are blurred by a Gaussian function which is defined by its standard deviation (std). The std is randomly sampled from [0, gaussian_filter_std].
- gaussian_filter_rate: Probability rate for gaussian filtering. 

Run:
```
python main.py -d 'data/' -i 'first_run' -l 'log/' 

-d, --data_dir, type=str, Path of data directory
-i, --id, type=str, Experiment ID; used for naming folders inside the log directory
-l, --log_path, type=str, Path of log directory 
```

Output:

In the specified log directory a subfolder with the name of the experiment ID will be created containing:
- log_file.csv: contains all augmentation parameters and accuracy, sensitivity and specificity
- model_weights.h5: contains weights of the best performing model
- train_vs_valid_no_da.csv: contains epoch, loss, sparse_categorical_accuracy, val_loss, val_sparse_categorical_accuracy of run without data augmentation
- train_vs_valid_da.csv: contains epoch, loss, sparse_categorical_accuracy, val_loss, val_sparse_categorical_accuracy of run with data augmentation

## Acknowledgments
- [1] The organization of the dataset and the code of the Data Generator is based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
