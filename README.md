# Segmentation Methods Benchmark

This project was heavily inspired by the works of *Alaudah et al.* in '**A Machine Learning Benchmark for Facies Classification**'  [[Paper]](https://arxiv.org/abs/1901.07659)[[Code]](https://github.com/yalaudah/facies_classification_benchmark).

Our goal is to propose a benchmark method for comparing the results of different machine learning approaches in the task of seismic facies segmentation. We provide three baseline encoder-decoder models (SegNet, U-Net and DeconvNet) as well as three open-source labeled datasets (F3 Netherlands, New Zeland Parihaka and Nova Scotia Penobscot).

## Instaling the Required Packages

The code was tested using Python 3.10. To install all the depedencies required, execute the following command within the main project folder:

```
pip install -r requirements.txt
```

## Downloading the Data

All three of the datasets listed above were compressed in a single `.zip` file, which can be obtained by running the following commands:

```
pip install gdown
gdown 163g5dKYmrDLHVrp84h-6_7hDpVRQ8zMY
```

> Note: folder names, as well as file names whithin the folders, should be kept the same after unziping in order for the code to work properly.

## Command Line Arguments

To train a model, simply run a command like the example:

```
python3 segment_seismic.py --architecture unet --data_path path/to/F3_alaudah
```

While ``architecture`` and ``data_path`` are the only required arguments, there are also several optional arguments to specify hyperparameters. The complete list of arguments, as well as their descriptions and default values, is given below:

|Argument|Description|Default|
|-|-|-|
|`--architecture`|Choose a model to train with. Options are `segnet`, `unet` and `deconvnet`.||
|`--data_path`|Path to the folder containing one of the datasets with its respective labels in `.npy` format.||
|`--batch_size`|Size of the training batch.|`16`|
|`--device`|Choose on which GPU to train on. Defaults to the CPU if the device isn't available.|`cuda:0`|
|`--loss_function`|Loss function to use. Currently limited to `cel` (Cross Entropy Loss).|`cel`|
|`--optimizer`|Optimizer to use. Options are `adam` and `sgd` (Stochastic Gradient Descent).|`adam`|
|`--learning_rate`|Learning rate to use during training.|`0.01`|
|`--class_weights`|Whether to use class weights in the loss function. The weights are used to assign a higher penalty to misclassifications of minority classes.|`False`|
|`--n_epochs`|Number of epochs during training. The actual number might be lower since early stopping is on by default.|`50`|
|`--orientation`|Choose an orientation for slices of the seismic cube to be sampled. Options are `in` for inlines and `cross` for crosslines.|`in`|
|`--test_ratio`|Percentage (from 0 to 1) of the data used for testing the model. The test set is currently being used for validation in a 5-fold cross validation.|`0.2`|
|`--store_results`|Whether to store training results. This generates a `.pt` file containing the model weights and a `.json` file containing metrics (i. e. class accuracy, mIoU).|`True`|
|`--results_path`|Directory for storing training results (if `store_results` is set to `True`).|`results`|

