# Seismic Segmentation Methods Benchmark

This project was heavily inspired by the works of *Alaudah et al.* in '**A Machine Learning Benchmark for Facies Classification**'  [[Paper]](https://arxiv.org/abs/1901.07659)[[Code]](https://github.com/yalaudah/facies_classification_benchmark).

Our goal is to propose a benchmark method for comparing the results of different machine learning approaches in the task of seismic facies segmentation. We provide three baseline encoder-decoder models (SegNet, U-Net and DeconvNet) as well as three open-source labeled datasets (F3 Netherlands, New Zeland Parihaka and Nova Scotia Penobscot).

## Instaling the Required Packages

The code was tested using Python 3.10. To install all the depedencies required, execute the following command within the main project folder:

```
pip install -r requirements.txt
```

## Downloading the Data

All three of the datasets listed above were preprocessed and compressed into a single `datasets.tar.xz` file. It can be obtained through the `gdown` package by running the following commands:

```
pip install gdown
gdown 1KtUr3dbcf_BBWKDkRWTil8f89CaW8e9W
```

To use the data, move the `.tar.xz` file to the desired folder and extract it by running the following command:

```
tar -xJf datasets.tar.xz
```

## Command Line Arguments

To train a model, simply run a command like the example:

```
python3 segment_seismic.py -t -a unet -d path/to/data.npy -l path/to/labels.npy
```

To test a previously stored model, simply drop the `-t` flag and specificy the path to a `.pt` file with `-m`:

```
python3 segment_seismic.py -a unet -d path/to/data.npy -l path/to/labels.npy -m path/to/model.pt
```

While `architecture`, `data_path` and `labels_path` are the only required arguments, there are also several optional arguments to specify hyperparameters. The complete list of arguments, as well as their descriptions and default values, is given below:

|Argument|Description|Default|
|-|-|-|
|`-a`, `--architecture`|Choose a model to train with. Options are `segnet`, `unet` and `deconvnet`.||
|`-d`, `--data-path`|Path to the seismic volume file in `.npy` or `.segy` format.||
|`-l`, `--labels-path`|Path to the labels file in `.npy` format.||
|`-t`, `--train`|Whether to train a model from scratch. If left on `False`, the path to a previously stored model must be provided.|`False`|
|`-b`, `--batch-size`|Size of the training batch.|`16`|
|`-D`, `--device`|Choose on which GPU to train on. Defaults to the CPU if the device isn't available.|`cuda:0`|
|`-v`, `--cross-validation`|Whether to train the model using 5-Fold Cross Validation.|`False`|
|`-L`, `--loss-function`|Loss function to use. Currently limited to `cel` (Cross Entropy Loss).|`cel`|
|`-o`, `--optimizer`|Optimizer to use. Options are `adam` and `sgd` (Stochastic Gradient Descent).|`adam`|
|`-r`, `--learning-rate`|Learning rate to use during training.|`1e-4`|
|`-w`, `--weight-decay`|Whether to use weight decay (L2 regularization) in the optimizer to prevent overfitting. A value of `0` indicates no decay.|`1e-5`|
|`-W`, `--weighted-loss`|Whether to use class weights in the loss function. The weights are used to assign a higher penalty to misclassifications of minority classes.|`False`|
|`-e`, `--n-epochs`|Number of epochs during training. The actual number might be lower since early stopping is on by default.|`20`|
|`-O`, `--orientation`|Choose an orientation for slices of the seismic cube to be sampled. Options are `in` for inlines and `cross` for crosslines.|`in`|
|`-f`, `--faulty-slices-list`|Path to a `.json` file with indexes of faulty slices in the data that contain artifacts. These slices will be discarded from the volume before training.|`False`|
|`-F`, `--few-shot`|Whether to swap the train and test sets to train the model using less data.|`False`|
|`-m`, `--model-path`|Path to a `.pt` file containing model weights from a previous training. This model will be ignored if the `--train` flag is set to `True`, but a future version of this code will allow resuming training from a stored model.|`None`|
|`-p`, `--results-path`|Directory for storing results. This will generate a `.json` file containing the metrics and a folder containing the predictions from the test set in `.png` format. If `--train` is `True`, a `.pt` file will also be created to store model weights.|`results`|