# Tutorial ResNet18 pre-training on ImageNet (and CUB_200_2011)

**Programming language:** Developed and tested using python 3.10

## About the ResNet18 training code
We are using the official code from PyTorch for training ResNet18 on ImageNet. 

The script is called "Image classification reference training scripts". We will be using the function train.py located at https://github.com/pytorch/vision/blob/main/references/classification/train.py

Usefull links: 
- ResNet page: https://pytorch.org/vision/stable/models/resnet.html  
- ResNet scientific paper: https://arxiv.org/pdf/1512.03385.pdf  
- Resnet18 page: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18 

<!-- You need to download the folder classification/ located on github at: https://github.com/pytorch/vision/blob/main/references/classification/ -->

## Data preparation for the ImageNet dataset

**Step 1**: Download dataset from kaggle
- Download the dataset at https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description. Go to the Data page and download data. It is contained in a folder named ILSVRC in Kaggle. You will obtained an archive named imagenet-object-localisation-challenge.zip **Warning: the archive weights 167.62 GB.**

- Extract the downloaded archive. Only keep the folder: ILSVRC/Data/train/ and name it images/. This folder contains subfolders corresponding to different classes of images. Each subfolder contains 1300 images. 

- Place the folder images/ in a location of your choice path_to_location/. For instance, you may create a folder /data/image_net and place the folder /images inside.

**Step 2**: Prepare the data
```Shellsession
demo> python preprocess_data_imagenet.py --data_path='path_to_location/'
```
The data will be prepared for the training. 

<!-- python preprocess_data_imagenet.py --data_path='/media/demouser/0434B71B34B70F24/image_net' -->


## Usage

**Training:**   
For arguments description including default values use:
```Shellsession
demo> python train.py --help
```   

**Example:**   
```Shellsession
demo> python train.py --epochs=2 --data-path='path_to_location/' --batch-size=16 --workers=8
```
<!-- python train.py --epochs=1 --data-path=/media/demouser/0434B71B34B70F24/image_net --batch-size=16 --workers=8 -->


## Appendix: data preparation for CUB_200_2011 dataset

**Step 1**: Download dataset from kaggle
- Download the dataset at https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images.
- Extract the downloaded archive. It contains two tar subfolders, one of which is named "CUB_200_2011.tgz".
- Place CUB_200_2011.tgz in the folder /energycalculatorsevaluation/data.

**Step 2**: Use *prepare_data()* function from *preprocess_data.py*  
Arguments:  
- ```data_path```: Path to downloaded data.   
- ```archive_name```: Name of downloaded archive   
- ```seed``` default(1234),   
- ```valid_data_amount``` default(0.9): Split data for training and validation.

For instance, open the terminal, go in the folder resnet_classification_pytorch_vision, open a python shell and run the following commands:
```python
from preprocess_data import prepare_data
prepare_data(data_path="../data", archive_name="CUB_200_2011.tgz")
```
The data will be prepared for the training. 