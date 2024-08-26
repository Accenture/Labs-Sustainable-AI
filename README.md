# Experimental protocol

By means of experiments, we compare a set of energy consumption evaluation tools and methods on different ML computing tasks: the training or fine-tuning of ML models for computer vision and Natural Language Programming (NLP). 

In the different ML contexts, we observe the relative energy consumption evaluation provided by these tools and methods (also compared to an external power meter), as they belong to different approaches: on-chip sensors, mixed on-chip sensors and analytical estimation model, and two different types of analytical estimation models.

We have chosen 4 different ML computing tasks:
- **Training an image classifier on the MNIST dataset.** Our reference training script is the PyTorch example ``Basic MNIST Example'' ([https://pytorch.org/examples/](https://pytorch.org/examples/)), for image classification using ConvNets, available on GitHub in the repository [pytorch/examples/tree/main/mnist](https://github.com/pytorch/examples/tree/main/mnist).

- **Training an image classifier on the CIFAR10 dataset.** Our reference training script is the PyTorch tutorial \``Training a classifier'', part of ``Deep Learning with PyTorch: A 60 Minute Blitz,'' available on the pytorch website at [tutorials/beginner/blitz/](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

- **Training Resnet18 on the ImageNet dataset.** Our reference training script is the recipe for training Resnet18 on ImageNet, provided by PyTorch. The corresponding code is available in the repository [pytorch/vision/references/classification/](https://github.com/pytorch/vision/tree/732551701c326b8338887a3812d189c845ff28a5/references/classification).

- **Fine-tuning Bert-base on the SQUADv1.1 dataset.** Our reference training script is the recipe for fine-tuning Bert-base (uncased) on the dataset SQUADv1-1, provided by google-research. It is available on GitHub in the repository [google-research/bert/](https://github.com/google-research/bert).


## 1. Installation

Clone this repository and create a folder ``data`` within the cloned repository (the datasets for MNIST and CIFAR10 will be automatically downloaded in this folder when running the experiments). 

Follow the requirements detailed in the next three sections.

Instructions to download and prepare the datasets ImageNet, CUB_200_2011 and SQUAD-v1-1 are provided in corresponding readme files:
- ``exp-3-resnet18/README-ResNet18.md`` (ImageNet, CUB_200_2011)
- ``exp-4-resnet18/README-Bert-SQUAD.md`` (SQUAD-v1-1)

## 2. Requirements for operating system and hardware

A Linux OS is needed to realize these tests. This is mainly due to the fact that the experiments are automatically launched by means of the sh file ``experiment.sh``. On the other side, the evaluation tool Carbon Tracker is only compatible with linux.

These tests have been realized with a desktop computer with an i9-9900K Intel CPU and two GeForce RTX 2080 Super Nvidia GPUs (though only one GPU has been used during training).

**Additonal hardware used:**   
For external measurements, we are using the smart plug Tapo P110 from TP-Link.

## 3. Python version and Python packages

### 3.1. Programming language
Developed and tested using python 3.10 and python 3.7.

### 3.2. Creating the virtual environment
Create and activate a virtual environment
- with pip: 
```Shellsession
demo> python -m venv venv_name       # create the virtual environment
demo> source venv_name/bin/activate  # activate the virtual environment (linux, mac)
```
another way of creating the virtual environment is `virtualenv venv_name`;
- with conda: 
```Shellsession
demo> conda create --name venv_name   # create the virtual environment
demo> conda activate venv_name        # activate the virtual environment
```
Source: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html.

### 3.3. Packages

**For the vision experiments:**   
For the experiences 1, 2 and 3, we need to install the packages torch and torchvision for CPU+GPU (i.e., with cuda). We have used a pip virtual environment for the former with Python 3.10.12.

*PyTorch:*  
Depending on your OS, use the installation commands provided at https://pytorch.org/. For instance we have used (for linux+pip+cuda): 
```Shellsession
demo> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu116
```

**For the NLP experiements:**   
For the experience 4, we need to install tensorflow. We have used a conda virtual environment for the latter with Python 3.7.16.

*TensorFlow:*   
We have installed the following packages: tensorflow, tf_slim. To install tensorflow, we have used pip, following the instruction of https://www.tensorflow.org/install/pip. Our installation process for tensorflow is described in detail in the file ``exp-4-bert-squad/README-Bert-SQUAD.md``, however one may now just use the command
```Shellsession
demo> pip install tensorflow[and-cuda]
```

*Modification of two TensorFlow files:*  
Then, the files 'estimator.py' and 'tpu_estimator.py' in the created virtual environment, should be replaced, respectively, with the files 
- ./-4-bert-squad/tf_updated_files/estimator.py
- ./exp-4-bert-squad/tf_updated_files/tpu_estimator.py. 

The files 'estimator.py' and 'tpu_estimator.py' are located in the conda environment folder, in the following respective in the folders 
- <path_to_venv>/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/
- <path_to_venv>/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/.

*Modification of the Carbon Tracker package:*

In the file ``<path_to_venv>/lib/python3.7/site-packages/carbontracker/components/gpu/nvidia.py``, the line
```Python
devices = [name.decode("utf-8") for name in names]
```
is replaced with
```Python
devices = [name for name in names]
```

**For all experiments:**

*Specific to the TapoP110:*   
We need to install a package to communicate with the smart plug Tapo P110: 
```Shellsession
demo> pip install --force-reinstall git+https://github.com/almottier/TapoP100.git@main
```
Previously we were using the origin package ``PyP100`` installed with ``pip install PyP100``, but the current firmware update (of the Tapo) is not compatible with the current version (0.1.2) of this package. The version of PyP100 is now 0.1.4.

*Other packages:* 
Finally, we need to install the following packages (pip install ...):
- carbontracker, codecarbon, eco2ai
- GPUtil
- matplotlib
- numpy
- requests
- tqdm
- pandas
- thop (for flops method)

Detailed version of the packages used in our work can be found in the file ``./requirements/requirements.txt``.

## 4. Providing rights for energy files of the CPU

We need to provide administrator rights (sudo) for several energy evaluation tools.

*For Carbon-Tracker:*

```Shellsession
demo> sudo chmod o+r /sys/class/powercap/intel-rapl\:0/energy_uj
```

*For Code-Carbon:*  

With reboot needed:
```Shellsession
demo> sudo apt install sysfsutils
```
Add this line in /etc/sysfs.conf : ``mode class/powercap/intel-rapl:0/energy_uj = 0444``. Then, reboot.

Without reboot: 
```Shellsession
demo> sudo chmod -R a+r /sys/class/powercap/intel-rapl 
```
However, this change will be lost at next boot.

(Source: https://github.com/mlco2/codecarbon/issues/244)

## 5. Usage

Run the file ``experiment.sh``:
```Shellsession
demo> ./experiment.sh 
```

A new log folder will be created with the results of the experiments.

One can change parameters in the paragraph "SET BY THE USER" at the top of the ``experiment.sh`` file. These parameters include:

**Data paths:**
- BERT_BASE_DIR: path to the **contents** of the folder ``/uncased_L-12_H-768_A-12``
- SQUAD_DIR: path to the folder ``/uncased_L-12_H-768_A-12``
- IMAGE_NET_DIR: path to the folder ``/imagenet``

**List of evaluation tools and methods:**
- 'code_carbon:online'
- 'carbon_tracker:measure'
- 'carbon_tracker:predict'
- 'eco2ai'
- 'green_algorithms:default'
- 'green_algorithms:automated_parallel'
- 'tapo'

**Usage of a GPU:** 'True' or 'False'.

**Number of iterations for the experiments:** integer.

**Choice of ML task:**
- 'idle': no task, corresponds to the idle state of the computer
- 'mnist': training an image classifier on MNIST
- 'cifar10': training an image classifier on CIFAR10
- 'image_net': training ResNet18 on ImageNet
- 'SQUAD-v1-1': fine-tuning Bert-base on SQUAD-v1-1
- 'CUB_200_2011': training ResNet18 on CUB_200_2011 (for tests)
- 'SQUAD-extracted': fine-tuning Bert-base on a subset of SQUAD-v1-1 (for tests)

