# SQuAD 1.1 energy consumption

**Programming language:**
Developed and tested using python 3.7.16


## About the BERT fine-tuning code

The fine-tuning scripts are the recipe for fine-tuning Bert developed by google-research, it can be found on github at ``google-research/bert/``. We have modified the original scripts as described in the file ``documentation_tf_update.md`` located in this folder.



## Download the BERT model   
https://github.com/google-research/bert   
*Model specification:* **BERT-Base, Uncased**   

## Download the SQUADv1.1 dataset
https://github.com/google-research/bert
- train-v1.1.json
- dev-v1.1.json
- evaluate-v1.1.py   


## Detail on the creation of the virtual environment and tensorflow installation

Below is the detail on how we have created the conda environment and installed tensorflow at the time.

Source: https://www.tensorflow.org/install/pip

```Shellsession
demo> conda create -n cenv_tf python=3.7

demo> conda activate cenv_tf

demo> conda install -c conda-forge cudatoolkit=11.8.0

demo> pip install nvidia-cudnn-cu11

demo> pip install tensorflow

demo> mkdir -p $CONDA_PREFIX/etc/conda/activate.d

demo> echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

demo> echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

demo> source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Verify install:
```Shellsession
demo> python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```