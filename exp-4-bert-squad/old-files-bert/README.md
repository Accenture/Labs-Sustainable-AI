# SQuAD 1.1 energy consumption

## Download model   
https://github.com/google-research/bert   
*Model specification:* **BERT-Base, Uncased**   

## Download data
https://github.com/google-research/bert
- train-v1.1.json
- dev-v1.1.json
- evaluate-v1.1.py   

## Environment preparation   
```
export BERT_BASE_DIR=/path/to/bert/downloaded_model
export SQUAD_DIR=/path/to/downloaded/data
```
## Installation
To install:
```
pip install requirements.txt
```

## Tensorflow estimator replacement    
Replace **estimator.py** and **tpu_estimator.py** files in created virtual environment directly. 

Updated tensorflow estimator files:
```
tf_updated_files/estimator.py
tf_updated_files/tpu_estimator.py
```

Files to replace:
```
<path_to_venv>/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py
```

```
venv/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py
```

## Modification to Carbon Tracker package

The line
```
devices = [name.decode("utf-8") for name in names]
```
is replaced with
```
devices = [name for name in names]
```
in the following file: /home/demouser/miniconda3/envs/cenv_tf/lib/python3.7/site-packages/carbontracker/components/gpu/nvidia.py

## Usage
**Programming language:**
Developed and tested using python 3.7.16

**RAPL usage:**
```
sudo chmod o+r /sys/class/powercap/intel-rapl\:0/energy_uj
```

**Create output directory:**
```
mkdir output
mkdir output/squad_base
mkdir output/calculator_output
```

## Start fine-tuning BERT 
```
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/output/squad_base/ \
  --calculator=<calculator_name>
```  

#### *Supported calculators*
- code_carbon
- carbon_tracker
- eco2ai
- impact_tracker



## Usage on the Alienware:

``` 
export BERT_BASE_DIR=/home/demouser/Documents/Demos/energycalculatorsevaluation/data/bert/uncased_L-12_H-768_A-12 
export SQUAD_DIR=/home/demouser/Documents/Demos/energycalculatorsevaluation/data/bert/data 
export OUTPUT_ESTIMATOR=./output/calculator_output 
``` 

Example without calculator:
```
python run_squad.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-extracted.json   --do_predict=False   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=12   --learning_rate=3e-5   --num_train_epochs=3.0   --max_seq_length=384   --doc_stride=128   --output_dir=output/squad_base/ 
```

```
python run_squad.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-extracted.json   --do_predict=False   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=12   --learning_rate=3e-5   --num_train_epochs=3.0   --doc_stride=128   --output_dir=output/squad_base/ 
```


Training with whole dataset 3 epochs:
```
python nlp_bert_squad/run_squad.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_predict=True --do_train=True  --train_file=$SQUAD_DIR/train-v1.1.json   --do_predict=False   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=3   --learning_rate=3e-5   --num_train_epochs=3.0   --doc_stride=128   --output_dir=nlp_bert_squad/output/squad_base/
```

Training with smaller dataset 1 epoch:
python nlp_bert_squad/run_squad.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_predict=True --do_train=True  --train_file=$SQUAD_DIR/train-extracted.json   --do_predict=False   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=12   --learning_rate=3e-5   --num_train_epochs=1.0   --doc_stride=128   --output_dir=nlp_bert_squad/output/squad_base/

Inference with smaller dataset (Super_Bowl_50 paragraph):
python nlp_bert_squad/run_squad.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_predict=True --do_train=False  --train_file=$SQUAD_DIR/train-extracted.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=3   --learning_rate=3e-5   --num_train_epochs=1.0   --doc_stride=128   --output_dir=nlp_bert_squad/output/squad_base/



Example with a calculator:

```
python run_squad.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-extracted.json   --do_predict=False   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=12   --learning_rate=3e-5   --num_train_epochs=3.0   --max_seq_length=384   --doc_stride=128   --output_dir=output/squad_base/   --calculator=carbon_tracker 
```





## Create the environment: 

source: https://www.tensorflow.org/install/pip

```
conda create -n cenv_tf python=3.7

conda activate cenv_tf

conda install -c conda-forge cudatoolkit=11.8.0

pip install nvidia-cudnn-cu11

pip install tensorflow

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Verify install:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Change the estimator.py and estimator-tpu.py files at locations:

- cenv_tf/lib/python3.7/site-packages/tensorflow_estimator/python/estimator

- cenv_tf/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu