# SQuAD 1.1 source code update

Original source code which is used to fine-tune BERT model for SQuAD dataset was prepared in tensorflow version 1.11.0. To start the training process locally using GPU, the Compute Unified Device Architecture (CUDA) is needed. Most of the GPUs do not support CUDA version which is required for older versions of tensorflow (1.11.0) anymore and  to get the best possible performance from GPU during the training process, there is a recommendation to get up to date version of CUDA, which is supported by used GPU. That is the reason to update the source code and use tensorflow version 2.0 or higher.

## Update code
### 1. Use command
There is an official way to update code from tensorflow version 1.x to 2.x provided by Tensorflow directly.    
*Source*: https://www.tensorflow.org/guide/migrate/upgrade   
*Command:*
```
tf_upgrade_v2 \  
--intree <path_current_version> \
--outtree <path_updated_version> \
--reportfile report.txt
```
### 2. Complete code update manually
Additional changes are inevitable because of specific source code structure and functions which are not transformed automatically in previous step.

*Replacements in* **run_squad.py**:
```
slim.tpu -> tf.compat.v1.estimator.tpu
```
```
tf.flags -> tf.compat.v1.flags
```
```
tf.io.gfile.Open -> tf.io.gfile.GFile
```
```
slim.data.map_and_batch -> tf.data.experimental.map_and_batch
```

*Replacements in* **modelling.py**:   
Replace the whole function *layer_norm()*.
```
def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
  return layer_norma(input_tensor)
```