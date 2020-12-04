# Measuring Pre Training Speed of BERT with Multi Gpus and Mixed Precision 

#### Cloned from [google-research's bert repository](https://github.com/google-research/bert).

## Modifications :

1. run_pretraining.py : 
    1. Added `strategy` for using multi gpus. It seems that `MultiWorkerMirroredStrategy`(`CollectiveAllReduceStrategy`) 
    enables faster training compared to `MirroredStrategy`.
    2. `Strategy` can only be passed to `tf.estimator.RunConfig`,
        `tf.estimator.RunConfig` can only be passed to `tf.estimator.Estimator`,
         instead of `tpu.RunConfig`. `tpu.TPUEstimator` respectiely.
     3. Also defined `tf.estimator.EstimatorSpec`, instead of `tpu.TPUEstimatorSpec`.
2. optimization.py :
    1. Adding `optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)`
        enables mixed precision training.
    2. For other modifications, referenced [guotong1988's BERT-GPU repository](https://github.com/guotong1988/BERT-GPU/).
3. site-packages/tensorflow/python/training/experimental/loss_scale.py line 311 :

    * I had to change dtype of `self._num_good_steps` into **`dtypes.int32`** from `dtypes.int64`
    as an error occurs while converting tensors for mixed precision training.
      
    * `self._num_good_steps = self._add_weight(
        name='good_steps', dtype=dtypes.int32, initial_value=0)`

## Requirements :
python 3.6

tensorflow-gpu == 1.14.0

## Usage :

```
python run_pretraining.py \ 
...
--do_train
--train_batch_size 8
--multi_gpu \
--use_fp16
```

## Experimental settings
bert-model : bert-base

max_sequence_length : 512

Used `MultiWorkerMirroredStrategy` in default.

## Results :

### RTX 2080 Ti

n_gpu | fp32(batch 4) | fp16(batch 6)
---------|---------------|------------
1        |       29      |       34
2        |       41.2    |       55.6
4        |       57.6    |       76.8

### V100(16G)

Didn't tested with 1, 2 GPUs.

Training with batch size 8 enables V100 to use tensor cores.

#### `MultiWorkerMirroredStrategy`
n_gpu | fp32(batch 4) | fp16(batch 8) | fp16(batch 7) | fp16(batch 4)
-----------|----------|---------------|---------------|--------------|
4          |     112  |     185.6     | 176           | 134

#### `MirroredStrategy`
n_gpu | fp32(batch 4) | fp16(batch 8) | fp16(batch 7) | fp16(batch 4)
-----------|----------|---------------|---------------|--------------|
4          |     95   |     160       | 146           | 110

#### horovod(same as NVIDIA's DeepLearningExamples)
n_gpu | fp32(batch 4) | fp16(batch 4) | fp16(batch 8) | fp16(batch 14) |
-----------|----------|---------------|---------------|---------------|
1          | NAN loss | 33            |     44        |               |
4          | NAN loss | 110           |     160       |     196       |

* Compared to [results of BERT by NVIDIA's DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT), 
my results using `MultiWorkerMirroredStrategy` show better results.
(But it seems to consume little more memory.)
NVIDIA reported 35, 110 examples/sec on 1, 4 GPUs with batch size 4.
Test with `MirroredStrategy`, horovod resulted same speed as NVIDIA's results. 
I think it is due to the batch size. 
Their batch size is 2, 4 for fp32, fp16 respectively.

* With horovod, I could increase batch size up to 14, resulted 196 samples per seconds.
Increasing batch size over 8 failed with both Tensorflow `Strategy`s.
Maybe horovod is efficient in memory.

* And batch size of multiple of 8 doesn't seem to dramatically increase
training speed even though I checked TensorCore was enabled.
It might be due to my vocab size which is not multiple of 8 but I'm not sure.

* **You might face OOM error during training with batch size 8.** 

#### Disclaimer
I didn't fully train BERT from scratch. Just measured training speed for about 2k steps. 
