# Measuring Pre Training Speed of BERT with Multi Gpus and Mixed Precision 

##### Cloned from google-research's bert repository.

#### Modifications :

1. run_pretraining.py : 
    1. Added `strategy`.
    2. `Strategy` can only be passed to `tf.estimator.RunConfig`,
        `tf.estimator.RunConfig` can only be passed to `tf.estimator.Estimator`,
         instead of `tpu.RunConfig`. `tpu.TPUEstimator` respectiely.
     3. Also defined `tf.estimator.EstimatorSpec`, instead of `tpu.TPUEstimatorSpec`.
2. optimization.py :
    1. Adding `optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)`
        enables mixed precision training.
    2. Referenced [guotong1988's BERT-GPU repository](https://github.com/guotong1988/BERT-GPU/).

#### Requirements :
python 3.6
tensorflow-gpu == 1.14.0

#### Usage :

```
python run_pretraining.py \ 
...
--multi_gpu \
--use_fp16
```

#### Experimental settings
bert-model : bert-base
max_sequence_length : 512

#### Results :

##### RTX 2080 Ti

##### V100(16G)

##### I didn't fully trained BERT from scratch. Just measured training speed for about 2k steps. 

**\*\*\*\*\* New March 11th, 2020: Smaller BERT Models \*\*\*\*\***

This is a release of 24 smaller BERT models (English only, uncased, trained with WordPiece masking) referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962).

We have shown that the standard BERT recipe (including model architecture and training objective) is effective on a wide range of model sizes, beyond BERT-Base and BERT-Large. The smaller BERT models are intended for environments with restricted computational resources. They can be fine-tuned in the same manner as the original BERT models. However, they are most effective in the context of knowledge distillation, where the fine-tuning labels are produced by a larger and more accurate teacher.

Our goal is to enable research in institutions with fewer computational resources and encourage the community to seek directions of innovation alternative to increasing model capacity.

You can download all 24 from [here][all], or individually from the table below:

|   |H=128|H=256|H=512|H=768|
|---|:---:|:---:|:---:|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**][2_128]|[2/256][2_256]|[2/512][2_512]|[2/768][2_768]|
| **L=4**  |[4/128][4_128]|[**4/256 (BERT-Mini)**][4_256]|[**4/512 (BERT-Small)**][4_512]|[4/768][4_768]|
| **L=6**  |[6/128][6_128]|[6/256][6_256]|[6/512][6_512]|[6/768][6_768]|
| **L=8**  |[8/128][8_128]|[8/256][8_256]|[**8/512 (BERT-Medium)**][8_512]|[8/768][8_768]|
| **L=10** |[10/128][10_128]|[10/256][10_256]|[10/512][10_512]|[10/768][10_768]|
| **L=12** |[12/128][12_128]|[12/256][12_256]|[12/512][12_512]|[**12/768 (BERT-Base)**][12_768]|

