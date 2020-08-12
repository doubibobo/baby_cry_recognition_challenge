#### 2020-08-04-01 采用随机取样，不使用dropout
> 原始数据：data_extend.csv  
> FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_bn_01.pkl"

```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0437 train_accuracy_sum:100.0000
 valid_loss_sum:1.3785 valid_accuracy_sum:66.7033
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08

########## 最终训练结果 ##########
train_loss:1.0436 train_accuracy:100.0000
```


#### 2020-08-08-01 采用按比例采样，不使用dropout
> 原始数据: test_mfcc_20_new_3s
> FILE_NAME = MLP_network_bs_32_epoch_1000_lr_small_bn_11
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0628 train_accuracy_sum:98.3219
 valid_loss_sum:1.3028 valid_accuracy_sum:74.1550
learning_rate:0.001
epoch_number:200
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:1.0539 train_accuracy:99.1624
```
#### 2020-08-04-02 采用随机取样，使用dropout
> 原始数据：data_extend.csv
> FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_bn_02.pkl"

```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.1117 train_accuracy_sum:95.8891
 valid_loss_sum:1.4152 valid_accuracy_sum:63.3568
learning_rate:0.001
epoch_number:200
batch_size:32
weight_delay:1e-08

########## 最终训练结果 ##########
train_loss:1.0910 train_accuracy:97.8214
```

#### 2020-08-04-03 采用平均取样，使用dropout
> 原始数据：data_extend.csv  
> FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_bn_03.pkl"
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0532 train_accuracy_sum:99.4191
 valid_loss_sum:1.3781 valid_accuracy_sum:65.7907
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:1.0523 train_accuracy:99.3464
```

#### 2020-08-04-04: 采用平均取样，不使用dropout
> 原始数据：data_extend.csv  
> FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_bn_04.pkl"

```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0437 train_accuracy_sum:100.0000
 valid_loss_sum:1.4006 valid_accuracy_sum:63.2758
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:1.0437 train_accuracy:100.0000
```
#### 2020-08-05-01 采用平均取样，不使用dropout
> 原始数据：data.csv  
> FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_bn_05.pkl"

```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0672 train_accuracy_sum:99.9034
 valid_loss_sum:1.4095 valid_accuracy_sum:67.6064
learning_rate:0.001
epoch_number:100
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:1.0612 train_accuracy:100.0000
```

#### 2020-08-05-02 采用平均取样，不使用dropout
> 原始数据：data.csv  
> FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_bn_06.pkl"

```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.3072 train_accuracy_sum:97.2310
 valid_loss_sum:1.5367 valid_accuracy_sum:64.8360
learning_rate:0.001
epoch_number:100
batch_size:32
weight_delay:1e-08
########## 最终训练结果 ##########
train_loss:1.2006 train_accuracy:98.4749
```

#### 2020-08-05-03 采用按比例取样，不使用dropout
> 原始数据: data_cqt_new.csv
> FILE_NAME = 'MLP_network_bs_32_epoch_1000_lr_small_bn_07.pkl'
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0544 train_accuracy_sum:99.2914
 valid_loss_sum:1.1498 valid_accuracy_sum:89.5185
learning_rate:0.001
epoch_number:100
batch_size:32
weight_delay:0
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:1.0497 train_accuracy:99.6077
```