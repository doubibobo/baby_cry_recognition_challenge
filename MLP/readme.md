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

#### 2020-08-17-01 随机采样，使用dropout
> 原始数据： data.csv
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_03
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0778 train_accuracy_sum:96.9109
 valid_loss_sum:1.4218 valid_accuracy_sum:61.7582
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08

########## 最终训练结果 ##########
train_loss:1.0829 train_accuracy:96.4052
```

#### 2020-08-17-02 随机采样，使用dropout
> 原始数据：test_mfcc_20_new_3s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081701
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.1071 train_accuracy_sum:93.9506
 valid_loss_sum:1.2971 valid_accuracy_sum:74.1910
learning_rate:0.001
epoch_number:1000
batch_size:128
weight_delay:1e-08

########## 最终训练结果 ##########
train_loss:1.1021 train_accuracy:94.2345
```

#### 2020-08-17-03 随机采样，使用dropout
> 原始数据：test_mfcc_20_new_3s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081702  
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.1235 train_accuracy_sum:92.1854
 valid_loss_sum:1.2964 valid_accuracy_sum:74.4834
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08

########## 最终训练结果 ##########
train_loss:1.1303 train_accuracy:91.3323
```

#### 2020-08-17-04 随机采样，使用dropout ------废弃
> 原始数据：train_mfcc_20_new_15s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081703  
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0509 train_accuracy_sum:99.4505
 valid_loss_sum:1.3490 valid_accuracy_sum:69.1209
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08

########## 最终训练结果 ##########
train_loss:1.0488 train_accuracy:99.7821
```

#### 2020-08-18-01 随机采样，使用dropout
> 原始数据：train_mfcc_20_new_15s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081703 
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0704 train_accuracy_sum:97.6313
 valid_loss_sum:1.3670 valid_accuracy_sum:67.4725
learning_rate:0.001
epoch_number:1500
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:1.0694 train_accuracy:97.9303
```

#### 2020-08-19-01 随机采样，使用dropout为0.3，冻结bn层
> train_file: train_mfcc_20_new_3s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081705.pkl
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:0.3707 train_accuracy_sum:87.3641
 valid_loss_sum:0.8656 valid_accuracy_sum:73.4503
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:0.4016 train_accuracy:86.4433
```

#### 2020-08-19-02 随机采样，使用dropout为0.3，冻结bn层，激活函数为tanh，最后一层没用tanh
> train_file: train_mfcc_20_new_3s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081706.pkl
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:0.1205 train_accuracy_sum:96.4977
 valid_loss_sum:0.9310 valid_accuracy_sum:76.0429
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:0.1367 train_accuracy:96.0460
```

#### 2020-08-19-03 随机采样，使用dropout为0.3，冻结bn层，使用tanh激活函数，最后一层使用tanh
> train_file: train_mfcc_20_new_3s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081901.pkl
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:0.6640 train_accuracy_sum:91.4555
 valid_loss_sum:0.9594 valid_accuracy_sum:73.9571
learning_rate:0.001
epoch_number:500
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型
########## 最终训练结果 ##########
train_loss:0.6788 train_accuracy:90.4753
```

#### 2020-08-19-03 随机采样，使用dropout为0.3，冻结bn层，使用tanh激活函数，最后一层使用tanh
> train_file: train_mfcc_20_new_3s  
> FILE_NAME = MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081902.pkl
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:0.6115 train_accuracy_sum:94.6372
 valid_loss_sum:0.9375 valid_accuracy_sum:75.5945
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:1e-08
数据已经转化为gpu类型

########## 最终训练结果 ##########
train_loss:0.6270 train_accuracy:93.7281
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