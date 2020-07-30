#### 第一轮超参数
```text
learning_rate:0.001 （动态调整：step_size=100, gamma=0.1）
epoch_number:300
batch_size:32
weight_delay:1e-08
```       
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.1221 train_accuracy_sum:92.0219
 valid_loss_sum:1.5214 valid_accuracy_sum:51.7943
```          

#### 第二轮超参数
```text
learning_rate:0.001 （动态调整：step_size=100, gamma=0.1）
epoch_number:100
batch_size:32
weight_delay:1e-08
```
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.2033 train_accuracy_sum:84.0900
 valid_loss_sum:1.5736 valid_accuracy_sum:45.6512
```

#### 第三轮超参数
```text
learning_rate:0.001 （动态调整：step_size=100, gamma=0.1）
epoch_number:200
batch_size:32
weight_delay:1e-08
```
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.1439 train_accuracy_sum:89.7706
 valid_loss_sum:1.5278 valid_accuracy_sum:50.8872
```

#### 第四轮超参数
```text
learning_rate:0.001 （动态调整：step_size=100, gamma=0.1）
epoch_number:1000
batch_size:32
weight_delay:1e-08
```
```text
train_loss_sum:1.1448 train_accuracy_sum:89.5675
 valid_loss_sum:1.5117 valid_accuracy_sum:52.3525
```

#### 第五轮超参数
```text
learning_rate:0.001 （动态调整：step_size=100, gamma=0.1）
epoch_number:1000
batch_size:32
weight_delay:0
```
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.1604 train_accuracy_sum:88.1126
 valid_loss_sum:1.5319 valid_accuracy_sum:50.1864
```

#### 第六轮超参数(**同样学习率下面的最优值**)
`````text
learning_rate:0.001
epoch_number:1000
batch_size:32
weight_delay:0
`````
```text
########## 最终k折交叉验证结果 ##########
train_loss_sum:1.0619 train_accuracy_sum:98.1415
 valid_loss_sum:1.4909 valid_accuracy_sum:54.8936
```
```text
在此参数下最终的训练结果
########## 最终训练结果 ##########
train_loss:1.0624 train_accuracy:98.0392
提交之后的验证结果
valid_accuracy: 0.45614
```