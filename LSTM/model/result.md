#### 第一次10折交叉验证结果
> batch_size = 512, epoch = 1000
##### 最终结果
train_loss_sum:1.2277 train_accuracy_sum:81.3553  
valid_loss_sum:1.4332 valid_accuracy_sum:60.9890  
##### 每一折的结果
************************* 第 1 折结束 *************************  
train_loss:1.310144 train_accuracy:73.7485  
 valid_loss:1.696510 valid_accuracy:35.1648  
************************* 第 2 折结束 *************************  
train_loss:1.260269 train_accuracy:77.8999  
 valid_loss:1.494688 valid_accuracy:53.8462  
************************* 第 3 折结束 *************************  
train_loss:1.257338 train_accuracy:78.0220  
 valid_loss:1.522062 valid_accuracy:51.6484  
************************* 第 4 折结束 *************************  
train_loss:1.295114 train_accuracy:74.8474  
 valid_loss:1.429790 valid_accuracy:61.5385  
************************* 第 5 折结束 *************************  
train_loss:1.220716 train_accuracy:81.8071  
 valid_loss:1.352357 valid_accuracy:69.2308  
************************* 第 6 折结束 *************************  
train_loss:1.188766 train_accuracy:84.9817  
 valid_loss:1.343581 valid_accuracy:70.3297  
************************* 第 7 折结束 *************************  
train_loss:1.248715 train_accuracy:79.8535  
 valid_loss:1.302917 valid_accuracy:74.7253  
************************* 第 8 折结束 *************************  
train_loss:1.159524 train_accuracy:88.0342  
 valid_loss:1.412549 valid_accuracy:62.6374  
************************* 第 9 折结束 *************************  
train_loss:1.184312 train_accuracy:85.5922  
 valid_loss:1.439101 valid_accuracy:60.4396  
************************* 第 10 折结束 *************************  
train_loss:1.152047 train_accuracy:88.7668  
 valid_loss:1.338548 valid_accuracy:70.3297  
 
 
#### 第二次10折交叉验证结果
> batch_size = 512, epoch = 10000
##### 最终结果
train_loss_sum:1.1649 train_accuracy_sum:87.6190  
 valid_loss_sum:1.4735 valid_accuracy_sum:56.5934  
##### 每一折的结果

************************* 第 1 折结束 *************************  
train_loss:1.425820 train_accuracy:61.6606  
 valid_loss:1.724975 valid_accuracy:31.8681  
************************* 第 2 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 2 折结束 *************************  
train_loss:1.349876 train_accuracy:69.3529  
 valid_loss:1.727717 valid_accuracy:30.7692  
************************* 第 3 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 3 折结束 *************************  
train_loss:1.161003 train_accuracy:87.7900  
 valid_loss:1.691985 valid_accuracy:32.9670  
************************* 第 4 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 4 折结束 *************************  
train_loss:1.135185 train_accuracy:90.4762  
 valid_loss:1.538069 valid_accuracy:50.5495  
************************* 第 5 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 5 折结束 *************************  
train_loss:1.121788 train_accuracy:91.8193  
 valid_loss:1.412958 valid_accuracy:62.6374  
************************* 第 6 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 6 折结束 *************************  
train_loss:1.117873 train_accuracy:92.3077  
 valid_loss:1.443857 valid_accuracy:58.2418  
************************* 第 7 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 7 折结束 *************************  
train_loss:1.102858 train_accuracy:93.7729  
 valid_loss:1.366585 valid_accuracy:67.0330  
************************* 第 8 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 8 折结束 *************************  
train_loss:1.081751 train_accuracy:95.9707  
 valid_loss:1.331737 valid_accuracy:70.3297  
************************* 第 9 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 9 折结束 *************************  
train_loss:1.076995 train_accuracy:96.4591  
 valid_loss:1.271090 valid_accuracy:79.1209  
************************* 第 10 折开始 *************************  
数据已经转化为gpu类型  
************************* 第 10 折结束 *************************  
train_loss:1.075728 train_accuracy:96.5812  
 valid_loss:1.226313 valid_accuracy:82.4176  
 
 
#### 第三次10折交叉验证结果
> batch_size = 512, epoch = 1000
##### 最终结果
train_loss_sum:1.1166 train_accuracy_sum:92.6252
 valid_loss_sum:1.3576 valid_accuracy_sum:68.1319
##### 每一折的结果
```text
LSTMClassify(
  (rnn): LSTM(26, 64, batch_first=True)
  (fc1): Linear(in_features=64, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=6, bias=True)

Model LSTMClassify : params: 0.326936M   
```
************************* 第 1 折结束 *************************  
train_loss:1.183503 train_accuracy:85.5922  
 valid_loss:1.686093 valid_accuracy:35.1648  
************************* 第 2 折结束 *************************  
train_loss:1.122839 train_accuracy:91.9414  
 valid_loss:1.530834 valid_accuracy:49.4505  
************************* 第 3 折结束 *************************  
train_loss:1.113583 train_accuracy:92.9182  
 valid_loss:1.460405 valid_accuracy:57.1429  
************************* 第 4 折结束 *************************  
train_loss:1.089576 train_accuracy:95.3602  
 valid_loss:1.341339 valid_accuracy:70.3297  
************************* 第 5 折结束 *************************  
train_loss:1.113866 train_accuracy:92.9182  
 valid_loss:1.303732 valid_accuracy:73.6264  
************************* 第 6 折结束 *************************  
train_loss:1.103325 train_accuracy:94.0171  
 valid_loss:1.335770 valid_accuracy:71.4286   
************************* 第 7 折结束 *************************  
train_loss:1.111968 train_accuracy:93.1624  
 valid_loss:1.257012 valid_accuracy:78.0220  
************************* 第 8 折结束 *************************  
train_loss:1.113189 train_accuracy:93.0403  
 valid_loss:1.243037 valid_accuracy:80.2198  
************************* 第 9 折结束 *************************  
train_loss:1.105863 train_accuracy:93.7729  
 valid_loss:1.228762 valid_accuracy:81.3187  
************************* 第 10 折开始 *************************  
train_loss:1.108238 train_accuracy:93.5287  
 valid_loss:1.189356 valid_accuracy:84.6154  

#### 第四次10折交叉验证结果
> batch_size = 512, epoch = 1000
##### 最终结果
train_loss_sum:1.0737 train_accuracy_sum:96.9841
 valid_loss_sum:1.2244 valid_accuracy_sum:81.8681
##### 每一折的结果
```text
LSTMClassify(
  (rnn): LSTM(26, 64, batch_first=True)
  (fc1): Linear(in_features=96128, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=6, bias=True)
)
Model LSTMClassify : params: 98.696472M
```
************************* 第 1 折结束 *************************  
train_loss:1.045965 train_accuracy:99.7558  
 valid_loss:1.716107 valid_accuracy:31.8681  
************************* 第 2 折结束 *************************  
train_loss:1.071675 train_accuracy:97.1917  
 valid_loss:1.505868 valid_accuracy:53.8462  
************************* 第 3 折结束 *************************  
train_loss:1.077780 train_accuracy:96.5812  
 valid_loss:1.141869 valid_accuracy:90.1099  
************************* 第 4 折结束 *************************  
train_loss:1.075338 train_accuracy:96.8254  
 valid_loss:1.110058 valid_accuracy:93.4066  
************************* 第 5 折结束 *************************  
train_loss:1.080222 train_accuracy:96.3370  
 valid_loss:1.297478 valid_accuracy:74.7253  
************************* 第 6 折结束 *************************  
train_loss:1.087548 train_accuracy:95.6044  
 valid_loss:1.120459 valid_accuracy:92.3077  
************************* 第 7 折结束 *************************  
train_loss:1.091211 train_accuracy:95.2381  
 valid_loss:1.122023 valid_accuracy:92.3077  
************************* 第 8 折结束 *************************  
train_loss:1.072896 train_accuracy:97.0696  
 valid_loss:1.076552 valid_accuracy:96.7033  
************************* 第 9 折结束 *************************  
train_loss:1.070454 train_accuracy:97.3138  
 valid_loss:1.054581 valid_accuracy:98.9011  
************************* 第 10 折结束 *************************  
train_loss:1.064349 train_accuracy:97.9243  
 valid_loss:1.098537 valid_accuracy:94.5055  
