import pandas as pd
import numpy
import random
import torch

"建立训练时的标签和语音类别的映射关系"
label_classes = {"awake": 0, "diaper": 1, "hug": 2,
                 "hungry": 3, "sleepy": 4, "uncomfortable": 5}


def csv_handle(filename):
    """
    处理获取的csv文件，如：删除对训练无用的filename等
    @:arg
        filename: csv的文件路径
    @:returns
        torch_data: 数据集
        labels: 标签
    """
    data = pd.read_csv(filename)
    data.head()
    print(data.shape)
    # 删除对训练数据无用的列，如文件名
    data = data.drop(['filename'], axis=1)

    # 对标签进行编码，用iloc函数提取最后一列[:, -1]，如果是取除最后一列以外的所有列[:, :-1]
    type_list = data.iloc[:, -1]
    labels = [label_classes.get(genre_list) for genre_list in type_list]

    # 预处理数据：去均值和方差规模化，即将特征数据的分布调整成标准正态分布（高斯分布），使得数据的均值为0,方差为1
    torch_data = torch.from_numpy(numpy.array(data.iloc[:, :-1], dtype=float))
    x_mean = torch_data.mean(dim=0, keepdim=True)
    x_standard = torch_data.std(dim=0, unbiased=False, keepdim=True)
    torch_data -= x_mean
    torch_data /= x_standard

    return torch_data, torch.from_numpy(numpy.array(labels))


def get_k_fold_data(k, number, data, label):
    """
    划分数据集为训练集（train）和验证集（validation）
    实现K折交叉验证方法
    :arg
        k: 折的数目
        number: 第几折
        data: 数据集
        y: 标签
    :returns
    """
    assert k > 1
    fold_size = data.shape[0] // k      # 确定每一折的个数
    list_index = [i for i in range(data.shape[0])]
    random.shuffle(list_index)

    data_train, label_train, data_validation, label_validation = None, None, None, None
    for i in range(k):
        index = slice(i * fold_size, (i + 1) * fold_size)                           # valid的索引
        data_part, label_part = data[list_index[index], :], label[list_index[index]]
        if i == number:
            data_validation, label_validation = data_part, label_part
        elif data_train is None:
            data_train, label_train = data_part, label_part
        else:
            data_train = torch.cat((data_train, data_part), dim=0)
            label_train = torch.cat((label_train, label_part), dim=0)
    return data_train.float(), label_train, data_validation.float(), label_validation
