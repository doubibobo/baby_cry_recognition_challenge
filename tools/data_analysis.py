import pandas as pd
import numpy
import random
import torch

"建立训练时的标签和语音类别的映射关系"
label_classes = {"awake": 0, "diaper": 1, "hug": 2,
                 "hungry": 3, "sleepy": 4, "uncomfortable": 5}


def csv_handle(filename, another_file=None, is_test=False):
    """
    处理获取的csv文件，如：删除对训练无用的filename等
    @:arg
        filename: csv的文件路径
        another_file: 判断是否有第二个csv文件
        is_test: 判断是否是测试集文件
    @:returns
        torch_data: 数据集
        labels: 标签

    """
    # 设置初始值
    file_name_col, frame_number_col = None, None
    data = pd.read_csv(filename)

    if another_file is not None:
        data_another = pd.read_csv(another_file)
        # 两个csv文件的行列进行首尾拼接
        data = pd.concat([data, data_another], axis=0)
        # 提取最后一列帧的序号
        frame_number_col = data['frame_number'].copy()

    # 提取第一列的文件名
    file_name_col = data['filename'].copy()

    # 删除对训练数据无用的列，文件名只是训练时的标志序号
    data = data.drop(['filename'], axis=1)
    # 删除对训练数据无用的列，frame_number只是训练时的标志序号
    if another_file is not None or is_test:
        data = data.drop(['frame_number'], axis=1)

    # 对标签进行编码，用iloc函数提取最后一列[:, -1]，如果是取除最后一列以外的所有列[:, :-1]
    type_list = data.iloc[:, -1]
    labels = [label_classes.get(genre_list) for genre_list in type_list]

    # 预处理数据：去均值和方差规模化，即将特征数据的分布调整成标准正态分布（高斯分布），使得数据的均值为0,方差为1
    torch_data = torch.from_numpy(numpy.array(data.iloc[:, :-1], dtype=float))
    x_mean = torch_data.mean(dim=0, keepdim=True)
    x_standard = torch_data.std(dim=0, unbiased=False, keepdim=True)
    torch_data -= x_mean
    torch_data /= x_standard

    if is_test:
        return torch_data, torch.from_numpy(numpy.array(labels)), file_name_col, frame_number_col
    else:
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
    fold_size = data.shape[0] // k  # 确定每一折的个数
    list_index = [i for i in range(data.shape[0])]
    random.shuffle(list_index)

    data_train, label_train, data_validation, label_validation = None, None, None, None
    for i in range(k):
        # TODO 这里的K折交叉验证好菜，没有考虑数据集平衡的问题，就是非常单纯的做了数据划分，啊噗
        # TODO 问题：每次生成的验证集最多含有两类数据，且被大批量取出来的类别的训练集数目会减少很多，导致训练失衡
        # TODO 思路：生成一个随机数种子seed，将其取值范围重置为0-K
        # TODO 结果：index = [i for i in range(len(data))] index[0:len(index):K]
        # TODO anyway, 第69行已经将list_index打乱了
        index = slice(i * fold_size, (i + 1) * fold_size)  # valid的索引
        data_part, label_part = data[list_index[index], :], label[list_index[index]]
        if i == number:
            data_validation, label_validation = data_part, label_part
        elif data_train is None:
            data_train, label_train = data_part, label_part
        else:
            data_train = torch.cat((data_train, data_part), dim=0)
            label_train = torch.cat((label_train, label_part), dim=0)
    return data_train.float(), label_train, data_validation.float(), label_validation


def split_train_test(data, label):
    """
    按照7:3的比例划分训练集和测试集
    :param data: 原始数据, list
    :param label: 原始数据的标签, list
    :return: 训练集/训练集标签/测试集/测试集标签
    """
    list_index = [i for i in range(len(data))]
    boarder = int(0.7 * len(data))
    random.shuffle(list_index)

    data, label = numpy.mat(data).T, numpy.mat(label).T
    data_train, data_test = data[list_index[0: boarder], :].T, data[list_index[boarder: len(data)], :].T
    label_train, label_test = label[list_index[0: boarder], :].T, label[list_index[boarder: len(label)], :].T
    return data_train.tolist()[0], label_train.tolist()[0], data_test.tolist()[0], label_test.tolist()[0]
