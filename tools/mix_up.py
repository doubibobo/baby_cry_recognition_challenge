import numpy
import torch

# 设置超参数（hyper parameter）
# mix_up的paper说，在0.2时能够取得最好的值
# 故第一次使用时可以选择alpha为0.2
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def mix_up_signal(signal_1, signal_2, label_1, label_2, index=2):
    """
    用简单的方式增加样本数据集合，注意：label均为one-hot编码
    :param signal_1: 第一个信号
    :param signal_2: 第二个信号
    :param label_1: 第一个信号的标签
    :param label_2: 第二个信号的标签
    :param index: 选取的alpha的值，默认为1,选择alpha为0.2
    :return: 新信号、新信号的标签
    """
    # 首先判断两者的信号长度是否一致, 不一致则补零
    signal_length_1 = len(signal_1)
    signal_length_2 = len(signal_2)
    if signal_length_1 < signal_length_2:
        signal_1 = numpy.append(signal_1, numpy.zeros(signal_length_2 - signal_length_1))
    elif signal_length_1 > signal_length_2:
        signal_2 = numpy.append(signal_2, numpy.zeros(signal_length_1 - signal_length_2))

    new_signal = [signal_1[i] * alpha[index] + signal_2[i] * (1 - alpha[index]) for i in range(max(signal_length_1,
                                                                                                   signal_length_2))]
    new_signal_label = [label_1[i] * alpha[index] + label_2[i] * (1 - alpha[index]) for i in range(len(label_1))]
    return new_signal, new_signal_label


def mix_data(data, label, alpha_value=10, gpu_available=False):
    """
    创建新的数据集，以一个batch_size为单位
    :param data: 数据集
    :param label: 标签
    :param alpha_value: 取的alpha的值，默认为1,选择alpha为0.2
    :param gpu_available: 是否使用GPU
    :return:
    """
    if alpha_value > 0:
        lam = numpy.random.beta(alpha[alpha_value], alpha[alpha_value])
    else:
        lam = 1.
    # print(lam)
    # 计算一个batch_size的长度
    batch_size = data.size()[0]
    # 生成一个无序序列
    if gpu_available:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_data = lam * data + (1 - lam) * data[index, :]
    label_a, label_b = label, label[index]
    return mixed_data, label_a, label_b, lam


def mix_criterion(label_a, label_b, lam):
    """
    mix之后的损失函数
    :param label_a: 第一个标签
    :param label_b: 第二个标签
    :param lam: lam的值（lam, 1 - lam）
    :return: 匿名函数
    """
    return lambda criterion, prediction: \
        lam * criterion(prediction, label_a) + (1 - lam) * criterion(prediction, label_b)
