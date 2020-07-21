import torch

from torch import nn
from torch.utils.data import DataLoader

from data.code.tools.network_tools import base_class as bc
from data.code.tools import data_analysis as da
from data.code.tools.network_tools.log_rmse import log_rmse


def train(network, dataset, data_validation, label_validation, learning_rate,
          epoch_number=30, weight_decay=0.0001, batch_size=32):
    """
    神经网络训练过程
    :param network: 构造的神经网络
    :param dataset: 封装好的训练集数据
    :param data_validation: 验证集
    :param label_validation: 验证集
    :param epoch_number: 迭代次数
    :param learning_rate: 学习率
    :param weight_decay: 在原本损失函数的基础上，加上L2正则化，而weight_decay就是这个正则化的lambda参数，一般设置为1e-8
    :param batch_size: 每组的样本个数，默认为32
    :return: 训练时损失，验证时损失
    """
    # 定义训练集的loss和accuracy为loss_train 测试集的loss和accuracy为loss_validation
    loss_train, loss_validation = [], []

    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    # 使用cross_entropy损失函数
    loss_function = nn.CrossEntropyLoss()

    # 使用Adam优化算法
    optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 分批训练
    for epoch in range(epoch_number):
        for X, y in train_iter:
            output = network(X)
            loss = loss_function(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 得到每个epoch的 loss 和 accuracy
        print("epoch is ", epoch)
        loss_train.append(log_rmse(False, network, dataset.x_data, dataset.y_data, loss_function))
        if data_validation is not None:
            loss_validation.append(log_rmse(True, network, data_validation, label_validation, loss_function))

    return loss_train, loss_validation


def train_process(data_train, label_train, network, k_number, learning_rate=0.001, epoch_number=30, batch_size=32,
                  weight_decay=1e-8, network_filename="net.pkl"):
    """
    k折划分后的训练过程，并且要求使用最好的神经网络
    :param data_train: 数据集
    :param label_train: 数据标签
    :param network: 神经网络对象
    :param k_number: 折的数目
    :param learning_rate: 学习率
    :param epoch_number: 迭代次数，默认为30次
    :param batch_size: 每组的样本个数，默认为32
    :param weight_decay: 原本损失函数的基础上，加上L2正则化，而weight_decay就是这个正则化的lambda参数，一般设置为1e-8
    :param network_filename: 神经网络默认保存文件名
    :return: 无返回值
    """
    best_loss_accuracy_validation = 0
    loss_train_sum, loss_validation_sum = 0, 0
    accuracy_train_sum, accuracy_validation_sum = 0, 0

    for i in range(k_number):
        data_train, label_train, data_validation, label_validation = da.get_k_fold_data(k_number, i, data_train,
                                                                                        label_train)
        # 将数据封装成DataLoader
        dataset = bc.TrainDataSet(data_train, label_train)
        # 对每一份数据进行训练
        loss_train, loss_validation = train(network, dataset, data_validation, label_validation, learning_rate,
                                            epoch_number, weight_decay, batch_size)
        # 输出这一批数据的最终训练结果
        print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('train_loss:%.6f' % loss_train[-1][0], 'train_accuracy:%.4f\n' % loss_train[-1][1],
              'valid_loss:%.6f' % loss_validation[-1][0], 'valid_accuracy:%.4f' % loss_validation[-1][1])

        # 确定并保存当前最好的训练结果，这里是保存整个网络
        if loss_validation[-1][1] >= best_loss_accuracy_validation:
            torch.save(network, network_filename)

        loss_train_sum += loss_train[-1][0]
        loss_validation_sum += loss_validation[-1][0]
        accuracy_train_sum += loss_train[-1][1]
        accuracy_validation_sum += loss_validation[-1][1]

    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    print('train_loss_sum:%.4f' % (loss_train_sum / k_number),
          'train_accuracy_sum:%.4f\n' % (accuracy_train_sum / k_number),
          'valid_loss_sum:%.4f' % (loss_validation_sum / k_number),
          'valid_accuracy_sum:%.4f' % (accuracy_validation_sum / k_number))


def test_process(data_test, network_filename="net.pkl"):
    """
    在测试集合上进行测试
    :param data_test: 测试集
    :param network_filename: 神经网络默认保存文件名
    :return: 针对每个样本的预测值
    """
    # 加载最好的模型，并返回预测值
    network = torch.load(network_filename)
    prediction = network(data_test.float())
    return torch.max(prediction, 1)[1]
