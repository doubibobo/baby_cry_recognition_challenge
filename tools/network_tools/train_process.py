import torch

from torch import nn
from torch.utils.data import DataLoader

from data.code.tools import data_analysis as da
from data.code.tools import mix_up as mu
from data.code.tools.network_tools import base_class as bc
from data.code.tools.network_tools import accuracy_loss_plotting as alp
from data.code.tools.network_tools.log_rmse import log_rmse
from data.code.tools.training_tools import statistics_counter as sc

from data.code.tools.algorithm import Adam as Adam_GCC


def train(network, data_train, label_train, data_validation, label_validation, learning_rate,
          epoch_number=30, weight_decay=0.0001, batch_size=32, gpu_available=False, alpha=0):
    """
    神经网络训练过程
    :param network: 构造的神经网络
    :param data_train: 训练数据集
    :param label_train: 训练标签集
    :param data_validation: 验证集
    :param label_validation: 验证集
    :param epoch_number: 迭代次数
    :param learning_rate: 学习率
    :param weight_decay: 在原本损失函数的基础上，加上L2正则化，而weight_decay就是这个正则化的lambda参数，一般设置为1e-8
    :param batch_size: 每组的样本个数，默认为32
    :param gpu_available: 是否使用GPU的选项，默认为False，不使用GPU
    :param alpha: 用于mix_up的alpha值
    :return: 训练时损失，验证时损失
    """
    # 定义训练集的loss和accuracy为loss_train 测试集的loss和accuracy为loss_validation
    loss_train, loss_validation = [], []
    best_accuracy_validation = 0
    early_stop_epoch = 0

    # 使用GPU进行训练
    if gpu_available:
        network = network.cuda()
        data_train, label_train = data_train.cuda(), label_train.cuda()
        if data_validation is not None and label_validation is not None:
            data_validation, label_validation = data_validation.cuda(), label_validation.cuda()
        print("数据已经转化为gpu类型")

    # 将数据封装成DataLoader
    dataset = bc.TrainDataSet(data_train, label_train)

    # 进行数据封装
    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    # 使用cross_entropy损失函数
    loss_function = nn.CrossEntropyLoss()

    # # 使用Adam优化算法
    # optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 使用改进的Adam_GC优化算法
    optimizer = Adam_GCC.Adam_GC(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 动态调整学习率
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # 使用GPU
    if gpu_available:
        loss_function = loss_function.cuda()

    # 分批训练
    for epoch in range(epoch_number):
        # scheduler.step(epoch)
        for batch_index, (inputs, targets) in enumerate(train_iter):
            if gpu_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets_a, targets_b, lam = mu.mix_data(inputs, targets, alpha, gpu_available)
            outputs = network(inputs)
            loss_func = mu.mix_criterion(targets_a, targets_b, lam)
            loss = loss_func(loss_function, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # for X, y in train_iter:
        #     output = network(X)
        #     loss = loss_function(output, y)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        # 得到每个epoch的 loss 和 accuracy
        # print("epoch is ", epoch)
        loss_train.append(log_rmse(False, network, dataset.x_data, dataset.y_data, loss_function))

        if data_validation is not None:
            loss_validation.append(log_rmse(True, network, data_validation, label_validation, loss_function, epoch))
        # scheduler.step(loss_validation[-1][1])

        # if data_validation is not None and loss_validation[-1][1] >= best_accuracy_validation:
        #     best_accuracy_validation = loss_validation[-1][1]
        # else:
        #     early_stop_epoch = early_stop_epoch + 1
        # if early_stop_epoch >= 10:
        #     break

    del data_train, label_train, data_validation, label_validation
    return loss_train, loss_validation


def train_process(data_train_input, label_train_input, network, k_number, learning_rate=0.001, epoch_number=30,
                  batch_size=32, weight_decay=1e-8, network_filename="net.pkl", gpu_available=False):
    """
    k折划分后的训练过程，并且要求使用最好的神经网络
    :param data_train_input: 数据集
    :param label_train_input: 数据标签
    :param network: 神经网络对象，是一个数组
    :param k_number: 折的数目
    :param learning_rate: 学习率
    :param epoch_number: 迭代次数，默认为30次
    :param batch_size: 每组的样本个数，默认为32
    :param weight_decay: 原本损失函数的基础上，加上L2正则化，而weight_decay就是这个正则化的lambda参数，一般设置为1e-8
    :param network_filename: 神经网络默认保存文件名
    :param gpu_available: 是否使用GPU的选项，默认为False，不使用GPU
    :return: 无返回值
    """
    best_loss_accuracy_validation = 0
    loss_train_sum, loss_validation_sum = 0, 0
    accuracy_train_sum, accuracy_validation_sum = 0, 0

    # 进行交叉验证数据集的划分
    divided_list_index = da.get_k_fold_data_by_proportion(k_number, label_train_input)

    for i in range(k_number):

        data_train, label_train, data_validation, label_validation = da.get_k_fold_data(
            k_number, i, data_train_input, label_train_input, divided_list_index)
        print('*' * 25, '第', i + 1, '折开始', '*' * 25)
        sc.counter_statistics(label_validation)
        sc.counter_statistics(label_train)
        # 对每一份数据及神经网络进行训练
        loss_train, loss_validation = train(network[i], data_train, label_train, data_validation,
                                            label_validation, learning_rate, epoch_number, weight_decay, batch_size,
                                            gpu_available, 2)
        # 画出训练过程中的accuracy和loss变换曲线
        alp.accuracy_loss_plotting(loss_train, epoch_number, i,  True)
        alp.accuracy_loss_plotting(loss_validation, epoch_number, i, False)

        # 输出这一批数据的最终训练结果
        print('*' * 25, '第', i + 1, '折结束', '*' * 25)
        print('train_loss:%.6f' % loss_train[-1][0], 'train_accuracy:%.4f\n' % loss_train[-1][1],
              'valid_loss:%.6f' % loss_validation[-1][0], 'valid_accuracy:%.4f' % loss_validation[-1][1])

        # TODO: 对K折交叉验证的理解有问题，导致整体的网络训练出现了问题
        # # 确定并保存当前最好的训练结果，这里是保存整个网络
        # if loss_validation[-1][1] >= best_loss_accuracy_validation:
        #     torch.save(network[i], network_filename)

        loss_train_sum += loss_train[-1][0]
        loss_validation_sum += loss_validation[-1][0]
        accuracy_train_sum += loss_train[-1][1]
        accuracy_validation_sum += loss_validation[-1][1]

    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    print('train_loss_sum:%.4f' % (loss_train_sum / k_number),
          'train_accuracy_sum:%.4f\n' % (accuracy_train_sum / k_number),
          'valid_loss_sum:%.4f' % (loss_validation_sum / k_number),
          'valid_accuracy_sum:%.4f\n' % (accuracy_validation_sum / k_number))

    # 打印当前的超参数，主要为： learning_rate, epoch_number, batch_size
    print('learning_rate:' + str(learning_rate) + '\n' +
          'epoch_number:' + str(epoch_number) + '\n' +
          'batch_size:' + str(batch_size) + '\n' +
          'weight_delay:' + str(weight_decay) + '\n')


def train_final_network(final_network, data_train, label_train, learning_rate, epoch_number, weight_decay, batch_size,
                        file_name="net.pkl", gpu_available=False):
    """
    :param final_network: 构造的神经网络
    :param data_train: 训练集
    :param label_train: 训练集标签
    :param learning_rate: 学习率
    :param epoch_number: 迭代次数
    :param weight_decay: 正则化项目
    :param batch_size: 每组的样本个数
    :param file_name: 神经网络保存的文件名
    :param gpu_available: 是否使用GPU进行训练
    :return: 无返回结果，直接打印输出
    """
    loss_train, _ = train(final_network, data_train, label_train, None, None, learning_rate, epoch_number, weight_decay,
                          batch_size, gpu_available)
    # 画出训练过程中的accuracy和loss变换曲线
    alp.accuracy_loss_plotting(loss_train, epoch_number, 100, True)
    print('#' * 10, '最终训练结果', '#' * 10)
    print('train_loss:%.4f' % (loss_train[-1][0]),
          'train_accuracy:%.4f\n' % (loss_train[-1][1]))
    # 保存神经网络
    torch.save(final_network, file_name)


def test_process(data_test, network_filename="net.pkl", gpu_available=False):
    """
    在测试集合上进行测试
    :param data_test: 测试集
    :param network_filename: 神经网络默认保存文件名
    :param gpu_available: 测试过程是否使用GPU，默认为False，不使用
    :return: 针对每个样本的预测值
    """
    # 加载最好的模型，并返回预测值
    network = torch.load(network_filename)
    if gpu_available:
        network = network.cuda()
        data_test = data_test.cuda()
    prediction = network(data_test.float())
    result = torch.max(prediction, 1)[1]
    # if gpu_available:
    #     return result.cpu()
    # return result
    return result
