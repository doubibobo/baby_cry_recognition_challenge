import data.code.build_file_index as bf
import data.code.feature_extrator as fe
import data.code.data_analysis as da
import data.code.network as net

import torch

K = 10                      # 进行10折交叉验证
epoch_number = 10000        # 循环迭代次数为20
learning_rate = 0.001       # 学习率


def train_process(data_train, label_train):
    """
    k折划分后的训练过程，并且要求使用最好的神经网络
    :param data_train:      数据集
    :param label_train:     数据标签
    :return:
    """
    best_loss_accuracy_validation = 0
    loss_train_sum, loss_validation_sum = 0, 0
    accuracy_train_sum, accuracy_validation_sum = 0, 0

    for i in range(K):
        data = da.get_k_fold_data(K, i, data_train, label_train)
        network = net.Network(len(data_train[1, :]), 6)

        # 对每一份数据进行训练
        loss_train, loss_validation = net.train(network, *data, learning_rate, epoch_number)

        # 输出这一批数据的最终训练结果
        print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('train_loss:%.6f' % loss_train[-1][0], 'train_accuracy:%.4f\n' % loss_train[-1][1],
              'valid_loss:%.6f' % loss_validation[-1][0], 'valid_accuracy:%.4f' % loss_validation[-1][1])

        # 确定并保存当前最好的训练结果，这里是保存整个网络
        if loss_validation[-1][1] >= best_loss_accuracy_validation:
            torch.save(network, 'net.pkl')

        loss_train_sum += loss_train[-1][0]
        loss_validation_sum += loss_validation[-1][0]
        accuracy_train_sum += loss_train[-1][1]
        accuracy_validation_sum += loss_validation[-1][1]

    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    print('train_loss_sum:%.4f' % (loss_train_sum / K), 'train_accuracy_sum:%.4f\n' % (accuracy_train_sum / K),
          'valid_loss_sum:%.4f' % (loss_validation_sum / K), 'valid_accuracy_sum:%.4f' % (accuracy_validation_sum / K))


def test_process(data_test):
    """
    在测试集合上进行测试
    :param data_test:   测试集
    :return: 无返回值
    """
    # 加载最好的模型，并返回预测值
    network = torch.load('net.pkl')
    prediction = network(data_test.float())
    return torch.max(prediction, 1)[1]


if __name__ == '__main__':
    # 建立文件路径与标签的索引
    # file_label_indexes = bf.get_filename("train")
    # 获取频谱图
    # fe.extract_spectrogram(file_label_indexes, "train")
    # 写入到csv文件中
    # headers = fe.extract_features()
    # fe.write_data_to_csv_file(headers, file_label_indexes, "data.csv", "train")
    # 读取数据
    torch_data, torch_label = da.csv_handle("data.csv")
    # 进行训练
    train_process(torch_data, torch_label)

    # 进行测试集合的验证
    # test_label_indexes = bf.get_filename("test")
    # fe.extract_spectrogram(test_label_indexes, "test")
    # fe.write_data_to_csv_file(headers, test_label_indexes, "test.csv", "test")

    test_data, _ = da.csv_handle("test.csv")
    fe.write_result_to_csv("test.csv", "result.csv", test_process(test_data))
