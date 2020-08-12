import torch
from data.code.tools.training_tools import statistics_counter as sc
from data.code.tools.feature_tools import feature_extractor as fe


def log_rmse(flag, network, x, y, loss_function, epoch=99):
    """
    网络模型评价指标
    :param flag: 是否是验证集
    :param network: 神经网络
    :param x: 样本
    :param y: 样本标签
    :param loss_function: 损失函数
    :param epoch: 迭代次数，默认为100
    :return: 计算值
    """
    if flag:
        # Sets the module in evaluation mode.
        network.eval()
    output = network(x)
    result = torch.max(output, 1)[1].view(y.size())  # 只返回最大值的每个索引
    corrects = (result.data == y.data).sum().item()

    # 计算每个类别的准确率
    if flag and epoch % 99 == 0 and epoch != 0:
        # sc.counter_statistics(result.csv_data.cpu())
        pass
    accuracy = corrects * 100 / len(y)
    loss = loss_function(output, y)
    # Sets the module in training mode.
    network.train()

    # 将data_valid的结果写入csv文档中
    if epoch == 199:
        if flag:
            fe.write_valid_dataset_csv("/home/zhuchuanbo/competition/data/code/MLP/data_valid/", 'results.csv', output, result)
        else:
            fe.write_valid_dataset_csv("/home/zhuchuanbo/competition/data/code/MLP/data_valid/", 'results_train.csv', output, result)

    return loss.data.item(), accuracy
