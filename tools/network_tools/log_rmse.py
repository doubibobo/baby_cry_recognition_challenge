import torch


def log_rmse(flag, network, x, y, loss_function):
    """
    网络模型评价指标
    :param flag: 是否是验证集
    :param network: 神经网络
    :param x: 样本
    :param y: 样本标签
    :param loss_function: 损失函数
    :return: 计算值
    """
    if flag:
        network.eval()
    output = network(x)
    result = torch.max(output, 1)[1].view(y.size())  # 只返回最大值的每个索引
    corrects = (result.data == y.data).sum().item()

    accuracy = corrects * 100 / len(y)
    loss = loss_function(output, y)
    network.train()

    return loss.data.item(), accuracy
