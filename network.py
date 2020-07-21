import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch.utils.data import DataLoader, Dataset


class Network(nn.Module):
    """
    构造神经网络结构，重写初始化函数__init__()和前向过程forward()
    """

    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.drop1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.5)
        self.layer4 = nn.Linear(128, 64)
        self.drop4 = nn.Dropout(0.5)
        self.layer5 = nn.Linear(64, 32)
        self.drop5 = nn.Dropout(0.5)
        self.layer6 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = functional.relu(self.drop1(x))
        x = self.layer2(x)
        x = functional.relu(self.drop2(x))
        x = self.layer3(x)
        x = functional.relu(self.drop3(x))
        x = self.layer4(x)
        x = functional.relu(self.drop4(x))
        x = self.layer5(x)
        x = functional.relu(self.drop5(x))
        x = self.layer6(x)
        x = functional.softmax(x)
        return x


class TrainDataSet(Dataset):
    """
    构造数据集类
    """

    def __init__(self, data_train, label_train):
        self.x_data = data_train
        self.y_data = label_train
        self.length = len(label_train)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.length


def log_rmse(flag, network, x, y, loss_function):
    """
    网络模型评价指标
    :param flag:
    :param network:
    :param x:
    :param y:
    :param loss_function:
    :return:
    """
    if flag:
        network.eval()
    output = network(x)
    result = torch.max(output, 1)[1].view(y.size())           # 只返回最大值的每个索引
    corrects = (result.data == y.data).sum().item()

    accuracy = corrects * 100 / len(y)
    loss = loss_function(output, y)
    network.train()

    return loss.data.item(), accuracy


def train(network, data_train, label_train, data_validation, label_validation,
          learning_rate, epoch_number=30, weight_decay=0.000000001, batch_size=512,
          gpu_available=False):
    """
    神经网络训练过程
    :param network: 构造的神经网络
    :param data_train: 训练集
    :param label_train: 训练集
    :param data_validation: 验证集
    :param label_validation: 验证集
    :param epoch_number: 迭代次数
    :param learning_rate: 学习率
    :param weight_decay: 在原本损失函数的基础上，加上L2正则化，而weight_decay就是这个正则化的lambda参数，一般设置为1e-8
    :param batch_size: 每组的样本个数，默认为32
    :param gpu_available: GPU的可用性
    :return:
    """
    # 定义训练集的loss和accuracy为loss_train 测试集的loss和accuracy为loss_validation
    loss_train, loss_validation = [], []

    # 将数据封装成DataLoader
    dataset = TrainDataSet(data_train, label_train)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    # 使用cross_entropy损失函数
    loss_function = nn.CrossEntropyLoss()

    if gpu_available:
        loss_function = loss_function.cuda()

    # 使用Adam优化算法
    optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 分批训练
    for epoch in range(epoch_number):
        for X, y in train_iter:
            X,y = X.cuda(), y.cuda()
            output = network(X)
            loss = loss_function(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 得到每个epoch的 loss 和 accuracy
        loss_train.append(log_rmse(False, network, data_train, label_train, loss_function))
        if data_validation is not None:
            loss_validation.append(log_rmse(True, network, data_validation, label_validation, loss_function))

    return loss_train, loss_validation
