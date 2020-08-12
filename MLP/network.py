import torch.nn as nn
import torch.nn.functional as functional


def batch_normalization_layer(channels):
    """
    batch_normalization层，使激活函数的输入保持在一个稳定状态来陷入梯度饱和区
    :param channels: 输入的特征维度
    :return: 某个batch_normalization层：（数据是不训练的）
    """
    return nn.BatchNorm1d(channels)


class Network(nn.Module):
    """
    构造神经网络结构，重写初始化函数__init__()和前向过程forward()
    """

    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_dim)
        self.layer1_bn = batch_normalization_layer(256)
        self.layer2_bn = batch_normalization_layer(128)
        self.layer3_bn = batch_normalization_layer(64)
        self.layer4_bn = batch_normalization_layer(output_dim)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_bn(x)
        x = self.drop1(x)
        x = functional.relu(x)

        x = self.layer2(x)
        x = self.layer2_bn(x)
        x = self.drop2(x)
        x = functional.relu(x)

        x = self.layer3(x)
        x = self.layer3_bn(x)
        x = self.drop3(x)
        x = functional.relu(x)

        x = self.layer4(x)
        x = self.layer4_bn(x)
        x = functional.softmax(x)
        return x
