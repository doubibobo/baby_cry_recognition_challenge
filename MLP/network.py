import torch.nn as nn
import torch.nn.functional as functional


def batch_normalization_layer(channels):
    """
    batch_normalization层，使激活函数的输入保持在一个稳定状态来陷入梯度饱和区
    :param channels: 输入的特征维度
    :return: 某个batch_normalization层：（数据是不训练的）
    """
    return nn.BatchNorm1d(channels)


def layer_normalization_layer(shape):
    return nn.LayerNorm(shape)


class Network(nn.Module):
    """
    构造神经网络结构，重写初始化函数__init__()和前向过程forward()
    """

    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()

        # self.layer_01 = nn.Linear(input_dim, 128)
        # self.layer01_bn = batch_normalization_layer(128)
        # self.drop01 = nn.Dropout(0.3)

        self.layer0 = nn.Linear(input_dim, 512)
        self.layer0_bn = batch_normalization_layer(512)
        self.drop0 = nn.Dropout(0.3)

        # self.layer1 = nn.Linear(input_dim, 256)
        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_dim)
        # self.layer1_bn = batch_normalization_layer(256)
        # self.layer2_bn = batch_normalization_layer(128)
        # self.layer3_bn = batch_normalization_layer(64)
        # self.layer4_bn = batch_normalization_layer(output_dim)
        self.layer1_bn = layer_normalization_layer(256)
        self.layer2_bn = layer_normalization_layer(128)
        self.layer3_bn = layer_normalization_layer(64)
        self.layer4_bn = layer_normalization_layer(output_dim)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.3)

    def forward(self, x):
        # x = self.layer_01(x)
        # x = self.layer01_bn(x)
        # x = self.drop01(x)
        # x = functional.tanh(x)

        x = self.layer0(x)
        x = self.layer0_bn(x)
        x = self.drop0(x)
        x = functional.tanh(x)

        x = self.layer1(x)
        x = self.layer1_bn(x)
        x = self.drop1(x)
        x = functional.tanh(x)

        x = self.layer2(x)
        x = self.layer2_bn(x)
        x = self.drop2(x)
        x = functional.tanh(x)

        x = self.layer3(x)
        x = self.layer3_bn(x)
        x = self.drop3(x)
        x = functional.tanh(x)

        x = self.layer4(x)
        x = self.layer4_bn(x)
        # x = self.drop4(x)
        x = functional.tanh(x)
        # x = functional.softmax(x)
        return x


class Network_2(nn.Module):
    """
    构造神经网络结构，重写初始化函数__init__()和前向过程forward()
    """

    def __init__(self, input_dim, output_dim):
        super(Network_2, self).__init__()

        self.layer_01 = nn.Linear(input_dim, 128)
        self.layer01_bn = batch_normalization_layer(128)
        self.drop01 = nn.Dropout(0.3)

        self.layer0 = nn.Linear(128, 512)
        self.layer0_bn = batch_normalization_layer(512)
        self.drop0 = nn.Dropout(0.3)

        # self.layer1 = nn.Linear(input_dim, 256)
        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_dim)
        # self.layer1_bn = batch_normalization_layer(256)
        # self.layer2_bn = batch_normalization_layer(128)
        # self.layer3_bn = batch_normalization_layer(64)
        # self.layer4_bn = batch_normalization_layer(output_dim)
        self.layer1_bn = layer_normalization_layer(256)
        self.layer2_bn = layer_normalization_layer(128)
        self.layer3_bn = layer_normalization_layer(64)
        self.layer4_bn = layer_normalization_layer(output_dim)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer_01(x)
        x = self.layer01_bn(x)
        x = self.drop01(x)
        x = functional.tanh(x)

        x = self.layer0(x)
        x = self.layer0_bn(x)
        x = self.drop0(x)
        x = functional.tanh(x)

        x = self.layer1(x)
        x = self.layer1_bn(x)
        x = self.drop1(x)
        x = functional.tanh(x)

        x = self.layer2(x)
        x = self.layer2_bn(x)
        x = self.drop2(x)
        x = functional.tanh(x)

        x = self.layer3(x)
        x = self.layer3_bn(x)
        x = self.drop3(x)
        x = functional.tanh(x)

        x = self.layer4(x)
        x = self.layer4_bn(x)
        # x = self.drop4(x)
        x = functional.tanh(x)
        # x = functional.softmax(x)
        return x


def weights_init(network):
    """
    神经网络参数初始化
    :param network： 待初始化参数的神经网络
    :return 无返回值
    """
    if isinstance(network, nn.Linear):
        print("123")
        # bias = network.bias
        nn.init.kaiming_normal_(network.weight)
        # nn.init.xavier_normal_(network.bias, 0)
    if isinstance(network, nn.BatchNorm1d):
        print("456")
        network.eval()
