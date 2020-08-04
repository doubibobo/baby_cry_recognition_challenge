from collections import OrderedDict
from torch import nn
from torch.nn import functional


class CNNClassify(nn.Module):
    """
    构建卷积神经网络，用于进一步学习LSTM的输出特征，并完成分类
    """

    def __init__(self, input_channels, class_number, kernel_size=(3, 1), stride=(3, 1), padding=0, pool_size=(3, 1)):
        """
        CNN初始化函数
        :param input_channels: 输入的维度
        :param class_number: 输出的种类数目
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding: 是否填充
        :param pool_size: MaxPool的尺寸
        """
        super(CNNClassify, self).__init__()
        # CNN 进一步提取特征模块
        # CNN 的输入应为：(batch_size, channel, height, width)
        #       input =====> batch_size, 1, seq_length, hidden_size
        # 经过卷积conv之后的height和width应为：
        #   height = \frac{height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1}{stride[0]} + 1
        #   width = \frac{width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1}{stride[1]} + 1
        #       output-of-conv1 =====> batch_size, 16, 500, 26 | batch_size, 16, 500, 21
        #       output-of-pool1 =====> batch_size, 16, 167, 26 | batch_size, 16, 167, 10
        #       output-of-conv2 =====> batch_size, 32, 55, 26  | batch_size, 32, 55, 3
        #       output-of-pool2 =====> batch_size, 32, 18, 26  | batch_size, 32, 18, 1
        #       output-of-conv3 =====> batch_size, 32, 6, 26
        #       output-of-pool3 =====> batch_size, 32, 2, 26
        self.cnn = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(input_channels, 16, kernel_size, stride, padding)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(pool_size)),
            ("conv2", nn.Conv2d(16, 32, kernel_size, stride, padding)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool2d(pool_size))
        ]))

        # 全连接分类模块
        self.fully_connection = nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(32 * 18 * 15, 1024)),
            ("ReLU1", nn.ReLU()),
            ("layer2", nn.Linear(1024, 512)),
            ("ReLU2", nn.ReLU()),
            ("layer3", nn.Linear(512, 256)),
            ("ReLU3", nn.ReLU()),
            ("layer4", nn.Linear(256, 64)),
            ("ReLU4", nn.ReLU()),
            ("layer5", nn.Linear(64, class_number)),
            ("soft1", nn.Softmax()),
        ]))
        # # 全连接分类模块
        # self.layer1 = nn.Linear(32 * 18 * 15, class_number)

    def forward(self, x):
        """
        前向传播函数
        :param x: 样本的特征表示
        :return: 返回class_number的概率分布
        """
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        # x = self.layer1(x)
        # x = functional.softmax(x)
        x = self.fully_connection(x)
        return x


class LSTMClassify(nn.Module):
    """
    构建LSTM神经网络，用于encoder和decoder问题（即序列问题）
    """

    def __init__(self, feature_dim, hidden_dim, layer_number, output_size=None):
        """
        LSTM网络初始化方法
        :param feature_dim: 某一时刻的输入特征维度
        :param hidden_dim: 隐含层维度
        :param layer_number: 神经网络层数
        :param output_size: 种类数
        """
        super(LSTMClassify, self).__init__()
        # 由于batch_first设置为true，所以输入数据形式为：(batch_size, seq_length, feature_dim)
        self.rnn = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=layer_number, batch_first=True)
        # 如果单独使用LSTM，则需要知道output_size
        if output_size is not None:
            self.layer1 = nn.Linear(hidden_dim, output_size)
            self.output_size = output_size
        else:
            self.output_size = None

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入值
        :return: 输出值
        """
        # rnn_output的数据格式为： (batch, seq_len, num_directions * hidden_size)
        rnn_output, (_, _) = self.rnn(x, None)

        if self.output_size is not None:
            # output的数据格式为：(batch, num_directions * hidden_size)
            # 这里选取最后一个时间节点的rnn_output输出，也就是h_n，
            # 在实践中发现，最后一个时间节点大多为补充帧，对模型的分类效果可能较差，这里选取倒数第二个时间节点数据。
            output = self.layer1(rnn_output[:, -2, :])
            return functional.softmax(output)
        return rnn_output


class CombineClassify(nn.Module):
    """
    组合LSTMClassify 和 CNNClassify的网络模型
    """

    def __init__(self, feature_dim, hidden_dim, layer_number, input_channels, class_number,
                 kernel_size=(3, 3), stride=(3, 2), padding=0):
        """
        CombineClassify的初始化方法
        :param feature_dim: 某一时刻的输入特征维度
        :param hidden_dim: 隐含层维度
        :param layer_number: LSTM cell中的隐含层数目
        :param input_channels: 输入的维度
        :param class_number: 输出的种类数目
        :param kernel_size: 卷积核的尺寸，默认为(3, 3)
        :param stride:  卷积核的步长，默认为(3, 2)
        :param padding: 填充
        """
        super(CombineClassify, self).__init__()
        self.lstm = LSTMClassify(feature_dim, hidden_dim, layer_number)
        self.cnn = CNNClassify(input_channels, class_number, kernel_size, stride, padding)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入值
        :return: 输出值
        """
        # 首先将数据送入LSTM中，输出为： (batch, seq_length, num_directions * hidden_size)
        rnn_output = self.lstm(x)
        # CNN需要的数据输入格式为：(batch_size, channel, height, width)
        #   也即： (batch_size, channel, seq_length, num_directions * hidden_size)
        shape = rnn_output.shape
        cnn_input = rnn_output.view(shape[0], -1, shape[1], shape[2])
        # 将数据送入CNN中，输出为线性的softmax
        cnn_output = self.cnn(cnn_input)
        return cnn_output
