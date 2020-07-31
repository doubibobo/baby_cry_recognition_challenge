import torch
from torch import nn
from torch.nn import functional


class LSTMClassify(nn.Module):
    """
    构建LSTM神经网络，用于encoder和decoder问题（即序列问题）
    """
    def __init__(self, feature_dim, hidden_dim, layer_number, class_number):
        """
        LSTM网络初始化方法
        :param feature_dim: 某一时刻的输入特征维度
        :param hidden_dim: 隐含层维度
        :param layer_number: 神经网络层数
        :param class_number: 种类数目
        """
        super(LSTMClassify, self).__init__()
        # 由于batch_first设置为true，所以输入数据形式为：(batch_size, seq_length, feature_dim)
        self.rnn = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=layer_number,
            bidirectional=True,
            batch_first=True
        )
        # # 卷积层
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        # # 池化层
        # self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, class_number)
        # # 输出层(不加CNN层的做法)
        # self.layer1 = nn.Linear(hidden_dim, class_number)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入值
        :return: 输出值
        """
        # rnn_output的数据格式为： (batch, seq_len, num_directions * hidden_size)
        rnn_output, (_, _) = self.rnn(x, None)
        # rnn_output = torch.reshape(rnn_output, (512, 1, 1502, 64))
        #
        # output = self.conv1(rnn_output)
        # output = self.pooling1(functional.relu(output))
        # output = self.conv2(output)
        # output = self.pooling2(functional.relu(output))
        # print(output.shape)
        # print(rnn_output.shape)
        # rnn_output = torch.reshape(rnn_output, (len(rnn_output[:, 0, 0]), 1502 * 64))
        # output = functional.relu(self.fc1(rnn_output[:, -2, :]))
        output = functional.relu(self.fc1(rnn_output[:, -2, :]))
        output = functional.relu(self.fc2(output))
        output = functional.relu(self.fc3(output))
        output = functional.softmax(self.fc4(output))

        # 不加CNN层的做法
        # output的数据格式为：(batch, num_directions * hidden_size)
        # 这里选取最后一个时间节点的rnn_output输出，也就是h_n
        # output = self.layer1(rnn_output[:, -2, :])
        # output = functional.softmax(output)
        return output
