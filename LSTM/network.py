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
            batch_first=True
        )

        # nn.LSTMCell
        # self.rnn_block = nn.Sequential(
        #     nn.LSTM(
        #         input_size=feature_dim,
        #         hidden_size=hidden_dim,
        #         num_layers=layer_number,
        #         batch_first=True
        #     )
        # )
        # 池化层
        # self.pooling1 = nn.MaxPool1d(3, stride=2)
        # TODO 后期需要考虑加入CNN，以提升训练结果
        # 输出层
        self.layer1 = nn.Linear(hidden_dim, class_number)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入值
        :return: 输出值
        """
        # rnn_output的数据格式为： (batch, seq_len, num_directions * hidden_size)
        rnn_output, (_, _) = self.rnn(x, None)

        # output的数据格式为：(batch, num_directions * hidden_size)
        # 这里选取最后一个时间节点的rnn_output输出，也就是h_n，
        # 在实践中发现，最后一个时间节点大多为补充帧，对模型的分类效果可能较差，这里选取倒数第二个时间节点数据。
        output = self.layer1(rnn_output[:, -2, :])
        output = functional.softmax(output)
        return output
