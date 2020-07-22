import torch.nn as nn
import torch.nn.functional as functional


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

    def forward(self, x):
        x = self.layer1(x)
        x = functional.relu(x)
        x = self.layer2(x)
        x = functional.relu(x)
        x = self.layer3(x)
        x = functional.relu(x)
        x = self.layer4(x)
        x = functional.softmax(x)
        return x
