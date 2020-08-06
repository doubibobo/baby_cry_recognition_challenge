import torch.nn as nn


class CNNBase(nn.Module):
    """
    自己构建的卷积神经网络的父类，所有处理图像问题的CNN子类应该继承于此
    """

    def __init__(self, input_channels, class_number, kernel_size=(3, 1), stride=(3, 1), dilation=1,
                 padding_mode='zeros', padding=0, pool_size=(3, 1)):
        """
        初始化函数
        :param input_channels: (int)输入的通道数
        :param class_number: (int)输出的类别数目
        :param kernel_size: (int or tuple): Size of the convolution kernel 卷积核大小
        :param stride: (int or tuple, optional): Stride of the convolution. Default: 1 步长大小
        :param dilation: (int or tuple, optional), Spacing between kernel elements. Default: 1
        :param padding_mode: (string, optional). Accepted values `zeros` and `circular` Default: `zeros` 填充模式
        :param padding: (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0 填充大小

        :param pool_size: (int or tuple, optional) 池化的尺寸
        """
        super(CNNBase, self).__init__()
        self.input_channels = input_channels
        self.class_number = class_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.padding = padding
        self.pool_size = pool_size
