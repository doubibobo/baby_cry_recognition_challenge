from torch import nn
from torchvision import models


class TransferVGG(nn.Module):
    def __init__(self):
        super(TransferVGG, self).__init__()
        # pre-trained为true，表示自动下载训练好的参数
        model = models.vgg16(pretrained=True)
