import torch
import torch.nn as nn
import torch.functional as functional

from torch.utils.data import DataLoader, Dataset


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,          # input of the
                out_channels=16,        # number of the filters
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=16,
        #         out_channels=
        #     )
        # )
if __name__ == '__main__':
    print("Hello, world!")