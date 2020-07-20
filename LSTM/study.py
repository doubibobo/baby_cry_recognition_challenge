import torch

from torch import nn
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(

        )


if __name__ == '__main__':
    dictionary = dict()
    dictionary['name'] = 'doubibobo'
    dictionary['age'] = 24
    print(dictionary)
    print(dictionary.items())
    print(dictionary.keys())
    dictionary.pop('value', "one")
    print(dictionary)
