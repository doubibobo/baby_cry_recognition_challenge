import torch
import numpy

from PIL import Image
from torchsummary import summary

from data.code.LSTM import network as net
from data.code.tools.image_tools import to_gray as tg
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.network_tools import train_process as tp
from data.code.tools.training_tools import gpu_selector as gs
from data.code.tools.feature_tools import build_file_index as bf


# 定义超参数
K = 10
EPOCH_NUMBER = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 16
OUTPUT_SIZE = 6
WEIGHT_DELAY = 1e-8
IMAGE_HEIGHT = 770
IMAGE_WIDTH = 775

FILE_NAME = "model/cnn_model_image_01.pkl"


if __name__ == '__main__':
    # 建立训练集文件路径与标签的索引
    file_label_indexes = bf.get_filename("train", tg.filePathGray)
    print(file_label_indexes)

    # 使用gpu进行训练
    gpu_available = gs.gpu_selector()

    # 构建卷积神经网络
    network = [net.CNNClassify(1, OUTPUT_SIZE, kernel_size=(3, 3), stride=(3, 3), padding=0, pool_size=(3, 3))
               for i in range(K)]

    # 打印神经网络的结构
    summary(network[0], (1, IMAGE_HEIGHT, IMAGE_WIDTH))

    # # 将数据读入内存
    # numpy_data, numpy_label = [], []
    # for key, value in file_label_indexes.items():
    #     image = Image.open(tg.filePathGray + "train/" + value + "/" + key)
    #     numpy_data.append(numpy.asarray(image))
    #     numpy_label.append(da.label_classes[value])
    #
    # torch_data, torch_label = torch.from_numpy(numpy.asarray(numpy_data)), torch.from_numpy(numpy.asarray(numpy_label))
    #
    # # 训练网络
    # tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
    #                  network_filename=FILE_NAME, gpu_available=gpu_available)
    #
    # # 将所有数据投入到最终的神经网络进行训练
    # final_network = net.CNNClassify(1, OUTPUT_SIZE, kernel_size=(3, 3), stride=(3, 3), padding=0, pool_size=(3, 3))
    # tp.train_final_network(final_network, torch_data, torch_label, LEARNING_RATE, EPOCH_NUMBER,
    #                        weight_decay=WEIGHT_DELAY, batch_size=BATCH_SIZE, file_name=FILE_NAME)
