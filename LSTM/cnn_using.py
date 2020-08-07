import numpy
import torch

from torchsummary import summary

from data.code.LSTM import network as net
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.training_tools import gpu_selector as gs
from data.code.tools.network_tools import train_process as tp
from data.code.tools.feature_tools import build_file_index as bf, feature_extractor as fe

# 定义超参数
K = 10
EPOCH_NUMBER = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TIME_STEP = 1502
INPUT_SIZE = 26
OUTPUT_SIZE = 6
WEIGHT_DELAY = 1e-8
FILE_NAME = "model/cnn_model_01.pkl"


if __name__ == '__main__':
    # # 输出csv文件的表头
    # headers = fe.extract_features(to_frame=True)
    #
    # # 建立训练集文件路径与标签的索引
    # file_label_indexes = bf.get_filename("train")
    # print(file_label_indexes)
    # # 进行训练集的特征提取，并将其写入csv文件中。
    # # 进行训练集的特征提取，并将其写入csv文件中。
    # fe.write_data_to_csv_file(headers, file_label_indexes, "../data/data_for_cnn_25ms_10ms.csv", "train", to_frame=True)
    #
    # # 建立测试集文件路径与标签的索引
    # test_label_indexes = bf.get_filename("test")
    # print(test_label_indexes)
    # # 进行测试集的特征提取，并将其写入csv文件中
    # fe.write_data_to_csv_file(headers, test_label_indexes, "../data/test_for_cnn_25ms_10ms.csv", "test", to_frame=True)

    # 使用gpu进行训练
    gpu_available = gs.gpu_selector()

    # 构建卷积神经网络
    network = [net.CNNClassify(1, OUTPUT_SIZE, (3, 2), (3, 2), 0, (2, 2)) for i in range(K)]

    # 打印神经网络的结构
    summary(network[0], (1, TIME_STEP, INPUT_SIZE), device="cpu")

    # 读取数据，并进行数据的重塑
    torch_data, torch_label = da.csv_handle("../data/data_for_rnn_1.csv", "../data/data_for_rnn.csv")
    torch_data = torch.reshape(torch_data, (-1, 1, TIME_STEP, INPUT_SIZE))
    torch_label = torch_label[0:len(torch_label):TIME_STEP]

    # 训练网络
    tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
                     network_filename=FILE_NAME, gpu_available=gpu_available)
    # 将所有数据投入到最终的神经网络进行训练
    final_network = net.CNNClassify(1, OUTPUT_SIZE, (3, 2), (3, 2), 0, (2, 2))
    tp.train_final_network(final_network, torch_data, torch_label, LEARNING_RATE, EPOCH_NUMBER,
                           weight_decay=WEIGHT_DELAY, batch_size=BATCH_SIZE, file_name=FILE_NAME)

    # 进行测试集合的验证
    test_data, file_name_col, _, _ = da.csv_handle("../data/test_for_rnn.csv", is_test=True)
    test_data = torch.reshape(test_data, (-1, TIME_STEP, INPUT_SIZE))
    fe.write_result_to_csv("../data/test_for_rnn.csv", "result/result_cnn_01.csv",
                           tp.test_process(test_data, FILE_NAME, gpu_available), TIME_STEP)
