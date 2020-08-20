import torch

from data.code.LSTM import network as net
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.feature_tools import feature_extractor as fe
from data.code.tools.network_tools import train_process as tp
from data.code.tools.training_tools import gpu_selector as gs
from torchsummary import summary

# 定义超参数
K = 10
EPOCH_NUMBER = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TIME_STEP = 201
INPUT_SIZE = 26
HIDDEN_SIZE = 64
OUTPUT_SIZE = 6
FILE_NAME = "model/LSTM_network_0819_01.pkl"


if __name__ == '__main__':
    gpu_available = gs.gpu_selector()

    # 读取数据，并进行数据的重塑
    torch_data, torch_label = da.csv_handle("../data/csv_data/train_mfcc_20_new_2s_time.csv")
    torch_data = torch.reshape(torch_data, (-1, TIME_STEP, INPUT_SIZE))
    torch_label = torch_label[0:len(torch_label):TIME_STEP]

    # 构建LSTM网络
    network = [net.LSTMClassify(INPUT_SIZE, HIDDEN_SIZE, 1, OUTPUT_SIZE, BATCH_SIZE, TIME_STEP) for i in range(K)]
    for i in range(len(network)):
        network[i].apply(net.weights_init)

    # 打印神经网络的结构
    # summary(network[0], (1, TIME_STEP, INPUT_SIZE), device="cpu")

    # 训练网络
    tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
                     network_filename=FILE_NAME, gpu_available=gpu_available)

    # 进行测试集合的验证
    test_data, file_name_col, _, _ = da.csv_handle("../data/csv_data/test_mfcc_20_new_2s_time.csv", is_test=True)
    test_data = torch.reshape(test_data, (-1, TIME_STEP, INPUT_SIZE))
    fe.write_result_to_csv("../data/csv_data/test_mfcc_20_new_2s_time.csv", "result/result_0812_01.csv",
                           tp.test_process(test_data, FILE_NAME, gpu_available), TIME_STEP)
