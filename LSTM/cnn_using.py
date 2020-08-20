import torch

from torchsummary import summary

from data.code.LSTM import network as net
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.training_tools import gpu_selector as gs
from data.code.tools.network_tools import train_process as tp
from data.code.tools.feature_tools import feature_extractor as fe

# 定义超参数
K = 10
EPOCH_NUMBER = 250
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TIME_STEP = 201
INPUT_SIZE = 26
OUTPUT_SIZE = 6
WEIGHT_DELAY = 1e-8
FILE_NAME = "model/cnn_model_0819_01.pkl"


if __name__ == '__main__':
    # 使用gpu进行训练
    gpu_available = gs.gpu_selector()

    # 构建卷积神经网络
    network = [net.CNNClassify(1, OUTPUT_SIZE, (3, 1), (3, 1), 0, (2, 2)) for i in range(K)]
    for i in range(len(network)):
        network[i].apply(net.weights_init)

    # 打印神经网络的结构
    summary(network[0], (1, TIME_STEP, INPUT_SIZE), device="cpu")

    # 读取数据，并进行数据的重塑
    torch_data, torch_label = da.csv_handle("../data/csv_data/train_mfcc_20_new_2s_time.csv")
    torch_data = torch.reshape(torch_data, (-1, 1, TIME_STEP, INPUT_SIZE))
    torch_label = torch_label[0:len(torch_label):TIME_STEP]

    # 训练网络
    tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
                     network_filename=FILE_NAME, gpu_available=gpu_available)
    # 将所有数据投入到最终的神经网络进行训练
    final_network = net.CNNClassify(1, OUTPUT_SIZE, (3, 1), (3, 1), 0, (2, 2))
    tp.train_final_network(final_network, torch_data, torch_label, LEARNING_RATE, EPOCH_NUMBER,
                           weight_decay=WEIGHT_DELAY, batch_size=BATCH_SIZE, file_name=FILE_NAME)

    # 进行测试集合的验证
    test_data, file_name_col, _, _ = da.csv_handle("../data/csv_data/test_mfcc_20_new_2s_time.csv", is_test=True)
    test_data = torch.reshape(test_data, (-1, TIME_STEP, INPUT_SIZE))
    fe.write_result_to_csv("../data/csv_data/test_mfcc_20_new_2s_time.csv", "result/result_cnn_0813_01.csv",
                           tp.test_process(test_data, FILE_NAME, gpu_available), TIME_STEP)
