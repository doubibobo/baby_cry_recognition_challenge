import numpy
import torch

from data.code.LSTM import network as net
from data.code.tools.feature_tools import build_file_index as bf, feature_extractor as fe
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.network_tools import train_process as tp
from data.code.tools.training_tools import gpu_selector as gs


# 定义超参数
K = 10
EPOCH_NUMBER = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TIME_STEP = 201
INPUT_SIZE = 26
HIDDEN_SIZE = 64
OUTPUT_SIZE = 6
FILE_NAME = "model/LSTM_network_04.pkl"


if __name__ == '__main__':
    gpu_available = gs.gpu_selector()

    # 读取数据，并进行数据的重塑
    torch_data, torch_label = da.csv_handle("../data/csv_data/train_mfcc_20_new_2s_time.csv")
    torch_data = torch.reshape(torch_data, (-1, TIME_STEP, INPUT_SIZE))
    torch_label = torch_label[0:len(torch_label):TIME_STEP]

    # 构建LSTM网络
    network = [net.LSTMClassify(INPUT_SIZE, HIDDEN_SIZE, 1, OUTPUT_SIZE) for i in range(K)]
    print(network)
    # 估测模型所占用的内存
    param = sum([numpy.prod(list(p.size())) for p in network[0].parameters()])
    # 下面的type_size是4，因为我们的参数是float32也就是4B，4个字节
    print('Model {} : params: {:4f}M'.format(network[0]._get_name(), param * 4 / 1000 / 1000))
    # 训练网络
    tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
                     network_filename=FILE_NAME, gpu_available=gpu_available)
    # # 进行测试集合的验证
    # test_data, file_name_col, _, _ = da.csv_handle("../csv_data/test_for_rnn.csv", is_test=True)
    # test_data = torch.reshape(test_data, (-1, TIME_STEP, INPUT_SIZE))
    # fe.write_result_to_csv("../csv_data/test_for_rnn.csv", "../csv_data/result.csv",
    #                        tp.test_process(test_data, FILE_NAME, gpu_available), TIME_STEP)
