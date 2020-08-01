import numpy
import torch

from data.code.LSTM import network as net

from data.code.tools import build_file_index as bf
from data.code.tools import data_analysis as da
from data.code.tools import feature_extrator as fe
from data.code.tools.network_tools import train_process as tp

# 定义超参数
K = 10
EPOCH_NUMBER = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 1
TIME_STEP = 1502
INPUT_SIZE = 26
HIDDEN_SIZE = 64
OUTPUT_SIZE = 6
FILE_NAME = "combine_network_01.pkl"


if __name__ == '__main__':
    # 建立文件路径与标签的索引
    file_label_indexes = bf.get_filename("train")
    print(file_label_indexes)
    tool_dictionary = file_label_indexes.copy()
    # 重新整合一边file_label_indexes，因为有不足15秒的数据导致程序异常
    for key, value in file_label_indexes.items():
        if key != "hug_1.wav":
            tool_dictionary.pop(key)
            continue
        else:
            break

    print(tool_dictionary)

    # 读取数据，并进行数据的重塑
    torch_data, torch_label, file_name, frame_number = da.csv_handle("../data/data_for_rnn_1.csv",
                                                                     "../data/data_for_rnn.csv")
    torch_data = torch.reshape(torch_data, (-1, TIME_STEP, INPUT_SIZE))
    torch_label = torch_label[0:len(torch_label):TIME_STEP]
    print(torch_data.shape)
    # 构建LSTM-CNN网络
    network = [net.CombineClassify(INPUT_SIZE, HIDDEN_SIZE, 1, 1, OUTPUT_SIZE) for i in range(K)]
    # 估测模型所占用的内存
    param = sum([numpy.prod(list(p.size())) for p in network[0].parameters()])
    # 下面的type_size是4，因为我们的参数是float32也就是4B，4个字节
    print('Model {} : params: {:4f}M'.format(network[0]._get_name(), param * 4 / 1000 / 1000))
    # 训练网络
    tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
                     network_filename=FILE_NAME)
    # 进行测试集合的验证
    test_data, _ = da.csv_handle("../data/test_for_rnn.csv")
    test_data = torch.reshape(test_data, (-1, TIME_STEP, INPUT_SIZE))
    fe.write_result_to_csv("../data/test_for_rnn.csv", "result/result_combine_01.csv", tp.test_process(test_data, FILE_NAME))
