import torch.cuda as cuda

import data.code.MLP.network as net
import data.code.tools.data_analysis as da

from data.code.tools import build_file_index as bf
from data.code.tools import feature_extractor as fe
from data.code.tools.network_tools import train_process as tp

K = 10                      # 进行10折交叉验证
EPOCH_NUMBER = 1000         # 循环迭代次数为20
LEARNING_RATE = 0.001        # 学习率
WEIGHT_DELAY = 0
BATCH_SIZE = 32
INPUT_SIZE = 38
OUTPUT_SIZE = 6
FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_01.pkl"


if __name__ == '__main__':
    # # 建立文件路径与标签的索引
    # file_label_indexes = bf.get_filename("train")
    # # 获取频谱图
    # fe.extract_spectrogram(file_label_indexes, "train")
    # # 写入到csv文件中
    # headers = fe.extract_features()
    # fe.write_data_to_csv_file(headers, file_label_indexes, "data.csv", "train")

    # 查看GPU相关信息
    gpu_available = cuda.is_available()
    device_name = cuda.get_device_name(1)
    device_capability = cuda.get_device_capability(1)
    print(gpu_available)
    print(device_name)
    print(device_capability)
    if gpu_available:
        print("device_number is ", 1)
        cuda.set_device(1)
    #
    # # 读取数据
    # torch_data, torch_label = da.csv_handle("../data/data_extend.csv")
    # # 构造神经网络
    # network = [net.Network(INPUT_SIZE, OUTPUT_SIZE) for _ in range(K)]
    # print(network[0])
    # # 进行训练
    # tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
    #                  network_filename=FILE_NAME, gpu_available=gpu_available)
    # #
    # # 进行测试集合的验证
    # test_label_indexes = bf.get_filename("test")
    # fe.extract_spectrogram(test_label_indexes, "test")
    # fe.write_data_to_csv_file(headers, test_label_indexes, "test.csv", "test")

    test_data, _ = da.csv_handle("../data/test_extend.csv")
    fe.write_result_to_csv("../data/test_extend.csv", "models/result-06.csv", tp.test_process(test_data, FILE_NAME, gpu_available))
