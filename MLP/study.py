from data.code.tools.feature_tools import build_file_index as bf, feature_extractor as fe
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.training_tools import gpu_selector as gs
from data.code.tools.network_tools import train_process as tp
from data.code.MLP import network as net

import torch

K = 10                          # 进行10折交叉验证
EPOCH_NUMBER = 200              # 循环迭代次数为20
LEARNING_RATE = 0.001           # 学习率
WEIGHT_DELAY = 0
BATCH_SIZE = 32
INPUT_SIZE = 26
OUTPUT_SIZE = 6
FILE_NAME = "MLP_network_bs_32_epoch_1000_lr_small_bn_13.pkl"


if __name__ == '__main__':
    # 建立文件路径与标签的索引
    # file_label_indexes = bf.get_filename("train")
    # # 获取频谱图
    # fe.extract_spectrogram(file_label_indexes, "train")
    # # 写入到csv文件中
    # headers = fe.extract_features()
    # # fe.write_data_to_csv_file(headers, file_label_indexes, "test_cqt_new.csv", "train")
    # #
    # # 测试集特征提取
    # test_label_indexes = bf.get_filename("test")
    # # fe.extract_spectrogram(test_label_indexes, "test")
    # fe.write_data_to_csv_file(headers, test_label_indexes, "test_cqt_new.csv", "test")

    gpu_available = gs.gpu_selector()

    # # 读取数据
    # torch_data, torch_label = da.csv_handle("../data/train_mfcc_20_new_2s_vad.csv")
    # # 构造神经网络
    # network = [net.Network(INPUT_SIZE, OUTPUT_SIZE) for _ in range(K)]
    # print(network[0])
    # # 进行训练
    # tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
    #                  network_filename=FILE_NAME, gpu_available=gpu_available)
    # # 不进行训练集合验证集合的划分，将全部数据拿来训练
    # final_network = net.Network(INPUT_SIZE, OUTPUT_SIZE)
    # tp.train_final_network(final_network, torch_data.float(), torch_label, LEARNING_RATE, EPOCH_NUMBER, WEIGHT_DELAY,
    #                        BATCH_SIZE, FILE_NAME, gpu_available)

    # 进行测试集验证
    test_data, file_name = da.csv_handle("../data/test_mfcc_20_new_2s_vad.csv", is_test=True)
    results = tp.test_process(test_data, FILE_NAME, gpu_available)

    position = torch.max(results, 1)[1]

    # 将数据转化为cpu类型和list类型
    results_torch = results.cpu()
    # file_name = file_name.numpy().tolist()
    results = results_torch.detach().numpy()
    # 创建结果字典
    results_dictionary = {}
    for i in range(len(file_name)):
        # 判断 file_name是否已经写入字典中
        if not file_name[i] in results_dictionary.keys():
            results_dictionary[file_name[i]] = [0, 0, 0, 0, 0, 0]

        # TODO 是否考虑对概率做一个加权平均，目前是舍弃掉精确度<=0.9的预测值
        if max(results[i]) >= 0.9:
            results_dictionary[file_name[i]] = results_dictionary[file_name[i]] + results[i]

    print(len(results_dictionary))

    vad = []

    # 获取整体的预测结果，取每段语音预测结果的最大值的下标
    for key, value in results_dictionary.items():
        value = value.tolist()
        results_dictionary[key] = value.index(max(value))
        vad.append(value)
    final_results, final_names = [], []
    for key, value in results_dictionary.items():
        final_results.append(value)
        final_names.append(key)
    final_results = list(results_dictionary.values())
    # final_names = list(results_dictionary.keys())

    fe.write_valid_dataset_csv("/home/zhuchuanbo/competition/data/code/MLP/data_test/", 'results_vad_02.csv', vad, final_results)


    print("file_name:")
    print(file_name)
    print("results")
    print(results)
    fe.write_result_to_csv("../data/test_mfcc_20_new_2s_vad.csv", "models/result-120.csv", final_results, final=final_names)
    # fe.write_result_to_csv("../data/test_mfcc_20_new_2s_vad.csv", "models/result-108.csv", results)

