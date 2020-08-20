from data.code.MLP import network as net
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.network_tools import train_process as tp
from data.code.tools.training_tools import gpu_selector as gs
from data.code.tools.feature_tools import feature_extractor as fe
from data.code.tools.network_tools import voting_test as vt

from torchsummary import summary


K = 10                          # 进行10折交叉验证
EPOCH_NUMBER = 1000          # 循环迭代次数为20
LEARNING_RATE = 0.001           # 学习率
WEIGHT_DELAY = 1e-8
BATCH_SIZE = 32
INPUT_SIZE = 26
OUTPUT_SIZE = 6
FILE_NAME = "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082030.pkl"


if __name__ == '__main__':
    gpu_available = gs.gpu_selector()

    # # 读取数据
    # torch_data, torch_label = da.csv_handle("../data/csv_data/train_mfcc_20_new_3s.csv")
    # # 构造神经网络
    # network = [net.Network_2(INPUT_SIZE, OUTPUT_SIZE) for _ in range(K)]
    # for i in range(len(network)):
    #     network[i].apply(net.weights_init)
    # print(network[0])
    #
    # # summary(network[0], (1, INPUT_SIZE), device="cpu")
    #
    # # 进行训练
    # tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
    #                  network_filename=FILE_NAME, gpu_available=gpu_available)
    # # 不进行训练集合验证集合的划分，将全部数据拿来训练
    # final_network = net.Network(INPUT_SIZE, OUTPUT_SIZE)
    # tp.train_final_network(final_network, torch_data.float(), torch_label, LEARNING_RATE, EPOCH_NUMBER, WEIGHT_DELAY,
    #                        BATCH_SIZE, FILE_NAME, gpu_available)

    # 进行测试集验证
    test_data, file_name = da.csv_handle("../data/csv_data/test_mfcc_20_new_3s.csv", is_test=True)

    # 对每一个模型进行测试
    combine_test_result = []
    combine_files = []
    PKL_FILE_NAME = ["models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl0",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl1",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl2",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl3",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl4",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl5",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl6",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl7",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081902.pkl",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081901.pkl",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081706.pkl",

                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082020.pkl0",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082020.pkl1",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082020.pkl2",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082020.pkl3",

                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082030.pkl0",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082030.pkl1",
                     "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082030.pkl2",

                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081705.pkl",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081701.pkl",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_081702.pkl",

                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl8",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl9",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082002.pkl",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl0",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl1",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl2",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl3",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl4",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl5",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl6",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl7",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl8",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl9",
                     # "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_082010.pkl"]
    ]

    for ONE_PKL_FILE in PKL_FILE_NAME:
        # 返回测试集的结果，为list
        test_result = tp.test_process(test_data, ONE_PKL_FILE, gpu_available)
        # 进行投票，获取最终的结果
        files_result_dictionary = vt.voting(test_result, file_name)
        combine_test_result.append(files_result_dictionary)

    final_files, final_result = vt.final_voting(combine_test_result)

    # 最后进行一轮总投票
    fe.write_result_to_csv("../data/csv_data/test_mfcc_20_new_3s.csv", "models/result-0820-combine-01.csv",
                           final_result, final=final_files)

    # fe.write_result_to_csv("../data/csv_data/test_mfcc_20_new_15s.csv", "models/result-082002.csv", result, final=files)
    # #
    # fe.write_result_to_csv("../data/csv_data/test_mfcc_20_new_15s.csv", "models/result-081804.csv",
    #                        tp.test_process(test_data, FILE_NAME, gpu_available))
