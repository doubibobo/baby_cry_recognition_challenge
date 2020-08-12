from data.code.MLP import network as net
from data.code.tools.data_tools import data_analysis as da
from data.code.tools.network_tools import train_process as tp
from data.code.tools.training_tools import gpu_selector as gs
from data.code.tools.feature_tools import feature_extractor as fe

K = 10                          # 进行10折交叉验证
EPOCH_NUMBER = 1000             # 循环迭代次数为20
LEARNING_RATE = 0.001           # 学习率
WEIGHT_DELAY = 0
BATCH_SIZE = 32
INPUT_SIZE = 134
OUTPUT_SIZE = 6
FILE_NAME = "models/MLP_network_bs_32_epoch_100_lr_small_bn_input_size_01.pkl"


if __name__ == '__main__':
    gpu_available = gs.gpu_selector()

    # 读取数据
    torch_data, torch_label = da.csv_handle("../csv_data/train_mfcc_128_new.csv")
    # 构造神经网络
    network = [net.Network(INPUT_SIZE, OUTPUT_SIZE) for _ in range(K)]
    print(network[0])
    # 进行训练
    tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
                     network_filename=FILE_NAME, gpu_available=gpu_available)
    # 不进行训练集合验证集合的划分，将全部数据拿来训练
    final_network = net.Network(INPUT_SIZE, OUTPUT_SIZE)
    tp.train_final_network(final_network, torch_data.float(), torch_label, LEARNING_RATE, EPOCH_NUMBER, WEIGHT_DELAY,
                           BATCH_SIZE, FILE_NAME, gpu_available)

    # 进行测试集验证
    test_data, file_name = da.csv_handle("../csv_data/test_extend.csv", is_test=True)
    # 返回测试集的结果，为list
    test_result = tp.test_process(test_data, FILE_NAME, gpu_available)

    fe.write_result_to_csv("../csv_data/test_extend.csv", "models/result-20.csv",
                           tp.test_process(test_data, FILE_NAME, gpu_available))
