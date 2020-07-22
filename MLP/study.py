import data.code.tools.data_analysis as da
import data.code.MLP.network as net

from data.code.tools import build_file_index as bf
from data.code.tools import feature_extrator as fe
from data.code.tools.network_tools import train_process as tp

K = 10                      # 进行10折交叉验证
EPOCH_NUMBER = 10000        # 循环迭代次数为20
LEARNING_RATE = 0.001       # 学习率
BATCH_SIZE = 512
INPUT_SIZE = 26
OUTPUT_SIZE = 6
FILE_NAME = "MLP_network.pkl"


if __name__ == '__main__':
    # 建立文件路径与标签的索引
    file_label_indexes = bf.get_filename("train")
    # 获取频谱图
    fe.extract_spectrogram(file_label_indexes, "train")
    # 写入到csv文件中
    headers = fe.extract_features()
    fe.write_data_to_csv_file(headers, file_label_indexes, "data.csv", "train")
    # 读取数据
    torch_data, torch_label = da.csv_handle("../data/data.csv")
    # 构造神经网络
    network = net.Network(INPUT_SIZE, OUTPUT_SIZE)
    # 进行训练
    tp.train_process(torch_data, torch_label, network, K, LEARNING_RATE, EPOCH_NUMBER, BATCH_SIZE,
                     network_filename=FILE_NAME)

    # 进行测试集合的验证
    test_label_indexes = bf.get_filename("test")
    fe.extract_spectrogram(test_label_indexes, "test")
    fe.write_data_to_csv_file(headers, test_label_indexes, "test.csv", "test")

    test_data, _ = da.csv_handle("test.csv")
    fe.write_result_to_csv("test.csv", "result.csv", tp.test_process(test_data, FILE_NAME))
