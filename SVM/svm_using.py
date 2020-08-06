import torch
import time
import pickle

from data.code.tools.data_tools.data_analysis import csv_handle
from data.code.tools.data_tools.data_analysis import get_k_fold_data
from data.code.SVM.multi_svm import MultiSVM

K_NUMBER = 10  # 进行10折交叉验证
CLASSES_NUMBER = 6  # 总共有六种类型的数据
SIGMA = 10  # 取高斯核的sigma为10


def gpu_setting(use_number):
    """
    训练时的GPU设置
    return: 是否可用GPU的标志
    """
    use_gpu = torch.cuda.is_available()
    gpu_number = torch.cuda.device_count()

    if use_gpu and use_number < gpu_number:
        print('*' * 25, "GPU信息展示", '*' * 25)
        print(torch.cuda.get_device_capability(use_number))
        print(torch.cuda.get_device_name(use_number))
        print(torch.cuda.get_device_properties(use_number))
        torch.cuda.set_device(0)
    return use_gpu


def train_processing(data_train, label_train):
    """
    k折划分后的训练过程，并且要求使用最好的神经网络
    :param data_train:      数据集
    :param label_train:     数据标签
    :return:
    """
    best_accuracy = 0
    # 首先将训练集的全部数据
    for m in range(K_NUMBER):
        print('*' * 25, '第', m + 1, '折SVM多分类器开始', '*' * 25)
        data_train_sum, label_train_sum, data_test_sum, label_test_sum = get_k_fold_data(K_NUMBER, m, data_train,
                                                                                         label_train)
        multi_svm = MultiSVM(CLASSES_NUMBER, data_train_sum, label_train_sum, SIGMA, K_NUMBER)
        # 训练每一个MultiSVM
        multi_svm.create_multi_svm()
        # 测试MultiSVM的训练结果
        accuracy = multi_svm.test(data_test_sum, label_test_sum)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            # torch.save(multi_svm, 'multi_svm.pkl')
            pkl_filename = "models/multi_svm.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(multi_svm, file)
        print('train_accuracy: %.6f' % accuracy)
        print('*' * 25, '第', m + 1, '折SVM多分类器结束', '*' * 25)
        break

    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    print('train_accuracy_sum: %.6f' % best_accuracy)


def test_processing(data_test):
    """
    在测试集合上进行SVM分类效果的测试
    :param data_test: 测试集合
    :return: 无返回值
    """
    # prediction = torch.load('multi_svm.pkl')(data_test.float())
    pk_filename = "models/multi_svm_1.pkl"
    with open(pk_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    prediction = []
    i = 0
    for test_sample in data_test:
        print('*' * 50)
        print("this is the %dth sample" % i)
        prediction.append(pickle_model.decide(test_sample))
        i += 1
        print('*' * 50)
    print(prediction)
    return prediction


if __name__ == '__main__':
    time_start = time.time()
    torch_data, torch_label = csv_handle("../data/data_extend.csv")
    # 进行训练
    train_processing(torch_data, torch_label)
    # # 进行测试集合的验证
    # headers = extract_features()

    # test_data, _ = csv_handle("test_1.csv")
    # write_result_to_csv("test_1.csv", "result_gpu.csv", test_processing(test_data))
    # print('time span: ', time.time() - time_start)
