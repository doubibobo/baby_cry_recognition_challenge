from data.code.SVM.SVM import SVM
from data.code.data_analysis import get_k_fold_data

import numpy
import torch


class MultiSVM:
    """
    用于多分类的SVM类
    """
    def __init__(self, classes_number, data, label, sigma, k_number):
        """
        构造方法
        :param classes_number:  类别的数量
        :param data:            训练集
        :param label:           训练集标签
        :param sigma:           高斯核中的sigma值
        :param k_number:        交叉验证折数
        """
        self.classes_number = classes_number
        self.data = data
        self.label = label
        self.sigma = sigma
        self.svm_number = int((classes_number * (classes_number - 1)) / 2)
        self.classes_index = []
        self.svm = numpy.empty(self.svm_number, dtype=SVM)
        self.accuracy = numpy.empty(self.svm_number, dtype=float)

        self.mapping = []

        self.K_NUMBER = k_number

    def create_label_index(self):
        """
        创建某一个类别的索引
        :return: 索引值
        """
        for i in range(self.classes_number):
            self.classes_index.append([])
        for i in range(len(self.label)):
            for j in range(self.classes_number):
                if self.label[i] == j:
                    self.classes_index[j].append(i)
                    continue

    def create_dataset(self, sequence):
        """
        处理数据集：单独取出某个类别的数据
        方法：取出label为sequence的index，然后取data[index]
        :param sequence: 类别号码
        :return: 特征，标签
        """
        return self.data[self.classes_index[sequence], :], self.label[self.classes_index[sequence]]

    def create_multi_svm(self):
        """
        构造用于多分类的svm
        :return: 无返回值
        """
        self.create_label_index()
        svm_number = -1
        for i in range(self.classes_number):
            data_i, label_i = self.create_dataset(i)
            # 将i类的标签设置为正类的1
            label_i = torch.ones(len(label_i))
            for j in range(i + 1, self.classes_number):
                data_j, label_j = self.create_dataset(j)
                # 将j类的标签设置为负类的-1
                label_j = (-1) * torch.ones(len(label_j))
                data_sum, label_sum = torch.cat((data_i, data_j)), torch.cat((label_i, label_j))
                # 初始化svm的最好精确度为0
                best_accuracy = 0
                svm_number += 1
                # 标记好是哪两类的svm
                self.mapping.append([i, j])
                # 用k折划分法对数据进行分类
                for k in range(self.K_NUMBER):
                    data_train, label_train, data_test, label_test = get_k_fold_data(self.K_NUMBER, k, data_sum, label_sum)
                    # 创建i和j类的SVM
                    svm = SVM(data_train, label_train, self.sigma)
                    # 对每一份数据进行训练
                    svm.train()
                    # 进行测试
                    accuracy = svm.test(data_test, label_test)
                    if accuracy >= best_accuracy:
                        self.svm[svm_number] = svm
                        self.accuracy[svm_number] = accuracy

    def decide(self, x):
        """
        给出样本x的类别预测结果
        :param x: 样本x
        :return: 结果
        """
        result = [0 for _ in range(self.classes_number)]
        for i in range(self.svm_number):
            predict_temp = self.svm[i].predict(x)
            # print("this is the %dth svm:" % i)
            # print(predict_temp)
            if predict_temp == 1:
                # print(self.mapping[i][0])
                result[self.mapping[i][0]] += 1
            else:
                # print(self.mapping[i][1])
                result[self.mapping[i][1]] += 1
        return result.index(max(result))

    def test(self, data_test, label_test):
        """
        测试多分类SVM的效果
        :param data_test:   测试数据集
        :param label_test:  测试标签
        :return: 正确率
        """
        error_count = 0
        for i in range(len(data_test)):
            print("test: %d, number: %d" % (i, len(data_test)))
            # 获取测试结果
            result = self.decide(data_test[i])
            if result != label_test[i]:
                error_count += 1
        return 1 - error_count / len(data_test)
