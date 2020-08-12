import math
import random

import numpy as np


class SVM:
    """
    支持向量机类
    """

    def __init__(self, features, labels, sigma, max_iteration=100, c=200, tolerance=0.001, kernel='linear'):
        """
        构造方法,初始化相关参数
            features:               特征数据集
            labels:                 数据集的标签
            tolerance:              为松弛变量（容忍度）,默认为0.001,用来判断某个变量是否为0
            E：                      SMO运算过程中的Ei
            k:                      高斯核函数矩阵，提前计算
            C：                      惩罚参数
            support_vector_index:   支持向量的索引
        :param max_iteration: 最大迭代次数
        :param kernel: 使用的核函数
        :param features: 特征数据集
        :param sigma 高斯核中的sigma
        :param labels: 数据集标签
        :return: 无返回值
        """
        self.max_iteration = max_iteration
        self._kernel = kernel
        self.sample_number, self.feature_number = features.shape
        self.data = features
        self.label = labels
        self.sigma = sigma
        self.b = 0.0

        # 将E(i)保存在一个列表里
        self.alpha = np.ones(self.sample_number)
        self.tolerance = tolerance
        self.C = c

        self.k = self.calculate_kernel()
        self.E = [self.calculate_error(i) for i in range(self.sample_number)]

        self.support_vector_index = []

    def calculate_error(self, i):
        """
        返回实际值和预测值之间的差距
        :param i: 元素的下标
        :return: 相关error
        """
        return self.calculate_gxi(i) - self.label[i]

    def calculate_kernel(self):
        """
        计算核函数
        这里使用的是高斯核
        :return: 高斯核矩阵
        """
        # 初始化高斯核结果矩阵，大小为sample_number * sample_number
        k = [[0 for i in range(self.sample_number)] for j in range(self.sample_number)]

        # 大循环遍历Xi
        for i in range(self.sample_number):
            # 得到当前的样本X
            X = self.data[i, :]
            for j in range(self.sample_number):
                Z = self.data[j, :]
                # 首先计算分子
                # ATTENTION 首先确定这样的计算方式是否有误
                # 通过点乘来计算数值
                factor = np.linalg.norm((X - Z))
                # 其次计算总的结果
                result = np.exp(-1 * factor / (2 * self.sigma ** 2))
                # 存放计算结果
                k[i][j] = result
                k[j][i] = result
        return k

    def calculate_gxi(self, i):
        """
        计算g(i) ，根据两个变量的二次规划的求解方法
        :param i: data的下标
        :return: g(x_i)的值
        """
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零alpha，i为非零alpha的下标
        for j in index:
            # 计算gxi
            gxi += self.alpha[j] * self.label[j] * self.k[j][i]
        # 累加偏置值
        gxi += self.b
        return gxi

    def is_satisfy_ktt(self, i):
        """
        判断第i个alpha(csv_data)是否满足KTT条件
        是SMO算法的第一个变量选择过程
        :param i alpha(csv_data)的下标
        :return:
            True ： 满足
            False ： 不满足
        """
        gxi = self.calculate_gxi(i)
        yi = self.label[i]
        if (self.alpha[i] == 0) and (yi * gxi >= 1):
            return True
        if (0 < self.alpha[i] < self.C) and (yi * gxi == 1):
            return True
        if (self.alpha[i] == self.C) and (yi * gxi <= 1):
            return True
        return False

    def select_alpha_second(self, e1, i):
        """
        SMO中选择第二个变量的方法
        :param e1: 第一个变量的e1
        :param i: 第一个变量alpha的下标
        :return: e2和第二个变量alpha的下标
        """
        e2 = 0              # 初始化e2
        max_e1_e2 = -1      # 初始化|e1 - e2|为-1
        max_index = -1      # 初始化第二个变量的下标为-1

        # 首先获得ei为0的元素对应的列表
        no_zero_e = [i for i, e_i in enumerate(self.E) if e_i != 0]
        for j in no_zero_e:
            e2_temp = self.calculate_error(j)
            if math.fabs(e1 - e2_temp) >= max_e1_e2:
                max_e1_e2 = math.fabs(e1 - e2_temp)
                e2 = e2_temp
                max_index = j

        # 如果列表中没有非零元素了，对应程序最开始运行的情况
        if max_index == -1:
            max_index = i
            while max_index == i:
                # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                max_index = int(random.uniform(0, self.sample_number))
            # 获得e2
            e2 = self.calculate_error(max_index)

        return e2, max_index

    def train(self):
        """
        训练过程
        :return: 无返回值
        """
        iteration_step = 0          # 迭代次数
        parameter_changed = 1       # 单次训练中如果参数有改变则增加1

        while (iteration_step < self.max_iteration) and parameter_changed > 0:
            print('iteration:%d max_iteration:%d' % (iteration_step + 1, self.max_iteration))
            iteration_step += 1
            parameter_changed = 0

            for i in range(self.sample_number):
                """
                第一步：选择优化变量alpha_i 和alpha_j
                """
                # 大循环选择第一个变量
                if not self.is_satisfy_ktt(i):
                    # 选择第二个变量
                    e1 = self.calculate_error(i)
                    e2, j = self.select_alpha_second(e1, i)
                    # 获得两个变量的标签
                    label_i = self.label[i]
                    label_j = self.label[j]
                    """
                    第二步：均是求解选择的两个变量的最优化问题（即：alpha_i 和 alpha_j的最优化问题）
                    """
                    # 复制初始可行解为alpha_old
                    alpha_old_i = self.alpha[i]
                    alpha_old_j = self.alpha[j]

                    # 根据标签是否一致确定裁剪盒子的边界(二分类问题)
                    if label_i == label_j:
                        L = max(0, alpha_old_i + alpha_old_j - self.C)
                        H = min(self.C, alpha_old_i + alpha_old_j)
                    else:
                        L = max(0, alpha_old_j - alpha_old_i)
                        H = min(self.C, self.C + alpha_old_j - alpha_old_i)
                    # 判断是否还能够继续优化
                    if L == H:
                        continue
                    # 求解alpha_new_j
                    alpha_new_j = alpha_old_j + ((label_j*(e1 - e2))/(self.k[i][i] + self.k[j][j] - 2*self.k[i][j]))
                    # 进一步裁剪盒子（盒子中线段的边界）
                    if alpha_new_j < L:
                        alpha_new_j = L
                    elif alpha_new_j > H:
                        alpha_new_j = H
                    # 将其再次带入约束条件，求解alpha_new_1
                    alpha_new_i = alpha_old_i + label_i * label_j * (alpha_old_j - alpha_new_j)

                    b_new_i = (-1) * e1 - label_i * self.k[i][i] * (alpha_new_i - alpha_old_i) \
                                        - label_j * self.k[j][i] * (alpha_new_j - alpha_old_j) + self.b
                    b_new_j = (-1) * e2 - label_i * self.k[i][j] * (alpha_new_i - alpha_old_i) \
                                        - label_j * self.k[j][j] * (alpha_new_j - alpha_old_j) + self.b
                    # 根据alpha_new_j 和 alpha_new_i的取值范围选择b的值
                    if 0 < alpha_new_i < self.C:
                        b_new = b_new_i
                    elif 0 < alpha_new_j < self.C:
                        b_new = b_new_j
                    else:
                        b_new = (b_new_i + b_new_j) / 2
                    """
                    第三步：根据alpha_i和alpha_j更新预测错误值Ei 和 偏置值b
                    """
                    # 更新alpha, b以及Ei的值
                    self.alpha[i], self.alpha[j] = alpha_new_i, alpha_new_j
                    self.b = b_new
                    self.E[i] = self.calculate_error(i)
                    self.E[j] = self.calculate_error(j)

                    # 判断alpha_new_j的改变量是否足够大，来证明已经开始优化了
                    if math.fabs(alpha_new_j - alpha_old_j) >= 0.00001:
                        parameter_changed += 1

                # 打印迭代轮数，i值
                # print("iteration: %d i:%d, pairs changed %d" % (iteration_step, i, parameter_changed))

        # 迭代完成，即计算完成之后，重新遍历一遍alpha，查找其中的支持向量
        for i in range(self.sample_number):
            if self.alpha[i] > 0:
                self.support_vector_index.append(i)

    def calculate_single_kernel(self, x1, x2):
        """
        单独计算核函数
        :param x1: 支持向量
        :param x2: 待判别向量
        :return: 计算结果
        """
        return np.exp((-1 * (np.linalg.norm((x1.double() - x2.double())))) / (2 * self.sigma ** 2))

    def predict(self, x):
        """
        SVM预测分类的结果
        :param x: 带判别的标签数据
        :return: 预测值
        """
        result = 0
        for i in self.support_vector_index:
            temp = self.calculate_single_kernel(self.data[i, :], x)
            result += self.alpha[i] * self.label[i] * temp
        result += self.b
        return np.sign(result)

    def test(self, data_test, label_test):
        """
        测试SVM的效果
        :param data_test:  测试数据集
        :param label_test: 测试标签
        :return: 正确率
        """
        error_count = 0     # 预测错误的样本数目
        for i in range(len(data_test)):
            # print("test: %d: %d" % (i, len(data_test)))
            # 获取预测结果
            result = self.predict(data_test[i])
            if result != label_test[i]:
                error_count += 1
        return 1 - error_count / len(data_test)
