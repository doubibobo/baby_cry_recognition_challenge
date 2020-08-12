import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from data.code.SVM.SVM import SVM


def create_data():
    """
    模拟生成训练用的数据集，iris是sk-learn自带的鸢尾花数据集
    数据集包含了四个特征：花萼长度,花萼宽度,花瓣长度,花瓣宽度
    共有三种不同种类的鸢尾花，包括：setosa, versicolor, virginica
    iris的内容：
        csv_data ： 鸢尾花样本数据
        target ： 目标分类结果
        target_names ： 目标分类结果名称
        feature_names : 目标特征的名称
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    # 这里没有使用全部的样本数据，只用了前100条训练数据
    # 亦没有使用全部的特征，只用了花萼长度(sepal length)和花萼宽度(sepal width)
    # 返回的是前两列数据data[:, :2] 和 标签数据data[:, -1]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]


if __name__ == '__main__':
    start_time = time.time()
    data, label = create_data()
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.25)
    # 做出相应的样本图, [:50, 0]表示前50行数据, [50:, 0]表示后50行数据
    # legend()为图例函数
    plt.scatter(data[:50, 0], data[:50, 1], label='0')
    plt.scatter(data[50:, 0], data[50:, 1], label='1')
    plt.legend()
    plt.show()

    # 初始化SVM类
    svm = SVM(data_train, label_train, 10)

    # 开始训练
    print("start to train")
    svm.train()
    print("train down")

    # 开始测试
    print("start to test")
    accuracy = svm.test(data_test, label_test)
    print("the accuracy is: %d" % (accuracy * 100), '%')
    print("test down")

    # 打印时间
    print('time span: ', time.time() - start_time)
