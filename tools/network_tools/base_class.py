import numpy
import torch
from PIL import Image
from torch.utils.data import Dataset


class TrainDataSet(Dataset):
    """
    构造数据集类
    """

    def __init__(self, data_train, label_train):
        self.x_data = data_train
        self.y_data = label_train
        self.length = len(label_train)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.length


# 建立训练时的标签和语音类别的映射关系
label_classes = {"awake": 0, "diaper": 1, "hug": 2, "hungry": 3, "sleepy": 4, "uncomfortable": 5}


class ImageDataSet(Dataset):
    """
    构造图像数据集
    """

    def __init__(self, filepath_label_indexes, root_path):
        """
        初始化函数
        :param filepath_label_indexes: <训练集文件名，标签>
        :param root_path: 训练集文件的根目录
        """
        self.image = []
        self.label = []
        for key, value in filepath_label_indexes:
            image = Image.open(root_path + '/' + value + "/" + key)
            self.image.append(torch.from_numpy(numpy.asarray(image)))
            self.label.append(label_classes[value])

        self.length = len(filepath_label_indexes)

    def __getitem__(self, item):
        # TODO 这里由于每次取值的时候都会读取图片，势必会降低GPU的使用效率
        return self.image[item], self.label[item]

    def __len__(self):
        return self.length
