import cv2
import os

filePath = "/home/doubibobo/桌面/婴儿啼哭识别 挑战赛/data/code/LSTM/image_data/"
filePathGray = "/home/doubibobo/桌面/婴儿啼哭识别 挑战赛/data/code/LSTM/image_data_gray/"
filePathGrayServer = "/home/zhuchuanbo/competition/data/code/LSTM/image_data_gray/"


def spectrogram_to_gray(file_name, is_train, image_data_gray="image_data_gray/"):
    """
    将频谱图转化为灰度图
    :param file_name: 原始文件名
    :param is_train: 是否是训练集
    :param image_data_gray： 新文件路径
    :return: 无返回值
    """
    selection = "train" if is_train else "test"
    # 创建对应类的频谱图文件夹
    for key, value in file_name.items():
        # k即是原始文件路径
        image_gray = cv2.imread((filePath + selection + "/" + value + "/" + key)
                                if is_train else key, cv2.IMREAD_GRAYSCALE)
        # 存储对应灰度图
        if is_train:
            cv2.imwrite(image_data_gray + selection + '/' + value + '/' + key, image_gray)
        else:
            cv2.imwrite(image_data_gray + selection + '/' + os.path.split(key)[1], image_gray)