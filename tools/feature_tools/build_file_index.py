"""
建立训练文件的索引，并将其存储到字典indexes中
格式为： (file_path, file_label)
"""
import os

filePath = "C:\\Users\\doubibobo\\Desktop\\baby_cry_competition\\data\\code\\data\\wav_data\\original-data\\"


def get_filename(selection, file=filePath):
    """
    获取音频文件
    :param selection: 选择使用哪个文件夹，可选择test和train两个
    :param file： 数据的根目录，默认为origin-csv_data
    :return: 字典列表
    """
    label_directory = os.listdir(file + selection)
    dictionary = {}
    for i in range(len(label_directory)):
        son_file_path = file + selection + "/" + label_directory[i]
        if selection == "train":
            dictionary[label_directory[i]] = os.listdir(son_file_path)
        elif selection == "test":
            dictionary[son_file_path] = "awake"
    if selection == "test":
        return dictionary

    # 遍历字典列表，字典的key为：文件名称，value为标签
    indexes = {}
    for key, value in dictionary.items():
        for file_name in range(len(value)):
            indexes[value[file_name]] = key

    del dictionary
    return indexes
