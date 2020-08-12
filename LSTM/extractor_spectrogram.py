from data.code.tools.feature_tools import build_file_index as bf
from data.code.tools.feature_tools import feature_extractor as fe
from data.code.tools.image_tools import to_gray as tg
if __name__ == '__main__':
    # 建立训练集文件路径与标签的索引
    file_label_indexes = bf.get_filename("train")
    print(file_label_indexes)
    # 提取频谱图
    fe.extract_spectrogram(file_label_indexes, "train", True)
    # 将其转化为灰度图
    tg.spectrogram_to_gray(file_label_indexes, True)

    # 建立测试集合文件路径与标签的索引
    file_label_indexes = bf.get_filename("test", tg.filePath)
    print(file_label_indexes)
    # 提取频谱图
    fe.extract_spectrogram(file_label_indexes, "test", True)

    # 将其转化为灰度图
    tg.spectrogram_to_gray(file_label_indexes, False)