import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import torch

import data.code.build_file_index as bf

"建立训练时的标签和语音类别的映射关系"
classes_labels = ["awake", "diaper", "hug", "hungry", "sleepy", "uncomfortable"]

types = 'awake diaper hug hungry sleepy uncomfortable'.split()


def extract_spectrogram(indexes, selection):
    """
    Extracting the Spectrogram for every video
    提取每条音频的频谱特征图
    """
    cmap = plt.get_cmap('inferno')
    plt.figure(figsize=(10, 10))
    # 创建对应类的频谱图文件夹
    for type in types:
        if not os.path.exists(selection + '/image_data/'):
            os.makedirs(selection + '/image_data/')
        if (selection == "train") and (not os.path.exists(selection + '/image_data/' + type)):
            os.makedirs(selection + '/image_data/' + type)
            print("the dir of" + selection + "/image_data/" + type + "is created!")
        else:
            print("the dir is exists!")

    for key, value in indexes.items():
        # 加载15s的音频，转换为单声道（mono）
        wav, sample_rate = librosa.load(((bf.filePath + selection + "/" + value + "/" + key) if selection == "train"
                                         else key), mono=True, duration=15)
        plt.specgram(wav, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap,
                     sides='default', mode='default', scale='dB')
        plt.axis("off")
        if selection == "train":
            plt.savefig(selection + '/image_data/' + value + "/" + key[: -3].replace(".", "") + ".png")
        elif selection == "test":
            plt.savefig(selection + '/image_data/' + os.path.split(key[:-3])[1] + "png")
        plt.clf()


def extract_features():
    """
    Extracting features form Spectrogram
    We will extract
        MFCC (20 in number)
        Spectral Centroid (光谱质心)
        Zero Crossing Rate (过零率)
        Chroma Frequencies (色度频率)
        Spectral Roll-off (光谱衰减)
    """
    header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    return header.split()


def write_data_to_csv_file(header, indexes, filename, selection):
    """
    Writing data to csv file
    Notation: we must close the file
    """
    file = open(filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        file.close()
        for key, value in indexes.items():
            wav, sample_rate = librosa.load(bf.filePath + selection + "/" + (value if selection == "train" else "") +
                                            "/" + (key if selection == "train" else os.path.split(key)[1]),
                                            mono=True, duration=15)
            chroma_stft = librosa.feature.chroma_stft(y=wav, sr=sample_rate)
            rms = librosa.feature.rms(y=wav)
            spec_cent = librosa.feature.spectral_centroid(y=wav, sr=sample_rate)
            spec_bw = librosa.feature.spectral_bandwidth(y=wav, sr=sample_rate)
            rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sample_rate)
            zcr = librosa.feature.zero_crossing_rate(y=wav)
            mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate)
            to_append = f'{(key if selection == "train" else os.path.split(key)[1])} {np.mean(chroma_stft)} ' \
                        f'{np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {value}'
            file = open(filename, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                file.close()


def write_result_to_csv(data_file, filename, results):
    """
    输出测试集合的结果到csv
    :param filename: 文件名称
    :param data_file: 测试集
    :param results: 识别结果
    :return:
    """
    file = open(filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow('id label'.split())
        file.close()
        data = pd.read_csv(data_file)
        wav_paths = data.iloc[:, [0]].T
        wav_paths = np.matrix.tolist(wav_paths)[0]

        dictionary = {}
        print(len(results))
        for i in range(len(results)):
            dictionary[wav_paths[i]] = results[i]
            to_append = f'{wav_paths[i]} {classes_labels[results[i].data.numpy()]} '
            file = open(filename, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                file.close()
