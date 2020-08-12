import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

import data.code.tools.feature_tools.build_file_index as bf

"建立训练时的标签和语音类别的映射关系"
classes_labels = ["awake", "diaper", "hug", "hungry", "sleepy", "uncomfortable"]

types = 'awake diaper hug hungry sleepy uncomfortable'.split()


def extract_spectrogram(indexes, selection, spectrogram=True, directory="image_data/", duration=15):
    """
    Extracting the Spectrogram for every video
    提取每条音频的频谱特征图
    :param indexes: 文件索引
    :param selection: 选择train文件或者test文件
    :param spectrogram: 是否是频谱图，默认为True;如果为语音波形图，则设置为False
    :param directory: 存储相关文件的目录
    :param duration: 语音段时长
    :return: 无返回值，生成频谱图之后直接存储到文件中
    """
    plt.figure(figsize=(10, 10))
    # 创建对应类的频谱图文件夹
    for label in types:
        if not os.path.exists(directory + selection):
            os.makedirs(directory + selection)
        if (selection == "train") and (not os.path.exists(directory + selection + '/' + label)):
            os.makedirs(directory + selection + '/' + label)
            print("the dir of " + directory + selection + '/' + label + " is created!")
        else:
            print("the dir is exists!")

    for key, value in indexes.items():
        # 加载15s的音频，转换为单声道（mono）
        wav, sample_rate = librosa.load(((bf.filePath + selection + "/" + value + "/" + key) if selection == "train"
                                         else key), mono=True, sr=None, duration=duration)
        if spectrogram:
            # 归一化语音数据，防止出现无效地方
            wav = wav * 1.0 / max(abs(wav))
            # 以25ms为一帧，以10ms作为步长，计算每一帧中的样本数目和重合的样本数
            frame_length, overlap_length = int(round(0.025 * sample_rate)), int(round((0.025 - 0.010) * sample_rate))

            # TODO 重要的是如何准确的设置参数
            plt.specgram(wav, NFFT=frame_length, Fs=sample_rate, Fc=0, noverlap=overlap_length, scale_by_freq=True,
                         window=np.hamming(frame_length), sides='default', mode='default', scale='dB')
            # plt.specgram(wav, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=c_map,
            #              sides='default', mode='default', scale='dB')
            plt.axis("off")

            fig = plt.gcf()
            if selection == "train":
                plt.ylabel('Frequency')
                plt.xlabel('Time(s)')
                # plt.title('Spectrogram')
                plt.savefig(directory + selection + '/' + value + "/" + key[: -3].replace(".", "") + ".png",
                            bbox_inches='tight', pad_inches=0, dpi=fig.dpi)
            elif selection == "test":
                plt.savefig(directory + selection + '/' + os.path.split(key[:-3])[1] + "png",
                            bbox_inches='tight', pad_inches=0, dpi=fig.dpi)
            plt.clf()
        else:
            plt.plot(wav)
            plt.axis("off")
            # # label the axes
            # plt.ylabel("Amplitude")
            # plt.xlabel("Time")
            # set the title
            # plt.title("Sample Wav")
            # display the plot
            if selection == "train":
                print(selection + directory + "/" + key[: -3].replace(".", "") + ".png")
                plt.savefig(selection + directory + "/" + key[: -3].replace(".", "") + ".png")
            elif selection == "test":
                plt.savefig(selection + directory + os.path.split(key[: -3])[1] + "png", bbox_inches='tight',
                            pad_inches=0.0)
            # plt.show()
            plt.close('all')


def extract_spectrogram_mfcc(indexes, selection, duration, directory="mel_data/", is_small=False):
    """
    提取语音的mfcc频谱图
    :param indexes: 文件索引
    :param selection: 选择train文件或者test文件
    :param duration: 声音段时长
    :param directory: 频谱图所在的根目录
    :param is_small: 是否对小单元进行分帧操作，默认为2s
    :return: 无返回值
    """
    for key, value in indexes.items():
        # 加载15s的音频，转换为单声道（mono）
        wav, sample_rate = librosa.load(((bf.filePath + selection + "/" + value + "/" + key) if selection == "train"
                                         else key), mono=True, sr=None, duration=duration)
        # 对语音数据进行预加重
        wav = librosa.effects.preemphasis(wav)

        # 以25ms为一帧，以10ms作为步长，计算每一帧中的样本数目和重合的样本数
        frame_length, overlap_length = int(round(0.025 * sample_rate)), int(round((0.025 - 0.010) * sample_rate))
        step_length = int(round((0.010 * sample_rate)))

        if not is_small:
            wav_collections = [wav]
        else:
            wav_collections, number = [], len(wav) // (2 * sample_rate)
            for i in range(number):
                wav_collections.append(wav[i*sample_rate: min((i+1)*sample_rate, len(wav))])

        count = 0
        for wav in wav_collections:
            # 提取mel spectrogram 图（梅尔频谱图【功率图】和频谱图【功率图】是一致的，只不过尺度发生了变换，
            # 默认画的是梅尔频谱功率图
            mel_power_spectrogram_feature = librosa.feature.melspectrogram(wav, sr=sample_rate, n_fft=frame_length,
                                                                           hop_length=step_length, n_mels=128,
                                                                           win_length=frame_length, window='hamm')
            # 转换为对数刻度，因为log运算的存在
            # 如果ref为1，转换公式相当于： log_mel_spectrogram_feature = 20 * log10(mel_spectrogram_feature)
            # 但还是相当于计算了某种功率
            log_mel_spectrogram_feature = librosa.power_to_db(mel_power_spectrogram_feature)

            # 绘制Mel频谱图
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(log_mel_spectrogram_feature, sr=sample_rate, x_axis='time', y_axis='mel')
            plt.axis("off")
            plt.gcf()
            # plt.title('Mel power spectrogram ')
            # plt.colorbar(format='%+02.0f dB')
            # plt.tight_layout()

            if selection == "test":
                plt.savefig(directory + selection + '/' + os.path.split(key[: -4])[1] + str(count) + ".png",
                            bbox_inches='tight', pad_inches=0.0)
            else:
                plt.savefig(directory + selection + '/' + value + "/" + key[: -4] + str(count) + ".png",
                            bbox_inches='tight', pad_inches=0.0)
            plt.close()
            count += 1

            # # 绘制MFCC系数图
            # mfcc = librosa.feature.mfcc(S=log_mel_spectrogram_feature, n_mfcc=20)
            #
            # # Let's pad on the first and second deltas while we're at it
            # # delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            #
            # plt.figure(figsize=(12, 4))
            # # librosa.display.specshow(delta2_mfcc)
            # librosa.display.specshow(mfcc)
            # plt.axis("off")
            # fig = plt.gcf()
            # plt.savefig(directory + selection + '/' + value + "/" + key[: -3].replace(".", "") + ".png",
            #             bbox_inches='tight', pad_inches=0, dpi=fig.dpi)
            # plt.clf()

            # plt.ylabel('MFCC coefficients')
            # plt.xlabel('Time')
            # plt.axis()
            #
            # plt.title('MFCC')
            # plt.colorbar()
            # plt.tight_layout()
            # plt.show()
            # print(key)
            # break


def signal_append(signal, sample_rate, signal_window):
    """
    补充语音信号
    :param signal: 信号值
    :param sample_rate: 采样率
    :param signal_window: 信号长度
    :return: 信号补充之后的结果
    """
    # 整段语音中包含的样本数目
    signal_length = int(round(signal_window * sample_rate))
    # 对不足signal_window长度的语音数据进行填充
    if len(signal) < signal_length:
        signal = np.append(signal, np.zeros(signal_length - len(signal)))
    else:
        pass
    return signal


def framing(signal, sample_rate, frame_window, shift_window, signal_window, hamming=False):
    """
    进行语音的分帧操作，如果hamming为True，则需要加hamming窗
    :param signal: 原始语音信号
    :param sample_rate: 采样率
    :param frame_window: 帧窗口，单位为ms
    :param shift_window: 步长窗口，单位为ms
    :param signal_window: 语音窗口，单位为ms
    :param hamming: 是否要进行加窗操作，hamming加窗，默认为False
    :return: 经过分帧处理以后的数据帧
    """
    # 每一帧包含的样本数目，以及每隔多少个样本到下一帧
    frame_length, step_length = int(round(frame_window * sample_rate)), int(round(shift_window * sample_rate))
    # 整段语音中包含的样本数目
    signal_length = int(round(signal_window * sample_rate))
    # 计算一段语音中包含的帧数
    frame_numbers = int(np.ceil(float(np.abs(signal_length - frame_length)) / step_length))
    # 对不足signal_window长度的语音数据进行填充
    signal = signal_append(signal, sample_rate, signal_window)
    # 对不足一帧的语音进行填充
    padding_signal_length = np.append(signal, np.zeros((frame_numbers * step_length + frame_length - signal_length)))
    # 计算每一个数据帧的索引
    # TODO 此处应该改为舍弃，因为数据集中大量样本的最后一帧都是补充帧，对分类效果影响较大
    indexes = np.tile(np.arange(0, frame_length), (frame_numbers, 1)) + np.tile(
        np.arange(0, frame_numbers * step_length, step_length), (frame_length, 1)).T
    # 进行hamming加窗操作
    if hamming:
        return padding_signal_length[indexes.astype(np.int32, copy=False)] * np.hamming(frame_length)
    return padding_signal_length[indexes.astype(np.int32, copy=False)]


def extract_features(to_frame=False):
    """
    Extracting features form Spectrogram
    We will extract
        MFCC (20 in number)
        Spectral Centroid (光谱质心)
        Zero Crossing Rate (过零率)
        Chroma Frequencies (色度频率)
        Spectral Roll-off (光谱衰减)
    :param to_frame: 是否进行分帧
    :return: csv文件头
    """
    header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    # for i in range(1, 21):
    #     header += f' delta{i}'
    # for i in range(1, 13):
    #     header += f' cqt{i}'
    header += ' label'
    if to_frame:
        header += ' frame_number'
    return header.split()


def write_data_to_csv_file(header, indexes, filename, selection, to_frame=False, root_path=bf.filePath):
    """
    将特征数据写入csv文件夹
    :param header: csv头部
    :param indexes: 文件的索引
    :param filename: csv文件名称
    :param selection: train或者test选择
    :param to_frame: 是否选择分帧
    :param root_path: 数据文件的根目录
    :return: 无返回值
    """
    file = open(filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        file.close()
        for key, value in indexes.items():
            # 获取样本的路径
            file_path = root_path + selection + "/" + (value if selection == "train" else "") + "/" + \
                        (key if selection == "train" else os.path.split(key)[1])
            wav, sample_rate = librosa.load(file_path, mono=True, sr=None)
            # wav, sample_rate = librosa.load(file_path, mono=True, duration=15)
            # # 不足15s的要补0
            # wav = np.append(wav, np.zeros(sample_rate * 15 - len(wav)))

            # # 获取样本的长度
            # length = librosa.get_duration(filename=file_path)
            # # 判断是否有新的数据集
            # if length < 20:
            #     continue
            # elif 20 <= length <= 30:
            #     # 存储新的数据
            #     wav, sample_rate = librosa.load(file_path, mono=True, offset=length-15, duration=15)
            #     wavfile.write("../csv_data/" + value + '/new_' + str(count) + '.wav', sample_rate, wav)
            #     count = count + 1
            # elif 30 < length <= 45:
            #     wav_1, sample_rate = librosa.load(file_path, mono=True, offset=15, duration=15)
            #     wav_2, _ = librosa.load(file_path, mono=True, offset=length-15, duration=15)
            #     wavfile.write("../csv_data/" + value + '/new_' + str(count) + '.wav', sample_rate, wav_1)
            #     count = count + 1
            #     wavfile.write("../csv_data/" + value + '/new_' + str(count) + '.wav', sample_rate, wav_2)
            #     count = count + 1
            # else:
            #     print(key + ' length is more than 45s!')
            # continue
            # print(sample_rate)
            # if sample_rate in sample_dictionary.keys():
            #     sample_dictionary[sample_rate] = sample_dictionary[sample_rate] + 1
            # else:
            #     sample_dictionary[sample_rate] = 1
            # continue
            # 对语音数据进行预处理
            wav = librosa.effects.preemphasis(wav)

            frame_length, overlap_length = int(round(0.025 * sample_rate)), int(round((0.025 - 0.010) * sample_rate))
            step_length = int(round((0.010 * sample_rate)))
            # mel_power_spectrogram_feature = librosa.feature.melspectrogram(wav, sr=sample_rate, n_fft=frame_length,
            #                                                                hop_length=step_length, n_mels=128,
            #                                                                win_length=frame_length, window='hamm')

            if to_frame:
                frames = framing(wav, sample_rate, 0.025, 0.01, 15, True)
            else:
                # # 数据增强，进行Time Stretch变换、Pitch Shift变换、roll变换（滚动变换）
                # wav_time_stretch = librosa.effects.time_stretch(wav, rate=1.2)
                # wav_pitch_shift = librosa.effects.pitch_shift(wav, sample_rate, n_steps=3.0)
                # wav_roll = np.roll(wav, sample_rate * 10)
                # frames = [wav, wav_time_stretch, wav_pitch_shift, wav_roll]
                frames = []
                # 以2s作为一个语音帧，无step进行划分，没有进行VAD操作
                # 计算共有多少语音帧
                # 计算一段语音中包含的帧数
                number = int(np.ceil(float(np.abs(len(wav) - 2 * sample_rate)) / (2 * sample_rate)))
                # number = len(wav) // (2 * sample_rate)
                for i in range(number):
                    frames.append(wav[int(2 * i * sample_rate): int(2 * (i + 1) * sample_rate)])
                # frames = [wav]
                print(os.path.split(file_path)[1])
            for i in range(len(frames)):
                chroma_stft = librosa.feature.chroma_stft(y=frames[i], sr=sample_rate, n_fft=frame_length,
                                                          hop_length=step_length,
                                                          win_length=frame_length, window='hamm').T
                spec_cent = librosa.feature.spectral_centroid(y=frames[i], sr=sample_rate, n_fft=frame_length,
                                                              hop_length=step_length,
                                                              win_length=frame_length, window='hamm').T
                spec_bw = librosa.feature.spectral_bandwidth(y=frames[i], sr=sample_rate, n_fft=frame_length,
                                                             hop_length=step_length,
                                                             win_length=frame_length, window='hamm').T
                rolloff = librosa.feature.spectral_rolloff(y=frames[i], sr=sample_rate, n_fft=frame_length,
                                                           hop_length=step_length,
                                                           win_length=frame_length, window='hamm').T
                mfcc = librosa.feature.mfcc(y=frames[i], sr=sample_rate, n_mfcc=20, n_mels=128, n_fft=frame_length,
                                            hop_length=step_length, win_length=frame_length, window='hamm').T
                rms = librosa.feature.rms(y=frames[i], hop_length=step_length, frame_length=frame_length).T
                zcr = librosa.feature.zero_crossing_rate(y=frames[i], hop_length=step_length, frame_length=frame_length).T

                # 保留时域特征，不取平均值，所有特征的第一维是：201，即帧的数目
                for j in range(len(mfcc)):
                    file_name = (key if selection == "train" else os.path.split(key)[1])[: -4] + '_' + str(i) + '.wav'
                    to_append = f'{file_name} {np.mean(chroma_stft[j])} {np.mean(rms[j])} {np.mean(spec_cent[j])} ' \
                                f'{np.mean(spec_bw[j])} {np.mean(rolloff[j])} {np.mean(zcr[j])} '
                    for e in mfcc[j]:
                        to_append += f'{e} '
                    to_append += f'{value} '
                    if to_frame:
                        to_append += f'{j} '
                    file = open(filename, 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
                        file.close()


def write_result_to_csv(data_file, filename, results, time_step=None, final=None):
    """
    输出测试集合的结果到csv
    :param filename: 文件名称
    :param data_file: 测试集
    :param results: 识别结果
    :param time_step: 测试语音文件是否分帧（每一个语音文件含有的时间步），默认为None，不分帧。
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
        if time_step is not None:
            wav_paths = wav_paths[0:len(wav_paths):time_step]
        print(wav_paths)

        if final is not None:
            wav_paths = final

        dictionary = {}
        print(len(results))
        print(results)
        for i in range(len(results)):
            dictionary[wav_paths[i]] = results[i]
            # to_append = f'{wav_paths[i]} {classes_labels[results[i].csv_data.numpy()]} '
            to_append = f'{wav_paths[i]} {classes_labels[results[i]]} '
            file = open(filename, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                file.close()


if __name__ == '__main__':
    from data.code.tools.image_tools import to_gray as tg

    # 建立训练集文件路径与标签的索引
    train_label_indexes = bf.get_filename("train")
    print(train_label_indexes)
    test_label_indexes = bf.get_filename("test")
    print(test_label_indexes)

    # # 获取训练集的原始频谱图
    # extract_spectrogram(train_label_indexes, "train", True, tg.fileOriginPath)
    # extract_spectrogram(test_label_indexes, "test", True, tg.fileOriginPath)
    #
    # # 获取训练集的MFCC频谱图
    # extract_spectrogram_mfcc(train_label_indexes, "train", 15, tg.fileMelPath)
    # extract_spectrogram_mfcc(test_label_indexes, "test", 15, tg.fileMelPath)
    # extract_spectrogram_mfcc(train_label_indexes, "train", None, tg.fileMel2sPath, True)
    # extract_spectrogram_mfcc(test_label_indexes, "test", None, tg.fileMel2sPath, True)

    # # 转化频谱图为灰度图
    # tg.spectrogram_to_gray(train_label_indexes, True, tg.fileOriginPath, tg.fileOriginGrayPath)
    # tg.spectrogram_to_gray(train_label_indexes, True, tg.fileMelPath, tg.fileMelGrayPath)
    # tg.spectrogram_to_gray(test_label_indexes, False, tg.fileOriginPath, tg.fileOriginGrayPath)
    # tg.spectrogram_to_gray(test_label_indexes, False, tg.fileMelPath, tg.fileMelGrayPath)

    # 将其特征写入csv文件中
    headers = extract_features()
    write_data_to_csv_file(headers, train_label_indexes, "../../data/csv_data/train_mfcc_20_new_2s_time.csv", "train")
    write_data_to_csv_file(headers, test_label_indexes, "../../data/csv_data/test_mfcc_20_new_2s_time.csv", "test")
