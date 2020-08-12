import matplotlib.pyplot as plt
import numpy
import os
import scipy.signal as signal
import struct
import webrtcvad

from data.code.tools.feature_tools import build_file_index as bf
from scipy.io import wavfile

original_path = "/csv_data/code/wav_data/original-csv_data/"

resampled_path = "/csv_data/code/wav_data/resampled-csv_data/"

vad_path = "/csv_data/code/wav_data/vad-csv_data/"


def resample_signal(old_path, new_path, new_sample_rate=16000):
    """
    对单个音频进行重采样
    :param old_path: 原始文件
    :param new_path: 新文件
    :param new_sample_rate: 新的采样率
    :return: 无返回值
    """
    sr, sig = wavfile.read(old_path)
    result = int((sig.shape[0]) / sr * new_sample_rate)
    sig = signal.resample(sig, result)
    sig = sig.astype(numpy.int16)
    print(sr)
    wavfile.write(new_path, new_sample_rate, sig)


def resample_wav(origin_path=original_path):
    """
    对原始数据进行重采样
    :param origin_path: 原始文件的路径
    :return: 无返回值
    """
    # train_indexes = bf.get_filename("train", origin_path)
    # print(train_indexes)

    test_indexes = bf.get_filename("test", origin_path)
    print(test_indexes)

    # for key, value in train_indexes.items():
    #     resample_signal(origin_path + 'train/' + value + '/' + key, resampled_path + 'train/' + value + '/' + key)

    for key, _ in test_indexes.items():
        resample_signal(key, resampled_path + 'test/' + os.path.split(key)[1])


def vad_with_webrtc(wav, sample_rate, frame_window=0.020, signal_byte=2):
    """
    对原始语音信号做vad处理，只返回有效信息
    :param wav: 原始声音信号
    :param sample_rate: 采样率
    :param frame_window: 每一帧的长度，以ms为单位
    :param signal_byte: 每个采样点的存储大小，默认是以int16的形式存储，为2字节
    :return: 无返回值，直接存储到新的文件夹中
    """
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    start, end = 0, len(wav)

    buffer_wav = struct.pack("%dh" % end, *wav)

    samples_per_window = int(frame_window * sample_rate + 0.4)

    # 保存vad识别的每一个片段
    segments = []
    try:
        for start in numpy.arange(0, end, samples_per_window):
            # print(start)
            stop = min(start + samples_per_window, end)

            is_speech = vad.is_speech(buffer_wav[start * signal_byte: stop * signal_byte],
                                      sample_rate=sample_rate, length=samples_per_window)
            # print(buffer_wav[start * signal_byte: stop * signal_byte])

            segments.append(dict(
                start=start,
                stop=stop,
                is_speech=is_speech
            ))
    except IndexError:
        # 最后一帧可能会因为不满足windows_duration的条件而使程序发生异常
        # 故直接舍弃最后一帧
        print("finally has an error for the last frame!")
    finally:
        speech_samples = numpy.concatenate(
                [wav[segment['start']: segment['stop']] for segment in segments if segment['is_speech']])
    return segments, speech_samples


def plot_segment(wav, segments):
    """
    画出vad之后有效的声音片段
    :param wav: 原始声音信号
    :param segments: vad识别片段
    :return: 无返回值
    """
    plt.figure(figsize=(10, 7))
    plt.plot(wav)
    y_max = max(wav)

    for segment in segments:
        if segment['is_speech']:
            plt.plot([segment['start'], segment['stop'] - 1], [y_max * 1.1, y_max * 1.1], color='orange')

    plt.xlabel('wav')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # # 将语音重采样到16kHz
    # resample_wav()

    # # 读取所有音频文件的d_type
    # train_indexes = bf.get_filename("train", file=resampled_path)
    # for key, value in train_indexes.items():
    #     sample_rate, wav = wavfile.read(resampled_path + 'train/' + value + '/' + key)
    #     segments, speech_samples = vad_with_webrtc(wav, sample_rate)
    #     wavfile.write(vad_path + 'train/' + value + '/' + key, sample_rate,  speech_samples)
    #     # if key == 'awake_87.wav':
    #     # plot_segment(wav, segments)

    test_indexes = bf.get_filename("test", file=resampled_path)
    for key, _ in test_indexes.items():
        sample_rate, wav = wavfile.read(key)
        print(wav.dtype)
        segments, speech_samples = vad_with_webrtc(wav, sample_rate)
        wavfile.write(vad_path + 'test/' + os.path.split(key)[1], sample_rate,  speech_samples)
        # plot_segment(wav, segments)

