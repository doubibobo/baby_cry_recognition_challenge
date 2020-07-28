"""
使用librosa包来提取MFCC特征
"""
import torchaudio


# 提取每段音频的MFCC特征
def get_mfcc_feature(indexes):
    mfcc = {}
    for key, _ in indexes.items():
        print(key)
        wav_form, sample_rate = torchaudio.load(key)
        mfcc[key] = torchaudio.transforms.MFCC()(wav_form)
    return mfcc
