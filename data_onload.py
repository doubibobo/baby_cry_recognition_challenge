import torch
import torchaudio
import matplotlib.pyplot as plt

filename = '/home/doubibobo/test-data/awake_0.wav'
wavform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(wavform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure("原始声音信号")
plt.plot(wavform.t().numpy())
plt.show()

spectrogram = torchaudio.transforms.MelSpectrogram()(wavform)
print("Shape of spectrogram: {}".format(spectrogram.size()))
plt.figure("原始声音信号的Mel图")
plt.imshow(spectrogram.log2()[0, :, :].detach().numpy(), cmap='gray')
plt.show()

mfcc = torchaudio.transforms.MFCC()(wavform)
print("Shape of mfcc: {}".format(mfcc.size()))
plt.figure("原始声音的mfcc特征")
plt.imshow(mfcc.log2()[0, :, :].detach().numpy(), origin='lower')
plt.show()

# 声音整形：重新对声音进行采样
new_sample_rate = sample_rate / 10
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(wavform[channel, :].view(1, -1))
print("Shape of transformed waveform: {}".format(transformed.size()))
plt.figure("经重采样处理过后的声音")
plt.plot(transformed[0, :].numpy())
plt.show()