from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms, utils
from data.code.tools import build_file_index as bf
from data.code.tools import data_analysis as da
from data.code.tools import feature_extractor as fe

import torch.cuda as cuda
import torch.nn as nn


import numpy
import torch
import os
import shutil
import matplotlib.pyplot as plt
import time

# pre-trained为true，表示自动下载训练好的参数
model = models.vgg16(pretrained=True)

print(model)

current_path = '/home/zhuchuanbo/competition/data/code/VGG/'
path = '/train/wav_plot/'

FILE_INPUT = '/wav_plot/'

label_encode = {
    "awake": 0,
    "diaper": 1,
    "hug": 2,
    "sleepy": 3,
    "uncomfortable": 4,
    "hungry": 5,
}

if __name__ == '__main__':
    # # 建立文件路径与标签的索引
    # # file_label_indexes的类型是 {
    # #   filename: label
    # # }
    # file_label_indexes = bf.get_filename("train")
    # # # 获取频谱图
    # # fe.extract_spectrogram(file_label_indexes, "test", False, FILE_INPUT)
    # # 将字典数据转化为列表元素
    # data, label = [], []
    # for key, value in file_label_indexes.items():
    #     data.append(key[:-3] + 'png')
    #     label.append(label_encode.get(value))

    # # 划分训练集,测试集(按照7:3的比例划分),得到list形式
    # data_train, label_train, data_test, label_test = da.split_train_test(data, label)

    # 移动文件,在train文件夹中明确划分data_train和data_test
    # if not os.path.exists(path + 'data_train'):
    #     os.makedirs(path + 'data_train')
    # if not os.path.exists(path + 'data_test'):
    #     os.makedirs(path + 'data_test')
    # for file_name in data_train:
    #     shutil.move(current_path + path + file_name, current_path + path + 'data_train')
    # for file_name in data_test:
    #     shutil.move(current_path + path + file_name, current_path + path + 'data_test')

    # 进行图片的变换
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # 数据集合列表
    data_image = {
        x: datasets.ImageFolder(root=os.path.join(current_path + path, x), transform=transform)
        for x in ["data_train", "data_test"]
    }
    data_loader_image = {
        x: DataLoader(dataset=data_image[x], batch_size=32, shuffle=True)
        for x in ["data_train", "data_test"]
    }
    classes = data_image["data_train"].classes
    classes_index = data_image["data_train"].class_to_idx
    use_gpu = torch.cuda.is_available()

    # 准备好的训练集合
    X_train, y_train = next(iter(data_loader_image["data_train"]))


    # 查看GPU相关信息
    gpu_available = cuda.is_available()
    device_name = cuda.get_device_name(1)
    device_capability = cuda.get_device_capability(1)
    print(gpu_available)
    print(device_name)
    print(device_capability)
    if gpu_available:
        print("device_number is ", 1)
        cuda.set_device(1)

    # 冻结VGG网络的参数
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096, 6)
    )
    if use_gpu:
        model = model.cuda()

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters())
    print(model)

    # 开始训练
    n_epochs = 100

    for epoch in range(n_epochs):
        since = time.time()
        print("Epoch{}/{}".format(epoch, n_epochs))
        print("-" * 10)
        for param in ["data_train", "data_test"]:
            if param == "data_train":
                model.train = True
            else:
                model.train = False

            running_loss = 0.0
            running_correct = 0
            batch = 0
            for data in data_loader_image[param]:
                batch += 1
                X, y = data
                if use_gpu:
                    X, y = Variable(X.cuda()), Variable(y.cuda())
                else:
                    X, y = Variable(X), Variable(y)

                optimizer.zero_grad()
                y_pred = model(X)
                _, pred = torch.max(y_pred.data, 1)

                loss = loss_function(y_pred, y)
                if param == "data_train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data
                running_correct += torch.sum(pred == y.data)
                if batch % 500 == 0 and param == "data_train":
                    print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                        batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

            epoch_loss = running_loss / len(data_image[param])
            epoch_correct = 100 * running_correct / len(data_image[param])

            print("{}  Loss:{:.4f},  Correct:{:.4f} ".format(param, epoch_loss, epoch_correct))

        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))

    torch.save(model.state_dict(), "model_vgg16_fine_tune.pkl")