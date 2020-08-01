import matplotlib.pyplot as plt


def accuracy_loss_plotting(loss_accuracy, epoch, k_number, train=True):
    """
    画训练过程中的accuracy和loss
    :param loss_accuracy: accuracy 和 loss的集合，输入为一个List[(loss, accuracy), (loss, accuracy)]
    :param epoch: 迭代次数
    :param k_number: 第几折数据
    :param train: 是否是训练集合，默认为True
    :return: 无返回值
    """
    loss_list = [loss_accuracy[i].__getitem__(0) for i in range(len(loss_accuracy))]
    accuracy_list = [loss_accuracy[i].__getitem__(1) for i in range(len(loss_accuracy))]
    x1, x2 = range(0, len(loss_list)), range(0, len(accuracy_list))
    y1, y2 = loss_list, accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title(("train" if train else "test") + ' loss vs. epochs')
    plt.ylabel(("train" if train else "test") + ' loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel(("train" if train else "test") + ' accuracy vs. epochs')
    plt.ylabel(("train" if train else "test") + ' accuracy')
    plt.savefig("images/" + ("train" if train else "test") + "/accuracy_loss_" + str(k_number) + ".jpg")
    plt.show()
