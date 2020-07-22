from torch.utils.data import Dataset


class TrainDataSet(Dataset):
    """
    构造数据集类
    """
    def __init__(self, data_train, label_train):
        self.x_data = data_train
        self.y_data = label_train
        self.length = len(label_train)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.length
