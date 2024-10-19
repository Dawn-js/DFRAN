import torch
from torch.utils.data import DataLoader, Dataset

# 自定义 Dataset 加载函数
class CustomMultimodalDataset(Dataset):
    def __init__(self, data_path):
        # 加载保存的.pt文件数据
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        images = self.data['images'][idx]
        texts = self.data['texts'][idx]
        labels = self.data['labels'][idx]
        return images, texts, labels

# 自定义 MultimodalDataLoader 类，用于加载三个数据集
class MultimodalDataLoader:
    def __init__(self, train_data_path, val_data_path, test_data_path, batch_size=16, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 加载训练、验证、和测试数据集
        self.train_dataset = CustomMultimodalDataset(train_data_path)
        self.val_dataset = CustomMultimodalDataset(val_data_path)
        self.test_dataset = CustomMultimodalDataset(test_data_path)

        # 创建数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)  # 通常验证数据集不打乱
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader


