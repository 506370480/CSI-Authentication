import numpy as np
from torch.utils.data import Dataset


class GaussianNoise(Dataset):
    """Gaussian Noise Dataset"""
    # 必须重写__len__、__getitem__方法，使得数据集可以与 torch.utils.data.DataLoader 配合使用
    # 后者可以提供批处理、样本洗牌、多线程数据加载等功能
    # 详细参考：https://pytorch.org/docs/stable/data.html
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    # 主要用于生成噪声数据集

    def __init__(self, size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0):
        self.size = size
        self.n_samples = n_samples
        self.mean = mean
        self.variance = variance
        # 定义的内部数据，用于存储生成的噪声数据
        # normal()函数用于生成正态分布的随机数，loc为均值，scale为标准差，size为输出的维度
        # 在Python中，如果你想创建一个只有一个元素的元组，你需要在该元素后面加上一个逗号。
        self.data = np.random.normal(loc=self.mean, scale=self.variance, size=(self.n_samples,) + self.size)#元组的拼接
        # 将 self.data 数组中的所有元素限制在0和1之间。如果数组中的元素小于0，则变为0；如果大于1，则变为1。
        self.data = np.clip(self.data, 0, 1) #将大小限制在0和1之间
        self.data = self.data.astype(np.float32) #强制类型转换

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


class UniformNoise(Dataset):
    """Uniform Noise Dataset"""

    def __init__(self, size=(3, 32, 32), n_samples=10000, low=0, high=1):
        self.size = size
        self.n_samples = n_samples
        self.low = low
        self.high = high
        self.data = np.random.uniform(low=self.low, high=self.high, size=(self.n_samples,) + self.size)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]