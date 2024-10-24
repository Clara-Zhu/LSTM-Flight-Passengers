import seaborn as sns  # 数据可视化
from numpy import ndarray  # 数值计算
from pandas import Series  # 数据处理
import torch  # PyTorch
from sklearn.preprocessing import MinMaxScaler  # 数据预处理


# 创建一个 PyTorch 数据集
class Data(torch.utils.data.Dataset):
    # 设置数据归一化器和转换器
    def __init__(self):
        # 初始化归一化器
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # 初始化转换器
        self.transform = ToTensor()
        # 加载数据集flights，seaborn库中内置的数据集
        tmp_flight_data = sns.load_dataset("flights")['passengers']
        # 归一化数据
        tmp_flight_data = self.scaler.fit_transform(tmp_flight_data.values.reshape(-1, 1))
        # 将数据转换为张量
        self.flight_data = self.transform(tmp_flight_data)

    # 返回数据集的长度
    def __len__(self):
        return len(self.flight_data)

    # 定义如何获取数据集中的一个元素
    def __getitem__(self, index: int):
        return torch.arange(0, len(self.flight_data), dtype=torch.int32)[index], self.flight_data[index]

    # 创建一个输入序列
    def create_inout_seq(self, input_data, train_window: int = 12):
        inout_seq = []
        for i in range(len(input_data) - train_window):
            train_seq = input_data[i: i + train_window]
            train_label = input_data[i + train_window: i + train_window + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def create_test_seq(self, input_data, min_seq_length: int = 12):
        inout_seq = []
        for i in range(len(input_data) - min_seq_length):
            in_seq = input_data[:i + min_seq_length]
            out_label = input_data[i + min_seq_length:i + min_seq_length + 1]
            inout_seq.append((in_seq, out_label))
        return inout_seq

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class ToTensor(object):
    def __call__(self, sample: object):
        if isinstance(sample, ndarray):
            return torch.from_numpy(sample).float().reshape(-1, 1)
        elif isinstance(sample, list) or isinstance(sample, Series):
            return torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)