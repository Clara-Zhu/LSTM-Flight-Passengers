import seaborn as sns
from numpy import ndarray
from pandas import Series
import torch
from sklearn.preprocessing import MinMaxScaler


class Data(torch.utils.data.Dataset):
    def __init__(self, test_data_size: int = 12):
        self.test_data_size = test_data_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.transform = ToTensor()
        tmp_flight_data = sns.load_dataset("flights")['passengers']
        tmp_flight_data = self.scaler.fit_transform(tmp_flight_data.values.reshape(-1, 1))
        self.flight_data = self.transform(tmp_flight_data)

    def __len__(self):
        return len(self.flight_data)

    def __getitem__(self, index: int):
        return torch.arange(0, len(self.flight_data), dtype=torch.int32)[index], self.flight_data[index]

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