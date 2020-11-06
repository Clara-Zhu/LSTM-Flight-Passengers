import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler


class Data():
    def __init__(self, test_data_size: int = 12):
        self.test_data_size = test_data_size
        self.scaler = MinMaxScaler(feature_range=(-1,1))

    def get_data(self):
        flight_data = sns.load_dataset("flights")
        data_vals = flight_data['passengers'].values
        data_vals = self.scaler.fit_transform(data_vals.reshape(-1, 1))
        train_data = flight_data[:-self.test_data_size]
        test_data = flight_data[-self.test_data_size:]
        train_vals = data_vals[:-self.test_data_size]
        test_vals = data_vals[-self.test_data_size:]
        train_data = train_data.index, train_vals
        test_data = test_data.index, test_vals
        return train_data, test_data

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
            in_seq = input_data[:i+min_seq_length]
            out_label = input_data[i+min_seq_length:i+min_seq_length+1]
            inout_seq.append((in_seq, out_label))
        return inout_seq

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
