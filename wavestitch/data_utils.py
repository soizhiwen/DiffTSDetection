import pandas as pd
import numpy as np
from scipy import io
from sklearn.preprocessing import StandardScaler

datasets = {
    "electricity": "datasets/electricity.csv",
    "energy": "datasets/energy.csv",
    "etth": "datasets/etth.csv",
    "ettm": "datasets/ettm.csv",
    "exchange_rate": "datasets/exchange_rate.csv",
    "fmri": "datasets/fmri.mat",
    "illness": "datasets/illness.csv",
    "stock": "datasets/stock.csv",
    "traffic": "datasets/traffic.csv",
    "weather": "datasets/weather.csv",
}


class Preprocessor:
    def __init__(self, name, window, proportion=0.8):
        self.scaler = StandardScaler()
        self.data = self.read_data(name)
        self.var_num = self.data.shape[-1]
        self.window = window
        self.train_data, self.test_data = self.divide(self.data, proportion)

        self.scaler = self.scaler.fit(self.train_data)
        self.train_data = self.scaler.transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)

    def read_data(self, name):
        if name == "fmri":
            data = io.loadmat(datasets[name])["ts"]
        else:
            df = pd.read_csv(datasets[name], header=0)
            df = df.drop(["date"], axis=1, errors="ignore")
            data = df.values
        return data

    def divide(self, data, ratio):
        num_train = int(np.ceil(data.shape[0] * ratio))
        train_data = data[:num_train]
        test_data = data[num_train:]
        return train_data, test_data

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
