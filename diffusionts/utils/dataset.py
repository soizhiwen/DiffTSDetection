import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        name,
        data_root,
        window=32,
        proportion=0.6,
        period="train",
        missing_ratio=None,
        style="separate",
        distribution="geometric",
        mean_mask_length=3,
    ):
        super(CustomDataset, self).__init__()
        assert period in ["train", "test"], "period must be train or test."
        self.name = name
        self.window = window
        self.missing_ratio = missing_ratio
        self.style = style
        self.distribution = distribution
        self.mean_mask_length = mean_mask_length

        raw_data = self.read_data(data_root)
        self.var_num = raw_data.shape[-1]

        if data_root.endswith(".npz"):
            train_data, test_data = self.divide(raw_data, proportion)
            self.samples = train_data if period == "train" else test_data
        else:
            train_data, test_data = self.divide(raw_data, proportion)
            train_data = self.__getsamples(train_data)
            test_data = self.__getsamples(test_data)
            self.total_samples = len(train_data) + len(test_data)
            self.scaler = StandardScaler()
            self.scaler = self.scaler.fit(train_data.reshape(-1, self.var_num))
            self.samples = train_data if period == "train" else test_data
            self.samples = self.normalize(self.samples)

        if missing_ratio is not None:
            self.masking = self.mask_data()

    def __getsamples(self, data):
        data = torch.tensor(data, dtype=torch.float32)
        return data.unfold(0, self.window, 1).transpose(1, 2).numpy()

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, d):
        d = self.scaler.transform(d)
        return d

    def __unnormalize(self, d):
        return self.scaler.inverse_transform(d)

    @staticmethod
    def divide(data, ratio):
        num_train = int(np.ceil(data.shape[0] * ratio))
        train_data = data[:num_train]
        test_data = data[num_train:]
        return train_data, test_data

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv"""
        if filepath.endswith(".npz"):
            data = np.load(filepath)["norm_synth"]
        elif filepath.endswith(".mat"):
            data = io.loadmat(filepath)["ts"]
        else:
            df = pd.read_csv(filepath, header=0)
            df = df.drop(["date"], axis=1, errors="ignore")
            data = df.values
        return data

    def mask_data(self):
        num_missing = int(self.samples.shape[1] * self.missing_ratio)
        conditional_mask = torch.ones_like(self.samples)
        mask_noise = torch.rand(self.samples.shape[0], self.samples.shape[1])
        indices_to_mask = mask_noise.topk(num_missing, dim=1).indices
        mask_2d = torch.ones(self.samples.shape[0], self.samples.shape[1])
        mask_2d.scatter_(dim=1, index=indices_to_mask, value=0.0)
        conditional_mask[:, :, :] = mask_2d.unsqueeze(-1)
        conditional_mask = conditional_mask.float()
        bool_mask = conditional_mask.bool()
        return bool_mask

    def __getitem__(self, ind):
        if self.missing_ratio is not None:
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.samples)
