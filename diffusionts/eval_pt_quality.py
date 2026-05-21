import os
import json
import torch
import numpy as np
import pandas as pd
from scipy import io
from sklearn.preprocessing import StandardScaler

from utils.io_utils import seed_everything
from utils.quality_metrics.context_fid import ContextFID
from utils.quality_metrics.cross_correlation import CrossCorrelationLoss
from utils.constants import DATASETS, VARIANTS, WINDOW_SIZES

DATASET_PATH = {
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
    def __init__(self, name, window, proportion=0.6):
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
            data = io.loadmat(DATASET_PATH[name])["ts"]
        else:
            df = pd.read_csv(DATASET_PATH[name], header=0)
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


def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx


def main(ori_data, fake_data, fid_iter=5, corr_iter=5):
    # Context-FID score
    context_fid_score = []
    for _ in range(fid_iter):
        context_fid = ContextFID(ori_data[:], fake_data[: ori_data.shape[0]])
        context_fid_score.append(context_fid)

    # Cross-correlation score
    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)
    correlational_score = []
    size = int(x_real.shape[0] / corr_iter)
    for _ in range(corr_iter):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelationLoss(x_real[real_idx, :, :])
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())

    return context_fid_score, correlational_score


if __name__ == "__main__":
    for window_size in WINDOW_SIZES:
        for dataset in DATASETS:
            preprocessor = Preprocessor(dataset, window_size, 0.6)
            ori_data = torch.from_numpy(preprocessor.test_data)
            ori_data = ori_data.unfold(0, window_size, 1).transpose(1, 2).numpy()

            for variant in VARIANTS:
                seed_everything(42)
                if variant == "real":
                    continue

                filename = f"../wavestitch-synth/datasets/{variant}/{variant}_{window_size}.json"
                if os.path.exists(filename):
                    with open(filename, "r") as file:
                        try:
                            res = json.load(file)
                        except json.JSONDecodeError:
                            res = {}
                else:
                    res = {}

                try:
                    fake_data = np.load(
                        f"../wavestitch-synth/datasets/{variant}/{dataset}_{window_size}.npz"
                    )["norm_synth"][-len(ori_data) :]

                    corr_iter = 100 if dataset in ["electricity", "traffic"] else 5
                    fid, corr = main(ori_data, fake_data, corr_iter=corr_iter)

                    if dataset not in res:
                        res[dataset] = {
                            "context_fid": {"mean": None, "std": None},
                            "correlational": {"mean": None, "std": None},
                            "discriminative": {"mean": None, "std": None},
                            "predictive": {"mean": None, "std": None},
                        }

                    res[dataset]["context_fid"]["mean"] = np.mean(fid).item()
                    res[dataset]["context_fid"]["std"] = np.std(fid).item()
                    res[dataset]["correlational"]["mean"] = np.mean(corr).item()
                    res[dataset]["correlational"]["std"] = np.std(corr).item()
                    with open(filename, "w") as file:
                        json.dump(res, file, indent=4)

                    print(
                        f"Results saved for Dataset: {dataset}, Variant: {variant}, Window Size: {window_size}"
                    )
                except Exception as e:
                    print(
                        f"[SKIP] Dataset: {dataset}, Variant: {variant}, Window Size: {window_size}"
                    )
