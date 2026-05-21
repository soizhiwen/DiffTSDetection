import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

import json
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import io
from sklearn.preprocessing import StandardScaler

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from utils.io_utils import seed_everything
from utils.quality_metrics.discriminative_metric import discriminative_score_metrics
from utils.quality_metrics.predictive_metric import predictive_score_metrics
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


def main(ori_data, fake_data, iterations=5):
    # Discriminative score
    discriminative_score = []
    for _ in range(iterations):
        disc, *_ = discriminative_score_metrics(
            ori_data[:], fake_data[: ori_data.shape[0]]
        )
        discriminative_score.append(disc)

    # Predictive score
    predictive_score = []
    for _ in range(iterations):
        pred = predictive_score_metrics(ori_data[:], fake_data[: ori_data.shape[0]])
        predictive_score.append(pred)

    return discriminative_score, predictive_score


if __name__ == "__main__":
    for window_size in WINDOW_SIZES:
        for dataset in DATASETS:
            preprocessor = Preprocessor(dataset, window_size, 0.6)
            ori_data = torch.from_numpy(preprocessor.test_data)
            ori_data = ori_data.unfold(0, window_size, 1).transpose(1, 2).numpy()

            for variant in VARIANTS:
                seed_everything(42)
                tf.random.set_seed(42)

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

                    disc, pred = main(ori_data, fake_data)

                    if dataset not in res:
                        res[dataset] = {
                            "context_fid": {"mean": None, "std": None},
                            "correlational": {"mean": None, "std": None},
                            "discriminative": {"mean": None, "std": None},
                            "predictive": {"mean": None, "std": None},
                        }

                    res[dataset]["discriminative"]["mean"] = np.mean(disc).item()
                    res[dataset]["discriminative"]["std"] = np.std(disc).item()
                    res[dataset]["predictive"]["mean"] = np.mean(pred).item()
                    res[dataset]["predictive"]["std"] = np.std(pred).item()
                    with open(filename, "w") as file:
                        json.dump(res, file, indent=4)

                    print(
                        f"Results saved for Dataset: {dataset}, Variant: {variant}, Window Size: {window_size}"
                    )
                except Exception as e:
                    print(
                        f"[SKIP] Dataset: {dataset}, Variant: {variant}, Window Size: {window_size}"
                    )
