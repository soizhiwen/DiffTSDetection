import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from aeon.classification.deep_learning import InceptionTimeClassifier
from scipy import io
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    roc_curve,
    auc,
    average_precision_score,
    accuracy_score,
)

from utils.io_utils import seed_everything


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--detector", type=str, default="sssd")
    parser.add_argument("--generator", type=str)
    parser.add_argument("--output", type=str, default="outputs_clf")

    # args for random
    parser.add_argument("--seed", type=int, default=42)

    # args for training
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fpr_tau", type=float, default=0.01)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    output = Path(args.output)
    save_dir = (
        output / f"clf_inceptiontime_{args.detector}_{args.dataset}_{args.window_size}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.train:
        X_train, y_train = [], []
        preprocessor = Preprocessor(args.dataset, args.window_size, proportion=0.6)
        for variant in ["real", args.detector]:
            d_vals_tensor = torch.from_numpy(preprocessor.train_data)
            training_samples = d_vals_tensor.unfold(0, args.window_size, 1).numpy()
            if variant != "real":
                training_samples = np.load(
                    f"../wavestitch-synth/datasets/{variant}/{args.dataset}_{args.window_size}.npz"
                )["norm_synth"][: len(training_samples)].transpose(0, 2, 1)

            X_train.extend(training_samples)

            if variant == "real":
                y_train.extend([1] * len(training_samples))
            else:
                y_train.extend([0] * len(training_samples))

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        model = InceptionTimeClassifier(
            n_classifiers=3,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            save_last_model=True,
            save_best_model=True,
            file_path=f"{str(save_dir)}/",
            random_state=args.seed,
            verbose=True,
        )
        model.fit(X_train, y_train)

    elif args.eval:
        X_test, y_test = [], []
        preprocessor = Preprocessor(args.dataset, args.window_size, proportion=0.6)
        for variant in ["real", args.generator]:
            d_vals_tensor = torch.from_numpy(preprocessor.test_data)
            training_samples = d_vals_tensor.unfold(0, args.window_size, 1).numpy()
            if variant != "real":
                training_samples = np.load(
                    f"../wavestitch-synth/datasets/{variant}/{args.dataset}_{args.window_size}.npz"
                )["norm_synth"][-len(training_samples) :].transpose(0, 2, 1)

            X_test.extend(training_samples)

            if variant == "real":
                y_test.extend([1] * len(training_samples))
            else:
                y_test.extend([0] * len(training_samples))

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        model = InceptionTimeClassifier.load_model(
            [
                save_dir / "best_model0.keras",
                save_dir / "best_model1.keras",
                save_dir / "best_model2.keras",
                save_dir / "last_model0.keras",
                save_dir / "last_model1.keras",
                save_dir / "last_model2.keras",
            ],
            np.array([0, 1]),
        )
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        ap = average_precision_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)

        # Calculate TPR at X% FPR
        valid_indices = fpr <= args.fpr_tau
        if valid_indices.any():
            tpr_at_x_fpr = tpr[valid_indices][-1]
        else:
            tpr_at_x_fpr = 0.0

        print(f"\n--- Results ---")
        print(f"F1 Score: {f1:.4f}")
        print(f"Avg Prec: {ap:.4f}")
        print(f"AUROC: {auc_score:.4f}")
        print(f"TPR@{args.fpr_tau * 100}%FPR: {tpr_at_x_fpr:.4f}")
        print(f"Accuracy: {acc:.4f}")

        res = {
            "dataset": args.dataset,
            "detector": args.detector,
            "generator": args.generator,
            "window_size": args.window_size,
            "fpr_tau": args.fpr_tau,
            "tpr_at_fpr_tau": tpr_at_x_fpr,
            "ap": ap,
            "f1": f1,
            "auc": auc_score,
            "acc": acc,
        }

        with open(save_dir / f"{args.generator}.json", "w") as file:
            json.dump(res, file, indent=4)

    else:
        raise ValueError("Please specify either --train or --eval.")


if __name__ == "__main__":
    main()
