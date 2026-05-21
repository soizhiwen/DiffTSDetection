import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    roc_curve,
    auc,
    average_precision_score,
    accuracy_score,
)

from utils.io_utils import seed_everything


class MLP(nn.Module):
    def __init__(self, in_dim, h1=64, h2=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_mse(dataset, variant, detector, window_size, flag) -> np.ndarray:
    file_path = Path(
        f"../wavestitch-synth/outputs/recon_{dataset}_{variant}_{detector}_{window_size}_{flag}/"
        f"recon_{dataset}_{variant}_{detector}_{window_size}_{flag}.parquet"
    )
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df.values
    # file_path = Path(
    #     f"../wavestitch-synth/outputs_clf/"
    #     f"onestep_{dataset}_{variant}_{detector}_{window_size}_{flag}.csv"
    # )
    # df = pd.read_csv(file_path)
    # columns = df.columns
    # df["mean"] = df[columns].mean(axis=1)
    # df["std"] = df[columns].std(axis=1)
    # return df.values


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--fpr_tau", type=float, default=0.01)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    output = Path(args.output)
    save_dir = output / f"clf_dire_{args.detector}_{args.dataset}_{args.window_size}"
    save_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        X_all, y_all = [], []
        for variant in ["real", args.detector]:
            mse = load_mse(args.dataset, variant, args.detector, args.window_size, 0)
            X_all.extend(mse)

            if variant == "real":
                y_all.extend([1] * len(mse))
            else:
                y_all.extend([0] * len(mse))

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=y_all,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = MLP(X_train.shape[-1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        epochs = args.epochs

        for epoch in range(epochs):
            model.train()
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x, y = x.to(device), y.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = torch.sigmoid(model(x)).cpu().numpy().ravel()
                val_logits.append(out)
                val_labels.append(y.numpy())

        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels)

        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = []

        for t in thresholds:
            preds = (val_logits >= t).astype(int)
            f1_scores.append(f1_score(val_labels, preds))

        f1_scores = np.array(f1_scores)
        best_f1 = np.max(f1_scores)

        # Get the mean of all thresholds that achieve the max F1
        best_thresholds = thresholds[f1_scores == best_f1]
        best_tau = np.mean(best_thresholds)

        print(f"Best τ*={best_tau:.3f} with F1={best_f1:.3f} on VALIDATION set")

        data = {
            "model": model.state_dict(),
            "scaler": scaler,
            "best_tau": best_tau,
        }
        torch.save(data, save_dir / "model.pt")

    elif args.eval:
        X_test, y_test = [], []
        for variant in ["real", args.generator]:
            mse = load_mse(args.dataset, variant, args.detector, args.window_size, 1)
            X_test.extend(mse)

            if variant == "real":
                y_test.extend([1] * len(mse))
            else:
                y_test.extend([0] * len(mse))

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        data = torch.load(save_dir / "model.pt", weights_only=False)
        X_test = data["scaler"].transform(X_test).astype(np.float32)

        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = MLP(X_test.shape[-1]).to(device)
        model.load_state_dict(data["model"])
        best_tau = data["best_tau"]

        test_logits, test_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = torch.sigmoid(model(x)).cpu().numpy().ravel()
                test_logits.append(out)
                test_labels.append(y.numpy())

        test_logits = np.concatenate(test_logits)
        test_labels = np.concatenate(test_labels)
        test_preds = (test_logits >= best_tau).astype(int)

        f1 = f1_score(test_labels, test_preds)
        fpr, tpr, _ = roc_curve(test_labels, test_logits)
        auc_score = auc(fpr, tpr)
        ap = average_precision_score(test_labels, test_logits)
        acc = accuracy_score(test_labels, test_preds)

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
