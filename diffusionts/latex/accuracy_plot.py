import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from utils.dataloader import build_dataloader
from utils.io_utils import load_yaml_config
from utils.constants import DATASETS, VARIANTS, PALETTE, WINDOW_SIZES


def load_acc(dataset, variant, window_size, scaler) -> np.ndarray:
    file_path = Path(
        f"outputs/recon_{dataset}_{variant}_{window_size}/"
        f"recon_{dataset}_{variant}_{window_size}.npz"
    )
    data = np.load(file_path)
    orig = data["orig"]
    recon = data["recon"]
    orig = scaler.transform(orig.reshape(-1, orig.shape[-1])).reshape(orig.shape)
    recon = scaler.transform(recon.reshape(-1, recon.shape[-1])).reshape(recon.shape)

    orig = torch.from_numpy(orig)
    recon = torch.from_numpy(recon)

    B, L, C = orig.shape
    orig = orig.permute(0, 2, 1).reshape(-1, L)
    recon = recon.permute(0, 2, 1).reshape(-1, L)

    q1 = torch.quantile(orig, 0.25, dim=1, keepdim=True)
    q2 = torch.quantile(orig, 0.5, dim=1, keepdim=True)
    q3 = torch.quantile(orig, 0.75, dim=1, keepdim=True)

    orig = torch.where(
        orig <= q1,
        0,
        torch.where(
            orig >= q3,
            1,
            torch.where((orig > q1) & (orig < q2), 2, 3),
        ),
    )

    recon = torch.where(
        recon <= q1,
        0,
        torch.where(
            recon >= q3,
            1,
            torch.where((recon > q1) & (recon < q2), 2, 3),
        ),
    )

    orig = orig.reshape(B, C, L).permute(0, 2, 1)
    recon = recon.reshape(B, C, L).permute(0, 2, 1)
    acc = np.mean((orig == recon).numpy(), axis=(1, 2))
    return acc


def build_plot_df(dataset, window_size, scaler) -> pd.DataFrame:
    rows = []
    for variant_key, variant_label in VARIANTS.items():
        acc = load_acc(dataset, variant_key, window_size, scaler)
        rows.append(pd.DataFrame({"value": acc, "variant": variant_label}))
    return pd.concat(rows, ignore_index=True)


def plot_acc_distribution(dataset, window_size, scaler, save_dir) -> None:
    plot_df = build_plot_df(dataset, window_size, scaler)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.histplot(
        data=plot_df,
        x="value",
        hue="variant",
        element="step",
        palette=PALETTE,
        ax=ax,
    )

    for variant, color in PALETTE.items():
        median_val = float(plot_df.loc[plot_df["variant"] == variant, "value"].median())
        ax.axvline(median_val, color=color, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title(f"{DATASETS[dataset]} - Window Size {window_size}")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("")

    fig.tight_layout()
    file_path = save_dir / f"accuracy_plot_{dataset}_{window_size}.pdf"
    fig.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Plot successfully saved to {file_path}")


if __name__ == "__main__":
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Times New Roman",
        font_scale=1.5,
        rc={"mathtext.fontset": "stix", "axes.formatter.use_mathtext": True},
    )

    save_dir = Path("plots/accuracy")
    save_dir.mkdir(parents=True, exist_ok=True)

    for window_size in WINDOW_SIZES:
        for dataset in DATASETS:
            try:
                config = load_yaml_config(f"./config/{dataset}.yaml")
                config["dataloader"]["train_dataset"]["params"]["window"] = window_size
                config["dataloader"]["test_dataset"]["params"]["window"] = window_size
                scaler = build_dataloader(config)["dataset"].scaler
                plot_acc_distribution(dataset, window_size, scaler, save_dir)
            except Exception as e:
                print(f"[SKIP] Dataset: {dataset}, Window Size: {window_size}")
