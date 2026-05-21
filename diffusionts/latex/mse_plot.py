import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

# from utils.constants import DATASETS, VARIANTS, WINDOW_SIZES

VARIANTS = {
    "real": "Real",
    "sssd": "SSSD",
    "tsdiff": "TSDiff",
    "diffusionts": "Diffusion-TS",
    "wavestitch": "WaveStitch",
}

DATASETS = {
    "electricity": "Electricity",
    "energy": "Energy",
    "etth": "ETTh",
    "ettm": "ETTm",
    "exchange_rate": "Exchange Rate",
    "fmri": "fMRI",
    "illness": "Illness",
    "stock": "Stock",
    "traffic": "Traffic",
    "weather": "Weather",
}

WINDOW_SIZES = [32, 64, 128]


def load_mse(dataset, variant, window_size, detector, flag) -> np.ndarray:
    file_path = Path(
        f"../wavestitch-synth/outputs/recon_{dataset}_{variant}_{detector}_{window_size}_{flag}/recon_{dataset}_{variant}_{detector}_{window_size}_{flag}.parquet"
        # f"./outputs/one_step_{dataset}_{variant}_{detector}_{window_size}_{flag}/one_step_{dataset}_{variant}_{detector}_{window_size}_{flag}.csv"
    )
    table = pq.read_table(file_path)
    df = table.to_pandas().mean(axis=1)
    return df.values
    # data = pd.read_csv(file_path)
    # mse = data.mean(axis=1).values
    # return mse


def build_plot_df(dataset, window_size, detector) -> pd.DataFrame:
    rows = []
    for variant_key, variant_label in VARIANTS.items():
        mse = load_mse(dataset, variant_key, window_size, detector, 1)
        rows.append(pd.DataFrame({"value": mse, "variant": variant_label}))
    return pd.concat(rows, ignore_index=True)


def plot_mse_distribution(dataset, window_size, save_dir, detector) -> None:
    plot_df = build_plot_df(dataset, window_size, detector)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="value",
        hue="variant",
        element="step",
        log_scale=True,
        ax=ax,
    )

    colors = sns.color_palette("bright", n_colors=len(VARIANTS))
    for variant, color in zip(VARIANTS.values(), colors):
        median_val = float(plot_df.loc[plot_df["variant"] == variant, "value"].median())
        ax.axvline(median_val, color=color, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.5)

    legend = ax.get_legend()
    if legend is not None:
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=5,
            title=None,
            frameon=False,
        )

    fig.tight_layout()
    file_path = save_dir / f"dire_mse_{dataset}_{window_size}.pdf"
    fig.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Plot successfully saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=str, default="sssd")
    args = parser.parse_args()

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Times New Roman",
        font_scale=2.25,
        palette="bright",
        rc={
            "mathtext.fontset": "stix",
            "axes.formatter.use_mathtext": True,
            "text.usetex": True,
        },
    )

    save_dir = Path(f"plots_dire/{args.detector}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for window_size in WINDOW_SIZES:
        for dataset in DATASETS:
            try:
                plot_mse_distribution(dataset, window_size, save_dir, args.detector)
            except Exception as e:
                print(f"[SKIP] Dataset: {dataset}, Window Size: {window_size}")
                print(f"Error: {e}")
