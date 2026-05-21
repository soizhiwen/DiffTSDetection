import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METHODS = ["dire", "disjointcnn"]
QUALITY_METRICS = ["context_fid", "correlational", "discriminative", "predictive"]
QUALITY_LOWER_IS_BETTER = {
    "context_fid": True,
    "correlational": True,
    "discriminative": True,
    "predictive": True,
}
DETECTABILITY_METRICS = ["tpr_at_fpr_tau", "ap", "f1", "auc", "acc"]
DEFAULT_WINDOW_SIZE = "32"
MARKERS = ["X", "o", "^", "D", "P", "s", "v", "<", ">", "*"]
PRETTY_GENERATOR_NAMES = {
    "sssd": "SSSD",
    "tsdiff": "TSDiff",
    "diffusionts": "Diffusion-TS",
    "wavestitch": "WaveStitch",
}
PRETTY_METHOD_NAMES = {
    "dire": "DIRE",
    "disjointcnn": "Disjoint-CNN",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_quality_raw(
    datasets_root: Path,
    quality_metrics: list[str],
    window_size: str,
) -> pd.DataFrame:
    rows = []

    for generator_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        generator = generator_dir.name
        metrics_file = generator_dir / f"{generator}_{window_size}.json"
        if not metrics_file.exists():
            continue

        payload = _read_json(metrics_file)
        dataset_count = len(payload)
        if dataset_count == 0:
            continue

        for dataset_name, dataset_metrics in payload.items():
            for metric in quality_metrics:
                metric_payload = dataset_metrics.get(metric)
                if isinstance(metric_payload, dict):
                    metric_value = _to_float_or_none(metric_payload.get("mean"))
                else:
                    metric_value = _to_float_or_none(metric_payload)

                rows.append(
                    {
                        "generator": generator,
                        "window_size": int(window_size),
                        "dataset": dataset_name,
                        "metric": metric,
                        "metric_value": metric_value,
                        "expected_dataset_count": dataset_count,
                    }
                )

    return pd.DataFrame(rows)


def normalize_quality_metrics(
    quality_raw_df: pd.DataFrame,
    lower_is_better: dict[str, bool],
) -> pd.DataFrame:
    norm_df = quality_raw_df.copy()
    norm_df["metric_norm"] = None

    # Normalize per (dataset, metric) across generators.
    for (dataset, metric), sub_df in norm_df.groupby(["dataset", "metric"]):
        valid = sub_df["metric_value"].dropna()
        if valid.empty:
            continue

        min_val = valid.min()
        max_val = valid.max()

        if max_val == min_val:
            metric_norm = pd.Series(1.0, index=sub_df.index)
        else:
            metric_norm = (sub_df["metric_value"] - min_val) / (max_val - min_val)

        if lower_is_better.get(metric, False):
            metric_norm = 1.0 - metric_norm

        norm_df.loc[sub_df.index, "metric_norm"] = metric_norm

    norm_df["metric_norm"] = pd.to_numeric(norm_df["metric_norm"], errors="coerce")
    return norm_df


def aggregate_quality(norm_quality_df: pd.DataFrame) -> pd.DataFrame:
    def _mean_non_null(series: pd.Series):
        valid = series.dropna()
        return valid.mean() if not valid.empty else None

    dataset_level = norm_quality_df.groupby(
        ["generator", "dataset"], as_index=False
    ).agg(
        dataset_quality_score=("metric_norm", _mean_non_null),
        quality_metric_count=("metric_norm", lambda s: int(s.notna().sum())),
    )

    total_dataset_count = int(norm_quality_df["dataset"].nunique())
    if total_dataset_count == 0:
        return pd.DataFrame(
            columns=["generator", "quality_score", "quality_metric_count"]
        )

    quality_df = (
        dataset_level.groupby(["generator"], as_index=False)
        .agg(
            quality_score=(
                "dataset_quality_score",
                lambda s: s.dropna().sum() / total_dataset_count,
            ),
            quality_metric_count=("quality_metric_count", "sum"),
            available_dataset_count=("dataset", "nunique"),
        )
        .sort_values(["generator"])
    )

    quality_df["expected_dataset_count"] = total_dataset_count
    return quality_df


def collect_detectability(
    detectability_root: Path,
    generator_names: set[str],
    window_size: str,
    detectability_metrics: list[str],
) -> pd.DataFrame:
    rows = []
    folder_re = re.compile(
        r"^clf_(?P<method>.+?)_sssd_(?P<dataset>.+)_(?P<window>\d+)$"
    )

    for clf_dir in sorted(p for p in detectability_root.iterdir() if p.is_dir()):
        match = folder_re.match(clf_dir.name)
        if not match:
            continue

        method = match.group("method")
        if method not in METHODS:
            continue
        dataset = match.group("dataset")
        window = match.group("window")
        if window != window_size:
            continue

        for generator in sorted(generator_names):
            result_file = clf_dir / f"{generator}.json"
            if not result_file.exists():
                continue

            payload = _read_json(result_file)
            vals = []
            for metric in detectability_metrics:
                value = _to_float_or_none(payload.get(metric))
                if value is not None:
                    vals.append(value)

            detectability_score = sum(vals) / len(vals) if vals else None
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "generator": generator,
                    "window_size": int(window),
                    "detectability_score": detectability_score,
                    "detectability_metric_count": len(vals),
                }
            )

    detect_df = pd.DataFrame(rows)
    if detect_df.empty:
        return detect_df

    detect_agg_df = (
        detect_df.groupby(["method", "generator"], as_index=False)
        .agg(
            detectability_score=("detectability_score", "mean"),
            detectability_dataset_count=("dataset", "nunique"),
            detectability_metric_count=("detectability_metric_count", "mean"),
        )
        .sort_values(["method", "generator"])
    )
    return detect_agg_df


def _build_method_markers(methods: list[str]) -> dict[str, str]:
    return {method: MARKERS[i % len(MARKERS)] for i, method in enumerate(methods)}


def plot_scatter(df: pd.DataFrame, output_path: Path):
    if df.empty:
        raise ValueError("No paired quality/detectability rows found to plot.")

    methods = sorted(df["method"].unique())
    method_markers = _build_method_markers(methods)
    generators = ["sssd", "tsdiff", "diffusionts", "wavestitch"]

    plot_df = df.copy()
    plot_df["Generator"] = plot_df["generator"].map(
        lambda g: PRETTY_GENERATOR_NAMES.get(g, g)
    )
    plot_df["Method"] = plot_df["method"].map(lambda m: PRETTY_METHOD_NAMES.get(m, m))

    generator_order = [g for g in generators if g in set(plot_df["generator"])]
    if not generator_order:
        generator_order = sorted(plot_df["generator"].unique())
    generator_pretty_order = [PRETTY_GENERATOR_NAMES.get(g, g) for g in generator_order]

    method_order = [m for m in METHODS if m in set(plot_df["method"])]
    if not method_order:
        method_order = methods
    method_pretty_order = [PRETTY_METHOD_NAMES.get(m, m) for m in method_order]
    method_pretty_markers = {
        PRETTY_METHOD_NAMES.get(m, m): marker for m, marker in method_markers.items()
    }
    palette = sns.color_palette("bright", n_colors=len(generator_pretty_order) + 1)[1:]
    generator_palette = dict(zip(generator_pretty_order, palette))

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
    plt.figure(figsize=(7, 6))
    ax = sns.scatterplot(
        data=plot_df,
        x="quality_score",
        y="detectability_score",
        hue="Generator",
        hue_order=generator_pretty_order,
        palette=generator_palette,
        style="Method",
        style_order=method_pretty_order,
        markers=method_pretty_markers,
        s=150,
        # alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        # legend=False,
    )

    ax.set_xlabel(r"Aggregated Quality $\rightarrow$")
    ax.set_ylabel(r"Aggregated Detectability $\rightarrow$")
    ax.grid(alpha=0.5)

    # Use rectangle patches for hue entries and marker symbols for method entries.
    generator_handles = [
        Patch(facecolor=generator_palette[g], label=g) for g in generator_pretty_order
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=method_pretty_markers[m],
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=12,
            label=m,
        )
        for m in method_pretty_order
    ]

    combined_handles = generator_handles + method_handles
    ax.legend(
        handles=combined_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=6,
        title=None,
        frameon=False,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a quality vs detectability scatter plot."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/home/ubuntu/wavestitch-synth/datasets"),
        help="Root folder containing generator subfolders and *_<window>.json quality files.",
    )
    parser.add_argument(
        "--detectability-root",
        type=Path,
        default=Path("/home/ubuntu/TimeDet/outputs_clf"),
        help="Root folder with clf_<method>_sssd_<dataset>_<window> folders.",
    )
    parser.add_argument(
        "--quality-metrics",
        type=str,
        default=",".join(QUALITY_METRICS),
        help=(
            "Comma-separated quality metrics to include. "
            "Example: context_fid,correlational,discriminative,predictive"
        ),
    )
    parser.add_argument(
        "--window",
        type=str,
        default=DEFAULT_WINDOW_SIZE,
        help="Single window size to include (e.g. 32, 64, or 128).",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Optional output plot path. Default includes the selected window size.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Default includes the selected window size.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    quality_metrics = [m.strip() for m in args.quality_metrics.split(",") if m.strip()]
    window = args.window.strip()
    if window not in {"32", "64", "128"}:
        raise ValueError("--window must be one of: 32, 64, 128")

    output_plot = args.output_plot or Path(
        f"plots/quality_detectability_scatter_{window}.pdf"
    )
    output_csv = args.output_csv or Path(
        f"plots/quality_detectability_scatter_data_{window}.csv"
    )

    quality_raw = collect_quality_raw(args.datasets_root, quality_metrics, window)
    if quality_raw.empty:
        raise ValueError("No quality rows found. Check datasets root and metric names.")

    quality_norm = normalize_quality_metrics(quality_raw, QUALITY_LOWER_IS_BETTER)
    quality_agg = aggregate_quality(quality_norm)

    generator_names = set(quality_agg["generator"].unique())
    detectability_agg = collect_detectability(
        args.detectability_root,
        generator_names=generator_names,
        window_size=window,
        detectability_metrics=DETECTABILITY_METRICS,
    )

    if detectability_agg.empty:
        raise ValueError(
            "No detectability rows found. Check detectability root folder structure."
        )

    merged = pd.merge(
        quality_agg,
        detectability_agg,
        on=["generator"],
        how="inner",
    ).sort_values(["generator", "method"])

    if merged.empty:
        raise ValueError("No matching rows after joining quality and detectability.")

    merged["window_size"] = int(window)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    plot_scatter(merged, output_plot)

    print(f"Saved plot: {output_plot}")
    print(f"Saved table: {output_csv}")
    print(f"Rows plotted: {len(merged)}")
    print(
        "Quality metrics used: "
        + ", ".join(quality_metrics)
        + f" | window: {window}"
        + " (normalized per dataset+metric across generators, then averaged over datasets)"
    )


if __name__ == "__main__":
    main()
