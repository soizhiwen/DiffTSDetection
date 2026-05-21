"""Generate LaTeX quality tables for synthetic time series generators.

The script reads JSON summaries from the wavestitch-synth dataset folder and
emits one table for the requested window size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

GENERATORS: Sequence[str] = ("sssd", "tsdiff", "diffusionts", "wavestitch")
GENERATOR_NAMES = {
    "sssd": "SSSD",
    "tsdiff": "TSDiff",
    "diffusionts": "Diffusion-TS",
    "wavestitch": "WaveStitch",
}
DATASET_NAMES = {
    "electricity": "Electricity",
    "energy": "Energy",
    "etth": "ETTh1",
    "ettm": "ETTm1",
    "exchange_rate": "Exchange Rate",
    "fmri": "fMRI",
    "illness": "Illness",
    "stock": "Stock",
    "traffic": "Traffic",
    "weather": "Weather",
}
METRICS: Sequence[tuple[str, str]] = (
    ("context_fid", "Context-FID"),
    ("correlational", "Correlational"),
    ("discriminative", "Discriminative"),
    ("predictive", "Predictive"),
)
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/wavestitch-synth/datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table for a single window size."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        required=True,
        choices=(32, 64, 128),
        help="Window size to render in the table.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder containing the generator subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table to a file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_value(mean: Any, std: Any) -> str:
    mean_value = float(mean)
    std_value = float(std)
    if mean_value < 10:
        precision = ".3f"
    elif mean_value < 100:
        precision = ".2f"
    elif mean_value < 1000:
        precision = ".1f"
    else:
        precision = ".0f"
    return f"\\val{{{mean_value:{precision}}}}{{{std_value:{precision}}}}{{}}"


def build_table(dataset_root: Path, window_size: int) -> str:
    generator_data = {}
    for generator in GENERATORS:
        json_path = dataset_root / generator / f"{generator}_{window_size}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing JSON file: {json_path}")
        generator_data[generator] = load_json(json_path)

    dataset_names = list(generator_data[GENERATORS[0]].keys())
    for generator in GENERATORS[1:]:
        current_names = list(generator_data[generator].keys())
        if current_names != dataset_names:
            raise ValueError(
                f"Dataset keys in {generator} do not match the first generator."
            )

    lines: List[str] = []
    lines.extend(
        [
            "\\begin{table}[ht!]",
            "",
            "\\centering",
            f"\\caption{{{window_size}-length synthetic time series quality of each generator across ten datasets. All metrics are lower the better.}}",
            f"\\label{{tab:quality_{window_size}}}",
            "% \\resizebox{\\linewidth}{!}{",
            "\\begin{tabular}{llcccc}",
            "",
            "\\toprule",
            "",
            "Dataset & Generator & Context-FID $\\downarrow$ & Correlational $\\downarrow$ & Discriminative $\\downarrow$ & Predictive $\\downarrow$ \\\\",
            "",
            "\\midrule",
            "",
        ]
    )

    for dataset_index, dataset_name in enumerate(dataset_names):
        display_dataset_name = DATASET_NAMES.get(dataset_name, dataset_name)
        for generator_index, generator in enumerate(GENERATORS):
            metrics = generator_data[generator][dataset_name]
            row_prefix = (
                f"\\multirow{{{len(GENERATORS)}}}{{*}}{{{display_dataset_name}}}"
            )
            display_name = GENERATOR_NAMES[generator]
            if generator_index == 0:
                row_start = f"{row_prefix} & {display_name}"
            else:
                row_start = f"& {display_name}"

            values = []
            for metric_key, _ in METRICS:
                metric_values = metrics[metric_key]
                values.append(format_value(metric_values["mean"], metric_values["std"]))
            lines.append(f"{row_start} & " + " & ".join(values) + " \\\\")

        if dataset_index < len(dataset_names) - 1:
            lines.extend(["", "\\cmidrule{1-6}", ""])

    lines.extend(
        [
            "",
            "\\bottomrule",
            "",
            "\\end{tabular}",
            "% }",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    table = build_table(args.dataset_root, args.window_size)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
    else:
        print(table)


if __name__ == "__main__":
    main()
