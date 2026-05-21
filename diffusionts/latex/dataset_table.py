"""Generate dataset-wise detectability tables for classifier outputs.

The script scans `outputs_clf`, groups results by window size, and renders a
large LaTeX table that shows per-dataset metrics for each generator/method pair
plus an average OOD block per dataset.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

GENERATORS: Sequence[Tuple[str, str]] = (
    ("sssd", "SSSD"),
    ("tsdiff", "TSDiff"),
    ("diffusionts", "Diffusion-TS"),
    ("wavestitch", "WaveStitch"),
)

METHODS: Sequence[Tuple[str, str]] = (
    ("dire", "DIRE"),
    ("disjointcnn", "Disjoint-CNN"),
)

METRICS: Sequence[Tuple[str, str]] = (
    ("f1", "F1"),
    ("acc", "Acc."),
    ("ap", "AP"),
    ("auc", "AUC"),
    ("tpr_at_fpr_tau", "TPR"),
)

DATASET_ORDER: Sequence[str] = (
    "electricity",
    "energy",
    "etth",
    "ettm",
    "exchange_rate",
    "fmri",
    "illness",
    "stock",
    "traffic",
    "weather",
)

DATASET_NAMES: Dict[str, str] = {
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

DEFAULT_OUTPUTS = ["outputs_clf"]


def parse_run_dir_name(run_dir_name: str) -> Dict[str, Any] | None:
    """Parse `clf_{method}_{detector}_{dataset}_{window_size}`."""

    parts = run_dir_name.split("_")
    if len(parts) < 5 or parts[0] != "clf":
        return None

    window_token = parts[-1]
    if not window_token.isdigit():
        return None

    method = parts[1]
    detector = parts[2]
    dataset = "_".join(parts[3:-1])
    window_size = int(window_token)

    if not dataset:
        return None

    return {
        "method": method,
        "detector": detector,
        "dataset": dataset,
        "window_size": window_size,
    }


def format_pct(value: float) -> str:
    """Format percentages in the same style as the existing classifier table."""

    formatted = f"{value:.1f}"
    if len(formatted) < 4:
        formatted = f"{value:.2f}"
    if formatted == "100.0":
        formatted = "100."
    return formatted


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def chunked(sequence: Sequence[str], size: int) -> List[List[str]]:
    return [
        list(sequence[index : index + size]) for index in range(0, len(sequence), size)
    ]


def column_ranges(dataset_count: int) -> List[str]:
    ranges: List[str] = []
    for index in range(dataset_count):
        start = 3 + index * len(METRICS)
        end = start + len(METRICS) - 1
        if dataset_count == 1:
            ranges.append(f"\\cmidrule{{{start}-{end}}}")
        elif index == 0:
            ranges.append(f"\\cmidrule(r){{{start}-{end}}}")
        elif index == dataset_count - 1:
            ranges.append(f"\\cmidrule(l){{{start}-{end}}}")
        else:
            ranges.append(f"\\cmidrule(rl){{{start}-{end}}}")
    return ranges


def load_results(
    outputs: Sequence[str], window_size: int, detector: str
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]:
    """Load all matching run JSON files for the requested window size."""

    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    method_names = {method for method, _ in METHODS}

    for base_path_raw in outputs:
        base_path = Path(base_path_raw)
        if not base_path.exists():
            continue

        candidate_dirs: Iterable[Path]
        if base_path.is_dir() and base_path.name.startswith("clf_"):
            candidate_dirs = [base_path]
        elif base_path.is_dir():
            candidate_dirs = sorted(
                path
                for path in base_path.iterdir()
                if path.is_dir() and path.name.startswith("clf_")
            )
        else:
            continue

        for run_dir in candidate_dirs:
            parsed = parse_run_dir_name(run_dir.name)
            if parsed is None or parsed["window_size"] != window_size:
                continue
            if parsed["detector"] != detector:
                continue
            if parsed["method"] not in method_names:
                continue

            dataset = parsed["dataset"]
            method = parsed["method"]
            detector = parsed["detector"]

            dataset_bucket = results.setdefault(dataset, {})
            method_bucket = dataset_bucket.setdefault(method, {})
            detector_bucket = method_bucket.setdefault(detector, {})

            for generator_name, _ in GENERATORS:
                json_path = run_dir / f"{generator_name}.json"
                if not json_path.exists():
                    continue

                data = read_json(json_path)
                if (
                    data.get("dataset") != dataset
                    or data.get("generator") != generator_name
                ):
                    continue

                detector_bucket[generator_name] = {
                    metric_key: float(data[metric_key])
                    for metric_key, _ in METRICS
                    if metric_key in data and isinstance(data[metric_key], (int, float))
                }

    return results


def compute_average_ood(
    values_by_generator: Dict[str, Dict[str, float]], detector: str
) -> Dict[str, float]:
    averages: Dict[str, float] = {}
    for metric_key, _ in METRICS:
        collected = [
            metric_values[metric_key]
            for generator_name, metric_values in values_by_generator.items()
            if generator_name != detector and metric_key in metric_values
        ]
        if collected:
            averages[metric_key] = sum(collected) / len(collected)
    return averages


def format_cell(
    value: float | None, highlight: bool = False, gray: bool = False
) -> str:
    if value is None:
        return "-"

    display = format_pct(value * 100)
    if highlight:
        display = f"\\cellcolor{{green!15}}{{{display}}}"
    if gray:
        display = f"\\textcolor{{gray}}{{{display}}}"
    return display


def build_dataset_block(
    dataset_names: Sequence[str],
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]],
) -> str:
    """Render a single 3-dataset-wide block."""

    num_dataset_cols = len(dataset_names)
    total_metric_columns = num_dataset_cols * len(METRICS)
    column_spec = "ll" + ("ccccc" * num_dataset_cols)

    lines: List[str] = [
        "\\toprule",
        "",
        "& "
        + " & ".join(
            f"\\multicolumn{{5}}{{c}}{{{DATASET_NAMES.get(dataset, dataset)}}}"
            for dataset in dataset_names
        )
        + " \\\\",
        "",
    ]
    lines.extend(column_ranges(num_dataset_cols))
    lines.extend(
        [
            "",
            "Generator & Method & "
            + " & ".join(
                f"{metric_label} $\\uparrow$"
                for _dataset in dataset_names
                for _, metric_label in METRICS
            )
            + " \\\\",
            "",
            "\\midrule",
            "",
        ]
    )

    for generator_index, (generator_name, generator_label) in enumerate(GENERATORS):
        for method_index, (method_name, method_label) in enumerate(METHODS):
            row_cells: List[str] = []
            for dataset_name in dataset_names:
                method_results = results.get(dataset_name, {}).get(method_name, {})
                detector_name = next(iter(method_results.keys()), None)
                generator_results = (
                    method_results.get(detector_name, {}) if detector_name else {}
                )
                metric_values = generator_results.get(generator_name, {})
                is_gray = detector_name == generator_name
                for metric_key, _ in METRICS:
                    row_cells.append(
                        format_cell(metric_values.get(metric_key), gray=is_gray)
                    )

            generator_cell = (
                f"\\multirow{{2}}{{*}}{{{generator_label}}} "
                if method_index == 0
                else ""
            )
            lines.append(
                f"{generator_cell}& {method_label} & " + " & ".join(row_cells) + " \\\\"
            )

        lines.extend(["\\cmidrule{1-%d}" % (2 + total_metric_columns), ""])

    average_lines: List[str] = []
    for method_index, (method_name, method_label) in enumerate(METHODS):
        row_cells: List[str] = []
        for dataset_name in dataset_names:
            method_results = results.get(dataset_name, {}).get(method_name, {})
            detector_name = next(iter(method_results.keys()), None)
            generator_results = (
                method_results.get(detector_name, {}) if detector_name else {}
            )
            averages = compute_average_ood(generator_results, detector_name or "")
            for metric_key, _ in METRICS:
                metric_value = averages.get(metric_key)
                row_cells.append(str(metric_value))

        average_lines.append(
            f"{('\\multirow{2}{*}{Avg. OOD} ' if method_index == 0 else '')}& {method_label} & "
            + " & ".join(row_cells)  # placeholder, formatted below
        )

    # Reformat the average rows with highlighting by metric and dataset.
    average_value_rows: Dict[str, List[List[float | None]]] = {}
    for method_name, method_label in METHODS:
        method_rows: List[List[float | None]] = []
        for dataset_name in dataset_names:
            method_results = results.get(dataset_name, {}).get(method_name, {})
            detector_name = next(iter(method_results.keys()), None)
            generator_results = (
                method_results.get(detector_name, {}) if detector_name else {}
            )
            averages = compute_average_ood(generator_results, detector_name or "")
            method_rows.append([averages.get(metric_key) for metric_key, _ in METRICS])
        average_value_rows[method_name] = method_rows

    average_rendered_rows: List[str] = []
    for method_index, (method_name, method_label) in enumerate(METHODS):
        row_cells: List[str] = []
        for dataset_index, dataset_name in enumerate(dataset_names):
            for metric_index, (metric_key, _) in enumerate(METRICS):
                candidates: List[Tuple[str, float]] = []
                for other_method_name, _ in METHODS:
                    value = average_value_rows[other_method_name][dataset_index][
                        metric_index
                    ]
                    if value is not None:
                        candidates.append((other_method_name, value))

                current_value = average_value_rows[method_name][dataset_index][
                    metric_index
                ]
                highlight = bool(
                    current_value is not None
                    and candidates
                    and current_value == max(candidate for _, candidate in candidates)
                )
                row_cells.append(format_cell(current_value, highlight=highlight))

        average_rendered_rows.append(
            f"{('\\multirow{2}{*}{Avg. OOD} ' if method_index == 0 else '')}& {method_label} & "
            + " & ".join(row_cells)
            + " \\\\"
        )

    lines.extend(average_rendered_rows)
    lines.extend(["", "\\bottomrule"])
    return column_spec + "\n" + "\n".join(lines)


def generate_latex_table(
    outputs: Sequence[str], window_size: int, detector: str
) -> str:
    results = load_results(outputs, window_size, detector)
    dataset_groups = chunked(DATASET_ORDER, 3)

    table_blocks: List[str] = []
    table_blocks.append("\\begin{table}[ht!]")
    table_blocks.append("% \\setlength{\\tabcolsep}{4pt}")
    table_blocks.append(
        f"\\caption{{Detectability across ten datasets for {window_size}-length. All metrics are higher the better, with TPR measured at a 1\\% FPR. Values in \\textcolor{{gray}}{{gray}} are ID, while values in black are OOD. The best average OOD scores are highlighted in \\colorbox{{green!15}}{{green}}.}}"
    )
    table_blocks.append(f"\\label{{tab:dataset_{window_size}}}")
    table_blocks.append("\\resizebox{\\linewidth}{!}{%")
    table_blocks.append(f"\\begin{{tabular}}{{ll{'ccccc' * 3}}}")
    table_blocks.append("")

    for group_index, dataset_group in enumerate(dataset_groups):
        if group_index > 0:
            table_blocks.append("\\toprule")
            table_blocks.append("")
        table_blocks.append(build_dataset_block(dataset_group, results))
        if group_index != len(dataset_groups) - 1:
            table_blocks.append("\\bottomrule")
            table_blocks.append("")

    table_blocks.append("\\end{tabular}")
    table_blocks.append("}")
    table_blocks.append("\\end{table}")
    return "\n".join(table_blocks)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--outputs", nargs="+", default=DEFAULT_OUTPUTS)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--detector", type=str, default="sssd")
    args = parser.parse_args()

    print(
        generate_latex_table(
            outputs=args.outputs, window_size=args.window_size, detector=args.detector
        )
    )


if __name__ == "__main__":
    main()
