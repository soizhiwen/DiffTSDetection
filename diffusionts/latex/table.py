import json
from pathlib import Path
from argparse import ArgumentParser


def parse_run_dir_name(run_dir_name):
    # Expected: clf_{method}_{detector}_{dataset}_{window_size}
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


def format_pct(value):
    formatted = f"{value:.1f}"
    if len(formatted) < 4:
        formatted = f"{value:.2f}"
    if formatted == "100.0":
        formatted = "100."
    return formatted


def generate_latex_table(outputs, window_size=32, detector="sssd"):
    generators = [
        ("sssd", "SSSD"),
        ("tsdiff", "TSDiff"),
        ("diffusionts", "Diffusion-TS"),
        # ("timeautodiff", "TimeAutoDiff"),
        ("wavestitch", "WaveStitch"),
    ]
    methods = [
        ("dire", "DIRE"),
        ("inceptiontime", "InceptionTime"),
        ("litetime", "LITE"),
        ("disjointcnn", "Disjoint-CNN"),
    ]
    method_names = {name for name, _ in methods}
    metric_rows = [
        ("f1", "F1"),
        ("acc", "Acc."),
        ("ap", "AP"),
        ("auc", "AUC"),
        ("tpr_at_fpr_tau", "TPR"),
    ]

    # Group results by (generator, method), aggregate each metric over datasets.
    grouped = {}

    for base_path in outputs:
        base_path = Path(base_path)
        if not base_path.exists():
            continue

        candidate_dirs = []
        if base_path.is_dir() and base_path.name.startswith("clf_"):
            candidate_dirs = [base_path]
        elif base_path.is_dir():
            candidate_dirs = [
                p
                for p in base_path.iterdir()
                if p.is_dir() and p.name.startswith("clf_")
            ]

        for run_dir in candidate_dirs:
            parsed = parse_run_dir_name(run_dir.name)
            if parsed is None or parsed["window_size"] != window_size:
                continue
            if parsed["detector"] != detector:
                continue

            method = parsed["method"]
            if method not in method_names:
                continue

            dataset = parsed["dataset"]

            for gen, _ in generators:
                json_path = run_dir / f"{gen}.json"
                if not json_path.exists():
                    continue

                with open(json_path, "r") as f:
                    data = json.load(f)

                if data.get("generator") != gen:
                    continue

                key = (gen, method)
                row_is_gray = parsed["detector"] == gen
                if key not in grouped:
                    grouped[key] = {
                        "datasets": set(),
                        "detectors": set(),
                        "values": {m[0]: [] for m in metric_rows},
                        "non_gray_values": {m[0]: [] for m in metric_rows},
                    }

                grouped[key]["datasets"].add(dataset)
                grouped[key]["detectors"].add(parsed["detector"])

                for metric_key, _ in metric_rows:
                    val = data.get(metric_key)
                    if isinstance(val, (int, float)):
                        float_val = float(val)
                        grouped[key]["values"][metric_key].append(float_val)
                        if not row_is_gray:
                            grouped[key]["non_gray_values"][metric_key].append(
                                float_val
                            )

    def format_metric_cell(values, highlight=False, gray=False):
        if not values:
            return "-"

        mean_pct = (sum(values) / len(values)) * 100
        display = format_pct(mean_pct)
        if highlight:
            display = f"\\cellcolor{{blue!15}}{{{display}}}"
        if gray:
            display = f"\\textcolor{{gray}}{{{display}}}"
        return display

    table_body_blocks = []
    empty_values = {m[0]: [] for m in metric_rows}

    for gen_index, (gen, gen_pretty) in enumerate(generators):
        block_lines = []
        for method_index, (method, method_label) in enumerate(methods):
            entry = grouped.get(
                (gen, method), {"values": empty_values, "detectors": set()}
            )
            values = entry["values"]
            gray_row = gen in entry["detectors"]
            metric_cells = [
                format_metric_cell(values[metric_key], gray=gray_row)
                for metric_key, _ in metric_rows
            ]
            generator_cell = (
                f"\\multirow{{{len(methods)}}}{{*}}{{{gen_pretty}}} "
                if method_index == 0
                else ""
            )
            block_lines.append(
                f"{generator_cell}& {method_label} & "
                + " & ".join(metric_cells)
                + " \\\\"
            )
        table_body_blocks.append("\n".join(block_lines))
        if gen_index != len(generators) - 1:
            table_body_blocks.append(r"\cmidrule{1-7}")

    average_block_lines = []
    for method_index, (method, method_label) in enumerate(methods):
        metric_cells = []
        for metric_key, _ in metric_rows:
            collected_vals = []
            for gen, _ in generators:
                collected_vals.extend(
                    grouped.get((gen, method), {"non_gray_values": {metric_key: []}})[
                        "non_gray_values"
                    ].get(metric_key, [])
                )
            metric_cells.append(format_metric_cell(collected_vals, highlight=True))

        generator_cell = "\\multirow{2}{*}{Avg. OOD} " if method_index == 0 else ""
        average_block_lines.append(
            f"{generator_cell}& {method_label} & " + " & ".join(metric_cells) + " \\\\"
        )

    table_body_blocks.append(r"\cmidrule{1-7}\morecmidrules\cmidrule{1-7}")
    table_body_blocks.append("\n".join(average_block_lines))

    table_body_str = "\n".join(table_body_blocks)

    latex_table = f"""\\begin{{table}}[ht!]
% \\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Window size {window_size}. All metrics are higher the better. TPR at 1\\% FPR. The final Average block is highlighted.}}
\\label{{tab:exp_{window_size}}}
% \\resizebox{{\\columnwidth}}{{!}}{{%\n\\begin{{tabular}}{{llccccc}}

\\toprule

Generator & Method & F1 & Acc. & AP & AUC & TPR \\\\

\\midrule

{table_body_str}

\\bottomrule
\\end{{tabular}}
% }}
\\end{{table}}"""

    print(latex_table)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outputs", nargs="+", default=["outputs_clf"])
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--detector", type=str, default="sssd")
    args = parser.parse_args()

    generate_latex_table(
        outputs=args.outputs, window_size=args.window_size, detector=args.detector
    )
