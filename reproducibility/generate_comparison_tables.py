"""Generate paper-ready AUC/AP LaTeX tables from one aligned matrix source."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata


def _read_matrix(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 3:
        raise ValueError(f"Matrix is empty or has too few columns: {path}")
    header = [cell.strip() for cell in rows[0]]
    methods = header[1:]
    datasets: List[str] = []
    values: List[List[float]] = []
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError(f"Malformed row in {path}: expected {len(header)} cells.")
        datasets.append(row[0].strip())
        values.append([float(cell) for cell in row[1:]])
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all() or len(set(datasets)) != len(datasets):
        raise ValueError(f"Matrix contains duplicate datasets or non-finite values: {path}")
    return datasets, methods, matrix


def _read_labels(path: Optional[Path], datasets: Sequence[str]) -> Dict[str, str]:
    if path is None:
        return {name: name for name in datasets}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {str(row.get("dataset", "")).strip(): str(row.get("label", "")).strip() for row in rows}
    if any(not key or not value for key, value in labels.items()):
        raise ValueError("Dataset labels must have non-empty dataset and label columns.")
    missing = [name for name in datasets if name not in labels]
    if missing:
        raise ValueError(f"Dataset labels are missing: {missing}")
    return labels


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _cell(value: float, bold: bool) -> str:
    text = f"{value:.4f}"
    return rf"\textbf{{{text}}}" if bold else text


def _table(metric: str, datasets: Sequence[str], methods: Sequence[str], values: np.ndarray,
           labels: Mapping[str, str]) -> str:
    rounded = np.round(values, 4)
    lines = [
        r"\begin{table*}[!t]",
        rf"\caption{{Per-dataset {('ROC AUC' if metric == 'auc' else 'average precision (AP)')} for the fixed 20-dataset benchmark with {len(methods)} methods. Higher is better; row maxima after four-decimal rounding are bold.}}",
        rf"\label{{tab:{metric}-all}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        rf"\begin{{tabular}}{{@{{}}l{'r' * len(methods)}@{{}}}}",
        r"\toprule",
        "Dataset & " + " & ".join(_latex_escape(method) for method in methods) + " \\\\",
        r"\midrule",
    ]
    for index, dataset in enumerate(datasets):
        row_max = float(np.max(rounded[index]))
        cells = [_cell(float(value), abs(float(value) - row_max) <= 1e-12) for value in rounded[index]]
        lines.append(_latex_escape(labels[dataset]) + " & " + " & ".join(cells) + " \\\\")
    means = values.mean(axis=0)
    ranks = np.mean(np.vstack([rankdata(-row, method="average") for row in values]), axis=0)
    lines.extend([
        r"\midrule",
        "Mean & " + " & ".join(f"{value:.4f}" for value in means) + " \\\\",
        "Avg. rank & " + " & ".join(f"{value:.3f}" for value in ranks) + " \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def generate(auc_matrix: Path, ap_matrix: Path, output_dir: Path,
             labels_path: Optional[Path] = None,
             statistics_dir: Optional[Path] = None) -> None:
    auc_datasets, auc_methods, auc_values = _read_matrix(auc_matrix)
    ap_datasets, ap_methods, ap_values = _read_matrix(ap_matrix)
    if auc_datasets != ap_datasets or auc_methods != ap_methods:
        raise ValueError("AUC and AP matrices must have identical dataset and method order.")
    labels = _read_labels(labels_path, auc_datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table_auc.tex").write_text(
        _table("auc", auc_datasets, auc_methods, auc_values, labels), encoding="utf-8"
    )
    (output_dir / "table_ap.tex").write_text(
        _table("ap", ap_datasets, ap_methods, ap_values, labels), encoding="utf-8"
    )
    if statistics_dir is not None:
        stats = {
            metric: json.loads((statistics_dir / f"{metric}_statistics.json").read_text(encoding="utf-8"))
            for metric in ("auc", "ap")
        }
        if "MFTAD" not in stats["auc"].get("average_ranks", {}):
            raise ValueError("Statistics do not contain an MFTAD column.")
        values = [
            rf"\newcommand{{\MFTADMeanAUC}}{{{auc_values.mean(axis=0)[auc_methods.index('MFTAD')]:.4f}}}",
            rf"\newcommand{{\MFTADMeanAP}}{{{ap_values.mean(axis=0)[ap_methods.index('MFTAD')]:.4f}}}",
            rf"\newcommand{{\MFTADAUCAverageRank}}{{{float(stats['auc']['average_ranks']['MFTAD']):.3f}}}",
            rf"\newcommand{{\MFTADAPAverageRank}}{{{float(stats['ap']['average_ranks']['MFTAD']):.3f}}}",
            rf"\newcommand{{\AUCFriedmanChiSquare}}{{{float(stats['auc']['friedman_statistic']):.4f}}}",
            rf"\newcommand{{\AUCImanDavenportStatistic}}{{{float(stats['auc']['iman_davenport_statistic']):.4f}}}",
            rf"\newcommand{{\AUCImanDavenportPValue}}{{{float(stats['auc']['iman_davenport_p']):.4g}}}",
            rf"\newcommand{{\APFriedmanChiSquare}}{{{float(stats['ap']['friedman_statistic']):.4f}}}",
            rf"\newcommand{{\APImanDavenportStatistic}}{{{float(stats['ap']['iman_davenport_statistic']):.4f}}}",
            rf"\newcommand{{\APImanDavenportPValue}}{{{float(stats['ap']['iman_davenport_p']):.4g}}}",
            rf"\newcommand{{\AUCNemenyiCD}}{{{float(stats['auc']['nemenyi_critical_difference']):.4f}}}",
            rf"\newcommand{{\APNemenyiCD}}{{{float(stats['ap']['nemenyi_critical_difference']):.4f}}}",
        ]
        (output_dir / "main_values.tex").write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auc-matrix", type=Path, required=True)
    parser.add_argument("--ap-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-labels", type=Path)
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        help="Optional Nemenyi output directory; also emits main_values.tex.",
    )
    args = parser.parse_args()
    generate(args.auc_matrix, args.ap_matrix, args.output_dir, args.dataset_labels, args.statistics_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
