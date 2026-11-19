"""Validate that paper-facing MFTAD tables use one fixed score source.

The check is intentionally independent of pandas and labels.  It compares the
full-precision benchmark rows with the M11 interaction rows, then verifies
that the displayed comparison matrices are their four-decimal projections.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _score_map(path: Path, variant: str) -> dict[str, tuple[float, float]]:
    rows = [row for row in _rows(path) if row.get("variant") == variant]
    if not rows:
        raise AssertionError(f"no rows for variant={variant!r} in {path}")
    result = {row["dataset"]: (float(row["auc"]), float(row["ap"])) for row in rows}
    if len(result) != len(rows):
        raise AssertionError(f"duplicate dataset rows for variant={variant!r}")
    return result


def _matrix_column(path: Path) -> dict[str, str]:
    rows = _rows(path)
    fieldnames = rows[0].keys() if rows else ()
    if "MFTAD" not in fieldnames:
        raise AssertionError(f"MFTAD column missing from {path}")
    return {row["dataset"]: row["MFTAD"].strip() for row in rows}


def validate(root: Path) -> None:
    benchmark = _score_map(
        root / "01_fixed20_benchmark" / "raw_results.csv", "relation"
    )
    full = _score_map(
        root / "02_fixed20_ablation" / "raw_results.csv",
        "r3_multiscale_leaf_path_gta",
    )
    if benchmark.keys() != full.keys():
        raise AssertionError("benchmark and M11 dataset sets differ")
    for dataset in sorted(benchmark):
        for metric_index, metric in enumerate(("AUC", "AP")):
            left = benchmark[dataset][metric_index]
            right = full[dataset][metric_index]
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(
                    f"M11 mismatch for {dataset} {metric}: benchmark={left!r}, M11={right!r}"
                )

    for metric_index, name in enumerate(("selected_auc_matrix.csv", "selected_ap_matrix.csv")):
        matrix = _matrix_column(root / "07_comparison_matrices" / name)
        for dataset, scores in benchmark.items():
            displayed = str(
                Decimal(str(scores[metric_index])).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
            )
            if matrix.get(dataset) != displayed:
                raise AssertionError(
                    f"display mismatch for {dataset} {name}: "
                    f"expected {displayed}, found {matrix.get(dataset)!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    validate(args.results_root)
    print("PASS: benchmark rows, M11 interaction rows, and displayed matrices agree.")


if __name__ == "__main__":
    main()
