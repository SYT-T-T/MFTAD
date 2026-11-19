"""Tests for fail-closed full-precision comparison-matrix generation."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from build_comparison_matrices import (  # noqa: E402
    BASELINE_METHODS,
    build_matrices,
)
from paper_conventions import PAPER_DATASET_FILES  # noqa: E402


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ComparisonMatrixTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        benchmark = root / "best_auc.csv"
        baseline = root / "results_long.csv"
        benchmark_rows = [
            {"dataset": dataset, "auc": f"0.{index + 10}", "ap": f"0.{index + 20}"}
            for index, dataset in reversed(list(enumerate(PAPER_DATASET_FILES)))
        ]
        baseline_rows = [
            {
                "dataset": dataset,
                "method": method,
                "auc": f"0.{method_index + 10}",
                "ap": f"0.{method_index + 20}",
                "error": "",
            }
            for dataset in PAPER_DATASET_FILES
            for method_index, method in enumerate(BASELINE_METHODS)
        ]
        _write_csv(benchmark, ["dataset", "auc", "ap"], benchmark_rows)
        _write_csv(
            baseline,
            ["dataset", "method", "auc", "ap", "error"],
            baseline_rows,
        )
        return benchmark, baseline

    def test_builds_aligned_matrices_in_fixed_dataset_order(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            benchmark, baseline = self._inputs(root)
            manifest = build_matrices(benchmark, baseline, root / "output")

            with (root / "output" / "selected_auc_matrix.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["dataset"] for row in rows], list(PAPER_DATASET_FILES))
            self.assertEqual(rows[0]["MFTAD"], "0.10")
            self.assertEqual(manifest["status"], "PASS")

    def test_duplicate_baseline_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            benchmark, baseline = self._inputs(root)
            with baseline.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [PAPER_DATASET_FILES[0], BASELINE_METHODS[0], "0.5", "0.4", ""]
                )

            with self.assertRaisesRegex(ValueError, "Duplicate baseline"):
                build_matrices(benchmark, baseline, root / "output")


if __name__ == "__main__":
    unittest.main()
