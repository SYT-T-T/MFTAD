"""Focused tests for the fixed-delta KFGOD comparator adapter."""

from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from build_comparison_matrices import build_matrices  # noqa: E402
from paper_conventions import PAPER_DATASET_FILES  # noqa: E402
from run_paper_experiments import _kfgod_scores  # noqa: E402


class KFGODAdapterTests(unittest.TestCase):
    def test_ball_scores_are_mapped_back_to_all_samples(self) -> None:
        gb = types.SimpleNamespace(
            get_GB=lambda matrix: [
                np.column_stack((matrix[[0, 2]], np.array([0.0, 2.0]))),
                np.column_stack((matrix[[1, 3]], np.array([1.0, 3.0]))),
            ]
        )
        kfgod = types.SimpleNamespace(KFGOD=lambda centers, delta: np.array([0.2, 0.8]))
        scores = _kfgod_scores(
            np.array([[0.0], [1.0], [2.0], [3.0]]), 0.3, gb, kfgod
        )
        np.testing.assert_allclose(scores, [0.2, 0.8, 0.2, 0.8])

    def test_extended_matrix_requires_one_kfgod_row_per_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            benchmark = root / "benchmark.csv"
            baseline = root / "baseline.csv"
            kfgod = root / "kfgod.csv"
            with benchmark.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("dataset", "auc", "ap"))
                writer.writeheader()
                writer.writerows(
                    {"dataset": name, "auc": "0.5", "ap": "0.4"}
                    for name in PAPER_DATASET_FILES
                )
            from build_comparison_matrices import BASELINE_METHODS

            with baseline.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("dataset", "method", "auc", "ap", "error")
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "dataset": dataset,
                        "method": method,
                        "auc": "0.5",
                        "ap": "0.4",
                        "error": "",
                    }
                    for dataset in PAPER_DATASET_FILES
                    for method in BASELINE_METHODS
                )
            with kfgod.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("dataset", "method", "auc", "ap", "error")
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "dataset": dataset,
                        "method": "KFGOD",
                        "auc": "0.6",
                        "ap": "0.5",
                        "error": "",
                    }
                    for dataset in PAPER_DATASET_FILES
                )
            manifest = build_matrices(benchmark, baseline, root / "out", kfgod)
            self.assertEqual(manifest["method_order"][-2:], ["KFGOD", "MFTAD"])
            with (root / "out" / "selected_auc_matrix.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                self.assertIn("KFGOD", next(csv.reader(handle)))


if __name__ == "__main__":
    unittest.main()
