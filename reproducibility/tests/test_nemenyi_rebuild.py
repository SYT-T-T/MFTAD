"""Input-contract tests for the strict Friedman/Nemenyi rebuild."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from paper_conventions import PAPER_DATASET_FILES  # noqa: E402
from rebuild_nemenyi import _nemenyi_groups, _read_matrix, rebuild  # noqa: E402


class NemenyiRebuildContractTests(unittest.TestCase):
    @staticmethod
    def _write_matrix(path: Path, methods=None, dataset_count: int = 20) -> None:
        methods = methods or ["MFTAD", "ECOD", "MFGAD"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", *methods])
            for index in range(dataset_count):
                writer.writerow(
                    [
                        PAPER_DATASET_FILES[index],
                        0.80 + index / 1000,
                        0.70,
                        0.60,
                    ]
                )

    def test_valid_matrix_preserves_order_and_20_rows(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "auc.csv"
            self._write_matrix(path)
            datasets, methods, values = _read_matrix(path)
            self.assertEqual(len(datasets), 20)
            self.assertEqual(methods, ["MFTAD", "ECOD", "MFGAD"])
            self.assertEqual(values.shape, (20, 3))

    def test_duplicate_required_method_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "auc.csv"
            self._write_matrix(path, methods=["ECOD", "ECOD", "MFGAD"])
            with self.assertRaisesRegex(ValueError, "duplicate columns"):
                _read_matrix(path)

    def test_case_insensitive_duplicate_method_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "auc.csv"
            self._write_matrix(path, methods=["ECOD", "MFGAD", "LOF", "lof"])
            with self.assertRaisesRegex(ValueError, "case-insensitive duplicate"):
                _read_matrix(path)

    def test_incomplete_dataset_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "auc.csv"
            self._write_matrix(path, dataset_count=19)
            with self.assertRaisesRegex(ValueError, "exactly 20"):
                _read_matrix(path)

    def test_missing_mfgad_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "auc.csv"
            self._write_matrix(path, methods=["MFTAD", "ECOD", "LOF"])
            with self.assertRaisesRegex(ValueError, "MFGAD exactly once"):
                _read_matrix(path)

    def test_auc_ap_method_order_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            auc = root / "auc.csv"
            ap = root / "ap.csv"
            self._write_matrix(auc)
            self._write_matrix(ap, methods=["ECOD", "MFTAD", "MFGAD"])
            with self.assertRaisesRegex(ValueError, "same method order"):
                rebuild(auc, ap, root / "output", source_root=root)

    def test_complete_rebuild_emits_one_source_chain(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            auc = root / "auc.csv"
            ap = root / "ap.csv"
            self._write_matrix(auc)
            self._write_matrix(ap)
            output = root / "output"
            manifest = rebuild(auc, ap, output, source_root=root)
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["dataset_count"], 20)
            self.assertEqual(manifest["datasets"][0], "Chess")
            for name in (
                "auc_statistics.json",
                "ap_statistics.json",
                "auc_average_ranks.csv",
                "ap_average_ranks.csv",
                "nemenyi_auc.svg",
                "nemenyi_ap.svg",
                "nemenyi_values.tex",
                "statistics_provenance.json",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_maximal_nemenyi_groups_are_deterministic(self) -> None:
        groups = _nemenyi_groups(
            {"A": 1.0, "B": 1.5, "C": 2.4, "D": 2.5},
            critical_difference=1.1,
        )
        self.assertEqual(groups, [["A", "B"], ["B", "C", "D"]])


if __name__ == "__main__":
    unittest.main()
