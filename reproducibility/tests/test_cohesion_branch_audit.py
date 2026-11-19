"""Tests for the per-dataset cohesion-branch execution audit."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from paper_conventions import PAPER_DATASET_FILES  # noqa: E402
from run_paper_experiments import _write_cohesion_branch_audit  # noqa: E402


class CohesionBranchAuditTests(unittest.TestCase):
    @staticmethod
    def _rows(count: int = 20, empty_rejections: int = 0):
        return [
            {
                "dataset": PAPER_DATASET_FILES[index],
                "tree_singleton_cohesion_calls": index,
                "tree_views_with_singleton_cohesion": int(index > 0),
                "tree_empty_cohesion_rejections": empty_rejections,
            }
            for index in range(count)
        ]

    def test_complete_20_dataset_audit_passes_and_uses_abbreviations(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            payload = _write_cohesion_branch_audit(self._rows(), output)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["observed_dataset_count"], 20)
            with (output / "cohesion_branch_audit.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["dataset"], "Chess")
            self.assertEqual(rows[-1]["dataset"], "WPBC")

    def test_incomplete_or_illegal_execution_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            payload = _write_cohesion_branch_audit(
                self._rows(count=19, empty_rejections=1), Path(folder)
            )
            self.assertEqual(payload["status"], "BLOCKED")
            self.assertEqual(payload["observed_dataset_count"], 19)
            self.assertEqual(payload["total_empty_cohesion_rejections"], 19)


if __name__ == "__main__":
    unittest.main()
