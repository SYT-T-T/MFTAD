"""Fail-closed tests for the paper-facing 20-dataset interaction export."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from run_paper_experiments import (  # noqa: E402
    INTERACTION_FAMILIES,
    PAPER_INTERACTION_FAMILY,
    _write_paper_interaction_outputs,
)
from paper_conventions import PAPER_DATASET_FILES  # noqa: E402


class PaperInteractionExportTests(unittest.TestCase):
    @staticmethod
    def _complete_rows(dataset_count: int = 20):
        family = next(
            item
            for item in INTERACTION_FAMILIES
            if item["family"] == PAPER_INTERACTION_FAMILY
        )
        cells = {
            code: family[f"cell_{code}"] for code in ("00", "10", "01", "11")
        }
        scores = {
            "00": (0.50, 0.40),
            "10": (0.70, 0.55),
            "01": (0.60, 0.50),
            "11": (0.90, 0.75),
        }
        by_variant = {
            variant: {
                PAPER_DATASET_FILES[index]: {
                    "dataset": PAPER_DATASET_FILES[index],
                    "dataset_type": "numeric",
                    "auc": scores[code][0],
                    "ap": scores[code][1],
                }
                for index in range(dataset_count)
            }
            for code, variant in cells.items()
        }
        return by_variant, list(by_variant)

    def test_complete_design_emits_all_required_artifacts(self) -> None:
        by_variant, names = self._complete_rows()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            manifest = _write_paper_interaction_outputs(output, by_variant, names)
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["dataset_count"], 20)
            expected = {
                "interaction_20datasets.csv",
                "interaction_20datasets.tex",
                "interaction_summary.tex",
            }
            self.assertTrue(all((output / name).is_file() for name in expected))
            with (output / "interaction_20datasets.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual(rows[0]["dataset"], "Chess")
            self.assertNotIn("chess_nowin_227_variant1", {row["dataset"] for row in rows})
            self.assertAlmostEqual(float(rows[0]["auc_interaction"]), 0.10)
            self.assertAlmostEqual(float(rows[0]["ap_interaction"]), 0.10)
            self.assertFalse((output / "interaction_20datasets_blockers.json").exists())

    def test_incomplete_design_is_blocked_without_partial_tables(self) -> None:
        by_variant, names = self._complete_rows()
        missing_variant = names[0]
        by_variant[missing_variant].pop(PAPER_DATASET_FILES[-1])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            manifest = _write_paper_interaction_outputs(output, by_variant, names)
            self.assertEqual(manifest["status"], "BLOCKED")
            self.assertEqual(manifest["aligned_dataset_count"], 19)
            self.assertFalse((output / "interaction_20datasets.csv").exists())
            payload = json.loads(
                (output / "interaction_20datasets_blockers.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["status"], "BLOCKED")
            self.assertIn("00", payload["missing_by_cell"])


if __name__ == "__main__":
    unittest.main()
