"""Publication-output tests that do not depend on benchmark results."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from experiment_utils import plot_critical_difference, plot_sensitivity  # noqa: E402


class ReportingFigureTests(unittest.TestCase):
    def test_sensitivity_plot_can_emit_vector_svg_without_best_marker(self) -> None:
        rows = [
            {"dataset": dataset, "value": value, "auc": score}
            for dataset, offset in (("Auto", 0.00), ("Glass", 0.02))
            for value, score in ((0.2, 0.70 + offset), (0.8, 0.72 + offset))
        ]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "sensitivity_delta_auc.svg"
            plot_sensitivity(
                rows,
                "value",
                "auc",
                output,
                reference_x=None,
            )
            self.assertTrue(output.is_file())
            text = output.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertIn('stroke-opacity="0.25"', text)
            self.assertIn("Dataset mean", text)
            self.assertIn("No best setting is marked", text)

    def test_critical_difference_plot_can_emit_editable_svg(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "nemenyi_auc.svg"
            plot_critical_difference(
                {"MFTAD": 1.2, "ECOD": 2.1, "MFGAD": 2.7},
                1.0,
                output,
                title="AUC critical difference",
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertEqual(text.count("MFTAD ("), 1)
            self.assertEqual(text.count("ECOD ("), 1)
            self.assertEqual(text.count("MFGAD ("), 1)
            self.assertIn("CD = 1.000", text)


if __name__ == "__main__":
    unittest.main()
