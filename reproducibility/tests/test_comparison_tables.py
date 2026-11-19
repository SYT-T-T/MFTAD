from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from generate_comparison_tables import generate  # noqa: E402


class ComparisonTableTests(unittest.TestCase):
    def test_generates_tex_for_explicit_method_columns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for metric, rows in {
                "auc": [("d1", "0.8", "0.9"), ("d2", "0.7", "0.6")],
                "ap": [("d1", "0.4", "0.5"), ("d2", "0.3", "0.2")],
            }.items():
                with (root / f"{metric}.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["dataset", "KFGOD", "MFTAD"])
                    writer.writerows(rows)
            labels = root / "labels.csv"
            with labels.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["dataset", "label"])
                writer.writerows([["d1", "D1"], ["d2", "D2"]])
            output = root / "tables"
            generate(root / "auc.csv", root / "ap.csv", output, labels)
            text = (output / "table_auc.tex").read_text(encoding="utf-8")
            self.assertIn("KFGOD & MFTAD", text)
            self.assertIn(r"\textbf{0.9000}", text)


if __name__ == "__main__":
    unittest.main()
