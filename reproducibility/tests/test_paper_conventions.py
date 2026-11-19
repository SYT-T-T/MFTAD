"""Tests for Table 2's paper-wide dataset abbreviations."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from paper_conventions import (  # noqa: E402
    PAPER_DATASET_ABBREVIATIONS,
    PAPER_DATASET_COUNT,
    paper_dataset_abbreviation,
)


class PaperConventionTests(unittest.TestCase):
    def test_table_2_defines_20_unique_abbreviations(self) -> None:
        self.assertEqual(PAPER_DATASET_COUNT, 20)
        self.assertEqual(len(set(PAPER_DATASET_ABBREVIATIONS.values())), 20)

    def test_raw_filename_and_existing_abbreviation_resolve_identically(self) -> None:
        self.assertEqual(paper_dataset_abbreviation("autos_variant1"), "Auto")
        self.assertEqual(paper_dataset_abbreviation("Auto"), "Auto")

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Table 2"):
            paper_dataset_abbreviation("unregistered_dataset")

    def test_executable_registry_matches_table_2_exactly(self) -> None:
        table_path = Path(__file__).resolve().parents[2] / "tables" / "table_datasets.tex"
        observed = {}
        pattern = re.compile(r"^(.+?) & ([A-Za-z]+) & [0-9,]+ &")
        for line in table_path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line.strip())
            if match:
                observed[match.group(1).replace(r"\_", "_")] = match.group(2)
        self.assertEqual(observed, PAPER_DATASET_ABBREVIATIONS)


if __name__ == "__main__":
    unittest.main()
