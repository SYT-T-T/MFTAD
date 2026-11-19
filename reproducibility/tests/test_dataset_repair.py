import sys
import unittest
from pathlib import Path


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
if str(REPRODUCIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from repair_dataset_results import (  # noqa: E402
    CORRECT_DATASET,
    WRONG_DATASET,
    merge_dataset_rows,
)


class DatasetRepairTests(unittest.TestCase):
    def test_replaces_rows_without_changing_configuration_keys(self) -> None:
        source = [
            {"dataset": WRONG_DATASET, "variant": "relation", "delta": "0.2", "delta_report": "N/A", "mu": "0.005", "min_views": "50", "max_views": "100", "enabled_components": "leaf,gta,path", "config_id": "a", "auc": "0.1"},
            {"dataset": WRONG_DATASET, "variant": "relation", "delta": "0.2", "delta_report": "N/A", "mu": "0.005", "min_views": "1", "max_views": "1", "enabled_components": "leaf", "config_id": "b", "auc": "0.2"},
            {"dataset": "other", "variant": "relation", "config_id": "c", "auc": "0.3"},
        ]
        replacement = [
            {"dataset": CORRECT_DATASET, "dataset_type": "nominal", "variant": "relation", "delta": "0.2", "delta_report": "N/A", "mu": "0.005", "min_views": "50", "max_views": "100", "enabled_components": "leaf,gta,path", "config_id": "a", "auc": "0.4"},
            {"dataset": CORRECT_DATASET, "dataset_type": "nominal", "variant": "relation", "delta": "0.2", "delta_report": "N/A", "mu": "0.005", "min_views": "1", "max_views": "1", "enabled_components": "leaf", "config_id": "b", "auc": "0.5"},
        ]
        merged, audit = merge_dataset_rows("benchmark", source, replacement)
        self.assertEqual(len(merged), 3)
        self.assertEqual({row["dataset"] for row in merged}, {CORRECT_DATASET, "other"})
        self.assertEqual(audit["removed_rows"], 2)

    def test_rejects_a_stale_or_mismatched_replacement(self) -> None:
        source = [{"dataset": WRONG_DATASET, "variant": "relation", "delta": "0.2", "delta_report": "N/A", "mu": "0.005", "min_views": "50", "max_views": "100", "enabled_components": "leaf", "config_id": "a"}]
        stale = [{"dataset": WRONG_DATASET, "dataset_type": "nominal", "variant": "relation", "delta": "0.2", "delta_report": "N/A", "mu": "0.005", "min_views": "50", "max_views": "100", "enabled_components": "leaf", "config_id": "a"}]
        mismatched = [{"dataset": CORRECT_DATASET, "dataset_type": "nominal", "variant": "relation", "delta": "0.2", "delta_report": "N/A", "mu": "0.005", "min_views": "1", "max_views": "1", "enabled_components": "leaf", "config_id": "b"}]
        with self.assertRaises(ValueError):
            merge_dataset_rows("benchmark", source, stale)
        with self.assertRaises(ValueError):
            merge_dataset_rows("benchmark", source, mismatched)


if __name__ == "__main__":
    unittest.main()
