"""Ranking-only semantics for exported interpretability cases."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from run_paper_experiments import _interpretability_selections  # noqa: E402


class InterpretabilityRoleTests(unittest.TestCase):
    def test_roles_are_rank_extrema_not_thresholded_predictions(self) -> None:
        scores = np.asarray([0.9, 0.9, 0.5, 0.1, 0.2])
        labels = np.asarray([0, 1, 1, 1, 0])
        selected, ranks = _interpretability_selections(scores, labels, top_k=2)
        self.assertEqual(
            selected,
            [
                ("top_score_rank", 0),
                ("top_score_rank", 1),
                ("highest_ranked_labeled_anomaly", 1),
                ("lowest_ranked_labeled_anomaly", 3),
                ("highest_ranked_labeled_normal", 0),
                ("lowest_ranked_labeled_normal", 4),
            ],
        )
        self.assertEqual(ranks[0], 1)
        self.assertEqual(ranks[1], 2)

    def test_top_k_is_exact_and_stable_across_a_cutoff_tie(self) -> None:
        selected, _ = _interpretability_selections(
            np.asarray([0.8, 0.7, 0.7, 0.6]),
            np.asarray([0, 0, 1, 1]),
            top_k=2,
        )
        self.assertEqual(selected[:2], [("top_score_rank", 0), ("top_score_rank", 1)])

    def test_nonfinite_scores_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _interpretability_selections(
                np.asarray([0.5, np.nan]), np.asarray([0, 1]), top_k=1
            )


if __name__ == "__main__":
    unittest.main()
