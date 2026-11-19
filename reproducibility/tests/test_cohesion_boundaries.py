"""Boundary tests for the audited relation-consistent tree implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from relation_mftad import RelationConsistentTree  # noqa: E402


class CohesionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = RelationConsistentTree.__new__(RelationConsistentTree)
        self.tree.R_view = np.asarray(
            [
                [1.0, 0.25, 0.00],
                [0.25, 1.0, 0.50],
                [0.00, 0.50, 1.0],
            ],
            dtype=np.float64,
        )
        self.tree.cohesion_singleton_calls = 0
        self.tree.cohesion_empty_rejections = 0

    def test_empty_index_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            self.tree._cohesion_statistics(np.asarray([], dtype=np.int64))
        self.assertEqual(self.tree.cohesion_empty_rejections, 1)

    def test_singleton_has_unit_cohesion_and_unit_cardinality(self) -> None:
        cohesion, cardinalities = self.tree._cohesion_statistics(
            np.asarray([1], dtype=np.int64)
        )
        self.assertEqual(cohesion, 1.0)
        np.testing.assert_array_equal(cardinalities, np.asarray([1.0]))
        self.assertEqual(self.tree.cohesion_singleton_calls, 1)

    def test_two_element_case_matches_normalized_pairwise_definition(self) -> None:
        cohesion, cardinalities = self.tree._cohesion_statistics(
            np.asarray([0, 1], dtype=np.int64)
        )
        self.assertAlmostEqual(cohesion, 0.25)
        np.testing.assert_allclose(cardinalities, np.asarray([1.25, 1.25]))

    def test_general_case_matches_normalized_pairwise_definition(self) -> None:
        cohesion, cardinalities = self.tree._cohesion_statistics(
            np.asarray([0, 1, 2], dtype=np.int64)
        )
        self.assertAlmostEqual(cohesion, 0.25)
        np.testing.assert_allclose(
            cardinalities,
            np.asarray([1.25, 1.75, 1.50]),
        )


if __name__ == "__main__":
    unittest.main()
