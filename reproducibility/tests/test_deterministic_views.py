"""Executable edge specification for deterministic attribute-view generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from relation_mftad import RelationConsistentMFTAD  # noqa: E402


def _model(
    n_features: int,
    *,
    min_views: int = 6,
    max_views: int = 10,
    min_ratio: float = 1.0,
    supplemental_ratio: float = 0.6,
    supplemental_count: int = 6,
    include_full_view: bool = True,
    importance: np.ndarray | None = None,
) -> RelationConsistentMFTAD:
    model = RelationConsistentMFTAD(
        min_views=min_views,
        max_views=max_views,
        min_view_dimension_ratio=min_ratio,
        supplemental_view_dimension_ratio=supplemental_ratio,
        supplemental_view_count=supplemental_count,
        include_full_view=include_full_view,
        use_numba=False,
    )
    model.X_ = np.zeros((2, n_features), dtype=np.float64)
    model.attribute_importance_ = (
        np.arange(n_features, 0, -1, dtype=np.float64)
        if importance is None
        else np.asarray(importance, dtype=np.float64)
    )
    return model


class DeterministicViewTests(unittest.TestCase):
    def test_feature_count_below_nominal_two_feature_floor_is_clamped(self) -> None:
        views = _model(1)._generate_views()
        self.assertEqual(views, [(0,)])

    def test_feature_count_equal_to_supplemental_dimension(self) -> None:
        views = _model(2, supplemental_ratio=1.0)._generate_views()
        self.assertEqual(views, [(0, 1)])

    def test_feature_count_above_supplemental_dimension_adds_valid_subsets(self) -> None:
        views = _model(5, supplemental_ratio=0.6)._generate_views()
        self.assertIn((0, 1, 2, 3, 4), views)
        self.assertTrue(any(len(view) == 3 for view in views))
        self.assertTrue(all(tuple(sorted(view)) == view for view in views))
        self.assertEqual(len(views), len(set(views)))

    def test_finite_combination_space_may_be_smaller_than_target(self) -> None:
        views = _model(
            3,
            min_views=20,
            max_views=20,
            supplemental_ratio=2.0 / 3.0,
            supplemental_count=20,
        )._generate_views()
        self.assertEqual(len(views), 4)
        self.assertEqual(
            views,
            [(0, 1, 2), (0, 1), (0, 2), (1, 2)],
        )

    def test_supplemental_views_never_exceed_max_views(self) -> None:
        views = _model(
            8,
            min_views=5,
            max_views=5,
            supplemental_count=12,
        )._generate_views()
        self.assertEqual(len(views), 5)

    def test_low_max_view_caps_are_strict_final_limits(self) -> None:
        for max_views in (25, 50, 75):
            with self.subTest(max_views=max_views):
                views = _model(
                    12,
                    min_views=max_views,
                    max_views=max_views,
                    supplemental_count=100,
                )._generate_views()
                self.assertLessEqual(len(views), max_views)

    def test_spaced_combination_indices_include_both_endpoints(self) -> None:
        self.assertEqual(RelationConsistentMFTAD._unrank_combination(5, 2, 0), (0, 1))
        self.assertEqual(RelationConsistentMFTAD._unrank_combination(5, 2, 9), (3, 4))
        views = _model(
            4,
            min_views=2,
            max_views=2,
            supplemental_ratio=0.5,
            supplemental_count=2,
            include_full_view=False,
            importance=np.ones(4),
        )._generate_views()
        self.assertEqual(views, [(0, 1), (2, 3)])

    def test_equal_importance_uses_original_feature_index_as_tie_breaker(self) -> None:
        views = _model(
            4,
            min_views=2,
            max_views=2,
            supplemental_ratio=0.5,
            supplemental_count=2,
            include_full_view=False,
            importance=np.ones(4),
        )._generate_views()
        self.assertEqual(views, [(0, 1), (2, 3)])

    def test_identical_input_produces_identical_ordered_view_list(self) -> None:
        first = _model(8, min_views=12, max_views=12, supplemental_count=12)
        second = _model(8, min_views=12, max_views=12, supplemental_count=12)
        self.assertEqual(first._generate_views(), second._generate_views())


if __name__ == "__main__":
    unittest.main()
