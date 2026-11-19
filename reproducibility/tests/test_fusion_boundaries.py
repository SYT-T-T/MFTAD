"""Boundary tests for deterministic normalization and evidence fusion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from relation_mftad import RelationConsistentMFTAD  # noqa: E402


class FusionBoundaryTests(unittest.TestCase):
    def test_fixed_component_mean_uses_equal_component_weights(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,)]
        model.enabled_components = ("leaf", "gta", "path")
        model.fusion_mode = "fixed_component_mean"
        model.retain_view_evidence = False
        raw = np.asarray(
            [[[0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [2.0, 0.0, 3.0], [3.0, 3.0, 0.0]]],
            dtype=np.float64,
        )

        score = model._fuse_raw_scores(raw)

        np.testing.assert_allclose(
            model.dynamic_weights_,
            np.tile(np.asarray([1.0 / 3.0] * 3), (4, 1)),
        )
        np.testing.assert_allclose(model.component_reliability_, [1.0 / 3.0] * 3)
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertEqual(model.fusion_degenerate_case_, "")

    def test_constant_minmax_input_returns_zero_information_vector(self) -> None:
        actual = RelationConsistentMFTAD._normalize_evidence(
            np.asarray([4.0, 4.0, 4.0])
        )
        np.testing.assert_array_equal(actual, np.zeros(3))

    def test_tied_ranks_use_stable_average_zero_based_ranks(self) -> None:
        actual = RelationConsistentMFTAD._rankdata(
            np.asarray([3.0, 1.0, 1.0, 2.0])
        )
        np.testing.assert_array_equal(actual, np.asarray([3.0, 0.5, 0.5, 2.0]))

    def test_spearman_is_zero_when_either_input_is_constant(self) -> None:
        self.assertEqual(
            RelationConsistentMFTAD._spearman(
                np.asarray([1.0, 1.0, 1.0]),
                np.asarray([1.0, 2.0, 3.0]),
            ),
            0.0,
        )

    def test_spearman_rejects_nonfinite_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            RelationConsistentMFTAD._spearman(
                np.asarray([1.0, np.nan]), np.asarray([1.0, 2.0])
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            RelationConsistentMFTAD._spearman(
                np.asarray([1.0, 2.0]), np.asarray([1.0, np.inf])
            )

    def test_zero_correlations_are_uniform_over_valid_components(self) -> None:
        actual = RelationConsistentMFTAD._normalize_correlation_weights(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([True, False, True]),
        )
        np.testing.assert_array_equal(actual, np.asarray([0.5, 0.0, 0.5]))

    def test_one_valid_component_receives_unit_weight(self) -> None:
        actual = RelationConsistentMFTAD._normalize_correlation_weights(
            np.asarray([0.0, -0.7, 0.4]),
            np.asarray([False, True, False]),
        )
        np.testing.assert_array_equal(actual, np.asarray([0.0, 1.0, 0.0]))

    def test_no_valid_component_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid component"):
            RelationConsistentMFTAD._normalize_correlation_weights(
                np.asarray([0.2, 0.3, 0.4]),
                np.asarray([False, False, False]),
            )

    def test_correlation_weights_use_only_positive_agreement(self) -> None:
        actual = RelationConsistentMFTAD._normalize_correlation_weights(
            np.asarray([0.2, -0.3, 0.0]),
            np.asarray([True, True, True]),
        )
        np.testing.assert_allclose(actual, np.asarray([1.0, 0.0, 0.0]))

    def test_all_nonpositive_correlations_use_uniform_valid_fallback(self) -> None:
        actual = RelationConsistentMFTAD._normalize_correlation_weights(
            np.asarray([-0.2, -0.3, 0.0]),
            np.asarray([True, True, False]),
        )
        np.testing.assert_allclose(actual, np.asarray([0.5, 0.5, 0.0]))

    def test_all_nonpositive_correlations_use_availability_fallback(self) -> None:
        actual = RelationConsistentMFTAD._normalize_correlation_weights(
            np.asarray([-0.2, -0.3, 0.0]),
            np.asarray([True, True, False]),
            np.asarray([0.25, 0.75, 0.0]),
        )
        np.testing.assert_allclose(actual, np.asarray([0.25, 0.75, 0.0]))

    def test_full_fusion_rejects_anti_agreement_as_reliability(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,), (0,), (0,), (0,)]
        model.enabled_components = ("leaf", "gta")
        model.fusion_mode = "structural_stability"
        model.retain_view_evidence = False
        ascending = np.asarray([0.0, 1.0, 2.0, 3.0])
        descending = ascending[::-1]
        raw = np.zeros((4, 4, 3), dtype=np.float64)
        raw[0, :, 0] = ascending
        raw[1, :, 0] = descending
        raw[2, :, 0] = ascending
        raw[3, :, 0] = descending
        raw[:, :, 1] = ascending

        model._fuse_raw_scores(raw)

        np.testing.assert_allclose(
            model.component_stability_[:2], [-1.0, 1.0]
        )
        self.assertTrue(np.isnan(model.component_stability_[2]))
        np.testing.assert_array_equal(
            model.component_stability_estimable_, [True, True, False]
        )
        self.assertEqual(
            model.component_stability_status_,
            (
                "estimated_negative",
                "estimated_positive",
                "stability_unestimable",
            ),
        )
        np.testing.assert_allclose(model.component_reliability_, [0.0, 1.0, 0.0])

    def test_full_fusion_records_nonpositive_availability_fallback(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,), (0,), (0,), (0,)]
        model.enabled_components = ("leaf", "gta")
        model.fusion_mode = "structural_stability"
        model.retain_view_evidence = False
        ascending = np.asarray([0.0, 1.0, 2.0, 3.0])
        descending = ascending[::-1]
        raw = np.zeros((4, 4, 3), dtype=np.float64)
        raw[0, :, 0] = ascending
        raw[1, :, 0] = descending
        raw[2, :, 0] = 0.5
        raw[3, :, 0] = 0.5
        raw[0, :, 1] = ascending
        raw[1, :, 1] = descending
        raw[2, :, 1] = ascending
        raw[3, :, 1] = descending

        model._fuse_raw_scores(raw)

        np.testing.assert_allclose(model.component_stability_[:2], [-1.0, -1.0])
        np.testing.assert_allclose(
            model.component_availability_, [0.5, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            model.component_reliability_, [1.0 / 3.0, 2.0 / 3.0, 0.0]
        )
        self.assertEqual(
            model.fusion_degenerate_case_,
            "stability_no_positive_agreement_availability_fallback",
        )

    def test_single_available_panel_is_marked_unestimable(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,)]
        model.enabled_components = ("leaf",)
        model.fusion_mode = "structural_stability"
        model.retain_view_evidence = False
        raw = np.zeros((1, 4, 3), dtype=np.float64)
        raw[0, :, 0] = np.asarray([0.0, 1.0, 2.0, 3.0])

        model._fuse_raw_scores(raw)

        self.assertTrue(np.isnan(model.component_stability_[0]))
        self.assertFalse(model.component_stability_estimable_[0])
        self.assertEqual(
            model.component_stability_status_[0], "stability_unestimable"
        )
        np.testing.assert_allclose(model.component_reliability_, [1.0, 0.0, 0.0])
        self.assertEqual(
            model.fusion_degenerate_case_,
            "stability_no_positive_agreement_availability_fallback",
        )

    def test_estimable_zero_correlation_is_not_marked_unestimable(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,), (0,)]
        model.enabled_components = ("leaf",)
        model.fusion_mode = "structural_stability"
        model.retain_view_evidence = False
        raw = np.zeros((2, 4, 3), dtype=np.float64)
        raw[0, :, 0] = np.asarray([0.0, 1.0, 2.0, 3.0])
        raw[1, :, 0] = np.asarray([1.0, 3.0, 0.0, 2.0])

        model._fuse_raw_scores(raw)

        self.assertAlmostEqual(model.component_stability_[0], 0.0)
        self.assertTrue(model.component_stability_estimable_[0])
        self.assertEqual(
            model.component_stability_status_[0], "estimated_zero"
        )

    def test_current_certainty_keeps_uniform_valid_component_reliability(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,), (0,)]
        model.enabled_components = ("leaf", "gta")
        model.fusion_mode = "current_certainty"
        model.retain_view_evidence = False
        raw = np.zeros((2, 4, 3), dtype=np.float64)
        raw[0, :, 0] = np.asarray([0.0, 1.0, 2.0, 3.0])
        raw[0, :, 1] = np.asarray([0.0, 1.0, 2.0, 3.0])
        raw[1, :, 1] = np.asarray([3.0, 2.0, 1.0, 0.0])

        model._fuse_raw_scores(raw)

        np.testing.assert_allclose(
            model.component_availability_, [0.5, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            model.component_reliability_, [0.5, 0.5, 0.0]
        )
        self.assertEqual(model.fusion_degenerate_case_, "")

    def test_full_fusion_is_finite_normalized_and_row_stochastic(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,), (0,), (0,), (0,)]
        model.enabled_components = ("leaf", "gta", "path")
        model.fusion_mode = "structural_stability"
        model.retain_view_evidence = False
        raw = np.asarray(
            [
                [[0.1, 0.8, 0.2], [0.2, 0.6, 0.4], [0.7, 0.3, 0.6], [0.9, 0.1, 0.8]],
                [[0.2, 0.7, 0.1], [0.3, 0.5, 0.3], [0.6, 0.4, 0.7], [0.8, 0.2, 0.9]],
                [[0.1, 0.9, 0.3], [0.4, 0.7, 0.5], [0.5, 0.2, 0.6], [0.7, 0.1, 0.8]],
                [[0.3, 0.8, 0.2], [0.4, 0.6, 0.4], [0.6, 0.3, 0.7], [0.9, 0.2, 0.9]],
            ],
            dtype=np.float64,
        )
        score = model._fuse_raw_scores(raw)
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertGreaterEqual(float(score.min()), 0.0)
        self.assertLessEqual(float(score.max()), 1.0)
        np.testing.assert_allclose(model.dynamic_weights_.sum(axis=1), 1.0)

    def test_full_fusion_returns_zero_information_when_enabled_evidence_is_constant(
        self,
    ) -> None:
        """Catch regressions that abort a valid single-component ablation."""

        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((4, 1), dtype=np.float64)
        model.views_ = [(0,)]
        model.enabled_components = ("leaf",)
        model.fusion_mode = "current_certainty"
        model.retain_view_evidence = False
        raw = np.full((1, 4, 3), 0.25, dtype=np.float64)

        score = model._fuse_raw_scores(raw)

        np.testing.assert_array_equal(score, np.zeros(4))
        np.testing.assert_array_equal(
            model.dynamic_weights_,
            np.tile(np.asarray([1.0, 0.0, 0.0]), (4, 1)),
        )
        np.testing.assert_array_equal(
            model.component_reliability_, np.asarray([1.0, 0.0, 0.0])
        )
        self.assertEqual(model.active_components_, ())
        self.assertEqual(model.fusion_degenerate_case_, "all_enabled_components_constant")

    def test_full_fusion_rejects_nonfinite_raw_input(self) -> None:
        model = RelationConsistentMFTAD.__new__(RelationConsistentMFTAD)
        model.X_ = np.zeros((2, 1), dtype=np.float64)
        model.views_ = [(0,), (0,)]
        model.enabled_components = ("leaf", "gta", "path")
        model.fusion_mode = "structural_stability"
        model.retain_view_evidence = False
        raw = np.ones((2, 2, 3), dtype=np.float64)
        raw[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            model._fuse_raw_scores(raw)


if __name__ == "__main__":
    unittest.main()
