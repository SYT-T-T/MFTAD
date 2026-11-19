"""Regression tests for paper-facing experiment protocols and defaults."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from experiment_utils import DatasetBundle  # noqa: E402
from run_paper_experiments import (  # noqa: E402
    PAPER_MAIN_ABLATION_NAMES,
    _grid_configurations,
    _resolve_ablation_variants,
    _selection_metadata_from_candidate_counts,
    _write_best_outputs,
    build_parser,
    run_smoke,
)


EXPECTED_PAPER_MAIN = (
    "r0_unified_single_view_core",
    "r1_relation_weighted_multiview",
    "r2_relation_diameter_raw_structure",
    "r3_multiscale_leaf_path_gta",
)


def _bundle(dataset_type: str) -> DatasetBundle:
    nominal_features = (0,) if dataset_type in {"nominal", "mixed"} else ()
    return DatasetBundle(
        name=f"fixture_{dataset_type}",
        path=Path(f"fixture_{dataset_type}.npz"),
        X=np.zeros((4, 2), dtype=np.float64),
        y=np.array([0, 0, 0, 1], dtype=np.int64),
        nominal_features=nominal_features,
        dataset_type=dataset_type,
        label_mapping={0: 0, 1: 1},
    )


class ExperimentProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = build_parser().parse_args(["benchmark"])

    def test_paper_main_resolves_to_adjacent_four_stage_chain(self) -> None:
        self.assertEqual(PAPER_MAIN_ABLATION_NAMES, EXPECTED_PAPER_MAIN)
        resolved = _resolve_ablation_variants("paper_main")
        self.assertEqual(tuple(name for name, _ in resolved), EXPECTED_PAPER_MAIN)

    def test_default_parameters_match_fixed_paper_protocol(self) -> None:
        self.assertEqual(self.args.deltas, "0.8")
        self.assertEqual(self.args.mus, "0.02")
        self.assertAlmostEqual(self.args.fixed_delta, 0.8)
        self.assertAlmostEqual(self.args.fixed_mu, 0.02)
        self.assertAlmostEqual(self.args.pure_nominal_fixed_delta, 0.2)
        self.assertAlmostEqual(self.args.pure_nominal_fixed_mu, 0.005)
        self.assertFalse(hasattr(self.args, "gta_complexity_penalty_eta"))

        sensitivity_args = build_parser().parse_args(["sensitivity"])
        self.assertEqual(
            sensitivity_args.sensitivity_parameters, "delta,mu,max_views"
        )
        self.assertFalse(
            hasattr(sensitivity_args, "sensitivity_gta_complexity_penalty_etas")
        )

    def test_revised_model_defaults_to_fixed_component_mean(self) -> None:
        from run_paper_experiments import REVISED_MODEL_OVERRIDES

        self.assertEqual(
            REVISED_MODEL_OVERRIDES["fusion_mode"], "fixed_component_mean"
        )

    def test_numeric_and_mixed_defaults_use_delta_08_mu_002(self) -> None:
        for dataset_type in ("numeric", "mixed"):
            with self.subTest(dataset_type=dataset_type):
                configurations = _grid_configurations(
                    self.args, _bundle(dataset_type)
                )
                self.assertEqual(len(configurations), 1)
                self.assertAlmostEqual(float(configurations[0]["delta"]), 0.8)
                self.assertAlmostEqual(float(configurations[0]["mu"]), 0.02)
                self.assertAlmostEqual(
                    float(configurations[0]["delta_report"]), 0.8
                )

    def test_pure_nominal_default_uses_mu_0005_and_reports_delta_na(self) -> None:
        configurations = _grid_configurations(self.args, _bundle("nominal"))
        self.assertEqual(len(configurations), 1)
        self.assertAlmostEqual(float(configurations[0]["delta"]), 0.2)
        self.assertAlmostEqual(float(configurations[0]["mu"]), 0.005)
        self.assertEqual(configurations[0]["delta_report"], "N/A")

    def test_smoke_uses_dataset_type_specific_defaults(self) -> None:
        captured: dict[str, dict[str, object]] = {}

        def record_configuration(bundle, _args, config, _variant):
            captured[bundle.dataset_type] = dict(config)
            return {"dataset": bundle.name}, None, np.zeros(len(bundle.y))

        with tempfile.TemporaryDirectory() as folder:
            smoke_args = build_parser().parse_args(
                ["smoke", "--output-dir", folder, "--no-numba"]
            )
            with patch(
                "run_paper_experiments._fit_and_record",
                side_effect=record_configuration,
            ):
                run_smoke(smoke_args)

        self.assertAlmostEqual(float(captured["numeric"]["delta"]), 0.8)
        self.assertAlmostEqual(float(captured["numeric"]["mu"]), 0.02)
        self.assertAlmostEqual(float(captured["mixed"]["delta"]), 0.8)
        self.assertAlmostEqual(float(captured["mixed"]["mu"]), 0.02)
        self.assertAlmostEqual(float(captured["nominal"]["delta"]), 0.2)
        self.assertAlmostEqual(float(captured["nominal"]["mu"]), 0.005)
        self.assertEqual(captured["nominal"]["delta_report"], "N/A")

    def test_candidate_counts_distinguish_fixed_and_label_oracle_protocols(self) -> None:
        fixed = _selection_metadata_from_candidate_counts([1, 1])
        self.assertEqual(fixed["selection_protocol"], "fixed_single_configuration")
        self.assertEqual(fixed["selection_metric"], "fixed")
        self.assertFalse(fixed["oracle_selection"])

        oracle = _selection_metadata_from_candidate_counts([1, 2])
        self.assertEqual(oracle["selection_protocol"], "label_oracle_auc")
        self.assertEqual(oracle["selection_metric"], "auc")
        self.assertTrue(oracle["oracle_selection"])

    def test_fixed_best_outputs_are_not_exported_as_oracle_results(self) -> None:
        rows = [
            {
                "dataset": "fixture_numeric",
                "variant": "relation",
                "config_id": "only-config",
                "auc": 0.75,
                "ap": 0.50,
                "delta": 0.8,
                "mu": 0.02,
                "max_views": 100,
            }
        ]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            best_auc, best_ap = _write_best_outputs(rows, output)
            summary = json.loads(
                (output / "best_summary.json").read_text(encoding="utf-8")
            )

            self.assertFalse(best_auc[0]["oracle_selection"])
            self.assertEqual(best_auc[0]["selection_metric"], "fixed")
            self.assertEqual(best_ap[0]["selection_metric"], "fixed")
            self.assertEqual(
                summary["selection_protocol"], "fixed_single_configuration"
            )
            self.assertFalse(summary["oracle_selection"])
            self.assertIsNone(summary["mean_independent_best_ap_oracle"])


if __name__ == "__main__":
    unittest.main()
