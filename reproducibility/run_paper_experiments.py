"""Complete experiment runner for the standalone relation MFTAD package."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import threading
import time
import types
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime diagnostic
    psutil = None

try:  # Support both ``python file.py`` and package imports.
    from .experiment_utils import (
        DatasetBundle,
        complementarity_record,
        compute_binary_metrics,
        dataset_type,
        discover_datasets,
        latex_wide_table,
        load_dataset,
        load_nominal_map,
        parse_number_grid,
        plot_critical_difference,
        plot_sensitivity,
        read_rows,
        save_json,
        select_best_per_dataset,
        statistical_comparison,
        system_manifest,
        write_rows,
    )
    from .paper_conventions import (
        PAPER_DATASET_COUNT,
        PAPER_DATASET_ORDER,
        paper_dataset_abbreviation,
    )
    from .relation_mftad import COMPONENTS, RelationConsistentMFTAD
except ImportError:  # pragma: no cover - script execution path
    from experiment_utils import (
        DatasetBundle,
        complementarity_record,
        compute_binary_metrics,
        dataset_type,
        discover_datasets,
        latex_wide_table,
        load_dataset,
        load_nominal_map,
        parse_number_grid,
        plot_critical_difference,
        plot_sensitivity,
        read_rows,
        save_json,
        select_best_per_dataset,
        statistical_comparison,
        system_manifest,
        write_rows,
    )
    from paper_conventions import (
        PAPER_DATASET_COUNT,
        PAPER_DATASET_ORDER,
        paper_dataset_abbreviation,
    )
    from relation_mftad import COMPONENTS, RelationConsistentMFTAD


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets1"
DEFAULT_NOMINAL_MAP = (
    SCRIPT_DIR.parent
    / "datasets1_schema_results"
    / "nominal_feature_map_datasets1.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_DELTAS = "0.8"
DEFAULT_MUS = "0.02"
DEFAULT_SENSITIVITY_PARAMETERS = "delta,mu,max_views"
DEFAULT_SENSITIVITY_DELTAS = "0.2,0.4,0.6,0.8,1.0,1.2,1.4"
DEFAULT_SENSITIVITY_MUS = "0.005,0.01,0.02,0.04,0.08"
DEFAULT_SENSITIVITY_VIEWS = "50,75,100,125,150"
DEFAULT_SENSITIVITY_AUC_MIN_SPAN = 0.15
DEFAULT_SENSITIVITY_AP_MIN_SPAN = 0.20
DEFAULT_SENSITIVITY_AXIS_PADDING = 0.03
DATASET_TYPES = ("numeric", "nominal", "mixed")
REPOSITORY_DETECTOR_METHODS = (
    "DCROD",
    "DFNO",
    "DIF",
    "GBFRD",
    "KFGOD",
    "MFGAD",
    "MGBOD",
    "ODGrCR",
)
SUPPORTED_DETECTOR_METHODS = ("relation", "IForest", *REPOSITORY_DETECTOR_METHODS)
DETECTOR_METHOD_ALIASES = {
    "relation": "relation",
    "mftad": "relation",
    "relationmftad": "relation",
    "iforest": "IForest",
    "isolationforest": "IForest",
    **{method.casefold(): method for method in REPOSITORY_DETECTOR_METHODS},
}


def _parse_csv_names(text: Optional[str]) -> Optional[List[str]]:
    if text is None or not str(text).strip() or str(text).strip().lower() == "all":
        return None
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _resolve_detector_methods(
    text: Optional[str], default: Sequence[str]
) -> List[str]:
    raw_tokens = [
        item.strip()
        for item in str(text or "").split(",")
        if item.strip()
    ]
    if not raw_tokens:
        raw_tokens = list(default)
    methods: List[str] = []
    for token in raw_tokens:
        normalized = token.casefold()
        if normalized == "all":
            expanded = list(SUPPORTED_DETECTOR_METHODS)
        elif normalized in {"repository", "repo"}:
            expanded = list(REPOSITORY_DETECTOR_METHODS)
        else:
            canonical = DETECTOR_METHOD_ALIASES.get(normalized)
            if canonical is None:
                raise ValueError(
                    f"Unknown detector method {token!r}. Supported methods: "
                    f"{', '.join(SUPPORTED_DETECTOR_METHODS)}; aliases: all, repository."
                )
            expanded = [canonical]
        for method in expanded:
            if method not in methods:
                methods.append(method)
    return methods


def _config_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _args_payload(args: argparse.Namespace) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def _limit_quick_bundles(
    bundles: Sequence[DatasetBundle], limit: int
) -> List[DatasetBundle]:
    if limit <= 0 or len(bundles) <= limit:
        return list(bundles)
    selected: List[DatasetBundle] = []
    selected_keys: set[Tuple[str, str]] = set()
    for kind in DATASET_TYPES:
        match = next((item for item in bundles if item.dataset_type == kind), None)
        key = (match.name, str(match.path)) if match is not None else None
        if match is not None and key not in selected_keys:
            selected.append(match)
            selected_keys.add(key)
    for bundle in bundles:
        key = (bundle.name, str(bundle.path))
        if key not in selected_keys:
            selected.append(bundle)
            selected_keys.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _load_bundles(args: argparse.Namespace) -> List[DatasetBundle]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nominal_map = load_nominal_map(args.nominal_map)
    requested = None if "all" in args.datasets else list(args.datasets)
    paths = discover_datasets(args.dataset_root, requested)
    allowed_types = set(DATASET_TYPES if "all" in args.dataset_types else args.dataset_types)
    bundles: List[DatasetBundle] = []
    failures: List[Dict[str, object]] = []
    schema_rows: List[Dict[str, object]] = []
    mapped_names = {str(name).casefold() for name in nominal_map}
    for path in paths:
        try:
            schema_present = path.stem.casefold() in mapped_names
            if not schema_present and bool(getattr(args, "require_schema_coverage", False)):
                raise ValueError(
                    f"Dataset {path.stem!r} is missing from nominal map {args.nominal_map}."
                )
            bundle = load_dataset(path, nominal_map)
            schema_rows.append(
                {
                    "dataset": bundle.name,
                    "source": str(path),
                    "schema_present": schema_present,
                    "dataset_type": bundle.dataset_type,
                    "n_features": int(bundle.X.shape[1]),
                    "nominal_feature_count": len(bundle.nominal_features),
                    "nominal_features": "|".join(
                        str(index) for index in bundle.nominal_features
                    ),
                }
            )
            if bundle.dataset_type not in allowed_types:
                continue
            if args.max_samples and len(bundle.y) > args.max_samples:
                indices = np.linspace(
                    0, len(bundle.y) - 1, int(args.max_samples), dtype=np.int64
                )
                bundle = DatasetBundle(
                    name=bundle.name,
                    path=bundle.path,
                    X=np.ascontiguousarray(bundle.X[indices]),
                    y=np.ascontiguousarray(bundle.y[indices]),
                    nominal_features=bundle.nominal_features,
                    dataset_type=bundle.dataset_type,
                    label_mapping=bundle.label_mapping,
                )
            bundles.append(bundle)
        except Exception as error:
            failures.append({"path": str(path), "error": repr(error)})
            if args.fail_on_error:
                write_rows(
                    failures,
                    output / "dataset_load_failures.csv",
                    fieldnames=("path", "error"),
                )
                raise
    write_rows(
        failures,
        output / "dataset_load_failures.csv",
        fieldnames=("path", "error"),
    )
    write_rows(
        schema_rows,
        output / "feature_type_report.csv",
        fieldnames=(
            "dataset",
            "source",
            "schema_present",
            "dataset_type",
            "n_features",
            "nominal_feature_count",
            "nominal_features",
        ),
    )
    bundles.sort(key=lambda item: (item.name.lower(), str(item.path).lower()))
    if args.quick:
        bundles = _limit_quick_bundles(bundles, int(args.quick_dataset_limit))
    if not bundles:
        raise ValueError(
            f"No datasets were loaded from {Path(args.dataset_root).resolve()}."
        )
    return bundles


def _base_model_kwargs(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    config: Optional[Mapping[str, object]] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    config = config or {}
    fixed_delta = float(args.fixed_delta)
    fixed_mu = float(args.fixed_mu)
    if bundle.dataset_type == "nominal":
        pure_nominal_delta = getattr(args, "pure_nominal_fixed_delta", None)
        pure_nominal_mu = getattr(args, "pure_nominal_fixed_mu", None)
        if pure_nominal_delta is not None:
            fixed_delta = float(pure_nominal_delta)
        if pure_nominal_mu is not None:
            fixed_mu = float(pure_nominal_mu)
    max_views = int(config.get("max_views", args.max_views))
    kwargs: Dict[str, object] = {
        "delta": float(config.get("delta", fixed_delta)),
        "cohesion_threshold": float(config.get("mu", fixed_mu)),
        "min_views": min(int(args.min_views), max_views),
        "max_views": max_views,
        "supplemental_view_count": int(args.supplemental_view_count),
        "min_view_dimension_ratio": float(args.min_view_dimension_ratio),
        "supplemental_view_dimension_ratio": float(
            args.supplemental_view_dimension_ratio
        ),
        "min_child_size": int(args.min_child_size),
        "strict_cohesion_gain": bool(args.strict_cohesion_gain),
        "fusion_mode": "fixed_component_mean",
        "importance_mode": "fuzzy_entropy",
        "nominal_relation_mode": "equality",
        "pivot_mode": "relation_diameter_root",
        "gain_mode": "raw",
        "gta_mode": "frontier",
        "enabled_components": COMPONENTS,
        "include_full_view": True,
        "nominal_features": bundle.nominal_features,
        "n_jobs": int(args.n_jobs),
        "memory_safety_ratio": float(args.memory_safety_ratio),
        "attribute_relation_cache_mb": float(args.attribute_relation_cache_mb),
        "use_numba": not bool(args.no_numba),
        "numba_threads": int(args.numba_threads),
        "retain_view_evidence": bool(args.retain_view_evidence),
        "verbose": bool(args.verbose),
    }
    if overrides:
        kwargs.update(overrides)
    kwargs["max_views"] = int(kwargs["max_views"])
    kwargs["min_views"] = min(int(kwargs["min_views"]), int(kwargs["max_views"]))
    return kwargs


def _grid_configurations(
    args: argparse.Namespace, bundle: DatasetBundle
) -> List[Dict[str, object]]:
    if bundle.dataset_type == "nominal":
        pure_nominal_delta = getattr(args, "pure_nominal_fixed_delta", None)
        deltas = [
            float(args.fixed_delta)
            if pure_nominal_delta is None
            else float(pure_nominal_delta)
        ]
    else:
        deltas = [float(item) for item in parse_number_grid(args.deltas)]
    pure_nominal_fixed_mu = getattr(args, "pure_nominal_fixed_mu", None)
    if bundle.dataset_type == "nominal" and pure_nominal_fixed_mu is not None:
        mus = [float(pure_nominal_fixed_mu)]
    else:
        mus = [float(item) for item in parse_number_grid(args.mus)]
    view_counts = [int(item) for item in parse_number_grid(args.view_grid, int)]
    if args.quick:
        deltas = deltas[:2]
        mus = mus[:2]
        view_counts = view_counts[:1]
    configurations: List[Dict[str, object]] = []
    for delta, mu, max_views in itertools.product(deltas, mus, view_counts):
        payload: Dict[str, object] = {
            "delta": float(delta),
            "mu": float(mu),
            "max_views": int(max_views),
            "delta_report": "N/A" if bundle.dataset_type == "nominal" else float(delta),
        }
        payload["grid_id"] = _config_id(payload)
        configurations.append(payload)
    return configurations


def _selected_config_lookup(path: Optional[str]) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    rows = read_rows(path)
    lookup: Dict[str, Dict[str, object]] = {}
    for row in rows:
        name = str(row.get("dataset", "")).strip()
        if not name:
            continue
        lookup[name.lower()] = {
            "delta": float(row.get("delta", 0.2)),
            "mu": float(row.get("mu", row.get("cohesion_threshold", 0.1))),
            "max_views": int(float(row.get("max_views", 100))),
        }
    return lookup


def _configuration_for_bundle(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    lookup: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    selected = dict(lookup.get(bundle.name.lower(), {}))
    pure_nominal_fixed_delta = getattr(args, "pure_nominal_fixed_delta", None)
    pure_nominal_fixed_mu = getattr(args, "pure_nominal_fixed_mu", None)
    fixed_delta = (
        pure_nominal_fixed_delta
        if bundle.dataset_type == "nominal"
        and pure_nominal_fixed_delta is not None
        else args.fixed_delta
    )
    fixed_mu = (
        pure_nominal_fixed_mu
        if bundle.dataset_type == "nominal" and pure_nominal_fixed_mu is not None
        else args.fixed_mu
    )
    delta = float(selected.get("delta", fixed_delta))
    return {
        "delta": delta,
        "mu": float(selected.get("mu", fixed_mu)),
        "max_views": int(selected.get("max_views", args.max_views)),
        "delta_report": "N/A" if bundle.dataset_type == "nominal" else delta,
    }


def _component_metrics(
    bundle: DatasetBundle, model: RelationConsistentMFTAD
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    if model.fused_scores_3d_ is None:
        return result
    for index, name in enumerate(COMPONENTS):
        metrics = compute_binary_metrics(bundle.y, model.fused_scores_3d_[:, index])
        result[f"{name}_auc"] = metrics["auc"]
        result[f"{name}_ap"] = metrics["ap"]
        if model.learned_score_weights_ is not None:
            result[f"{name}_mean_weight"] = float(
                model.learned_score_weights_[index]
            )
        if model.component_availability_ is not None:
            result[f"{name}_availability"] = float(
                model.component_availability_[index]
            )
        if model.component_stability_ is not None:
            estimable = bool(
                model.component_stability_estimable_ is not None
                and model.component_stability_estimable_[index]
            )
            result[f"{name}_stability"] = (
                float(model.component_stability_[index]) if estimable else None
            )
            result[f"{name}_stability_status"] = (
                model.component_stability_status_[index]
                if model.component_stability_status_ is not None
                else "stability_unestimable"
            )
        if (
            model.component_reliability_ is not None
            and model.fusion_mode != "fixed_component_mean"
        ):
            result[f"{name}_reliability"] = float(
                model.component_reliability_[index]
            )
        if model.component_weights_ is not None:
            result[f"{name}_component_weight"] = float(
                model.component_weights_[index]
            )
    return result


def _fit_and_record(
    bundle: DatasetBundle,
    args: argparse.Namespace,
    config: Mapping[str, object],
    variant: str,
    overrides: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[str, object], RelationConsistentMFTAD, np.ndarray]:
    kwargs = _base_model_kwargs(args, bundle, config, overrides)
    result_id = _config_id(
        {
            "dataset": bundle.name,
            "data_digest": _array_digest(bundle.X),
            "labels_digest": _array_digest(bundle.y),
            "variant": variant,
            "model": kwargs,
        }
    )
    start = time.perf_counter()
    model = RelationConsistentMFTAD(**kwargs)
    scores = model.fit_predict(bundle.X)
    elapsed = time.perf_counter() - start
    views_used = len(model.views_)
    configured_max_views = int(kwargs["max_views"])
    if views_used > configured_max_views:
        raise RuntimeError(
            f"Dataset {bundle.name!r} generated {views_used} views, exceeding "
            f"max_views={configured_max_views}."
        )
    metrics = compute_binary_metrics(bundle.y, scores)
    row: Dict[str, object] = {
        "dataset": bundle.name,
        "source": str(bundle.path),
        "dataset_type": bundle.dataset_type,
        "n_samples": int(bundle.X.shape[0]),
        "n_features": int(bundle.X.shape[1]),
        "n_anomalies": int(bundle.y.sum()),
        "variant": variant,
        "config_id": result_id,
        "grid_id": config.get("grid_id", "selected"),
        "delta": float(kwargs["delta"]),
        "delta_report": (
            float(kwargs["delta"])
            if bundle.dataset_type == "nominal"
            and model.nominal_relation_mode == "numeric_code"
            else config.get("delta_report", kwargs["delta"])
        ),
        "mu": float(kwargs["cohesion_threshold"]),
        "min_views": int(kwargs["min_views"]),
        "max_views": int(kwargs["max_views"]),
        "views_used": views_used,
        "enabled_components": ",".join(model.enabled_components),
        "importance_mode": model.importance_mode,
        "fusion_mode": model.fusion_mode,
        "fusion_degenerate_case": model.fusion_degenerate_case_,
        "nominal_relation_mode": model.nominal_relation_mode,
        "pivot_mode": model.pivot_mode,
        "gain_mode": model.gain_mode,
        "gta_mode": model.gta_mode,
        "auc": metrics["auc"],
        "ap": metrics["ap"],
        "elapsed_seconds": elapsed,
        "used_strategy": model.used_strategy_,
    }
    row.update(_component_metrics(bundle, model))
    row.update({f"runtime_{key}": value for key, value in model.runtime_breakdown_.items()})
    row.update({f"tree_{key}": value for key, value in model.tree_stats_.items()})
    row.update(
        {f"relation_cache_{key}": value for key, value in model.relation_cache_stats_.items()}
    )
    return row, model, scores


def _save_score_archive(
    output: Path,
    bundle: DatasetBundle,
    row: Mapping[str, object],
    model: RelationConsistentMFTAD,
    scores: np.ndarray,
) -> None:
    folder = output / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    component = (
        model.fused_scores_3d_
        if model.fused_scores_3d_ is not None
        else np.empty((len(scores), 0), dtype=np.float64)
    )
    np.savez_compressed(
        folder / f"{bundle.name}__{row['variant']}__{row['config_id']}.npz",
        y=bundle.y,
        score=scores,
        component_scores=component,
        dynamic_weights=(
            model.dynamic_weights_
            if model.dynamic_weights_ is not None
            else np.empty((len(scores), 0), dtype=np.float64)
        ),
    )


def _run_grid(
    args: argparse.Namespace,
    variants: Sequence[Tuple[str, Mapping[str, object]]],
    experiment_name: str,
    variant_mu_grids: Optional[Mapping[str, str]] = None,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(args)
    raw_path = output / "raw_results.csv"
    rows: List[Dict[str, object]] = (
        [dict(row) for row in read_rows(raw_path)]
        if args.resume and raw_path.is_file()
        else []
    )
    completed = {
        (str(row.get("dataset")), str(row.get("variant")), str(row.get("config_id")))
        for row in rows
    }
    failures: List[Dict[str, object]] = []

    def configurations_for(
        bundle: DatasetBundle, variant: str
    ) -> List[Dict[str, object]]:
        if not variant_mu_grids or variant not in variant_mu_grids:
            return _grid_configurations(args, bundle)
        variant_args = copy.copy(args)
        variant_args.mus = str(variant_mu_grids[variant])
        return _grid_configurations(variant_args, bundle)

    candidate_counts = [
        len(configurations_for(bundle, variant))
        for bundle in bundles
        for variant, _ in variants
    ]
    selection_metadata = _selection_metadata_from_candidate_counts(candidate_counts)
    total_candidates = sum(candidate_counts)
    candidate_index = 0
    for bundle in bundles:
        if variant_mu_grids:
            candidates = [
                (config, variant, overrides)
                for variant, overrides in variants
                for config in configurations_for(bundle, variant)
            ]
        else:
            candidates = [
                (config, variant, overrides)
                for config in _grid_configurations(args, bundle)
                for variant, overrides in variants
            ]
        for config, variant, overrides in candidates:
            candidate_index += 1
            kwargs = _base_model_kwargs(args, bundle, config, overrides)
            result_id = _config_id(
                {
                    "dataset": bundle.name,
                    "data_digest": _array_digest(bundle.X),
                    "labels_digest": _array_digest(bundle.y),
                    "variant": variant,
                    "model": kwargs,
                }
            )
            key = (bundle.name, variant, result_id)
            if key in completed:
                print(
                    f"[{candidate_index}/{total_candidates}] resume {bundle.name} {variant}",
                    flush=True,
                )
                continue
            try:
                row, model, scores = _fit_and_record(
                    bundle, args, config, variant, overrides
                )
                rows.append(row)
                completed.add(key)
                if args.save_scores:
                    _save_score_archive(output, bundle, row, model, scores)
                print(
                    f"[{candidate_index}/{total_candidates}] {bundle.name} {variant} "
                    f"delta={row['delta_report']} mu={row['mu']} "
                    f"AUC={float(row['auc']):.6f} AP={float(row['ap']):.6f} "
                    f"time={float(row['elapsed_seconds']):.3f}s",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "dataset": bundle.name,
                    "source": str(bundle.path),
                    "variant": variant,
                    "delta": config.get("delta"),
                    "mu": config.get("mu"),
                    "max_views": config.get("max_views"),
                    "error": repr(error),
                }
                failures.append(failure)
                print(
                    f"[{candidate_index}/{total_candidates}] FAILED {bundle.name} "
                    f"{variant}: {error!r}",
                    flush=True,
                )
                if args.fail_on_error:
                    write_rows(rows, raw_path)
                    write_rows(
                        failures,
                        output / "failures.csv",
                        fieldnames=(
                            "dataset",
                            "source",
                            "variant",
                            "delta",
                            "mu",
                            "max_views",
                            "error",
                        ),
                    )
                    raise
            write_rows(rows, raw_path)
            write_rows(
                failures,
                output / "failures.csv",
                fieldnames=(
                    "dataset",
                    "source",
                    "variant",
                    "delta",
                    "mu",
                    "max_views",
                    "error",
                ),
            )
    save_json(
        system_manifest(
            {
                "experiment": experiment_name,
                "arguments": _args_payload(args),
                "variants": [name for name, _ in variants],
                "variant_mu_grids": (
                    dict(variant_mu_grids) if variant_mu_grids else None
                ),
                **selection_metadata,
            }
        ),
        output / "run_config.json",
    )
    return rows, failures, selection_metadata


def _as_numeric_rows(
    rows: Sequence[Mapping[str, object]]
) -> List[Dict[str, object]]:
    converted: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        try:
            item["auc"] = float(item["auc"])
            item["ap"] = float(item["ap"])
            item["delta"] = float(item["delta"])
            item["mu"] = float(item["mu"])
            item["max_views"] = int(float(item["max_views"]))
        except (KeyError, TypeError, ValueError):
            continue
        converted.append(item)
    return converted


def _selection_metadata_from_candidate_counts(
    candidate_counts: Sequence[int],
) -> Dict[str, object]:
    counts = [int(value) for value in candidate_counts]
    if not counts or any(value < 1 for value in counts):
        raise ValueError("Every dataset/variant must have at least one configuration.")
    oracle_selection = any(value > 1 for value in counts)
    return {
        "selection_protocol": (
            "label_oracle_auc" if oracle_selection else "fixed_single_configuration"
        ),
        "selection_metric": "auc" if oracle_selection else "fixed",
        "oracle_selection": oracle_selection,
        "candidate_configurations_min": min(counts),
        "candidate_configurations_max": max(counts),
        "ap_reporting_rule": (
            "Report AP from the AUC-selected row; selection uses labels."
            if oracle_selection
            else "Report AUC and AP from the single fixed configuration."
        ),
    }


def _selection_metadata_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str], set[str]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("variant", "relation")))
        config_id = str(row.get("config_id", row.get("grid_id", "")))
        if not key[0] or not config_id:
            raise ValueError("Cannot infer selection protocol from incomplete rows.")
        grouped.setdefault(key, set()).add(config_id)
    return _selection_metadata_from_candidate_counts(
        [len(configurations) for configurations in grouped.values()]
    )


def _write_best_outputs(
    rows: Sequence[Mapping[str, object]],
    output: Path,
    selection_metadata: Optional[Mapping[str, object]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    numeric = _as_numeric_rows(rows)
    metadata = dict(
        selection_metadata or _selection_metadata_from_rows(numeric)
    )
    oracle_selection = bool(metadata["oracle_selection"])
    best_auc = select_best_per_dataset(numeric, "auc")
    best_ap_oracle = select_best_per_dataset(numeric, "ap")
    for row in best_auc:
        row["selection_metric"] = metadata["selection_metric"]
        row["selection_protocol"] = metadata["selection_protocol"]
        row["oracle_selection"] = oracle_selection
    for row in best_ap_oracle:
        row["selection_metric"] = "ap" if oracle_selection else "fixed"
        row["selection_protocol"] = metadata["selection_protocol"]
        row["oracle_selection"] = oracle_selection
    write_rows(best_auc, output / "best_auc.csv")
    write_rows(best_auc, output / "auc_selected_metrics.csv")
    write_rows(best_auc, output / "best_ap.csv")
    write_rows(best_ap_oracle, output / "best_ap_oracle.csv")
    save_json(
        {
            "dataset_count": len(best_auc),
            "mean_auc": (
                float(np.mean([float(row["auc"]) for row in best_auc]))
                if best_auc
                else float("nan")
            ),
            "mean_ap_at_auc_selected_config": (
                float(np.mean([float(row["ap"]) for row in best_auc]))
                if best_auc
                else float("nan")
            ),
            "mean_independent_best_ap_oracle": (
                float(np.mean([float(row["ap"]) for row in best_ap_oracle]))
                if best_ap_oracle and oracle_selection
                else None
            ),
            **metadata,
            "selection_note": (
                "Primary AUC and AP use the single fixed configuration; "
                "best_ap_oracle.csv is an identical compatibility export, not an oracle."
                if not oracle_selection
                else "Primary AUC and AP reporting uses the same AUC-selected row. "
                "best_ap_oracle.csv is diagnostic label-based oracle evidence only."
            ),
        },
        output / "best_summary.json",
    )
    return best_auc, best_ap_oracle


def _write_cohesion_branch_audit(
    selected_rows: Sequence[Mapping[str, object]], output: Path
) -> Dict[str, object]:
    """Record whether singleton/empty cohesion branches execute on every dataset."""

    audit_rows: List[Dict[str, object]] = []
    naming_errors: List[str] = []
    seen: set[str] = set()
    for source_row in selected_rows:
        raw_name = str(source_row.get("dataset", "")).strip()
        try:
            abbreviation = paper_dataset_abbreviation(raw_name)
        except ValueError as error:
            naming_errors.append(str(error))
            continue
        if abbreviation in seen:
            naming_errors.append(
                f"Duplicate selected row for Table 2 abbreviation {abbreviation!r}."
            )
            continue
        seen.add(abbreviation)
        singleton_calls = int(
            float(source_row.get("tree_singleton_cohesion_calls", 0))
        )
        views_triggered = int(
            float(source_row.get("tree_views_with_singleton_cohesion", 0))
        )
        empty_rejections = int(
            float(source_row.get("tree_empty_cohesion_rejections", 0))
        )
        audit_rows.append(
            {
                "dataset": abbreviation,
                "singleton_branch_triggered": singleton_calls > 0,
                "singleton_cohesion_calls": singleton_calls,
                "views_with_singleton_cohesion": views_triggered,
                "empty_cohesion_rejections": empty_rejections,
            }
        )
    audit_rows.sort(key=lambda row: PAPER_DATASET_ORDER[str(row["dataset"])])
    write_rows(
        audit_rows,
        output / "cohesion_branch_audit.csv",
        fieldnames=(
            "dataset",
            "singleton_branch_triggered",
            "singleton_cohesion_calls",
            "views_with_singleton_cohesion",
            "empty_cohesion_rejections",
        ),
    )
    empty_total = sum(int(row["empty_cohesion_rejections"]) for row in audit_rows)
    status = (
        "PASS"
        if len(audit_rows) == PAPER_DATASET_COUNT
        and not naming_errors
        and empty_total == 0
        else "BLOCKED"
    )
    payload = {
        "status": status,
        "expected_dataset_count": PAPER_DATASET_COUNT,
        "observed_dataset_count": len(audit_rows),
        "datasets_triggering_singleton_branch": [
            row["dataset"]
            for row in audit_rows
            if row["singleton_branch_triggered"]
        ],
        "singleton_trigger_dataset_count": sum(
            bool(row["singleton_branch_triggered"]) for row in audit_rows
        ),
        "total_singleton_cohesion_calls": sum(
            int(row["singleton_cohesion_calls"]) for row in audit_rows
        ),
        "total_empty_cohesion_rejections": empty_total,
        "naming_errors": naming_errors,
        "interpretation": (
            "Singleton cohesion was already one before the empty-input guard was "
            "added. A zero empty-rejection count proves that successful tree builds "
            "did not depend on the newly rejected illegal case; other corrected "
            "mechanisms are assessed separately."
        ),
    }
    save_json(payload, output / "cohesion_branch_audit.json")
    return payload


def _merge_baseline(
    baseline_path: Optional[str],
    selected_rows: Sequence[Mapping[str, object]],
    metric: str,
    output: Path,
) -> None:
    if not baseline_path:
        return
    with Path(baseline_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        baseline_rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError(f"Baseline table has no header: {baseline_path}")
    dataset_field = next(
        (name for name in fields if name.lower() in {"dataset", "data", "name"}),
        fields[0],
    )
    selected = {str(row["dataset"]).lower(): row for row in selected_rows}
    aligned: List[Dict[str, object]] = []
    missing_selected: List[str] = []
    for row in baseline_rows:
        dataset = str(row.get(dataset_field, "")).strip()
        match = selected.get(dataset.lower())
        if match is None:
            missing_selected.append(dataset)
            continue
        if any(
            str(row.get(field, "")).strip() == ""
            for field in fields
            if field != dataset_field
        ):
            continue
        merged = dict(row)
        merged["relation"] = float(match[metric])
        aligned.append(merged)
    write_rows(aligned, output)
    save_json(
        {
            "baseline_rows": len(baseline_rows),
            "selected_rows": len(selected_rows),
            "aligned_complete_rows": len(aligned),
            "baseline_without_relation": missing_selected,
            "rule": "Only complete aligned rows are exported.",
        },
        output.with_name(f"{output.stem}_alignment.json"),
    )


def run_benchmark(args: argparse.Namespace) -> None:
    rows, failures, selection_metadata = _run_grid(
        args, [("relation", {})], experiment_name="benchmark"
    )
    output = Path(args.output_dir)
    best_auc, _ = _write_best_outputs(rows, output, selection_metadata)
    _write_cohesion_branch_audit(best_auc, output)
    _merge_baseline(
        args.baseline_auc, best_auc, "auc", output / "comparison_auc.csv"
    )
    _merge_baseline(
        args.baseline_ap, best_auc, "ap", output / "comparison_ap.csv"
    )
    print(
        f"[benchmark complete] rows={len(rows)} failures={len(failures)} output={output}",
        flush=True,
    )


def run_repair_benchmark(args: argparse.Namespace) -> None:
    """Replace invalid dataset grids and regenerate every derived benchmark file."""

    original = Path(args.benchmark_dir)
    replacement = Path(args.replacement_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = [str(name) for name in args.datasets]
    if not targets:
        raise ValueError("--datasets must name at least one dataset to repair.")
    original_rows = read_rows(original / "raw_results.csv")
    replacement_rows = read_rows(replacement / "raw_results.csv")
    target_set = set(targets)
    selected_replacements: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    for dataset in targets:
        old = [row for row in original_rows if row.get("dataset") == dataset]
        candidates = [
            row for row in replacement_rows if row.get("dataset") == dataset
        ]
        if not old:
            raise ValueError(f"Dataset {dataset!r} is absent from the benchmark.")
        if not candidates:
            raise ValueError(f"Dataset {dataset!r} is absent from the replacement.")
        required = sorted(
            {
                (
                    str(row.get("variant", "relation")),
                    float(row["mu"]),
                    int(float(row["max_views"])),
                )
                for row in old
            }
        )
        replacement_lookup: Dict[Tuple[str, float, int], List[Dict[str, str]]] = {}
        for row in candidates:
            if str(row.get("dataset_type")) != "nominal":
                continue
            key = (
                str(row.get("variant", "relation")),
                float(row["mu"]),
                int(float(row["max_views"])),
            )
            replacement_lookup.setdefault(key, []).append(row)
        missing = [key for key in required if key not in replacement_lookup]
        if missing:
            raise ValueError(
                f"Replacement for {dataset!r} is missing required configurations: {missing}."
            )
        for key in required:
            matches = replacement_lookup[key]
            matches.sort(key=lambda row: str(row.get("config_id", "")))
            selected_replacements.append(dict(matches[0]))
        audit.append(
            {
                "dataset": dataset,
                "removed_rows": len(old),
                "inserted_rows": len(required),
                "required_mus": "|".join(str(key[1]) for key in required),
                "replacement_dataset_type": "nominal",
            }
        )
    merged = [
        dict(row) for row in original_rows if row.get("dataset") not in target_set
    ] + selected_replacements
    merged.sort(
        key=lambda row: (
            str(row.get("dataset", "")).casefold(),
            str(row.get("variant", "")),
            float(row.get("delta", 0.0)),
            float(row.get("mu", 0.0)),
            int(float(row.get("max_views", 0))),
        )
    )
    write_rows(merged, output / "raw_results.csv")
    best_auc, _ = _write_best_outputs(_as_numeric_rows(merged), output)
    _merge_baseline(
        args.baseline_auc, best_auc, "auc", output / "comparison_auc.csv"
    )
    _merge_baseline(
        args.baseline_ap, best_auc, "ap", output / "comparison_ap.csv"
    )
    for filename in ("failures.csv", "dataset_load_failures.csv"):
        source = original / filename
        failure_rows = read_rows(source) if source.is_file() else []
        write_rows(failure_rows, output / filename)
    write_rows(audit, output / "repair_audit.csv")
    save_json(
        {
            "experiment": "repair-benchmark",
            "source_benchmark": str(original.resolve()),
            "replacement_results": str(replacement.resolve()),
            "datasets": targets,
            "source_rows": len(original_rows),
            "corrected_rows": len(merged),
            "validation": "All original variant/mu/max_views keys must be present as nominal replacement rows.",
        },
        output / "run_config.json",
    )


SINGLE_FULL_VIEW_CONTROL: Mapping[str, object] = {
    "min_views": 1,
    "max_views": 1,
    "supplemental_view_count": 0,
    "include_full_view": True,
}


LEGACY_MODEL_OVERRIDES: Mapping[str, object] = {
    "pivot_mode": "relation_medoid",
    "gain_mode": "normalized",
    "gta_mode": "frontier",
    "fusion_mode": "structural_stability",
}


CUMULATIVE_ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = (
    (
        "u0_minimal_unified_core",
        {
            "importance_mode": "uniform",
            **SINGLE_FULL_VIEW_CONTROL,
            "pivot_mode": "relation_farthest_pair",
            "gain_mode": "raw",
            "gta_mode": "single_scale",
            "fusion_mode": "current_certainty",
        },
    ),
    (
        "u1_relation_entropy_weighting",
        {
            **SINGLE_FULL_VIEW_CONTROL,
            "pivot_mode": "relation_farthest_pair",
            "gain_mode": "raw",
            "gta_mode": "single_scale",
            "fusion_mode": "current_certainty",
        },
    ),
    (
        "u2_deterministic_multiview",
        {
            "pivot_mode": "relation_farthest_pair",
            "gain_mode": "raw",
            "gta_mode": "single_scale",
            "fusion_mode": "current_certainty",
        },
    ),
    (
        "u3_support_aware_pivots",
        {
            "pivot_mode": "relation_medoid",
            "gain_mode": "raw",
            "gta_mode": "single_scale",
            "fusion_mode": "current_certainty",
        },
    ),
    (
        "u4_normalized_cohesion_gain",
        {
            **LEGACY_MODEL_OVERRIDES,
            "gta_mode": "single_scale",
            "fusion_mode": "current_certainty",
        },
    ),
    (
        "u5_multiscale_gta_frontier",
        {**LEGACY_MODEL_OVERRIDES, "fusion_mode": "current_certainty"},
    ),
    ("full_model", LEGACY_MODEL_OVERRIDES),
)

# Backward-compatible name: `main` now means the cumulative unified-framework chain.
MAIN_ABLATIONS = CUMULATIVE_ABLATIONS


REVISED_FIXED_DELTA = 0.8
REVISED_PURE_NOMINAL_FIXED_DELTA = 0.2
REVISED_FIXED_MU = 0.02
REVISED_PURE_NOMINAL_FIXED_MU = 0.005
REVISED_FINAL_VARIANT = "r3_multiscale_leaf_path_gta"


REVISED_MODEL_OVERRIDES: Mapping[str, object] = {
    "pivot_mode": "relation_diameter_root",
    "gain_mode": "raw",
    "gta_mode": "frontier",
    "fusion_mode": "fixed_component_mean",
}


PIVOT_REDESIGN_ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = (
    (
        "pivot_relation_medoid",
        {**REVISED_MODEL_OVERRIDES, "pivot_mode": "relation_medoid"},
    ),
    (
        "pivot_relation_diameter",
        {**REVISED_MODEL_OVERRIDES, "pivot_mode": "relation_farthest_pair"},
    ),
)


REVISED_CUMULATIVE_ABLATIONS: Tuple[
    Tuple[str, Mapping[str, object]], ...
] = (
    (
        "r0_unified_single_view_core",
        {
            "importance_mode": "uniform",
            **SINGLE_FULL_VIEW_CONTROL,
            "pivot_mode": "relation_medoid",
            "gain_mode": "raw",
            "gta_mode": "single_scale",
            "enabled_components": ("leaf",),
            "fusion_mode": "fixed_component_mean",
        },
    ),
    (
        "r1_relation_weighted_multiview",
        {
            "pivot_mode": "relation_medoid",
            "gain_mode": "raw",
            "gta_mode": "single_scale",
            "enabled_components": ("leaf",),
            "fusion_mode": "fixed_component_mean",
        },
    ),
    (
        "r2_relation_diameter_raw_structure",
        {
            **REVISED_MODEL_OVERRIDES,
            "gta_mode": "single_scale",
            "enabled_components": ("leaf",),
            "fusion_mode": "fixed_component_mean",
        },
    ),
    (
        "r3_multiscale_leaf_path_gta",
        {**REVISED_MODEL_OVERRIDES, "fusion_mode": "fixed_component_mean"},
    ),
)


GAIN_COMPLEXITY_VARIANTS: Tuple[Tuple[str, Mapping[str, object]], ...] = (
    ("raw_gain", REVISED_MODEL_OVERRIDES),
    (
        "normalized_gain",
        {**REVISED_MODEL_OVERRIDES, "gain_mode": "normalized"},
    ),
)


LEAVE_ONE_OUT_ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = (
    ("full_model", LEGACY_MODEL_OVERRIDES),
    (
        "uniform_attribute_weights",
        {**LEGACY_MODEL_OVERRIDES, "importance_mode": "uniform"},
    ),
    ("single_full_view", {**LEGACY_MODEL_OVERRIDES, **SINGLE_FULL_VIEW_CONTROL}),
    (
        "relation_farthest_pivots",
        {**LEGACY_MODEL_OVERRIDES, "pivot_mode": "relation_farthest_pair"},
    ),
    ("raw_cohesion_gain", {**LEGACY_MODEL_OVERRIDES, "gain_mode": "raw"}),
    ("single_scale_gta", {**LEGACY_MODEL_OVERRIDES, "gta_mode": "single_scale"}),
    (
        "certainty_only_fusion",
        {**LEGACY_MODEL_OVERRIDES, "fusion_mode": "current_certainty"},
    ),
)


EVIDENCE_ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = (
    (
        "without_leaf",
        {**LEGACY_MODEL_OVERRIDES, "enabled_components": ("gta", "path")},
    ),
    (
        "without_gta",
        {**LEGACY_MODEL_OVERRIDES, "enabled_components": ("leaf", "path")},
    ),
    (
        "without_path",
        {**LEGACY_MODEL_OVERRIDES, "enabled_components": ("leaf", "gta")},
    ),
)


# Evidence leave-one-out for the currently frozen revised method.  Keep the
# legacy evidence family above available for reproducibility of old diagnostic
# runs, but route the public ``evidence`` suite to this revised family.
REVISED_EVIDENCE_ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = (
    (
        "revised_without_leaf",
        {**REVISED_MODEL_OVERRIDES, "enabled_components": ("gta", "path")},
    ),
    (
        "revised_without_gta",
        {**REVISED_MODEL_OVERRIDES, "enabled_components": ("leaf", "path")},
    ),
    (
        "revised_without_path",
        {**REVISED_MODEL_OVERRIDES, "enabled_components": ("leaf", "gta")},
    ),
)


INTERACTION_ONLY_ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = (
    (
        "int_weight_view_00",
        {
            **LEGACY_MODEL_OVERRIDES,
            "importance_mode": "uniform",
            **SINGLE_FULL_VIEW_CONTROL,
        },
    ),
    (
        "int_pivot_gain_00",
        {
            **LEGACY_MODEL_OVERRIDES,
            "pivot_mode": "relation_farthest_pair",
            "gain_mode": "raw",
        },
    ),
    (
        "int_gain_gta_00",
        {
            **LEGACY_MODEL_OVERRIDES,
            "gain_mode": "raw",
            "gta_mode": "single_scale",
        },
    ),
)

STRUCTURE_EVIDENCE_ONLY_ABLATIONS: Tuple[
    Tuple[str, Mapping[str, object]], ...
] = (
    (
        "int_structure_evidence_00",
        {
            "pivot_mode": "relation_medoid",
            "gain_mode": "raw",
            "gta_mode": "single_scale",
            "enabled_components": ("leaf",),
            "fusion_mode": "fixed_component_mean",
        },
    ),
    (
        "int_structure_evidence_01",
        {
            "pivot_mode": "relation_medoid",
            "gain_mode": "raw",
            "gta_mode": "frontier",
            "fusion_mode": "fixed_component_mean",
        },
    ),
)


INTERACTION_FAMILIES: Tuple[Mapping[str, str], ...] = (
    {
        "family": "relation_entropy_weighting_x_multiview",
        "mechanism_a": "Relation-entropy attribute weighting",
        "mechanism_b": "Deterministic multi-view ensemble",
        "cell_00": "int_weight_view_00",
        "cell_10": "single_full_view",
        "cell_01": "uniform_attribute_weights",
        "cell_11": "full_model",
    },
    {
        "family": "support_aware_pivots_x_normalized_gain",
        "mechanism_a": "Support-aware relation-medoid pivots",
        "mechanism_b": "Headroom-normalized cohesion gain",
        "cell_00": "int_pivot_gain_00",
        "cell_10": "raw_cohesion_gain",
        "cell_01": "relation_farthest_pivots",
        "cell_11": "full_model",
    },
    {
        "family": "normalized_gain_x_multiscale_gta",
        "mechanism_a": "Headroom-normalized cohesion gain",
        "mechanism_b": "Branch-adaptive multiscale GTA frontier",
        "cell_00": "int_gain_gta_00",
        "cell_10": "single_scale_gta",
        "cell_01": "raw_cohesion_gain",
        "cell_11": "full_model",
    },
    {
        "family": "hierarchical_structure_x_multiscale_evidence",
        "mechanism_a": "Relation-diameter root-pivot structure with raw gain",
        "mechanism_b": "Leaf-Path-GTA multiscale evidence with fixed component mean",
        "cell_00": "int_structure_evidence_00",
        "cell_10": "r2_relation_diameter_raw_structure",
        "cell_01": "int_structure_evidence_01",
        "cell_11": "r3_multiscale_leaf_path_gta",
    },
)

PAPER_INTERACTION_FAMILY = "hierarchical_structure_x_multiscale_evidence"

PAPER_MAIN_ABLATION_NAMES: Tuple[str, ...] = (
    "r0_unified_single_view_core",
    "r1_relation_weighted_multiview",
    "r2_relation_diameter_raw_structure",
    "r3_multiscale_leaf_path_gta",
)


def _ordered_unique_names(names: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(names))


INTERACTION_ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = tuple(
    (
        name,
        dict(
            LEAVE_ONE_OUT_ABLATIONS
            + INTERACTION_ONLY_ABLATIONS
            + REVISED_CUMULATIVE_ABLATIONS
            + STRUCTURE_EVIDENCE_ONLY_ABLATIONS
        )[name],
    )
    for name in _ordered_unique_names(
        family[cell]
        for family in INTERACTION_FAMILIES
        for cell in ("cell_00", "cell_10", "cell_01", "cell_11")
    )
)


ABLATIONS: Tuple[Tuple[str, Mapping[str, object]], ...] = tuple(
    dict(
        CUMULATIVE_ABLATIONS
        + PIVOT_REDESIGN_ABLATIONS
        + REVISED_CUMULATIVE_ABLATIONS
        + LEAVE_ONE_OUT_ABLATIONS
        + EVIDENCE_ABLATIONS
        + REVISED_EVIDENCE_ABLATIONS
        + INTERACTION_ONLY_ABLATIONS
        + STRUCTURE_EVIDENCE_ONLY_ABLATIONS
    ).items()
)


ABLATION_METADATA: Mapping[str, Mapping[str, str]] = {
    "u0_minimal_unified_core": {
        "group": "cumulative",
        "mechanism": "Minimal unified relation-consistent core",
        "control": "Uniform weights, one full view, farthest pivots, raw gain, single-scale GTA, certainty fusion",
        "cumulative_stage": "0",
        "introduced_mechanism": "Unified relation-consistent core",
    },
    "u1_relation_entropy_weighting": {
        "group": "cumulative",
        "mechanism": "Relation-entropy attribute weighting",
        "control": "U0 plus relation-entropy weights",
        "cumulative_stage": "1",
        "introduced_mechanism": "Relation-entropy attribute weighting",
    },
    "u2_deterministic_multiview": {
        "group": "cumulative",
        "mechanism": "Deterministic multi-view ensemble",
        "control": "U1 plus deterministic multi-view construction",
        "cumulative_stage": "2",
        "introduced_mechanism": "Deterministic multi-view ensemble",
    },
    "u3_support_aware_pivots": {
        "group": "cumulative",
        "mechanism": "Support-aware relation-medoid pivots",
        "control": "U2 plus support-aware relation-medoid pivots",
        "cumulative_stage": "3",
        "introduced_mechanism": "Support-aware relation-medoid pivots",
    },
    "u4_normalized_cohesion_gain": {
        "group": "cumulative",
        "mechanism": "Headroom-normalized cohesion gain",
        "control": "U3 plus headroom-normalized cohesion gain",
        "cumulative_stage": "4",
        "introduced_mechanism": "Headroom-normalized cohesion gain",
    },
    "u5_multiscale_gta_frontier": {
        "group": "cumulative",
        "mechanism": "Branch-adaptive multiscale GTA frontier",
        "control": "U4 plus branch-adaptive multiscale GTA frontier",
        "cumulative_stage": "5",
        "introduced_mechanism": "Branch-adaptive multiscale GTA frontier",
    },
    "full_model": {
        "group": "reference",
        "mechanism": "Complete unified relation-consistent framework",
        "control": "None",
        "cumulative_stage": "6",
        "introduced_mechanism": "Structural availability-stability reliability",
    },
    "pivot_relation_medoid": {
        "group": "pivot_redesign",
        "mechanism": "Relation-medoid pivots at every level",
        "control": "Same revised framework with relation-medoid pivots",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "pivot_relation_diameter": {
        "group": "pivot_redesign",
        "mechanism": "Exact relation-diameter pivots at every level",
        "control": "Same revised framework with exact farthest pairs",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "r0_unified_single_view_core": {
        "group": "revised_cumulative",
        "mechanism": "Unified single-view relation core",
        "control": "Uniform weighting, one full view, medoid pivots, raw gain, Leaf evidence",
        "cumulative_stage": "0",
        "introduced_mechanism": "Unified relation-consistent core",
    },
    "r1_relation_weighted_multiview": {
        "group": "revised_cumulative",
        "mechanism": "Relation-weighted deterministic multi-view representation",
        "control": "R0 plus relation-entropy weighting and deterministic multi-view construction",
        "cumulative_stage": "1",
        "introduced_mechanism": "Relation-weighted deterministic multi-view representation",
    },
    "r2_relation_diameter_raw_structure": {
        "group": "revised_cumulative",
        "mechanism": "Relation-diameter raw-gain hierarchical structure learning",
        "control": "R1 plus the root relation-diameter pivot under raw cohesion gain",
        "cumulative_stage": "2",
        "introduced_mechanism": "Relation-diameter hierarchical structure learning",
    },
    "r3_multiscale_leaf_path_gta": {
        "group": "revised_cumulative",
        "mechanism": "Leaf-Path-GTA multiscale evidence",
        "control": "R2 plus Path evidence and branch-adaptive multiscale GTA frontier with fixed component mean",
        "cumulative_stage": "3",
        "introduced_mechanism": "Leaf-Path-GTA multiscale evidence",
    },
    "revised_without_leaf": {
        "group": "evidence",
        "mechanism": "Revised model without Leaf evidence",
        "control": "Frozen revised model with GTA and Path evidence only",
        "cumulative_stage": "",
        "introduced_mechanism": "Leaf evidence removal",
    },
    "revised_without_gta": {
        "group": "evidence",
        "mechanism": "Revised model without GTA evidence",
        "control": "Frozen revised model with Leaf and Path evidence only",
        "cumulative_stage": "",
        "introduced_mechanism": "GTA evidence removal",
    },
    "revised_without_path": {
        "group": "evidence",
        "mechanism": "Revised model without Path evidence",
        "control": "Frozen revised model with Leaf and GTA evidence only",
        "cumulative_stage": "",
        "introduced_mechanism": "Path evidence removal",
    },
    "uniform_attribute_weights": {
        "group": "leave_one_out",
        "mechanism": "Relation-entropy attribute weighting",
        "control": "Equal weights over non-constant attributes",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "single_full_view": {
        "group": "leave_one_out",
        "mechanism": "Deterministic multi-view ensemble",
        "control": "One full-feature relation view",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "relation_farthest_pivots": {
        "group": "leave_one_out",
        "mechanism": "Support-aware relation-medoid pivots",
        "control": "Most dissimilar pair in the same relation space",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "raw_cohesion_gain": {
        "group": "leave_one_out",
        "mechanism": "Headroom-normalized cohesion gain",
        "control": "Unnormalized absolute cohesion gain",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "single_scale_gta": {
        "group": "leave_one_out",
        "mechanism": "Branch-adaptive multiscale GTA frontier",
        "control": "Best common tree depth for all GTA anchors",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "certainty_only_fusion": {
        "group": "leave_one_out",
        "mechanism": "Structural availability-stability reliability",
        "control": "Sample certainty without structural reliability",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "without_leaf": {
        "group": "evidence",
        "mechanism": "Leaf evidence",
        "control": "Fuse GTA and Path only",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "without_gta": {
        "group": "evidence",
        "mechanism": "GTA evidence",
        "control": "Fuse Leaf and Path only",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "without_path": {
        "group": "evidence",
        "mechanism": "Path evidence",
        "control": "Fuse Leaf and GTA only",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "int_weight_view_00": {
        "group": "interaction",
        "mechanism": "Weighting-view interaction baseline",
        "control": "Uniform weights and one full view",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "int_pivot_gain_00": {
        "group": "interaction",
        "mechanism": "Pivot-gain interaction baseline",
        "control": "Relation-farthest pivots and raw cohesion gain",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "int_gain_gta_00": {
        "group": "interaction",
        "mechanism": "Gain-GTA interaction baseline",
        "control": "Raw cohesion gain and single-scale GTA",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "int_structure_evidence_00": {
        "group": "structure_evidence_interaction",
        "mechanism": "Structure-evidence interaction baseline",
        "control": "Relation-medoid raw-gain structure with Leaf evidence only",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
    "int_structure_evidence_01": {
        "group": "structure_evidence_interaction",
        "mechanism": "Multiscale evidence on the control structure",
        "control": "Relation-medoid raw-gain structure with Leaf-Path-GTA evidence and fixed component mean",
        "cumulative_stage": "",
        "introduced_mechanism": "",
    },
}


def _resolve_ablation_variants(text: str) -> List[Tuple[str, Mapping[str, object]]]:
    lookup = dict(ABLATIONS)
    cumulative_names = [name for name, _ in CUMULATIVE_ABLATIONS]
    revised_cumulative_names = [
        name for name, _ in REVISED_CUMULATIVE_ABLATIONS
    ]
    structure_evidence_names = [
        family[cell]
        for family in INTERACTION_FAMILIES
        if family["family"] == "hierarchical_structure_x_multiscale_evidence"
        for cell in ("cell_00", "cell_10", "cell_01", "cell_11")
    ]
    pivot_redesign_names = [name for name, _ in PIVOT_REDESIGN_ABLATIONS]
    leave_one_out_names = [name for name, _ in LEAVE_ONE_OUT_ABLATIONS]
    interaction_names = [name for name, _ in INTERACTION_ABLATIONS]
    legacy_evidence_names = [
        "full_model",
        *(name for name, _ in EVIDENCE_ABLATIONS),
    ]
    revised_evidence_names = [
        REVISED_FINAL_VARIANT,
        *(name for name, _ in REVISED_EVIDENCE_ABLATIONS),
    ]
    suites = {
        "cumulative": cumulative_names,
        "main": cumulative_names,
        "paper_main": list(PAPER_MAIN_ABLATION_NAMES),
        "pivot_redesign": pivot_redesign_names,
        "revised_cumulative": revised_cumulative_names,
        "revised_main": revised_cumulative_names,
        "structure_evidence": structure_evidence_names,
        "structure_evidence_interaction": structure_evidence_names,
        "interaction": interaction_names,
        "leave_one_out": leave_one_out_names,
        "evidence": revised_evidence_names,
        "revised_evidence": revised_evidence_names,
        "legacy_evidence": legacy_evidence_names,
        "all": _ordered_unique_names(
            [
                *cumulative_names,
                *revised_cumulative_names,
                *pivot_redesign_names,
                *interaction_names,
                *leave_one_out_names,
                *legacy_evidence_names,
                *revised_evidence_names,
            ]
        ),
    }
    tokens = [item.strip() for item in str(text).split(",") if item.strip()]
    if not tokens:
        tokens = ["revised_cumulative"]
    names: List[str] = []
    for token in tokens:
        expanded = suites.get(token, [token])
        for name in expanded:
            if name not in lookup:
                raise ValueError(
                    f"Unknown ablation variant or suite {name!r}. "
                    f"Choose from {sorted([*lookup, *suites])}."
                )
            if name not in names:
                names.append(name)
    revised_names = (
        set(revised_cumulative_names)
        | set(pivot_redesign_names)
        | set(structure_evidence_names)
        | set(revised_evidence_names)
    )
    if any(name in revised_names for name in names):
        if REVISED_FINAL_VARIANT not in names:
            names.insert(0, REVISED_FINAL_VARIANT)
    elif "full_model" not in names:
        names.insert(0, "full_model")
    return [(name, lookup[name]) for name in names]


def _ablation_requires_fixed_protocol(names: Sequence[str]) -> bool:
    selected = set(names)
    has_cumulative_step = any(
        before in selected and after in selected
        for before, after, _ in CUMULATIVE_COMPARISONS
    )
    has_complete_interaction = any(
        all(
            family[cell] in selected
            for cell in ("cell_00", "cell_10", "cell_01", "cell_11")
        )
        for family in INTERACTION_FAMILIES
    )
    revised_evidence_names = {
        REVISED_FINAL_VARIANT,
        *(name for name, _ in REVISED_EVIDENCE_ABLATIONS),
    }
    has_revised_evidence_comparison = (
        REVISED_FINAL_VARIANT in selected
        and bool(selected & (revised_evidence_names - {REVISED_FINAL_VARIANT}))
    )
    return (
        has_cumulative_step
        or has_complete_interaction
        or has_revised_evidence_comparison
    )


def _descending_average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


CUMULATIVE_COMPARISONS: Tuple[Tuple[str, str, str], ...] = (
    (
        "u0_minimal_unified_core",
        "u1_relation_entropy_weighting",
        "Relation-entropy attribute weighting",
    ),
    (
        "u1_relation_entropy_weighting",
        "u2_deterministic_multiview",
        "Deterministic multi-view ensemble",
    ),
    (
        "u2_deterministic_multiview",
        "u3_support_aware_pivots",
        "Support-aware relation-medoid pivots",
    ),
    (
        "u3_support_aware_pivots",
        "u4_normalized_cohesion_gain",
        "Headroom-normalized cohesion gain",
    ),
    (
        "u4_normalized_cohesion_gain",
        "u5_multiscale_gta_frontier",
        "Branch-adaptive multiscale GTA frontier",
    ),
    (
        "u5_multiscale_gta_frontier",
        "full_model",
        "Structural availability-stability reliability",
    ),
    (
        "r0_unified_single_view_core",
        "r1_relation_weighted_multiview",
        "Relation-weighted deterministic multi-view representation",
    ),
    (
        "r1_relation_weighted_multiview",
        "int_structure_evidence_01",
        "Leaf-Path-GTA multiscale evidence on the relation-medoid hierarchy",
    ),
    (
        "int_structure_evidence_01",
        REVISED_FINAL_VARIANT,
        "Relation-diameter root-pivot structure optimization",
    ),
    (
        "r1_relation_weighted_multiview",
        "r2_relation_diameter_raw_structure",
        "Relation-diameter raw-gain hierarchical structure learning",
    ),
    (
        "r2_relation_diameter_raw_structure",
        "r3_multiscale_leaf_path_gta",
        "Leaf-Path-GTA multiscale evidence",
    ),
)


ABLATION_DIAGNOSTIC_FIELDS = (
    "views_used",
    "tree_mean_nodes",
    "tree_mean_depth",
    "tree_mean_gtas",
    "tree_mean_internal_gtas",
    "tree_internal_gta_fraction",
    "tree_mean_gta_depth",
    "tree_mean_frontier_lambda",
    "tree_mean_frontier_s0",
    "tree_mean_frontier_sl",
    "tree_mean_frontier_gain_span",
    "tree_frontier_calibrated_rate",
    "tree_frontier_root_unsplit_rate",
    "tree_frontier_no_optional_refinement_rate",
    "tree_frontier_no_positive_span_rate",
    "tree_coarsest_frontier_rate",
    "tree_all_leaf_frontier_rate",
    "tree_mixed_frontier_rate",
    "tree_root_split_rate",
    "tree_root_split_balance",
    "tree_mean_split_balance",
    "tree_mean_normalized_gain",
    "tree_medoid_pivot_attempts",
    "tree_farthest_pivot_attempts",
    "tree_medoid_pivot_accepts",
    "tree_farthest_pivot_accepts",
    "tree_pivot_selection_seconds",
    "runtime_total_seconds",
)


def _row_number(row: Mapping[str, object], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _holm_adjust_rows(rows: List[Dict[str, object]], alpha: float = 0.05) -> None:
    order = sorted(
        range(len(rows)),
        key=lambda index: float(rows[index].get("wilcoxon_p", 1.0)),
    )
    running = 0.0
    for position, index in enumerate(order):
        raw_p = float(rows[index].get("wilcoxon_p", 1.0))
        adjusted = min(1.0, (len(order) - position) * raw_p)
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["holm_reject"] = bool(running <= alpha)


def _remove_generated_files(output: Path, names: Iterable[str]) -> None:
    for name in names:
        path = output / name
        if path.is_file():
            path.unlink()


def _write_cumulative_outputs(
    output: Path,
    by_variant: Mapping[str, Mapping[str, Mapping[str, object]]],
    aligned: Sequence[str],
    names: Sequence[str],
) -> Dict[str, object]:
    output_names = (
        "cumulative_effects.csv",
        "cumulative_effect_summary.csv",
        "cumulative_pairwise.csv",
    )
    active = [
        item
        for item in CUMULATIVE_COMPARISONS
        if item[0] in names and item[1] in names
    ]
    if not active:
        _remove_generated_files(output, output_names)
        return {"adjacent_comparison_count": 0, "total_comparison": False}

    effect_rows: List[Dict[str, object]] = []
    comparisons: List[Tuple[str, str, str, str]] = [
        ("adjacent", before, after, mechanism)
        for before, after, mechanism in active
    ]
    cumulative_chains = (
        (
            CUMULATIVE_ABLATIONS[0][0],
            "full_model",
            "Complete cumulative framework",
        ),
        (
            REVISED_CUMULATIVE_ABLATIONS[0][0],
            REVISED_FINAL_VARIANT,
            "Complete revised cumulative framework with fixed component mean",
        ),
    )
    for base, reference, mechanism in cumulative_chains:
        if base in names and reference in names:
            comparisons.append(("total", base, reference, mechanism))
    for comparison_kind, before_name, after_name, mechanism in comparisons:
        comparison_id = f"{before_name}_to_{after_name}"
        for dataset in aligned:
            before = by_variant[before_name][dataset]
            after = by_variant[after_name][dataset]
            row: Dict[str, object] = {
                "dataset": dataset,
                "dataset_type": after.get("dataset_type", "unknown"),
                "comparison_kind": comparison_kind,
                "comparison_id": comparison_id,
                "stage_from": before_name,
                "stage_to": after_name,
                "introduced_mechanism": mechanism,
                "auc_before": float(before["auc"]),
                "auc_after": float(after["auc"]),
                "auc_increment_after_minus_before": float(after["auc"])
                - float(before["auc"]),
                "ap_before": float(before["ap"]),
                "ap_after": float(after["ap"]),
                "ap_increment_after_minus_before": float(after["ap"])
                - float(before["ap"]),
            }
            for field in ABLATION_DIAGNOSTIC_FIELDS:
                before_value = _row_number(before, field)
                after_value = _row_number(after, field)
                row[f"{field}_before"] = before_value
                row[f"{field}_after"] = after_value
                row[f"{field}_delta_after_minus_before"] = after_value - before_value
            effect_rows.append(row)
    write_rows(effect_rows, output / "cumulative_effects.csv")

    tolerance = 1e-12
    summary: List[Dict[str, object]] = []
    comparison_ids = list(dict.fromkeys(str(row["comparison_id"]) for row in effect_rows))
    for comparison_id in comparison_ids:
        comparison_rows = [
            row for row in effect_rows if row["comparison_id"] == comparison_id
        ]
        dataset_types = [
            "all",
            *sorted({str(row["dataset_type"]) for row in comparison_rows}),
        ]
        for dataset_type in dataset_types:
            selected = (
                comparison_rows
                if dataset_type == "all"
                else [
                    row
                    for row in comparison_rows
                    if row["dataset_type"] == dataset_type
                ]
            )
            auc = np.asarray(
                [float(row["auc_increment_after_minus_before"]) for row in selected]
            )
            ap = np.asarray(
                [float(row["ap_increment_after_minus_before"]) for row in selected]
            )
            first = selected[0]
            item: Dict[str, object] = {
                "comparison_kind": first["comparison_kind"],
                "comparison_id": comparison_id,
                "stage_from": first["stage_from"],
                "stage_to": first["stage_to"],
                "introduced_mechanism": first["introduced_mechanism"],
                "dataset_type": dataset_type,
                "dataset_count": len(selected),
                "mean_auc_increment": float(auc.mean()),
                "median_auc_increment": float(np.median(auc)),
                "auc_after_wins": int(np.sum(auc > tolerance)),
                "auc_ties": int(np.sum(np.abs(auc) <= tolerance)),
                "auc_after_losses": int(np.sum(auc < -tolerance)),
                "mean_ap_increment": float(ap.mean()),
                "median_ap_increment": float(np.median(ap)),
                "ap_after_wins": int(np.sum(ap > tolerance)),
                "ap_ties": int(np.sum(np.abs(ap) <= tolerance)),
                "ap_after_losses": int(np.sum(ap < -tolerance)),
            }
            for field in ABLATION_DIAGNOSTIC_FIELDS:
                key = f"{field}_delta_after_minus_before"
                item[f"mean_{key}"] = _finite_mean(
                    [float(row[key]) for row in selected]
                )
            summary.append(item)
    write_rows(summary, output / "cumulative_effect_summary.csv")

    pairwise: List[Dict[str, object]] = []
    adjacent_rows = [
        row for row in effect_rows if row["comparison_kind"] == "adjacent"
    ]
    if len(aligned) >= 2:
        for metric in ("auc", "ap"):
            metric_rows: List[Dict[str, object]] = []
            for before_name, after_name, mechanism in active:
                pair = np.asarray(
                    [
                        [
                            float(by_variant[after_name][dataset][metric]),
                            float(by_variant[before_name][dataset][metric]),
                        ]
                        for dataset in aligned
                    ],
                    dtype=np.float64,
                )
                item = dict(
                    statistical_comparison(pair, [after_name, before_name])[
                        "wilcoxon_holm"
                    ][0]
                )
                item.update(
                    {
                        "metric": metric,
                        "comparison_id": f"{before_name}_to_{after_name}",
                        "introduced_mechanism": mechanism,
                    }
                )
                metric_rows.append(item)
            _holm_adjust_rows(metric_rows)
            pairwise.extend(metric_rows)
    write_rows(pairwise, output / "cumulative_pairwise.csv")
    return {
        "adjacent_comparison_count": len(active),
        "total_comparison": any(
            row["comparison_kind"] == "total" for row in effect_rows
        ),
        "effect_row_count": len(effect_rows),
        "adjacent_effect_row_count": len(adjacent_rows),
    }


def _latex_escape_text(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _paper_interaction_latex(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Per-dataset structure--evidence four-cell interaction. All cells use the same fixed configuration selected by AUC; AP is reported from that configuration. Higher is better.}",
        r"\label{tab:interaction_20datasets}",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Dataset & \multicolumn{5}{c}{AUC} & \multicolumn{5}{c}{AP} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r" & $M_{00}$ & $M_{10}$ & $M_{01}$ & $M_{11}$ & $I_d$ & $M_{00}$ & $M_{10}$ & $M_{01}$ & $M_{11}$ & $I_d$ \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            _latex_escape_text(row["dataset"]),
            *[
                f"{float(row[field]):.4f}"
                for field in (
                    "auc_m00",
                    "auc_m10",
                    "auc_m01",
                    "auc_m11",
                    "auc_interaction",
                    "ap_m00",
                    "ap_m10",
                    "ap_m01",
                    "ap_m11",
                    "ap_interaction",
                )
            ],
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _paper_interaction_summary_latex(
    rows: Sequence[Mapping[str, object]],
    tolerance: float = 1e-12,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Summary of the observational structure--evidence interaction.}",
        r"\label{tab:interaction_summary}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Metric & $N$ & Mean & Median & $I_d>0$ & $I_d=0$ & $I_d<0$ \\",
        r"\midrule",
    ]
    for metric in ("auc", "ap"):
        values = np.asarray(
            [float(row[f"{metric}_interaction"]) for row in rows],
            dtype=np.float64,
        )
        lines.append(
            " & ".join(
                (
                    metric.upper(),
                    str(len(values)),
                    f"{float(values.mean()):.4f}",
                    f"{float(np.median(values)):.4f}",
                    str(int(np.sum(values > tolerance))),
                    str(int(np.sum(np.abs(values) <= tolerance))),
                    str(int(np.sum(values < -tolerance))),
                )
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _write_paper_interaction_outputs(
    output: Path,
    by_variant: Mapping[str, Mapping[str, Mapping[str, object]]],
    names: Sequence[str],
) -> Dict[str, object]:
    """Emit the paper's 20-dataset interaction only from a complete 2x2 design."""

    output_names = (
        "interaction_20datasets.csv",
        "interaction_20datasets.tex",
        "interaction_summary.tex",
    )
    blocker_path = output / "interaction_20datasets_blockers.json"
    family = next(
        item
        for item in INTERACTION_FAMILIES
        if item["family"] == PAPER_INTERACTION_FAMILY
    )
    cells = {
        code: str(family[f"cell_{code}"])
        for code in ("00", "10", "01", "11")
    }
    missing_variants = [
        variant
        for variant in cells.values()
        if variant not in names or variant not in by_variant
    ]
    if missing_variants:
        _remove_generated_files(output, output_names)
        payload = {
            "status": "BLOCKED",
            "reason": "One or more required four-cell variants were not requested.",
            "missing_variants": missing_variants,
            "required_cells": cells,
        }
        save_json(payload, blocker_path)
        return payload

    dataset_sets = {
        code: set(by_variant[variant]) for code, variant in cells.items()
    }
    union = set().union(*dataset_sets.values())
    aligned = set.intersection(*dataset_sets.values())
    missing_by_cell = {
        code: sorted(union - datasets)
        for code, datasets in dataset_sets.items()
        if union - datasets
    }
    if len(aligned) != PAPER_DATASET_COUNT or missing_by_cell:
        _remove_generated_files(output, output_names)
        payload = {
            "status": "BLOCKED",
            "reason": (
                "The structure--evidence design is not a complete aligned "
                f"{PAPER_DATASET_COUNT}-dataset matrix."
            ),
            "expected_dataset_count": PAPER_DATASET_COUNT,
            "union_dataset_count": len(union),
            "aligned_dataset_count": len(aligned),
            "missing_by_cell": missing_by_cell,
            "required_cells": cells,
        }
        save_json(payload, blocker_path)
        return payload

    abbreviations: Dict[str, str] = {}
    naming_errors: List[str] = []
    for dataset in aligned:
        try:
            abbreviation = paper_dataset_abbreviation(dataset)
        except ValueError as error:
            naming_errors.append(str(error))
            continue
        if abbreviation in abbreviations.values():
            naming_errors.append(
                f"Multiple input rows map to the Table 2 abbreviation {abbreviation!r}."
            )
        abbreviations[dataset] = abbreviation
    if naming_errors or len(abbreviations) != PAPER_DATASET_COUNT:
        _remove_generated_files(output, output_names)
        payload = {
            "status": "BLOCKED",
            "reason": "Dataset names do not map one-to-one to the paper's Table 2.",
            "naming_errors": naming_errors,
            "required_cells": cells,
        }
        save_json(payload, blocker_path)
        return payload

    rows: List[Dict[str, object]] = []
    invalid_values: List[str] = []
    for dataset in sorted(
        aligned,
        key=lambda name: PAPER_DATASET_ORDER[abbreviations[name]],
    ):
        cell_rows = {
            code: by_variant[variant][dataset] for code, variant in cells.items()
        }
        row: Dict[str, object] = {
            "dataset": abbreviations[dataset],
            "dataset_type": cell_rows["11"].get("dataset_type", "unknown"),
        }
        for metric in ("auc", "ap"):
            values = {
                code: float(cell_rows[code][metric])
                for code in ("00", "10", "01", "11")
            }
            if not np.all(np.isfinite(list(values.values()))):
                invalid_values.append(f"{dataset}:{metric}")
            for code, value in values.items():
                row[f"{metric}_m{code}"] = value
            row[f"{metric}_interaction"] = (
                values["11"] - values["10"] - values["01"] + values["00"]
            )
        rows.append(row)
    if invalid_values:
        _remove_generated_files(output, output_names)
        payload = {
            "status": "BLOCKED",
            "reason": "The four-cell matrix contains non-finite values.",
            "invalid_dataset_metrics": invalid_values,
            "required_cells": cells,
        }
        save_json(payload, blocker_path)
        return payload

    fieldnames = (
        "dataset",
        "dataset_type",
        "auc_m00",
        "auc_m10",
        "auc_m01",
        "auc_m11",
        "auc_interaction",
        "ap_m00",
        "ap_m10",
        "ap_m01",
        "ap_m11",
        "ap_interaction",
    )
    write_rows(rows, output / "interaction_20datasets.csv", fieldnames=fieldnames)
    (output / "interaction_20datasets.tex").write_text(
        _paper_interaction_latex(rows), encoding="utf-8"
    )
    (output / "interaction_summary.tex").write_text(
        _paper_interaction_summary_latex(rows), encoding="utf-8"
    )
    if blocker_path.is_file():
        blocker_path.unlink()
    return {
        "status": "PASS",
        "dataset_count": len(rows),
        "required_cells": cells,
        "outputs": list(output_names),
        "interpretation": (
            "Positive, zero, and negative contrasts describe super-additive, "
            "additive, and sub-additive observational differences; they are not "
            "causal effects."
        ),
    }


def _write_interaction_outputs(
    output: Path,
    by_variant: Mapping[str, Mapping[str, Mapping[str, object]]],
    aligned: Sequence[str],
    names: Sequence[str],
) -> Dict[str, object]:
    output_names = (
        "interaction_effects.csv",
        "interaction_effect_summary.csv",
        "interaction_pairwise.csv",
        "interaction_cell_metadata.csv",
    )
    active = [
        family
        for family in INTERACTION_FAMILIES
        if all(
            family[cell] in names
            for cell in ("cell_00", "cell_10", "cell_01", "cell_11")
        )
    ]
    paper_manifest = _write_paper_interaction_outputs(output, by_variant, names)
    if not active:
        _remove_generated_files(output, output_names)
        return {
            "family_count": 0,
            "effect_row_count": 0,
            "paper_structure_evidence": paper_manifest,
        }

    effect_rows: List[Dict[str, object]] = []
    for family in active:
        for dataset in aligned:
            cells = {
                code: by_variant[family[f"cell_{code}"]][dataset]
                for code in ("00", "10", "01", "11")
            }
            row: Dict[str, object] = {
                "dataset": dataset,
                "dataset_type": cells["11"].get("dataset_type", "unknown"),
                "family": family["family"],
                "mechanism_a": family["mechanism_a"],
                "mechanism_b": family["mechanism_b"],
            }
            for metric in ("auc", "ap"):
                scores = {code: float(cells[code][metric]) for code in cells}
                for code, score in scores.items():
                    row[f"{metric}_{code}"] = score
                row[f"{metric}_effect_a_when_b0"] = scores["10"] - scores["00"]
                row[f"{metric}_effect_a_when_b1"] = scores["11"] - scores["01"]
                row[f"{metric}_effect_b_when_a0"] = scores["01"] - scores["00"]
                row[f"{metric}_effect_b_when_a1"] = scores["11"] - scores["10"]
                row[f"{metric}_interaction"] = (
                    scores["11"] - scores["10"] - scores["01"] + scores["00"]
                )
            for field in ABLATION_DIAGNOSTIC_FIELDS:
                values = {code: _row_number(cells[code], field) for code in cells}
                row[f"{field}_interaction"] = (
                    values["11"] - values["10"] - values["01"] + values["00"]
                )
            effect_rows.append(row)
    write_rows(effect_rows, output / "interaction_effects.csv")

    tolerance = 1e-12
    summary: List[Dict[str, object]] = []
    for family in active:
        family_rows = [
            row for row in effect_rows if row["family"] == family["family"]
        ]
        dataset_types = [
            "all",
            *sorted({str(row["dataset_type"]) for row in family_rows}),
        ]
        for dataset_type in dataset_types:
            selected = (
                family_rows
                if dataset_type == "all"
                else [row for row in family_rows if row["dataset_type"] == dataset_type]
            )
            item: Dict[str, object] = {
                "family": family["family"],
                "mechanism_a": family["mechanism_a"],
                "mechanism_b": family["mechanism_b"],
                "dataset_type": dataset_type,
                "dataset_count": len(selected),
            }
            for metric in ("auc", "ap"):
                values = np.asarray(
                    [float(row[f"{metric}_interaction"]) for row in selected]
                )
                item[f"mean_{metric}_interaction"] = float(values.mean())
                item[f"median_{metric}_interaction"] = float(np.median(values))
                item[f"{metric}_positive_interactions"] = int(
                    np.sum(values > tolerance)
                )
                item[f"{metric}_zero_interactions"] = int(
                    np.sum(np.abs(values) <= tolerance)
                )
                item[f"{metric}_negative_interactions"] = int(
                    np.sum(values < -tolerance)
                )
                for effect in (
                    "effect_a_when_b0",
                    "effect_a_when_b1",
                    "effect_b_when_a0",
                    "effect_b_when_a1",
                ):
                    item[f"mean_{metric}_{effect}"] = _finite_mean(
                        [float(row[f"{metric}_{effect}"]) for row in selected]
                    )
            for field in ABLATION_DIAGNOSTIC_FIELDS:
                item[f"mean_{field}_interaction"] = _finite_mean(
                    [float(row[f"{field}_interaction"]) for row in selected]
                )
            summary.append(item)
    write_rows(summary, output / "interaction_effect_summary.csv")

    pairwise: List[Dict[str, object]] = []
    if len(aligned) >= 2:
        for metric in ("auc", "ap"):
            metric_rows: List[Dict[str, object]] = []
            for family in active:
                values = np.asarray(
                    [
                        float(row[f"{metric}_interaction"])
                        for row in effect_rows
                        if row["family"] == family["family"]
                    ],
                    dtype=np.float64,
                )
                pair = np.column_stack((values, np.zeros_like(values)))
                item = dict(
                    statistical_comparison(
                        pair, ["observed_interaction", "additive_null"]
                    )["wilcoxon_holm"][0]
                )
                item.update(
                    {
                        "metric": metric,
                        "family": family["family"],
                        "mechanism_a": family["mechanism_a"],
                        "mechanism_b": family["mechanism_b"],
                    }
                )
                metric_rows.append(item)
            _holm_adjust_rows(metric_rows)
            pairwise.extend(metric_rows)
    write_rows(pairwise, output / "interaction_pairwise.csv")

    write_rows(
        [
            {
                "family": family["family"],
                "mechanism_a": family["mechanism_a"],
                "mechanism_b": family["mechanism_b"],
                "cell": code,
                "a_enabled": code[0] == "1",
                "b_enabled": code[1] == "1",
                "variant": family[f"cell_{code}"],
            }
            for family in active
            for code in ("00", "10", "01", "11")
        ],
        output / "interaction_cell_metadata.csv",
    )
    return {
        "family_count": len(active),
        "effect_row_count": len(effect_rows),
        "paper_structure_evidence": paper_manifest,
    }


def run_ablation(args: argparse.Namespace) -> None:
    variants = _resolve_ablation_variants(args.ablation_variants)
    variant_names = [name for name, _ in variants]
    if (
        args.ablation_protocol != "fixed"
        and _ablation_requires_fixed_protocol(variant_names)
    ):
        raise ValueError(
            "Cumulative increments and interaction contrasts require "
            "--ablation-protocol fixed so every cell uses the same configuration."
        )
    run_args = copy.copy(args)
    if args.ablation_protocol == "fixed":
        run_args.deltas = str(args.fixed_delta)
        run_args.mus = str(args.fixed_mu)
        run_args.view_grid = str(args.max_views)
    rows, failures, _selection_metadata = _run_grid(
        run_args, variants, experiment_name="ablation"
    )
    output = Path(args.output_dir)
    numeric = _as_numeric_rows(rows)
    selected_auc: List[Dict[str, object]] = []
    selected_ap_diagnostic: List[Dict[str, object]] = []
    for variant, _ in variants:
        variant_rows = [row for row in numeric if row["variant"] == variant]
        auc_rows = select_best_per_dataset(variant_rows, "auc")
        ap_rows = select_best_per_dataset(variant_rows, "ap")
        if args.ablation_protocol == "fixed":
            for row in [*auc_rows, *ap_rows]:
                row["selection_metric"] = "fixed"
                row["oracle_selection"] = False
        selected_auc.extend(auc_rows)
        selected_ap_diagnostic.extend(ap_rows)
    write_rows(selected_auc, output / "ablation_primary_results.csv")
    if args.ablation_protocol == "fixed":
        write_rows(selected_auc, output / "ablation_fixed_results.csv")
        for stale_name in (
            "ablation_best.csv",
            "ablation_selected_by_auc.csv",
            "ablation_best_ap_oracle.csv",
        ):
            stale_path = output / stale_name
            if stale_path.is_file():
                stale_path.unlink()
    else:
        fixed_path = output / "ablation_fixed_results.csv"
        if fixed_path.is_file():
            fixed_path.unlink()
        write_rows(selected_auc, output / "ablation_best.csv")
        write_rows(selected_auc, output / "ablation_selected_by_auc.csv")
        write_rows(
            selected_ap_diagnostic,
            output / "ablation_best_ap_oracle.csv",
        )

    by_variant = {
        variant: {
            str(row["dataset"]): row
            for row in selected_auc
            if row["variant"] == variant
        }
        for variant, _ in variants
    }
    dataset_sets = [set(items) for items in by_variant.values()]
    aligned = sorted(set.intersection(*dataset_sets)) if dataset_sets else []
    auc_ranks: Dict[str, List[float]] = {variant: [] for variant, _ in variants}
    ap_ranks: Dict[str, List[float]] = {variant: [] for variant, _ in variants}
    names = [variant for variant, _ in variants]
    for name in aligned:
        auc_values = np.asarray(
            [float(by_variant[variant][name]["auc"]) for variant in names]
        )
        ap_values = np.asarray(
            [float(by_variant[variant][name]["ap"]) for variant in names]
        )
        for variant, rank in zip(names, _descending_average_ranks(auc_values)):
            auc_ranks[variant].append(float(rank))
        for variant, rank in zip(names, _descending_average_ranks(ap_values)):
            ap_ranks[variant].append(float(rank))

    summary: List[Dict[str, object]] = []
    for variant in names:
        complete_rows = [by_variant[variant][name] for name in aligned]
        summary.append(
            {
                "variant": variant,
                "aligned_dataset_count": len(complete_rows),
                "available_dataset_count": len(by_variant[variant]),
                "mean_auc": (
                    float(np.mean([float(row["auc"]) for row in complete_rows]))
                    if complete_rows
                    else float("nan")
                ),
                "mean_ap_at_auc_selected_config": (
                    float(np.mean([float(row["ap"]) for row in complete_rows]))
                    if complete_rows
                    else float("nan")
                ),
                "mean_auc_rank": (
                    float(np.mean(auc_ranks[variant]))
                    if auc_ranks[variant]
                    else float("nan")
                ),
                "mean_ap_rank": (
                    float(np.mean(ap_ranks[variant]))
                    if ap_ranks[variant]
                    else float("nan")
                ),
            }
        )
    write_rows(summary, output / "ablation_summary.csv")
    auc_wide = [
        {
            "dataset": dataset,
            **{
                variant: float(by_variant[variant][dataset]["auc"])
                for variant in names
            },
        }
        for dataset in aligned
    ]
    ap_wide = [
        {
            "dataset": dataset,
            **{
                variant: float(by_variant[variant][dataset]["ap"])
                for variant in names
            },
        }
        for dataset in aligned
    ]
    write_rows(auc_wide, output / "ablation_auc_wide.csv")
    write_rows(ap_wide, output / "ablation_ap_wide.csv")
    reference = (
        "full_model" if "full_model" in by_variant else REVISED_FINAL_VARIANT
    )
    effect_rows: List[Dict[str, object]] = []
    if reference in by_variant:
        for variant in names:
            if variant == reference:
                continue
            metadata = ABLATION_METADATA[variant]
            for dataset in aligned:
                full_row = by_variant[reference][dataset]
                variant_row = by_variant[variant][dataset]
                is_removal = metadata["group"] in {"leave_one_out", "evidence"}
                effect_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_type": full_row.get("dataset_type", "unknown"),
                        "variant": variant,
                        "group": metadata["group"],
                        "mechanism": metadata["mechanism"],
                        "ablated_mechanism": (
                            metadata["mechanism"] if is_removal else ""
                        ),
                        "control": metadata["control"],
                        "full_auc": float(full_row["auc"]),
                        "variant_auc": float(variant_row["auc"]),
                        "auc_contribution_full_minus_variant": float(full_row["auc"])
                        - float(variant_row["auc"]),
                        "full_ap": float(full_row["ap"]),
                        "variant_ap": float(variant_row["ap"]),
                        "ap_contribution_full_minus_variant": float(full_row["ap"])
                        - float(variant_row["ap"]),
                    }
                )
    write_rows(effect_rows, output / "ablation_effects.csv")

    effect_summary: List[Dict[str, object]] = []
    tolerance = 1e-12
    for variant in names:
        if variant == reference:
            continue
        variant_effects = [row for row in effect_rows if row["variant"] == variant]
        dataset_types = [
            "all",
            *sorted({str(row["dataset_type"]) for row in variant_effects}),
        ]
        for kind in dataset_types:
            selected = (
                variant_effects
                if kind == "all"
                else [row for row in variant_effects if row["dataset_type"] == kind]
            )
            auc_effect = np.asarray(
                [float(row["auc_contribution_full_minus_variant"]) for row in selected]
            )
            ap_effect = np.asarray(
                [float(row["ap_contribution_full_minus_variant"]) for row in selected]
            )
            effect_summary.append(
                {
                    "variant": variant,
                    "group": ABLATION_METADATA[variant]["group"],
                    "mechanism": ABLATION_METADATA[variant]["mechanism"],
                    "ablated_mechanism": (
                        ABLATION_METADATA[variant]["mechanism"]
                        if ABLATION_METADATA[variant]["group"]
                        in {"leave_one_out", "evidence"}
                        else ""
                    ),
                    "dataset_type": kind,
                    "dataset_count": len(selected),
                    "mean_auc_contribution": float(auc_effect.mean()),
                    "median_auc_contribution": float(np.median(auc_effect)),
                    "auc_full_wins": int(np.sum(auc_effect > tolerance)),
                    "auc_ties": int(np.sum(np.abs(auc_effect) <= tolerance)),
                    "auc_full_losses": int(np.sum(auc_effect < -tolerance)),
                    "mean_ap_contribution": float(ap_effect.mean()),
                    "median_ap_contribution": float(np.median(ap_effect)),
                    "ap_full_wins": int(np.sum(ap_effect > tolerance)),
                    "ap_ties": int(np.sum(np.abs(ap_effect) <= tolerance)),
                    "ap_full_losses": int(np.sum(ap_effect < -tolerance)),
                }
            )
    write_rows(effect_summary, output / "ablation_effect_summary.csv")

    cumulative_manifest = _write_cumulative_outputs(
        output, by_variant, aligned, names
    )
    interaction_manifest = _write_interaction_outputs(
        output, by_variant, aligned, names
    )

    reference_pairwise: List[Dict[str, object]] = []
    if len(aligned) >= 2 and reference in names:
        for metric, wide in (("auc", auc_wide), ("ap", ap_wide)):
            metric_rows: List[Dict[str, object]] = []
            for variant in names:
                if variant == reference:
                    continue
                pair = np.asarray(
                    [
                        [float(row[reference]), float(row[variant])]
                        for row in wide
                    ],
                    dtype=np.float64,
                )
                item = dict(
                    statistical_comparison(pair, [reference, variant])["wilcoxon_holm"][0]
                )
                item["metric"] = metric
                item["mechanism"] = ABLATION_METADATA[variant]["mechanism"]
                item["ablated_mechanism"] = (
                    ABLATION_METADATA[variant]["mechanism"]
                    if ABLATION_METADATA[variant]["group"]
                    in {"leave_one_out", "evidence"}
                    else ""
                )
                metric_rows.append(item)
            _holm_adjust_rows(metric_rows)
            reference_pairwise.extend(metric_rows)
    write_rows(reference_pairwise, output / "ablation_reference_pairwise.csv")

    write_rows(
        [
            {"variant": name, **ABLATION_METADATA[name]}
            for name in names
        ],
        output / "ablation_variant_metadata.csv",
    )
    if len(aligned) >= 2 and len(names) >= 2:
        for metric, wide in (("auc", auc_wide), ("ap", ap_wide)):
            matrix = np.asarray(
                [[float(row[name]) for name in names] for row in wide],
                dtype=np.float64,
            )
            result = statistical_comparison(matrix, names)
            result["metric"] = metric
            result["datasets"] = aligned
            save_json(result, output / f"ablation_{metric}_statistics.json")
    save_json(
        {
            "aligned_datasets": aligned,
            "variant_dataset_counts": {
                variant: len(items) for variant, items in by_variant.items()
            },
            "rule": (
                "Aggregates and ranks use only datasets complete for every requested "
                "variant; all variants use the same fixed configuration."
                if args.ablation_protocol == "fixed"
                else "Aggregates and ranks use only datasets complete for every requested "
                "variant; AP comes from each variant's AUC-selected row."
            ),
            "protocol": args.ablation_protocol,
            "reference_variant": reference,
            "cumulative_outputs": cumulative_manifest,
            "interaction_outputs": interaction_manifest,
        },
        output / "ablation_alignment.json",
    )
    save_json(
        system_manifest(
            {
                "experiment": "ablation",
                "arguments": _args_payload(args),
                "effective_grid": {
                    "deltas": run_args.deltas,
                    "mus": run_args.mus,
                    "pure_nominal_fixed_delta": args.pure_nominal_fixed_delta,
                    "pure_nominal_fixed_mu": args.pure_nominal_fixed_mu,
                    "view_grid": run_args.view_grid,
                },
                "variants": names,
                "protocol": args.ablation_protocol,
                "reference_variant": reference,
                "primary_effect": (
                    "Adjacent cumulative increment and targeted two-factor "
                    "interaction; full_model-minus-variant is supplementary."
                ),
                "cumulative_outputs": cumulative_manifest,
                "interaction_outputs": interaction_manifest,
                "oracle_selection": args.ablation_protocol == "label_oracle",
                "ap_reporting_rule": (
                    "AP at the common fixed configuration."
                    if args.ablation_protocol == "fixed"
                    else "AP from the AUC-selected configuration."
                ),
            }
        ),
        output / "run_config.json",
    )
    print(
        f"[ablation complete] rows={len(rows)} failures={len(failures)} "
        f"aligned={len(aligned)}",
        flush=True,
    )


def _gain_complexity_distance(
    raw_row: Mapping[str, object],
    normalized_row: Mapping[str, object],
    metric: str,
) -> Tuple[float, float, float]:
    raw_nodes = _row_number(raw_row, "tree_mean_nodes")
    normalized_nodes = _row_number(normalized_row, "tree_mean_nodes")
    raw_depth = _row_number(raw_row, "tree_mean_depth")
    normalized_depth = _row_number(normalized_row, "tree_mean_depth")
    node_distance = abs(math.log1p(raw_nodes) - math.log1p(normalized_nodes))
    depth_distance = abs(raw_depth - normalized_depth) / max(
        raw_depth, normalized_depth, 1.0
    )
    if metric == "nodes":
        distance = node_distance
    elif metric == "depth":
        distance = depth_distance
    else:
        distance = node_distance + depth_distance
    return float(distance), float(node_distance), float(depth_distance)


def _write_gain_complexity_outputs(
    output: Path,
    rows: Sequence[Mapping[str, object]],
    metric: str,
    target_normalized_mu: Optional[float] = None,
    quality_thresholds: Sequence[float] = (0.3, 0.5),
) -> List[Dict[str, object]]:
    thresholds = sorted({float(value) for value in quality_thresholds})
    if any(value < 0.0 for value in thresholds):
        raise ValueError("Complexity-match quality thresholds must be non-negative.")
    datasets = sorted({str(row["dataset"]) for row in rows})
    pair_grid: List[Dict[str, object]] = []
    matches: List[Dict[str, object]] = []
    for dataset in datasets:
        raw_rows = [
            row
            for row in rows
            if row["dataset"] == dataset and row["variant"] == "raw_gain"
        ]
        normalized_rows = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["variant"] == "normalized_gain"
        ]
        candidates: List[Dict[str, object]] = []
        for normalized_row in normalized_rows:
            normalized_candidates: List[Dict[str, object]] = []
            for raw_row in raw_rows:
                distance, node_distance, depth_distance = (
                    _gain_complexity_distance(raw_row, normalized_row, metric)
                )
                item: Dict[str, object] = {
                    "dataset": dataset,
                    "dataset_type": normalized_row.get(
                        "dataset_type", raw_row.get("dataset_type", "unknown")
                    ),
                    "match_metric": metric,
                    "structural_distance": distance,
                    "log_node_distance": node_distance,
                    "relative_depth_distance": depth_distance,
                    "raw_config_id": raw_row.get("config_id", ""),
                    "normalized_config_id": normalized_row.get("config_id", ""),
                    "raw_mu": float(raw_row["mu"]),
                    "normalized_mu": float(normalized_row["mu"]),
                    "raw_mean_nodes": _row_number(raw_row, "tree_mean_nodes"),
                    "normalized_mean_nodes": _row_number(
                        normalized_row, "tree_mean_nodes"
                    ),
                    "raw_mean_depth": _row_number(raw_row, "tree_mean_depth"),
                    "normalized_mean_depth": _row_number(
                        normalized_row, "tree_mean_depth"
                    ),
                    "raw_auc": float(raw_row["auc"]),
                    "normalized_auc": float(normalized_row["auc"]),
                    "auc_normalized_minus_raw": float(normalized_row["auc"])
                    - float(raw_row["auc"]),
                    "raw_ap": float(raw_row["ap"]),
                    "normalized_ap": float(normalized_row["ap"]),
                    "ap_normalized_minus_raw": float(normalized_row["ap"])
                    - float(raw_row["ap"]),
                    "raw_elapsed_seconds": _row_number(
                        raw_row, "elapsed_seconds"
                    ),
                    "normalized_elapsed_seconds": _row_number(
                        normalized_row, "elapsed_seconds"
                    ),
                    "selected_match": False,
                    "primary_frozen_match": False,
                }
                candidates.append(item)
                normalized_candidates.append(item)
            if not normalized_candidates:
                continue
            selected = min(
                normalized_candidates,
                key=lambda item: (
                    float(item["structural_distance"]),
                    abs(float(item["raw_mu"]) - float(item["normalized_mu"])),
                    float(item["raw_mu"]),
                ),
            )
            selected["selected_match"] = True
            if target_normalized_mu is not None and math.isclose(
                float(selected["normalized_mu"]),
                float(target_normalized_mu),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                selected["primary_frozen_match"] = True
            matches.append(dict(selected))
        if not candidates:
            continue
        pair_grid.extend(candidates)
    write_rows(pair_grid, output / "gain_complexity_pair_grid.csv")
    write_rows(matches, output / "gain_complexity_matches.csv")
    primary_matches = [row for row in matches if row["primary_frozen_match"]]
    if target_normalized_mu is not None and len(primary_matches) != len(datasets):
        missing = sorted(
            set(datasets) - {str(row["dataset"]) for row in primary_matches}
        )
        raise ValueError(
            f"The mu grid must contain fixed_mu={target_normalized_mu} for every "
            f"dataset; missing primary matches for {missing}."
        )
    write_rows(
        primary_matches, output / "gain_complexity_primary_matches.csv"
    )

    tolerance = 1e-12
    summary: List[Dict[str, object]] = []
    dataset_types = [
        "all", *sorted({str(row["dataset_type"]) for row in primary_matches})
    ]
    for dataset_type in dataset_types:
        selected = (
            primary_matches
            if dataset_type == "all"
            else [
                row
                for row in primary_matches
                if row["dataset_type"] == dataset_type
            ]
        )
        if not selected:
            continue
        auc = np.asarray(
            [float(row["auc_normalized_minus_raw"]) for row in selected]
        )
        ap = np.asarray(
            [float(row["ap_normalized_minus_raw"]) for row in selected]
        )
        summary.append(
            {
                "dataset_type": dataset_type,
                "dataset_count": len(selected),
                "match_metric": metric,
                "mean_structural_distance": float(
                    np.mean(
                        [float(row["structural_distance"]) for row in selected]
                    )
                ),
                "mean_auc_normalized_minus_raw": float(auc.mean()),
                "median_auc_normalized_minus_raw": float(np.median(auc)),
                "auc_normalized_wins": int(np.sum(auc > tolerance)),
                "auc_ties": int(np.sum(np.abs(auc) <= tolerance)),
                "auc_normalized_losses": int(np.sum(auc < -tolerance)),
                "mean_ap_normalized_minus_raw": float(ap.mean()),
                "median_ap_normalized_minus_raw": float(np.median(ap)),
                "ap_normalized_wins": int(np.sum(ap > tolerance)),
                "ap_ties": int(np.sum(np.abs(ap) <= tolerance)),
                "ap_normalized_losses": int(np.sum(ap < -tolerance)),
            }
        )
    write_rows(summary, output / "gain_complexity_summary.csv")

    quality_rows: List[Dict[str, object]] = []
    pairwise_rows: List[Dict[str, object]] = []
    for dataset_type in dataset_types:
        type_rows = (
            primary_matches
            if dataset_type == "all"
            else [
                row
                for row in primary_matches
                if row["dataset_type"] == dataset_type
            ]
        )
        subsets: List[Tuple[str, Optional[float], List[Dict[str, object]]]] = [
            ("all_primary", None, list(type_rows))
        ]
        for threshold in thresholds:
            qualified = [
                row
                for row in type_rows
                if float(row["structural_distance"]) <= threshold
            ]
            subsets.append(("distance_threshold", threshold, qualified))
            item: Dict[str, object] = {
                "dataset_type": dataset_type,
                "dataset_count": len(type_rows),
                "match_metric": metric,
                "distance_threshold": threshold,
                "qualified_count": len(qualified),
                "qualified_fraction": (
                    float(len(qualified) / len(type_rows)) if type_rows else float("nan")
                ),
                "mean_qualified_structural_distance": (
                    _finite_mean(
                        [float(row["structural_distance"]) for row in qualified]
                    )
                    if qualified
                    else float("nan")
                ),
            }
            for score_name in ("auc", "ap"):
                differences = np.asarray(
                    [
                        float(row[f"{score_name}_normalized_minus_raw"])
                        for row in qualified
                    ],
                    dtype=np.float64,
                )
                item[f"mean_{score_name}_normalized_minus_raw"] = (
                    float(differences.mean()) if len(differences) else float("nan")
                )
                item[f"median_{score_name}_normalized_minus_raw"] = (
                    float(np.median(differences))
                    if len(differences)
                    else float("nan")
                )
                item[f"{score_name}_normalized_wins"] = int(
                    np.sum(differences > tolerance)
                )
                item[f"{score_name}_ties"] = int(
                    np.sum(np.abs(differences) <= tolerance)
                )
                item[f"{score_name}_normalized_losses"] = int(
                    np.sum(differences < -tolerance)
                )
            quality_rows.append(item)

        for subset_name, threshold, selected in subsets:
            if len(selected) < 2:
                continue
            for score_name in ("auc", "ap"):
                pair = np.asarray(
                    [
                        [
                            float(row[f"normalized_{score_name}"]),
                            float(row[f"raw_{score_name}"]),
                        ]
                        for row in selected
                    ],
                    dtype=np.float64,
                )
                comparison = dict(
                    statistical_comparison(
                        pair, ["normalized_gain", "raw_gain"]
                    )["wilcoxon_holm"][0]
                )
                comparison.update(
                    {
                        "metric": score_name,
                        "dataset_type": dataset_type,
                        "subset": subset_name,
                        "distance_threshold": (
                            "" if threshold is None else threshold
                        ),
                        "dataset_count": len(selected),
                        "diagnostic_subset": threshold is not None,
                    }
                )
                pairwise_rows.append(comparison)
    write_rows(
        quality_rows, output / "gain_complexity_match_quality.csv"
    )
    write_rows(pairwise_rows, output / "gain_complexity_pairwise.csv")
    return matches


def run_gain_complexity(args: argparse.Namespace) -> None:
    run_args = copy.copy(args)
    run_args.deltas = str(args.fixed_delta)
    run_args.view_grid = str(args.max_views)
    raw_mus = str(args.raw_mus or args.mus)
    normalized_mus = str(args.normalized_mus or args.mus)
    raw_mu_values = parse_number_grid(raw_mus)
    normalized_mu_values = parse_number_grid(normalized_mus)
    quality_thresholds = parse_number_grid(args.complexity_match_thresholds)
    if not raw_mu_values:
        raise ValueError("--raw-mus must contain at least one value.")
    if not normalized_mu_values:
        raise ValueError("--normalized-mus must contain at least one value.")
    if not quality_thresholds:
        raise ValueError("--complexity-match-thresholds must not be empty.")
    if not any(
        math.isclose(value, float(args.fixed_mu), rel_tol=0.0, abs_tol=1e-12)
        for value in normalized_mu_values
    ):
        raise ValueError(
            f"--normalized-mus must contain fixed_mu={args.fixed_mu}."
        )
    variant_mu_grids = {
        "raw_gain": raw_mus,
        "normalized_gain": normalized_mus,
    }
    rows, failures, _selection_metadata = _run_grid(
        run_args,
        GAIN_COMPLEXITY_VARIANTS,
        experiment_name="gain-complexity",
        variant_mu_grids=variant_mu_grids,
    )
    output = Path(args.output_dir)
    numeric = _as_numeric_rows(rows)
    write_rows(numeric, output / "gain_complexity_results.csv")
    matches = _write_gain_complexity_outputs(
        output,
        numeric,
        args.complexity_match_metric,
        target_normalized_mu=args.fixed_mu,
        quality_thresholds=quality_thresholds,
    )
    save_json(
        system_manifest(
            {
                "experiment": "gain-complexity",
                "arguments": _args_payload(args),
                "effective_grid": {
                    "deltas": run_args.deltas,
                    "raw_mus": raw_mus,
                    "normalized_mus": normalized_mus,
                    "view_grid": run_args.view_grid,
                },
                "variants": [name for name, _ in GAIN_COMPLEXITY_VARIANTS],
                "matching_rule": (
                    "For every normalized mu and dataset, select the raw-mu run "
                    "with the smallest requested structural distance. The primary "
                    "paper comparison fixes normalized mu to fixed_mu and matches "
                    "only the raw side. Labels, AUC, and AP are excluded."
                ),
                "primary_normalized_mu": args.fixed_mu,
                "match_metric": args.complexity_match_metric,
                "match_quality_thresholds": quality_thresholds,
                "oracle_selection": False,
                "ap_reporting_rule": (
                    "AP is reported on the same label-free structurally matched pair."
                ),
            }
        ),
        output / "run_config.json",
    )
    print(
        f"[gain-complexity complete] rows={len(rows)} matches={len(matches)} "
        f"failures={len(failures)}",
        flush=True,
    )


def _write_selected_run_config(
    args: argparse.Namespace, experiment: str, extra: Optional[Mapping[str, object]] = None
) -> None:
    payload: Dict[str, object] = {
        "experiment": experiment,
        "arguments": _args_payload(args),
        "selection": "AUC-selected configuration when --selected-configs is supplied.",
    }
    if extra:
        payload.update(extra)
    save_json(system_manifest(payload), Path(args.output_dir) / "run_config.json")


def run_complementarity(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(args)
    lookup = _selected_config_lookup(args.selected_configs)
    rows: List[Dict[str, object]] = []
    details: Dict[str, object] = {}
    failures: List[Dict[str, object]] = []
    for index, bundle in enumerate(bundles, start=1):
        config = _configuration_for_bundle(args, bundle, lookup)
        try:
            row, model, scores = _fit_and_record(
                bundle, args, config, "relation", {"enabled_components": COMPONENTS}
            )
            component_scores = {
                name: model.fused_scores_3d_[:, position]
                for position, name in enumerate(COMPONENTS)
            }
            detail = complementarity_record(
                bundle.y,
                component_scores,
                top_fraction=float(args.top_fraction),
                top_k=(
                    int(bundle.y.sum())
                    if args.top_k_rule == "anomaly_count"
                    else None
                ),
            )
            details[bundle.name] = detail
            for key, value in detail.items():
                row[f"complementarity_{key}"] = (
                    value
                    if isinstance(value, (str, int, float, bool)) or value is None
                    else json.dumps(value, sort_keys=True, default=str)
                )
            rows.append(row)
            if args.save_scores:
                _save_score_archive(output, bundle, row, model, scores)
            print(
                f"[{index}/{len(bundles)}] complementarity {bundle.name}",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {"dataset": bundle.name, "source": str(bundle.path), "error": repr(error)}
            )
            if args.fail_on_error:
                raise
    write_rows(rows, output / "complementarity_results.csv")
    write_rows(
        failures,
        output / "failures.csv",
        fieldnames=("dataset", "source", "error"),
    )
    save_json(details, output / "complementarity_details.json")
    _write_selected_run_config(args, "complementarity")


def _sensitivity_parameter_values(
    args: argparse.Namespace,
) -> Dict[str, List[float | int]]:
    values: Dict[str, List[float | int]] = {
        "delta": [float(item) for item in parse_number_grid(args.sensitivity_deltas)],
        "mu": [float(item) for item in parse_number_grid(args.sensitivity_mus)],
        "max_views": [
            int(item) for item in parse_number_grid(args.sensitivity_views, int)
        ],
        "forest_size": [
            int(item) for item in parse_number_grid(args.sensitivity_forest_sizes, int)
        ],
        "min_child_size": [
            int(item) for item in parse_number_grid(args.sensitivity_min_child, int)
        ],
        "supplemental_view_count": [
            int(item)
            for item in parse_number_grid(args.sensitivity_supplemental_views, int)
        ],
    }
    if args.quick:
        return {key: items[:2] for key, items in values.items()}
    return values


def _sensitivity_axis_limits(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    min_span: float,
    padding: float,
) -> Tuple[float, float]:
    if not 0.0 < float(min_span) <= 1.0:
        raise ValueError("Sensitivity minimum y-axis span must be in (0, 1].")
    if not 0.0 <= float(padding) < 1.0:
        raise ValueError("Sensitivity y-axis padding must be in [0, 1).")
    values: List[float] = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return (0.0, 1.0)

    observed_lower = min(values)
    observed_upper = max(values)
    lower = max(0.0, observed_lower - float(padding))
    upper = min(1.0, observed_upper + float(padding))
    if upper - lower < float(min_span):
        midpoint = 0.5 * (observed_lower + observed_upper)
        half_span = 0.5 * float(min_span)
        lower = midpoint - half_span
        upper = midpoint + half_span
        if lower < 0.0:
            upper = min(1.0, upper - lower)
            lower = 0.0
        if upper > 1.0:
            lower = max(0.0, lower - (upper - 1.0))
            upper = 1.0
    return (float(lower), float(upper))


def _sensitivity_reference_value(
    args: argparse.Namespace,
    bundles: Sequence[DatasetBundle],
    parameter: str,
) -> Optional[float]:
    dataset_types = {bundle.dataset_type for bundle in bundles}
    if parameter == "delta":
        return float(args.fixed_delta) if dataset_types - {"nominal"} else None
    if parameter == "mu":
        candidates = {
            float(args.pure_nominal_fixed_mu)
            if dataset_type_name == "nominal"
            else float(args.fixed_mu)
            for dataset_type_name in dataset_types
        }
        return next(iter(candidates)) if len(candidates) == 1 else None
    if parameter == "max_views":
        return float(args.max_views)
    if parameter == "min_child_size":
        return float(args.min_child_size)
    if parameter == "supplemental_view_count":
        return float(args.supplemental_view_count)
    return None


def run_sensitivity(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(args)
    lookup = _selected_config_lookup(args.selected_configs)
    requested = _parse_csv_names(args.sensitivity_parameters)
    all_values = _sensitivity_parameter_values(args)
    parameters = list(all_values) if requested is None else requested
    unknown = sorted(set(parameters) - set(all_values))
    if unknown:
        raise ValueError(f"Unknown sensitivity parameters: {unknown}")
    for metric, min_span in (
        ("auc", args.sensitivity_auc_min_span),
        ("ap", args.sensitivity_ap_min_span),
    ):
        _sensitivity_axis_limits(
            [], metric, float(min_span), float(args.sensitivity_axis_padding)
        )
    result_path = output / "sensitivity_results.csv"
    rows: List[Dict[str, object]] = (
        [dict(row) for row in read_rows(result_path)]
        if bool(getattr(args, "resume", False)) and result_path.is_file()
        else []
    )
    completed = {
        (str(row.get("dataset")), str(row.get("parameter")), str(row.get("value")))
        for row in rows
    }
    failures: List[Dict[str, object]] = []
    for bundle in bundles:
        baseline = _configuration_for_bundle(args, bundle, lookup)
        for parameter in parameters:
            if parameter == "delta" and bundle.dataset_type == "nominal":
                if (bundle.name, parameter, "N/A") in completed:
                    continue
                rows.append(
                    {
                        "dataset": bundle.name,
                        "dataset_type": bundle.dataset_type,
                        "parameter": parameter,
                        "value": "N/A",
                        "status": "inactive",
                        "reason": "Equality relations make delta inactive.",
                    }
                )
                completed.add((bundle.name, parameter, "N/A"))
                continue
            for value in all_values[parameter]:
                if (bundle.name, parameter, str(value)) in completed:
                    continue
                config = dict(baseline)
                overrides: Dict[str, object] = {}
                if parameter in {
                    "delta",
                    "mu",
                    "max_views",
                }:
                    config[parameter] = value
                    if parameter == "max_views":
                        config["max_views"] = int(value)
                else:
                    if parameter == "forest_size":
                        config["max_views"] = int(value)
                        overrides.update(
                            {
                                "min_views": int(value),
                                "supplemental_view_count": int(value),
                            }
                        )
                    else:
                        overrides[parameter] = int(value)
                try:
                    row, _, _ = _fit_and_record(
                        bundle, args, config, "relation", overrides
                    )
                    row.update(
                        {"parameter": parameter, "value": value, "status": "ok"}
                    )
                    rows.append(row)
                    completed.add((bundle.name, parameter, str(value)))
                except Exception as error:
                    failures.append(
                        {
                            "dataset": bundle.name,
                            "parameter": parameter,
                            "value": value,
                            "error": repr(error),
                        }
                    )
                    if args.fail_on_error:
                        raise
                write_rows(rows, result_path)
    write_rows(rows, result_path)
    write_rows(
        failures,
        output / "failures.csv",
        fieldnames=("dataset", "parameter", "value", "error"),
    )
    sensitivity_summary: List[Dict[str, object]] = []
    sensitivity_alignment: Dict[str, object] = {}
    for parameter in parameters:
        parameter_rows = [
            row
            for row in rows
            if row.get("parameter") == parameter and row.get("status") == "ok"
        ]
        by_value: Dict[str, Dict[str, Mapping[str, object]]] = {}
        for row in parameter_rows:
            by_value.setdefault(str(row["value"]), {})[str(row["dataset"])] = row
        dataset_sets = [set(items) for items in by_value.values()]
        aligned = sorted(set.intersection(*dataset_sets)) if dataset_sets else []
        sensitivity_alignment[parameter] = {
            "aligned_datasets": aligned,
            "value_dataset_counts": {
                value: len(items) for value, items in by_value.items()
            },
        }
        for value in sorted(by_value, key=lambda item: float(item)):
            selected = [by_value[value][dataset] for dataset in aligned]
            auc_values = np.asarray(
                [float(row["auc"]) for row in selected], dtype=np.float64
            )
            ap_values = np.asarray(
                [float(row["ap"]) for row in selected], dtype=np.float64
            )
            views_used = np.asarray(
                [float(row.get("views_used", float("nan"))) for row in selected],
                dtype=np.float64,
            )
            views_used = views_used[np.isfinite(views_used)]
            if parameter == "max_views" and len(views_used):
                at_requested_cap = int(
                    np.count_nonzero(np.isclose(views_used, float(value)))
                )
                cap_rate: object = float(at_requested_cap / len(views_used))
            else:
                at_requested_cap = ""
                cap_rate = ""
            sensitivity_summary.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "aligned_dataset_count": len(aligned),
                    "mean_auc": float(auc_values.mean()) if len(auc_values) else float("nan"),
                    "std_auc": (
                        float(auc_values.std(ddof=1)) if len(auc_values) > 1 else 0.0
                    ),
                    "mean_ap": float(ap_values.mean()) if len(ap_values) else float("nan"),
                    "std_ap": (
                        float(ap_values.std(ddof=1)) if len(ap_values) > 1 else 0.0
                    ),
                    "mean_views_used": (
                        float(views_used.mean()) if len(views_used) else float("nan")
                    ),
                    "min_views_used": (
                        float(views_used.min()) if len(views_used) else float("nan")
                    ),
                    "max_views_used": (
                        float(views_used.max()) if len(views_used) else float("nan")
                    ),
                    "datasets_at_requested_view_cap": at_requested_cap,
                    "view_cap_saturation_rate": cap_rate,
                }
            )
    write_rows(sensitivity_summary, output / "sensitivity_summary.csv")
    save_json(
        {
            "parameters": sensitivity_alignment,
            "rule": "Each parameter is summarized on the dataset intersection complete for every tested value.",
        },
        output / "sensitivity_alignment.json",
    )
    all_plot_rows = [
        row
        for row in rows
        if row.get("parameter") in parameters and row.get("status") == "ok"
    ]
    axis_limits = {
        "auc": _sensitivity_axis_limits(
            all_plot_rows,
            "auc",
            float(args.sensitivity_auc_min_span),
            float(args.sensitivity_axis_padding),
        ),
        "ap": _sensitivity_axis_limits(
            all_plot_rows,
            "ap",
            float(args.sensitivity_ap_min_span),
            float(args.sensitivity_axis_padding),
        ),
    }
    reference_values = {
        parameter: _sensitivity_reference_value(args, bundles, parameter)
        for parameter in parameters
    }
    save_json(
        {
            "axis_policy": (
                "All parameter panels in this run share one y-axis range per metric. "
                "Limits include every finite per-dataset observation, add fixed absolute "
                "padding, and enforce a minimum span without clipping to a favorable range."
            ),
            "axis_limits": {
                metric: {"lower": limits[0], "upper": limits[1]}
                for metric, limits in axis_limits.items()
            },
            "minimum_spans": {
                "auc": float(args.sensitivity_auc_min_span),
                "ap": float(args.sensitivity_ap_min_span),
            },
            "absolute_padding": float(args.sensitivity_axis_padding),
            "x_scales": {
                parameter: "log" if parameter == "mu" else "linear"
                for parameter in parameters
            },
            "frozen_reference_values": reference_values,
            "reference_marker": (
                "Reference settings are recorded for reproducibility but are not "
                "drawn in the panels; the figure does not identify a best value."
            ),
        },
        output / "sensitivity_plot_config.json",
    )
    for parameter in parameters:
        plot_rows = [
            row
            for row in rows
            if row.get("parameter") == parameter and row.get("status") == "ok"
        ]
        if not plot_rows:
            continue
        for metric in ("auc", "ap"):
            try:
                plot_sensitivity(
                    plot_rows,
                    "value",
                    metric,
                    output / f"sensitivity_{parameter}_{metric}.svg",
                    y_limits=axis_limits[metric],
                    x_scale="log" if parameter == "mu" else "linear",
                    reference_x=None,
                )
            except Exception as error:  # pragma: no cover - optional plotting backend
                save_json(
                    {"parameter": parameter, "metric": metric, "error": repr(error)},
                    output / f"sensitivity_{parameter}_{metric}_plot_error.json",
                )
    _write_selected_run_config(args, "sensitivity", {"parameters": parameters})


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _score_similarity(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, object]:
    difference = np.abs(np.asarray(reference) - np.asarray(candidate))
    left = _rankdata(reference)
    right = _rankdata(candidate)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "max_abs_difference": float(difference.max(initial=0.0)),
        "mean_abs_difference": float(difference.mean()),
        "spearman": float(np.dot(left, right) / denominator) if denominator else 1.0,
    }


def _relabel_nominal(
    X: np.ndarray, nominal_features: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    result = np.asarray(X, dtype=np.float64).copy()
    for feature in nominal_features:
        unique, inverse = np.unique(result[:, feature], return_inverse=True)
        replacement = rng.permutation(len(unique)).astype(np.float64) * 17.0 + 101.0
        result[:, feature] = replacement[inverse]
    return result


def _fit_transformed(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    config: Mapping[str, object],
) -> np.ndarray:
    model = RelationConsistentMFTAD(**_base_model_kwargs(args, bundle, config))
    return model.fit_predict(bundle.X)


def run_invariance(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(args)
    lookup = _selected_config_lookup(args.selected_configs)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for offset, bundle in enumerate(bundles):
        config = _configuration_for_bundle(args, bundle, lookup)
        rng = np.random.default_rng(int(args.seed) + offset)
        try:
            reference = _fit_transformed(args, bundle, config)
            permutation = rng.permutation(len(bundle.y))
            permuted_bundle = DatasetBundle(
                name=bundle.name,
                path=bundle.path,
                X=np.ascontiguousarray(bundle.X[permutation]),
                y=np.ascontiguousarray(bundle.y[permutation]),
                nominal_features=bundle.nominal_features,
                dataset_type=bundle.dataset_type,
                label_mapping=bundle.label_mapping,
            )
            permuted_scores = _fit_transformed(args, permuted_bundle, config)
            restored = np.empty_like(permuted_scores)
            restored[permutation] = permuted_scores

            feature_order = rng.permutation(bundle.X.shape[1])
            nominal_set = set(bundle.nominal_features)
            remapped_nominal = tuple(
                position
                for position, original in enumerate(feature_order)
                if int(original) in nominal_set
            )
            feature_bundle = DatasetBundle(
                name=bundle.name,
                path=bundle.path,
                X=np.ascontiguousarray(bundle.X[:, feature_order]),
                y=bundle.y,
                nominal_features=remapped_nominal,
                dataset_type=dataset_type(bundle.X.shape[1], remapped_nominal),
                label_mapping=bundle.label_mapping,
            )
            feature_scores = _fit_transformed(args, feature_bundle, config)

            transformed_scores = None
            if bundle.nominal_features:
                transformed_bundle = DatasetBundle(
                    name=bundle.name,
                    path=bundle.path,
                    X=_relabel_nominal(bundle.X, bundle.nominal_features, rng),
                    y=bundle.y,
                    nominal_features=bundle.nominal_features,
                    dataset_type=bundle.dataset_type,
                    label_mapping=bundle.label_mapping,
                )
                transformed_scores = _fit_transformed(args, transformed_bundle, config)

            candidates = [
                ("sample_order", restored),
                ("feature_order", feature_scores),
            ]
            if transformed_scores is not None:
                candidates.append(("nominal_relabeling", transformed_scores))
            for transformation, candidate in candidates:
                similarity = _score_similarity(reference, candidate)
                similarity.update(
                    {
                        "dataset": bundle.name,
                        "dataset_type": bundle.dataset_type,
                        "transformation": transformation,
                        "tolerance": float(args.invariance_tolerance),
                        "passed": float(similarity["max_abs_difference"])
                        <= float(args.invariance_tolerance),
                    }
                )
                rows.append(similarity)
        except Exception as error:
            failures.append(
                {"dataset": bundle.name, "source": str(bundle.path), "error": repr(error)}
            )
            if args.fail_on_error:
                raise
    write_rows(rows, output / "invariance_results.csv")
    write_rows(
        failures,
        output / "failures.csv",
        fieldnames=("dataset", "source", "error"),
    )
    _write_selected_run_config(args, "invariance")


def _interpretability_selections(
    scores: np.ndarray, labels: np.ndarray, top_k: int
) -> Tuple[List[Tuple[str, int]], Dict[int, int]]:
    """Select ranking diagnostics without defining a deployment threshold.

    Exactly ``min(top_k, n)`` score-only cases are returned. Stable mergesort
    makes the original sample index the deterministic tie-breaker. Label-based
    roles are extrema within the labeled anomaly/normal subsets and are
    explicitly post-hoc ranking diagnostics, not detected/undetected classes.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(values) != len(y) or len(values) == 0:
        raise ValueError("Interpretability scores and labels must align and be nonempty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Interpretability scores must be finite.")
    if top_k < 1:
        raise ValueError("explain_top_k must be positive.")
    score_order = np.argsort(-values, kind="mergesort")
    overall_rank = {
        int(sample): rank for rank, sample in enumerate(score_order, start=1)
    }
    selections: List[Tuple[str, int]] = [
        ("top_score_rank", int(sample))
        for sample in score_order[: min(int(top_k), len(values))]
    ]
    anomaly_indices = np.flatnonzero(y == 1)
    normal_indices = np.flatnonzero(y == 0)
    if len(anomaly_indices):
        selections.extend(
            (
                (
                    "highest_ranked_labeled_anomaly",
                    int(anomaly_indices[np.argmax(values[anomaly_indices])]),
                ),
                (
                    "lowest_ranked_labeled_anomaly",
                    int(anomaly_indices[np.argmin(values[anomaly_indices])]),
                ),
            )
        )
    if len(normal_indices):
        selections.extend(
            (
                (
                    "highest_ranked_labeled_normal",
                    int(normal_indices[np.argmax(values[normal_indices])]),
                ),
                (
                    "lowest_ranked_labeled_normal",
                    int(normal_indices[np.argmin(values[normal_indices])]),
                ),
            )
        )
    return selections, overall_rank


def run_interpretability(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(args)
    lookup = _selected_config_lookup(args.selected_configs)
    attribute_rows: List[Dict[str, object]] = []
    view_rows: List[Dict[str, object]] = []
    tree_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    sample_path_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}
    for bundle in bundles:
        config = _configuration_for_bundle(args, bundle, lookup)
        try:
            dataset_node_count = 0
            row, model, scores = _fit_and_record(
                bundle,
                args,
                config,
                "relation",
                {"retain_view_evidence": True, "enabled_components": COMPONENTS},
            )
            nominal = set(bundle.nominal_features)
            for feature, importance in enumerate(model.attribute_importance_):
                attribute_rows.append(
                    {
                        "dataset": bundle.name,
                        "feature": feature,
                        "feature_type": "nominal" if feature in nominal else "numeric",
                        "importance": float(importance),
                    }
                )
            for view_index, (view, tree) in enumerate(zip(model.views_, model.forest_)):
                item: Dict[str, object] = {
                    "dataset": bundle.name,
                    "view_index": view_index,
                    "features": "|".join(str(value) for value in view),
                    "dimension": len(view),
                    "nominal_count": sum(feature in nominal for feature in view),
                    "node_count": len(tree.tree),
                    "leaf_count": sum(node.is_leaf for node in tree.tree.values()),
                    "gta_count": len(tree.local_gtas),
                    "max_depth": max(node.scale_level for node in tree.tree.values()),
                }
                for component in COMPONENTS:
                    weights = model.view_weights_.get(component)
                    item[f"{component}_view_weight"] = (
                        float(weights[view_index]) if weights is not None else float("nan")
                    )
                view_rows.append(item)
                if dataset_node_count < int(args.max_explanation_nodes):
                    for node_id in sorted(tree.tree):
                        if dataset_node_count >= int(args.max_explanation_nodes):
                            break
                        node = tree.tree[node_id]
                        tree_rows.append(
                            {
                                "dataset": bundle.name,
                                "view_index": view_index,
                                "node_id": node.id,
                                "parent_id": node.parent_id,
                                "depth": node.scale_level,
                                "sample_count": len(node.sample_indices),
                                "is_leaf": node.is_leaf,
                                "is_gta": node.is_gta,
                                "cohesion": node.cohesion,
                                "raw_cohesion_gain": node.raw_cohesion_gain,
                                "normalized_cohesion_gain": node.normalized_cohesion_gain,
                                "anchor_strength": node.anchor_strength,
                                "pivot_a": node.pivot_a,
                                "pivot_b": node.pivot_b,
                            }
                        )
                        dataset_node_count += 1
            selections, overall_rank = _interpretability_selections(
                scores, bundle.y, int(args.explain_top_k)
            )
            view_strength = np.mean(
                np.column_stack(
                    [model.view_weights_[component] for component in COMPONENTS]
                ),
                axis=1,
            )
            selected_views = np.argsort(-view_strength, kind="mergesort")[
                : min(int(args.explain_top_views), len(model.views_))
            ]
            for case_role, sample in selections:
                paths = [len(tree.evolution_paths[int(sample)]) for tree in model.forest_]
                item = {
                    "dataset": bundle.name,
                    "case_role": case_role,
                    "rank": overall_rank[int(sample)],
                    "sample_index": int(sample),
                    "label": int(bundle.y[sample]),
                    "score": float(scores[sample]),
                    "mean_path_length": float(np.mean(paths)),
                    "min_path_length": min(paths),
                    "max_path_length": max(paths),
                }
                for position, component in enumerate(COMPONENTS):
                    item[f"{component}_score"] = float(
                        model.fused_scores_3d_[sample, position]
                    )
                    item[f"{component}_dynamic_weight"] = float(
                        model.dynamic_weights_[sample, position]
                    )
                sample_rows.append(item)
                for view_index in selected_views:
                    tree = model.forest_[int(view_index)]
                    path = tree.evolution_paths[int(sample)]
                    path_item: Dict[str, object] = {
                        "dataset": bundle.name,
                        "case_role": case_role,
                        "sample_index": int(sample),
                        "label": int(bundle.y[sample]),
                        "score": float(scores[sample]),
                        "view_index": int(view_index),
                        "view_features": "|".join(
                            str(value) for value in model.views_[int(view_index)]
                        ),
                        "view_weight": float(view_strength[int(view_index)]),
                        "leaf_node": int(tree.sample_leaf_map[int(sample)]),
                        "path_nodes": "|".join(str(node_id) for node_id in path),
                        "gta_nodes": "|".join(
                            str(node_id) for node_id in tree.local_gtas
                        ),
                    }
                    for position, component in enumerate(COMPONENTS):
                        path_item[f"{component}_view_evidence"] = float(
                            model.view_evidence_[int(view_index), int(sample), position]
                        )
                        path_item[f"{component}_view_weight"] = float(
                            model.view_weights_[component][int(view_index)]
                        )
                    sample_path_rows.append(path_item)
            summaries[bundle.name] = {
                "metrics": {"auc": row["auc"], "ap": row["ap"]},
                "tree_stats": model.tree_stats_,
                "relation_cache_stats": model.relation_cache_stats_,
                "active_components": list(model.enabled_components),
            }
            if args.save_scores:
                _save_score_archive(output, bundle, row, model, scores)
        except Exception as error:
            failures.append(
                {"dataset": bundle.name, "source": str(bundle.path), "error": repr(error)}
            )
            if args.fail_on_error:
                raise
    write_rows(attribute_rows, output / "attribute_importance.csv")
    write_rows(view_rows, output / "view_explanations.csv")
    write_rows(tree_rows, output / "tree_explanations.csv")
    write_rows(sample_rows, output / "sample_explanations.csv")
    write_rows(sample_path_rows, output / "sample_path_explanations.csv")
    write_rows(
        failures,
        output / "failures.csv",
        fieldnames=("dataset", "source", "error"),
    )
    save_json(summaries, output / "structural_explanations.json")
    _write_selected_run_config(args, "interpretability")


class _PeakRSSMonitor:
    def __init__(self, interval: float = 0.02) -> None:
        self.interval = float(interval)
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_PeakRSSMonitor":
        if psutil is None:
            return self
        process = psutil.Process()
        self.start_rss = process.memory_info().rss
        self.peak_rss = self.start_rss

        def monitor() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self.peak_rss = max(self.peak_rss, process.memory_info().rss)
                except Exception:
                    return

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_delta_mb(self) -> float:
        return max(0.0, (self.peak_rss - self.start_rss) / 1024**2)


_EXTERNAL_MODULE_CACHE: Dict[str, object] = {}


def _load_external_module(name: str, path: Path) -> object:
    cached = _EXTERNAL_MODULE_CACHE.get(name)
    if cached is not None:
        return cached
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import baseline module from {path}.")
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(path.parent))
    try:
        specification.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    _EXTERNAL_MODULE_CACHE[name] = module
    return module


def _baseline_matrix(bundle: DatasetBundle) -> np.ndarray:
    """Scale numeric columns to [0, 1] while preserving categorical codes."""

    from sklearn.preprocessing import MinMaxScaler

    values = np.asarray(bundle.X, dtype=np.float64).copy()
    nominal = set(bundle.nominal_features)
    numeric = [index for index in range(values.shape[1]) if index not in nominal]
    if numeric:
        values[:, numeric] = MinMaxScaler().fit_transform(values[:, numeric])
    return values


def _baseline_root() -> Path:
    """Locate the supplied comparison implementations across checkout layouts."""

    candidates = (
        SCRIPT_DIR.parent / "对比算法代码",
        PROJECT_ROOT / "对比算法代码",
        PROJECT_ROOT.parent / "对比算法代码",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Comparison-method root was not found; searched: {searched}")


def _kfgod_scores(
    values: np.ndarray,
    delta: float,
    gb_module: object,
    kfgod_module: object,
) -> np.ndarray:
    """Run the authors' fixed-delta KFGOD pipeline and map ball scores to samples."""

    from sklearn.preprocessing import MinMaxScaler

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("KFGOD requires a nonempty two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("KFGOD requires finite inputs.")
    scaled = MinMaxScaler().fit_transform(matrix)
    balls = list(gb_module.get_GB(scaled))
    if not balls:
        return np.zeros(len(scaled), dtype=np.float64)
    centers = np.vstack(
        [np.asarray(ball, dtype=np.float64)[:, :-1].mean(axis=0) for ball in balls]
    )
    ball_scores = np.asarray(
        kfgod_module.KFGOD(centers, float(delta)), dtype=np.float64
    ).reshape(-1)
    if len(ball_scores) != len(balls) or not np.isfinite(ball_scores).all():
        raise ValueError("KFGOD returned invalid granular-ball scores.")
    scores = np.empty(len(scaled), dtype=np.float64)
    assigned = np.zeros(len(scaled), dtype=bool)
    for ball, score in zip(balls, ball_scores):
        indices = np.asarray(ball, dtype=np.float64)[:, -1].astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= len(scores)):
            raise ValueError("KFGOD granular-ball membership contains invalid indices.")
        scores[indices] = float(score)
        assigned[indices] = True
    if not assigned.all():
        raise ValueError("KFGOD granular balls do not cover every sample.")
    return scores


def _load_repository_kfgod(baseline_root: Path) -> Tuple[object, object]:
    code_root = baseline_root / "KFGOD-main" / "code"
    try:
        gb_module = _load_external_module(
            "relation_runtime_kfgod_gb", code_root / "GB_generation_with_idx.py"
        )
        kfgod_module = _load_external_module(
            "relation_runtime_kfgod", code_root / "KFGOD.py"
        )
    except ModuleNotFoundError as error:
        if error.name == "cupy":
            raise ImportError(
                "KFGOD requires a CUDA-matched CuPy installation (for example "
                "cupy-cuda12x) and a visible NVIDIA GPU."
            ) from error
        raise
    cupy = getattr(kfgod_module, "cp", None)
    if cupy is None or int(cupy.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("KFGOD requires at least one visible CUDA GPU.")
    return gb_module, kfgod_module


def _kfgod_runtime_metadata() -> Dict[str, object]:
    try:
        import cupy

        count = int(cupy.cuda.runtime.getDeviceCount())
        devices = []
        for index in range(count):
            properties = cupy.cuda.runtime.getDeviceProperties(index)
            raw_name = properties.get("name", "unknown")
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            devices.append({"index": index, "name": name})
        return {
            "cupy_version": getattr(cupy, "__version__", "unknown"),
            "cuda_device_count": count,
            "cuda_devices": devices,
            "device_memory_measured": False,
        }
    except Exception as error:
        return {"available": False, "error": repr(error), "device_memory_measured": False}


def _load_repository_dif(baseline_root: Path) -> object:
    """Load repository DIF while preserving its original package imports."""

    cached = _EXTERNAL_MODULE_CACHE.get("relation_runtime_dif")
    if cached is not None:
        return cached

    dif_dir = baseline_root / "DIF"
    net_module = _load_external_module(
        "relation_runtime_dif_net_torch", dif_dir / "net_torch.py"
    )
    algorithms_package = types.ModuleType("algorithms")
    algorithms_package.__path__ = [str(dif_dir)]
    algorithms_package.net_torch = net_module
    previous_package = sys.modules.get("algorithms")
    previous_net = sys.modules.get("algorithms.net_torch")
    sys.modules["algorithms"] = algorithms_package
    sys.modules["algorithms.net_torch"] = net_module
    try:
        return _load_external_module("relation_runtime_dif", dif_dir / "dif.py")
    finally:
        if previous_package is None:
            sys.modules.pop("algorithms", None)
        else:
            sys.modules["algorithms"] = previous_package
        if previous_net is None:
            sys.modules.pop("algorithms.net_torch", None)
        else:
            sys.modules["algorithms.net_torch"] = previous_net


def _load_repository_odgrcr(baseline_root: Path) -> object:
    """Load ODGrCR without requiring its unused plotting/notebook imports."""

    cached = _EXTERNAL_MODULE_CACHE.get("relation_runtime_odgrcr")
    if cached is not None:
        return cached

    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    matplotlib_module.__path__ = []
    matplotlib_module.use = lambda *args, **kwargs: None
    matplotlib_module.pyplot = pyplot_module
    ipython_module = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    ipython_module.__path__ = []
    display_module.display = lambda *args, **kwargs: None
    display_module.clear_output = lambda *args, **kwargs: None
    ipython_module.display = display_module

    replacements = {
        "matplotlib": matplotlib_module,
        "matplotlib.pyplot": pyplot_module,
        "IPython": ipython_module,
        "IPython.display": display_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        return _load_external_module(
            "relation_runtime_odgrcr",
            baseline_root / "ODGrCR" / "ODGrCR_no.py",
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _make_detector_runner(
    method: str,
    bundle: DatasetBundle,
    args: argparse.Namespace,
    config: Mapping[str, object],
    seed: int,
) -> Callable[[], np.ndarray]:
    normalized = method.casefold()
    if normalized in {"relation", "mftad", "relationmftad"}:
        return lambda: RelationConsistentMFTAD(
            **_base_model_kwargs(args, bundle, config)
        ).fit_predict(bundle.X)
    if normalized in {"iforest", "isolationforest"}:
        from sklearn.ensemble import IsolationForest

        def run_iforest() -> np.ndarray:
            values = _baseline_matrix(bundle)
            model = IsolationForest(
                n_estimators=int(args.iforest_trees),
                max_samples="auto",
                contamination="auto",
                random_state=int(seed),
                n_jobs=1,
            )
            return -model.fit(values).score_samples(values)

        return run_iforest
    baseline_root = _baseline_root()
    if normalized == "dcrod":
        module = _load_external_module(
            "relation_runtime_dcrod", baseline_root / "DCROD-main" / "dcrod.py"
        )
        if len(bundle.X) < 3:
            raise ValueError("DCROD requires at least three samples.")
        k = min(max(2, int(args.dcrod_k)), len(bundle.X) - 1)

        def run_dcrod() -> np.ndarray:
            model = module.DCROD(n_neighbors=k, n_jobs=1).fit(
                _baseline_matrix(bundle)
            )
            return np.asarray(model.decision_scores_, dtype=np.float64)

        return run_dcrod
    if normalized == "dfno":
        module = _load_external_module(
            "relation_runtime_dfno", baseline_root / "DFNO" / "DFNO.py"
        )
        k = min(max(1, int(args.dfno_k)), len(bundle.X) - 1)
        return lambda: module.DFNO(_baseline_matrix(bundle), k)
    if normalized == "dif":
        module = _load_repository_dif(baseline_root)

        def run_dif() -> np.ndarray:
            values = _baseline_matrix(bundle)
            model = module.DIF(
                n_ensemble=int(args.dif_ensembles),
                n_estimators=int(args.dif_estimators),
                max_samples=min(int(args.dif_max_samples), len(values)),
                random_state=int(seed),
                device="cpu",
                n_processes=1,
                verbose=0,
            ).fit(values)
            return model.decision_function(values)

        return run_dif
    if normalized == "gbfrd":
        module = _load_external_module(
            "relation_runtime_gbfrd",
            baseline_root / "GBFRD-main" / "Code" / "GBFRD.py",
        )
        return lambda: module.GBFRD(
            _baseline_matrix(bundle), float(args.gbfrd_sigma)
        )
    if normalized == "kfgod":
        gb_module, kfgod_module = _load_repository_kfgod(baseline_root)
        return lambda: _kfgod_scores(
            bundle.X, float(args.kfgod_delta), gb_module, kfgod_module
        )
    if normalized == "mgbod":
        module = _load_external_module(
            "relation_runtime_mgbod", baseline_root / "MGBOD" / "MGBOD.py"
        )
        return lambda: module.MGBOD(
            _baseline_matrix(bundle), float(args.mgbod_sigma)
        )
    if normalized == "mfgad":
        module = _load_external_module(
            "relation_runtime_mfgad", baseline_root / "MFGAD" / "MFGAD.py"
        )
        return lambda: module.MFGAD(
            _baseline_matrix(bundle), float(args.mfgad_delta)
        )
    if normalized == "odgrcr":
        module = _load_repository_odgrcr(baseline_root)
        n_clusters = max(2, int(args.odgrcr_clusters))

        def run_odgrcr() -> np.ndarray:
            random_state = np.random.get_state()
            np.random.seed(int(seed))
            try:
                return np.asarray(
                    module.ODGrCR_with_FCM(
                        np.asarray(bundle.X, dtype=np.float64),
                        n_clusters=n_clusters,
                        show_progress=False,
                    ),
                    dtype=np.float64,
                )
            finally:
                np.random.set_state(random_state)

        return run_odgrcr
    raise ValueError(f"Unknown method {method!r}.")


def _method_summary(
    rows: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    runs_per_group: int,
) -> List[Dict[str, object]]:
    keys = sorted(
        {(str(row["dataset"]), str(row["method"])) for row in (*rows, *failures)}
    )
    summary: List[Dict[str, object]] = []
    for dataset, method in keys:
        selected = [
            row
            for row in rows
            if str(row["dataset"]) == dataset and str(row["method"]) == method
        ]
        failed = sum(
            str(row["dataset"]) == dataset and str(row["method"]) == method
            for row in failures
        )
        item: Dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "requested_runs": runs_per_group,
            "successful_runs": len(selected),
            "failed_runs": failed,
            "unique_score_hashes": len(
                {str(row.get("score_hash", "")) for row in selected}
            ),
        }
        for key, output_name in (
            ("auc", "auc"),
            ("ap", "ap"),
            ("elapsed_seconds", "seconds"),
            ("peak_rss_mb", "peak_rss_mb"),
            ("peak_rss_delta_mb", "peak_rss_delta_mb"),
        ):
            values = np.asarray(
                [float(row[key]) for row in selected if key in row], dtype=np.float64
            )
            if len(values):
                item[f"mean_{output_name}"] = float(values.mean())
                item[f"std_{output_name}"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
        item["deterministic_hash"] = bool(
            selected and item["unique_score_hashes"] == 1
        )
        summary.append(item)
    return summary


def run_real_runtime(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(args)
    lookup = _selected_config_lookup(args.selected_configs)
    methods = _resolve_detector_methods(
        args.runtime_methods,
        ("relation", "IForest", "GBFRD", "DFNO", "MGBOD"),
    )
    repeats = 1 if args.quick else int(args.runtime_repeats)
    raw_path = output / "real_runtime_raw.csv"
    rows: List[Dict[str, object]] = (
        [dict(row) for row in read_rows(raw_path)]
        if bool(args.resume) and raw_path.is_file()
        else []
    )
    completed = {
        (str(row.get("dataset")), str(row.get("method")), int(float(row.get("repeat", 0))))
        for row in rows
    }
    failures: List[Dict[str, object]] = []
    if (
        bool(args.runtime_warmup)
        and not bool(args.no_numba)
        and any(method.casefold() in {"relation", "mftad", "relationmftad"} for method in methods)
    ):
        warm_X, warm_nominal = _runtime_synthetic(24, 4, "mixed", int(args.seed))
        warm_bundle = DatasetBundle(
            name="warmup",
            path=Path("<synthetic>"),
            X=warm_X,
            y=np.asarray([0] * 20 + [1] * 4, dtype=np.int8),
            nominal_features=warm_nominal,
            dataset_type="mixed",
            label_mapping={0: 0, 1: 1},
        )
        RelationConsistentMFTAD(
            **_base_model_kwargs(
                args,
                warm_bundle,
                {"delta": args.fixed_delta, "mu": args.fixed_mu, "max_views": 3},
                {"min_views": 2, "supplemental_view_count": 2},
            )
        ).fit_predict(warm_X)
    if bool(args.runtime_warmup) and "KFGOD" in methods:
        warm_X, warm_nominal = _runtime_synthetic(24, 4, "mixed", int(args.seed))
        warm_bundle = DatasetBundle(
            name="kfgod-warmup",
            path=Path("<synthetic>"),
            X=warm_X,
            y=np.asarray([0] * 20 + [1] * 4, dtype=np.int8),
            nominal_features=warm_nominal,
            dataset_type="mixed",
            label_mapping={0: 0, 1: 1},
        )
        _make_detector_runner("KFGOD", warm_bundle, args, {}, int(args.seed))()
    for bundle_index, bundle in enumerate(bundles, start=1):
        config = _configuration_for_bundle(args, bundle, lookup)
        for method in methods:
            for repeat in range(repeats):
                if (bundle.name, method, repeat) in completed:
                    continue
                seed = int(args.seed) + repeat
                try:
                    runner = _make_detector_runner(method, bundle, args, config, seed)
                    gc.collect()
                    with _PeakRSSMonitor() as monitor:
                        start = time.perf_counter()
                        scores = np.asarray(runner(), dtype=np.float64).reshape(-1)
                        elapsed = time.perf_counter() - start
                    metrics = compute_binary_metrics(bundle.y, scores)
                    rows.append(
                        {
                            "dataset": bundle.name,
                            "dataset_type": bundle.dataset_type,
                            "n_samples": len(bundle.y),
                            "n_features": bundle.X.shape[1],
                            "method": method,
                            "repeat": repeat,
                            "seed": seed,
                            "auc": metrics["auc"],
                            "ap": metrics["ap"],
                            "elapsed_seconds": elapsed,
                            "peak_rss_mb": monitor.peak_rss / 1024**2,
                            "peak_rss_delta_mb": monitor.peak_delta_mb,
                            "score_hash": _array_digest(scores),
                        }
                    )
                    completed.add((bundle.name, method, repeat))
                except Exception as error:
                    failures.append(
                        {
                            "dataset": bundle.name,
                            "method": method,
                            "repeat": repeat,
                            "seed": seed,
                            "error": repr(error),
                        }
                    )
                    if args.fail_on_error:
                        raise
                write_rows(rows, raw_path)
                write_rows(failures, output / "failures.csv")
        print(
            f"[{bundle_index}/{len(bundles)}] real-runtime {bundle.name}", flush=True
        )
    write_rows(rows, raw_path)
    write_rows(_method_summary(rows, failures, repeats), output / "real_runtime_summary.csv")
    write_rows(failures, output / "failures.csv")
    _write_selected_run_config(
        args,
        "real-runtime",
        {
            "methods": methods,
            "memory_measurement": (
                "host process RSS delta; KFGOD GPU device memory is excluded"
                if "KFGOD" in methods
                else "host process RSS delta"
            ),
            "preprocessing_included": True,
            "numba_warmup_excluded": bool(args.runtime_warmup),
            "kfgod": _kfgod_runtime_metadata() if "KFGOD" in methods else None,
        },
    )


def run_repeatability(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(args)
    lookup = _selected_config_lookup(args.selected_configs)
    methods = _resolve_detector_methods(
        args.repeat_methods, ("relation", "IForest")
    )
    seeds = [int(value) for value in parse_number_grid(args.repeat_seeds, int)]
    if args.quick:
        seeds = seeds[:2]
    raw_path = output / "repeat_runs.csv"
    rows: List[Dict[str, object]] = (
        [dict(row) for row in read_rows(raw_path)]
        if bool(args.resume) and raw_path.is_file()
        else []
    )
    completed = {
        (str(row.get("dataset")), str(row.get("method")), int(float(row.get("seed", 0))))
        for row in rows
    }
    failures: List[Dict[str, object]] = []
    for bundle_index, bundle in enumerate(bundles, start=1):
        config = _configuration_for_bundle(args, bundle, lookup)
        for seed in seeds:
            for method in methods:
                if (bundle.name, method, seed) in completed:
                    continue
                try:
                    runner = _make_detector_runner(method, bundle, args, config, seed)
                    start = time.perf_counter()
                    scores = np.asarray(runner(), dtype=np.float64).reshape(-1)
                    elapsed = time.perf_counter() - start
                    metrics = compute_binary_metrics(bundle.y, scores)
                    rows.append(
                        {
                            "dataset": bundle.name,
                            "dataset_type": bundle.dataset_type,
                            "method": method,
                            "seed": seed,
                            "auc": metrics["auc"],
                            "ap": metrics["ap"],
                            "elapsed_seconds": elapsed,
                            "score_hash": _array_digest(scores),
                        }
                    )
                    completed.add((bundle.name, method, seed))
                except Exception as error:
                    failures.append(
                        {
                            "dataset": bundle.name,
                            "method": method,
                            "seed": seed,
                            "error": repr(error),
                        }
                    )
                    if args.fail_on_error:
                        raise
                write_rows(rows, raw_path)
                write_rows(failures, output / "failures.csv")
        print(
            f"[{bundle_index}/{len(bundles)}] repeatability {bundle.name}", flush=True
        )
    summary = _method_summary(rows, failures, len(seeds))
    write_rows(rows, raw_path)
    write_rows(summary, output / "repeat_summary.csv")
    write_rows(
        [
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "unique_score_hashes": row["unique_score_hashes"],
                "deterministic_hash": row["deterministic_hash"],
            }
            for row in summary
        ],
        output / "determinism_audit.csv",
    )
    write_rows(failures, output / "failures.csv")
    _write_selected_run_config(
        args, "repeatability", {"methods": methods, "common_seeds": seeds}
    )


def _runtime_synthetic(
    n_samples: int, n_features: int, data_type_name: str, seed: int
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    if data_type_name == "numeric":
        return rng.normal(size=(n_samples, n_features)), ()
    if data_type_name == "nominal":
        return rng.integers(0, 5, size=(n_samples, n_features)).astype(float), tuple(
            range(n_features)
        )
    if data_type_name == "mixed":
        nominal_count = max(1, n_features // 2)
        numeric_count = n_features - nominal_count
        numeric = rng.normal(size=(n_samples, numeric_count))
        nominal_values = rng.integers(0, 5, size=(n_samples, nominal_count))
        X = np.column_stack((numeric, nominal_values)).astype(float)
        return X, tuple(range(numeric_count, n_features))
    raise ValueError(f"Unknown runtime type: {data_type_name}")


def _runtime_mode_args(
    args: argparse.Namespace, mode: str
) -> argparse.Namespace:
    child = copy.copy(args)
    if mode == "numpy_serial":
        child.no_numba = True
        child.n_jobs = 1
    elif mode == "numba_serial":
        child.no_numba = False
        child.n_jobs = 1
    elif mode == "numba_parallel":
        child.no_numba = False
        child.n_jobs = max(2, int(args.runtime_parallel_jobs))
    else:
        raise ValueError(f"Unknown runtime mode: {mode}")
    return child


def run_runtime(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = [int(item) for item in parse_number_grid(args.runtime_samples, int)]
    features = [int(item) for item in parse_number_grid(args.runtime_features, int)]
    types = _parse_csv_names(args.runtime_types) or list(DATASET_TYPES)
    modes = _parse_csv_names(args.runtime_modes) or [
        "numpy_serial",
        "numba_serial",
        "numba_parallel",
    ]
    repeats = int(args.runtime_repeats)
    if args.quick:
        samples = samples[:2]
        features = features[:1]
        types = types[:2]
        modes = modes[:2]
        repeats = 1
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for mode in modes:
        mode_args = _runtime_mode_args(args, mode)
        if not mode_args.no_numba and args.runtime_warmup:
            warm_X, warm_nominal = _runtime_synthetic(24, 4, "mixed", args.seed)
            warm_bundle = DatasetBundle(
                name="warmup",
                path=Path("<synthetic>"),
                X=warm_X,
                y=np.asarray([0] * 20 + [1] * 4, dtype=np.int8),
                nominal_features=warm_nominal,
                dataset_type="mixed",
                label_mapping={0: 0, 1: 1},
            )
            warm_config = {
                "delta": args.fixed_delta,
                "mu": args.fixed_mu,
                "max_views": min(3, int(args.runtime_views)),
            }
            _fit_transformed(mode_args, warm_bundle, warm_config)
        for n_samples, n_features, kind in itertools.product(samples, features, types):
            for repeat in range(repeats):
                X, nominal = _runtime_synthetic(
                    n_samples, n_features, kind, args.seed + repeat
                )
                y = np.zeros(n_samples, dtype=np.int8)
                y[-max(1, n_samples // 10) :] = 1
                bundle = DatasetBundle(
                    name=f"runtime_{kind}_{n_samples}_{n_features}",
                    path=Path("<synthetic>"),
                    X=X,
                    y=y,
                    nominal_features=nominal,
                    dataset_type=kind,
                    label_mapping={0: 0, 1: 1},
                )
                config = _configuration_for_bundle(mode_args, bundle, {})
                config["max_views"] = int(args.runtime_views)
                try:
                    with _PeakRSSMonitor() as monitor:
                        start = time.perf_counter()
                        model = RelationConsistentMFTAD(
                            **_base_model_kwargs(mode_args, bundle, config)
                        )
                        model.fit_predict(X)
                        elapsed = time.perf_counter() - start
                    rows.append(
                        {
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "dataset_type": kind,
                            "mode": mode,
                            "repeat": repeat,
                            "elapsed_seconds": elapsed,
                            "peak_rss_delta_mb": monitor.peak_delta_mb,
                            "views_used": len(model.views_),
                            "numba_used": model.use_numba,
                            "n_jobs": model.n_jobs,
                        }
                    )
                except Exception as error:
                    failures.append(
                        {
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "dataset_type": kind,
                            "mode": mode,
                            "repeat": repeat,
                            "error": repr(error),
                        }
                    )
                    if args.fail_on_error:
                        raise
                write_rows(rows, output / "runtime_raw.csv")
    summary: List[Dict[str, object]] = []
    keys = sorted(
        {
            (row["n_samples"], row["n_features"], row["dataset_type"], row["mode"])
            for row in rows
        },
        key=str,
    )
    for n_samples, n_features, kind, mode in keys:
        selected = [
            row
            for row in rows
            if (
                row["n_samples"],
                row["n_features"],
                row["dataset_type"],
                row["mode"],
            )
            == (n_samples, n_features, kind, mode)
        ]
        elapsed = np.asarray([float(row["elapsed_seconds"]) for row in selected])
        memory = np.asarray([float(row["peak_rss_delta_mb"]) for row in selected])
        summary.append(
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "dataset_type": kind,
                "mode": mode,
                "repeats": len(selected),
                "mean_seconds": float(elapsed.mean()),
                "std_seconds": float(elapsed.std(ddof=1)) if len(elapsed) > 1 else 0.0,
                "mean_peak_rss_delta_mb": float(memory.mean()),
                "max_peak_rss_delta_mb": float(memory.max()),
            }
        )
    write_rows(rows, output / "runtime_raw.csv")
    write_rows(summary, output / "runtime_summary.csv")
    write_rows(
        failures,
        output / "failures.csv",
        fieldnames=(
            "n_samples",
            "n_features",
            "dataset_type",
            "mode",
            "repeat",
            "error",
        ),
    )
    _write_selected_run_config(
        args,
        "runtime",
        {"numba_warmup_excluded_from_timing": bool(args.runtime_warmup)},
    )


def _synthetic_bundle(
    name: str, X: np.ndarray, y: np.ndarray, nominal: Sequence[int]
) -> DatasetBundle:
    nominal_tuple = tuple(int(item) for item in nominal)
    return DatasetBundle(
        name=name,
        path=Path("<synthetic>"),
        X=np.ascontiguousarray(X, dtype=np.float64),
        y=np.ascontiguousarray(y, dtype=np.int8),
        nominal_features=nominal_tuple,
        dataset_type=dataset_type(X.shape[1], nominal_tuple),
        label_mapping={0: 0, 1: 1},
    )


def _mechanism_datasets(
    seed: int, n_normal: int, n_anomaly: int
) -> List[DatasetBundle]:
    rng = np.random.default_rng(seed)
    labels = np.asarray([0] * n_normal + [1] * n_anomaly, dtype=np.int8)

    global_numeric = np.vstack(
        (
            rng.normal(0.0, 1.0, size=(n_normal, 6)),
            rng.normal(3.5, 0.45, size=(n_anomaly, 6)),
        )
    )

    nominal_normal = rng.integers(0, 3, size=(n_normal, 6))
    nominal_anomaly = np.tile(np.asarray([4, 4, 4, 3, 3, 3]), (n_anomaly, 1))
    nominal_rare = np.vstack((nominal_normal, nominal_anomaly)).astype(float)

    mixed_numeric_normal = rng.normal(size=(n_normal, 3))
    mixed_nominal_normal = rng.integers(0, 3, size=(n_normal, 3))
    mixed_numeric_anomaly = rng.normal(1.75, 0.35, size=(n_anomaly, 3))
    mixed_nominal_anomaly = np.tile(np.asarray([3, 3, 2]), (n_anomaly, 1))
    mixed = np.vstack(
        (
            np.column_stack((mixed_numeric_normal, mixed_nominal_normal)),
            np.column_stack((mixed_numeric_anomaly, mixed_nominal_anomaly)),
        )
    ).astype(float)

    compact = np.vstack(
        (
            rng.normal(0.0, 1.0, size=(n_normal, 6)),
            rng.normal(2.4, 0.08, size=(n_anomaly, 6)),
        )
    )
    return [
        _synthetic_bundle("global_numeric", global_numeric, labels, ()),
        _synthetic_bundle("rare_nominal", nominal_rare, labels, range(6)),
        _synthetic_bundle("mixed_intersection", mixed, labels, (3, 4, 5)),
        _synthetic_bundle("compact_cluster", compact, labels, ()),
    ]


def run_mechanism(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    n_normal = 60 if args.quick else int(args.mechanism_normal)
    n_anomaly = 10 if args.quick else int(args.mechanism_anomaly)
    bundles = _mechanism_datasets(args.seed, n_normal, n_anomaly)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for bundle in bundles:
        config = _configuration_for_bundle(args, bundle, {})
        config["max_views"] = min(
            int(args.max_views), 8 if args.quick else int(args.max_views)
        )
        try:
            row, model, scores = _fit_and_record(bundle, args, config, "relation")
            rows.append(row)
            if args.save_scores:
                _save_score_archive(output, bundle, row, model, scores)
        except Exception as error:
            failures.append({"dataset": bundle.name, "error": repr(error)})
            if args.fail_on_error:
                raise
    write_rows(rows, output / "mechanism_results.csv")
    write_rows(
        failures,
        output / "failures.csv",
        fieldnames=("dataset", "error"),
    )
    _write_selected_run_config(
        args,
        "mechanism",
        {"n_normal": n_normal, "n_anomaly": n_anomaly, "controlled_synthetic": True},
    )


def _stress_bundle(
    scenario: str,
    n_samples: int,
    anomaly_fraction: float,
    compactness: float,
    separation: float,
    seed: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_anomaly = max(2, int(round(n_samples * anomaly_fraction)))
    n_normal = n_samples - n_anomaly
    labels = np.asarray([0] * n_normal + [1] * n_anomaly, dtype=np.int8)
    if scenario == "numeric_compact_cluster":
        normal = rng.normal(0.0, 1.0, size=(n_normal, 6))
        anomaly = rng.normal(separation, compactness, size=(n_anomaly, 6))
        values = np.vstack((normal, anomaly))
        nominal: Tuple[int, ...] = ()
    elif scenario == "nominal_rare_combination":
        normal = rng.integers(0, 3, size=(n_normal, 6))
        anomaly = rng.integers(0, 3, size=(n_anomaly, 6))
        special_count = min(6, max(1, int(round(separation))))
        anomaly[:, :special_count] = 3
        mutations = rng.random((n_anomaly, special_count)) < compactness
        replacements = rng.integers(0, 3, size=(n_anomaly, special_count))
        anomaly[:, :special_count] = np.where(
            mutations, replacements, anomaly[:, :special_count]
        )
        values = np.vstack((normal, anomaly)).astype(np.float64)
        nominal = tuple(range(6))
    elif scenario == "mixed_interaction":
        normal_numeric = rng.normal(0.0, 1.0, size=(n_normal, 3))
        normal_nominal = rng.integers(0, 3, size=(n_normal, 3))
        anomaly_numeric = rng.normal(separation, compactness, size=(n_anomaly, 3))
        anomaly_nominal = rng.integers(0, 3, size=(n_anomaly, 3))
        special_count = min(3, max(1, int(math.ceil(separation / 2.0))))
        anomaly_nominal[:, :special_count] = 3
        mutations = rng.random((n_anomaly, special_count)) < compactness
        replacements = rng.integers(0, 3, size=(n_anomaly, special_count))
        anomaly_nominal[:, :special_count] = np.where(
            mutations, replacements, anomaly_nominal[:, :special_count]
        )
        values = np.vstack(
            (
                np.column_stack((normal_numeric, normal_nominal)),
                np.column_stack((anomaly_numeric, anomaly_nominal)),
            )
        ).astype(np.float64)
        nominal = (3, 4, 5)
    else:
        raise ValueError(f"Unknown stress scenario {scenario!r}.")
    return _synthetic_bundle(scenario, values, labels, nominal)


def run_stress(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = _parse_csv_names(args.stress_scenarios) or [
        "numeric_compact_cluster",
        "nominal_rare_combination",
        "mixed_interaction",
    ]
    fractions = [float(value) for value in parse_number_grid(args.stress_anomaly_ratios)]
    compactness_values = [
        float(value) for value in parse_number_grid(args.stress_compactness)
    ]
    separations = [float(value) for value in parse_number_grid(args.stress_separations)]
    seeds = [int(value) for value in parse_number_grid(args.stress_seeds, int)]
    if args.quick:
        fractions = fractions[:2]
        compactness_values = compactness_values[:2]
        separations = separations[:2]
        seeds = seeds[:1]
    result_path = output / "stress_results.csv"
    rows: List[Dict[str, object]] = (
        [dict(row) for row in read_rows(result_path)]
        if bool(args.resume) and result_path.is_file()
        else []
    )
    completed = {
        (
            str(row.get("scenario")),
            float(row.get("anomaly_fraction", 0.0)),
            float(row.get("compactness", 0.0)),
            float(row.get("separation", 0.0)),
            int(float(row.get("seed", 0))),
        )
        for row in rows
    }
    failures: List[Dict[str, object]] = []
    combinations = list(
        itertools.product(
            scenarios, fractions, compactness_values, separations, seeds
        )
    )
    for index, (scenario, fraction, compactness, separation, seed) in enumerate(
        combinations, start=1
    ):
        if (scenario, fraction, compactness, separation, seed) in completed:
            print(f"[stress {index}/{len(combinations)}] resume {scenario}", flush=True)
            continue
        try:
            bundle = _stress_bundle(
                scenario,
                int(args.stress_samples),
                fraction,
                compactness,
                separation,
                seed,
            )
            config = _configuration_for_bundle(args, bundle, {})
            config["max_views"] = min(
                int(args.max_views), 12 if args.quick else int(args.max_views)
            )
            row, model, scores = _fit_and_record(
                bundle, args, config, "relation"
            )
            row.update(
                {
                    "scenario": scenario,
                    "anomaly_fraction": fraction,
                    "compactness": compactness,
                    "separation": separation,
                    "seed": seed,
                }
            )
            rows.append(row)
            completed.add((scenario, fraction, compactness, separation, seed))
            if args.save_scores:
                _save_score_archive(output, bundle, row, model, scores)
        except Exception as error:
            failures.append(
                {
                    "scenario": scenario,
                    "anomaly_fraction": fraction,
                    "compactness": compactness,
                    "separation": separation,
                    "seed": seed,
                    "error": repr(error),
                }
            )
            if args.fail_on_error:
                raise
        write_rows(rows, result_path)
        print(f"[stress {index}/{len(combinations)}] {scenario}", flush=True)
    write_rows(rows, result_path)
    write_rows(failures, output / "failures.csv")
    summary: List[Dict[str, object]] = []
    keys = sorted(
        {
            (
                str(row["scenario"]),
                float(row["anomaly_fraction"]),
                float(row["compactness"]),
                float(row["separation"]),
            )
            for row in rows
        }
    )
    for scenario, fraction, compactness, separation in keys:
        selected = [
            row
            for row in rows
            if (
                str(row["scenario"]),
                float(row["anomaly_fraction"]),
                float(row["compactness"]),
                float(row["separation"]),
            )
            == (scenario, fraction, compactness, separation)
        ]
        auc_values = np.asarray([float(row["auc"]) for row in selected])
        ap_values = np.asarray([float(row["ap"]) for row in selected])
        summary.append(
            {
                "scenario": scenario,
                "anomaly_fraction": fraction,
                "compactness": compactness,
                "separation": separation,
                "successful_seeds": len(selected),
                "mean_auc": float(auc_values.mean()),
                "std_auc": (
                    float(auc_values.std(ddof=1)) if len(auc_values) > 1 else 0.0
                ),
                "mean_ap": float(ap_values.mean()),
                "std_ap": (
                    float(ap_values.std(ddof=1)) if len(ap_values) > 1 else 0.0
                ),
            }
        )
    write_rows(summary, output / "stress_summary.csv")
    _write_selected_run_config(
        args,
        "stress",
        {
            "scenarios": scenarios,
            "anomaly_ratios": fractions,
            "compactness": compactness_values,
            "separations": separations,
            "seeds": seeds,
        },
    )


def run_smoke(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundles = _mechanism_datasets(args.seed, 32, 8)[:3]
    child = copy.copy(args)
    child.min_views = min(int(args.min_views), 3)
    child.max_views = min(int(args.max_views), 5)
    child.supplemental_view_count = min(int(args.supplemental_view_count), 3)
    child.n_jobs = 1
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for bundle in bundles:
        config = _configuration_for_bundle(child, bundle, {})
        config["max_views"] = child.max_views
        config["delta_report"] = (
            "N/A" if bundle.dataset_type == "nominal" else config["delta"]
        )
        try:
            row, _, _ = _fit_and_record(bundle, child, config, "relation")
            rows.append(row)
        except Exception as error:
            failures.append({"dataset": bundle.name, "error": repr(error)})
            if args.fail_on_error:
                raise
    write_rows(rows, output / "smoke_results.csv")
    write_rows(
        failures,
        output / "failures.csv",
        fieldnames=("dataset", "error"),
    )
    save_json(
        system_manifest(
            {
                "experiment": "smoke",
                "arguments": _args_payload(child),
                "workflow_validation_only": True,
            }
        ),
        output / "run_config.json",
    )
    if failures or len(rows) != 3:
        raise RuntimeError(f"Smoke validation failed: {failures}")
    print(f"[smoke complete] rows={len(rows)} output={output}", flush=True)


def _wide_numeric_table(
    path: str,
    dataset_column: Optional[str],
    source_group_column: Optional[str] = None,
) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    if not fields or not rows:
        raise ValueError("The statistics input must contain a header and data rows.")
    duplicate_fields = sorted(
        {field for field in fields if fields.count(field) > 1}
    )
    if duplicate_fields:
        raise ValueError(
            f"The statistics input contains duplicate columns: {duplicate_fields}"
        )
    dataset_field = dataset_column or next(
        (field for field in fields if field.lower() in {"dataset", "data", "name"}),
        fields[0],
    )
    if source_group_column and source_group_column not in fields:
        raise ValueError(
            f"Source-group column {source_group_column!r} is not in the input table."
        )
    candidates: List[str] = []
    for field in fields:
        if field in {dataset_field, source_group_column}:
            continue
        numeric_count = 0
        for row in rows:
            try:
                float(str(row.get(field, "")).strip())
                numeric_count += 1
            except ValueError:
                pass
        if numeric_count >= 2 and numeric_count >= math.ceil(0.5 * len(rows)):
            candidates.append(field)
    if len(candidates) < 2:
        raise ValueError("At least two numeric method columns are required.")
    complete: List[Tuple[str, str, List[float]]] = []
    dropped: List[str] = []
    for row in rows:
        name = str(row.get(dataset_field, "")).strip()
        try:
            numeric = [float(str(row[field]).strip()) for field in candidates]
        except (KeyError, ValueError):
            dropped.append(name)
            continue
        if not np.all(np.isfinite(numeric)):
            dropped.append(name)
            continue
        group = (
            str(row.get(source_group_column, "")).strip()
            if source_group_column
            else name
        )
        if not group:
            dropped.append(name)
            continue
        complete.append((name, group, numeric))
    datasets: List[str] = []
    values: List[List[float]] = []
    grouped: Dict[str, List[List[float]]] = {}
    for _, group, numeric in complete:
        grouped.setdefault(group, []).append(numeric)
    for group in sorted(grouped):
        datasets.append(group)
        values.append(np.mean(np.asarray(grouped[group], dtype=np.float64), axis=0).tolist())
    if len(values) < 2:
        raise ValueError("At least two complete aligned dataset rows are required.")
    return datasets, candidates, np.asarray(values, dtype=np.float64), dropped


def run_statistics(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets, methods, values, dropped = _wide_numeric_table(
        args.input_csv, args.dataset_column, args.source_group_column
    )
    result = statistical_comparison(
        values,
        methods,
        alpha=float(args.alpha),
        tie_tolerance=float(args.tie_tolerance),
    )
    result["datasets"] = datasets
    result["dropped_incomplete_datasets"] = dropped
    result["metric"] = args.metric_name
    save_json(result, output / "statistical_tests.json")
    wilcoxon_rows = list(result.get("wilcoxon_holm", []))
    write_rows(wilcoxon_rows, output / "wilcoxon_holm.csv")
    if args.reference_method:
        if args.reference_method not in methods:
            raise ValueError(
                f"Reference method {args.reference_method!r} is not in {methods}."
            )
        reference_rows = [
            row
            for row in wilcoxon_rows
            if args.reference_method in {row["method_a"], row["method_b"]}
        ]
        write_rows(reference_rows, output / "reference_pairwise.csv")
    average_ranks = result.get("average_ranks", {})
    write_rows(
        [
            {"method": method, "average_rank": average_ranks[method]}
            for method in methods
        ],
        output / "average_ranks.csv",
    )
    critical_difference = float(result.get("nemenyi_critical_difference", float("nan")))
    if np.isfinite(critical_difference):
        try:
            plot_critical_difference(
                average_ranks,
                critical_difference,
                output / "critical_difference.svg",
                title=f"{args.metric_name.upper()} critical difference",
            )
        except Exception as error:  # pragma: no cover - optional plotting backend
            save_json({"error": repr(error)}, output / "plot_error.json")
    table = latex_wide_table(
        datasets,
        methods,
        values,
        caption=f"Aligned {args.metric_name.upper()} results.",
        label=f"tab:relation_{args.metric_name}",
    )
    (output / "aligned_table.tex").write_text(table, encoding="utf-8")
    save_json(
        {
            "input_rows": len(datasets) + len(dropped),
            "aligned_complete_rows": len(datasets),
            "dropped_incomplete_datasets": dropped,
            "methods": methods,
            "source_group_column": args.source_group_column,
            "source_family_aggregation": (
                "mean within source group" if args.source_group_column else "none"
            ),
        },
        output / "alignment.json",
    )
    _write_selected_run_config(args, "statistics")


def _child_args(args: argparse.Namespace, folder: str) -> argparse.Namespace:
    child = copy.copy(args)
    child.output_dir = Path(args.output_dir) / folder
    return child


def run_all(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    completed: List[str] = []

    benchmark_args = _child_args(args, "01_benchmark")
    run_benchmark(benchmark_args)
    completed.append("benchmark")
    selected = root / "01_benchmark" / "best_auc.csv"

    ablation_args = _child_args(args, "02_ablation")
    run_ablation(ablation_args)
    completed.append("ablation")

    for folder, function in (
        ("03_complementarity", run_complementarity),
        ("04_sensitivity", run_sensitivity),
        ("05_invariance", run_invariance),
        ("06_interpretability", run_interpretability),
    ):
        child = _child_args(args, folder)
        if not child.selected_configs and selected.is_file():
            child.selected_configs = str(selected)
        function(child)
        completed.append(folder.split("_", 1)[1])

    run_runtime(_child_args(args, "07_runtime"))
    completed.append("runtime")
    run_mechanism(_child_args(args, "08_mechanism"))
    completed.append("mechanism")

    statistics_completed: List[str] = []
    for metric, baseline in (("auc", args.baseline_auc), ("ap", args.baseline_ap)):
        comparison = root / "01_benchmark" / f"comparison_{metric}.csv"
        if baseline and comparison.is_file():
            child = _child_args(args, f"09_statistics_{metric}")
            child.input_csv = str(comparison)
            child.metric_name = metric
            run_statistics(child)
            statistics_completed.append(metric)
    save_json(
        {
            "completed": completed,
            "statistics_completed": statistics_completed,
            "primary_selection": "AUC per dataset",
            "primary_ap": "AP from the same AUC-selected configuration",
            "quick": bool(args.quick),
            "note": (
                "The all command is intentionally expensive. Quick mode validates "
                "the workflow and is not paper evidence."
            ),
        },
        root / "paper_all_manifest.json",
    )


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--numba-threads", type=int, default=1)
    parser.add_argument("--memory-safety-ratio", type=float, default=0.8)
    parser.add_argument("--attribute-relation-cache-mb", type=float, default=512.0)
    parser.add_argument("--retain-view-evidence", action="store_true")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-dataset-limit", type=int, default=3)


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        choices=("all", *DATASET_TYPES),
        default=["all"],
    )
    parser.add_argument("--nominal-map", type=Path, default=DEFAULT_NOMINAL_MAP)
    parser.add_argument(
        "--require-schema-coverage",
        action="store_true",
        help="Fail when any requested dataset is absent from the nominal map.",
    )
    parser.add_argument("--max-samples", type=int, default=0)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixed-delta", type=float, default=REVISED_FIXED_DELTA)
    parser.add_argument("--fixed-mu", type=float, default=REVISED_FIXED_MU)
    parser.add_argument(
        "--pure-nominal-fixed-delta",
        type=float,
        default=REVISED_PURE_NOMINAL_FIXED_DELTA,
        help=(
            "Fixed runtime delta placeholder for pure-nominal datasets; "
            "the reported value remains N/A because equality relations ignore delta."
        ),
    )
    parser.add_argument(
        "--pure-nominal-fixed-mu",
        type=float,
        default=REVISED_PURE_NOMINAL_FIXED_MU,
        help=(
            "Use this fixed cohesion threshold only for pure-nominal datasets; "
            "numeric and mixed datasets continue to use --fixed-mu/--mus."
        ),
    )
    parser.add_argument("--min-views", type=int, default=50)
    parser.add_argument("--max-views", type=int, default=100)
    parser.add_argument("--supplemental-view-count", type=int, default=50)
    parser.add_argument("--min-view-dimension-ratio", type=float, default=0.1)
    parser.add_argument(
        "--supplemental-view-dimension-ratio", type=float, default=0.7
    )
    parser.add_argument("--min-child-size", type=int, default=1)
    parser.add_argument("--strict-cohesion-gain", action="store_true")


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deltas", default=DEFAULT_DELTAS)
    parser.add_argument("--mus", default=DEFAULT_MUS)
    parser.add_argument("--view-grid", default="100")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-auc")
    parser.add_argument("--baseline-ap")


def _add_selected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selected-configs")


def _add_sensitivity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sensitivity-parameters",
        default=DEFAULT_SENSITIVITY_PARAMETERS,
    )
    parser.add_argument(
        "--sensitivity-deltas", default=DEFAULT_SENSITIVITY_DELTAS
    )
    parser.add_argument("--sensitivity-mus", default=DEFAULT_SENSITIVITY_MUS)
    parser.add_argument("--sensitivity-views", default=DEFAULT_SENSITIVITY_VIEWS)
    parser.add_argument("--sensitivity-forest-sizes", default="10,20,50,100")
    parser.add_argument("--sensitivity-min-child", default="1,2,3,5")
    parser.add_argument("--sensitivity-supplemental-views", default="0,10,25,50")
    parser.add_argument(
        "--sensitivity-auc-min-span",
        type=float,
        default=DEFAULT_SENSITIVITY_AUC_MIN_SPAN,
    )
    parser.add_argument(
        "--sensitivity-ap-min-span",
        type=float,
        default=DEFAULT_SENSITIVITY_AP_MIN_SPAN,
    )
    parser.add_argument(
        "--sensitivity-axis-padding",
        type=float,
        default=DEFAULT_SENSITIVITY_AXIS_PADDING,
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-samples", default="200,500,1000")
    parser.add_argument("--runtime-features", default="10,20,40")
    parser.add_argument("--runtime-types", default="numeric,nominal,mixed")
    parser.add_argument(
        "--runtime-modes", default="numpy_serial,numba_serial,numba_parallel"
    )
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument("--runtime-views", type=int, default=20)
    parser.add_argument("--runtime-parallel-jobs", type=int, default=2)
    parser.add_argument(
        "--runtime-warmup", action=argparse.BooleanOptionalAction, default=True
    )


def _add_mechanism_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mechanism-normal", type=int, default=180)
    parser.add_argument("--mechanism-anomaly", type=int, default=20)


def _add_stress_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stress-scenarios",
        default="numeric_compact_cluster,nominal_rare_combination,mixed_interaction",
    )
    parser.add_argument("--stress-samples", type=int, default=240)
    parser.add_argument("--stress-anomaly-ratios", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--stress-compactness", default="0.05,0.20,0.50")
    parser.add_argument("--stress-separations", default="1.0,2.0,3.0,4.0")
    parser.add_argument("--stress-seeds", default="1:10:1")


def _add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iforest-trees", type=int, default=100)
    parser.add_argument("--dcrod-k", type=int, default=5)
    parser.add_argument("--dfno-k", type=int, default=3)
    parser.add_argument("--dif-ensembles", type=int, default=50)
    parser.add_argument("--dif-estimators", type=int, default=6)
    parser.add_argument("--dif-max-samples", type=int, default=256)
    parser.add_argument("--gbfrd-sigma", type=float, default=0.6)
    parser.add_argument(
        "--kfgod-delta",
        type=float,
        default=0.3,
        help="Fixed KFGOD kernel delta; never selected with evaluation labels.",
    )
    parser.add_argument("--mfgad-delta", type=float, default=1.0)
    parser.add_argument("--mgbod-sigma", type=float, default=0.6)
    parser.add_argument("--odgrcr-clusters", type=int, default=3)


def _add_interpretability_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--explain-top-k", type=int, default=5)
    parser.add_argument("--max-explanation-nodes", type=int, default=5000)
    parser.add_argument("--explain-top-views", type=int, default=10)


def _add_statistics_options(
    parser: argparse.ArgumentParser, require_input: bool
) -> None:
    if require_input:
        parser.add_argument("--input-csv", required=True)
        parser.add_argument("--metric-name", choices=("auc", "ap"), required=True)
    parser.add_argument("--dataset-column")
    parser.add_argument("--source-group-column")
    parser.add_argument("--reference-method")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--alpha", type=float, default=0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone relation-consistent MFTAD and complete paper experiments. "
            "Label-based parameter selection is recorded explicitly."
        )
    )
    execution = argparse.ArgumentParser(add_help=False)
    _add_execution_arguments(execution)
    data = argparse.ArgumentParser(add_help=False)
    _add_dataset_arguments(data)
    model = argparse.ArgumentParser(add_help=False)
    _add_model_arguments(model)
    grid = argparse.ArgumentParser(add_help=False)
    _add_grid_arguments(grid)
    selected = argparse.ArgumentParser(add_help=False)
    _add_selected_arguments(selected)

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "smoke", parents=[execution, model], help="Run three synthetic workflow checks."
    )

    mechanism = subparsers.add_parser(
        "mechanism",
        parents=[execution, model],
        help="Run controlled numeric, nominal, mixed, and compact-cluster studies.",
    )
    _add_mechanism_arguments(mechanism)

    stress = subparsers.add_parser(
        "stress",
        parents=[execution, model],
        help="Sweep anomaly ratio, compactness, and separation for three mechanisms.",
    )
    _add_stress_arguments(stress)
    stress.add_argument("--resume", action="store_true")

    subparsers.add_parser(
        "benchmark",
        parents=[execution, data, model, grid],
        help="Run the delta x mu x view-count benchmark grid.",
    )

    repair = subparsers.add_parser(
        "repair-benchmark",
        parents=[execution],
        help="Replace invalid dataset grids and regenerate benchmark summaries.",
    )
    repair.add_argument("--benchmark-dir", type=Path, required=True)
    repair.add_argument("--replacement-dir", type=Path, required=True)
    repair.add_argument("--datasets", nargs="+", required=True)
    repair.add_argument("--baseline-auc")
    repair.add_argument("--baseline-ap")

    ablation = subparsers.add_parser(
        "ablation",
        parents=[execution, data, model, grid],
        help="Run fixed-protocol unified-framework mechanism ablations.",
    )
    ablation.add_argument(
        "--ablation-variants",
        default="revised_cumulative",
        help=(
            "Suite (cumulative, paper_main, pivot_redesign, revised_cumulative, "
            "structure_evidence, interaction, leave_one_out, evidence, "
            "revised_evidence, legacy_evidence, all; "
            "main/revised_main and structure_evidence_interaction are aliases) or "
            "comma-separated variant names. The revised suites use raw gain."
        ),
    )
    ablation.add_argument(
        "--ablation-protocol",
        choices=("fixed", "label_oracle"),
        default="fixed",
        help="Use one common fixed configuration unless explicitly running a diagnostic oracle grid.",
    )

    gain_complexity = subparsers.add_parser(
        "gain-complexity",
        parents=[execution, data, model, grid],
        help="Compare raw and normalized gain on a label-free matched-complexity mu grid.",
    )
    gain_complexity.add_argument(
        "--complexity-match-metric",
        choices=("nodes", "depth", "joint"),
        default="joint",
        help="Structural quantity used to match raw and normalized gain runs.",
    )
    gain_complexity.add_argument(
        "--raw-mus",
        help="Raw-gain mu grid; defaults to --mus for backward compatibility.",
    )
    gain_complexity.add_argument(
        "--normalized-mus",
        help=(
            "Normalized-gain mu grid; defaults to --mus and must include "
            "--fixed-mu."
        ),
    )
    gain_complexity.add_argument(
        "--complexity-match-thresholds",
        default="0.3,0.5",
        help="Distance thresholds used for label-free match-quality diagnostics.",
    )

    complementarity = subparsers.add_parser(
        "complementarity",
        parents=[execution, data, model, selected],
        help="Measure Leaf/GTA/Path complementarity.",
    )
    complementarity.add_argument("--top-fraction", type=float, default=0.1)
    complementarity.add_argument(
        "--top-k-rule", choices=("anomaly_count", "fraction"), default="anomaly_count"
    )

    sensitivity = subparsers.add_parser(
        "sensitivity",
        parents=[execution, data, model, selected],
        help="Run one-factor parameter sensitivity.",
    )
    _add_sensitivity_arguments(sensitivity)
    sensitivity.add_argument("--resume", action="store_true")

    invariance = subparsers.add_parser(
        "invariance",
        parents=[execution, data, model, selected],
        help="Test sample order, feature order, and nominal recoding invariance.",
    )
    invariance.add_argument("--invariance-tolerance", type=float, default=1e-10)

    runtime = subparsers.add_parser(
        "runtime",
        parents=[execution, model],
        help="Run synthetic sample/feature scaling and acceleration comparisons.",
    )
    _add_runtime_arguments(runtime)

    real_runtime = subparsers.add_parser(
        "real-runtime",
        parents=[execution, data, model, selected],
        help="Measure real-data runtime and peak RSS against repository baselines.",
    )
    real_runtime.add_argument(
        "--runtime-methods",
        default="relation,IForest,GBFRD,DFNO,MGBOD",
        help=(
            "Comma-separated methods. Supported: "
            f"{', '.join(SUPPORTED_DETECTOR_METHODS)}. "
            "Use 'repository' for every detector under 对比算法代码 or 'all' "
            "to include RelationMFTAD and IForest as well."
        ),
    )
    real_runtime.add_argument("--runtime-repeats", type=int, default=3)
    real_runtime.add_argument(
        "--runtime-warmup", action=argparse.BooleanOptionalAction, default=True
    )
    real_runtime.add_argument("--resume", action="store_true")
    _add_baseline_arguments(real_runtime)

    repeatability = subparsers.add_parser(
        "repeatability",
        parents=[execution, data, model, selected],
        help="Run common seeds, hashes, means, standard deviations, and failures.",
    )
    repeatability.add_argument(
        "--repeat-methods",
        default="relation,IForest",
        help=(
            "Comma-separated methods. Supported: "
            f"{', '.join(SUPPORTED_DETECTOR_METHODS)}. "
            "Use 'repository' or 'all' for group expansion."
        ),
    )
    repeatability.add_argument("--repeat-seeds", default="1:20:1")
    repeatability.add_argument("--resume", action="store_true")
    _add_baseline_arguments(repeatability)

    interpretability = subparsers.add_parser(
        "interpretability",
        parents=[execution, data, model, selected],
        help="Export attribute, view, tree, path, and sample explanations.",
    )
    _add_interpretability_arguments(interpretability)

    statistics = subparsers.add_parser(
        "statistics",
        parents=[execution],
        help="Run aligned Friedman/Iman-Davenport/Nemenyi analysis.",
    )
    _add_statistics_options(statistics, require_input=True)

    all_parser = subparsers.add_parser(
        "all",
        parents=[execution, data, model, grid, selected],
        help="Run the complete paper experiment suite.",
    )
    all_parser.add_argument("--ablation-variants", default="revised_cumulative")
    all_parser.add_argument(
        "--ablation-protocol",
        choices=("fixed", "label_oracle"),
        default="fixed",
    )
    all_parser.add_argument("--top-fraction", type=float, default=0.1)
    all_parser.add_argument(
        "--top-k-rule", choices=("anomaly_count", "fraction"), default="anomaly_count"
    )
    all_parser.add_argument("--invariance-tolerance", type=float, default=1e-10)
    _add_sensitivity_arguments(all_parser)
    _add_runtime_arguments(all_parser)
    _add_mechanism_arguments(all_parser)
    _add_interpretability_arguments(all_parser)
    _add_statistics_options(all_parser, require_input=False)
    return parser


COMMANDS = {
    "smoke": run_smoke,
    "mechanism": run_mechanism,
    "stress": run_stress,
    "benchmark": run_benchmark,
    "repair-benchmark": run_repair_benchmark,
    "ablation": run_ablation,
    "gain-complexity": run_gain_complexity,
    "complementarity": run_complementarity,
    "sensitivity": run_sensitivity,
    "invariance": run_invariance,
    "runtime": run_runtime,
    "real-runtime": run_real_runtime,
    "repeatability": run_repeatability,
    "interpretability": run_interpretability,
    "statistics": run_statistics,
    "all": run_all,
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
