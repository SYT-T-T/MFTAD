"""Repair a single misidentified dataset in a completed paper-results bundle.

The command consumes fresh single-dataset reruns and writes a new result bundle.
It never overwrites the downloaded source bundle. Use ``--apply`` only after a
successful dry run. Derived outputs are regenerated from merged raw rows; no
full-dataset refit is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from paper_conventions import PAPER_DATASET_FILES


WRONG_DATASET = "chess_nowin_145_variant1"
CORRECT_DATASET = "chess_nowin_227_variant1"
EXPERIMENT_GLOBS = {
    "benchmark": ("01_*_benchmark", "01_benchmark"),
    "ablation": ("02_*_ablation", "02_ablation"),
    "invariance": ("03_*_invariance", "03_invariance"),
    "complementarity": ("04_*_complementarity", "04_complementarity"),
    "sensitivity": ("05_*_sensitivity", "05_sensitivity"),
    "interpretability": ("06_*_interpretability", "06_interpretability"),
}
PRIMARY_FILES = {
    "benchmark": "raw_results.csv",
    "ablation": "raw_results.csv",
    "invariance": "invariance_results.csv",
    "complementarity": "complementarity_results.csv",
    "sensitivity": "sensitivity_results.csv",
}
KEY_FIELDS = {
    # config_id includes data/label digests, so it must change when replacing
    # one dataset. Compare the semantic runtime configuration instead.
    "benchmark": (
        "dataset",
        "variant",
        "delta",
        "delta_report",
        "mu",
        "min_views",
        "max_views",
        "enabled_components",
    ),
    "ablation": (
        "dataset",
        "variant",
        "delta",
        "delta_report",
        "mu",
        "min_views",
        "max_views",
        "enabled_components",
    ),
    "invariance": ("dataset", "transformation"),
    "complementarity": ("dataset",),
    "sensitivity": ("dataset", "parameter", "value"),
    "feature_report": ("dataset",),
}
BASELINE_METHODS = (
    "IForest",
    "COPOD",
    "ECOD",
    "KNN",
    "LOF",
    "DCROD",
    "ODGrCR",
    "MFGAD",
    "GBFRD",
    "DFNO",
    "MGBOD",
)
MATRIX_METHODS = (*BASELINE_METHODS, "MFTAD")


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_dir(root: Path, patterns: Sequence[str], label: str) -> Path:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_dir())
    unique = sorted(set(matches))
    if len(unique) > 1:
        fixed20 = [path for path in unique if "fixed20" in path.name]
        if len(fixed20) == 1:
            return fixed20[0]
    if len(unique) != 1:
        raise ValueError(f"Expected exactly one {label} directory under {root}; found {unique}")
    return unique[0]


def _dataset_set(rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {str(row.get("dataset", "")) for row in rows}


def _key(row: Mapping[str, str], fields: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def merge_dataset_rows(
    family: str,
    original_rows: Sequence[Mapping[str, str]],
    replacement_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Replace one dataset while requiring identical configuration keys."""

    if not original_rows or not replacement_rows:
        raise ValueError(f"{family}: source and replacement rows must be non-empty")
    if WRONG_DATASET not in _dataset_set(original_rows):
        raise ValueError(f"{family}: source has no {WRONG_DATASET}")
    if CORRECT_DATASET in _dataset_set(original_rows):
        raise ValueError(f"{family}: source already contains {CORRECT_DATASET}")
    if _dataset_set(replacement_rows) != {CORRECT_DATASET}:
        raise ValueError(
            f"{family}: replacement must contain only {CORRECT_DATASET}; "
            f"observed {sorted(_dataset_set(replacement_rows))}"
        )
    if family != "feature_report" and any(
        str(row.get("dataset_type", "")) != "nominal" for row in replacement_rows
    ):
        raise ValueError(f"{family}: chess 227 replacement must use the nominal protocol")
    fields = KEY_FIELDS[family]
    old_suffixes = {_key(row, fields)[1:] for row in original_rows if row.get("dataset") == WRONG_DATASET}
    new_suffixes = {_key(row, fields)[1:] for row in replacement_rows}
    if old_suffixes != new_suffixes:
        raise ValueError(
            f"{family}: replacement configuration keys differ; "
            f"missing={sorted(old_suffixes - new_suffixes)}, "
            f"extra={sorted(new_suffixes - old_suffixes)}"
        )
    replacement_keys = [_key(row, fields) for row in replacement_rows]
    if len(set(replacement_keys)) != len(replacement_keys):
        raise ValueError(f"{family}: replacement contains duplicate configuration keys")
    merged = [dict(row) for row in original_rows if row.get("dataset") != WRONG_DATASET]
    merged.extend(dict(row) for row in replacement_rows)
    if len(merged) != len(original_rows):
        raise AssertionError(f"{family}: row count changed during replacement")
    return merged, {
        "removed_rows": len(old_suffixes),
        "inserted_rows": len(new_suffixes),
        "old_keys": sorted("|".join(key) for key in old_suffixes),
        "new_keys": sorted("|".join(key) for key in new_suffixes),
    }


def _merge_feature_report(source: Path, replacement: Path, output: Path) -> Dict[str, Any]:
    fields, source_rows = read_csv(source)
    _, replacement_rows = read_csv(replacement)
    merged, audit = merge_dataset_rows("feature_report", source_rows, replacement_rows)
    write_csv(output, fields, merged)
    return audit


def _annotate_config(
    path: Path,
    family: str,
    source: Path,
    replacement: Path,
    selected_configs: Path | None = None,
) -> None:
    payload = read_json(path)
    payload["dataset_repair"] = {
        "family": family,
        "removed_dataset": WRONG_DATASET,
        "inserted_dataset": CORRECT_DATASET,
        "source_bundle": str(source.resolve()),
        "replacement_bundle": str(replacement.resolve()),
        "derived_outputs_regenerated": family in {"benchmark", "ablation", "sensitivity"},
    }
    if selected_configs is not None and isinstance(payload.get("arguments"), dict):
        if "selected_configs" in payload["arguments"]:
            payload["arguments"]["selected_configs"] = str(selected_configs.resolve())
    if isinstance(payload.get("arguments"), dict) and "output_dir" in payload["arguments"]:
        payload["arguments"]["output_dir"] = str(path.parent.resolve())
    write_json(path, payload)


def _args_from_config(path: Path, output: Path, selected_configs: Path | None) -> argparse.Namespace:
    payload = read_json(path)
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"Missing arguments object in {path}")
    args = SimpleNamespace(**arguments)
    args.output_dir = str(output)
    args.datasets = [CORRECT_DATASET]
    args.dataset_types = ["all"]
    args.resume = True
    args.fail_on_error = True
    if selected_configs is not None:
        args.selected_configs = str(selected_configs)
    return args


def _import_runner() -> Any:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import run_paper_experiments as runner

    return runner


def regenerate_benchmark_or_ablation(
    runner: Any,
    family: str,
    output: Path,
    config_path: Path,
    merged_rows: Sequence[Mapping[str, str]],
) -> None:
    args = _args_from_config(config_path, output, None)
    original_grid = runner._run_grid
    runner._run_grid = lambda *_args, **_kwargs: ([dict(row) for row in merged_rows], [])
    try:
        if family == "benchmark":
            runner.run_benchmark(args)
        else:
            runner.run_ablation(args)
    finally:
        runner._run_grid = original_grid


def regenerate_sensitivity(
    runner: Any,
    output: Path,
    config_path: Path,
    merged_rows: Sequence[Mapping[str, str]],
) -> None:
    benchmark_dir = find_dir(output.parent, EXPERIMENT_GLOBS["benchmark"], "benchmark")
    args = _args_from_config(config_path, output, benchmark_dir / "best_auc.csv")
    dataset_types: Dict[str, str] = {}
    for row in merged_rows:
        name = str(row.get("dataset", ""))
        if name and row.get("dataset_type"):
            dataset_types[name] = str(row["dataset_type"])

    class Bundle:
        def __init__(self, name: str, dataset_type: str) -> None:
            self.name = name
            self.dataset_type = dataset_type

    dummy_bundles = [Bundle(name, dataset_types.get(name, "unknown")) for name in sorted(dataset_types)]
    original_loader = runner._load_bundles
    runner._load_bundles = lambda _args: dummy_bundles
    try:
        runner.run_sensitivity(args)
    finally:
        runner._load_bundles = original_loader


def _replace_complementarity_details(source: Path, replacement: Path, output: Path) -> None:
    source_value = read_json(source)
    replacement_value = read_json(replacement)
    if WRONG_DATASET not in source_value or CORRECT_DATASET not in replacement_value:
        raise ValueError("Complementarity details do not contain the expected dataset keys")
    if CORRECT_DATASET in source_value:
        raise ValueError("Complementarity details already contain the correct dataset")
    source_value.pop(WRONG_DATASET)
    source_value[CORRECT_DATASET] = replacement_value[CORRECT_DATASET]
    write_json(output, source_value)


def _replace_scores(source_dir: Path, replacement_dir: Path, output_dir: Path) -> Dict[str, Any]:
    source_scores = source_dir / "scores"
    replacement_scores = replacement_dir / "scores"
    output_scores = output_dir / "scores"
    old = sorted(source_scores.glob(f"{WRONG_DATASET}__*.npz")) if source_scores.is_dir() else []
    new = sorted(replacement_scores.glob(f"{CORRECT_DATASET}__*.npz")) if replacement_scores.is_dir() else []
    if len(new) != 1:
        raise ValueError(f"Expected one corrected benchmark score archive, found {new}")
    for path in output_scores.glob(f"{WRONG_DATASET}__*.npz"):
        path.unlink()
    output_scores.mkdir(parents=True, exist_ok=True)
    shutil.copy2(new[0], output_scores / new[0].name)
    return {"removed_archives": [path.name for path in old], "inserted_archive": new[0].name}


def _load_baselines(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    _, rows = read_csv(path)
    required = {"dataset", "method", "auc", "ap"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Baseline long table must contain {sorted(required)}: {path}")
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        method = str(row["method"])
        if method in MATRIX_METHODS and method != "MFTAD":
            key = (str(row["dataset"]), method)
            if key in lookup:
                raise ValueError(f"Duplicate baseline row: {key}")
            lookup[key] = row
    return lookup


def rebuild_matrices(
    source_root: Path,
    output_root: Path,
    baseline_long: Path,
) -> Dict[str, Any]:
    benchmark_fields, benchmark_rows = read_csv(output_root / "01_fixed20_benchmark" / "best_auc.csv")
    benchmark = {str(row["dataset"]): row for row in benchmark_rows}
    if len(benchmark) != 20 or WRONG_DATASET in benchmark or CORRECT_DATASET not in benchmark:
        raise ValueError("Corrected benchmark must contain exactly 20 datasets including chess 227")
    baselines = _load_baselines(baseline_long)
    matrix_info: Dict[str, Any] = {}
    for metric, filename in (("auc", "selected_auc_matrix.csv"), ("ap", "selected_ap_matrix.csv")):
        fields, old_rows = read_csv(source_root / filename)
        if fields[0] != "dataset" or not set(MATRIX_METHODS).issubset(fields):
            raise ValueError(f"Unexpected matrix columns in {filename}: {fields}")
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for old in old_rows:
            old_name = str(old["dataset"])
            name = CORRECT_DATASET if old_name == WRONG_DATASET else old_name
            if name in seen:
                raise ValueError(f"Duplicate matrix dataset after repair: {name}")
            seen.add(name)
            values: Dict[str, Any] = {"dataset": name}
            for method in BASELINE_METHODS:
                row = baselines.get((name, method))
                if row is None:
                    raise ValueError(f"Missing full-precision baseline for {(name, method)}")
                values[method] = row[metric]
            values["MFTAD"] = benchmark[name][metric]
            rows.append(values)
        if seen != set(benchmark):
            raise ValueError(f"{filename} dataset set differs from corrected benchmark")
        write_csv(output_root / filename, fields, rows)
        matrix_info[metric] = {
            "path": str((output_root / filename).resolve()),
            "sha256": sha256(output_root / filename),
            "rows": len(rows),
        }
    return matrix_info


def rebuild_nemenyi(output_root: Path) -> Dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from rebuild_nemenyi import rebuild

    return rebuild(
        output_root / "selected_auc_matrix.csv",
        output_root / "selected_ap_matrix.csv",
        output_root / "nemenyi",
        source_root=output_root,
    )


def _scan_stale(root: Path) -> List[str]:
    stale: List[str] = []
    needle = WRONG_DATASET.encode("utf-8")
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "dataset_repair_manifest.json":
            continue
        relative = path.relative_to(root)
        # Config files preserve historical command/provenance metadata, and
        # the runtime bundle is outside this single-dataset repair scope. They
        # must not invalidate the corrected benchmark/ablation artifacts.
        if path.name == "run_config.json" or (
            relative.parts and relative.parts[0].startswith("08_")
        ):
            continue
        try:
            if needle in path.read_bytes():
                stale.append(str(relative))
        except OSError:
            continue
    return sorted(stale)


def _validate_bundle(root: Path) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for family, filename in PRIMARY_FILES.items():
        directory = find_dir(root, EXPERIMENT_GLOBS[family], family)
        _, rows = read_csv(directory / filename)
        datasets = _dataset_set(rows)
        if len(datasets) != 20 or WRONG_DATASET in datasets or CORRECT_DATASET not in datasets:
            raise ValueError(f"{family}: expected 20 corrected datasets, observed {sorted(datasets)}")
        checks[family] = {"rows": len(rows), "datasets": len(datasets)}
    fields, auc_rows = read_csv(root / "selected_auc_matrix.csv")
    _, ap_rows = read_csv(root / "selected_ap_matrix.csv")
    if len(auc_rows) != 20 or len(ap_rows) != 20:
        raise ValueError("Root AUC/AP matrices must each contain 20 rows")
    if WRONG_DATASET in {row["dataset"] for row in auc_rows + ap_rows}:
        raise ValueError("Wrong dataset remains in root matrices")
    stale = _scan_stale(root)
    if stale:
        raise ValueError(f"Stale wrong-dataset references remain: {stale}")
    checks["matrices"] = {"auc_rows": len(auc_rows), "ap_rows": len(ap_rows)}
    checks["stale_references"] = []
    return checks


def repair(source_root: Path, replacement_root: Path, output_root: Path, baseline_long: Path, apply: bool) -> Dict[str, Any]:
    source_dirs = {
        family: find_dir(source_root, EXPERIMENT_GLOBS[family], family)
        for family in EXPERIMENT_GLOBS
    }
    replacement_dirs = {
        family: find_dir(replacement_root, EXPERIMENT_GLOBS[family], f"replacement {family}")
        for family in PRIMARY_FILES
    }
    merged_rows: Dict[str, List[Dict[str, str]]] = {}
    audit: Dict[str, Any] = {"source": str(source_root.resolve()), "replacement": str(replacement_root.resolve())}
    for family, filename in PRIMARY_FILES.items():
        _, old_rows = read_csv(source_dirs[family] / filename)
        _, new_rows = read_csv(replacement_dirs[family] / filename)
        merged_rows[family], audit[family] = merge_dataset_rows(family, old_rows, new_rows)
    _, source_feature = read_csv(source_dirs["benchmark"] / "feature_type_report.csv")
    _, replacement_feature = read_csv(replacement_dirs["benchmark"] / "feature_type_report.csv")
    expected_source = (set(PAPER_DATASET_FILES) - {CORRECT_DATASET}) | {WRONG_DATASET}
    if _dataset_set(source_feature) != expected_source:
        raise ValueError("Source benchmark feature report does not match the pre-repair 20 datasets")
    if _dataset_set(replacement_feature) != {CORRECT_DATASET}:
        raise ValueError("Corrected benchmark feature report must contain chess 227 only")
    audit["apply"] = bool(apply)
    audit["expected_dataset_count"] = 20
    if not apply:
        return audit
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    shutil.copytree(source_root, output_root, ignore=shutil.ignore_patterns("dataset_repair_work"))
    nemenyi = output_root / "nemenyi"
    if nemenyi.exists():
        shutil.rmtree(nemenyi)
    for family, filename in PRIMARY_FILES.items():
        target_dir = find_dir(output_root, EXPERIMENT_GLOBS[family], family)
        fields, _ = read_csv(source_dirs[family] / filename)
        write_csv(target_dir / filename, fields, merged_rows[family])
    for family in PRIMARY_FILES:
        target_dir = find_dir(output_root, EXPERIMENT_GLOBS[family], family)
        fields, _ = read_csv(source_dirs[family] / "feature_type_report.csv")
        _, replacement_rows = read_csv(replacement_dirs[family] / "feature_type_report.csv")
        _, old_rows = read_csv(source_dirs[family] / "feature_type_report.csv")
        merged, _feature_audit = merge_dataset_rows("feature_report", old_rows, replacement_rows)
        write_csv(target_dir / "feature_type_report.csv", fields, merged)
    target_benchmark = find_dir(output_root, EXPERIMENT_GLOBS["benchmark"], "benchmark")
    audit["scores"] = _replace_scores(
        source_dirs["benchmark"], replacement_dirs["benchmark"], target_benchmark
    )
    target_complement = find_dir(output_root, EXPERIMENT_GLOBS["complementarity"], "complementarity")
    _replace_complementarity_details(
        source_dirs["complementarity"] / "complementarity_details.json",
        replacement_dirs["complementarity"] / "complementarity_details.json",
        target_complement / "complementarity_details.json",
    )
    runner = _import_runner()
    regenerate_benchmark_or_ablation(
        runner, "benchmark", target_benchmark, target_benchmark / "run_config.json", merged_rows["benchmark"]
    )
    target_ablation = find_dir(output_root, EXPERIMENT_GLOBS["ablation"], "ablation")
    regenerate_benchmark_or_ablation(
        runner, "ablation", target_ablation, target_ablation / "run_config.json", merged_rows["ablation"]
    )
    target_sensitivity = find_dir(output_root, EXPERIMENT_GLOBS["sensitivity"], "sensitivity")
    regenerate_sensitivity(
        runner, target_sensitivity, target_sensitivity / "run_config.json", merged_rows["sensitivity"]
    )
    for family in ("benchmark", "ablation", "sensitivity", "invariance", "complementarity"):
        target = find_dir(output_root, EXPERIMENT_GLOBS[family], family)
        _annotate_config(
            target / "run_config.json",
            family,
            source_dirs[family],
            replacement_dirs[family],
            selected_configs=(target_benchmark / "best_auc.csv"),
        )
    audit["matrices"] = rebuild_matrices(source_root, output_root, baseline_long)
    audit["nemenyi"] = rebuild_nemenyi(output_root)
    audit["validation"] = _validate_bundle(output_root)
    write_json(output_root / "dataset_repair_manifest.json", audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-long", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Write a fresh corrected bundle")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = args.source_root.resolve()
    replacement_root = args.replacement_root.resolve()
    output_root = args.output_root.resolve()
    baseline_long = args.baseline_long.resolve()
    if not source_root.is_dir() or not replacement_root.is_dir() or not baseline_long.is_file():
        raise FileNotFoundError("Source root, replacement root, and baseline-long must exist")
    audit = repair(source_root, replacement_root, output_root, baseline_long, bool(args.apply))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
