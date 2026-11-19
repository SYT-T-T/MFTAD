"""Build aligned full-precision AUC/AP matrices from fresh MFTAD results.

The historical baseline long table is treated as an immutable supplied input.
KFGOD can be added from a fresh fixed-delta one-run real-runtime CSV, and MFTAD
is taken from the fresh benchmark's fixed rows. The script fails closed on
missing, duplicate, non-finite, or misaligned data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Support both ``python file.py`` and package imports.
    from .paper_conventions import PAPER_DATASET_FILES
except ImportError:  # pragma: no cover - script execution path
    from paper_conventions import PAPER_DATASET_FILES


BASELINE_METHODS: Tuple[str, ...] = (
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
EXTENDED_BASELINE_METHODS: Tuple[str, ...] = (*BASELINE_METHODS, "KFGOD")
MATRIX_METHODS: Tuple[str, ...] = (*BASELINE_METHODS, "MFTAD")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _finite_text(row: Mapping[str, str], field: str, context: str) -> str:
    text = str(row.get(field, "")).strip()
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"Missing or nonnumeric {field} for {context}.") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} for {context}.")
    return text


def _unique_lookup(
    rows: Iterable[Mapping[str, str]],
    key_fields: Sequence[str],
    label: str,
) -> Dict[Tuple[str, ...], Mapping[str, str]]:
    lookup: Dict[Tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")).strip() for field in key_fields)
        if any(not value for value in key):
            raise ValueError(f"{label} row has an empty key field: {key_fields}.")
        if key in lookup:
            raise ValueError(f"Duplicate {label} row: {key}.")
        lookup[key] = row
    return lookup


def _read_dataset_list(path: Optional[Path]) -> Tuple[str, ...]:
    """Read an explicit paper dataset order, or use the legacy registry."""

    if path is None:
        return tuple(PAPER_DATASET_FILES)
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset list does not exist: {resolved}")
    names = tuple(
        line.strip()
        for line in resolved.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not names or len(set(names)) != len(names):
        raise ValueError("Dataset list must contain unique, non-empty names.")
    return names


def build_matrices(
    benchmark_csv: Path,
    baseline_long: Path,
    output_dir: Path,
    kfgod_runtime_csv: Optional[Path] = None,
    dataset_list: Optional[Path] = None,
) -> Dict[str, object]:
    benchmark_csv = benchmark_csv.resolve()
    baseline_long = baseline_long.resolve()
    if not benchmark_csv.is_file() or not baseline_long.is_file():
        raise FileNotFoundError("Benchmark CSV and baseline long table must exist.")

    datasets = _read_dataset_list(dataset_list)
    _, benchmark_rows = _read_rows(benchmark_csv)
    benchmark = _unique_lookup(benchmark_rows, ("dataset",), "benchmark")
    expected_datasets = set(datasets)
    observed_datasets = {key[0] for key in benchmark}
    if observed_datasets != expected_datasets:
        missing = sorted(expected_datasets - observed_datasets)
        extra = sorted(observed_datasets - expected_datasets)
        raise ValueError(
            "Fresh benchmark must match the paper's 20-dataset protocol; "
            f"missing={missing}, extra={extra}."
        )

    _, baseline_rows = _read_rows(baseline_long)
    relevant_baselines = [
        row for row in baseline_rows if str(row.get("method", "")) in BASELINE_METHODS
    ]
    baselines = _unique_lookup(
        relevant_baselines, ("dataset", "method"), "baseline"
    )
    kfgod_rows: Dict[Tuple[str, str], Mapping[str, str]] = {}
    if kfgod_runtime_csv is not None:
        _, runtime_rows = _read_rows(kfgod_runtime_csv.resolve())
        selected_runtime = [
            row
            for row in runtime_rows
            if str(row.get("method", "")).strip().casefold() == "kfgod"
        ]
        if not selected_runtime:
            raise ValueError("KFGOD runtime CSV contains no KFGOD rows.")
        kfgod_rows = _unique_lookup(
            selected_runtime, ("dataset", "method"), "KFGOD runtime"
        )

    output_dir = output_dir.resolve()
    targets = {
        "auc": output_dir / "selected_auc_matrix.csv",
        "ap": output_dir / "selected_ap_matrix.csv",
        "manifest": output_dir / "comparison_matrix_manifest.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite comparison-matrix artifacts: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_rows: Dict[str, List[Dict[str, str]]] = {"auc": [], "ap": []}
    for dataset in datasets:
        relation_row = benchmark[(dataset,)]
        for metric in ("auc", "ap"):
            output_row: Dict[str, str] = {"dataset": dataset}
            for method in BASELINE_METHODS:
                baseline = baselines.get((dataset, method))
                if baseline is None:
                    raise ValueError(
                        f"Missing full-precision baseline for {(dataset, method)}."
                    )
                error_text = str(baseline.get("error", "")).strip()
                if error_text:
                    raise ValueError(
                        f"Baseline {(dataset, method)} has an error: {error_text}"
                    )
                output_row[method] = _finite_text(
                    baseline, metric, f"baseline {(dataset, method)}"
                )
            if kfgod_runtime_csv is not None:
                kfgod = kfgod_rows.get((dataset, "KFGOD"))
                if kfgod is None:
                    kfgod = kfgod_rows.get((dataset, "kfgod"))
                if kfgod is None:
                    raise ValueError(f"Missing KFGOD runtime result for {dataset}.")
                if str(kfgod.get("error", "")).strip():
                    raise ValueError(
                        f"KFGOD runtime result for {dataset} has an error: "
                        f"{kfgod['error']}"
                    )
                output_row["KFGOD"] = _finite_text(
                    kfgod, metric, f"KFGOD dataset {dataset}"
                )
            output_row["MFTAD"] = _finite_text(
                relation_row, metric, f"MFTAD dataset {dataset}"
            )
            matrix_rows[metric].append(output_row)

    method_order = (
        (*BASELINE_METHODS, "KFGOD", "MFTAD")
        if kfgod_runtime_csv is not None
        else MATRIX_METHODS
    )
    fieldnames = ("dataset", *method_order)
    _write_rows(targets["auc"], fieldnames, matrix_rows["auc"])
    _write_rows(targets["ap"], fieldnames, matrix_rows["ap"])
    manifest: Dict[str, object] = {
        "status": "PASS",
        "dataset_count": len(datasets),
        "datasets": list(datasets),
        "method_order": list(method_order),
        "selection_policy": (
            "AUC and AP use the same fresh MFTAD row selected by AUC; baseline "
            "values are copied from the supplied long table without rounding."
        ),
        "inputs": {
            "benchmark": {
                "path": str(benchmark_csv),
                "sha256": _sha256(benchmark_csv),
            },
            "baseline_long": {
                "path": str(baseline_long),
                "sha256": _sha256(baseline_long),
            },
            "dataset_list": (
                {
                    "path": str(dataset_list.resolve()),
                    "sha256": _sha256(dataset_list.resolve()),
                }
                if dataset_list is not None
                else None
            ),
            "kfgod_runtime": (
                {
                    "path": str(kfgod_runtime_csv.resolve()),
                    "sha256": _sha256(kfgod_runtime_csv.resolve()),
                }
                if kfgod_runtime_csv is not None
                else None
            ),
        },
        "outputs": {
            metric: {"path": str(path), "sha256": _sha256(path)}
            for metric, path in (("auc", targets["auc"]), ("ap", targets["ap"]))
        },
    }
    targets["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--baseline-long", type=Path, required=True)
    parser.add_argument(
        "--kfgod-runtime-csv",
        type=Path,
        help=(
            "Optional fixed-delta KFGOD real-runtime CSV. When supplied, the "
            "output has 12 baselines plus MFTAD (13 methods total)."
        ),
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        help=(
            "Optional explicit dataset order (one raw dataset name per line). "
            "When omitted, the legacy Table-2 registry is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_matrices(
        args.benchmark_csv,
        args.baseline_long,
        args.output_dir,
        args.kfgod_runtime_csv,
        args.dataset_list,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
