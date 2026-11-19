"""Strict Friedman/Nemenyi rebuild from the mandated raw source directory.

This script deliberately fails closed. It will not consume values copied from a
manuscript table or a derived PDF, and it requires the AUC and AP matrices to
contain the same 20 datasets and the same ordered, unique method columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Support both ``python file.py`` and package imports.
    from .experiment_utils import plot_critical_difference, statistical_comparison
    from .paper_conventions import (
        PAPER_DATASET_COUNT,
        paper_dataset_abbreviation,
    )
except ImportError:  # pragma: no cover - script execution path
    from experiment_utils import plot_critical_difference, statistical_comparison
    from paper_conventions import PAPER_DATASET_COUNT, paper_dataset_abbreviation


REQUIRED_SINGLE_METHODS = ("ECOD", "MFGAD")
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "audit" / "derived_fixed20"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "recomputed_results" / "nemenyi"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_dataset_list(path: Optional[Path]) -> Optional[Tuple[str, ...]]:
    if path is None:
        return None
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


def _assert_under_source(path: Path, source_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Statistics input must be inside the mandated source directory: {source_root}"
        ) from error
    return resolved


def _read_matrix(
    path: Path, expected_datasets: Optional[Sequence[str]] = None
) -> Tuple[List[str], List[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 3:
        raise ValueError(f"{path} must contain a dataset column and at least two methods.")
    header = [cell.strip() for cell in rows[0]]
    if any(not cell for cell in header):
        raise ValueError(f"{path} contains an empty header cell.")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path} contains duplicate columns: {duplicates}")
    methods = header[1:]
    if len(set(methods)) != len(methods):
        raise ValueError(f"{path} contains duplicate method names.")
    folded = [method.casefold() for method in methods]
    folded_duplicates = sorted(
        {
            methods[index]
            for index, normalized in enumerate(folded)
            if folded.count(normalized) > 1
        }
    )
    if folded_duplicates:
        raise ValueError(
            f"{path} contains case-insensitive duplicate method names: "
            f"{folded_duplicates}"
        )
    for required in REQUIRED_SINGLE_METHODS:
        count = folded.count(required.casefold())
        if count != 1:
            raise ValueError(
                f"{path} must contain {required} exactly once; observed {count}."
            )
    expected_count = len(expected_datasets) if expected_datasets is not None else PAPER_DATASET_COUNT
    if len(rows) - 1 != expected_count:
        raise ValueError(
            f"{path} must contain exactly {expected_count} datasets; "
            f"observed {len(rows) - 1}."
        )
    datasets: List[str] = []
    values: List[List[float]] = []
    expected_width = len(header)
    for line_number, raw in enumerate(rows[1:], start=2):
        if len(raw) != expected_width:
            raise ValueError(
                f"{path}:{line_number} has {len(raw)} cells; expected {expected_width}."
            )
        dataset = raw[0].strip()
        if not dataset:
            raise ValueError(f"{path}:{line_number} has an empty dataset name.")
        numeric: List[float] = []
        for column, cell in zip(methods, raw[1:]):
            text = cell.strip()
            if not text:
                raise ValueError(
                    f"{path}:{line_number} is missing {column!r}."
                )
            try:
                value = float(text)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} contains a nonnumeric value for {column!r}."
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"{path}:{line_number} contains a non-finite value for {column!r}."
                )
            numeric.append(value)
        datasets.append(dataset)
        values.append(numeric)
    duplicate_datasets = sorted(
        {name for name in datasets if datasets.count(name) > 1}
    )
    if duplicate_datasets:
        raise ValueError(f"{path} contains duplicate datasets: {duplicate_datasets}")
    if expected_datasets is not None and datasets != list(expected_datasets):
        raise ValueError(
            f"{path} dataset rows do not match the explicit dataset list order."
        )
    return datasets, methods, np.asarray(values, dtype=np.float64)


def _nemenyi_groups(
    average_ranks: Mapping[str, float],
    critical_difference: float,
    tolerance: float = 1e-12,
) -> List[List[str]]:
    """Return maximal contiguous non-rejected groups in rank order."""

    ordered = sorted(average_ranks, key=lambda name: (average_ranks[name], name))
    candidates: List[Tuple[str, ...]] = []
    for start in range(len(ordered)):
        stop = start
        while (
            stop + 1 < len(ordered)
            and average_ranks[ordered[stop + 1]] - average_ranks[ordered[start]]
            <= critical_difference + tolerance
        ):
            stop += 1
        if stop > start:
            candidates.append(tuple(ordered[start : stop + 1]))
    maximal = [
        candidate
        for candidate in candidates
        if not any(
            set(candidate) < set(other)
            for other in candidates
        )
    ]
    return [list(group) for group in dict.fromkeys(maximal)]


def _latex_statistics(metric: str, result: Mapping[str, object]) -> str:
    def latex_number(value: float) -> str:
        mantissa, exponent = f"{float(value):.6g}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"

    prefix = metric.upper()
    groups = result.get("nemenyi_groups", [])
    group_text = "; ".join(
        ", ".join(str(method).replace("_", r"\_") for method in group)
        for group in groups
    )
    return "\n".join(
        (
            rf"\newcommand{{\{prefix}FriedmanStatistic}}{{{float(result['friedman_statistic']):.6f}}}",
            rf"\newcommand{{\{prefix}FriedmanPValue}}{{{latex_number(float(result['friedman_p']))}}}",
            rf"\newcommand{{\{prefix}NemenyiCD}}{{{float(result['nemenyi_critical_difference']):.6f}}}",
            rf"\newcommand{{\{prefix}NemenyiGroups}}{{{group_text}}}",
            "",
        )
    )


def rebuild(
    auc_path: Path,
    ap_path: Path,
    output: Path,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    alpha: float = 0.05,
    tie_tolerance: float = 1e-12,
    dataset_list: Optional[Path] = None,
) -> Dict[str, object]:
    auc_path = _assert_under_source(auc_path, source_root)
    ap_path = _assert_under_source(ap_path, source_root)
    explicit_datasets = _read_dataset_list(dataset_list)
    auc_datasets, auc_methods, auc_values = _read_matrix(auc_path, explicit_datasets)
    ap_datasets, ap_methods, ap_values = _read_matrix(ap_path, explicit_datasets)
    if explicit_datasets is None:
        auc_abbreviations = [paper_dataset_abbreviation(name) for name in auc_datasets]
        ap_abbreviations = [paper_dataset_abbreviation(name) for name in ap_datasets]
        if len(set(auc_abbreviations)) != PAPER_DATASET_COUNT:
            raise ValueError("AUC dataset rows do not map one-to-one to Table 2 abbreviations.")
        if len(set(ap_abbreviations)) != PAPER_DATASET_COUNT:
            raise ValueError("AP dataset rows do not map one-to-one to Table 2 abbreviations.")
    else:
        auc_abbreviations = list(auc_datasets)
        ap_abbreviations = list(ap_datasets)
    if auc_methods != ap_methods:
        raise ValueError(
            "AUC and AP must use exactly the same method order; "
            f"AUC={auc_methods}, AP={ap_methods}."
        )
    if auc_abbreviations != ap_abbreviations:
        raise ValueError(
            "AUC and AP must use exactly the same ordered dataset rows."
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "status": "PASS",
        "source_root": str(source_root.resolve()),
        "dataset_count": len(auc_abbreviations),
        "datasets": auc_abbreviations,
        "method_count": len(auc_methods),
        "method_order": auc_methods,
        "inputs": {
            "auc": {"path": str(auc_path), "sha256": _sha256(auc_path)},
            "ap": {"path": str(ap_path), "sha256": _sha256(ap_path)},
            "dataset_list": (
                {"path": str(dataset_list.resolve()), "sha256": _sha256(dataset_list.resolve())}
                if dataset_list is not None
                else None
            ),
        },
        "tie_policy": "Average ranks for exact ties; lower average rank is better.",
        "alpha": alpha,
        "tie_tolerance": tie_tolerance,
    }
    latex_parts: List[str] = []
    for metric, values in (("auc", auc_values), ("ap", ap_values)):
        result = statistical_comparison(
            values,
            auc_methods,
            alpha=alpha,
            tie_tolerance=tie_tolerance,
        )
        result["nemenyi_groups"] = _nemenyi_groups(
            result["average_ranks"],
            float(result["nemenyi_critical_difference"]),
            tolerance=tie_tolerance,
        )
        with (output / f"{metric}_statistics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        with (output / f"{metric}_average_ranks.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("method", "average_rank"))
            writer.writeheader()
            writer.writerows(
                {
                    "method": method,
                    "average_rank": result["average_ranks"][method],
                }
                for method in auc_methods
            )
        plot_critical_difference(
            result["average_ranks"],
            float(result["nemenyi_critical_difference"]),
            output / f"nemenyi_{metric}.svg",
            title=f"{metric.upper()} critical difference",
        )
        latex_parts.append(_latex_statistics(metric, result))
        manifest[metric] = {
            "friedman_statistic": result["friedman_statistic"],
            "friedman_p": result["friedman_p"],
            "nemenyi_critical_difference": result["nemenyi_critical_difference"],
            "nemenyi_groups": result["nemenyi_groups"],
        }
    (output / "nemenyi_values.tex").write_text("\n".join(latex_parts), encoding="utf-8")
    with (output / "statistics_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auc-matrix", type=Path, required=True)
    parser.add_argument("--ap-matrix", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset-list",
        type=Path,
        help="Optional explicit dataset order (one raw dataset name per line).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    manifest = rebuild(
        args.auc_matrix,
        args.ap_matrix,
        args.output_dir,
        source_root=args.source_root,
        alpha=args.alpha,
        tie_tolerance=args.tie_tolerance,
        dataset_list=args.dataset_list,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
