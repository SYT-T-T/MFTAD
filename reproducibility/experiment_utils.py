"""Standalone experiment utilities for relation-consistent MFTAD.

The helpers deliberately keep raw runs, failures, selected rows, and manifests
separate.  This makes label-based best-performance reporting explicit and
auditable instead of hiding it inside a final summary table.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.io import loadmat
    from scipy.stats import (
        friedmanchisquare,
        norm,
        rankdata,
        spearmanr,
        studentized_range,
        wilcoxon,
    )
except Exception as error:  # pragma: no cover - dependency guard
    raise ImportError(
        "experiment_utils requires scipy. Install it in the active Python environment."
    ) from error

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception as error:  # pragma: no cover - dependency guard
    raise ImportError(
        "experiment_utils requires scikit-learn. "
        "Install it in the active Python environment."
    ) from error


DATASET_SUFFIXES = {".mat", ".npz", ".csv", ".tsv"}
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "relation_mftad_mpl_cache"),
)


@dataclass
class DatasetBundle:
    name: str
    path: Path
    X: np.ndarray
    y: np.ndarray
    nominal_features: Tuple[int, ...]
    dataset_type: str
    label_mapping: Dict[object, int]


def _json_ready(value: object) -> object:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(payload: object, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_ready(payload), ensure_ascii=False, indent=2)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)


def read_json(path: Path | str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rows(
    rows: Sequence[Mapping[str, object]],
    path: Path | str,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fieldnames = ordered
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_ready(value), ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple, set))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    os.replace(temporary, target)


def read_rows(path: Path | str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_binary_labels(
    labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[object, int]]:
    """Map the minority class to anomaly label 1.

    Ties are resolved by first appearance only to make the mapping deterministic;
    a warning-worthy 50/50 dataset remains fully recorded in the manifest.
    """

    raw = np.asarray(labels).reshape(-1)
    if raw.size < 2:
        raise ValueError("At least two labels are required.")
    values, first_indices, counts = np.unique(
        raw, return_index=True, return_counts=True
    )
    if len(values) != 2:
        raise ValueError(
            f"Binary anomaly detection requires exactly two labels, got {values!r}."
        )
    anomaly_position = min(
        range(2), key=lambda index: (int(counts[index]), int(first_indices[index]))
    )
    anomaly_value = values[anomaly_position]
    normal_value = values[1 - anomaly_position]
    normalized = (raw == anomaly_value).astype(int)
    mapping = {
        normal_value.item() if hasattr(normal_value, "item") else normal_value: 0,
        anomaly_value.item() if hasattr(anomaly_value, "item") else anomaly_value: 1,
    }
    return normalized, mapping


def compute_binary_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int).reshape(-1)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != values.size:
        raise ValueError("labels and scores must have the same length.")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contain NaN or infinite values.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must contain both 0 and 1.")
    return {
        "auc": float(roc_auc_score(y, values)),
        "ap": float(average_precision_score(y, values)),
    }


def _extract_xy_from_mapping(
    mapping: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    usable = {
        key: np.asarray(value)
        for key, value in mapping.items()
        if not key.startswith("__")
    }
    normalized_keys = {
        str(key).lower(): key for key in usable
    }
    if "trandata" in normalized_keys:
        packed = np.asarray(usable[normalized_keys["trandata"]])
        if packed.ndim != 2 or packed.shape[1] < 2:
            raise ValueError(
                "trandata must be a two-dimensional matrix with at least "
                "one feature column and one label column."
            )
        labels = np.asarray(packed[:, -1]).reshape(-1)
        if np.unique(labels).size != 2:
            raise ValueError(
                "The last trandata column must contain exactly two labels."
            )
        return np.asarray(packed[:, :-1], dtype=float), labels
    x_keys = ("X", "x", "data", "Data", "features", "fea")
    y_keys = ("y", "Y", "label", "labels", "target", "gnd")
    X = next(
        (
            usable[key]
            for key in x_keys
            if key in usable and np.asarray(usable[key]).ndim == 2
        ),
        None,
    )
    y = next(
        (
            usable[key]
            for key in y_keys
            if key in usable and np.asarray(usable[key]).size > 1
        ),
        None,
    )
    if X is None:
        matrices = [
            value
            for value in usable.values()
            if value.ndim == 2 and min(value.shape) > 1
        ]
        if not matrices:
            raise ValueError("Could not locate a two-dimensional feature matrix.")
        X = max(matrices, key=lambda value: value.size)
    if y is None:
        candidates = [
            value.reshape(-1)
            for value in usable.values()
            if value.size in X.shape and np.unique(value).size == 2
        ]
        if not candidates:
            raise ValueError("Could not locate a binary label vector.")
        y = candidates[0]
    y = np.asarray(y).reshape(-1)
    X = np.asarray(X)
    if X.shape[0] != y.size and X.shape[1] == y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError(
            f"Feature/label size mismatch: X={X.shape}, y={y.shape}."
        )
    return np.asarray(X, dtype=float), y


def load_nominal_map(path: Optional[Path | str]) -> Dict[str, object]:
    if path is None:
        return {}
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    payload = read_json(target)
    if not isinstance(payload, dict):
        raise ValueError("Nominal feature map must be a JSON object.")
    return payload


def _lookup_metadata(
    dataset_name: str, nominal_map: Mapping[str, object]
) -> Optional[object]:
    candidates = (
        dataset_name,
        dataset_name.lower(),
        Path(dataset_name).stem,
        Path(dataset_name).stem.lower(),
    )
    lower_map = {str(key).lower(): value for key, value in nominal_map.items()}
    for candidate in candidates:
        if candidate in nominal_map:
            return nominal_map[candidate]
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def resolve_nominal_features(
    dataset_name: str,
    n_features: int,
    nominal_map: Mapping[str, object],
    inferred: Sequence[int] = (),
) -> Tuple[int, ...]:
    metadata = _lookup_metadata(dataset_name, nominal_map)
    if metadata is None:
        indices = tuple(sorted(set(int(item) for item in inferred)))
    else:
        if isinstance(metadata, Mapping):
            for key in (
                "nominal_features",
                "categorical_features",
                "nominal",
                "categorical",
            ):
                if key in metadata:
                    metadata = metadata[key]
                    break
            else:
                dataset_type = str(metadata.get("dataset_type", "")).lower()
                if dataset_type in {"nominal", "categorical", "pure_nominal"}:
                    metadata = "all"
        if isinstance(metadata, str):
            if metadata.lower() not in {"all", "nominal", "categorical"}:
                raise ValueError(
                    f"Unsupported nominal metadata for {dataset_name}: {metadata!r}"
                )
            indices = tuple(range(n_features))
        else:
            indices = tuple(sorted(set(int(item) for item in metadata)))
    invalid = [index for index in indices if index < 0 or index >= n_features]
    if invalid:
        raise ValueError(
            f"Invalid nominal indices for {dataset_name}: {invalid}; d={n_features}."
        )
    return indices


def dataset_type(n_features: int, nominal_features: Sequence[int]) -> str:
    count = len(set(nominal_features))
    if count == 0:
        return "numeric"
    if count == n_features:
        return "nominal"
    return "mixed"


def _load_delimited(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    try:
        import pandas as pd
    except Exception as error:  # pragma: no cover - dependency guard
        raise ImportError("CSV/TSV loading requires pandas.") from error
    separator = "\t" if path.suffix.lower() == ".tsv" else None
    frame = pd.read_csv(path, sep=separator, engine="python")
    label_candidates = [
        column
        for column in frame.columns
        if str(column).lower()
        in {"y", "label", "labels", "target", "class", "outlier"}
    ]
    label_column = label_candidates[0] if label_candidates else frame.columns[-1]
    y = frame.pop(label_column).to_numpy()
    inferred_nominal = []
    columns = []
    for index, column in enumerate(frame.columns):
        series = frame[column]
        if (
            str(series.dtype) in {"object", "category", "bool"}
            or not np.issubdtype(series.dtype, np.number)
        ):
            inferred_nominal.append(index)
            codes, _ = pd.factorize(series, sort=True)
            columns.append(codes.astype(float))
        else:
            columns.append(series.to_numpy(dtype=float))
    X = np.column_stack(columns)
    return X, y, tuple(inferred_nominal)


def load_dataset(
    path: Path | str,
    nominal_map: Optional[Mapping[str, object]] = None,
) -> DatasetBundle:
    target = Path(path)
    suffix = target.suffix.lower()
    inferred: Tuple[int, ...] = ()
    if suffix == ".mat":
        X, raw_y = _extract_xy_from_mapping(loadmat(target))
    elif suffix == ".npz":
        with np.load(target, allow_pickle=False) as archive:
            X, raw_y = _extract_xy_from_mapping(
                {key: archive[key] for key in archive.files}
            )
    elif suffix in {".csv", ".tsv"}:
        X, raw_y, inferred = _load_delimited(target)
    else:
        raise ValueError(f"Unsupported dataset suffix: {suffix}")
    y, mapping = normalize_binary_labels(raw_y)
    metadata = nominal_map or {}
    nominal = resolve_nominal_features(target.stem, X.shape[1], metadata, inferred)
    return DatasetBundle(
        name=target.stem,
        path=target.resolve(),
        X=np.asarray(X, dtype=float),
        y=y,
        nominal_features=nominal,
        dataset_type=dataset_type(X.shape[1], nominal),
        label_mapping=mapping,
    )


def discover_datasets(
    dataset_root: Path | str,
    names: Optional[Sequence[str]] = None,
) -> List[Path]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DATASET_SUFFIXES
    )
    if names:
        requested = {str(name).lower() for name in names}
        selected = [
            path
            for path in files
            if path.stem.lower() in requested or path.name.lower() in requested
        ]
        missing = sorted(
            requested
            - {path.stem.lower() for path in selected}
            - {path.name.lower() for path in selected}
        )
        if missing:
            raise FileNotFoundError(f"Datasets not found: {missing}")
        files = selected
    if not files:
        raise FileNotFoundError(f"No supported datasets found under {root}.")
    return files


def parse_number_grid(text: str, cast=float) -> List[object]:
    values: List[float] = []
    for item in str(text).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            values.append(float(stripped))
            continue
        fields = [float(field) for field in stripped.split(":")]
        if len(fields) != 3:
            raise ValueError(f"Invalid range {stripped!r}; use start:stop:step.")
        start, stop, step = fields
        if step <= 0.0 or stop < start:
            raise ValueError(f"Invalid increasing range {stripped!r}.")
        count = int(math.floor((stop - start) / step + 1e-10)) + 1
        values.extend(start + index * step for index in range(count))
    if not values:
        raise ValueError("Grid cannot be empty.")
    converted = [cast(round(value, 12)) for value in values]
    return list(dict.fromkeys(converted))


def select_best_per_dataset(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    dataset_key: str = "dataset",
) -> List[Dict[str, object]]:
    """Select label-based per-dataset oracle rows with deterministic tie breaks."""

    if metric not in {"auc", "ap"}:
        raise ValueError("metric must be 'auc' or 'ap'.")
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[dataset_key]), []).append(row)
    selected = []
    for dataset in sorted(grouped):
        candidates = [
            row
            for row in grouped[dataset]
            if np.isfinite(float(row.get(metric, float("nan"))))
        ]
        if not candidates:
            continue
        other = "ap" if metric == "auc" else "auc"
        best = max(
            candidates,
            key=lambda row: (
                float(row[metric]),
                float(row.get(other, float("-inf"))),
                -float(row.get("runtime_seconds", 0.0)),
                str(row.get("config_id", "")),
            ),
        )
        output = dict(best)
        output["selection_metric"] = metric
        output["oracle_selection"] = True
        selected.append(output)
    return selected


def statistical_comparison(
    values: np.ndarray,
    method_names: Sequence[str],
    alpha: float = 0.05,
    tie_tolerance: float = 1e-12,
) -> Dict[str, object]:
    """Omnibus ranks plus paired Wilcoxon-Holm tests on complete rows."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (datasets, methods).")
    n_datasets, n_methods = matrix.shape
    if len(method_names) != n_methods:
        raise ValueError("method_names length does not match values.")
    if n_datasets < 2 or n_methods < 2:
        raise ValueError("At least two datasets and methods are required.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Statistical comparison requires complete finite rows.")
    ranks = np.vstack(
        [rankdata(-row, method="average") for row in matrix]
    )
    average_ranks = np.mean(ranks, axis=0)
    if n_methods >= 3:
        try:
            friedman_stat, friedman_p = friedmanchisquare(
                *[matrix[:, index] for index in range(n_methods)]
            )
        except ValueError as error:
            if "identical" not in str(error).lower():
                raise
            friedman_stat, friedman_p = 0.0, 1.0
    else:
        friedman_stat, friedman_p = float("nan"), float("nan")
    denominator = n_datasets * (n_methods - 1) - float(friedman_stat)
    if (
        n_methods >= 3
        and np.isfinite(friedman_stat)
        and denominator > EPSILON_LOCAL
    ):
        iman_stat = (
            (n_datasets - 1) * float(friedman_stat) / denominator
        )
        from scipy.stats import f as f_distribution

        iman_p = float(
            f_distribution.sf(
                iman_stat,
                n_methods - 1,
                (n_methods - 1) * (n_datasets - 1),
            )
        )
    else:
        iman_stat, iman_p = float("nan"), float("nan")
    # scipy's infinite-df studentized-range inversion can take minutes on some
    # Windows builds.  These are the standard Nemenyi q_0.05 values already
    # divided by sqrt(2), as used in the critical-difference formula.
    q_005 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
        16: 3.426,
        17: 3.458,
        18: 3.489,
        19: 3.517,
        20: 3.544,
    }
    if abs(alpha - 0.05) <= 1e-12 and n_methods in q_005:
        q_alpha = q_005[n_methods]
    else:
        q_alpha = float(
            studentized_range.ppf(1.0 - alpha, n_methods, np.inf)
            / math.sqrt(2.0)
        )
    critical_difference = q_alpha * math.sqrt(
        n_methods * (n_methods + 1) / (6.0 * n_datasets)
    )
    standard_error = math.sqrt(
        n_methods * (n_methods + 1) / (6.0 * n_datasets)
    )
    pairwise = []
    wilcoxon_rows = []
    for first in range(n_methods):
        for second in range(first + 1, n_methods):
            difference = abs(average_ranks[first] - average_ranks[second])
            z_value = difference / max(standard_error, EPSILON_LOCAL)
            p_value = float(2.0 * norm.sf(z_value))
            pairwise.append(
                {
                    "method_a": str(method_names[first]),
                    "method_b": str(method_names[second]),
                    "rank_difference": float(difference),
                    "unadjusted_normal_p": p_value,
                    "nemenyi_significant": bool(
                        difference > critical_difference
                    ),
                }
            )
            differences = matrix[:, first] - matrix[:, second]
            wins = int(np.sum(differences > tie_tolerance))
            losses = int(np.sum(differences < -tie_tolerance))
            ties = int(n_datasets - wins - losses)
            nonzero = differences[np.abs(differences) > tie_tolerance]
            if len(nonzero):
                paired_p = float(
                    wilcoxon(
                        nonzero,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                )
                absolute_ranks = rankdata(np.abs(nonzero), method="average")
                positive = float(absolute_ranks[nonzero > 0].sum())
                negative = float(absolute_ranks[nonzero < 0].sum())
                rank_biserial = (positive - negative) / max(
                    positive + negative, EPSILON_LOCAL
                )
            else:
                paired_p = 1.0
                rank_biserial = 0.0
            wilcoxon_rows.append(
                {
                    "method_a": str(method_names[first]),
                    "method_b": str(method_names[second]),
                    "mean_difference_a_minus_b": float(differences.mean()),
                    "median_difference_a_minus_b": float(np.median(differences)),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "paired_rank_biserial": float(rank_biserial),
                    "wilcoxon_p": paired_p,
                }
            )
    order = sorted(range(len(wilcoxon_rows)), key=lambda index: wilcoxon_rows[index]["wilcoxon_p"])
    running_adjusted = 0.0
    comparison_count = len(order)
    for position, index in enumerate(order):
        raw_p = float(wilcoxon_rows[index]["wilcoxon_p"])
        adjusted = min(1.0, (comparison_count - position) * raw_p)
        running_adjusted = max(running_adjusted, adjusted)
        wilcoxon_rows[index]["holm_adjusted_p"] = running_adjusted
        wilcoxon_rows[index]["holm_reject"] = bool(running_adjusted <= alpha)
    return {
        "n_datasets": n_datasets,
        "n_methods": n_methods,
        "average_ranks": {
            str(name): float(rank)
            for name, rank in zip(method_names, average_ranks)
        },
        "friedman_statistic": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "iman_davenport_statistic": float(iman_stat),
        "iman_davenport_p": float(iman_p),
        "alpha": float(alpha),
        "nemenyi_cd": float(critical_difference),
        "nemenyi_critical_difference": float(critical_difference),
        "pairwise": pairwise,
        "wilcoxon_holm": wilcoxon_rows,
        "effect_size": "paired rank-biserial correlation; positive favors method_a",
        "tie_tolerance": float(tie_tolerance),
        "rank_matrix": ranks.tolist(),
    }


EPSILON_LOCAL = 1e-12


def complementarity_record(
    labels: np.ndarray,
    component_scores: Mapping[str, np.ndarray],
    top_fraction: float = 0.1,
    top_k: Optional[int] = None,
) -> Dict[str, object]:
    y = np.asarray(labels, dtype=int)
    names = list(component_scores)
    if not names:
        raise ValueError("component_scores cannot be empty.")
    n = len(y)
    requested_top_k = top_k
    top_k = (
        max(1, min(n, int(requested_top_k)))
        if requested_top_k is not None
        else max(1, int(math.ceil(n * top_fraction)))
    )
    top_sets = {
        name: set(
            np.argsort(-np.asarray(component_scores[name]), kind="mergesort")[:top_k]
        )
        for name in names
    }
    spearman = {}
    overlap = {}
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            coefficient = spearmanr(
                component_scores[first], component_scores[second]
            ).statistic
            key = f"{first}__{second}"
            spearman[key] = (
                0.0 if not np.isfinite(coefficient) else float(coefficient)
            )
            overlap[key] = float(
                len(top_sets[first] & top_sets[second]) / top_k
            )
    anomaly_indices = set(np.flatnonzero(y == 1))
    recalls = {
        name: (
            float(len(indices & anomaly_indices) / len(anomaly_indices))
            if anomaly_indices
            else float("nan")
        )
        for name, indices in top_sets.items()
    }
    stacked = np.vstack([component_scores[name] for name in names])
    dominant_positions = np.argmax(stacked, axis=0)
    dominant_counts = {
        name: int(np.sum(dominant_positions == index))
        for index, name in enumerate(names)
    }
    detected_true_anomalies = anomaly_indices & set().union(*top_sets.values())
    detected_dominant_counts = {
        name: int(
            sum(dominant_positions[index] == position for index in detected_true_anomalies)
        )
        for position, name in enumerate(names)
    }
    exclusive_anomalies = {}
    for name in names:
        others = set().union(
            *(top_sets[other] for other in names if other != name)
        )
        exclusive_anomalies[name] = int(
            len((top_sets[name] - others) & anomaly_indices)
        )
    return {
        "top_k": top_k,
        "top_fraction": (top_k / n),
        "top_k_rule": "explicit" if requested_top_k is not None else "fraction",
        "spearman": spearman,
        "top_k_overlap": overlap,
        "top_k_anomaly_recall": recalls,
        "dominant_cue_counts": dominant_counts,
        "detected_true_anomaly_count": len(detected_true_anomalies),
        "detected_true_anomaly_dominant_cue_counts": detected_dominant_counts,
        "exclusive_top_k_anomalies": exclusive_anomalies,
    }


def system_manifest(extra: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    packages = {"numpy": np.__version__}
    for name in ("scipy", "sklearn", "pandas", "matplotlib", "numba", "psutil"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = "unavailable"
    payload: Dict[str, object] = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }
    if extra:
        payload.update(extra)
    return payload


def _write_svg(target: Path, lines: Sequence[str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _plot_sensitivity_svg(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    metric: str,
    output: Path | str,
    group_key: str,
    y_limits: Optional[Tuple[float, float]],
    x_scale: str,
) -> None:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    all_x = sorted({float(row[x_key]) for row in rows})
    if not all_x:
        raise ValueError("Sensitivity plots require at least one x value.")
    if x_scale not in {"linear", "log"}:
        raise ValueError("x_scale must be 'linear' or 'log'.")
    if x_scale == "log" and any(value <= 0.0 for value in all_x):
        raise ValueError("Log-scaled sensitivity plots require positive x values.")
    aggregate: List[float] = []
    for x_value in all_x:
        values = [
            float(row[metric])
            for row in rows
            if float(row[x_key]) == x_value
            and np.isfinite(float(row[metric]))
        ]
        aggregate.append(float(np.mean(values)) if values else float("nan"))
    finite_y = [
        float(row[metric])
        for row in rows
        if np.isfinite(float(row[metric]))
    ]
    if not finite_y:
        raise ValueError("Sensitivity plots require finite metric values.")
    if y_limits is None:
        lower = min(finite_y)
        upper = max(finite_y)
        span = upper - lower
        padding = 0.05 * span if span > 0.0 else 0.05
        lower -= padding
        upper += padding
    else:
        lower, upper = (float(y_limits[0]), float(y_limits[1]))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("y_limits must be two finite increasing values.")
    transformed_x = [math.log10(value) if x_scale == "log" else value for value in all_x]
    x_lower, x_upper = min(transformed_x), max(transformed_x)
    if x_upper <= x_lower:
        x_lower -= 0.5
        x_upper += 0.5
    width, height = 720.0, 460.0
    left, right, top, bottom = 76.0, 28.0, 30.0, 66.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_position(value: float) -> float:
        transformed = math.log10(value) if x_scale == "log" else value
        return left + (transformed - x_lower) / (x_upper - x_lower) * plot_width

    def y_position(value: float) -> float:
        return top + (upper - value) / (upper - lower) * plot_height

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="460" viewBox="0 0 720 460">',
        "<desc>Per-dataset sensitivity traces are translucent; the dataset mean is opaque. No best setting is marked.</desc>",
        '<rect width="720" height="460" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#222">',
    ]
    for tick_index in range(6):
        tick_value = lower + (upper - lower) * tick_index / 5.0
        y = y_position(tick_value)
        lines.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{width-right:.2f}" y2="{y:.2f}" stroke="#d9dde3" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-9:.2f}" y="{y+4:.2f}" text-anchor="end">{tick_value:.3f}</text>'
        )
    for value in all_x:
        x = x_position(value)
        label = f"{value:g}"
        lines.append(
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{height-bottom:.2f}" stroke="#eef0f3" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{height-bottom+22:.2f}" text-anchor="middle">{html.escape(label)}</text>'
        )
    lines.extend(
        (
            f'<line x1="{left:.2f}" y1="{height-bottom:.2f}" x2="{width-right:.2f}" y2="{height-bottom:.2f}" stroke="#222" stroke-width="1.2"/>',
            f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{height-bottom:.2f}" stroke="#222" stroke-width="1.2"/>',
        )
    )
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: float(row[x_key]))
        points = " ".join(
            f"{x_position(float(row[x_key])):.2f},{y_position(float(row[metric])):.2f}"
            for row in ordered
            if np.isfinite(float(row[metric]))
        )
        if points:
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="#7d8793" stroke-opacity="0.25" stroke-width="1.1"/>'
            )
    aggregate_points = " ".join(
        f"{x_position(x_value):.2f},{y_position(value):.2f}"
        for x_value, value in zip(all_x, aggregate)
        if np.isfinite(value)
    )
    lines.append(
        f'<polyline points="{aggregate_points}" fill="none" stroke="#1f5aa6" stroke-width="2.8"/>'
    )
    for x_value, value in zip(all_x, aggregate):
        if np.isfinite(value):
            lines.append(
                f'<circle cx="{x_position(x_value):.2f}" cy="{y_position(value):.2f}" r="3.2" fill="#1f5aa6"/>'
            )
    x_label = f"{x_key} (log scale)" if x_scale == "log" else x_key
    lines.extend(
        (
            f'<text x="{left + plot_width/2:.2f}" y="{height-16:.2f}" text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text x="18" y="{top + plot_height/2:.2f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height/2:.2f})">{html.escape(metric.upper())}</text>',
            f'<line x1="{width-185:.2f}" y1="{top+12:.2f}" x2="{width-145:.2f}" y2="{top+12:.2f}" stroke="#1f5aa6" stroke-width="2.8"/>',
            f'<text x="{width-137:.2f}" y="{top+16:.2f}">Dataset mean</text>',
            "</g>",
            "</svg>",
        )
    )
    _write_svg(Path(output), lines)


def plot_sensitivity(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    metric: str,
    output: Path | str,
    group_key: str = "dataset",
    y_limits: Optional[Tuple[float, float]] = None,
    x_scale: str = "linear",
    reference_x: Optional[float] = None,
) -> None:
    target = Path(output)
    if target.suffix.lower() == ".svg":
        _plot_sensitivity_svg(
            rows,
            x_key,
            metric,
            target,
            group_key,
            y_limits,
            x_scale,
        )
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - dependency guard
        raise ImportError("Plotting requires matplotlib.") from error
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    all_x = sorted({float(row[x_key]) for row in rows})
    if x_scale not in {"linear", "log"}:
        raise ValueError("x_scale must be 'linear' or 'log'.")
    if x_scale == "log" and any(value <= 0.0 for value in all_x):
        raise ValueError("Log-scaled sensitivity plots require positive x values.")
    aggregate = []
    for x_value in all_x:
        values = [
            float(row[metric])
            for row in rows
            if float(row[x_key]) == x_value
            and np.isfinite(float(row[metric]))
        ]
        aggregate.append(float(np.mean(values)) if values else np.nan)
    for group, group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda row: float(row[x_key]))
        axis.plot(
            [float(row[x_key]) for row in ordered],
            [float(row[metric]) for row in ordered],
            color="#9aa0a6",
            alpha=0.25,
            linewidth=0.8,
        )
    axis.plot(
        all_x,
        aggregate,
        marker="o",
        color="#1f5aa6",
        linewidth=2.2,
        label="Dataset mean",
    )
    if reference_x is not None and all_x and all_x[0] <= reference_x <= all_x[-1]:
        axis.axvline(
            float(reference_x),
            color="#b33a3a",
            linestyle="--",
            linewidth=1.3,
            label="Frozen setting",
        )
    axis.set_xscale(x_scale)
    axis.set_xlabel(f"{x_key} (log scale)" if x_scale == "log" else x_key)
    axis.set_ylabel(metric.upper())
    if y_limits is not None:
        lower, upper = (float(y_limits[0]), float(y_limits[1]))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("y_limits must be two finite increasing values.")
        axis.set_ylim(lower, upper)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=220)
    plt.close(figure)


def _plot_critical_difference_svg(
    average_ranks: Mapping[str, float],
    critical_difference: float,
    output: Path | str,
    title: str,
) -> None:
    if not average_ranks:
        raise ValueError("A critical-difference plot requires at least one method.")
    if not np.isfinite(float(critical_difference)) or critical_difference < 0.0:
        raise ValueError("The critical difference must be finite and nonnegative.")
    ordered = sorted(
        average_ranks.items(), key=lambda item: (float(item[1]), str(item[0]))
    )
    method_count = len(ordered)
    width = 900.0
    height = max(360.0, 170.0 + 30.0 * method_count)
    left, right = 80.0, 260.0
    axis_left, axis_right = left, width - right
    axis_width = axis_right - axis_left
    minimum_rank = 1.0
    maximum_rank = float(max(2, method_count))

    def x_position(rank: float) -> float:
        return axis_left + (rank - minimum_rank) / (maximum_rank - minimum_rank) * axis_width

    rank_map = {name: float(rank) for name, rank in ordered}
    candidates: List[Tuple[int, int]] = []
    for start in range(method_count):
        stop = start
        while (
            stop + 1 < method_count
            and ordered[stop + 1][1] - ordered[start][1]
            <= critical_difference + 1e-12
        ):
            stop += 1
        if stop > start:
            candidates.append((start, stop))
    groups = [
        candidate
        for candidate in candidates
        if not any(
            other[0] <= candidate[0]
            and candidate[1] <= other[1]
            and candidate != other
            for other in candidates
        )
    ]
    axis_y = 82.0 + 9.0 * len(groups)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height:.0f}" viewBox="0 0 900 {height:.0f}">',
        "<desc>Average-rank critical-difference diagram with maximal non-rejected Nemenyi groups.</desc>",
        f'<rect width="900" height="{height:.0f}" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#222">',
        f'<text x="450" y="24" text-anchor="middle" font-size="15" font-weight="600">{html.escape(title)}</text>',
    ]
    for group_index, (start, stop) in enumerate(groups):
        y = 42.0 + 9.0 * group_index
        lines.append(
            f'<line x1="{x_position(float(ordered[start][1])):.2f}" y1="{y:.2f}" x2="{x_position(float(ordered[stop][1])):.2f}" y2="{y:.2f}" stroke="#333" stroke-width="4" stroke-linecap="round"/>'
        )
    lines.append(
        f'<line x1="{axis_left:.2f}" y1="{axis_y:.2f}" x2="{axis_right:.2f}" y2="{axis_y:.2f}" stroke="#222" stroke-width="1.4"/>'
    )
    for rank in range(1, method_count + 1):
        x = x_position(float(rank))
        lines.append(
            f'<line x1="{x:.2f}" y1="{axis_y-5:.2f}" x2="{x:.2f}" y2="{axis_y+5:.2f}" stroke="#222" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{axis_y+20:.2f}" text-anchor="middle">{rank}</text>'
        )
    list_top = axis_y + 48.0
    for index, (name, rank) in enumerate(ordered):
        y = list_top + 29.0 * index
        x = x_position(float(rank))
        lines.extend(
            (
                f'<circle cx="{x:.2f}" cy="{axis_y:.2f}" r="3.4" fill="#1f5aa6"/>',
                f'<line x1="{x:.2f}" y1="{axis_y+4:.2f}" x2="{x:.2f}" y2="{y-4:.2f}" stroke="#9aa0a6" stroke-width="0.8"/>',
                f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{axis_right+28:.2f}" y2="{y:.2f}" stroke="#9aa0a6" stroke-width="0.8"/>',
                f'<text x="{axis_right+36:.2f}" y="{y+4:.2f}">{html.escape(str(name))} ({rank:.3f})</text>',
            )
        )
    cd_start = axis_left
    cd_stop = min(axis_right, x_position(minimum_rank + critical_difference))
    cd_y = height - 42.0
    lines.extend(
        (
            f'<line x1="{cd_start:.2f}" y1="{cd_y:.2f}" x2="{cd_stop:.2f}" y2="{cd_y:.2f}" stroke="#b33a3a" stroke-width="3"/>',
            f'<line x1="{cd_start:.2f}" y1="{cd_y-5:.2f}" x2="{cd_start:.2f}" y2="{cd_y+5:.2f}" stroke="#b33a3a" stroke-width="1.5"/>',
            f'<line x1="{cd_stop:.2f}" y1="{cd_y-5:.2f}" x2="{cd_stop:.2f}" y2="{cd_y+5:.2f}" stroke="#b33a3a" stroke-width="1.5"/>',
            f'<text x="{(cd_start+cd_stop)/2:.2f}" y="{cd_y-9:.2f}" text-anchor="middle" fill="#8c2525">CD = {critical_difference:.3f}</text>',
            f'<text x="{(axis_left+axis_right)/2:.2f}" y="{height-12:.2f}" text-anchor="middle">Average rank (lower is better)</text>',
            "</g>",
            "</svg>",
        )
    )
    _write_svg(Path(output), lines)


def plot_critical_difference(
    average_ranks: Mapping[str, float],
    critical_difference: float,
    output: Path | str,
    title: str,
) -> None:
    target = Path(output)
    if target.suffix.lower() == ".svg":
        _plot_critical_difference_svg(
            average_ranks,
            critical_difference,
            target,
            title,
        )
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib.") from error
    ordered = sorted(average_ranks.items(), key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(8.0, max(3.0, 0.38 * len(ordered) + 1.5)))
    y = np.arange(len(ordered))
    axis.scatter([rank for _, rank in ordered], y, color="#1f5aa6", zorder=3)
    for position, (name, rank) in enumerate(ordered):
        axis.text(rank + 0.03, position, f"{name} ({rank:.2f})", va="center")
    axis.set_yticks([])
    axis.set_xlabel("Average rank (lower is better)")
    axis.set_title(f"{title}; Nemenyi CD={critical_difference:.3f}")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=220)
    plt.close(figure)


def latex_wide_table(
    datasets: Sequence[str],
    method_names: Sequence[str],
    values: np.ndarray,
    caption: str,
    label: str,
) -> str:
    matrix = np.asarray(values, dtype=float)
    header = "Dataset & " + " & ".join(method_names) + r" \\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(method_names) + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for dataset, row in zip(datasets, matrix):
        finite = np.isfinite(row)
        best = float(np.max(row[finite])) if np.any(finite) else float("nan")
        cells = []
        for value in row:
            if not np.isfinite(value):
                cells.append("--")
            elif abs(float(value) - best) <= 1e-12:
                cells.append(rf"\textbf{{{float(value):.4f}}}")
            else:
                cells.append(f"{float(value):.4f}")
        lines.append(dataset.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "DatasetBundle",
    "complementarity_record",
    "compute_binary_metrics",
    "dataset_type",
    "discover_datasets",
    "latex_wide_table",
    "load_dataset",
    "load_nominal_map",
    "normalize_binary_labels",
    "parse_number_grid",
    "plot_critical_difference",
    "plot_sensitivity",
    "read_json",
    "read_rows",
    "resolve_nominal_features",
    "save_json",
    "select_best_per_dataset",
    "statistical_comparison",
    "system_manifest",
    "write_rows",
]
