"""Relation-consistent multi-view fuzzy-tree anomaly detection.

The implementation is independent of the historical MFTAD modules.  A single
view relation controls pivot selection, normalized cohesion refinement, and
the Leaf/GTA/Path support-flow evidence for numerical, nominal, and mixed
attribute views.  No dataset-type branch is used after the single-attribute
relations have been constructed.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function

        return decorator

    prange = range

    def set_num_threads(count):
        return None


ArrayLike = Union[np.ndarray, "torch.Tensor"]
COMPONENTS = ("leaf", "gta", "path")


@dataclass
class GranularBallNode:
    """One node in a view-specific fuzzy-granulation tree."""

    id: int
    scale_level: int
    sample_indices: np.ndarray
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    is_leaf: bool = False
    cohesion: float = 0.0
    is_gta: bool = False
    fuzzy_cardinalities: Optional[np.ndarray] = None
    pivot_a: Optional[int] = None
    pivot_b: Optional[int] = None
    raw_cohesion_gain: float = 0.0
    normalized_cohesion_gain: float = 0.0
    anchor_strength: float = 0.0


def _as_numpy_2d(X: ArrayLike) -> np.ndarray:
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    result = np.asarray(X, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {result.shape}.")
    if result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError("X must contain at least two samples and one feature.")
    if not np.all(np.isfinite(result)):
        raise ValueError("X contains NaN or infinite values.")
    return np.ascontiguousarray(result)


def _single_attribute_relation(
    column: np.ndarray,
    delta: float,
    nominal: bool,
    nominal_relation_mode: str = "equality",
) -> np.ndarray:
    distances = np.abs(column[:, None] - column[None, :])
    if nominal and nominal_relation_mode == "equality":
        return (distances <= 1e-12).astype(np.float64)
    std = float(np.std(column))
    if std <= 1e-12:
        return np.ones((len(column), len(column)), dtype=np.float64)
    epsilon = std / delta
    return 1.0 - np.clip(distances / epsilon, 0.0, 1.0)


class AttributeRelationProvider:
    """Memory-bounded provider for reusable single-attribute relations."""

    def __init__(
        self,
        X: np.ndarray,
        delta: float,
        nominal_features: frozenset[int],
        max_cache_mb: float,
        nominal_relation_mode: str = "equality",
    ) -> None:
        self.X = X
        self.delta = delta
        self.nominal_features = nominal_features
        self.nominal_relation_mode = nominal_relation_mode
        relation_bytes = X.shape[0] * X.shape[0] * np.dtype(np.float64).itemsize
        cache_bytes = max(0, int(max_cache_mb * 1024**2))
        self.capacity = min(X.shape[1], cache_bytes // max(1, relation_bytes))
        self.cache: Dict[int, np.ndarray] = {}
        self.pinned_features: frozenset[int] = frozenset()
        self.computations = 0
        self.cache_hits = 0
        self._stats_lock = Lock()

    def get(self, feature: int) -> np.ndarray:
        cached = self.cache.get(feature)
        if cached is not None:
            with self._stats_lock:
                self.cache_hits += 1
            return cached
        relation = _single_attribute_relation(
            self.X[:, feature],
            self.delta,
            feature in self.nominal_features,
            self.nominal_relation_mode,
        )
        with self._stats_lock:
            self.computations += 1
        if feature in self.pinned_features:
            self.cache[feature] = relation
        return relation

    def pin(self, features: Iterable[int]) -> None:
        selected = tuple(sorted(set(int(feature) for feature in features)))
        if len(selected) > self.capacity:
            raise ValueError("Pinned relation count exceeds the cache capacity.")
        self.pinned_features = frozenset(selected)
        for feature in selected:
            if feature not in self.cache:
                self.cache[feature] = self.get(feature)

    @property
    def cache_memory_mb(self) -> float:
        return sum(relation.nbytes for relation in self.cache.values()) / 1024**2


def _attribute_importance(
    X: np.ndarray,
    relation_provider: AttributeRelationProvider,
    mode: str,
) -> np.ndarray:
    n_samples, n_features = X.shape
    importance = np.zeros(n_features, dtype=np.float64)
    for feature in range(n_features):
        column = X[:, feature]
        is_constant = np.ptp(column) <= 1e-12
        if is_constant and mode in {"fuzzy_entropy", "uniform"}:
            importance[feature] = 0.0
            continue
        relation = relation_provider.get(feature)
        cardinality = np.maximum(relation.sum(axis=1), 1e-12)
        if mode == "fuzzy_entropy":
            importance[feature] = -np.mean(np.log2(cardinality / n_samples))
        elif mode == "uniform":
            importance[feature] = 1.0
        elif mode == "legacy_uniform":
            # Reproduces the historical missing /n term.  Its non-positive
            # values subsequently trigger uniform view weights.
            importance[feature] = -np.mean(np.log2(cardinality))
        else:
            raise ValueError(
                "importance_mode must be 'fuzzy_entropy', 'uniform', or "
                "'legacy_uniform'."
            )
    return importance


class RelationConsistentTree:
    """Build one hierarchy whose pivots, assignment, and scores share R_view."""

    def __init__(
        self,
        X: np.ndarray,
        view: Sequence[int],
        delta: float,
        attribute_importance: np.ndarray,
        nominal_features: frozenset[int],
        cohesion_threshold: float,
        min_child_size: int,
        strict_cohesion_gain: bool,
        importance_mode: str,
        relation_provider: AttributeRelationProvider,
        pivot_mode: str,
        gain_mode: str,
        gta_mode: str,
        gta_max_count: int,
        gta_min_popularity_ratio: float,
    ) -> None:
        self.X = X
        self.view_indices = tuple(sorted(int(index) for index in view))
        self.X_view = np.ascontiguousarray(X[:, self.view_indices])
        self.n_samples = X.shape[0]
        self.delta = float(delta)
        self.attribute_importance = attribute_importance
        self.nominal_features = nominal_features
        self.cohesion_threshold = float(cohesion_threshold)
        self.min_child_size = int(min_child_size)
        self.strict_cohesion_gain = bool(strict_cohesion_gain)
        self.importance_mode = importance_mode
        self.relation_provider = relation_provider
        self.pivot_mode = pivot_mode
        self.gain_mode = gain_mode
        self.gta_mode = gta_mode
        self.gta_max_count = int(gta_max_count)
        self.gta_min_popularity_ratio = float(gta_min_popularity_ratio)

        self.R_view: Optional[np.ndarray] = None
        self.tree: Dict[int, GranularBallNode] = {}
        self.sample_leaf_map: Dict[int, int] = {}
        self.evolution_paths: Dict[int, List[int]] = {}
        self.evolution_cardinalities: Dict[int, np.ndarray] = {}
        self.local_gtas: List[int] = []
        self.anchor_weights: Dict[int, float] = {}
        self.view_attribute_weights: Optional[np.ndarray] = None
        self.pivot_attempt_counts = {
            "relation_medoid": 0,
            "relation_farthest_pair": 0,
        }
        self.pivot_accepted_counts = {
            "relation_medoid": 0,
            "relation_farthest_pair": 0,
        }
        self.pivot_selection_seconds = 0.0
        self.accepted_split_balances: List[float] = []
        self.root_split_balance = 0.0
        self.cohesion_singleton_calls = 0
        self.cohesion_empty_rejections = 0
        self.gta_frontier_lambda = 0.0
        self.gta_frontier_s0 = 0.0
        self.gta_frontier_sl = 0.0
        self.gta_frontier_k0 = 0
        self.gta_frontier_leaf_count = 0
        self.gta_frontier_status = "uninitialized"

    def build(self) -> "RelationConsistentTree":
        self.R_view = self._compute_view_relation()
        root = GranularBallNode(
            id=0,
            scale_level=0,
            sample_indices=np.arange(self.n_samples, dtype=np.int32),
        )
        root.cohesion, root.fuzzy_cardinalities = self._cohesion_statistics(
            root.sample_indices
        )
        self.tree = {0: root}
        queue: deque[GranularBallNode] = deque([root])
        next_id = 1

        while queue:
            node = queue.popleft()
            if len(node.sample_indices) <= 1:
                node.is_leaf = True
                continue
            if len(node.sample_indices) < 2 * self.min_child_size:
                node.is_leaf = True
                continue
            pivots = self._find_split_pivots(
                node.sample_indices, scale_level=node.scale_level
            )
            if pivots is None:
                node.is_leaf = True
                continue
            child_a_indices, child_b_indices = self._assign_samples(
                node.sample_indices, pivots
            )
            if (
                len(child_a_indices) < self.min_child_size
                or len(child_b_indices) < self.min_child_size
            ):
                node.is_leaf = True
                continue

            cohesion_a, cardinalities_a = self._cohesion_statistics(child_a_indices)
            cohesion_b, cardinalities_b = self._cohesion_statistics(child_b_indices)
            gain = (
                len(child_a_indices) * cohesion_a
                + len(child_b_indices) * cohesion_b
            ) / len(node.sample_indices) - node.cohesion
            remaining_gain = 1.0 - node.cohesion
            normalized_gain = (
                gain / remaining_gain if remaining_gain > 1e-15 else 0.0
            )
            tested_gain = normalized_gain if self.gain_mode == "normalized" else gain
            accepted = (
                tested_gain > self.cohesion_threshold
                if self.strict_cohesion_gain
                else tested_gain + 1e-15 >= self.cohesion_threshold
            )
            if gain <= 1e-15 or not accepted:
                node.is_leaf = True
                continue

            effective_pivot_mode = self._effective_pivot_mode(node.scale_level)
            if effective_pivot_mode in self.pivot_accepted_counts:
                self.pivot_accepted_counts[effective_pivot_mode] += 1
            split_balance = (
                4.0
                * len(child_a_indices)
                * len(child_b_indices)
                / float(len(node.sample_indices) ** 2)
            )
            self.accepted_split_balances.append(float(split_balance))
            if node.scale_level == 0:
                self.root_split_balance = float(split_balance)

            node.pivot_a, node.pivot_b = pivots
            node.raw_cohesion_gain = float(gain)
            node.normalized_cohesion_gain = float(normalized_gain)

            child_a = GranularBallNode(
                id=next_id,
                scale_level=node.scale_level + 1,
                sample_indices=child_a_indices,
                parent_id=node.id,
                cohesion=cohesion_a,
                fuzzy_cardinalities=cardinalities_a,
            )
            next_id += 1
            child_b = GranularBallNode(
                id=next_id,
                scale_level=node.scale_level + 1,
                sample_indices=child_b_indices,
                parent_id=node.id,
                cohesion=cohesion_b,
                fuzzy_cardinalities=cardinalities_b,
            )
            next_id += 1
            node.children_ids = [child_a.id, child_b.id]
            self.tree[child_a.id] = child_a
            self.tree[child_b.id] = child_b
            queue.extend((child_a, child_b))

        self._build_paths_and_gtas()
        return self

    def _compute_view_relation(self) -> np.ndarray:
        weights = self.attribute_importance[list(self.view_indices)].astype(
            np.float64, copy=True
        )
        if self.importance_mode == "legacy_uniform" or weights.sum() <= 1e-15:
            weights.fill(1.0 / len(weights))
        else:
            weights /= weights.sum()
        self.view_attribute_weights = weights.copy()

        relation = np.zeros((self.n_samples, self.n_samples), dtype=np.float64)
        for weight, feature in zip(weights, self.view_indices):
            relation += weight * self.relation_provider.get(feature)
        return relation

    def _cohesion_statistics(
        self, indices: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        if len(indices) == 0:
            self.cohesion_empty_rejections = (
                getattr(self, "cohesion_empty_rejections", 0) + 1
            )
            raise ValueError("Cohesion is undefined for an empty index set.")
        sub_relation = self.R_view[np.ix_(indices, indices)]
        cardinalities = sub_relation.sum(axis=1)
        if len(indices) == 1:
            self.cohesion_singleton_calls = (
                getattr(self, "cohesion_singleton_calls", 0) + 1
            )
            return 1.0, cardinalities
        off_diagonal_sum = float(cardinalities.sum() - np.trace(sub_relation))
        cohesion = off_diagonal_sum / (len(indices) * (len(indices) - 1))
        return cohesion, cardinalities

    def _effective_pivot_mode(self, scale_level: int) -> str:
        if self.pivot_mode == "relation_diameter_root":
            return (
                "relation_farthest_pair"
                if scale_level == 0
                else "relation_medoid"
            )
        return self.pivot_mode

    def _find_split_pivots(
        self, indices: np.ndarray, scale_level: int = 0
    ) -> Optional[Tuple[int, int]]:
        if len(indices) < 2:
            return None
        effective_mode = self._effective_pivot_mode(scale_level)
        if effective_mode in self.pivot_attempt_counts:
            self.pivot_attempt_counts[effective_mode] += 1
        start = time.perf_counter()
        try:
            return self._select_split_pivots(indices, effective_mode)
        finally:
            self.pivot_selection_seconds += time.perf_counter() - start

    def _select_split_pivots(
        self, indices: np.ndarray, effective_mode: str
    ) -> Optional[Tuple[int, int]]:
        if effective_mode == "feature_center":
            center = self.X_view[indices].mean(axis=0)
            distances = np.linalg.norm(self.X_view[indices] - center, axis=1)
            closest_order = np.lexsort((indices, distances))
            farthest_order = np.lexsort((indices, -distances))
            pivot_a = int(indices[closest_order[0]])
            pivot_b = int(indices[farthest_order[0]])
            if pivot_a == pivot_b:
                for local_index in farthest_order[1:]:
                    candidate = int(indices[local_index])
                    if candidate != pivot_a:
                        pivot_b = candidate
                        break
            return None if pivot_a == pivot_b else (pivot_a, pivot_b)
        sub_relation = self.R_view[np.ix_(indices, indices)]
        if effective_mode == "relation_farthest_pair":
            local_a, local_b = np.triu_indices(len(indices), k=1)
            order = np.lexsort(
                (
                    indices[local_b],
                    indices[local_a],
                    sub_relation[local_a, local_b],
                )
            )
            selected = int(order[0])
            return int(indices[local_a[selected]]), int(indices[local_b[selected]])
        total_support = sub_relation.sum(axis=1)
        center_order = np.lexsort((indices, -total_support))
        pivot_a = int(indices[center_order[0]])
        similarities = self.R_view[pivot_a, indices]
        margin_order = np.lexsort((indices, total_support, similarities))

        pivot_b = pivot_a
        for local_index in margin_order:
            candidate = int(indices[local_index])
            if candidate != pivot_a:
                pivot_b = candidate
                break
        return None if pivot_a == pivot_b else (pivot_a, pivot_b)

    def _assign_samples(
        self, indices: np.ndarray, pivots: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        pivot_a, pivot_b = pivots
        similarity_a = self.R_view[indices, pivot_a]
        similarity_b = self.R_view[indices, pivot_b]
        mask_a = similarity_a >= similarity_b
        return indices[mask_a].copy(), indices[~mask_a].copy()

    def _build_paths_and_gtas(self) -> None:
        for node_id in sorted(self.tree):
            node = self.tree[node_id]
            if node.is_leaf:
                for sample_index in node.sample_indices:
                    self.sample_leaf_map[int(sample_index)] = node_id

        for sample_index in range(self.n_samples):
            current = self.sample_leaf_map.get(sample_index)
            path: List[int] = []
            while current is not None:
                path.append(current)
                current = self.tree[current].parent_id
            ordered_path = list(reversed(path))
            self.evolution_paths[sample_index] = ordered_path
            cardinalities = []
            for node_id in ordered_path:
                node = self.tree[node_id]
                position = int(np.searchsorted(node.sample_indices, sample_index))
                cardinalities.append(node.fuzzy_cardinalities[position])
            self.evolution_cardinalities[sample_index] = np.asarray(
                cardinalities, dtype=np.float64
            )

        if self.gta_mode == "top_k":
            self.gta_frontier_status = "not_applicable_top_k"
            candidates: List[Tuple[float, int]] = []
            for node_id in sorted(self.tree):
                node = self.tree[node_id]
                if node.is_leaf:
                    continue
                popularity = len(node.sample_indices) / self.n_samples
                if popularity + 1e-15 < self.gta_min_popularity_ratio:
                    continue
                candidates.append((popularity * node.cohesion, node_id))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            self.local_gtas = [
                node_id for _, node_id in candidates[: self.gta_max_count]
            ]
        elif self.gta_mode == "single_scale":
            self.gta_frontier_status = "not_applicable_single_scale"
            by_depth: Dict[int, List[int]] = {}
            for node_id in sorted(self.tree):
                node = self.tree[node_id]
                if node_id == 0 or node.is_leaf:
                    continue
                by_depth.setdefault(node.scale_level, []).append(node_id)
            if by_depth:
                selected_depth = min(
                    by_depth,
                    key=lambda depth: (
                        -sum(
                            self._anchor_strength(node_id)
                            for node_id in by_depth[depth]
                        ),
                        depth,
                    ),
                )
                self.local_gtas = by_depth[selected_depth]
            else:
                self.local_gtas = []
        else:
            self.local_gtas = self._select_gta_frontier()
        if not self.local_gtas:
            self.anchor_weights = {}
            return
        strengths = {
            node_id: self._anchor_strength(node_id) for node_id in self.local_gtas
        }
        maximum = max(strengths.values())
        if maximum <= 1e-15:
            self.local_gtas = []
            self.anchor_weights = {}
            return
        self.anchor_weights = {
            node_id: strength / maximum for node_id, strength in strengths.items()
        }
        for node_id, weight in self.anchor_weights.items():
            node = self.tree[node_id]
            node.is_gta = True
            node.anchor_strength = float(weight)

    def _anchor_strength(self, node_id: int) -> float:
        node = self.tree[node_id]
        return (len(node.sample_indices) / self.n_samples) * node.cohesion

    def _calibrate_frontier_complexity(self) -> bool:
        """Calibrate the exact-cover cost from this tree's attainable range."""

        root_children = list(self.tree[0].children_ids)
        terminal_leaves = [
            node_id
            for node_id in sorted(self.tree)
            if self.tree[node_id].is_leaf or not self.tree[node_id].children_ids
        ]
        self.gta_frontier_k0 = len(root_children)
        self.gta_frontier_leaf_count = len(terminal_leaves)
        if not root_children:
            self.gta_frontier_s0 = 0.0
            self.gta_frontier_sl = 0.0
            self.gta_frontier_lambda = 0.0
            self.gta_frontier_status = "root_unsplit"
            return False
        self.gta_frontier_s0 = float(
            sum(self._anchor_strength(node_id) for node_id in root_children)
        )
        self.gta_frontier_sl = float(
            sum(self._anchor_strength(node_id) for node_id in terminal_leaves)
        )
        self.gta_frontier_lambda = 0.0
        complexity_span = self.gta_frontier_leaf_count - self.gta_frontier_k0
        if complexity_span <= 0:
            self.gta_frontier_status = "no_optional_refinement"
            return False
        utility_span = self.gta_frontier_sl - self.gta_frontier_s0
        tolerance = (
            100.0
            * np.finfo(np.float64).eps
            * max(1.0, abs(self.gta_frontier_s0), abs(self.gta_frontier_sl))
        )
        if utility_span <= tolerance:
            self.gta_frontier_status = "no_positive_utility_span"
            return False
        self.gta_frontier_lambda = float(utility_span / complexity_span)
        self.gta_frontier_status = "calibrated"
        return True

    def _select_gta_frontier(self) -> List[int]:
        """Return the parameter-free exact-cover frontier below the root."""

        calibrated = self._calibrate_frontier_complexity()
        root_children = list(self.tree[0].children_ids)
        if not root_children:
            return []
        if not calibrated:
            return root_children
        selected, _ = self._anchor_frontier(0, allow_self=False)
        return selected

    def _anchor_selection_utility(self, node_id: int) -> float:
        """Return strength minus this tree's calibrated per-anchor cost."""

        return self._anchor_strength(node_id) - self.gta_frontier_lambda

    def _anchor_frontier(
        self, node_id: int, allow_self: bool = True
    ) -> Tuple[List[int], float]:
        """Select a complexity-regularized exact-cover frontier."""
        node = self.tree[node_id]
        if node.is_leaf or not node.children_ids:
            if allow_self:
                return [node_id], self._anchor_selection_utility(node_id)
            return [], 0.0
        descendants: List[int] = []
        descendant_utility = 0.0
        for child_id in node.children_ids:
            child_nodes, child_utility = self._anchor_frontier(child_id)
            descendants.extend(child_nodes)
            descendant_utility += child_utility
        if not allow_self:
            return descendants, descendant_utility
        self_utility = self._anchor_selection_utility(node_id)
        if self_utility + 1e-15 >= descendant_utility:
            return [node_id], self_utility
        return descendants, descendant_utility


@njit(cache=True, parallel=True)
def _leaf_scores_numba(
    relation, member_indices, sizes, offsets, n_samples
):  # pragma: no cover - exercised when numba is installed
    scores = np.full(n_samples, 0.5)
    for leaf_index in prange(len(sizes)):
        start = offsets[leaf_index]
        stop = start + sizes[leaf_index]
        members = member_indices[start:stop]
        if len(members) <= 1:
            for sample in members:
                scores[sample] = 1.0
            continue
        for i in range(len(members)):
            total = 0.0
            for j in range(len(members)):
                if i != j:
                    total += relation[members[i], members[j]]
            scores[members[i]] = 1.0 - total / (len(members) - 1)
    return scores


@njit(cache=True, parallel=True)
def _gta_scores_numba(
    relation, member_indices, sizes, offsets, anchor_weights, n_samples
):  # pragma: no cover - exercised when numba is installed
    scores = np.full(n_samples, 0.5)
    for sample in prange(n_samples):
        maximum = 0.0
        for gta_index in range(len(sizes)):
            start = offsets[gta_index]
            stop = start + sizes[gta_index]
            total = 0.0
            contains_sample = False
            for position in range(start, stop):
                member = member_indices[position]
                if member == sample:
                    contains_sample = True
                else:
                    total += relation[sample, member]
            denominator = sizes[gta_index] - (1 if contains_sample else 0)
            trust = (
                anchor_weights[gta_index] * total / denominator
                if denominator > 0
                else 0.0
            )
            if trust > maximum:
                maximum = trust
        scores[sample] = 1.0 - maximum
    return scores


@njit(cache=True, parallel=True)
def _path_scores_numba(
    relation,
    path_nodes,
    path_sizes,
    path_offsets,
    node_members,
    node_sizes,
    node_offsets,
    n_samples,
):  # pragma: no cover - exercised when numba is installed
    scores = np.zeros(n_samples)
    for sample in prange(n_samples):
        path_size = path_sizes[sample]
        if path_size <= 1:
            continue
        cardinalities = np.zeros(path_size)
        path_start = path_offsets[sample]
        for path_position in range(path_size):
            node_id = path_nodes[path_start + path_position]
            node_start = node_offsets[node_id]
            node_stop = node_start + node_sizes[node_id]
            total = 0.0
            for member_position in range(node_start, node_stop):
                total += relation[sample, node_members[member_position]]
            cardinalities[path_position] = total
        maximum = np.max(cardinalities)
        if maximum > 0.0:
            cardinalities /= maximum
        standard_deviation = np.std(cardinalities)
        scores[sample] = 1.0 - 1.0 / (1.0 + standard_deviation)
    return scores


@njit(cache=True, parallel=True)
def _cached_path_scores_numba(
    cardinalities_flat, path_sizes, path_offsets, n_samples
):  # pragma: no cover - exercised when numba is installed
    scores = np.zeros(n_samples)
    for sample in prange(n_samples):
        path_size = path_sizes[sample]
        if path_size <= 1:
            continue
        start = path_offsets[sample]
        stop = start + path_size
        values = cardinalities_flat[start:stop].copy()
        maximum = np.max(values)
        if maximum > 0.0:
            values /= maximum
        standard_deviation = np.std(values)
        scores[sample] = 1.0 - 1.0 / (1.0 + standard_deviation)
    return scores


class RelationConsistentMFTAD:
    """Unified heterogeneous-attribute MFTAD with relation-diameter hierarchy."""

    def __init__(
        self,
        delta: float = 0.2,
        min_views: int = 50,
        max_views: int = 100,
        supplemental_view_count: Optional[int] = 50,
        min_view_dimension_ratio: float = 0.1,
        supplemental_view_dimension_ratio: float = 0.7,
        cohesion_threshold: float = 0.0225,
        min_child_size: int = 1,
        strict_cohesion_gain: bool = False,
        fusion_mode: str = "fixed_component_mean",
        importance_mode: str = "fuzzy_entropy",
        nominal_relation_mode: str = "equality",
        pivot_mode: str = "relation_diameter_root",
        gain_mode: str = "raw",
        gta_mode: str = "frontier",
        gta_max_count: int = 5,
        gta_min_popularity_ratio: float = 0.1,
        enabled_components: Iterable[str] = COMPONENTS,
        include_full_view: bool = True,
        nominal_features: Optional[Iterable[int]] = None,
        n_jobs: int = 1,
        memory_safety_ratio: float = 0.8,
        attribute_relation_cache_mb: float = 512.0,
        use_numba: bool = True,
        numba_threads: int = 1,
        retain_view_evidence: bool = False,
        verbose: bool = False,
    ) -> None:
        self.delta = float(delta)
        self.min_views = int(min_views)
        self.max_views = int(max_views)
        self.supplemental_view_count = supplemental_view_count
        self.min_view_dimension_ratio = float(min_view_dimension_ratio)
        self.supplemental_view_dimension_ratio = float(
            supplemental_view_dimension_ratio
        )
        self.cohesion_threshold = float(cohesion_threshold)
        self.min_child_size = int(min_child_size)
        self.strict_cohesion_gain = bool(strict_cohesion_gain)
        self.fusion_mode = str(fusion_mode)
        self.importance_mode = str(importance_mode)
        self.nominal_relation_mode = str(nominal_relation_mode)
        self.pivot_mode = str(pivot_mode)
        self.gain_mode = str(gain_mode)
        self.gta_mode = str(gta_mode)
        self.gta_max_count = int(gta_max_count)
        self.gta_min_popularity_ratio = float(gta_min_popularity_ratio)
        component_input = (
            (enabled_components,)
            if isinstance(enabled_components, str)
            else enabled_components
        )
        requested_components = tuple(
            dict.fromkeys(str(item) for item in component_input)
        )
        self.enabled_components = requested_components
        self.include_full_view = bool(include_full_view)
        self.nominal_features = frozenset(int(i) for i in (nominal_features or []))
        self.n_jobs = max(1, mp.cpu_count() if n_jobs == -1 else int(n_jobs))
        self.memory_safety_ratio = float(memory_safety_ratio)
        self.attribute_relation_cache_mb = float(attribute_relation_cache_mb)
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
        self.numba_threads = int(numba_threads)
        self.retain_view_evidence = bool(retain_view_evidence)
        self.verbose = bool(verbose)
        self._validate_parameters()
        if self.use_numba:
            set_num_threads(self.numba_threads)

        self.X_: Optional[np.ndarray] = None
        self.attribute_importance_: Optional[np.ndarray] = None
        self.views_: List[Tuple[int, ...]] = []
        self.forest_: List[RelationConsistentTree] = []
        self.fused_scores_3d_: Optional[np.ndarray] = None
        self.dynamic_weights_: Optional[np.ndarray] = None
        self.learned_score_weights_: Optional[np.ndarray] = None
        self.view_weights_: Dict[str, np.ndarray] = {}
        self.active_components_: Tuple[int, ...] = (0, 1, 2)
        self.component_availability_: Optional[np.ndarray] = None
        self.component_stability_: Optional[np.ndarray] = None
        self.component_stability_estimable_: Optional[np.ndarray] = None
        self.component_stability_status_: Optional[Tuple[str, ...]] = None
        self.component_reliability_: Optional[np.ndarray] = None
        self.component_weights_: Optional[np.ndarray] = None
        self.view_component_available_: Optional[np.ndarray] = None
        self.view_evidence_: Optional[np.ndarray] = None
        self.runtime_breakdown_: Dict[str, float] = {}
        self.tree_stats_: Dict[str, object] = {}
        self.used_strategy_: str = "unfitted"
        self.relation_cache_stats_: Dict[str, float] = {}
        self._relation_provider: Optional[AttributeRelationProvider] = None

    def _validate_parameters(self) -> None:
        if self.delta <= 0:
            raise ValueError("delta must be positive.")
        if not 0.0 <= self.cohesion_threshold <= 1.0:
            raise ValueError("Normalized cohesion threshold must be in [0, 1].")
        if self.min_views < 1 or self.max_views < 1:
            raise ValueError("min_views and max_views must be positive.")
        if self.min_views > self.max_views:
            raise ValueError("min_views cannot exceed max_views.")
        if self.min_child_size < 1:
            raise ValueError("min_child_size must be positive.")
        if not 0.0 < self.memory_safety_ratio <= 1.0:
            raise ValueError("memory_safety_ratio must be in (0, 1].")
        if self.attribute_relation_cache_mb < 0.0:
            raise ValueError("attribute_relation_cache_mb cannot be negative.")
        if self.numba_threads < 1:
            raise ValueError("numba_threads must be positive.")
        if self.importance_mode not in {"fuzzy_entropy", "uniform", "legacy_uniform"}:
            raise ValueError("Invalid importance_mode.")
        if self.nominal_relation_mode not in {"equality", "numeric_code"}:
            raise ValueError("Invalid nominal_relation_mode.")
        if self.pivot_mode not in {
            "relation_medoid",
            "relation_farthest_pair",
            "relation_diameter_root",
            "feature_center",
        }:
            raise ValueError("Invalid pivot_mode.")
        if self.gain_mode not in {"normalized", "raw"}:
            raise ValueError("Invalid gain_mode.")
        if self.gta_mode not in {"frontier", "single_scale", "top_k"}:
            raise ValueError("Invalid gta_mode.")
        if self.gta_max_count < 1:
            raise ValueError("gta_max_count must be positive.")
        if not 0.0 <= self.gta_min_popularity_ratio <= 1.0:
            raise ValueError("gta_min_popularity_ratio must be in [0, 1].")
        invalid_components = set(self.enabled_components) - set(COMPONENTS)
        if invalid_components or not self.enabled_components:
            raise ValueError(
                f"enabled_components must be a non-empty subset of {COMPONENTS}."
            )
        if self.fusion_mode not in {
            "fixed_component_mean",
            "structural_stability",
            "current_certainty",
        }:
            raise ValueError(
                "fusion_mode must be 'fixed_component_mean', "
                "'structural_stability', or 'current_certainty'."
            )

    def fit(self, X: ArrayLike) -> "RelationConsistentMFTAD":
        self._fit_and_score(X)
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        return self._fit_and_score(X)

    def _fit_and_score(self, X: ArrayLike) -> np.ndarray:
        self.X_ = _as_numpy_2d(X)
        n_features = self.X_.shape[1]
        invalid_nominal = [i for i in self.nominal_features if not 0 <= i < n_features]
        if invalid_nominal:
            raise ValueError(f"nominal feature indices out of range: {invalid_nominal}")

        total_start = time.perf_counter()
        start = time.perf_counter()
        self._relation_provider = AttributeRelationProvider(
            self.X_,
            self.delta,
            self.nominal_features,
            self.attribute_relation_cache_mb,
            self.nominal_relation_mode,
        )
        if self._relation_provider.capacity >= n_features:
            self._relation_provider.pin(range(n_features))
        self.attribute_importance_ = _attribute_importance(
            self.X_, self._relation_provider, self.importance_mode
        )
        self.views_ = self._generate_views()
        view_time = time.perf_counter() - start
        if not self.views_:
            raise RuntimeError("Deterministic view generation produced no views.")
        if 0 < self._relation_provider.capacity < n_features:
            frequencies = np.zeros(n_features, dtype=np.int64)
            for view in self.views_:
                frequencies[list(view)] += 1
            frequent_features = np.lexsort(
                (np.arange(n_features), -frequencies)
            )[: self._relation_provider.capacity]
            self._relation_provider.pin(frequent_features)

        start = time.perf_counter()
        (
            self.forest_,
            raw_scores,
            tree_build_cpu_time,
            raw_score_cpu_time,
        ) = self._build_and_score_forest()
        forest_score_time = time.perf_counter() - start
        if len(self.forest_) != len(self.views_):
            raise RuntimeError("One or more trees failed to build.")

        start = time.perf_counter()
        scores = self._fuse_raw_scores(raw_scores)
        fusion_time = time.perf_counter() - start
        self.relation_cache_stats_ = {
            "capacity": int(self._relation_provider.capacity),
            "cached_relations": int(len(self._relation_provider.cache)),
            "relation_computations": int(self._relation_provider.computations),
            "cache_hits": int(self._relation_provider.cache_hits),
            "cache_memory_mb": float(self._relation_provider.cache_memory_mb),
        }
        self._relation_provider = None
        self.runtime_breakdown_ = {
            "view_generation_seconds": view_time,
            "forest_and_raw_score_seconds": forest_score_time,
            "tree_build_cpu_seconds": tree_build_cpu_time,
            "raw_score_cpu_seconds": raw_score_cpu_time,
            "fusion_seconds": fusion_time,
            "total_seconds": time.perf_counter() - total_start,
        }
        self._update_tree_stats()
        if self.verbose:
            print(
                f"RelationConsistentMFTAD[relation/{self.fusion_mode}] "
                f"views={len(self.views_)} "
                f"trees={len(self.forest_)} time={self.runtime_breakdown_['total_seconds']:.3f}s"
            )
        return scores

    @staticmethod
    def _canonicalize_views(views: Iterable[Sequence[int]]) -> List[Tuple[int, ...]]:
        unique = {tuple(sorted(int(i) for i in view)) for view in views if view}
        return sorted(unique, key=lambda view: (-len(view), view))

    @staticmethod
    def _unrank_combination(n: int, k: int, rank: int) -> Tuple[int, ...]:
        """Return the zero-based lexicographic combination at ``rank``."""
        if not 0 <= rank < math.comb(n, k):
            raise ValueError("Combination rank is out of range.")
        result: List[int] = []
        minimum = 0
        for remaining in range(k, 0, -1):
            for value in range(minimum, n):
                suffix_count = math.comb(n - value - 1, remaining - 1)
                if rank < suffix_count:
                    result.append(value)
                    minimum = value + 1
                    break
                rank -= suffix_count
        return tuple(result)

    def _generate_views(self) -> List[Tuple[int, ...]]:
        n_features = self.X_.shape[1]
        # Descending importance with original feature index as the tie-breaker.
        rank_desc = list(
            np.lexsort((np.arange(n_features), -self.attribute_importance_))
        )
        rank_asc = list(
            np.lexsort((np.arange(n_features), self.attribute_importance_))
        )
        minimum_dimension = min(
            n_features, max(2, int(math.floor(n_features * self.min_view_dimension_ratio)))
        )

        initial: List[Tuple[int, ...]] = (
            [tuple(range(n_features))] if self.include_full_view else []
        )
        forward = set(range(n_features))
        backward = set(range(n_features))
        while len(forward) > minimum_dimension:
            forward.remove(rank_asc[len(range(n_features)) - len(forward)])
            initial.append(tuple(sorted(forward)))
        while len(backward) > minimum_dimension:
            backward.remove(rank_desc[len(range(n_features)) - len(backward)])
            initial.append(tuple(sorted(backward)))
        views = self._canonicalize_views(initial)

        if len(views) > self.max_views:
            if self.max_views == 1:
                views = views[:1]
            else:
                positions = [
                    int(math.floor(q * (len(views) - 1) / (self.max_views - 1)))
                    for q in range(self.max_views)
                ]
                views = [views[position] for position in positions]
            self.used_strategy_ = "index_spaced_truncation"
        else:
            self.used_strategy_ = "progressive_views"

        if len(views) < self.min_views:
            required = self.min_views - len(views)
            if self.supplemental_view_count is not None:
                required = max(required, int(self.supplemental_view_count))
            required = min(required, self.max_views - len(views))
            supplemental_dimension = min(
                n_features,
                max(
                    2,
                    int(
                        math.floor(
                            n_features * self.supplemental_view_dimension_ratio
                        )
                    ),
                ),
            )
            combination_count = math.comb(n_features, supplemental_dimension)
            selected_count = min(required, combination_count)
            if selected_count == 1:
                combination_ranks = [0]
            else:
                combination_ranks = [
                    int(
                        math.floor(
                            q * (combination_count - 1) / (selected_count - 1)
                        )
                    )
                    for q in range(selected_count)
                ]
            candidates: List[Tuple[int, ...]] = []
            for combination_rank in combination_ranks:
                ranked_positions = self._unrank_combination(
                    n_features, supplemental_dimension, combination_rank
                )
                candidates.append(
                    tuple(sorted(rank_desc[position] for position in ranked_positions))
                )
            views = self._canonicalize_views([*views, *candidates])
            self.used_strategy_ += "+combination_spaced_supplement"

        if len(views) > self.max_views:
            raise RuntimeError(
                "Generated view count exceeds the configured max_views limit."
            )
        return views

    def _safe_worker_count(self) -> int:
        if self.n_jobs <= 1 or len(self.views_) <= 1:
            return 1
        n_samples = self.X_.shape[0]
        estimated_per_tree = max(n_samples * n_samples * 8 * 2.2, 16 * 1024**2)
        if psutil is None:
            return min(self.n_jobs, len(self.views_))
        cache_bytes = (
            self._relation_provider.cache_memory_mb * 1024**2
            if self._relation_provider is not None
            else 0.0
        )
        safe_bytes = max(
            estimated_per_tree,
            psutil.virtual_memory().available * self.memory_safety_ratio
            - cache_bytes,
        )
        memory_workers = max(1, int(safe_bytes / estimated_per_tree))
        return min(self.n_jobs, len(self.views_), memory_workers)

    def _view_has_nominal(self, view: Sequence[int]) -> bool:
        return any(feature in self.nominal_features for feature in view)

    def _view_is_pure_nominal(self, view: Sequence[int]) -> bool:
        return bool(view) and all(
            feature in self.nominal_features for feature in view
        )

    def _view_type(self, view: Sequence[int]) -> str:
        if self._view_is_pure_nominal(view):
            return "pure_nominal"
        if self._view_has_nominal(view):
            return "mixed"
        return "pure_numeric"

    def _build_one_tree(self, index: int) -> Tuple[int, RelationConsistentTree]:
        view = self.views_[index]
        tree = RelationConsistentTree(
            X=self.X_,
            view=view,
            delta=self.delta,
            attribute_importance=self.attribute_importance_,
            nominal_features=self.nominal_features,
            cohesion_threshold=self.cohesion_threshold,
            min_child_size=self.min_child_size,
            strict_cohesion_gain=self.strict_cohesion_gain,
            importance_mode=self.importance_mode,
            relation_provider=self._relation_provider,
            pivot_mode=self.pivot_mode,
            gain_mode=self.gain_mode,
            gta_mode=self.gta_mode,
            gta_max_count=self.gta_max_count,
            gta_min_popularity_ratio=self.gta_min_popularity_ratio,
        ).build()
        return index, tree

    def _build_and_score_one(
        self, index: int
    ) -> Tuple[int, RelationConsistentTree, np.ndarray, float, float]:
        start = time.perf_counter()
        _, tree = self._build_one_tree(index)
        build_time = time.perf_counter() - start
        start = time.perf_counter()
        raw_scores = self._raw_scores(tree)
        score_time = time.perf_counter() - start
        for node in tree.tree.values():
            node.fuzzy_cardinalities = None
        tree.evolution_cardinalities.clear()
        tree.R_view = None
        tree.X_view = None
        tree.view_attribute_weights = None
        tree.X = None
        tree.relation_provider = None
        return index, tree, raw_scores, build_time, score_time

    def _build_and_score_forest(
        self,
    ) -> Tuple[List[RelationConsistentTree], np.ndarray, float, float]:
        workers = self._safe_worker_count()
        if workers == 1:
            results = [
                self._build_and_score_one(i) for i in range(len(self.views_))
            ]
            forest = [result[1] for result in results]
            raw = np.stack([result[2] for result in results], axis=0)
            return (
                forest,
                raw,
                float(sum(result[3] for result in results)),
                float(sum(result[4] for result in results)),
            )
        ordered: List[Optional[RelationConsistentTree]] = [None] * len(self.views_)
        ordered_raw: List[Optional[np.ndarray]] = [None] * len(self.views_)
        build_times = np.zeros(len(self.views_), dtype=np.float64)
        score_times = np.zeros(len(self.views_), dtype=np.float64)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._build_and_score_one, i)
                for i in range(len(self.views_))
            ]
            for future in as_completed(futures):
                index, tree, raw_scores, build_time, score_time = future.result()
                ordered[index] = tree
                ordered_raw[index] = raw_scores
                build_times[index] = build_time
                score_times[index] = score_time
        forest = [tree for tree in ordered if tree is not None]
        raw = np.stack(
            [scores for scores in ordered_raw if scores is not None], axis=0
        )
        return forest, raw, float(build_times.sum()), float(score_times.sum())

    @staticmethod
    def _flatten_node_members(
        tree: RelationConsistentTree, node_ids: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        members: List[int] = []
        sizes: List[int] = []
        offsets: List[int] = []
        for node_id in node_ids:
            node_members = tree.tree[node_id].sample_indices
            offsets.append(len(members))
            sizes.append(len(node_members))
            members.extend(int(i) for i in node_members)
        return (
            np.asarray(members, dtype=np.int32),
            np.asarray(sizes, dtype=np.int32),
            np.asarray(offsets, dtype=np.int32),
        )

    def _raw_scores(self, tree: RelationConsistentTree) -> np.ndarray:
        leaf_ids = [
            node_id for node_id in sorted(tree.tree) if tree.tree[node_id].is_leaf
        ]
        gta_members, gta_sizes, gta_offsets = self._flatten_node_members(
            tree, tree.local_gtas
        )
        gta_weights = np.asarray(
            [tree.anchor_weights[node_id] for node_id in tree.local_gtas],
            dtype=np.float64,
        )
        n_samples = self.X_.shape[0]

        leaf = np.full(n_samples, 0.5)
        for node_id in leaf_ids:
            node = tree.tree[node_id]
            members = node.sample_indices
            if len(members) <= 1:
                leaf[members] = 1.0
            else:
                leaf[members] = 1.0 - (
                    (node.fuzzy_cardinalities - 1.0) / (len(members) - 1)
                )

        if self.use_numba and len(gta_sizes):
            gta = _gta_scores_numba(
                tree.R_view,
                gta_members,
                gta_sizes,
                gta_offsets,
                gta_weights,
                n_samples,
            )
        elif tree.local_gtas:
            trust_columns = []
            for node_id, anchor_weight in zip(tree.local_gtas, gta_weights):
                members = tree.tree[node_id].sample_indices
                total = tree.R_view[:, members].sum(axis=1)
                denominator = np.full(n_samples, len(members), dtype=np.float64)
                total[members] -= 1.0
                denominator[members] -= 1.0
                peer_trust = np.divide(
                    total,
                    denominator,
                    out=np.zeros_like(total),
                    where=denominator > 0,
                )
                trust_columns.append(anchor_weight * peer_trust)
            trust = np.column_stack(trust_columns)
            gta = 1.0 - trust.max(axis=1)
        else:
            gta = np.full(n_samples, 0.5)

        cardinalities_flat: List[float] = []
        path_sizes: List[int] = []
        path_offsets: List[int] = []
        for sample in range(n_samples):
            cardinalities = tree.evolution_cardinalities[sample]
            path_offsets.append(len(cardinalities_flat))
            path_sizes.append(len(cardinalities))
            cardinalities_flat.extend(float(value) for value in cardinalities)
        cardinalities_array = np.asarray(cardinalities_flat, dtype=np.float64)
        path_sizes_array = np.asarray(path_sizes, dtype=np.int32)
        path_offsets_array = np.asarray(path_offsets, dtype=np.int32)
        if self.use_numba:
            path = _cached_path_scores_numba(
                cardinalities_array,
                path_sizes_array,
                path_offsets_array,
                n_samples,
            )
        else:
            path = np.zeros(n_samples)
            for sample in range(n_samples):
                start = path_offsets_array[sample]
                stop = start + path_sizes_array[sample]
                cardinalities = cardinalities_array[start:stop]
                if len(cardinalities) <= 1:
                    continue
                maximum = cardinalities.max()
                normalized = (
                    cardinalities / maximum if maximum > 0 else cardinalities
                )
                path[sample] = 1.0 - 1.0 / (1.0 + normalized.std())
        return np.column_stack((leaf, gta, path))

    @staticmethod
    def _normalize_evidence(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("Evidence must be a nonempty one-dimensional vector.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Evidence must contain only finite values.")
        minimum = float(scores.min())
        maximum = float(scores.max())
        if maximum <= minimum:
            return np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)

    @staticmethod
    def _entropy_view_weights(evidence: np.ndarray) -> np.ndarray:
        evidence = np.asarray(evidence, dtype=np.float64)
        if evidence.ndim != 2 or len(evidence) == 0:
            raise ValueError("View evidence must be a nonempty two-dimensional array.")
        if not np.all(np.isfinite(evidence)):
            raise ValueError("View evidence must contain only finite values.")
        clipped = np.clip(evidence, 1e-8, 1.0 - 1e-8)
        entropy = -np.mean(
            clipped * np.log2(clipped)
            + (1.0 - clipped) * np.log2(1.0 - clipped),
            axis=1,
        )
        weights = 1.0 - entropy
        if weights.sum() <= 1e-15:
            return np.full(len(weights), 1.0 / len(weights))
        return weights / weights.sum()

    @staticmethod
    def _rankdata(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Ranks require a nonempty one-dimensional vector.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Ranks require finite values.")
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

    @classmethod
    def _spearman(cls, left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
            raise ValueError("Spearman inputs must be same-length vectors.")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("Spearman inputs must contain only finite values.")
        if len(left) < 2:
            return 0.0
        left_rank = cls._rankdata(left)
        right_rank = cls._rankdata(right)
        left_rank -= left_rank.mean()
        right_rank -= right_rank.mean()
        denominator = float(
            np.sqrt(np.dot(left_rank, left_rank) * np.dot(right_rank, right_rank))
        )
        correlation = (
            float(np.dot(left_rank, right_rank) / denominator)
            if denominator > 1e-15
            else 0.0
        )
        if not np.isfinite(correlation):
            raise ValueError("Spearman correlation must be finite.")
        return correlation

    @staticmethod
    def _normalize_correlation_weights(
        correlations: np.ndarray,
        valid_mask: np.ndarray,
        availability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        correlations = np.asarray(correlations, dtype=np.float64)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if correlations.ndim != 1 or correlations.shape != valid_mask.shape:
            raise ValueError("Correlations and valid-component mask must align.")
        if not np.all(np.isfinite(correlations)):
            raise ValueError("Correlation weights require finite values.")
        if availability is None:
            availability = np.ones_like(correlations)
        availability = np.asarray(availability, dtype=np.float64)
        if availability.shape != correlations.shape:
            raise ValueError("Availability and correlation weights must align.")
        if not np.all(np.isfinite(availability)) or np.any(availability < 0.0):
            raise ValueError("Availability weights must be finite and nonnegative.")
        count = int(valid_mask.sum())
        if count == 0:
            raise ValueError("Fusion requires at least one valid component.")
        weights = np.zeros_like(correlations)
        agreement = np.maximum(correlations[valid_mask], 0.0)
        denominator = float(agreement.sum())
        if denominator <= 1e-15:
            fallback = availability[valid_mask]
            fallback_sum = float(fallback.sum())
            if fallback_sum <= 1e-15:
                weights[valid_mask] = 1.0 / count
            else:
                weights[valid_mask] = fallback / fallback_sum
        else:
            weights[valid_mask] = agreement / denominator
        return weights

    @staticmethod
    def _uniform_valid_weights(valid_mask: np.ndarray) -> np.ndarray:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.ndim != 1 or not np.any(valid_mask):
            raise ValueError("Fusion requires at least one valid component.")
        weights = np.zeros(valid_mask.shape, dtype=np.float64)
        weights[valid_mask] = 1.0 / float(valid_mask.sum())
        return weights

    def _panel_fusion(
        self, evidence: np.ndarray, view_indices: np.ndarray
    ) -> np.ndarray:
        if len(view_indices) == 0:
            return np.zeros(self.X_.shape[0], dtype=np.float64)
        selected = evidence[view_indices]
        return self._entropy_view_weights(selected) @ selected

    def _fuse_raw_scores(self, raw: np.ndarray) -> np.ndarray:
        n_samples = self.X_.shape[0]
        n_views = len(self.views_)
        raw = np.asarray(raw, dtype=np.float64)
        expected_shape = (n_views, n_samples, len(COMPONENTS))
        if raw.shape != expected_shape:
            raise ValueError(
                f"Raw evidence must have shape {expected_shape}, got {raw.shape}."
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Raw evidence must contain only finite values.")
        fused = np.zeros((n_samples, 3), dtype=np.float64)
        names = COMPONENTS
        enabled_mask = np.asarray(
            [name in self.enabled_components for name in names], dtype=bool
        )
        evidence_all = np.empty_like(raw, dtype=np.float64)
        view_weights: Dict[str, np.ndarray] = {}
        availability = np.ptp(raw, axis=1) > 1e-15
        stability = np.full(3, np.nan, dtype=np.float64)
        stability_estimable = np.zeros(3, dtype=bool)
        even = np.arange(0, n_views, 2, dtype=np.int64)
        odd = np.arange(1, n_views, 2, dtype=np.int64)
        for dimension, name in enumerate(names):
            evidence = np.stack(
                [self._normalize_evidence(scores) for scores in raw[:, :, dimension]],
                axis=0,
            )
            evidence_all[:, :, dimension] = evidence
            available_views = availability[:, dimension]
            weights = np.zeros(n_views, dtype=np.float64)
            if np.any(available_views):
                weights[available_views] = self._entropy_view_weights(
                    evidence[available_views]
                )
            view_weights[name] = weights
            fused[:, dimension] = weights @ evidence
            even_available = even[available_views[even]]
            odd_available = odd[available_views[odd]]
            even_panel = self._panel_fusion(evidence, even_available)
            odd_panel = self._panel_fusion(evidence, odd_available)
            if self.fusion_mode != "fixed_component_mean" and (
                n_samples >= 2
                and len(even_available) > 0
                and len(odd_available) > 0
                and np.ptp(even_panel) > 1e-15
                and np.ptp(odd_panel) > 1e-15
            ):
                stability[dimension] = self._spearman(even_panel, odd_panel)
                stability_estimable[dimension] = True

        component_availability = availability.mean(axis=0)
        valid_mask = enabled_mask & (component_availability > 0.0)
        fusion_degenerate_case = ""
        stability_for_weights = np.where(stability_estimable, stability, 0.0)
        if not np.any(valid_mask):
            # A requested component can be well-defined but constant for every
            # sample/view (for example, Leaf in a single-view core ablation).
            # This is a valid zero-information detector, not an invalid empty
            # component configuration.  Keep the enabled components as a
            # deterministic fallback so the run yields finite all-zero scores.
            reliability = self._normalize_correlation_weights(
                np.zeros(3, dtype=np.float64),
                enabled_mask,
                component_availability,
            )
            fusion_degenerate_case = "all_enabled_components_constant"
        elif self.fusion_mode == "fixed_component_mean":
            # Production fusion: each available component contributes equally.
            # View-level entropy weighting above remains active; this branch
            # deliberately excludes correlation priors and sample-level
            # midpoint-certainty weights.
            reliability = self._uniform_valid_weights(valid_mask)
        elif self.fusion_mode == "current_certainty":
            reliability = self._uniform_valid_weights(valid_mask)
        else:
            reliability = self._normalize_correlation_weights(
                stability_for_weights,
                valid_mask,
                component_availability,
            )
            if not np.any((stability_for_weights > 0.0) & valid_mask):
                fusion_degenerate_case = (
                    "stability_no_positive_agreement_availability_fallback"
                )

        if self.fusion_mode == "fixed_component_mean":
            dynamic_weights = np.broadcast_to(
                reliability, (n_samples, len(COMPONENTS))
            ).copy()
        else:
            certainty = np.abs(fused - 0.5) * 2.0
            weighted_certainty = certainty * reliability[None, :]
            denominator = weighted_certainty.sum(axis=1, keepdims=True)
            dynamic_weights = np.empty_like(weighted_certainty)
            informative = denominator[:, 0] > 1e-15
            dynamic_weights[informative] = (
                weighted_certainty[informative] / denominator[informative]
            )
            dynamic_weights[~informative] = reliability
        final = np.sum(fused * dynamic_weights, axis=1)
        final = self._normalize_evidence(final)

        self.active_components_ = tuple(int(i) for i in np.flatnonzero(valid_mask))
        self.fused_scores_3d_ = fused
        self.dynamic_weights_ = dynamic_weights
        self.learned_score_weights_ = dynamic_weights.mean(axis=0)
        self.view_weights_ = view_weights
        self.view_component_available_ = availability
        self.component_availability_ = component_availability
        self.component_stability_ = stability
        self.component_stability_estimable_ = stability_estimable
        self.component_stability_status_ = tuple(
            (
                "estimated_positive"
                if stability[index] > 1e-15
                else "estimated_negative"
                if stability[index] < -1e-15
                else "estimated_zero"
            )
            if value
            else "stability_unestimable"
            for index, value in enumerate(stability_estimable)
        )
        self.component_reliability_ = reliability
        self.component_weights_ = reliability.copy()
        self.fusion_degenerate_case_ = fusion_degenerate_case
        self.view_evidence_ = evidence_all if self.retain_view_evidence else None
        return final

    def _update_tree_stats(self) -> None:
        node_counts = np.asarray([len(tree.tree) for tree in self.forest_])
        depths = np.asarray(
            [max(node.scale_level for node in tree.tree.values()) for tree in self.forest_]
        )
        leaf_counts = np.asarray(
            [sum(node.is_leaf for node in tree.tree.values()) for tree in self.forest_]
        )
        gta_nodes = [
            tree.tree[node_id]
            for tree in self.forest_
            for node_id in tree.local_gtas
        ]
        internal_gta_counts = np.asarray(
            [
                sum(bool(tree.tree[node_id].children_ids) for node_id in tree.local_gtas)
                for tree in self.forest_
            ],
            dtype=np.float64,
        )
        gta_depths = np.asarray(
            [node.scale_level for node in gta_nodes], dtype=np.float64
        )
        if len(gta_depths):
            unique_gta_depths, gta_depth_counts = np.unique(
                gta_depths.astype(np.int64), return_counts=True
            )
            gta_depth_histogram = ",".join(
                f"{int(depth)}:{int(count)}"
                for depth, count in zip(unique_gta_depths, gta_depth_counts)
            )
        else:
            gta_depth_histogram = ""
        frontier_lambdas = np.asarray(
            [tree.gta_frontier_lambda for tree in self.forest_], dtype=np.float64
        )
        frontier_s0 = np.asarray(
            [tree.gta_frontier_s0 for tree in self.forest_], dtype=np.float64
        )
        frontier_sl = np.asarray(
            [tree.gta_frontier_sl for tree in self.forest_], dtype=np.float64
        )
        frontier_k0 = np.asarray(
            [tree.gta_frontier_k0 for tree in self.forest_], dtype=np.float64
        )
        frontier_leaf_counts = np.asarray(
            [tree.gta_frontier_leaf_count for tree in self.forest_],
            dtype=np.float64,
        )
        frontier_calibrated = np.asarray(
            [tree.gta_frontier_status == "calibrated" for tree in self.forest_],
            dtype=bool,
        )
        frontier_root_unsplit = np.asarray(
            [tree.gta_frontier_status == "root_unsplit" for tree in self.forest_],
            dtype=np.float64,
        )
        frontier_no_optional_refinement = np.asarray(
            [
                tree.gta_frontier_status == "no_optional_refinement"
                for tree in self.forest_
            ],
            dtype=np.float64,
        )
        frontier_no_positive_span = np.asarray(
            [
                tree.gta_frontier_status == "no_positive_utility_span"
                for tree in self.forest_
            ],
            dtype=np.float64,
        )
        coarsest_frontiers: List[float] = []
        all_leaf_frontiers: List[float] = []
        mixed_frontiers: List[float] = []
        for tree in self.forest_:
            selected = set(tree.local_gtas)
            root_children = set(tree.tree[0].children_ids)
            terminal_leaves = {
                node_id
                for node_id, node in tree.tree.items()
                if node.is_leaf or not node.children_ids
            }
            coarsest_frontiers.append(
                float(bool(root_children) and selected == root_children)
            )
            all_leaf_frontiers.append(
                float(
                    len(terminal_leaves) > len(root_children)
                    and selected == terminal_leaves
                )
            )
            selected_internal = any(
                bool(tree.tree[node_id].children_ids) for node_id in selected
            )
            selected_leaves = any(
                not tree.tree[node_id].children_ids for node_id in selected
            )
            mixed_frontiers.append(float(selected_internal and selected_leaves))
        split_nodes = [
            node
            for tree in self.forest_
            for node in tree.tree.values()
            if node.children_ids
        ]
        normalized_gains = np.asarray(
            [node.normalized_cohesion_gain for node in split_nodes],
            dtype=np.float64,
        )
        root_split_rate = float(
            np.mean([bool(tree.tree[0].children_ids) for tree in self.forest_])
        )
        root_split_balances = np.asarray(
            [
                tree.root_split_balance
                for tree in self.forest_
                if tree.tree[0].children_ids
            ],
            dtype=np.float64,
        )
        split_balances = np.asarray(
            [
                balance
                for tree in self.forest_
                for balance in tree.accepted_split_balances
            ],
            dtype=np.float64,
        )
        view_types = [self._view_type(view) for view in self.views_]
        self.tree_stats_ = {
            "n_views": int(len(self.views_)),
            "mean_nodes": float(node_counts.mean()),
            "max_nodes": int(node_counts.max()),
            "mean_depth": float(depths.mean()),
            "max_depth": int(depths.max()),
            "mean_leaves": float(leaf_counts.mean()),
            "mean_gtas": float(
                np.mean([len(tree.local_gtas) for tree in self.forest_])
            ),
            "mean_internal_gtas": float(internal_gta_counts.mean()),
            "internal_gta_fraction": float(
                internal_gta_counts.sum() / len(gta_nodes) if gta_nodes else 0.0
            ),
            "mean_gta_depth": float(gta_depths.mean() if len(gta_depths) else 0.0),
            "median_gta_depth": float(
                np.median(gta_depths) if len(gta_depths) else 0.0
            ),
            "std_gta_depth": float(gta_depths.std() if len(gta_depths) else 0.0),
            "max_gta_depth": int(gta_depths.max() if len(gta_depths) else 0),
            "gta_depth_histogram": gta_depth_histogram,
            "mean_frontier_lambda": float(
                frontier_lambdas[frontier_calibrated].mean()
                if np.any(frontier_calibrated)
                else 0.0
            ),
            "mean_frontier_s0": float(frontier_s0.mean()),
            "mean_frontier_sl": float(frontier_sl.mean()),
            "mean_frontier_gain_span": float((frontier_sl - frontier_s0).mean()),
            "mean_frontier_k0": float(frontier_k0.mean()),
            "mean_frontier_leaf_count": float(frontier_leaf_counts.mean()),
            "frontier_calibrated_rate": float(frontier_calibrated.mean()),
            "frontier_root_unsplit_rate": float(frontier_root_unsplit.mean()),
            "frontier_no_optional_refinement_rate": float(
                frontier_no_optional_refinement.mean()
            ),
            "frontier_no_positive_span_rate": float(
                frontier_no_positive_span.mean()
            ),
            "coarsest_frontier_rate": float(np.mean(coarsest_frontiers)),
            "all_leaf_frontier_rate": float(np.mean(all_leaf_frontiers)),
            "mixed_frontier_rate": float(np.mean(mixed_frontiers)),
            "root_split_rate": root_split_rate,
            "root_split_balance": float(
                root_split_balances.mean() if len(root_split_balances) else 0.0
            ),
            "mean_split_balance": float(
                split_balances.mean() if len(split_balances) else 0.0
            ),
            "medoid_pivot_attempts": int(
                sum(
                    tree.pivot_attempt_counts["relation_medoid"]
                    for tree in self.forest_
                )
            ),
            "farthest_pivot_attempts": int(
                sum(
                    tree.pivot_attempt_counts["relation_farthest_pair"]
                    for tree in self.forest_
                )
            ),
            "medoid_pivot_accepts": int(
                sum(
                    tree.pivot_accepted_counts["relation_medoid"]
                    for tree in self.forest_
                )
            ),
            "farthest_pivot_accepts": int(
                sum(
                    tree.pivot_accepted_counts["relation_farthest_pair"]
                    for tree in self.forest_
                )
            ),
            "pivot_selection_seconds": float(
                sum(tree.pivot_selection_seconds for tree in self.forest_)
            ),
            "mean_normalized_gain": float(
                normalized_gains.mean() if len(normalized_gains) else 0.0
            ),
            "min_normalized_gain": float(
                normalized_gains.min() if len(normalized_gains) else 0.0
            ),
            "pure_numeric_views": int(view_types.count("pure_numeric")),
            "pure_nominal_views": int(view_types.count("pure_nominal")),
            "mixed_views": int(view_types.count("mixed")),
            "retained_relation_matrices": int(
                sum(tree.R_view is not None for tree in self.forest_)
            ),
            "singleton_cohesion_calls": int(
                sum(tree.cohesion_singleton_calls for tree in self.forest_)
            ),
            "views_with_singleton_cohesion": int(
                sum(tree.cohesion_singleton_calls > 0 for tree in self.forest_)
            ),
            "empty_cohesion_rejections": int(
                sum(tree.cohesion_empty_rejections for tree in self.forest_)
            ),
        }


# Compact aliases for experiment scripts.
MFTAD = RelationConsistentMFTAD
RelationMFTAD = RelationConsistentMFTAD
RelationConsistentFuzzyTree = RelationConsistentTree


__all__ = [
    "RelationConsistentMFTAD",
    "RelationMFTAD",
    "MFTAD",
    "RelationConsistentTree",
    "RelationConsistentFuzzyTree",
    "GranularBallNode",
    "NUMBA_AVAILABLE",
    "COMPONENTS",
]
