"""Executable specification of the parameter-free GTA frontier."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


REPRODUCIBILITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from relation_mftad import (  # noqa: E402
    GranularBallNode,
    RelationConsistentMFTAD,
    RelationConsistentTree,
)


TIE_TOLERANCE = 1e-15


def _node(
    node_id: int,
    members: list[int],
    cohesion: float,
    *,
    parent_id: int | None = None,
    children: list[int] | None = None,
) -> GranularBallNode:
    children = [] if children is None else children
    return GranularBallNode(
        id=node_id,
        scale_level=0,
        sample_indices=np.asarray(members, dtype=np.int32),
        parent_id=parent_id,
        children_ids=children,
        is_leaf=not children,
        cohesion=cohesion,
    )


def _tree(
    nodes: list[GranularBallNode], n_samples: int
) -> RelationConsistentTree:
    tree = RelationConsistentTree.__new__(RelationConsistentTree)
    tree.n_samples = n_samples
    tree.tree = {node.id: node for node in nodes}
    tree.gta_frontier_lambda = 0.0
    tree.gta_frontier_s0 = 0.0
    tree.gta_frontier_sl = 0.0
    tree.gta_frontier_k0 = 0
    tree.gta_frontier_leaf_count = 0
    tree.gta_frontier_status = "uninitialized"
    return tree


def _parameter_free_tree() -> RelationConsistentTree:
    return _tree(
        [
            _node(0, list(range(8)), 0.1, children=[1, 2]),
            _node(1, [0, 1, 2, 3], 0.2, parent_id=0, children=[3, 4]),
            _node(2, [4, 5, 6, 7], 0.1, parent_id=0, children=[5, 6]),
            _node(3, [0, 1], 0.3, parent_id=1),
            _node(4, [2, 3], 0.3, parent_id=1),
            _node(5, [4, 5], 0.9, parent_id=2),
            _node(6, [6, 7], 0.9, parent_id=2),
        ],
        n_samples=8,
    )


def _all_frontiers(
    tree: RelationConsistentTree, node_id: int, allow_self: bool = True
) -> list[tuple[list[int], float]]:
    """Exhaustively enumerate exact-cover antichains below one node."""

    node = tree.tree[node_id]
    if node.is_leaf or not node.children_ids:
        return (
            [([node_id], tree._anchor_selection_utility(node_id))]
            if allow_self
            else [([], 0.0)]
        )
    child_options: list[tuple[list[int], float]] = [([], 0.0)]
    for child_id in node.children_ids:
        combined: list[tuple[list[int], float]] = []
        for prefix_nodes, prefix_utility in child_options:
            for child_nodes, child_utility in _all_frontiers(tree, child_id):
                combined.append(
                    (prefix_nodes + child_nodes, prefix_utility + child_utility)
                )
        child_options = combined
    if allow_self:
        child_options.append(
            ([node_id], tree._anchor_selection_utility(node_id))
        )
    return child_options


def _terminal_leaf_ids(tree: RelationConsistentTree) -> set[int]:
    return {
        node_id
        for node_id, node in tree.tree.items()
        if node.is_leaf or not node.children_ids
    }


def _frontier_strength(tree: RelationConsistentTree, selected: list[int]) -> float:
    return float(sum(tree._anchor_strength(node_id) for node_id in selected))


def _normalized_frontier_objective(
    tree: RelationConsistentTree, selected: list[int]
) -> float:
    utility_span = tree.gta_frontier_sl - tree.gta_frontier_s0
    complexity_span = tree.gta_frontier_leaf_count - tree.gta_frontier_k0
    return (
        (_frontier_strength(tree, selected) - tree.gta_frontier_s0) / utility_span
        - (len(selected) - tree.gta_frontier_k0) / complexity_span
    )


class GTAFrontierTests(unittest.TestCase):
    def test_model_api_has_no_gta_complexity_hyperparameter(self) -> None:
        self.assertNotIn(
            "gta_complexity_penalty_eta",
            inspect.signature(RelationConsistentMFTAD).parameters,
        )

    def test_leaf_is_the_dynamic_program_base_case(self) -> None:
        tree = _tree([_node(0, [0, 1], 0.25)], n_samples=2)
        selected, utility = tree._anchor_frontier(0)
        self.assertEqual(selected, [0])
        self.assertAlmostEqual(utility, 0.25)

    def test_root_without_split_has_no_gta_frontier(self) -> None:
        tree = _tree([_node(0, [0, 1], 0.25)], n_samples=2)
        self.assertEqual(tree._select_gta_frontier(), [])
        self.assertEqual(tree.gta_frontier_status, "root_unsplit")
        self.assertEqual(tree.gta_frontier_s0, 0.0)
        self.assertEqual(tree.gta_frontier_sl, 0.0)

    def test_root_direct_leaf_children_are_the_coarsest_frontier(self) -> None:
        tree = _tree(
            [
                _node(0, [0, 1], 0.25, children=[1, 2]),
                _node(1, [0], 1.0, parent_id=0),
                _node(2, [1], 1.0, parent_id=0),
            ],
            n_samples=2,
        )
        self.assertEqual(tree._select_gta_frontier(), [1, 2])
        self.assertEqual(tree.gta_frontier_status, "no_optional_refinement")

    def test_tree_calibrates_from_actual_utility_range(self) -> None:
        tree = _parameter_free_tree()
        self.assertTrue(tree._calibrate_frontier_complexity())
        self.assertAlmostEqual(tree.gta_frontier_s0, 0.15)
        self.assertAlmostEqual(tree.gta_frontier_sl, 0.60)
        self.assertEqual(tree.gta_frontier_k0, 2)
        self.assertEqual(tree.gta_frontier_leaf_count, 4)
        self.assertAlmostEqual(tree.gta_frontier_lambda, 0.225)
        self.assertLess(tree.gta_frontier_sl, 1.0)

    def test_endpoint_objectives_are_both_zero(self) -> None:
        tree = _parameter_free_tree()
        selected = tree._select_gta_frontier()
        root_children = tree.tree[0].children_ids
        terminal_leaves = sorted(_terminal_leaf_ids(tree))
        self.assertAlmostEqual(
            _normalized_frontier_objective(tree, root_children), 0.0
        )
        self.assertAlmostEqual(
            _normalized_frontier_objective(tree, terminal_leaves), 0.0
        )
        self.assertEqual(selected, [1, 5, 6])
        self.assertGreater(_normalized_frontier_objective(tree, selected), 0.0)

    def test_positive_gain_tree_can_mix_internal_and_leaf_anchors(self) -> None:
        tree = _parameter_free_tree()
        selected = tree._select_gta_frontier()
        self.assertEqual(selected, [1, 5, 6])
        self.assertTrue(any(tree.tree[node_id].children_ids for node_id in selected))
        self.assertTrue(any(not tree.tree[node_id].children_ids for node_id in selected))

    def test_zero_utility_span_returns_the_coarsest_frontier(self) -> None:
        tree = _tree(
            [
                _node(0, list(range(4)), 0.1, children=[1, 2]),
                _node(1, [0, 1], 0.2, parent_id=0, children=[3, 4]),
                _node(2, [2, 3], 0.2, parent_id=0, children=[5, 6]),
                _node(3, [0], 0.2, parent_id=1),
                _node(4, [1], 0.2, parent_id=1),
                _node(5, [2], 0.2, parent_id=2),
                _node(6, [3], 0.2, parent_id=2),
            ],
            n_samples=4,
        )
        self.assertEqual(tree._select_gta_frontier(), [1, 2])
        self.assertEqual(tree.gta_frontier_status, "no_positive_utility_span")

    def test_terminal_node_never_yields_an_empty_frontier(self) -> None:
        tree = _tree([_node(0, [0], 0.1)], n_samples=1)
        selected, utility = tree._anchor_frontier(0)
        self.assertEqual(selected, [0])
        self.assertAlmostEqual(utility, 0.1)

    def test_tie_within_tolerance_selects_current_node_deterministically(self) -> None:
        tree = _tree(
            [
                _node(1, [0, 1, 2, 3], 0.25, children=[2, 3]),
                _node(2, [0, 1], 0.5, parent_id=1),
                _node(3, [2, 3], 0.5, parent_id=1),
            ],
            n_samples=8,
        )
        tree.gta_frontier_lambda = 0.125
        selected, utility = tree._anchor_frontier(1)
        self.assertEqual(selected, [1])
        self.assertLessEqual(abs(utility), TIE_TOLERANCE)

    def test_dynamic_program_matches_exhaustive_exact_cover_optimum(self) -> None:
        tree = _parameter_free_tree()
        selected = tree._select_gta_frontier()
        candidates = _all_frontiers(tree, 0, allow_self=False)
        optimum = max(value for _, value in candidates)
        selected_utility = sum(
            tree._anchor_selection_utility(node_id) for node_id in selected
        )
        self.assertAlmostEqual(selected_utility, optimum)
        self.assertTrue(
            any(
                candidate == selected and abs(value - optimum) <= TIE_TOLERANCE
                for candidate, value in candidates
            )
        )
        self.assertTrue(selected)


if __name__ == "__main__":
    unittest.main()
