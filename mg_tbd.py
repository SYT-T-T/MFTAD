import gc
import multiprocessing as mp
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import psutil
import torch
from numba import jit, prange
warnings.filterwarnings('ignore')

@dataclass
class GranularBallNode:
    id: int
    scale_level: int
    sample_indices: Set[int]
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    is_leaf: bool = False
    cohesion: float = 0.0
    is_gta: bool = False

class FRS_Tree_Builder:
    def __init__(self, X: torch.Tensor, view_features: Set[int], delta: float, tree_params: Dict):
        self.view_indices = list(view_features)
        if self.view_indices:
            self.X_view = X[:, self.view_indices]
        else:
            self.X_view = torch.empty(X.shape[0], 0)
        self.n_samples = X.shape[0]
        self.delta = delta
        self.params = tree_params
        self.R_view: Optional[np.ndarray] = None
        self.tree: Dict[int, GranularBallNode] = {}
        self.sample_leaf_map: Dict[int, int] = {}
        self.evolution_paths: Dict[int, List[int]] = {}
        self.local_gtas: List[int] = []

    def build(self):
        self.R_view = self._compute_view_similarity_relation()
        root_node = GranularBallNode(
            id=0,
            scale_level=0,
            sample_indices=set(range(self.n_samples)),
        )
        root_node.cohesion = self._calculate_cohesion(root_node.sample_indices)
        self.tree[0] = root_node
        split_queue = [root_node]
        gb_id_counter = 1
        while split_queue:
            current_ball = split_queue.pop(0)
            pivots = self._find_split_pivots(current_ball)
            if pivots is None:
                current_ball.is_leaf = True
                continue
            child_A_samples, child_B_samples = self._assign_samples(current_ball, pivots)
            if not child_A_samples or not child_B_samples:
                current_ball.is_leaf = True
                continue
            cohesion_gain = self._calculate_cohesion_gain(current_ball, child_A_samples, child_B_samples)
            if cohesion_gain < self.params.get('cohesion_improvement_threshold', 0.01):
                current_ball.is_leaf = True
                continue
            child_A = self._create_child_ball(gb_id_counter, current_ball, child_A_samples)
            gb_id_counter += 1
            child_B = self._create_child_ball(gb_id_counter, current_ball, child_B_samples)
            gb_id_counter += 1
            current_ball.children_ids = [child_A.id, child_B.id]
            self.tree[child_A.id] = child_A
            self.tree[child_B.id] = child_B
            split_queue.extend([child_A, child_B])
        self._build_sample_leaf_map()
        self._build_evolution_paths()
        self._identify_local_gtas()

    def _compute_view_similarity_relation(self) -> np.ndarray:
        data_np = self.X_view.numpy()
        n_view_features = self.X_view.shape[1]

        if n_view_features == 0:
            return np.ones((self.n_samples, self.n_samples))

        varepsilon = np.zeros(n_view_features)
        for j in range(n_view_features):
            std_dev = np.std(data_np[:, j])
            if std_dev > 1e-8:
                varepsilon[j] = std_dev / self.delta

        weighted_similarity = np.zeros((self.n_samples, self.n_samples))
        attribute_weights = np.zeros(n_view_features)

        for j in range(n_view_features):
            dist_matrix = np.abs(data_np[:, j][:, np.newaxis] - data_np[:, j])

            if varepsilon[j] > 1e-8:
                R_j = 1 - np.clip(dist_matrix / varepsilon[j], 0, 1)
            else:
                R_j = (dist_matrix < 1e-8).astype(float)

            cardinalities = np.sum(R_j, axis=1)
            log_card = np.log2(
                cardinalities,
                where=cardinalities > 0,
                out=np.zeros_like(cardinalities, dtype=float),
            )
            attribute_weights[j] = -np.mean(log_card)

            del R_j, dist_matrix

        if np.sum(attribute_weights) > 0:
            attribute_weights /= np.sum(attribute_weights)
        else:
            attribute_weights = np.ones(n_view_features) / n_view_features

        for j in range(n_view_features):
            dist_matrix = np.abs(data_np[:, j][:, np.newaxis] - data_np[:, j])

            if varepsilon[j] > 1e-8:
                R_j = 1 - np.clip(dist_matrix / varepsilon[j], 0, 1)
            else:
                R_j = (dist_matrix < 1e-8).astype(float)

            weighted_similarity += attribute_weights[j] * R_j

            del R_j, dist_matrix

        return weighted_similarity

    def _calculate_cohesion(self, sample_indices: Set[int]) -> float:
        if len(sample_indices) <= 1:
            return 1.0
        indices = list(sample_indices)
        sub_matrix = self.R_view[np.ix_(indices, indices)]
        if sub_matrix.shape[0] <= 1:
            return 1.0
        if sub_matrix.shape[0] != sub_matrix.shape[1]:
            return 0.0
        return np.mean(sub_matrix[np.triu_indices(len(indices), k=1)])

    def _calculate_geometric_center(self, sample_indices: Set[int]) -> torch.Tensor:
        if not sample_indices:
            return torch.zeros(self.X_view.shape[1])
        return self.X_view[list(sample_indices)].mean(dim=0)

    def _find_split_pivots(self, ball: GranularBallNode) -> Optional[Tuple[int, int]]:
        indices = list(ball.sample_indices)
        if len(indices) < 2:
            return None
        pivot_center = self._calculate_geometric_center(ball.sample_indices)
        sub_X_view = self.X_view[indices]
        distances_to_center = torch.cdist(sub_X_view, pivot_center.unsqueeze(0)).squeeze()
        closest_point_local_idx = torch.argmin(distances_to_center)
        pivot_A_idx = indices[closest_point_local_idx]
        farthest_point_local_idx = torch.argmax(distances_to_center)
        pivot_B_idx = indices[farthest_point_local_idx]
        if pivot_A_idx == pivot_B_idx:
            if len(indices) > 1:
                sorted_dist_indices = torch.argsort(distances_to_center, descending=True)
                if len(sorted_dist_indices) > 1:
                    second_farthest_local_idx = sorted_dist_indices[1].item()
                    pivot_B_idx = indices[second_farthest_local_idx]
                else:
                    return None
            else:
                return None
        return pivot_A_idx, pivot_B_idx

    def _assign_samples(self, ball: GranularBallNode, pivots: Tuple[int, int]) -> Tuple[Set[int], Set[int]]:
        pivot_A, pivot_B = pivots
        indices = list(ball.sample_indices)
        sim_to_A = self.R_view[indices, pivot_A]
        sim_to_B = self.R_view[indices, pivot_B]
        mask_A = sim_to_A >= sim_to_B
        arr_indices = np.array(indices)
        return set(arr_indices[mask_A]), set(arr_indices[~mask_A])

    def _calculate_cohesion_gain(self, parent_ball: GranularBallNode, samples_A: Set[int], samples_B: Set[int]) -> float:
        cohesion_A = self._calculate_cohesion(samples_A)
        cohesion_B = self._calculate_cohesion(samples_B)
        len_A, len_B = len(samples_A), len(samples_B)
        if len_A + len_B == 0:
            return -1.0
        cohesion_after = (len_A * cohesion_A + len_B * cohesion_B) / (len_A + len_B)
        return cohesion_after - parent_ball.cohesion
    
    def _create_child_ball(self, gb_id: int, parent_ball: GranularBallNode, samples: Set[int]) -> GranularBallNode:
        child_ball = GranularBallNode(id=gb_id, scale_level=parent_ball.scale_level + 1, sample_indices=samples, parent_id=parent_ball.id)
        child_ball.cohesion = self._calculate_cohesion(samples)
        return child_ball

    def _build_sample_leaf_map(self):
        for ball_id, ball in self.tree.items():
            if ball.is_leaf:
                for sample_idx in ball.sample_indices:
                    self.sample_leaf_map[sample_idx] = ball_id
    
    def _build_evolution_paths(self):
        for sample_idx in range(self.n_samples):
            leaf_id = self.sample_leaf_map.get(sample_idx)
            if leaf_id is None:
                continue
            path = []
            curr_id = leaf_id
            while curr_id is not None:
                if curr_id not in self.tree:
                    break
                path.append(curr_id)
                curr_id = self.tree[curr_id].parent_id
            self.evolution_paths[sample_idx] = path[::-1]

    def _identify_local_gtas(self):
        candidates = []
        min_pop_ratio = self.params.get('gta_min_popularity_ratio', 0.1)
        max_gta_count = self.params.get('gta_max_count', 5)
        for ball in self.tree.values():
            if ball.is_leaf:
                continue
            popularity_ratio = len(ball.sample_indices) / self.n_samples
            if popularity_ratio < min_pop_ratio:
                continue
            gta_score = popularity_ratio * ball.cohesion
            candidates.append({'id': ball.id, 'score': gta_score})
        candidates.sort(key=lambda x: x['score'], reverse=True)
        self.local_gtas = [c['id'] for c in candidates[:max_gta_count]]
        for gta_id in self.local_gtas:
            self.tree[gta_id].is_gta = True


@jit(nopython=True, parallel=True)
def _compute_leaf_scores_numba(R_view, leaf_indices_flat, leaf_sizes, leaf_offsets, n_samples):
    scores = np.full(n_samples, 0.5)
    
    for leaf_idx in prange(len(leaf_sizes)):
        start_idx = leaf_offsets[leaf_idx]
        end_idx = start_idx + leaf_sizes[leaf_idx]
        leaf_members = leaf_indices_flat[start_idx:end_idx]
        
        if len(leaf_members) <= 1:
            for i in leaf_members:
                scores[i] = 1.0
            continue
        
        for i, sample_i in enumerate(leaf_members):
            total_sim = 0.0
            count = 0
            for j, sample_j in enumerate(leaf_members):
                if i != j:
                    total_sim += R_view[sample_i, sample_j]
                    count += 1
            
            if count > 0:
                avg_sim = total_sim / count
                scores[sample_i] = 1.0 - avg_sim
            else:
                scores[sample_i] = 1.0
    
    return scores

@jit(nopython=True, parallel=True)
def _compute_gta_scores_numba(R_view, gta_indices_flat, gta_sizes, gta_offsets, n_samples):
    scores = np.full(n_samples, 0.5)
    
    if len(gta_sizes) == 0:
        return scores
    
    for sample_idx in prange(n_samples):
        max_trust = 0.0
        
        for gta_idx in range(len(gta_sizes)):
            start_idx = gta_offsets[gta_idx]
            end_idx = start_idx + gta_sizes[gta_idx]
            gta_members = gta_indices_flat[start_idx:end_idx]
            
            if len(gta_members) == 0:
                continue
            
            total_sim = 0.0
            for member in gta_members:
                total_sim += R_view[sample_idx, member]
            avg_trust = total_sim / len(gta_members)
            
            if avg_trust > max_trust:
                max_trust = avg_trust
        
        scores[sample_idx] = 1.0 - max_trust
    
    return scores

@jit(nopython=True)
def _compute_path_scores_numba(R_view, evolution_paths_flat, path_sizes, path_offsets, 
                               ball_indices_flat, ball_sizes, ball_offsets, 
                               ball_id_to_idx_keys, ball_id_to_idx_values, n_samples):
    scores = np.zeros(n_samples)
    
    for sample_idx in range(len(path_sizes)):
        if path_sizes[sample_idx] <= 1:
            continue
            
        start_idx = path_offsets[sample_idx]
        end_idx = start_idx + path_sizes[sample_idx]
        path_balls = evolution_paths_flat[start_idx:end_idx]
        
        cardinalities = np.zeros(len(path_balls))
        for i, ball_id in enumerate(path_balls):
            ball_idx = -1
            for j in range(len(ball_id_to_idx_keys)):
                if ball_id_to_idx_keys[j] == ball_id:
                    ball_idx = ball_id_to_idx_values[j]
                    break
            
            if ball_idx >= 0:
                ball_start = ball_offsets[ball_idx]
                ball_end = ball_start + ball_sizes[ball_idx]
                ball_members = ball_indices_flat[ball_start:ball_end]
                
                total_sim = 0.0
                for member in ball_members:
                    total_sim += R_view[sample_idx, member]
                cardinalities[i] = total_sim
        
        max_card = np.max(cardinalities)
        if max_card > 0:
            norm_cards = cardinalities / max_card
        else:
            norm_cards = cardinalities
        
        std_card = np.std(norm_cards)
        scores[sample_idx] = 1.0 - (1.0 / (1.0 + std_card))
    
    return scores


class MVT_FRS:
    def __init__(self,
                 mfgad_lower_bound: int = 50,
                 mfgad_upper_bound: int = 100,
                 random_views_if_needed: int = 50,
                 tree_params: Dict = None,
                 delta: float = 2.5,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 memory_safety_ratio: float = 0.8,
                 use_numba_acceleration: bool = True):
        self.mfgad_lower_bound = mfgad_lower_bound
        self.mfgad_upper_bound = mfgad_upper_bound
        self.random_views_if_needed = random_views_if_needed
        self.tree_params = tree_params if tree_params is not None else {}
        self.delta = delta
        self.learned_score_weights: Optional[np.ndarray] = None
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.memory_safety_ratio = memory_safety_ratio
        self.use_numba_acceleration = use_numba_acceleration
        self.X: Optional[torch.Tensor] = None
        self.n_samples: int = 0
        self.n_features: int = 0
        self.attribute_weights: Optional[np.ndarray] = None
        self.forest: List[FRS_Tree_Builder] = []
        self.view_weights: Optional[np.ndarray] = None
        self.used_strategy: str = "Unknown"

    def _get_available_memory_gb(self) -> float:
        try:
            available_memory = psutil.virtual_memory().available
            return available_memory / (1024**3)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Warning: Failed to get memory info: {e}, using default 8GB")
            return 8.0

    def _calculate_safe_parallel_jobs(self, views: List[Set[int]]) -> int:
        if not views:
            return 1
            
        available_memory_gb = self._get_available_memory_gb()
        safe_memory_gb = available_memory_gb * self.memory_safety_ratio
        
        avg_view_features = np.mean([len(view) for view in views])
        single_tree_memory_gb = self._estimate_single_tree_memory_gb(self.n_samples, int(avg_view_features))
        
        if single_tree_memory_gb <= 0:
            safe_parallel_jobs = self.n_jobs
        else:
            safe_parallel_jobs = max(1, int(safe_memory_gb / single_tree_memory_gb))
        
        safe_parallel_jobs = min(safe_parallel_jobs, self.n_jobs, len(views))
        
        if self.verbose:
            print(f"  🧠 Memory Analysis:")
            print(f"     - Available memory: {available_memory_gb:.2f} GB")
            print(f"     - Safe memory limit: {safe_memory_gb:.2f} GB")
            print(f"     - Single tree memory: {single_tree_memory_gb:.3f} GB")
            print(f"     - Requested parallel jobs: {self.n_jobs}")
            print(f"     - Safe parallel jobs: {safe_parallel_jobs}")
        
        return safe_parallel_jobs

    def _build_single_tree(self, view_data: Tuple[int, Set[int]]) -> Tuple[int, FRS_Tree_Builder]:
        view_index, view = view_data
        try:
            builder = FRS_Tree_Builder(self.X, view, self.delta, self.tree_params)
            builder.build()
            return view_index, builder
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Warning: Tree {view_index} building failed: {e}")
            return view_index, None

    def _build_forest_parallel(self, views: List[Set[int]]) -> List[FRS_Tree_Builder]:
        if len(views) <= 1:
            return self._build_forest_serial(views)
        
        safe_parallel_jobs = self._calculate_safe_parallel_jobs(views)
        
        if safe_parallel_jobs == 1:
            if self.verbose:
                print(f"  🔄 Memory constraints require serial processing")
            return self._build_forest_serial(views)
        elif safe_parallel_jobs < len(views):
            if self.verbose:
                print(f"  🔄 Using batch parallel processing: {safe_parallel_jobs} workers")
            return self._build_forest_batch_parallel(views, safe_parallel_jobs)
        else:
            if self.verbose:
                print(f"  🔄 Using full parallel processing: {safe_parallel_jobs} workers")
            return self._build_forest_full_parallel(views, safe_parallel_jobs)

    def _build_forest_full_parallel(self, views: List[Set[int]], n_workers: int) -> List[FRS_Tree_Builder]:
        indexed_views = [(i, view) for i, view in enumerate(views)]
        forest_results = [None] * len(views)
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_index = {
                executor.submit(self._build_single_tree, view_data): view_data[0] 
                for view_data in indexed_views
            }
            
            completed_count = 0
            for future in as_completed(future_to_index):
                try:
                    view_index, builder = future.result()
                    forest_results[view_index] = builder
                    completed_count += 1
                    
                    if self.verbose and completed_count % max(1, len(views) // 10) == 0:
                        progress = (completed_count / len(views)) * 100
                        print(f"  📊 Progress: {completed_count}/{len(views)} trees ({progress:.1f}%)")
                        
                except Exception as e:
                    view_index = future_to_index[future]
                    if self.verbose:
                        print(f"  ❌ Tree {view_index} failed: {e}")
                    forest_results[view_index] = None
        
        successful_builders = [builder for builder in forest_results if builder is not None]
        
        if len(successful_builders) < len(views):
            failed_count = len(views) - len(successful_builders)
            if self.verbose:
                print(f"  ⚠️  {failed_count} trees failed to build, continuing with {len(successful_builders)} trees")
        
        return successful_builders

    def _build_forest_batch_parallel(self, views: List[Set[int]], n_workers: int) -> List[FRS_Tree_Builder]:
        all_builders = []
        batch_size = n_workers
        
        for batch_start in range(0, len(views), batch_size):
            batch_end = min(batch_start + batch_size, len(views))
            batch_views = views[batch_start:batch_end]
            
            if self.verbose:
                print(f"  📦 Processing batch {batch_start//batch_size + 1}/{(len(views) + batch_size - 1)//batch_size}: views {batch_start}-{batch_end-1}")
            
            batch_builders = self._build_forest_full_parallel(batch_views, n_workers)
            all_builders.extend(batch_builders)
            
            gc.collect()
        
        return all_builders

    def _build_forest_serial(self, views: List[Set[int]]) -> List[FRS_Tree_Builder]:
        forest = []
        for i, view in enumerate(views):
            try:
                builder = FRS_Tree_Builder(self.X, view, self.delta, self.tree_params)
                builder.build()
                forest.append(builder)
                
                if self.verbose and (i + 1) % max(1, len(views) // 10) == 0:
                    progress = ((i + 1) / len(views)) * 100
                    print(f"  📊 Progress: {i + 1}/{len(views)} trees ({progress:.1f}%)")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Warning: Tree {i} building failed: {e}")
                continue
        
        return forest

    def fit_predict(self, X: torch.Tensor) -> np.ndarray:
        self.X = X
        self.n_samples, self.n_features = X.shape

        if self.verbose:
            acceleration_status = "ON" if self.use_numba_acceleration else "OFF"
            print(f"🚀 MVT-FRS v6.0 (Numba Turbo: {acceleration_status}) 启动... on data {X.shape}")
            print(f"  🔧 Max parallel workers: {self.n_jobs}")
            print(f"  🧠 Memory safety ratio: {self.memory_safety_ratio}")
            
            phase1_memory = self._estimate_phase1_memory_gb()
            available_memory = self._get_available_memory_gb()
            print(f"  📊 Memory Analysis:")
            print(f"     - Phase 1 estimated: {phase1_memory:.2f} GB")
            print(f"     - Available memory: {available_memory:.2f} GB")

        total_start_time = time.time()

        phase1_start = time.time()
        views = self._generate_adaptive_views()
        phase1_time = time.time() - phase1_start

        if not views: 
            warnings.warn("警告：没有生成任何视角。")
            return np.random.rand(self.n_samples)

        phase2_start = time.time()
        if self.verbose: 
            print(f"\n--- Building Forest of {len(views)} FRS-Trees (Smart Parallel) using '{self.used_strategy}' strategy ---")
        
        self.forest = self._build_forest_parallel(views)
        
        if self.verbose: 
            print(f"  ✅ Successfully built {len(self.forest)} trees")
        
        if not self.forest:
            warnings.warn("警告：没有成功构建任何树。")
            return np.random.rand(self.n_samples)
        phase2_time = time.time() - phase2_start

        phase3_start = time.time()
        if self.verbose: 
            acceleration_msg = "Numba-Accelerated" if self.use_numba_acceleration else "Standard"
            print(f"\n--- Computing and Fusing Anomaly Scores ({acceleration_msg}) ---")
        final_scores = self._compute_and_fuse_scores_turbo()
        phase3_time = time.time() - phase3_start

        if self.verbose:
            print(f"\n⏱️ 阶段耗时统计:")
            print(f"  - 视图生成: {phase1_time:.3f}s")
            print(f"  - 森林构建: {phase2_time:.3f}s")
            print(f"  - 分数计算: {phase3_time:.3f}s ⚡")

        total_time = time.time() - total_start_time
        if self.verbose: 
            print(f"\n📊 Total Algorithm Time: {total_time:.3f}s")

        return final_scores

    def _compute_and_fuse_scores_turbo(self) -> np.ndarray:
        if not self.forest: 
            return np.full(self.n_samples, 0.5)

        all_view_scores = []

        if self.use_numba_acceleration:
            for builder in self.forest:
                scores_leaf = self._compute_leaf_scores_turbo(builder)
                scores_gta = self._compute_gta_scores_turbo(builder)
                scores_path = self._compute_path_scores_turbo(builder)

                all_view_scores.append(np.stack([scores_leaf, scores_gta, scores_path], axis=1))
        else:
            for builder in self.forest:
                scores_leaf = self._compute_all_leaf_scores_vectorized(builder)
                scores_gta = self._compute_all_gta_scores_vectorized(builder)
                scores_path = self._compute_all_path_scores_vectorized(builder)

                all_view_scores.append(np.stack([scores_leaf, scores_gta, scores_path], axis=1))

        fused_scores_3d = np.zeros((self.n_samples, 3))
        score_names = ['Leaf', 'GTA', 'Path']

        for dim in range(3):
            scores_for_dim = [s[:, dim] for s in all_view_scores]
            fused_scores_3d[:, dim] = self._fuse_single_dimension_scores(scores_for_dim)
            if self.verbose and self.view_weights is not None:
                print(f"  - Fused '{score_names[dim]}' scores. Avg view weight: {np.mean(self.view_weights):.3f}")

        certainty = np.abs(fused_scores_3d - 0.5) * 2
        total_certainty = np.sum(certainty, axis=1, keepdims=True)
        total_certainty[total_certainty < 1e-8] = 1.0
        dynamic_weights = certainty / total_certainty
        
        self.learned_score_weights = np.mean(dynamic_weights, axis=0)
        if self.verbose:
            print(f"  - Average Dynamic Score Weights (Leaf, GTA, Path): "
                  f"[{self.learned_score_weights[0]:.3f}, "
                  f"{self.learned_score_weights[1]:.3f}, "
                  f"{self.learned_score_weights[2]:.3f}]")

        final_scores = np.sum(fused_scores_3d * dynamic_weights, axis=1)

        min_s, max_s = np.min(final_scores), np.max(final_scores)
        if max_s > min_s:
            final_scores = (final_scores - min_s) / (max_s - min_s)

        return final_scores

    def _compute_leaf_scores_turbo(self, builder: FRS_Tree_Builder) -> np.ndarray:
        leaf_nodes = [(ball_id, list(ball.sample_indices)) 
                      for ball_id, ball in builder.tree.items() if ball.is_leaf]

        if not leaf_nodes:
            return np.full(self.n_samples, 0.5)

        leaf_indices_flat = []
        leaf_sizes = []
        leaf_offsets = []

        current_offset = 0
        for _, indices in leaf_nodes:
            leaf_indices_flat.extend(indices)
            leaf_sizes.append(len(indices))
            leaf_offsets.append(current_offset)
            current_offset += len(indices)

        leaf_indices_flat = np.array(leaf_indices_flat, dtype=np.int32)
        leaf_sizes = np.array(leaf_sizes, dtype=np.int32)
        leaf_offsets = np.array(leaf_offsets, dtype=np.int32)

        return _compute_leaf_scores_numba(builder.R_view, leaf_indices_flat, 
                                        leaf_sizes, leaf_offsets, self.n_samples)

    def _compute_gta_scores_turbo(self, builder: FRS_Tree_Builder) -> np.ndarray:
        if not builder.local_gtas:
            return np.full(self.n_samples, 0.5)

        gta_indices_flat = []
        gta_sizes = []
        gta_offsets = []

        current_offset = 0
        for gta_id in builder.local_gtas:
            gta_indices = list(builder.tree[gta_id].sample_indices)
            gta_indices_flat.extend(gta_indices)
            gta_sizes.append(len(gta_indices))
            gta_offsets.append(current_offset)
            current_offset += len(gta_indices)

        gta_indices_flat = np.array(gta_indices_flat, dtype=np.int32)
        gta_sizes = np.array(gta_sizes, dtype=np.int32)
        gta_offsets = np.array(gta_offsets, dtype=np.int32)

        return _compute_gta_scores_numba(builder.R_view, gta_indices_flat, 
                                       gta_sizes, gta_offsets, self.n_samples)

    def _compute_path_scores_turbo(self, builder: FRS_Tree_Builder) -> np.ndarray:
        if not builder.evolution_paths:
            return np.zeros(self.n_samples)

        evolution_paths_flat = []
        path_sizes = []
        path_offsets = []

        current_offset = 0
        for sample_idx in range(self.n_samples):
            path_ids = builder.evolution_paths.get(sample_idx, [])
            evolution_paths_flat.extend(path_ids)
            path_sizes.append(len(path_ids))
            path_offsets.append(current_offset)
            current_offset += len(path_ids)

        ball_indices_flat = []
        ball_sizes = []
        ball_offsets = []
        ball_id_to_idx = {}

        current_offset = 0
        for idx, (ball_id, ball) in enumerate(builder.tree.items()):
            ball_indices = list(ball.sample_indices)
            ball_indices_flat.extend(ball_indices)
            ball_sizes.append(len(ball_indices))
            ball_offsets.append(current_offset)
            ball_id_to_idx[ball_id] = idx
            current_offset += len(ball_indices)

        evolution_paths_flat = np.array(evolution_paths_flat, dtype=np.int32)
        path_sizes = np.array(path_sizes, dtype=np.int32)
        path_offsets = np.array(path_offsets, dtype=np.int32)
        ball_indices_flat = np.array(ball_indices_flat, dtype=np.int32)
        ball_sizes = np.array(ball_sizes, dtype=np.int32)
        ball_offsets = np.array(ball_offsets, dtype=np.int32)

        ball_id_to_idx_keys = np.array(list(ball_id_to_idx.keys()), dtype=np.int32)
        ball_id_to_idx_values = np.array(list(ball_id_to_idx.values()), dtype=np.int32)

        return _compute_path_scores_numba(builder.R_view, evolution_paths_flat, 
                                        path_sizes, path_offsets, ball_indices_flat, 
                                        ball_sizes, ball_offsets, 
                                        ball_id_to_idx_keys, ball_id_to_idx_values, 
                                        self.n_samples)

    def _compute_all_leaf_scores_vectorized(self, builder: FRS_Tree_Builder) -> np.ndarray:
        scores = np.full(self.n_samples, 0.5)
        leaf_to_samples = {ball_id: list(ball.sample_indices) 
                           for ball_id, ball in builder.tree.items() if ball.is_leaf}

        for leaf_id, indices in leaf_to_samples.items():
            if len(indices) <= 1:
                scores[indices] = 1.0
                continue

            sub_matrix = builder.R_view[np.ix_(indices, indices)]
            sum_of_sims = np.sum(sub_matrix, axis=1) - 1.0
            num_others = len(indices) - 1

            if num_others > 0:
                avg_sims = sum_of_sims / num_others
                scores[indices] = 1.0 - avg_sims
            else:
                scores[indices] = 1.0
        return scores

    def _compute_all_gta_scores_vectorized(self, builder: FRS_Tree_Builder) -> np.ndarray:
        if not builder.local_gtas:
            return np.full(self.n_samples, 0.5)

        gta_trust_matrix = np.zeros((self.n_samples, len(builder.local_gtas)))
        for j, gta_id in enumerate(builder.local_gtas):
            gta_indices = list(builder.tree[gta_id].sample_indices)
            if not gta_indices:
                continue

            gta_trust_matrix[:, j] = np.mean(builder.R_view[:, gta_indices], axis=1)

        max_gta_trust = np.max(gta_trust_matrix, axis=1)
        return 1.0 - max_gta_trust

    def _compute_all_path_scores_vectorized(self, builder: FRS_Tree_Builder) -> np.ndarray:
        scores = np.zeros(self.n_samples)

        if not builder.evolution_paths:
            return scores

        ball_to_indices = {}
        for ball_id, ball in builder.tree.items():
            ball_to_indices[ball_id] = np.array(list(ball.sample_indices))

        computation_pairs = []
        sample_to_result_idx = {}
        result_shapes = []

        for sample_idx, path_ids in builder.evolution_paths.items():
            if len(path_ids) > 1:
                sample_to_result_idx[sample_idx] = len(result_shapes)
                result_shapes.append(len(path_ids))
                for ball_id in path_ids:
                    computation_pairs.append((sample_idx, ball_id))

        if not computation_pairs:
            return scores

        sample_indices = np.array([pair[0] for pair in computation_pairs])
        ball_indices_list = [ball_to_indices[pair[1]] for pair in computation_pairs]

        batch_similarities = []
        for i, ball_indices in enumerate(ball_indices_list):
            sample_idx = sample_indices[i]
            if len(ball_indices) > 0:
                sim_sum = np.sum(builder.R_view[sample_idx, ball_indices])
            else:
                sim_sum = 0.0
            batch_similarities.append(sim_sum)

        batch_similarities = np.array(batch_similarities)

        start_idx = 0
        for sample_idx, path_ids in builder.evolution_paths.items():
            if len(path_ids) <= 1:
                continue

            path_length = len(path_ids)
            end_idx = start_idx + path_length

            cardinalities = batch_similarities[start_idx:end_idx]

            max_card = np.max(cardinalities)
            if max_card > 0:
                norm_cards = cardinalities / max_card
            else:
                norm_cards = cardinalities

            std_card = np.std(norm_cards)
            scores[sample_idx] = 1.0 - (1.0 / (1.0 + std_card))

            start_idx = end_idx

        return scores

    def _estimate_phase1_memory_gb(self) -> float:
        single_matrix_memory = self.n_samples * self.n_samples * 8 / (1024**3)
        temp_memory = single_matrix_memory * 0.5
        return single_matrix_memory + temp_memory
    
    def _check_phase1_memory_safety(self) -> bool:
        estimated_memory = self._estimate_phase1_memory_gb()
        available_memory = self._get_available_memory_gb()

        if self.verbose:
            print(f"  🧠 Phase 1 Memory Check:")
            print(f"     - Estimated memory per matrix: {estimated_memory:.2f} GB")
            print(f"     - Available memory: {available_memory:.2f} GB")
            print(f"     - Memory safety ratio: {self.memory_safety_ratio}")

        return estimated_memory < available_memory * self.memory_safety_ratio
    
    def _generate_adaptive_views(self) -> List[Set[int]]:
        if self.verbose:
            print("\n--- Phase 1: Generating Views with Adaptive Strategy ---")
        
        if not self._check_phase1_memory_safety():
            if self.verbose:
                print("  ⚠️  Warning: Phase 1 may cause memory overflow, using simplified strategy")
            self.attribute_weights = np.ones(self.n_features) / self.n_features
        else:
            self._compute_attribute_weights_optimized()
        
        sorted_indices = np.argsort(self.attribute_weights)
        min_features = max(2, int(self.n_features * 0.1))
        mfgad_views = []
        current_fs = list(range(self.n_features))
        if self.n_features > min_features:
            for i in range(self.n_features - min_features):
                idx_to_remove = sorted_indices[i]
                if idx_to_remove in current_fs: current_fs.remove(idx_to_remove)
                mfgad_views.append(set(current_fs))
        current_rs = list(range(self.n_features))
        if self.n_features > min_features:
            for i in range(self.n_features - min_features):
                idx_to_remove = sorted_indices[-(i+1)]
                if idx_to_remove in current_rs: current_rs.remove(idx_to_remove)
                mfgad_views.append(set(current_rs))
        num_mfgad_views = len(set(map(frozenset, mfgad_views)))
        if self.verbose: print(f"  - MFGAD strategy can generate {num_mfgad_views} unique views.")
        final_views = []
        if num_mfgad_views > self.mfgad_upper_bound:
            self.used_strategy = "Structured Sampled MFGAD Forest (MFGAD views > upper bound)"
            if self.verbose: print(f"  - Decision: Switching to {self.used_strategy}")
            
            unique_mfgad_views = list(set(map(frozenset, mfgad_views)))
            unique_mfgad_views = [set(view) for view in unique_mfgad_views]
            
            keep_ratio = self.mfgad_upper_bound / num_mfgad_views
            
            half_point = len(unique_mfgad_views) // 2
            
            first_half_keep = min(half_point, int(self.mfgad_upper_bound * 0.5))
            if first_half_keep > 0 and half_point > 0:
                step_size = max(1, half_point // first_half_keep)
                for i in range(0, half_point, step_size):
                    if len(final_views) < self.mfgad_upper_bound:
                        final_views.append(unique_mfgad_views[i])
            
            second_half_keep = self.mfgad_upper_bound - len(final_views)
            if second_half_keep > 0 and (len(unique_mfgad_views) - half_point) > 0:
                step_size = max(1, (len(unique_mfgad_views) - half_point) // second_half_keep)
                for i in range(half_point, len(unique_mfgad_views), step_size):
                    if len(final_views) < self.mfgad_upper_bound:
                        final_views.append(unique_mfgad_views[i])
            
            if self.verbose:
                print(f"  - Sampled {len(final_views)} views from original {num_mfgad_views} MFGAD views")
        elif num_mfgad_views < self.mfgad_lower_bound:
            self.used_strategy = "Hybrid Forest (MFGAD views < lower bound)"
            if self.verbose: print(f"  - Decision: Switching to {self.used_strategy}")
            final_views.extend(mfgad_views)
            n_sub_features = max(2, int(self.n_features * 0.7))
            for _ in range(self.random_views_if_needed):
                if self.n_features >= n_sub_features:
                    final_views.append(set(np.random.choice(self.n_features, n_sub_features, replace=False)))
        else:
            self.used_strategy = "Pure MFGAD Sequence (view count is optimal)"
            if self.verbose: print(f"  - Decision: Using {self.used_strategy}")
            final_views.extend(mfgad_views)
        unique_views = [set(v) for v in set(map(frozenset, final_views))]
        if self.verbose: print(f"  - Final unique views generated: {len(unique_views)}")
        return unique_views
        
    def _compute_attribute_weights_optimized(self):
        data_np = self.X.numpy()
        attribute_weights = np.zeros(self.n_features)
        
        for j in range(self.n_features):
            std_dev = np.std(data_np[:, j])
            varepsilon = std_dev / self.delta if std_dev > 1e-8 else 1e-8
            
            dist_matrix = np.abs(data_np[:, j][:, np.newaxis] - data_np[:, j])
            R_j = 1 - np.clip(dist_matrix / varepsilon, 0, 1) if varepsilon > 1e-8 else (dist_matrix < 1e-8).astype(float)
            
            cardinalities = np.sum(R_j, axis=1)
            log_card = np.log2(cardinalities, where=cardinalities > 0, out=np.zeros_like(cardinalities, dtype=float))
            attribute_weights[j] = -np.mean(log_card)
            
            del dist_matrix, R_j
        
        if np.sum(attribute_weights) > 0:
            self.attribute_weights = attribute_weights / np.sum(attribute_weights)
        else:
            self.attribute_weights = np.ones(self.n_features) / self.n_features
        
    def _fuse_single_dimension_scores(self, scores_list: List[np.ndarray]) -> np.ndarray:
        if not scores_list: return np.full(self.n_samples, 0.5)
        all_probs = []
        for scores in scores_list:
            min_s, max_s = np.min(scores), np.max(scores)
            probs = (scores - min_s) / (max_s - min_s) if max_s > min_s else np.full_like(scores, 0.5)
            all_probs.append(probs)
        view_weights = []
        for probs in all_probs:
            probs_clipped = np.clip(probs, 1e-8, 1 - 1e-8)
            entropy = -np.mean(probs_clipped * np.log2(probs_clipped) + (1 - probs_clipped) * np.log2(1 - probs_clipped))
            view_weights.append(1 - entropy)
        self.view_weights = np.array(view_weights)
        if np.sum(self.view_weights) > 0: self.view_weights /= np.sum(self.view_weights)
        else: self.view_weights = np.ones(len(self.view_weights)) / len(self.view_weights)
        return np.einsum('i,ij->j', self.view_weights, np.array(all_probs))

    def _estimate_single_tree_memory_gb(self, n_samples: int, n_view_features: int) -> float:
        x_view_memory = n_samples * n_view_features * 4 / (1024**3)
        similarity_matrix_memory = n_samples * n_samples * 8 / (1024**3)
        estimated_nodes = min(n_samples * 2, 1000)
        tree_structure_memory = estimated_nodes * 0.0001
        python_overhead = (x_view_memory + similarity_matrix_memory) * 0.1
        
        if n_samples < 5000:
            memory_factor = 0.5
        elif n_samples < 20000:
            memory_factor = 0.7
        else:
            memory_factor = 1.0
        
        total_memory = (x_view_memory + similarity_matrix_memory + tree_structure_memory + python_overhead) * memory_factor
        
        return max(total_memory, 0.01)
