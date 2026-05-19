from typing import List, Set, Dict
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

from mg_tbd import MVT_FRS
from mg_tbd_utils import load_and_preprocess_data, evaluate_performance

warnings.filterwarnings('ignore')

class MVT_FRS_Ablation(MVT_FRS):
    def __init__(self, ablation_flags: dict, **kwargs):
        super().__init__(**kwargs)
        self.ablation_flags = ablation_flags

    def fit_predict(self, X):
        self._ablation_X = X.numpy() if torch.is_tensor(X) else X
        return super().fit_predict(X)

    def _generate_adaptive_views(self) -> List[Set[int]]:
        if not self.ablation_flags['multi_view']:
            views = [set(range(self.n_features))]
        else:
            views = super()._generate_adaptive_views()
        self._ablation_views = views
        return views

    def _compute_and_fuse_scores_turbo(self) -> np.ndarray:
        if not self.forest:
            return np.full(self.n_samples, 0.5)

        if not self.ablation_flags['use_tree']:
            all_view_scores = []
            for view_features in self._ablation_views:
                view_cols = list(view_features)
                X_view = self._ablation_X[:, view_cols]
                global_center = np.mean(X_view, axis=0)
                distances = np.linalg.norm(X_view - global_center, axis=1)
                all_view_scores.append(distances)

            fused = np.mean(all_view_scores, axis=0)
            min_s, max_s = np.min(fused), np.max(fused)
            if max_s > min_s:
                fused = (fused - min_s) / (max_s - min_s)
            return fused

        all_view_scores = []
        for builder in self.forest:
            scores_to_stack = []

            if self.ablation_flags.get('use_leaf', False):
                s_leaf = self._compute_leaf_scores_turbo(builder) if self.use_numba_acceleration else self._compute_all_leaf_scores_vectorized(builder)
                scores_to_stack.append(s_leaf)

            if self.ablation_flags.get('use_gta', False):
                s_gta = self._compute_gta_scores_turbo(builder) if self.use_numba_acceleration else self._compute_all_gta_scores_vectorized(builder)
                scores_to_stack.append(s_gta)

            if self.ablation_flags.get('use_path', False):
                s_path = self._compute_path_scores_turbo(builder) if self.use_numba_acceleration else self._compute_all_path_scores_vectorized(builder)
                scores_to_stack.append(s_path)

            all_view_scores.append(np.stack(scores_to_stack, axis=1))

        num_dims = len(all_view_scores[0][0])
        fused_scores_nd = np.zeros((self.n_samples, num_dims))
        for dim in range(num_dims):
            scores_for_dim = [s[:, dim] for s in all_view_scores]
            fused_scores_nd[:, dim] = self._fuse_single_dimension_scores(scores_for_dim)

        if self.ablation_flags.get('dynamic_weight', False) and num_dims > 1:
            certainty = np.abs(fused_scores_nd - 0.5) * 2
            total_certainty = np.sum(certainty, axis=1, keepdims=True)
            total_certainty[total_certainty < 1e-8] = 1.0
            dynamic_weights = certainty / total_certainty
            final_scores = np.sum(fused_scores_nd * dynamic_weights, axis=1)
        else:
            final_scores = fused_scores_nd[:, 0] if num_dims == 1 else np.mean(fused_scores_nd, axis=1)

        min_s, max_s = np.min(final_scores), np.max(final_scores)
        if max_s > min_s:
            final_scores = (final_scores - min_s) / (max_s - min_s)

        return final_scores


MACRO_CONFIGS = {
    'w/o MV':      {'multi_view': False, 'use_tree': True,  'use_leaf': True,  'use_gta': True,  'use_path': True,  'dynamic_weight': True},
    'w/o Tree':    {'multi_view': True,  'use_tree': False, 'use_leaf': False, 'use_gta': False, 'use_path': False, 'dynamic_weight': False},
    'w/o Dynamic': {'multi_view': True,  'use_tree': True,  'use_leaf': True,  'use_gta': True,  'use_path': True,  'dynamic_weight': False},
    'MFTAD(Ours)': {'multi_view': True,  'use_tree': True,  'use_leaf': True,  'use_gta': True,  'use_path': True,  'dynamic_weight': True}
}

MICRO_CONFIGS = {
    'Only Path':   {'multi_view': True, 'use_tree': True, 'use_leaf': False, 'use_gta': False, 'use_path': True,  'dynamic_weight': False},
    'Only Leaf':   {'multi_view': True, 'use_tree': True, 'use_leaf': True,  'use_gta': False, 'use_path': False, 'dynamic_weight': False},
    'Only GTA':    {'multi_view': True, 'use_tree': True, 'use_leaf': False, 'use_gta': True,  'use_path': False, 'dynamic_weight': False},
    'Leaf + GTA':  {'multi_view': True, 'use_tree': True, 'use_leaf': True,  'use_gta': True,  'use_path': False, 'dynamic_weight': False},
    'MFTAD(All)':  {'multi_view': True, 'use_tree': True, 'use_leaf': True,  'use_gta': True,  'use_path': True,  'dynamic_weight': True}
}

def _print_table(title: str, res_df: pd.DataFrame, headers: List[tuple]):
    print("\n" + "="*90)
    print(f"📄 {title}")
    print("="*90)

    for comp_name, marks in headers:
        row_str = f"{comp_name:<20} |"
        for m in marks:
            row_str += f"      {m}      |"
        print(row_str)
    print("-" * (20 + 3 + 14 * len(res_df.columns)))

    row_str = f"{'Dataset':<20} |"
    for c in res_df.columns:
        row_str += f" {c:^11} |"
    print(row_str)
    print("-" * (20 + 3 + 14 * len(res_df.columns)))

    for index, row in res_df.iterrows():
        row_str = f"{index:<20} |"
        for val in row:
            row_str += f"   {val:.4f}   |" if pd.notna(val) else "    N/A    |"
        print(row_str)

def _run_experiment(configs: Dict[str, dict], dataset_files, delta_dict, base_config, title, headers, out_csv):
    results = []

    for d_idx, dataset_file in enumerate(dataset_files):
        ds_name = dataset_file.split('.')[0]
        full_path = os.path.join('../datasets/', dataset_file)

        X, y = load_and_preprocess_data(full_path)
        if X is None or X.shape[0] == 0:
            continue

        delta = delta_dict.get(ds_name, 1.0)
        row_result = {'Dataset': ds_name}
        print(f"\n[{d_idx+1}/{len(dataset_files)}] 测试数据集: {ds_name} (Using delta: {delta:.2f})")

        for model_name, flags in configs.items():
            np.random.seed(1)
            torch.manual_seed(1)
            random.seed(1)
            os.environ["PYTHONHASHSEED"] = str(1)

            try:
                detector = MVT_FRS_Ablation(ablation_flags=flags, delta=delta, **base_config)
                scores = detector.fit_predict(X)
                auc = evaluate_performance(y.numpy(), scores)['auc']
                print(f"   ► {model_name:<12}: AUC = {auc:.4f}")
                row_result[model_name] = auc
            except Exception as e:
                print(f"   ► {model_name:<12}: 运行失败 ({str(e)[:60]})")
                row_result[model_name] = np.nan

        results.append(row_result)

    if results:
        res_df = pd.DataFrame(results).set_index('Dataset')
        res_df.loc['Average'] = res_df.mean()
        _print_table(title, res_df, headers)
        res_df.to_csv(out_csv)
        print(f"\n💾 结果已保存至: {out_csv}")

def main():
    dataset_folder = '../datasets/'
    delta_file = '../best_delta.csv'

    delta_dict = {}
    if os.path.exists(delta_file):
        try:
            df = pd.read_csv(delta_file)
            delta_dict = dict(zip(df['Dataset'], df['delta']))
        except:
            pass

    dataset_files = sorted([f for f in os.listdir(dataset_folder) if f.endswith(('.mat', '.npz'))])

    base_config = {
        "mfgad_lower_bound": 50,
        "mfgad_upper_bound": 100,
        "random_views_if_needed": 50,
        "tree_params": {'cohesion_improvement_threshold': 0.008},
        "n_jobs": -1,
        'use_numba_acceleration': True,
        "verbose": False
    }

    macro_headers = [
        ("Multi-View (MV)",   ['✗', '✓', '✓', '✓']),
        ("Fuzzy Tree",        ['✓', '✗', '✓', '✓']),
        ("Dynamic Weight",    ['✓', '✗', '✗', '✓'])
    ]
    print("\n" + "="*80)
    print("🚀 表格1：框架级消融实验 (Macro)")
    print("="*80)
    _run_experiment(MACRO_CONFIGS, dataset_files, delta_dict, base_config,
                    "表格1：框架级消融实验 (Macro)", macro_headers, "ablation_macro_results.csv")

    micro_headers = [
        ("Leaf Score", ['✗', '✓', '✗', '✓', '✓']),
        ("GTA Score",  ['✗', '✗', '✓', '✓', '✓']),
        ("Path Score", ['✓', '✗', '✗', '✗', '✓'])
    ]
    print("\n" + "="*80)
    print("🚀 表格2：三维打分机制消融 (Micro)")
    print("="*80)
    _run_experiment(MICRO_CONFIGS, dataset_files, delta_dict, base_config,
                    "表格2：三维打分机制消融 (Micro)", micro_headers, "ablation_micro_results.csv")

if __name__ == "__main__":
    main()
