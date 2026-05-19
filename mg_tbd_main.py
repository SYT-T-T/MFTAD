import os
import random
import time
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import torch

from mg_tbd import MVT_FRS
from mg_tbd_utils import evaluate_performance, load_and_preprocess_data

warnings.filterwarnings('ignore')

def run_with_best_delta(dataset_path: str, delta_dict: Dict):
    print("\n" + "="*80)
    print(f"🔬 Running with fixed delta=1 on: {os.path.basename(dataset_path)}")
    print("="*80)
    
    X, y = load_and_preprocess_data(dataset_path)
    if X is None or X.shape[0] == 0: 
        print("  - 数据加载失败或数据为空，跳过。")
        return None

    print(f"  - Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"  - Anomaly ratio: {y.numpy().mean():.2%}")

    ds_name = os.path.basename(dataset_path).split('.')[0]

    current_delta = 1
    print(f"  - 使用固定 delta 值: {current_delta}")

    base_config = {
        "mfgad_lower_bound": 50,
        "mfgad_upper_bound": 100,
        "random_views_if_needed": 50,
        "tree_params": {
            'cohesion_improvement_threshold': 0.008
        },
        "n_jobs": -1,
        'use_numba_acceleration': True,
        "verbose": False
    }

    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    os.environ["PYTHONHASHSEED"] = str(1)
    
    start_time = time.time()
    
    current_config = base_config.copy()
    current_config['delta'] = current_delta

    detector = MVT_FRS(**current_config)
    scores = detector.fit_predict(X)

    duration = time.time() - start_time
    metrics = evaluate_performance(y.numpy(), scores)
    auc = metrics['auc']
    
    print(f"  - AUC: {auc:.4f}")
    print(f"  - 运行时间: {duration:.2f}秒")

    roc_df = pd.DataFrame({
        'dataset_id': ds_name,
        'method_id': 'MFTAD',
        'y_true': y,
        'anomaly_score': scores,
        'score_direction_flag': 'greater'
    })

    output_dir = "result_ROC"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    roc_df.to_csv(f"{output_dir}/ROC_{ds_name}.csv", index=False)
    print(f"  - ROC数据已保存到: {output_dir}/ROC_{ds_name}.csv")
    
    return {
        'Dataset': ds_name,
        'Best_AUC': auc
    }


def main():
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    os.environ["PYTHONHASHSEED"] = str(1)

    dataset_folder = '../datasets/'
    delta_dict = {}

    best_configs_per_dataset = []

    try:
        dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith(('.mat', '.npz'))]
        if not dataset_files:
            raise FileNotFoundError(f"在 '{dataset_folder}' 文件夹中未找到数据集文件。")

        for dataset_file in dataset_files:
            full_path = os.path.join(dataset_folder, dataset_file)
            if os.path.exists(full_path):
                result = run_with_best_delta(full_path, delta_dict)
                if result:
                    best_configs_per_dataset.append(result)
            
    except Exception as e:
        import traceback
        print(f"\n程序发生意外错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
