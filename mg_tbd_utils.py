import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    from units import load_data

    try:
        X, y = load_data(file_path)
    except Exception as e:
        print(f"Error in load_data for {file_path}: {e}")
        return torch.empty(0, 0), torch.empty(0)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    if X.shape[0] > 0:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.numpy())
        X = torch.from_numpy(X_scaled).float()

    return X, y

def evaluate_performance(y_true: np.ndarray, scores: np.ndarray, anomaly_ratio: float = None) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        warnings.warn("真实标签中只存在一种类别，无法计算AUC等指标。")
        return {'auc': 0.5, 'precision': 0, 'recall': 0, 'f1': 0}
        
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        return {'auc': 0.5, 'precision': 0, 'recall': 0, 'f1': 0}
    
    if anomaly_ratio is None:
        anomaly_ratio = np.sum(y_true) / len(y_true)
        
    if anomaly_ratio == 0 or np.sum(y_true) == 0:
        return {'auc': auc, 'precision': 0, 'recall': 0, 'f1': 0}
        
    threshold = np.percentile(scores, 100 * (1 - anomaly_ratio))
    y_pred = (scores >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    return {'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}

def visualize_forest_analysis(mvt_frs_detector, scores: np.ndarray, config_name: str, save_path: str = None):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    fig.suptitle(f'MVT-FRS Forest Analysis - Config: "{config_name}"', fontsize=20)

    if mvt_frs_detector.view_weights is not None and len(mvt_frs_detector.view_weights) > 0:
        weights = mvt_frs_detector.view_weights
        sorted_weights = sorted(weights, reverse=True)
        ax1.bar(range(len(sorted_weights)), sorted_weights, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of View (Tree) Weights')
        ax1.set_xlabel('View Index (Sorted by Weight)')
        ax1.set_ylabel('Adaptive Weight (1 - Entropy)')
        ax1.grid(True, linestyle='--', alpha=0.6)
    else:
        ax1.text(0.5, 0.5, 'No view weights available.', ha='center', va='center')
        ax1.set_title('Distribution of View (Tree) Weights')

    if mvt_frs_detector.forest:
        tree_sizes = [len(builder.tree) for builder in mvt_frs_detector.forest]
        ax2.hist(tree_sizes, bins=20, color='mediumseagreen', alpha=0.8, edgecolor='black')
        ax2.set_title('Distribution of Tree Sizes in Forest')
        ax2.set_xlabel('Number of Nodes in Tree')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, 'No forest built.', ha='center', va='center')
        ax2.set_title('Distribution of Tree Sizes in Forest')

    ax3.hist(scores, bins=50, color='salmon', alpha=0.8, edgecolor='black')
    ax3.set_title('Final Fused Anomaly Score Distribution')
    ax3.set_xlabel('Anomaly Score')
    ax3.set_ylabel('Frequency')
    ax3.set_yscale('log')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
