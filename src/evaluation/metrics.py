"""
Comprehensive evaluation metrics for link prediction and recommendation.
"""
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    ndcg_score,
)
from typing import Callable, Dict, List, Optional, Tuple


def compute_auc_ap(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute AUC-ROC and Average Precision."""
    return {
        "auc": roc_auc_score(labels, probs),
        "ap": average_precision_score(labels, probs),
    }


def compute_ranking_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    ks: List[int] = [5, 10, 20, 50],
) -> Dict[str, float]:
    """
    Compute ranking-based metrics (Precision@K, Recall@K, NDCG@K, Hit@K).

    Note: This assumes binary classification setting.
    For recommendation setting with multiple positives per user,
    use compute_recommendation_metrics instead.
    """
    metrics = {}

    # Sort by predicted probability descending
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]

    for k in ks:
        k = min(k, len(labels))
        top_k_labels = sorted_labels[:k]

        # Precision@K
        precision = top_k_labels.sum() / k
        metrics[f"precision@{k}"] = precision

        # Recall@K
        total_positives = labels.sum()
        recall = top_k_labels.sum() / total_positives if total_positives > 0 else 0.0
        metrics[f"recall@{k}"] = recall

        # Hit@K (at least one positive in top-k)
        hit = 1.0 if top_k_labels.sum() > 0 else 0.0
        metrics[f"hit@{k}"] = hit

        # NDCG@K
        dcg = np.sum(top_k_labels / np.log2(np.arange(2, k + 2)))
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = np.sum(ideal_labels / np.log2(np.arange(2, k + 2)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"ndcg@{k}"] = ndcg

    return metrics


def compute_recommendation_metrics(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    test_interactions: Dict[int, List[int]],
    ks: List[int] = [5, 10, 20, 50],
) -> Dict[str, float]:
    """
    Compute recommendation metrics for user-item setting.

    Args:
        user_embeddings: (n_users, d)
        item_embeddings: (n_items, d)
        test_interactions: Dict mapping user_id -> list of test item_ids
        ks: List of K values

    Returns:
        Dict of metrics (mean across users)
    """
    n_users = user_embeddings.shape[0]
    n_items = item_embeddings.shape[0]

    all_precision = {k: [] for k in ks}
    all_recall = {k: [] for k in ks}
    all_ndcg = {k: [] for k in ks}
    all_hit = {k: [] for k in ks}

    # Compute all scores
    scores = user_embeddings @ item_embeddings.T  # (n_users, n_items)

    for u in range(n_users):
        test_items = test_interactions.get(u, [])
        if not test_items:
            continue

        # Rank items for this user
        user_scores = scores[u]
        ranked_items = np.argsort(-user_scores)

        for k in ks:
            k = min(k, n_items)
            top_k = ranked_items[:k]
            hits = np.isin(top_k, test_items).astype(float)

            # Precision@K
            all_precision[k].append(hits.sum() / k)

            # Recall@K
            all_recall[k].append(hits.sum() / len(test_items))

            # Hit@K
            all_hit[k].append(1.0 if hits.sum() > 0 else 0.0)

            # NDCG@K
            dcg = np.sum(hits / np.log2(np.arange(2, k + 2)))
            ideal_hits = np.ones(min(len(test_items), k))
            idcg = np.sum(ideal_hits / np.log2(np.arange(2, min(len(test_items), k) + 2)))
            all_ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {}
    for k in ks:
        metrics[f"precision@{k}"] = np.mean(all_precision[k]) if all_precision[k] else 0.0
        metrics[f"recall@{k}"] = np.mean(all_recall[k]) if all_recall[k] else 0.0
        metrics[f"hit@{k}"] = np.mean(all_hit[k]) if all_hit[k] else 0.0
        metrics[f"ndcg@{k}"] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0

    return metrics


def compute_mrr(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean Reciprocal Rank."""
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]

    # Find first positive
    pos_indices = np.where(sorted_labels == 1)[0]
    if len(pos_indices) == 0:
        return 0.0

    first_pos_rank = pos_indices[0] + 1  # 1-indexed
    return 1.0 / first_pos_rank


def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    ks: List[int] = [5, 10, 20, 50],
) -> Dict[str, float]:
    """
    Compute all available metrics.

    Args:
        probs: Predicted probabilities (N,)
        labels: Ground truth labels (N,)
        ks: K values for ranking metrics

    Returns:
        Dict of all metrics
    """
    metrics = {}

    # AUC/AP
    metrics.update(compute_auc_ap(probs, labels))

    # Ranking metrics
    metrics.update(compute_ranking_metrics(probs, labels, ks))

    # MRR
    metrics["mrr"] = compute_mrr(probs, labels)

    return metrics


def compute_all_metrics_torch(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ks: List[int] = [5, 10, 20, 50],
) -> Dict[str, float]:
    """Torch tensor version."""
    return compute_all_metrics(probs.cpu().numpy(), labels.cpu().numpy(), ks)


class MetricTracker:
    """Track metrics across epochs."""

    def __init__(self):
        self.history = {}

    def update(self, metrics: Dict[str, float], epoch: int):
        for k, v in metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append((epoch, v))

    def get_best(self, metric: str, mode: str = "max") -> Tuple[int, float]:
        """Get best epoch and value for a metric."""
        if metric not in self.history:
            return -1, None

        values = self.history[metric]
        if mode == "max":
            best = max(values, key=lambda x: x[1])
        else:
            best = min(values, key=lambda x: x[1])
        return best

    def get_last(self, metric: str) -> float:
        """Get last value for a metric."""
        if metric in self.history and self.history[metric]:
            return self.history[metric][-1][1]
        return None


def bootstrap_confidence_interval(
    probs: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Returns:
        (mean, lower_ci, upper_ci)
    """
    n = len(labels)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        score = metric_fn(probs[idx], labels[idx])
        bootstrap_scores.append(score)

    mean_score = np.mean(bootstrap_scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

    return mean_score, lower, upper


# For backward compatibility
def precision_at_k(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Precision at K."""
    sorted_indices = np.argsort(-probs)
    top_k = sorted_indices[:k]
    return labels[top_k].sum() / k


def recall_at_k(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Recall at K."""
    sorted_indices = np.argsort(-probs)
    top_k = sorted_indices[:k]
    total_pos = labels.sum()
    return labels[top_k].sum() / total_pos if total_pos > 0 else 0.0


def ndcg_at_k(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """NDCG at K."""
    sorted_indices = np.argsort(-probs)
    top_k = sorted_indices[:k]
    dcg = np.sum(labels[top_k] / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(labels)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, min(len(ideal), k) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Hit Rate at K."""
    sorted_indices = np.argsort(-probs)
    top_k = sorted_indices[:k]
    return 1.0 if labels[top_k].sum() > 0 else 0.0