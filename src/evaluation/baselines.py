"""
Classical link prediction heuristics (non-neural baselines).
These are essential for showing GNNs actually improve over simple methods.
"""
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix


def common_neighbors(G: nx.Graph, u: int, v: int) -> int:
    """Number of common neighbors between u and v."""
    return len(set(G.neighbors(u)) & set(G.neighbors(v)))


def jaccard_coefficient(G: nx.Graph, u: int, v: int) -> float:
    """Jaccard coefficient: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|."""
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = len(neighbors_u & neighbors_v)
    union = len(neighbors_u | neighbors_v)
    return intersection / union if union > 0 else 0.0


def adamic_adar(G: nx.Graph, u: int, v: int) -> float:
    """Adamic-Adar index: sum_{w in N(u)∩N(v)} 1/log(|N(w)|)."""
    common = set(G.neighbors(u)) & set(G.neighbors(v))
    score = 0.0
    for w in common:
        deg = G.degree(w)
        if deg > 1:
            score += 1.0 / np.log(deg)
    return score


def resource_allocation(G: nx.Graph, u: int, v: int) -> float:
    """Resource Allocation index: sum_{w in N(u)∩N(v)} 1/|N(w)|."""
    common = set(G.neighbors(u)) & set(G.neighbors(v))
    score = 0.0
    for w in common:
        deg = G.degree(w)
        if deg > 0:
            score += 1.0 / deg
    return score


def preferential_attachment(G: nx.Graph, u: int, v: int) -> float:
    """Preferential Attachment: |N(u)| * |N(v)|."""
    return G.degree(u) * G.degree(v)


def katz_index(G: nx.Graph, u: int, v: int, beta: float = 0.01, max_len: int = 3) -> float:
    """Katz index: sum_{l=1}^{max_len} beta^l * |paths of length l|."""
    try:
        # Use networkx's katz centrality approximation
        paths = nx.all_simple_paths(G, u, v, cutoff=max_len)
        score = 0.0
        for path in paths:
            l = len(path) - 1
            score += (beta ** l)
        return score
    except nx.NetworkXNoPath:
        return 0.0


def compute_heuristic_scores(
    G: nx.Graph,
    edge_index: torch.Tensor,
    heuristic: str = "common_neighbors",
) -> np.ndarray:
    """
    Compute heuristic scores for all edges in edge_index.

    Args:
        G: NetworkX graph
        edge_index: (2, E) edge tensor
        heuristic: One of 'common_neighbors', 'jaccard', 'adamic_adar',
                   'resource_allocation', 'preferential_attachment', 'katz'

    Returns:
        Scores array (E,)
    """
    heuristic_fns = {
        "common_neighbors": common_neighbors,
        "jaccard": jaccard_coefficient,
        "adamic_adar": adamic_adar,
        "resource_allocation": resource_allocation,
        "preferential_attachment": preferential_attachment,
        "katz": katz_index,
    }

    if heuristic not in heuristic_fns:
        raise ValueError(f"Unknown heuristic: {heuristic}. Available: {list(heuristic_fns.keys())}")

    fn = heuristic_fns[heuristic]
    scores = []

    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        scores.append(fn(G, u, v))

    return np.array(scores, dtype=np.float32)


def compute_all_heuristics(G: nx.Graph, edge_index: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute all heuristic scores."""
    return {
        "common_neighbors": compute_heuristic_scores(G, edge_index, "common_neighbors"),
        "jaccard": compute_heuristic_scores(G, edge_index, "jaccard"),
        "adamic_adar": compute_heuristic_scores(G, edge_index, "adamic_adar"),
        "resource_allocation": compute_heuristic_scores(G, edge_index, "resource_allocation"),
        "preferential_attachment": compute_heuristic_scores(G, edge_index, "preferential_attachment"),
    }


def evaluate_heuristics(
    data: Data,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all classical heuristics on a link prediction split.

    Args:
        data: Data object with edge_index (message passing edges)
        edge_label_index: Edges to predict
        edge_label: Ground truth labels

    Returns:
        Dict mapping heuristic name -> metrics dict
    """
    from src.evaluation.metrics import compute_all_metrics

    G = to_networkx(data, to_undirected=True)
    results = {}

    heuristics = compute_all_heuristics(G, edge_label_index)

    for name, scores in heuristics.items():
        # Normalize scores to [0, 1] for AUC/AP
        if scores.max() > scores.min():
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_norm = scores

        metrics = compute_all_metrics(scores_norm, edge_label.numpy())
        results[name] = metrics

    return results


class HeuristicPredictor:
    """
    Wrapper to make heuristics compatible with GNN evaluation pipeline.
    """

    def __init__(self, data: Data, heuristic: str = "adamic_adar"):
        self.G = to_networkx(data, to_undirected=True)
        self.heuristic = heuristic

    def predict(self, edge_label_index: torch.Tensor) -> np.ndarray:
        scores = compute_heuristic_scores(self.G, edge_label_index, self.heuristic)
        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def predict_proba(self, edge_label_index: torch.Tensor) -> np.ndarray:
        """Return probabilities for compatibility."""
        return self.predict(edge_label_index)


def run_baseline_comparison(
    train_data: Data,
    val_data: Data,
    test_data: Data,
) -> Dict[str, Dict[str, float]]:
    """
    Run all baselines on val and test sets.

    Returns:
        Dict with 'val' and 'test' keys, each mapping heuristic -> metrics
    """
    print("Running classical link prediction baselines...")

    # Build graph from training edges only (transductive setting)
    G = to_networkx(train_data, to_undirected=True)

    val_results = evaluate_heuristics(train_data, val_data.edge_label_index, val_data.edge_label)
    test_results = evaluate_heuristics(train_data, test_data.edge_label_index, test_data.edge_label)

    print("\nBaseline Results (Test Set):")
    print(f"{'Heuristic':<25} {'AUC':>8} {'AP':>8} {'P@10':>8} {'R@10':>8} {'NDCG@10':>8}")
    print("-" * 70)

    for name in ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"]:
        m = test_results[name]
        print(f"{name:<25} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['precision@10']:>8.4f} {m['recall@10']:>8.4f} {m['ndcg@10']:>8.4f}")

    return {"val": val_results, "test": test_results}


# Efficient sparse implementations for large graphs
def common_neighbors_sparse(adj: csr_matrix, u: int, v: int) -> int:
    """Sparse matrix version of common neighbors."""
    return len(set(adj[u].indices) & set(adj[v].indices))


def adamic_adar_sparse(adj: csr_matrix, u: int, v: int) -> float:
    """Sparse Adamic-Adar."""
    common = set(adj[u].indices) & set(adj[v].indices)
    score = 0.0
    for w in common:
        deg = len(adj[w].indices)
        if deg > 1:
            score += 1.0 / np.log(deg)
    return score


def resource_allocation_sparse(adj: csr_matrix, u: int, v: int) -> float:
    """Sparse Resource Allocation."""
    common = set(adj[u].indices) & set(adj[v].indices)
    score = 0.0
    for w in common:
        deg = len(adj[w].indices)
        if deg > 0:
            score += 1.0 / deg
    return score


def batch_heuristic_scores(
    adj: csr_matrix,
    edge_list: np.ndarray,
    heuristic: str = "adamic_adar",
) -> np.ndarray:
    """
    Batch compute heuristic scores using sparse matrices (faster for large graphs).

    Args:
        adj: Scipy CSR adjacency matrix
        edge_list: (E, 2) array of edges
        heuristic: Heuristic name

    Returns:
        Scores array (E,)
    """
    heuristics = {
        "common_neighbors": common_neighbors_sparse,
        "adamic_adar": adamic_adar_sparse,
        "resource_allocation": resource_allocation_sparse,
    }

    if heuristic not in heuristics:
        raise ValueError(f"Sparse version not available for {heuristic}")

    fn = heuristics[heuristic]
    scores = np.zeros(edge_list.shape[0], dtype=np.float32)

    for i, (u, v) in enumerate(edge_list):
        scores[i] = fn(adj, u, v)

    return scores