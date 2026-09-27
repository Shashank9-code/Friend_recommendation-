"""
Structural feature engineering for graph neural networks.
Supports: Identity, Degree, PageRank, RWPE, Laplacian PE, and learnable embeddings.
"""
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dense_adj
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from typing import List, Optional


def compute_degree_features(data: Data, normalize: bool = True) -> torch.Tensor:
    """Compute normalized node degree features."""
    G = to_networkx(data, to_undirected=True)
    degree_dict = dict(G.degree())
    degree_vals = np.array([degree_dict.get(i, 0) for i in range(data.num_nodes)], dtype=np.float32)

    if normalize:
        max_deg = degree_vals.max() if degree_vals.max() > 0 else 1.0
        degree_vals = degree_vals / max_deg

    return torch.tensor(degree_vals, dtype=torch.float).unsqueeze(1)


def compute_pagerank_features(data: Data, alpha: float = 0.85, normalize: bool = True) -> torch.Tensor:
    """Compute PageRank centrality features."""
    G = to_networkx(data, to_undirected=True)
    pr_dict = nx.pagerank(G, alpha=alpha)
    pr_vals = np.array([pr_dict.get(i, 0.0) for i in range(data.num_nodes)], dtype=np.float32)

    if normalize:
        max_pr = pr_vals.max() if pr_vals.max() > 0 else 1.0
        pr_vals = pr_vals / max_pr

    return torch.tensor(pr_vals, dtype=torch.float).unsqueeze(1)


def compute_rwpe(data: Data, walk_length: int = 16) -> torch.Tensor:
    """
    Random Walk Positional Encoding (RWPE).
    Computes the diagonal of powers of the random walk transition matrix.
    """
    # Build transition matrix P = D^{-1} A
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Degree vector
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.float))
    deg = deg.clamp(min=1.0)

    # Random walk: P = D^{-1} A
    # We compute diagonal of P^k for k=1..walk_length
    pe_list = []
    adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)  # (N, N)

    # Normalize rows to get transition matrix
    row_sum = adj_dense.sum(dim=1, keepdim=True).clamp(min=1.0)
    P = adj_dense / row_sum

    P_k = P.clone()
    for k in range(walk_length):
        pe_list.append(torch.diag(P_k).unsqueeze(1))
        P_k = P_k @ P

    rwpe = torch.cat(pe_list, dim=1)  # (N, walk_length)
    return rwpe


def compute_laplacian_pe(data: Data, k: int = 16) -> torch.Tensor:
    """
    Laplacian Positional Encoding using eigenvectors of normalized Laplacian.
    Returns first k non-trivial eigenvectors.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Build adjacency
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).numpy()

    # Degree matrix
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = np.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt

    # Compute k smallest eigenvalues/vectors (skip first trivial eigenvalue ~0)
    try:
        eigvals, eigvecs = eigsh(L, k=k+1, which="SM", maxiter=10000)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        # Skip first eigenvector (constant)
        pe = eigvecs[:, 1:k+1]
    except Exception:
        # Fallback: random initialization if eigensolver fails
        pe = np.random.randn(num_nodes, k).astype(np.float32)

    return torch.tensor(pe, dtype=torch.float)


def compute_identity_features(data: Data) -> torch.Tensor:
    """One-hot identity features (not scalable for large graphs)."""
    return torch.eye(data.num_nodes, dtype=torch.float)


def compute_learnable_embeddings(num_nodes: int, dim: int, init_std: float = 0.1) -> torch.nn.Parameter:
    """Learnable node embeddings (for inductive settings or large graphs)."""
    emb = torch.nn.Parameter(torch.randn(num_nodes, dim) * init_std)
    return emb


class FeatureEngineer:
    """
    Composes multiple structural features into a single feature matrix.

    Supported features:
    - identity: One-hot (N x N) - use only for small graphs
    - degree: Normalized degree (N x 1)
    - pagerank: Normalized PageRank (N x 1)
    - rwpe: Random Walk PE (N x walk_length)
    - laplacian_pe: Laplacian eigenvectors (N x k)
    """

    def __init__(
        self,
        features: List[str] = None,
        rwpe_length: int = 16,
        laplacian_k: int = 16,
    ):
        """
        Args:
            features: List of feature names to include. Default: ["degree", "pagerank"]
            rwpe_length: Walk length for RWPE
            laplacian_k: Number of eigenvectors for Laplacian PE
        """
        if features is None:
            features = ["degree", "pagerank"]

        self.features = features
        self.rwpe_length = rwpe_length
        self.laplacian_k = laplacian_k

        # Validate features
        valid_features = {"identity", "degree", "pagerank", "rwpe", "laplacian_pe"}
        invalid = set(features) - valid_features
        if invalid:
            raise ValueError(f"Unknown features: {invalid}. Valid: {valid_features}")

    def __call__(self, data: Data) -> Data:
        """Apply feature engineering to data object."""
        feature_tensors = []

        if "identity" in self.features:
            print("  Computing identity features...")
            feature_tensors.append(compute_identity_features(data))

        if "degree" in self.features:
            print("  Computing degree features...")
            feature_tensors.append(compute_degree_features(data))

        if "pagerank" in self.features:
            print("  Computing PageRank features...")
            feature_tensors.append(compute_pagerank_features(data))

        if "rwpe" in self.features:
            print(f"  Computing RWPE (walk_length={self.rwpe_length})...")
            feature_tensors.append(compute_rwpe(data, self.rwpe_length))

        if "laplacian_pe" in self.features:
            print(f"  Computing Laplacian PE (k={self.laplacian_k})...")
            feature_tensors.append(compute_laplacian_pe(data, self.laplacian_k))

        if not feature_tensors:
            raise ValueError("No features selected!")

        data.x = torch.cat(feature_tensors, dim=1)
        print(f"  Final feature dim: {data.x.size(1)}")
        return data

    def get_feature_dim(self, num_nodes: int) -> int:
        """Calculate total feature dimension without computing features."""
        dim = 0
        if "identity" in self.features:
            dim += num_nodes
        if "degree" in self.features:
            dim += 1
        if "pagerank" in self.features:
            dim += 1
        if "rwpe" in self.features:
            dim += self.rwpe_length
        if "laplacian_pe" in self.features:
            dim += self.laplacian_k
        return dim