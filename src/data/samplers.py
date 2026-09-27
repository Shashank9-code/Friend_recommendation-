"""
Negative sampling and edge sampling utilities for link prediction.
"""
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, to_undirected
from typing import Tuple, Optional
import numpy as np


def uniform_negative_sampling(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    exclude_edges: Optional[torch.Tensor] = None,
    method: str = "sparse",
) -> torch.Tensor:
    """
    Uniform negative sampling for link prediction.

    Args:
        edge_index: Positive edge index (2, E)
        num_nodes: Number of nodes in graph
        num_neg_samples: Number of negative samples to generate
        exclude_edges: Additional edges to exclude from sampling
        method: 'sparse' (default) or 'dense'

    Returns:
        Negative edge index (2, num_neg_samples)
    """
    # Combine edges to exclude
    if exclude_edges is not None:
        all_edges = torch.cat([edge_index, exclude_edges], dim=1)
    else:
        all_edges = edge_index

    # PyG's negative_sampling handles exclusion
    neg_edge_index = negative_sampling(
        edge_index=all_edges,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method=method,
    )
    return neg_edge_index


def structured_negative_sampling(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    max_trials: int = 10,
) -> torch.Tensor:
    """
    Structured negative sampling that avoids high-degree nodes
    to prevent bias toward popular nodes.
    """
    # Compute degree-based sampling probabilities
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.float))

    # Inverse degree weighting (lower prob for high-degree nodes)
    prob = 1.0 / (deg + 1.0)
    prob = prob / prob.sum()

    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        # Sample source and destination independently
        src = torch.multinomial(prob, num_neg_samples * 2, replacement=True)
        dst = torch.multinomial(prob, num_neg_samples * 2, replacement=True)

        # Filter self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        # Filter existing edges
        if len(src) > 0:
            candidate_edges = torch.stack([src, dst], dim=0)
            # Check against existing edges (could optimize with hash set for large graphs)
            existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            is_new = torch.tensor([(int(s), int(d)) not in existing for s, d in zip(src, dst)])
            new_edges = candidate_edges[:, is_new]
            neg_edges.append(new_edges)

        if len(neg_edges) >= num_neg_samples:
            break

    neg_edge_index = torch.cat(neg_edges, dim=1)[:, :num_neg_samples]
    return neg_edge_index


def dynamic_edge_masking(
    edge_index: torch.Tensor,
    mask_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask a fraction of edges for self-supervised pretraining
    (like GraphMAE, MaskGAE).

    Returns:
        (masked_edge_index, masked_edges)
    """
    if seed is not None:
        torch.manual_seed(seed)

    num_edges = edge_index.size(1)
    num_mask = int(num_edges * mask_ratio)

    perm = torch.randperm(num_edges)
    mask_idx = perm[:num_mask]
    keep_idx = perm[num_mask:]

    masked_edges = edge_index[:, mask_idx]
    masked_edge_index = edge_index[:, keep_idx]

    return masked_edge_index, masked_edges


def bpr_triplet_sampling(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    neg_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate (user, pos_item, neg_item) triplets for BPR loss.
    Assumes bipartite user-item graph or treats all nodes uniformly.
    """
    num_pos = pos_edge_index.size(1)
    num_neg = int(num_pos * neg_ratio)

    # Sample negative destinations
    neg_dst = torch.randint(0, num_nodes, (num_neg,), dtype=torch.long)

    # Repeat positive edges to match negative count if needed
    if num_neg > num_pos:
        repeat = (num_neg + num_pos - 1) // num_pos
        pos_edge_index = pos_edge_index.repeat(1, repeat)[:, :num_neg]

    src = pos_edge_index[0, :num_neg]
    pos_dst = pos_edge_index[1, :num_neg]

    return src, pos_dst, neg_dst


class NegativeSampler:
    """
    Configurable negative sampler for training.
    """

    def __init__(
        self,
        method: str = "uniform",  # uniform, structured
        neg_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.method = method
        self.neg_ratio = neg_ratio
        self.seed = seed

    def __call__(self, data: Data) -> Data:
        """Add negative edges to data for training."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        num_pos = pos_edge_index.size(1)
        num_neg = int(num_pos * self.neg_ratio)

        if self.method == "uniform":
            neg_edge_index = uniform_negative_sampling(
                pos_edge_index, data.num_nodes, num_neg
            )
        elif self.method == "structured":
            neg_edge_index = structured_negative_sampling(
                pos_edge_index, data.num_nodes, num_neg
            )
        else:
            raise ValueError(f"Unknown negative sampling method: {self.method}")

        # Create labels
        neg_labels = torch.zeros(num_neg, dtype=torch.float)
        pos_labels = data.edge_label[data.edge_label == 1]

        # Combine
        new_edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        new_edge_label = torch.cat([pos_labels, neg_labels], dim=0)

        # Shuffle
        perm = torch.randperm(new_edge_label.size(0))
        data.edge_label_index = new_edge_label_index[:, perm]
        data.edge_label = new_edge_label[perm]

        return data


class NeighborSampler:
    """
    Mini-batch neighbor sampling for scalable GNN training (GraphSAGE-style).
    """

    def __init__(
        self,
        sizes: list,  # e.g., [15, 10, 5] for 3-layer
        batch_size: int = 512,
        shuffle: bool = True,
    ):
        self.sizes = sizes
        self.batch_size = batch_size
        self.shuffle = shuffle

    def sample(self, edge_index: torch.Tensor, node_idx: torch.Tensor) -> list:
        """
        Sample neighbors for each layer.

        Args:
            edge_index: (2, E) graph connectivity
            node_idx: (B,) target node indices

        Returns:
            List of (batch_size, num_sampled) neighbor indices per layer
        """
        from torch_geometric.utils import k_hop_subgraph
        # Simplified: use PyG's NeighborSampler in production
        # This is a placeholder for the interface
        raise NotImplementedError("Use torch_geometric.loader.NeighborSampler for production")