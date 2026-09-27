"""
Dataset loading utilities for SNAP ego-networks and OGB link prediction benchmarks.
"""
import os
from typing import Optional, List, Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import SNAPDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected


class EgoNetworkDataset:
    """
    Wrapper for SNAP ego-network datasets (Facebook, Twitter, Gplus).
    """

    AVAILABLE_DATASETS = {
        "facebook": "ego-Facebook",
        "twitter": "ego-Twitter",
        "gplus": "ego-Gplus",
    }

    def __init__(
        self,
        root: str = "./data",
        name: str = "facebook",
        val_ratio: float = 0.05,
        test_ratio: float = 0.10,
        neg_sampling_ratio: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            root: Root directory for dataset storage
            name: Dataset name (facebook, twitter, gplus)
            val_ratio: Fraction of edges for validation
            test_ratio: Fraction of edges for testing
            neg_sampling_ratio: Negative to positive edge ratio
            seed: Random seed for reproducibility
        """
        if name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.AVAILABLE_DATASETS.keys())}")

        self.root = root
        self.name = name
        self.snap_name = self.AVAILABLE_DATASETS[name]
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.seed = seed

        self._data: Optional[Data] = None
        self._splits: Optional[Tuple[Data, Data, Data]] = None

    def load(self) -> Data:
        """Load the raw ego-network graph."""
        if self._data is not None:
            return self._data

        print(f"Loading SNAP {self.snap_name} dataset...")
        dataset = SNAPDataset(root=self.root, name=self.snap_name)
        self._data = dataset[0]

        # Ensure undirected
        self._data.edge_index = to_undirected(self._data.edge_index)

        print(f"  Nodes: {self._data.num_nodes:,}")
        print(f"  Edges: {self._data.num_edges:,} (undirected)")
        return self._data

    def get_splits(self) -> Tuple[Data, Data, Data]:
        """Get train/val/test splits with negative sampling."""
        if self._splits is not None:
            return self._splits

        data = self.load()

        splitter = RandomLinkSplit(
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=self.neg_sampling_ratio,
            split_labels=False,
        )

        # Set seed for reproducible splits
        torch.manual_seed(self.seed)
        self._splits = splitter(data)

        for name, split in [("Train", self._splits[0]), ("Val", self._splits[1]), ("Test", self._splits[2])]:
            num_pos = int(split.edge_label.sum().item())
            num_neg = len(split.edge_label) - num_pos
            print(f"  {name:5s} | message edges: {split.edge_index.size(1):>7,} | pos: {num_pos:>6,} neg: {num_neg:>6,}")

        return self._splits

    @property
    def num_nodes(self) -> int:
        return self.load().num_nodes

    @property
    def num_edges(self) -> int:
        return self.load().num_edges


class OGBLinkPredictionDataset:
    """
    Wrapper for OGB link prediction datasets (ogbl-collab, ogbl-ppa, ogbl-ddi).
    Requires ogb package: pip install ogb
    """

    AVAILABLE_DATASETS = ["ogbl-collab", "ogbl-ppa", "ogbl-ddi", "ogbl-citation2"]

    def __init__(
        self,
        root: str = "./data",
        name: str = "ogbl-collab",
        seed: int = 42,
    ):
        if name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown OGB dataset: {name}. Available: {self.AVAILABLE_DATASETS}")

        self.root = root
        self.name = name
        self.seed = seed
        self._dataset = None
        self._splits = None

    def load(self):
        """Load OGB dataset."""
        if self._dataset is not None:
            return self._dataset

        try:
            from ogb.linkproppred import LinkPropPredDataset
        except ImportError:
            raise ImportError("ogb package required: pip install ogb")

        print(f"Loading OGB {self.name} dataset...")
        self._dataset = LinkPropPredDataset(name=self.name, root=self.root)
        return self._dataset

    def get_splits(self):
        """Get train/val/test splits from OGB evaluator."""
        if self._splits is not None:
            return self._splits

        dataset = self.load()
        split_edge = dataset.get_edge_split()
        self._splits = split_edge

        print(f"  Train edges: {split_edge['train']['edge'].size(0):,}")
        print(f"  Val edges:   {split_edge['valid']['edge'].size(0):,} (pos) | {split_edge['valid']['edge_neg'].size(0):,} (neg)")
        print(f"  Test edges:  {split_edge['test']['edge'].size(0):,} (pos) | {split_edge['test']['edge_neg'].size(0):,} (neg)")

        return self._splits

    def to_pyg_data(self) -> Data:
        """Convert to PyG Data object."""
        dataset = self.load()
        data = dataset[0]
        return data


def load_dataset(config: dict) -> Tuple[Data, Tuple[Data, Data, Data]]:
    """
    Factory function to load dataset based on config.

    Args:
        config: Dict with keys: dataset.name, dataset.root, dataset.val_ratio, etc.

    Returns:
        (full_data, (train_data, val_data, test_data))
    """
    ds_config = config.get("dataset", {})
    name = ds_config.get("name", "facebook").lower()
    root = ds_config.get("root", "./data")
    val_ratio = ds_config.get("val_ratio", 0.05)
    test_ratio = ds_config.get("test_ratio", 0.10)
    neg_ratio = ds_config.get("neg_sampling_ratio", 1.0)
    seed = ds_config.get("seed", 42)

    if name.startswith("ogbl-"):
        ds = OGBLinkPredictionDataset(root=root, name=name, seed=seed)
        full_data = ds.to_pyg_data()
        splits = ds.get_splits()
        # OGB returns dict, convert to tuple format for compatibility
        return full_data, splits
    else:
        ds = EgoNetworkDataset(
            root=root,
            name=name,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            neg_sampling_ratio=neg_ratio,
            seed=seed,
        )
        full_data = ds.load()
        splits = ds.get_splits()
        return full_data, splits