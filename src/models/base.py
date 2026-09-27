"""
Base classes for GNN models.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseGNNEncoder(nn.Module, ABC):
    """
    Abstract base class for GNN encoders.

    All encoders should implement forward(x, edge_index) -> node_embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_residual: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)

        Returns:
            Node embeddings (N, out_channels)
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for logging/checkpointing."""
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_residual": self.use_residual,
            "use_layer_norm": self.use_layer_norm,
        }


class BaseLinkPredictor(nn.Module, ABC):
    """
    Abstract base class for link prediction heads.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score edges.

        Args:
            z: Node embeddings (N, d)
            edge_label_index: Edges to score (2, M)

        Returns:
            Scores (M,)
        """
        pass


class GNNModel(nn.Module):
    """
    Complete GNN model combining encoder + predictor.
    """

    def __init__(
        self,
        encoder: BaseGNNEncoder,
        predictor: BaseLinkPredictor,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass: encode -> predict.

        Returns:
            Edge scores (M,)
        """
        z = self.encoder(x, edge_index)
        return self.predictor(z, edge_label_index)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes to embeddings."""
        return self.encoder(x, edge_index)

    def predict(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge scores from embeddings."""
        return self.predictor(z, edge_label_index)

    def recommend(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        user_id: int,
        top_k: int = 10,
        exclude_existing: bool = True,
    ) -> torch.Tensor:
        """
        Generate top-K recommendations for a user.

        Args:
            x: Node features
            edge_index: Graph edges (for message passing)
            user_id: Target user index
            top_k: Number of recommendations
            exclude_existing: Whether to filter out existing friends

        Returns:
            (top_k,) tensor of recommended node indices
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x, edge_index)
            user_emb = z[user_id]

            # Score all nodes
            scores = self.predictor.score_all(z, user_emb)  # (N,)

            if exclude_existing:
                # Mask existing neighbors
                mask = edge_index[0] == user_id
                existing = edge_index[1][mask]
                scores[existing] = -float("inf")
                scores[user_id] = -float("inf")

            top_scores, top_indices = torch.topk(scores, k=top_k)
            return top_indices


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}