"""
Link prediction heads: Dot-product, Cosine, MLP, Bilinear.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseLinkPredictor


class DotProductPredictor(BaseLinkPredictor):
    """
    Dot-product decoder: score(u, v) = z_u^T z_v
    Most common for link prediction with GNNs.
    """

    def __init__(self, in_channels: int):
        super().__init__(in_channels)

    def forward(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def score_all(self, z: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        """Score user against all nodes."""
        return (z * user_emb).sum(dim=-1)


class CosinePredictor(BaseLinkPredictor):
    """
    Cosine similarity decoder: score(u, v) = cos(z_u, z_v)
    Normalized dot product, range [-1, 1].
    """

    def __init__(self, in_channels: int):
        super().__init__(in_channels)

    def forward(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return F.cosine_similarity(src, dst, dim=-1)

    def score_all(self, z: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(z, user_emb.unsqueeze(0).expand_as(z), dim=-1)


class BilinearPredictor(BaseLinkPredictor):
    """
    Bilinear decoder: score(u, v) = z_u^T W z_v
    Learnable weight matrix for asymmetric scoring.
    """

    def __init__(self, in_channels: int):
        super().__init__(in_channels)
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # src @ W @ dst^T -> (src @ W) * dst summed
        return (src @ self.weight * dst).sum(dim=-1)

    def score_all(self, z: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        return (z @ self.weight * user_emb).sum(dim=-1)


class MLPPredictor(BaseLinkPredictor):
    """
    MLP decoder: score(u, v) = MLP([z_u || z_v || |z_u - z_v| || z_u * z_v])
    Concatenates multiple interaction features for expressive scoring.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__(in_channels)
        self.hidden_channels = hidden_channels

        # Input: concat of [z_u, z_v, |z_u - z_v|, z_u * z_v] -> 4 * in_channels
        input_dim = 4 * in_channels

        layers = []
        layers.append(nn.Linear(input_dim, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_channels, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]

        # Interaction features
        diff = torch.abs(src - dst)
        prod = src * dst
        features = torch.cat([src, dst, diff, prod], dim=-1)

        return self.mlp(features).squeeze(-1)

    def score_all(self, z: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        """Score user against all nodes."""
        N = z.size(0)
        user_emb_expanded = user_emb.unsqueeze(0).expand(N, -1)

        diff = torch.abs(z - user_emb_expanded)
        prod = z * user_emb_expanded
        features = torch.cat([z, user_emb_expanded, diff, prod], dim=-1)

        return self.mlp(features).squeeze(-1)


class HadamardMLPPredictor(BaseLinkPredictor):
    """
    Hadamard product + MLP (common in knowledge graph completion).
    score(u, v) = MLP(z_u * z_v)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__(in_channels)

        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_channels, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        hadamard = src * dst
        return self.mlp(hadamard).squeeze(-1)

    def score_all(self, z: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        hadamard = z * user_emb
        return self.mlp(hadamard).squeeze(-1)


def create_predictor(
    predictor_type: str,
    in_channels: int,
    **kwargs,
) -> BaseLinkPredictor:
    """
    Factory to create predictor by name.

    Args:
        predictor_type: One of 'dot', 'cosine', 'bilinear', 'mlp', 'hadamard_mlp'
        **kwargs: Additional arguments (hidden_channels, num_layers, dropout)
    """
    predictor_type = predictor_type.lower()

    if predictor_type == "dot":
        return DotProductPredictor(in_channels)
    elif predictor_type == "cosine":
        return CosinePredictor(in_channels)
    elif predictor_type == "bilinear":
        return BilinearPredictor(in_channels)
    elif predictor_type == "mlp":
        return MLPPredictor(in_channels, **kwargs)
    elif predictor_type == "hadamard_mlp":
        return HadamardMLPPredictor(in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown predictor: {predictor_type}. "
                         f"Available: dot, cosine, bilinear, mlp, hadamard_mlp")