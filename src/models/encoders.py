"""
GNN Encoder implementations: GCN, GAT, GATv2, GIN, GraphConv, LightGCN.
All support residual connections and LayerNorm for deep architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GATv2Conv,
    GINConv,
    GraphConv,
    SAGEConv,
)
from typing import Optional, List
from src.models.base import BaseGNNEncoder


class ResidualBlock(nn.Module):
    """Residual connection with optional LayerNorm."""

    def __init__(
        self,
        channels: int,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """x: transformed features, residual: skip connection input."""
        out = x + residual
        if self.use_layer_norm:
            out = self.norm(out)
        out = self.dropout(out)
        return out


class GCNEncoder(BaseGNNEncoder):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).

    Supports residual connections and LayerNorm for deep architectures.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        cached: bool = False,
    ):
        super().__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, dropout, use_residual, use_layer_norm
        )

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList() if use_residual else None
        self.norms = nn.ModuleList() if use_layer_norm else None

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, cached=cached))
        else:
            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels))

            # Middle layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=cached))
                if use_layer_norm:
                    self.norms.append(nn.LayerNorm(hidden_channels))
                if use_residual:
                    self.residuals.append(ResidualBlock(hidden_channels, use_layer_norm, dropout))

            # Last layer
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and i < len(self.convs) - 1 else None

            x = conv(x, edge_index)

            if i < len(self.convs) - 1:  # Not last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.use_residual and residual is not None and x.size(-1) == residual.size(-1):
                    x = self.residuals[i - 1](x, residual)

        return x


class GATEncoder(BaseGNNEncoder):
    """
    Graph Attention Network (Veličković et al., 2018).

    Multi-head attention with residual and LayerNorm support.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.0,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        concat: bool = True,
    ):
        super().__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, dropout, use_residual, use_layer_norm
        )
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.concat = concat

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList() if use_residual else None
        self.norms = nn.ModuleList() if use_layer_norm else None

        if num_layers == 1:
            self.convs.append(GATConv(
                in_channels, out_channels,
                heads=1, concat=False, dropout=attn_dropout
            ))
        else:
            # First layer: multi-head with concat
            self.convs.append(GATConv(
                in_channels, hidden_channels,
                heads=heads, concat=concat, dropout=attn_dropout
            ))
            first_out = hidden_channels * heads if concat else hidden_channels
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(first_out))

            # Middle layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(
                    first_out, hidden_channels,
                    heads=heads, concat=concat, dropout=attn_dropout
                ))
                if use_layer_norm:
                    self.norms.append(nn.LayerNorm(first_out))
                if use_residual:
                    self.residuals.append(ResidualBlock(first_out, use_layer_norm, dropout))

            # Last layer: single head, no concat
            self.convs.append(GATConv(
                first_out, out_channels,
                heads=1, concat=False, dropout=attn_dropout
            ))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and i < len(self.convs) - 1 else None

            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.use_residual and residual is not None and x.size(-1) == residual.size(-1):
                    x = self.residuals[i - 1](x, residual)

        return x


class GATv2Encoder(BaseGNNEncoder):
    """
    GATv2: How Attentive are Graph Attention Networks? (Brody et al., 2022).
    More expressive attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.0,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        concat: bool = True,
    ):
        super().__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, dropout, use_residual, use_layer_norm
        )
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.concat = concat

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList() if use_residual else None
        self.norms = nn.ModuleList() if use_layer_norm else None

        if num_layers == 1:
            self.convs.append(GATv2Conv(
                in_channels, out_channels,
                heads=1, concat=False, dropout=attn_dropout
            ))
        else:
            self.convs.append(GATv2Conv(
                in_channels, hidden_channels,
                heads=heads, concat=concat, dropout=attn_dropout
            ))
            first_out = hidden_channels * heads if concat else hidden_channels
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(first_out))

            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(
                    first_out, hidden_channels,
                    heads=heads, concat=concat, dropout=attn_dropout
                ))
                if use_layer_norm:
                    self.norms.append(nn.LayerNorm(first_out))
                if use_residual:
                    self.residuals.append(ResidualBlock(first_out, use_layer_norm, dropout))

            self.convs.append(GATv2Conv(
                first_out, out_channels,
                heads=1, concat=False, dropout=attn_dropout
            ))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and i < len(self.convs) - 1 else None

            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.use_residual and residual is not None and x.size(-1) == residual.size(-1):
                    x = self.residuals[i - 1](x, residual)

        return x


class GINEncoder(BaseGNNEncoder):
    """
    Graph Isomorphism Network (Xu et al., 2019).
    Maximally expressive GNN for graph-level tasks, also works for node tasks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, dropout, use_residual, use_layer_norm
        )

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList() if use_residual else None
        self.norms = nn.ModuleList() if use_layer_norm else None

        def make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_dim),
            )

        if num_layers == 1:
            self.convs.append(GINConv(make_mlp(in_channels, out_channels), eps=eps, train_eps=train_eps))
        else:
            # First layer
            self.convs.append(GINConv(make_mlp(in_channels, hidden_channels), eps=eps, train_eps=train_eps))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels))

            # Middle layers
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels), eps=eps, train_eps=train_eps))
                if use_layer_norm:
                    self.norms.append(nn.LayerNorm(hidden_channels))
                if use_residual:
                    self.residuals.append(ResidualBlock(hidden_channels, use_layer_norm, dropout))

            # Last layer
            self.convs.append(GINConv(make_mlp(hidden_channels, out_channels), eps=eps, train_eps=train_eps))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and i < len(self.convs) - 1 else None

            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.use_residual and residual is not None:
                    x = self.residuals[i - 1](x, residual)

        return x


class GraphConvEncoder(BaseGNNEncoder):
    """
    Vanilla Message Passing (GraphConv from PyG).
    No symmetric normalization - learns aggregation weights.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        aggr: str = "mean",
    ):
        super().__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, dropout, use_residual, use_layer_norm
        )

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList() if use_residual else None
        self.norms = nn.ModuleList() if use_layer_norm else None

        if num_layers == 1:
            self.convs.append(GraphConv(in_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr=aggr))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr=aggr))
                if use_layer_norm:
                    self.norms.append(nn.LayerNorm(hidden_channels))
                if use_residual:
                    self.residuals.append(ResidualBlock(hidden_channels, use_layer_norm, dropout))

            self.convs.append(GraphConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and i < len(self.convs) - 1 else None

            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.use_residual and residual is not None:
                    x = self.residuals[i - 1](x, residual)

        return x


class LightGCNEncoder(BaseGNNEncoder):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (He et al., 2020).
    No feature transformation, no non-linearity - just weighted neighborhood aggregation.
    Designed for collaborative filtering but works for friend recommendation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,  # LightGCN typically no dropout
        use_residual: bool = False,  # Not applicable
        use_layer_norm: bool = False,
    ):
        # LightGCN uses embedding layer as input, then propagates
        super().__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, dropout, use_residual, use_layer_norm
        )

        # Learnable embeddings replace input features
        self.embedding = nn.Embedding(in_channels, out_channels)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(out_channels, out_channels, cached=False, add_self_loops=True))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: Node indices (N,) or one-hot (N, num_nodes) - we use indices
        """
        # If x is one-hot, convert to indices
        if x.dim() == 2 and x.size(1) == self.in_channels:
            x = x.argmax(dim=1)

        # Get embeddings
        z = self.embedding(x)

        # Layer-wise propagation with residual (sum of all layers)
        all_embeddings = [z]
        for conv in self.convs:
            z = conv(z, edge_index)
            all_embeddings.append(z)

        # Final embedding = sum of all layers (LightGCN)
        final_z = torch.stack(all_embeddings, dim=0).sum(dim=0)
        return final_z


class SAGEEncoder(BaseGNNEncoder):
    """
    GraphSAGE (Hamilton et al., 2017).
    Inductive encoder with mean/max/LSTM aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        aggr: str = "mean",
    ):
        super().__init__(
            in_channels, hidden_channels, out_channels,
            num_layers, dropout, use_residual, use_layer_norm
        )

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList() if use_residual else None
        self.norms = nn.ModuleList() if use_layer_norm else None

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
                if use_layer_norm:
                    self.norms.append(nn.LayerNorm(hidden_channels))
                if use_residual:
                    self.residuals.append(ResidualBlock(hidden_channels, use_layer_norm, dropout))

            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 and i < len(self.convs) - 1 else None

            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if self.use_residual and residual is not None:
                    x = self.residuals[i - 1](x, residual)

        return x


# Factory function
def create_encoder(
    model_type: str,
    in_channels: int,
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    use_residual: bool = False,
    use_layer_norm: bool = False,
    **kwargs,
) -> BaseGNNEncoder:
    """
    Factory to create encoder by name.

    Args:
        model_type: One of 'gcn', 'gat', 'gatv2', 'gin', 'graphconv', 'lightgcn', 'sage'
        **kwargs: Additional model-specific arguments (heads, attn_dropout, aggr, etc.)

    Returns:
        Encoder instance
    """
    model_type = model_type.lower()

    if model_type == "gcn":
        return GCNEncoder(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, use_residual, use_layer_norm, **kwargs
        )
    elif model_type == "gat":
        return GATEncoder(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout=dropout, use_residual=use_residual, use_layer_norm=use_layer_norm, **kwargs
        )
    elif model_type == "gatv2":
        return GATv2Encoder(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout=dropout, use_residual=use_residual, use_layer_norm=use_layer_norm, **kwargs
        )
    elif model_type == "gin":
        return GINEncoder(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, use_residual, use_layer_norm, **kwargs
        )
    elif model_type == "graphconv":
        return GraphConvEncoder(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, use_residual, use_layer_norm, **kwargs
        )
    elif model_type == "lightgcn":
        return LightGCNEncoder(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, use_residual, use_layer_norm, **kwargs
        )
    elif model_type == "sage":
        return SAGEEncoder(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, use_residual, use_layer_norm, **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {model_type}. "
                         f"Available: gcn, gat, gatv2, gin, graphconv, lightgcn, sage")