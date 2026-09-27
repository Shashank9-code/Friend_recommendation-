"""
=============================================================================
Ablation Study — Effect of GCN Depth on Link Prediction Performance
=============================================================================

This script trains three GCN variants with different depths:

    • 1-layer GCN  — captures only immediate (1-hop) neighbourhood
    • 2-layer GCN  — captures 2-hop neighbourhood (baseline)
    • 5-layer GCN  — captures 5-hop neighbourhood (deep)

The goal is to demonstrate the **over-smoothing problem** in deep GNNs:
as the number of layers increases, node embeddings converge to the same
vector, making it impossible to distinguish between users.

The final Test AUC for each model is plotted as a bar chart and saved
as ``ablation_results.png``.

Author  : Shashank Prabhakar
Date    : April 2026
"""

# ── Standard library ─────────────────────────────────────────────────────
import random

# ── Third-party ──────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

# ── PyTorch Geometric ────────────────────────────────────────────────────
from torch_geometric.datasets import SNAPDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx

# ──────────────────────────────────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  +  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
def load_facebook_dataset(root: str = "./data"):
    """Load SNAP Facebook dataset with engineered structural features."""
    print("=" * 60)
    print("  Loading SNAP Facebook dataset")
    print("=" * 60)

    dataset = SNAPDataset(root=root, name="ego-Facebook")
    data = dataset[0]

    print(f"  Nodes: {data.num_nodes}  |  Edges: {data.num_edges}")

    # ── Feature Engineering ──────────────────────────────────────────────
    print("  🔧  Engineering structural features …")

    G = to_networkx(data, to_undirected=True)

    # Node Degree (normalised)
    degree_dict = dict(G.degree())
    degree_vals = np.array([degree_dict[i] for i in range(data.num_nodes)],
                           dtype=np.float32)
    deg_max = degree_vals.max() if degree_vals.max() > 0 else 1.0
    degree_vals /= deg_max

    # PageRank (normalised)
    pr_dict = nx.pagerank(G, alpha=0.85)
    pr_vals = np.array([pr_dict[i] for i in range(data.num_nodes)],
                       dtype=np.float32)
    pr_max = pr_vals.max() if pr_vals.max() > 0 else 1.0
    pr_vals /= pr_max

    identity = torch.eye(data.num_nodes, dtype=torch.float)
    degree_tensor = torch.tensor(degree_vals).unsqueeze(1)
    pr_tensor     = torch.tensor(pr_vals).unsqueeze(1)

    data.x = torch.cat([identity, degree_tensor, pr_tensor], dim=1)
    print(f"  ✅  Feature dim: {data.x.size(1)}\n")

    return data


def prepare_splits(data):
    """Split edges into train / val / test."""
    splitter = RandomLinkSplit(
        num_val=0.05, num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        split_labels=False,
    )
    return splitter(data)


# ══════════════════════════════════════════════════════════════════════════
# 2.  FLEXIBLE-DEPTH GCN ENCODER
# ══════════════════════════════════════════════════════════════════════════
class FlexGCNEncoder(nn.Module):
    """
    A GCN encoder whose depth (number of layers) is configurable.

    Parameters
    ----------
    in_channels     : int  — input feature dimensionality
    hidden_channels : int  — width of all hidden layers
    out_channels    : int  — dimensionality of final embeddings
    num_layers      : int  — total number of GCNConv layers
    dropout         : float — dropout probability
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            # Single layer: input → output directly
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            # First layer: input → hidden
            self.convs.append(GCNConv(in_channels, hidden_channels))
            # Middle layers: hidden → hidden
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            # Final layer: hidden → output
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 3.  DECODER + TRAIN + EVAL
# ══════════════════════════════════════════════════════════════════════════
def dot_product_decode(z, edge_label_index):
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=-1)


def train_one_epoch(model, optimizer, criterion, train_data):
    model.train()
    optimizer.zero_grad()
    z = model(train_data.x, train_data.edge_index)
    logits = dot_product_decode(z, train_data.edge_label_index)
    loss = criterion(logits, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    z = model(data.x, data.edge_index)
    logits = dot_product_decode(z, data.edge_label_index)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.cpu().numpy()
    auc = roc_auc_score(labels, probs)
    ap  = average_precision_score(labels, probs)
    return auc, ap


# ══════════════════════════════════════════════════════════════════════════
# 4.  ABLATION — TRAIN & COMPARE
# ══════════════════════════════════════════════════════════════════════════
def run_ablation(train_data, val_data, test_data, in_channels,
                 layer_configs=(1, 2, 5),
                 hidden=128, out=64, lr=0.01, epochs=100, dropout=0.3):
    """
    Train a GCN for each layer configuration and return test metrics.

    Returns
    -------
    results : dict  — {num_layers: {'test_auc': float, 'test_ap': float}}
    """
    results = {}

    for n_layers in layer_configs:
        print("=" * 60)
        print(f"  Training {n_layers}-layer GCN …")
        print("=" * 60)

        # Reset seed for fair comparison
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = FlexGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=out,
            num_layers=n_layers,
            dropout=dropout,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0.0

        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, optimizer, criterion, train_data)

            if epoch % 20 == 0 or epoch == 1:
                val_auc, val_ap = evaluate(model, val_data)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), f"ablation_{n_layers}L.pt")
                print(f"    Epoch {epoch:3d}  |  Loss: {loss:.4f}"
                      f"  |  Val AUC: {val_auc:.4f}")

        # Load best model and evaluate on test
        model.load_state_dict(
            torch.load(f"ablation_{n_layers}L.pt", weights_only=True))
        test_auc, test_ap = evaluate(model, test_data)

        results[n_layers] = {"test_auc": test_auc, "test_ap": test_ap}
        print(f"  ✅  {n_layers}-layer GCN  →  Test AUC: {test_auc:.4f}"
              f"  |  Test AP: {test_ap:.4f}\n")

    return results


# ══════════════════════════════════════════════════════════════════════════
# 5.  VISUALIZATION — Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def plot_ablation_results(results, save_path="ablation_results.png"):
    """
    Plot a bar chart comparing Test AUC across different GCN depths
    to illustrate the over-smoothing problem.
    """
    layers = sorted(results.keys())
    aucs   = [results[l]["test_auc"] for l in layers]
    aps    = [results[l]["test_ap"]  for l in layers]
    labels = [f"{l}-Layer" for l in layers]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_auc = ax.bar(x - width / 2, aucs, width, label="Test AUC",
                      color=["#2ecc71", "#3498db", "#e74c3c"],
                      edgecolor="black", linewidth=0.8)
    bars_ap  = ax.bar(x + width / 2, aps, width, label="Test AP",
                      color=["#82e0aa", "#85c1e9", "#f1948a"],
                      edgecolor="black", linewidth=0.8)

    # Add value labels on bars
    for bar in bars_auc:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar in bars_ap:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Model Depth", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Ablation Study: GCN Depth vs. Link Prediction Performance\n"
                 "(Demonstrating the Over-Smoothing Problem)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0.0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Ablation results saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    HIDDEN_CHANNELS = 128
    OUT_CHANNELS    = 64
    LEARNING_RATE   = 0.01
    EPOCHS          = 100
    DROPOUT         = 0.3

    # 1. Load engineered dataset
    data = load_facebook_dataset(root="./data")

    # 2. Split edges
    train_data, val_data, test_data = prepare_splits(data)
    train_data = train_data.to(DEVICE)
    val_data   = val_data.to(DEVICE)
    test_data  = test_data.to(DEVICE)

    in_channels = train_data.x.size(1)

    # 3. Run ablation study
    results = run_ablation(
        train_data, val_data, test_data,
        in_channels=in_channels,
        layer_configs=(1, 2, 5),
        hidden=HIDDEN_CHANNELS,
        out=OUT_CHANNELS,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        dropout=DROPOUT,
    )

    # 4. Visualise
    plot_ablation_results(results, save_path="ablation_results.png")

    # 5. Summary
    print("\n" + "=" * 60)
    print("  ABLATION STUDY — SUMMARY")
    print("=" * 60)
    for n_layers in sorted(results.keys()):
        r = results[n_layers]
        print(f"    {n_layers}-Layer GCN  →  AUC: {r['test_auc']:.4f}"
              f"  |  AP: {r['test_ap']:.4f}")

    print("\n  Observation:")
    print("  • Shallow (1-layer) may underfit — limited receptive field.")
    print("  • Optimal depth (2-layer) typically gives the best AUC.")
    print("  • Deep (5-layer) suffers from over-smoothing — node")
    print("    embeddings converge, degrading discrimination ability.")
    print("\n  Done! ✨\n")
