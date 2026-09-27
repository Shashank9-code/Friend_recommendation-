"""
=============================================================================
Friend Recommendation System using Link Prediction (GraphConv — Vanilla MPNN)
=============================================================================

This variant replaces PyG's GCNConv (Kipf & Welling normalised convolution)
with **GraphConv**, a basic message-passing neural network (MPNN) layer that
does NOT apply symmetric normalisation.  All other pipeline components
(feature engineering, edge splitting, decoding, evaluation, and
visualisations) remain identical to the GCN version.

GraphConv computes:
    h_v^{(l+1)} = W_1 h_v^{(l)} + W_2 · mean_{u ∈ N(v)} h_u^{(l)}

Unlike GCNConv, there is no 1/√(d_u · d_v) normalisation — the model treats
all neighbour messages equally, relying on learned weights to calibrate.

Author  : Shashank Prabhakar
Date    : April 2026
"""

# ── Standard library ─────────────────────────────────────────────────────
import os
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
from sklearn.manifold import TSNE
try:
    import community.community_louvain as louvain  # python-louvain
except ImportError:
    louvain = None  # graceful fallback if not installed

# ── PyTorch Geometric ────────────────────────────────────────────────────
from torch_geometric.datasets import SNAPDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GraphConv                    # ← Changed
from torch_geometric.utils import negative_sampling, to_networkx

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
def load_facebook_dataset(root: str = "./data") -> "torch_geometric.data.Data":
    """
    Download and load the SNAP Facebook ego-network dataset with
    engineered structural features (Node Degree + PageRank).
    """
    print("=" * 60)
    print("  Step 1 — Loading SNAP Facebook dataset")
    print("=" * 60)

    dataset = SNAPDataset(root=root, name="ego-Facebook")
    data = dataset[0]

    print(f"  Nodes          : {data.num_nodes}")
    print(f"  Edges          : {data.num_edges}  (directed count)")
    print(f"  Node features  : {data.num_node_features}")
    print(f"  Has self-loops : {data.has_self_loops()}")
    print()

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

    print(f"  ✅  Features: identity({data.num_nodes}) + degree(1) + pagerank(1)")
    print(f"      Final feature dim: {data.x.size(1)}")
    print()

    return data


# ══════════════════════════════════════════════════════════════════════════
# 2.  EDGE SPLITTING  &  NEGATIVE SAMPLING
# ══════════════════════════════════════════════════════════════════════════
def prepare_link_prediction_splits(data):
    """Split graph edges into train / val / test with negative sampling."""
    print("=" * 60)
    print("  Step 2 — Splitting edges & generating negative samples")
    print("=" * 60)

    splitter = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        split_labels=False,
    )

    train_data, val_data, test_data = splitter(data)

    for name, split in [("Train", train_data),
                         ("Val",   val_data),
                         ("Test",  test_data)]:
        num_pos = int(split.edge_label.sum().item())
        num_neg = len(split.edge_label) - num_pos
        print(f"  {name:5s}  |  message edges: {split.edge_index.size(1):>7,}"
              f"  |  supervision pos: {num_pos:>6,}  neg: {num_neg:>6,}")

    print()
    return train_data, val_data, test_data


# ══════════════════════════════════════════════════════════════════════════
# 3.  GraphConv ENCODER  (Vanilla MPNN)
# ══════════════════════════════════════════════════════════════════════════
class GraphConvEncoder(nn.Module):
    """
    A 2-layer vanilla Message-Passing Neural Network using PyG's GraphConv.

    Unlike GCNConv, GraphConv does NOT use symmetric normalisation.
    It computes:
        h_v' = W_1 h_v + W_2 · AGG({h_u : u ∈ N(v)})
    where AGG is mean aggregation by default.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, dropout: float = 0.3):
        super().__init__()

        self.conv1 = GraphConv(in_channels, hidden_channels)   # ← GraphConv
        self.conv2 = GraphConv(hidden_channels, out_channels)  # ← GraphConv

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.conv1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        z = self.conv2(z, edge_index)
        return z


# ══════════════════════════════════════════════════════════════════════════
# 4.  LINK PREDICTOR  (Dot-Product Decoder)
# ══════════════════════════════════════════════════════════════════════════
def dot_product_decode(z, edge_label_index):
    """Dot-product link scoring — identical to GCN variant."""
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=-1)


# ══════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, optimizer, criterion, train_data):
    """Execute one training epoch."""
    model.train()
    optimizer.zero_grad()

    z = model(train_data.x, train_data.edge_index)
    logits = dot_product_decode(z, train_data.edge_label_index)
    loss = criterion(logits, train_data.edge_label)

    loss.backward()
    optimizer.step()
    return loss.item()


# ══════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, data):
    """Evaluate with AUC, AP, and loss."""
    model.eval()
    z = model(data.x, data.edge_index)
    logits = dot_product_decode(z, data.edge_label_index)

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, data.edge_label).item()

    probs = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.cpu().numpy()

    auc = roc_auc_score(labels, probs)
    ap  = average_precision_score(labels, probs)
    return auc, ap, loss


# ══════════════════════════════════════════════════════════════════════════
# 7.  FRIEND RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def recommend_friends(model, data, user_id: int, top_k: int = 10):
    """Recommend top-K friends for a given user."""
    model.eval()
    z = model(data.x, data.edge_index)
    user_emb = z[user_id]

    scores = (z * user_emb).sum(dim=-1)
    scores = torch.sigmoid(scores)

    edge_index = data.edge_index
    mask = edge_index[0] == user_id
    existing_friends = set(edge_index[1][mask].cpu().tolist())
    existing_friends.add(user_id)

    for nid in existing_friends:
        scores[nid] = -1.0

    top_scores, top_indices = torch.topk(scores, k=top_k)
    return list(zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()))


# ══════════════════════════════════════════════════════════════════════════
# 8.  VISUALIZATION — Learning Curves
# ══════════════════════════════════════════════════════════════════════════
def plot_learning_curves(history, save_path="learning_curves_gnn.png"):
    """Plot training/validation loss and validation metrics vs epochs."""
    epochs = history["epochs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss",
             color="#e74c3c", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss",
             color="#3498db", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (BCE)", fontsize=12)
    ax1.set_title("Loss vs. Epochs (GraphConv)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_auc"], label="Val AUC",
             color="#2ecc71", linewidth=2)
    ax2.plot(epochs, history["val_ap"], label="Val AP",
             color="#9b59b6", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Validation Metrics vs. Epochs (GraphConv)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Learning curves saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 9.  VISUALIZATION — t-SNE Embeddings
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def get_embeddings(model, data):
    """Extract node embeddings from the model."""
    model.eval()
    z = model(data.x, data.edge_index)
    return z.cpu().numpy()


def plot_tsne_embeddings(emb_before, emb_after,
                         data=None,
                         save_path="tsne_clusters_gnn.png"):
    """
    Side-by-side t-SNE: untrained vs. trained embeddings.
    The trained panel is coloured by Louvain community so distinct social
    cliques appear as separate colour islands.
    """
    print("  🔄  Running t-SNE … (this may take a minute)")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)

    proj_before = tsne.fit_transform(emb_before)
    proj_after  = tsne.fit_transform(emb_after)

    # ── Louvain community labels ──────────────────────────────────────────
    cluster_labels = None
    num_communities = 0
    if louvain is not None and data is not None:
        print("  🔍  Detecting Louvain communities for cluster colouring …")
        G_nx = to_networkx(data, to_undirected=True)
        partition = louvain.best_partition(G_nx)
        cluster_labels = np.array([partition[i] for i in range(data.num_nodes)])
        num_communities = cluster_labels.max() + 1
        print(f"  ✅  Found {num_communities} communities")
    else:
        if louvain is None:
            print("  ⚠️   python-louvain not found — install with: "
                  "pip install python-louvain  (falling back to rainbow)")
        cluster_labels = np.arange(len(proj_after))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Untrained (rainbow for contrast) ─────────────────────────────────
    ax1.scatter(proj_before[:, 0], proj_before[:, 1],
                c=np.arange(len(proj_before)), cmap="Spectral", s=5, alpha=0.7)
    ax1.set_title("Epoch 1 — Untrained Embeddings", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE Dim 1"); ax1.set_ylabel("t-SNE Dim 2")
    ax1.grid(True, alpha=0.2)

    # ── Trained (Louvain community colours) ──────────────────────────────
    scatter = ax2.scatter(proj_after[:, 0], proj_after[:, 1],
                          c=cluster_labels, cmap="tab20", s=10, alpha=0.8)
    title_suffix = (
        f" ({num_communities} Louvain communities)"
        if num_communities > 0 else ""
    )
    ax2.set_title(
        f"Final Epoch — Trained Embeddings{title_suffix}",
        fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("t-SNE Dim 1"); ax2.set_ylabel("t-SNE Dim 2")
    ax2.grid(True, alpha=0.2)

    if num_communities > 0:
        cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
        cbar.set_label("Community ID", fontsize=10)

    plt.suptitle(
        "t-SNE Visualization of GraphConv Node Embeddings\n"
        "(Right panel coloured by Louvain social community)",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  t-SNE embeddings saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    HIDDEN_CHANNELS = 128
    OUT_CHANNELS    = 64
    LEARNING_RATE   = 0.01
    EPOCHS          = 100
    DROPOUT         = 0.3

    # 1. Load data
    data = load_facebook_dataset(root="./data")

    # 2. Split edges
    train_data, val_data, test_data = prepare_link_prediction_splits(data)
    train_data = train_data.to(DEVICE)
    val_data   = val_data.to(DEVICE)
    test_data  = test_data.to(DEVICE)

    # 3. Initialise model
    in_channels = train_data.x.size(1)
    model = GraphConvEncoder(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("=" * 60)
    print("  Step 3 — Model architecture (GraphConv / Vanilla MPNN)")
    print("=" * 60)
    print(model)
    print(f"\n  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # 4. Capture Epoch-1 embeddings
    emb_epoch1 = get_embeddings(model, train_data)

    # 5. Training
    print("=" * 60)
    print("  Step 4 — Training")
    print("=" * 60)

    best_val_auc = 0.0
    best_epoch   = 0
    history = {"epochs": [], "train_loss": [], "val_loss": [],
               "val_auc": [], "val_ap": []}

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, criterion, train_data)
        val_auc, val_ap, val_loss = evaluate(model, val_data)

        history["epochs"].append(epoch)
        history["train_loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)

        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            torch.save(model.state_dict(), "best_model_gnn.pt")
            marker = "  ◀ best"

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}"
                  f"  |  Loss: {loss:.4f}"
                  f"  |  Val AUC: {val_auc:.4f}"
                  f"  |  Val AP: {val_ap:.4f}{marker}")

    print(f"\n  Best Val AUC: {best_val_auc:.4f}  (epoch {best_epoch})\n")

    # 6. Final embeddings + plots
    emb_final = get_embeddings(model, train_data)
    plot_learning_curves(history, save_path="learning_curves_gnn.png")
    plot_tsne_embeddings(emb_epoch1, emb_final, data=data, save_path="tsne_clusters_gnn.png")

    # 7. Test evaluation
    print("=" * 60)
    print("  Step 5 — Test evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load("best_model_gnn.pt", weights_only=True))
    test_auc, test_ap, test_loss = evaluate(model, test_data)
    print(f"  ✅  Test AUC: {test_auc:.4f}")
    print(f"  ✅  Test AP:  {test_ap:.4f}\n")

    # 8. Demo
    print("=" * 60)
    print("  Step 6 — Friend recommendations (demo)")
    print("=" * 60)

    demo_user = random.randint(0, data.num_nodes - 1)
    recommendations = recommend_friends(model, train_data,
                                        user_id=demo_user, top_k=10)

    print(f"\n  Top 10 friend recommendations for User #{demo_user}:\n")
    print(f"  {'Rank':<6}{'User ID':<12}{'Score':<10}")
    print(f"  {'─' * 6}{'─' * 12}{'─' * 10}")
    for rank, (node_id, score) in enumerate(recommendations, start=1):
        print(f"  {rank:<6}{node_id:<12}{score:<10.4f}")

    print("\n  Done! ✨\n")
