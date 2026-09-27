"""
=============================================================================
Friend Recommendation System using Link Prediction (GCN — Graph Convolutional Network)
=============================================================================

This script builds a friend recommendation engine on the SNAP Facebook
ego-network dataset.  The core idea is **Link Prediction**: given a social
graph, predict which pairs of users (nodes) are likely to become friends
(edges) in the future.

Pipeline overview
-----------------
1.  Load the Facebook ego-network graph.
2.  Engineer structural features (Node Degree, PageRank) and append to
    identity features for richer node representations.
3.  Split existing edges into train / val / test sets and generate
    *negative* edges (pairs that are NOT friends) for each split.
4.  Train a 2-layer GCN that learns a low-dimensional embedding for every
    user.
5.  Score candidate links with a dot-product decoder.
6.  Evaluate with AUC and Average Precision on held-out test edges.
7.  Visualise learning curves and t-SNE node embeddings.

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
matplotlib.use("Agg")                       # non-interactive backend
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
from torch_geometric.nn import GCNConv
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

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  +  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
def load_facebook_dataset(root: str = "./data") -> "torch_geometric.data.Data":
    """
    Download and load the SNAP Facebook ego-network dataset with
    engineered structural features.

    Feature engineering
    ───────────────────
    Instead of a pure identity (one-hot) matrix, we also compute:
      • **Node Degree** — the number of friends each user has.
      • **PageRank**    — a global importance/centrality measure.

    These two features are appended as extra columns to the identity
    matrix, giving the GCN richer input signals about graph topology.

    Returns
    -------
    data : torch_geometric.data.Data
        The single graph object with enriched `x` and `edge_index`.
    """
    print("=" * 60)
    print("  Step 1 — Loading SNAP Facebook dataset")
    print("=" * 60)

    dataset = SNAPDataset(root=root, name="ego-Facebook")

    # SNAPDataset stores the graph(s) as a list; Facebook has exactly one.
    data = dataset[0]

    print(f"  Nodes          : {data.num_nodes}")
    print(f"  Edges          : {data.num_edges}  (directed count)")
    print(f"  Node features  : {data.num_node_features}")
    print(f"  Has self-loops : {data.has_self_loops()}")
    print()

    # ── Feature Engineering ──────────────────────────────────────────────
    print("  🔧  Engineering structural features …")

    # Convert to NetworkX for structural feature calculation
    G = to_networkx(data, to_undirected=True)

    # 1) Node Degree
    degree_dict = dict(G.degree())
    degree_vals = np.array([degree_dict[i] for i in range(data.num_nodes)],
                           dtype=np.float32)
    # Normalise to [0, 1]
    deg_max = degree_vals.max() if degree_vals.max() > 0 else 1.0
    degree_vals /= deg_max

    # 2) PageRank
    pr_dict = nx.pagerank(G, alpha=0.85)
    pr_vals = np.array([pr_dict[i] for i in range(data.num_nodes)],
                       dtype=np.float32)
    # Normalise to [0, 1]
    pr_max = pr_vals.max() if pr_vals.max() > 0 else 1.0
    pr_vals /= pr_max

    # Start with identity (one-hot) features
    identity = torch.eye(data.num_nodes, dtype=torch.float)

    # Append degree and PageRank as extra columns 
    degree_tensor = torch.tensor(degree_vals).unsqueeze(1)   # (N, 1)
    pr_tensor     = torch.tensor(pr_vals).unsqueeze(1)       # (N, 1)

    data.x = torch.cat([identity, degree_tensor, pr_tensor], dim=1)

    print(f"  ✅  Features: identity({data.num_nodes}) + degree(1) + pagerank(1)")
    print(f"      Final feature dim: {data.x.size(1)}")
    print()

    return data


# ══════════════════════════════════════════════════════════════════════════
# 2.  EDGE SPLITTING  &  NEGATIVE SAMPLING
# ══════════════════════════════════════════════════════════════════════════
def prepare_link_prediction_splits(data):
    """
    Split the graph edges into training, validation, and test sets.

    How it works
    ────────────
    ``RandomLinkSplit`` from PyG performs the following:

    1.  **Positive edges** — The existing edges in the graph are randomly
        partitioned into three disjoint sets:
            • Training   (85 %)
            • Validation ( 5 %)
            • Test       (10 %)

    2.  **Negative edges** — For each split, PyG also samples an equal
        number of *negative* edges — pairs of nodes that are NOT connected
        in the original graph.  These serve as "non-friend" examples so the
        model can learn the difference between real friendships and random
        pairs.

    3.  **Message-passing edges vs. supervision edges** —
        During training the GCN sees *only* the training edges for message
        passing (aggregating neighbour features).  The validation and test
        positive edges are hidden from the GCN – they are used exclusively
        to evaluate how well the model predicts unseen links.

    4.  ``add_negative_train_samples=True`` ensures we also have negatives
        for the training split, which we need for the BCE loss.

    Returns
    -------
    train_data, val_data, test_data : Data objects
        Each object contains:
        • ``edge_index``      — edges available for GCN message passing
        • ``edge_label_index``— edges to predict (both pos & neg)
        • ``edge_label``      — ground-truth labels (1 = friend, 0 = not)
    """
    print("=" * 60)
    print("  Step 2 — Splitting edges & generating negative samples")
    print("=" * 60)

    splitter = RandomLinkSplit(
        num_val=0.05,                     # 5 % of edges for validation
        num_test=0.10,                    # 10 % of edges for testing
        is_undirected=True,               # Facebook friendships are mutual
        add_negative_train_samples=True,  # Need negatives for training BCE
        neg_sampling_ratio=1.0,           # 1 negative per positive edge
        split_labels=False,               # Keep pos & neg in one tensor
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
# 3.  GCN ENCODER  (Node Embedding Generator)
# ══════════════════════════════════════════════════════════════════════════
class GCNEncoder(nn.Module):
    """
    A 2-layer Graph Convolutional Network that produces a dense embedding
    vector for every node in the graph.

    Architecture
    ────────────
        Input features  ──►  GCNConv(in → hidden)  ──►  ReLU + Dropout
                          ──►  GCNConv(hidden → out)

    The GCN aggregates information from a node's neighbours (friends) via
    the *message-passing* paradigm:

        h_v^{(l+1)} = σ( Σ_{u ∈ N(v)} (1/√(d_u · d_v)) · W^{(l)} · h_u^{(l)} )

    After two rounds of aggregation, each node's embedding captures the
    structure of its 2-hop neighbourhood — enough to detect local
    connectivity patterns that indicate likely friendships.

    Parameters
    ----------
    in_channels  : int   — dimensionality of input node features
    hidden_channels : int — width of the hidden layer
    out_channels : int    — dimensionality of the final node embeddings
    dropout      : float  — dropout probability (regularisation)
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, dropout: float = 0.3):
        super().__init__()

        # First GCN layer: maps raw features → hidden space
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Second GCN layer: maps hidden → final embedding space
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute node embeddings.

        Parameters
        ----------
        x          : (N, in_channels)  – node feature matrix
        edge_index : (2, E)            – graph connectivity (COO format)

        Returns
        -------
        z : (N, out_channels) – learned node embeddings
        """
        # Layer 1: aggregate neighbour features, apply non-linearity
        z = self.conv1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # Layer 2: refine embeddings with a second round of aggregation
        z = self.conv2(z, edge_index)

        return z


# ══════════════════════════════════════════════════════════════════════════
# 4.  LINK PREDICTOR  (Dot-Product Decoder)
# ══════════════════════════════════════════════════════════════════════════
def dot_product_decode(z: torch.Tensor,
                       edge_label_index: torch.Tensor) -> torch.Tensor:
    """
    Predict the probability of a link between pairs of nodes using the
    **dot-product** of their embeddings.

    Intuition
    ─────────
    If two users have similar embedding vectors (i.e. they occupy a similar
    "region" of the learned social space), their dot product will be large
    → high probability of friendship.

    Formally:
        score(u, v) = z_u · z_v  =  Σ_i  z_u[i] * z_v[i]

    We return *raw logits* (no sigmoid) because ``BCEWithLogitsLoss``
    applies sigmoid internally for numerical stability.

    Parameters
    ----------
    z                : (N, d)  – node embeddings from the GCN encoder
    edge_label_index : (2, M)  – pairs of node indices to score

    Returns
    -------
    scores : (M,) – raw logit scores for each candidate edge
    """
    src = z[edge_label_index[0]]  # embeddings of source nodes
    dst = z[edge_label_index[1]]  # embeddings of destination nodes

    # Element-wise product followed by sum → dot product per pair
    scores = (src * dst).sum(dim=-1)

    return scores


# ══════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, optimizer, criterion, train_data):
    """
    Execute one training epoch.

    Steps
    ─────
    1.  Run the GCN encoder on the training graph to get node embeddings.
    2.  Score every supervision edge (both positive friends and negative
        non-friends) using the dot-product decoder.
    3.  Compute BCE loss between predicted scores and ground-truth labels.
    4.  Back-propagate and update model weights.

    Returns
    -------
    loss : float – scalar training loss for this epoch
    """
    model.train()
    optimizer.zero_grad()

    # --- Encode: generate embeddings using ONLY training edges -----------
    z = model(train_data.x, train_data.edge_index)

    # --- Decode: score the supervision edges (pos + neg) ----------------
    logits = dot_product_decode(z, train_data.edge_label_index)

    # --- Loss: Binary Cross Entropy between predictions and labels ------
    loss = criterion(logits, train_data.edge_label)

    # --- Back-prop -------------------------------------------------------
    loss.backward()
    optimizer.step()

    return loss.item()


# ══════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION  (AUC + Average Precision)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, data):
    """
    Evaluate the model on a given split (validation or test).

    Metrics
    ───────
    • **AUC** — Area Under the ROC Curve.
    • **AP**  — Average Precision (area under the Precision-Recall curve).

    Parameters
    ----------
    model : GCNEncoder – the trained GCN
    data  : Data       – split data (val or test) with edge_label_index & edge_label

    Returns
    -------
    auc : float – AUC score on this split
    ap  : float – Average Precision score on this split
    loss : float – BCE loss on this split
    """
    model.eval()

    # Encode nodes using the message-passing edges of this split
    z = model(data.x, data.edge_index)

    # Decode: score the supervision edges
    logits = dot_product_decode(z, data.edge_label_index)

    # Compute loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, data.edge_label).item()

    # Convert logits → probabilities via sigmoid
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.cpu().numpy()

    auc = roc_auc_score(labels, probs)
    ap  = average_precision_score(labels, probs)
    return auc, ap, loss


# ══════════════════════════════════════════════════════════════════════════
# 7.  FRIEND RECOMMENDATION (Inference)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def recommend_friends(model, data, user_id: int, top_k: int = 10):
    """
    Given a user, recommend the top-K most likely new friends.

    How it works
    ────────────
    1.  Compute embeddings for all nodes using the FULL training graph.
    2.  Calculate the dot-product similarity between the target user and
        every other node.
    3.  Exclude nodes that are already friends (existing edges).
    4.  Return the top-K highest-scoring non-friend nodes.

    Parameters
    ----------
    model   : GCNEncoder – trained model
    data    : Data       – the training split (for message-passing edges)
    user_id : int        – index of the target user node
    top_k   : int        – number of recommendations to return

    Returns
    -------
    recommendations : list[tuple[int, float]]
        List of (node_id, score) tuples, sorted by descending score.
    """
    model.eval()

    z = model(data.x, data.edge_index)
    user_emb = z[user_id]  # (d,)

    # Dot-product similarity with every other node
    scores = (z * user_emb).sum(dim=-1)  # (N,)
    scores = torch.sigmoid(scores)        # convert to probability

    # Find existing friends to exclude them
    edge_index = data.edge_index
    # Neighbours of user_id in the graph
    mask = edge_index[0] == user_id
    existing_friends = set(edge_index[1][mask].cpu().tolist())
    existing_friends.add(user_id)  # exclude self

    # Zero out existing friends and self
    for nid in existing_friends:
        scores[nid] = -1.0

    # Get top-K
    top_scores, top_indices = torch.topk(scores, k=top_k)
    recommendations = list(zip(top_indices.cpu().tolist(),
                               top_scores.cpu().tolist()))

    return recommendations


# ══════════════════════════════════════════════════════════════════════════
# 8.  VISUALIZATION — Learning Curves
# ══════════════════════════════════════════════════════════════════════════
def plot_learning_curves(history: dict, save_path: str = "learning_curves.png"):
    """
    Plot training/validation loss and validation metrics (AUC, AP) vs epochs.

    Parameters
    ----------
    history : dict
        Keys: 'train_loss', 'val_loss', 'val_auc', 'val_ap', 'epochs'
    save_path : str
        File path to save the figure.
    """
    epochs = history["epochs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Subplot 1: Loss vs. Epochs ───────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], label="Train Loss",
             color="#e74c3c", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss",
             color="#3498db", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (BCE)", fontsize=12)
    ax1.set_title("Loss vs. Epochs", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: AUC & AP vs. Epochs ───────────────────────────────────
    ax2.plot(epochs, history["val_auc"], label="Val AUC",
             color="#2ecc71", linewidth=2)
    ax2.plot(epochs, history["val_ap"], label="Val AP",
             color="#9b59b6", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Validation Metrics vs. Epochs", fontsize=14, fontweight="bold")
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


def plot_tsne_embeddings(emb_before: np.ndarray, emb_after: np.ndarray,
                         data=None,
                         save_path: str = "tsne_clusters.png"):
    """
    Generate a side-by-side scatter plot comparing untrained vs. trained
    2D t-SNE embeddings to show how the model learns to cluster users.

    The trained-embeddings panel is coloured by **Louvain community** so
    that each social clique appears as a distinct colour island, proving the
    GCN has grouped people into their actual social cliques.

    Parameters
    ----------
    emb_before : (N, d) – node embeddings at Epoch 1 (before learning)
    emb_after  : (N, d) – node embeddings at the final epoch (after learning)
    data       : torch_geometric.data.Data, optional
        Original graph object used to detect Louvain communities.
        If None (or python-louvain is not installed), falls back to a
        per-node rainbow colouring.
    save_path  : str    – file path to save the figure
    """
    print("  🔄  Running t-SNE … (this may take a minute)")

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=15, max_iter=1000)

    proj_before = tsne.fit_transform(emb_before)
    proj_after  = tsne.fit_transform(emb_after)

    # ── Louvain community labels for the trained-embeddings panel ─────────
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
        cluster_labels = np.arange(len(proj_after))  # fallback

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Untrained embeddings (rainbow by node index for contrast) ─────────
    ax1.scatter(proj_before[:, 0], proj_before[:, 1],
                c=np.arange(len(proj_before)), cmap="Spectral",
                s=5, alpha=0.7)
    ax1.set_title("Epoch 1 — Untrained Embeddings", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax1.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax1.grid(True, alpha=0.2)

    # ── Trained embeddings coloured by Louvain community ──────────────────
    scatter = ax2.scatter(proj_after[:, 0], proj_after[:, 1],
                          c=cluster_labels, cmap="tab20",
                          s=10, alpha=0.8)
    title_suffix = (
        f" ({num_communities} Louvain communities)"
        if num_communities > 0 else ""
    )
    ax2.set_title(
        f"Final Epoch — Trained Embeddings{title_suffix}",
        fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax2.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax2.grid(True, alpha=0.2)

    # Add a colour-bar legend for community IDs
    if num_communities > 0:
        cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
        cbar.set_label("Community ID", fontsize=10)

    plt.suptitle(
        "t-SNE Visualization of GCN Node Embeddings\n"
        "(Right panel coloured by Louvain social community)",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  t-SNE embeddings saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 10.  MAIN — PUTTING IT ALL TOGETHER
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Hyper-parameters ─────────────────────────────────────────────────
    HIDDEN_CHANNELS = 128   # Width of the GCN hidden layer
    OUT_CHANNELS    = 64    # Dimensionality of node embeddings
    LEARNING_RATE   = 0.005  # Adam learning rate
    EPOCHS          = 100   # Number of training epochs
    DROPOUT         = 0.7   # Dropout rate for regularisation

    # ── 1. Load data ─────────────────────────────────────────────────────
    data = load_facebook_dataset(root="./data")

    # ── 2. Split edges ───────────────────────────────────────────────────
    train_data, val_data, test_data = prepare_link_prediction_splits(data)

    # Move all splits to the target device
    train_data = train_data.to(DEVICE)
    val_data   = val_data.to(DEVICE)
    test_data  = test_data.to(DEVICE)

    # ── 3. Initialise model, loss, optimiser ─────────────────────────────
    in_channels = train_data.x.size(1)

    model = GCNEncoder(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("=" * 60)
    print("  Step 3 — Model architecture")
    print("=" * 60)
    print(model)
    print(f"\n  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── 4. Capture Epoch-1 (untrained) embeddings for t-SNE ──────────────
    emb_epoch1 = get_embeddings(model, train_data)

    # ── 5. Training ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 4 — Training")
    print("=" * 60)

    best_val_auc = 0.0
    best_epoch   = 0

    # History for plotting
    history = {
        "epochs":     [],
        "train_loss": [],
        "val_loss":   [],
        "val_auc":    [],
        "val_ap":     [],
    }

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, criterion, train_data)

        # Evaluate on validation set every epoch (for smooth curves)
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
            # Save the best model checkpoint
            torch.save(model.state_dict(), "best_model.pt")
            marker = "  ◀ best"

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}"
                  f"  |  Loss: {loss:.4f}"
                  f"  |  Val AUC: {val_auc:.4f}"
                  f"  |  Val AP: {val_ap:.4f}{marker}")

    print(f"\n  Best Val AUC: {best_val_auc:.4f}  (epoch {best_epoch})\n")

    # ── 6. Capture final trained embeddings for t-SNE ────────────────────
    emb_final = get_embeddings(model, train_data)

    # ── 7. Plot learning curves ──────────────────────────────────────────
    plot_learning_curves(history, save_path="learning_curves.png")

    # ── 8. Plot t-SNE embeddings ─────────────────────────────────────────
    plot_tsne_embeddings(emb_epoch1, emb_final, data=data, save_path="tsne_clusters.png")

    # ── 9. Test evaluation ───────────────────────────────────────────────
    print("=" * 60)
    print("  Step 5 — Test evaluation")
    print("=" * 60)

    # Load best model weights
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))

    test_auc, test_ap, test_loss = evaluate(model, test_data)
    print(f"  ✅  Test AUC: {test_auc:.4f}")
    print(f"  ✅  Test AP:  {test_ap:.4f}\n")

    if test_auc >= 0.90:
        print("  🎉  Excellent!  The model discriminates friends from")
        print("      non-friends with high confidence.\n")
    elif test_auc >= 0.75:
        print("  👍  Good performance.  Consider tuning hyper-parameters")
        print("      or using a deeper/wider model for improvement.\n")
    else:
        print("  ⚠️   Performance is below expectations.  Try increasing")
        print("      the number of epochs, adjusting the learning rate,")
        print("      or using a more expressive architecture (e.g. GAT).\n")

    # ── 10. Demo: Friend recommendations ─────────────────────────────────
    print("=" * 60)
    print("  Step 6 — Friend recommendations (demo)")
    print("=" * 60)

    # Pick a random user for the demo
    demo_user = random.randint(0, data.num_nodes - 1)
    recommendations = recommend_friends(model, train_data,
                                        user_id=demo_user, top_k=10)

    print(f"\n  Top 10 friend recommendations for User #{demo_user}:\n")
    print(f"  {'Rank':<6}{'User ID':<12}{'Score':<10}")
    print(f"  {'─' * 6}{'─' * 12}{'─' * 10}")
    for rank, (node_id, score) in enumerate(recommendations, start=1):
        print(f"  {rank:<6}{node_id:<12}{score:<10.4f}")

    print("\n  Done! ✨\n")
