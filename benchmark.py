"""
Multi-Seed & Multi-Split Benchmark Runner for Link Prediction.

Runs all classical heuristic baselines (CN, AA, JC, PA, RA) and neural
architectures (GCN, GAT, GIN, LightGCN) across multiple random seeds,
aggregates results (Mean ± Std), and writes a Markdown comparison table
to results/benchmark_summary.md.

Usage:
    python benchmark.py
    python benchmark.py --seeds 42 123 456 --models gcn gat
    python benchmark.py --dataset facebook --epochs 50
"""
import os
import sys
import time
import argparse
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx, to_undirected

from src.data.dataset import EgoNetworkDataset
from src.evaluation.baselines import compute_heuristic_scores
from src.evaluation.metrics import (
    compute_auc_ap,
    compute_ranking_metrics,
    compute_mrr,
)
from src.models.encoders import create_encoder
from src.models.predictors import create_predictor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [42, 123, 456, 789, 999]
DEFAULT_HEURISTICS = [
    "common_neighbors",
    "adamic_adar",
    "jaccard",
    "resource_allocation",
    "preferential_attachment",
]
DEFAULT_NEURAL_MODELS = ["gcn", "gat", "gin", "lightgcn"]
HEURISTIC_DISPLAY = {
    "common_neighbors": "Common Neighbors (CN)",
    "adamic_adar": "Adamic-Adar (AA)",
    "jaccard": "Jaccard Coefficient (JC)",
    "resource_allocation": "Resource Allocation (RA)",
    "preferential_attachment": "Preferential Attach. (PA)",
}
NEURAL_DISPLAY = {
    "gcn": "GCN",
    "gat": "GAT",
    "gin": "GIN",
    "lightgcn": "LightGCN",
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(
    data: Data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[Data, Data, Data]:
    """Create train/val/test edge splits with negative sampling."""
    set_seed(seed)
    splitter = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        split_labels=False,
    )
    return splitter(data)


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """Evaluate a score vector against binary labels."""
    # Ensure float arrays
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    metrics = {}

    # AUC / AP
    try:
        auc_ap = compute_auc_ap(scores, labels)
        metrics["AUC"] = auc_ap["auc"]
        metrics["AP"] = auc_ap["ap"]
    except Exception:
        metrics["AUC"] = 0.0
        metrics["AP"] = 0.0

    # Ranking metrics
    ranking = compute_ranking_metrics(scores, labels, ks)
    for k in ks:
        metrics[f"Recall@{k}"] = ranking.get(f"recall@{k}", 0.0)
        metrics[f"NDCG@{k}"] = ranking.get(f"ndcg@{k}", 0.0)

    # MRR
    metrics["MRR"] = compute_mrr(scores, labels)

    return metrics


# ─────────────────────────────────────────────────────────────────────
# Heuristic Baseline Runner
# ─────────────────────────────────────────────────────────────────────
def run_heuristic(
    G,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    heuristic: str,
) -> Dict[str, float]:
    """Run a single heuristic and return evaluation metrics."""
    scores = compute_heuristic_scores(G, edge_label_index, heuristic)

    # Normalize scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)

    labels = edge_label.cpu().numpy()
    return evaluate_scores(scores, labels)


# ─────────────────────────────────────────────────────────────────────
# Neural Model Runner
# ─────────────────────────────────────────────────────────────────────
def train_and_evaluate_neural(
    train_data: Data,
    val_data: Data,
    test_data: Data,
    model_type: str,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.005,
    hidden: int = 128,
    out_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 10,
) -> Dict[str, float]:
    """Train a GNN encoder + dot-product predictor and evaluate on test set."""
    in_channels = train_data.x.size(1) if train_data.x is not None else train_data.num_nodes

    # Handle identity features for models that need them
    if train_data.x is None:
        train_data.x = torch.eye(train_data.num_nodes)
        val_data.x = torch.eye(val_data.num_nodes)
        test_data.x = torch.eye(test_data.num_nodes)
        in_channels = train_data.num_nodes

    # LightGCN uses nn.Embedding: needs integer node indices, not float features
    if model_type == "lightgcn":
        in_channels = train_data.num_nodes
        train_data = train_data.clone()
        val_data = val_data.clone()
        test_data = test_data.clone()
        train_data.x = torch.arange(train_data.num_nodes, dtype=torch.long)
        val_data.x = torch.arange(val_data.num_nodes, dtype=torch.long)
        test_data.x = torch.arange(test_data.num_nodes, dtype=torch.long)

    encoder = create_encoder(
        model_type=model_type,
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=out_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    predictor = create_predictor("dot", in_channels=out_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )

    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    best_val_auc = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        encoder.train()
        predictor.train()
        optimizer.zero_grad()

        z = encoder(train_data.x, train_data.edge_index)
        logits = predictor(z, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), 1.0
        )
        optimizer.step()

        # ── Validate ──
        if epoch % 5 == 0 or epoch == 1:
            encoder.eval()
            predictor.eval()
            with torch.no_grad():
                z = encoder(val_data.x, val_data.edge_index)
                val_logits = predictor(z, val_data.edge_label_index)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_labels = val_data.edge_label.cpu().numpy()
                try:
                    val_auc = compute_auc_ap(val_probs, val_labels)["auc"]
                except Exception:
                    val_auc = 0.0

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {
                    "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                    "predictor": {k: v.cpu().clone() for k, v in predictor.state_dict().items()},
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    # ── Test ──
    if best_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder"].items()})
        predictor.load_state_dict({k: v.to(device) for k, v in best_state["predictor"].items()})

    encoder.eval()
    predictor.eval()
    with torch.no_grad():
        z = encoder(test_data.x, test_data.edge_index)
        test_logits = predictor(z, test_data.edge_label_index)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_labels = test_data.edge_label.cpu().numpy()

    return evaluate_scores(test_probs, test_labels)


# ─────────────────────────────────────────────────────────────────────
# Aggregation & Markdown Output
# ─────────────────────────────────────────────────────────────────────
def aggregate_results(
    all_results: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Dict[str, str]]:
    """Aggregate per-seed results into Mean ± Std strings."""
    summary = {}
    for model_name, seed_results in all_results.items():
        metric_values = defaultdict(list)
        for result in seed_results:
            for metric, value in result.items():
                metric_values[metric].append(value)

        summary[model_name] = {}
        for metric, values in metric_values.items():
            mean = np.mean(values)
            std = np.std(values)
            summary[model_name][metric] = f"{mean:.4f} ± {std:.4f}"
    return summary


def write_markdown_table(
    summary: Dict[str, Dict[str, str]],
    output_path: str,
    seeds: List[int],
    dataset_name: str,
):
    """Write benchmark results as a Markdown comparison table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define column order
    columns = ["AUC", "AP", "Recall@10", "NDCG@10", "MRR"]

    lines = []
    lines.append(f"# Benchmark Summary: {dataset_name.upper()} Dataset")
    lines.append("")
    lines.append(f"**Seeds**: {seeds}")
    lines.append(f"**Split**: 85% train / 5% val / 10% test")
    lines.append(f"**Negative Sampling Ratio**: 1:1")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Classical Baselines ──
    lines.append("## Classical Heuristic Baselines")
    lines.append("")
    header = f"| {'Method':<28} | " + " | ".join(f"{c:>16}" for c in columns) + " |"
    separator = f"|{'-'*30}|" + "|".join(f"{'-'*18}" for _ in columns) + "|"
    lines.append(header)
    lines.append(separator)

    for h_key, h_display in HEURISTIC_DISPLAY.items():
        if h_key in summary:
            row = f"| {h_display:<28} | "
            row += " | ".join(f"{summary[h_key].get(c, 'N/A'):>16}" for c in columns)
            row += " |"
            lines.append(row)

    lines.append("")

    # ── Neural Models ──
    lines.append("## Neural Architectures")
    lines.append("")
    lines.append(header)
    lines.append(separator)

    for m_key, m_display in NEURAL_DISPLAY.items():
        if m_key in summary:
            row = f"| {m_display:<28} | "
            row += " | ".join(f"{summary[m_key].get(c, 'N/A'):>16}" for c in columns)
            row += " |"
            lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Values shown as Mean ± Std across seeds.*")

    content = "\n".join(lines) + "\n"

    with open(output_path, "w") as f:
        f.write(content)

    print(f"\n✅ Benchmark summary written to: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Main Benchmark Pipeline
# ─────────────────────────────────────────────────────────────────────
def run_benchmark(args):
    """Execute the full benchmark pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Dataset: {args.dataset}")
    print(f"Heuristics: {args.heuristics}")
    print(f"Neural models: {args.models}")
    print("=" * 70)

    # Collect results: model_name -> list of per-seed metric dicts
    all_results: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    # Load base dataset once
    ds = EgoNetworkDataset(root=args.data_root, name=args.dataset)
    base_data = ds.load()

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'='*70}")
        print(f"  SEED {seed} ({seed_idx + 1}/{len(args.seeds)})")
        print(f"{'='*70}")

        set_seed(seed)
        train_data, val_data, test_data = make_splits(
            base_data,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=seed,
        )

        # ── Heuristic Baselines ──
        G = to_networkx(train_data, to_undirected=True)

        for h_name in args.heuristics:
            print(f"  [{h_name}] ... ", end="", flush=True)
            t0 = time.time()
            metrics = run_heuristic(
                G, test_data.edge_label_index, test_data.edge_label, h_name
            )
            elapsed = time.time() - t0
            print(f"AUC={metrics['AUC']:.4f}  AP={metrics['AP']:.4f}  "
                  f"R@10={metrics['Recall@10']:.4f}  ({elapsed:.1f}s)")
            all_results[h_name].append(metrics)

        # ── Neural Models ──
        for m_name in args.models:
            print(f"  [{m_name.upper()}] training ... ", end="", flush=True)
            t0 = time.time()
            try:
                metrics = train_and_evaluate_neural(
                    train_data, val_data, test_data,
                    model_type=m_name,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden=args.hidden,
                    out_dim=args.out_dim,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    patience=args.patience,
                )
                elapsed = time.time() - t0
                print(f"AUC={metrics['AUC']:.4f}  AP={metrics['AP']:.4f}  "
                      f"R@10={metrics['Recall@10']:.4f}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                metrics = {k: 0.0 for k in ["AUC", "AP", "Recall@10", "NDCG@10", "MRR"]}
            all_results[m_name].append(metrics)

    # ── Aggregate and Write ──
    print("\n" + "=" * 70)
    print("  AGGREGATING RESULTS")
    print("=" * 70)

    summary = aggregate_results(all_results)

    # Print to console
    columns = ["AUC", "AP", "Recall@10", "NDCG@10", "MRR"]
    print(f"\n{'Method':<28} | " + " | ".join(f"{c:>16}" for c in columns))
    print("-" * 120)
    for name in list(HEURISTIC_DISPLAY.keys()) + list(NEURAL_DISPLAY.keys()):
        if name in summary:
            display = HEURISTIC_DISPLAY.get(name, NEURAL_DISPLAY.get(name, name))
            row = f"{display:<28} | "
            row += " | ".join(f"{summary[name].get(c, 'N/A'):>16}" for c in columns)
            print(row)

    # Write Markdown
    write_markdown_table(summary, args.output, args.seeds, args.dataset)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Benchmark Runner for Link Prediction"
    )
    parser.add_argument(
        "--dataset", type=str, default="facebook",
        choices=["facebook", "twitter", "gplus"],
        help="Dataset name",
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random seeds for reproducibility",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--heuristics", type=str, nargs="+", default=DEFAULT_HEURISTICS,
        help="Heuristic baselines to evaluate",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=DEFAULT_NEURAL_MODELS,
        help="Neural architectures to train and evaluate",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--output", type=str, default="results/benchmark_summary.md",
        help="Output path for Markdown results table",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
