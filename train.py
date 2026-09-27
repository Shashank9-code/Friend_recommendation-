#!/usr/bin/env python
"""
Main training entry point for GNN Friend Recommendation System.

Usage:
    python train.py --config configs/default.yaml --model gcn
    python train.py --config configs/default.yaml --model gat --baseline true
    python train.py --config configs/default.yaml --model gcn --epochs 200 --lr 0.01
"""
import argparse
import sys
import os
import random
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config, get_default_config, validate_config, merge_configs, Config
from src.utils.logger import setup_logger, MetricLogger
from src.data import (
    EgoNetworkDataset,
    OGBLinkPredictionDataset,
    load_dataset,
    FeatureEngineer,
)
from src.models import (
    create_encoder,
    create_predictor,
    GNNModel,
    count_parameters,
)
from src.training import (
    LinkPredictionTrainer,
    create_optimizer,
    create_loss_fn,
    create_early_stopping,
    create_lr_scheduler,
)
from src.evaluation import (
    compute_all_metrics,
    run_baseline_comparison,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GNN Friend Recommendation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )

    # Model selection (overrides config)
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["gcn", "gat", "gatv2", "gin", "graphconv", "lightgcn", "sage"],
        help="Model type (overrides config)",
    )

    # Quick overrides
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--hidden", type=int, help="Hidden channels")
    parser.add_argument("--out", type=int, help="Output channels")
    parser.add_argument("--layers", type=int, help="Number of layers")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--heads", type=int, help="GAT attention heads")
    parser.add_argument("--residual", action="store_true", help="Enable residual connections")
    parser.add_argument("--layer-norm", action="store_true", help="Enable LayerNorm")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Dataset
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["facebook", "twitter", "gplus", "ogbl-collab", "ogbl-ppa", "ogbl-ddi"],
        help="Dataset name",
    )

    # Features
    parser.add_argument(
        "--features",
        type=str,
        choices=["structural", "identity", "rwpe", "laplacian_pe", "learnable"],
        help="Feature type",
    )

    # Predictor
    parser.add_argument(
        "--predictor", "-p",
        type=str,
        choices=["dot", "cosine", "bilinear", "mlp", "hadamard_mlp"],
        help="Link predictor type",
    )

    # Baseline
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run classical baselines for comparison",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        help="Path to save best model checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )

    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply command line overrides to config."""
    # Model overrides
    if args.model:
        config.model.type = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.lr = args.lr
    if args.hidden:
        config.model.hidden_channels = args.hidden
    if args.out:
        config.model.out_channels = args.out
    if args.layers:
        config.model.num_layers = args.layers
    if args.dropout:
        config.model.dropout = args.dropout
    if args.heads:
        config.model.heads = args.heads
    if args.residual:
        config.model.use_residual = True
    if args.layer_norm:
        config.model.use_layer_norm = True
    if args.seed:
        config.seed = args.seed

    # Dataset
    if args.dataset:
        config.dataset.name = args.dataset

    # Features
    if args.features:
        config.features.type = args.features

    # Predictor
    if args.predictor:
        config.predictor.type = args.predictor

    # Logging
    if args.wandb:
        config.logging.use_wandb = True
    if args.log_level:
        config.logging.level = args.log_level

    # Device
    if args.device:
        config.device = args.device

    return config


def get_device(device_config: str) -> torch.device:
    """Get torch device from config."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def main():
    args = parse_args()

    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = get_default_config()

    # Apply CLI overrides
    config = apply_overrides(config, args)

    # Validate
    validate_config(config)

    # Setup seed
    set_seed(config.seed)

    # Setup device
    device = get_device(config.device)
    print(f"Using device: {device}")

    # Setup logger
    logger = setup_logger(level=config.logging.level)

    # Dry run
    if args.dry_run:
        print("Config:")
        import yaml
        print(yaml.dump(config.to_dict(), default_flow_style=False))
        return

    # Setup metric logger
    metric_logger = MetricLogger(
        log_dir=config.logging.log_dir,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        wandb_entity=config.logging.wandb_entity,
        config=config.to_dict(),
    )

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    full_data, (train_data, val_data, test_data) = load_dataset(config.to_dict())

    # Feature engineering
    print("\n" + "=" * 60)
    print("Feature engineering...")
    print("=" * 60)

    feature_engineer = FeatureEngineer(
        features=config.features.structural,
        rwpe_length=config.features.rwpe_length,
        laplacian_k=config.features.laplacian_k,
    )

    # Apply features to all splits
    full_data = feature_engineer(full_data)
    train_data = feature_engineer(train_data)
    val_data = feature_engineer(val_data)
    test_data = feature_engineer(test_data)

    in_channels = full_data.x.size(1)
    print(f"Input feature dimension: {in_channels}")

    # Create model
    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)

    encoder = create_encoder(
        model_type=config.model.type,
        in_channels=in_channels,
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        use_residual=config.model.use_residual,
        use_layer_norm=config.model.use_layer_norm,
        heads=config.model.heads,
        attn_dropout=config.model.attn_dropout,
        aggr=config.model.aggr,
    )

    predictor = create_predictor(
        predictor_type=config.predictor.type,
        in_channels=config.model.out_channels,
        hidden_channels=config.predictor.hidden_channels,
        num_layers=config.predictor.num_layers,
        dropout=config.predictor.dropout,
    )

    model = GNNModel(encoder, predictor)
    model.to(device)

    param_counts = count_parameters(model)
    print(f"Model: {encoder.__class__.__name__} + {predictor.__class__.__name__}")
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")

    # Setup training
    optimizer = create_optimizer(model, config.training.to_dict())
    loss_fn = create_loss_fn(config.training.to_dict())

    early_stopping = None
    if config.early_stopping.enabled:
        early_stopping = create_early_stopping(config.early_stopping.to_dict())

    lr_scheduler = None
    if config.lr_scheduler.enabled:
        lr_scheduler = create_lr_scheduler(optimizer, config.lr_scheduler.to_dict())

    # Checkpoint path
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif config.checkpoint.save_dir:
        Path(config.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(Path(config.checkpoint.save_dir) / f"best_{config.model.type}.pt")

    trainer = LinkPredictionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        grad_clip=config.training.grad_clip,
        log_interval=config.training.log_interval,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)

    # Training
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=config.training.epochs,
        use_bpr=config.training.loss == "bpr",
        checkpoint_path=checkpoint_path,
    )

    # Log final validation metrics
    val_metrics = {k: v[-1] for k, v in history.items() if k != "epochs" and v}
    metric_logger.log(val_metrics, step=trainer.best_epoch, prefix="val_")

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test evaluation...")
    print("=" * 60)

    # Load best model
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_metrics = trainer.evaluate(test_data)
    print(f"Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    metric_logger.log(test_metrics, step=trainer.best_epoch, prefix="test_")

    # Run baselines if requested
    if args.baseline or config.evaluation.run_baselines:
        print("\n" + "=" * 60)
        print("Running classical baselines...")
        print("=" * 60)
        baseline_results = run_baseline_comparison(train_data, val_data, test_data)
        metric_logger.log(
            {f"baseline_{k}_{m}": v for k, res in baseline_results["test"].items() for m, v in res.items()},
            step=trainer.best_epoch,
        )

    # Demo recommendations
    print("\n" + "=" * 60)
    print("Friend recommendation demo...")
    print("=" * 60)

    demo_user = random.randint(0, full_data.num_nodes - 1)
    recommendations = model.recommend(
        full_data.x.to(device),
        full_data.edge_index.to(device),
        user_id=demo_user,
        top_k=10,
    )

    print(f"\nTop 10 recommendations for User #{demo_user}:")
    print(f"  {'Rank':<6} {'User ID':<12} {'Score':<10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10}")
    for rank, node_id in enumerate(recommendations.cpu().tolist(), 1):
        # Score computation for display
        with torch.no_grad():
            z = model.encode(full_data.x.to(device), full_data.edge_index.to(device))
            user_emb = z[demo_user]
            score = torch.sigmoid((z[node_id] * user_emb).sum()).item()
        print(f"  {rank:<6} {node_id:<12} {score:<10.4f}")

    metric_logger.finish()
    print("\nDone! ✨")


if __name__ == "__main__":
    main()