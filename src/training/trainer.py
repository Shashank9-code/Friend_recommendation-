"""
Generic training engine for GNN link prediction.
Supports BCEWithLogitsLoss, BPR loss, and custom losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, Optional, Callable, Any, Tuple
from tqdm import tqdm
import time

from src.models.base import GNNModel
from src.training.early_stopping import EarlyStopping, LRScheduler


class BPR_Loss(nn.Module):
    """
    Bayesian Personalized Ranking loss for implicit feedback.
    Loss = -sum(log(sigmoid(pos_score - neg_score)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_scores: Positive edge scores (B,)
            neg_scores: Negative edge scores (B,) or (B, K) for K negatives
        """
        if neg_scores.dim() > 1:
            # Multiple negatives per positive: average
            neg_scores = neg_scores.mean(dim=1)

        diff = pos_scores - neg_scores
        loss = -F.logsigmoid(diff).mean()
        return loss


class LinkPredictionTrainer:
    """
    Generic trainer for link prediction with GNNs.
    """

    def __init__(
        self,
        model: GNNModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: Optional[nn.Module] = None,
        early_stopping: Optional[EarlyStopping] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        grad_clip: float = 1.0,
        log_interval: int = 10,
    ):
        """
        Args:
            model: GNNModel (encoder + predictor)
            optimizer: PyTorch optimizer
            device: torch.device
            loss_fn: Loss function (default: BCEWithLogitsLoss)
            early_stopping: EarlyStopping instance
            lr_scheduler: LRScheduler instance
            grad_clip: Gradient clipping max norm
            log_interval: Log every N epochs
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        self.model.to(device)
        self.history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_ap": [],
            "val_precision@10": [],
            "val_recall@10": [],
            "val_ndcg@10": [],
            "val_hit@10": [],
        }
        self.best_val_auc = 0.0
        self.best_epoch = 0

    def train_epoch(self, train_data: Data) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(
            train_data.x.to(self.device),
            train_data.edge_index.to(self.device),
            train_data.edge_label_index.to(self.device),
        )

        # Compute loss
        labels = train_data.edge_label.to(self.device)
        loss = self.loss_fn(logits, labels)

        # Backward
        loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    def train_epoch_bpr(self, train_data: Data) -> float:
        """Train for one epoch using BPR loss (requires triplet sampling)."""
        self.model.train()
        self.optimizer.zero_grad()

        # Get positive edges
        pos_mask = train_data.edge_label == 1
        pos_edge_index = train_data.edge_label_index[:, pos_mask]

        # Get negative edges
        neg_mask = train_data.edge_label == 0
        neg_edge_index = train_data.edge_label_index[:, neg_mask]

        # Encode
        z = self.model.encode(
            train_data.x.to(self.device),
            train_data.edge_index.to(self.device),
        )

        # Score positive and negative edges
        pos_scores = self.model.predict(
            z, pos_edge_index.to(self.device)
        )
        neg_scores = self.model.predict(
            z, neg_edge_index.to(self.device)
        )

        # BPR loss
        loss = self.loss_fn(pos_scores, neg_scores)

        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, data: Data, metrics_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Evaluate model on a data split.

        Args:
            data: Data split (val or test)
            metrics_fn: Function that computes metrics from (probs, labels)

        Returns:
            Dict of metrics
        """
        from src.evaluation.metrics import compute_all_metrics

        self.model.eval()

        logits = self.model(
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.edge_label_index.to(self.device),
        )

        labels = data.edge_label.to(self.device)

        # Loss
        loss = self.loss_fn(logits, labels).item()

        # Metrics
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()

        if metrics_fn is not None:
            metrics = metrics_fn(probs, labels_np)
        else:
            metrics = compute_all_metrics(probs, labels_np)

        metrics["loss"] = loss
        return metrics

    def fit(
        self,
        train_data: Data,
        val_data: Data,
        epochs: int = 100,
        use_bpr: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Maximum epochs
            use_bpr: Use BPR loss instead of BCE
            checkpoint_path: Path to save best model

        Returns:
            Training history
        """
        train_data = train_data.to(self.device)
        val_data = val_data.to(self.device)

        print(f"Starting training on {self.device}...")
        print(f"Model: {self.model.encoder.__class__.__name__} + {self.model.predictor.__class__.__name__}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Training
            if use_bpr:
                train_loss = self.train_epoch_bpr(train_data)
            else:
                train_loss = self.train_epoch(train_data)

            # Validation
            val_metrics = self.evaluate(val_data)

            # Update history
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            for k, v in val_metrics.items():
                if k in self.history:
                    self.history[k].append(v)

            # Logging
            if epoch % self.log_interval == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val AUC: {val_metrics.get('auc', 0):.4f} | "
                      f"Val AP: {val_metrics.get('ap', 0):.4f} | "
                      f"LR: {lr:.2e}")

            # Early stopping
            if self.early_stopping is not None:
                val_auc = val_metrics.get("auc", 0)
                if self.early_stopping(val_auc, self.model, epoch):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # LR scheduling
            if self.lr_scheduler is not None:
                val_auc = val_metrics.get("auc", 0)
                self.lr_scheduler.step(val_auc)

            # Checkpoint best model
            val_auc = val_metrics.get("auc", 0)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                if checkpoint_path:
                    torch.save(self.model.state_dict(), checkpoint_path)

        print(f"\nTraining complete. Best Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")

        # Load best weights if early stopping restored
        if self.early_stopping and self.early_stopping.best_weights:
            self.model.load_state_dict(self.early_stopping.best_weights)

        return self.history

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        return checkpoint


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    opt_type = config.get("optimizer", "adam").lower()
    lr = config.get("lr", 0.005)
    weight_decay = config.get("weight_decay", 0.0)

    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def create_loss_fn(config: dict) -> nn.Module:
    """Create loss function from config."""
    loss_type = config.get("loss", "bce").lower()

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "bpr":
        return BPR_Loss()
    elif loss_type == "focal":
        # Focal loss for imbalanced data
        return FocalLoss(
            alpha=config.get("focal_alpha", 0.25),
            gamma=config.get("focal_gamma", 2.0),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()