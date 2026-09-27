"""
Early stopping and learning rate scheduling utilities.
"""
import torch
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    Early stopping with patience and optional learning rate reduction.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",  # 'max' for AUC/AP, 'min' for loss
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' (higher is better) or 'min' (lower is better)
            restore_best_weights: Whether to restore best model weights on stop
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == "max":
            self._is_better = lambda current, best: current > best + min_delta
            self.best_score = -float("inf")
        elif mode == "min":
            self._is_better = lambda current, best: current < best - min_delta
            self.best_score = float("inf")
        else:
            raise ValueError(f"Mode must be 'max' or 'min', got {mode}")

    def __call__(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric
            model: Model to potentially save weights from
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"  EarlyStopping: New best score = {score:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: Triggered at epoch {epoch}. Best was epoch {self.best_epoch} with score {self.best_score:.6f}")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("  EarlyStopping: Restored best weights")

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.early_stop = state["early_stop"]


class LRScheduler:
    """
    Wrapper for ReduceLROnPlateau with consistent interface.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "max",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
        verbose: bool = True,
    ):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=verbose,
        )

    def step(self, metric: float):
        self.scheduler.step(metric)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state)


def create_early_stopping(config: dict) -> EarlyStopping:
    """Create EarlyStopping from config."""
    return EarlyStopping(
        patience=config.get("patience", 10),
        min_delta=config.get("min_delta", 1e-4),
        mode=config.get("mode", "max"),
        restore_best_weights=config.get("restore_best_weights", True),
        verbose=config.get("verbose", True),
    )


def create_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> LRScheduler:
    """Create LRScheduler from config."""
    return LRScheduler(
        optimizer,
        mode=config.get("lr_mode", "max"),
        factor=config.get("lr_factor", 0.5),
        patience=config.get("lr_patience", 5),
        min_lr=config.get("min_lr", 1e-6),
        verbose=config.get("lr_verbose", True),
    )