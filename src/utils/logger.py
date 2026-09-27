"""
Standardized logging utilities.
"""
import logging
import sys
import os
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    name: str = "gnn_friend_rec",
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Setup logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        use_colors: Use colored output in console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if use_colors:
        formatter = ColorFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """
    Logger for training metrics with optional W&B / TensorBoard support.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        use_wandb: bool = False,
        wandb_project: str = "gnn-friend-rec",
        wandb_entity: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.wandb_run = None

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=config,
                    dir=str(self.log_dir),
                )
            except ImportError:
                print("wandb not installed, disabling...")
                self.use_wandb = False

        # JSON log file
        self.json_log_path = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.step = 0

    def log(self, metrics: dict, step: Optional[int] = None, prefix: str = ""):
        """Log metrics."""
        if step is not None:
            self.step = step

        # Add prefix to keys
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        prefixed_metrics["step"] = self.step

        # Console log
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in prefixed_metrics.items() if k != "step")
        print(f"[Step {self.step}] {metric_str}")

        # JSON log
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(prefixed_metrics) + "\n")

        # W&B
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log(prefixed_metrics, step=self.step)

    def log_config(self, config: dict):
        """Log configuration."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update(config)

    def finish(self):
        """Finish logging."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()


class ProgressLogger:
    """Simple progress logging for epochs."""

    def __init__(self, total_epochs: int, log_interval: int = 1):
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics."""
        if epoch % self.log_interval != 0 and epoch != 1 and epoch != self.total_epochs:
            return

        elapsed = time.time() - self.start_time
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch:3d}/{self.total_epochs} | {metric_str} | Time: {elapsed:.1f}s")


def get_logger(name: str = "gnn_friend_rec") -> logging.Logger:
    """Get existing logger."""
    return logging.getLogger(name)