"""
Utility functions for training.
"""

import os
import random
import csv
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_dirs(base_dir: str = "results"):
    """
    Create necessary directories for results.
    
    Args:
        base_dir: Base directory for results
    """
    dirs = [
        os.path.join(base_dir, "plots"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "checkpoints"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    extra_info: Optional[Dict] = None,
):
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save the checkpoint
        extra_info: Additional information to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to: {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to the checkpoint
        optimizer: Optional optimizer to load state into
        device: Device to load the model on
    
    Returns:
        Checkpoint dictionary with additional info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


class MetricsLogger:
    """
    Simple logger for training metrics to CSV.
    """
    
    def __init__(self, log_path: str, fieldnames: list):
        """
        Initialize the logger.
        
        Args:
            log_path: Path to the CSV log file
            fieldnames: List of column names
        """
        self.log_path = log_path
        self.fieldnames = fieldnames
        
        # Create directory and file with header
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def log(self, metrics: Dict[str, Any]):
        """
        Log a row of metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)
    
    def log_batch(self, metrics_list: list):
        """
        Log multiple rows of metrics.
        
        Args:
            metrics_list: List of metric dictionaries
        """
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for metrics in metrics_list:
                writer.writerow(metrics)


def get_experiment_name(prefix: str = "exp") -> str:
    """
    Generate a unique experiment name with timestamp.
    
    Args:
        prefix: Prefix for the experiment name
    
    Returns:
        Experiment name string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print a summary of the model.
    
    Args:
        model: PyTorch model
        model_name: Name to display
    """
    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"{model_name} Summary")
    print(f"{'='*50}")
    print(f"Total trainable parameters: {n_params:,}")
    print(f"{'='*50}\n")
