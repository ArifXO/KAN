"""
Training script for the tabular classification experiment.

This script trains both KAN and MLP models on the breast cancer dataset
and compares their classification performance.

Usage:
    python -m src.train.train_tabular --config configs/tabular.yaml
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import softmax as scipy_softmax

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.kan import KANModel, PYKAN_AVAILABLE
from src.models.mlp import MLP
from src.data.tabular import get_tabular_dataloaders, get_dataset_info
from src.eval.metrics import compute_classification_metrics, get_confusion_matrix, get_classification_report
from src.eval.plots import plot_loss_curve, plot_confusion_matrix, plot_comparison_bar
from src.train.utils import (
    set_seed,
    get_device,
    load_config,
    ensure_dirs,
    save_checkpoint,
    MetricsLogger,
    print_model_summary,
    get_experiment_name,
    count_parameters,
)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        num_classes: Number of classes
        device: Device to run on
    
    Returns:
        Tuple of (average loss, y_true, y_pred, y_prob)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_targets)
    
    # Convert logits to predictions and probabilities
    # Use scipy's numerically stable softmax to avoid overflow with large logits
    y_prob = scipy_softmax(logits, axis=1)
    y_pred = logits.argmax(axis=1)
    
    return total_loss / n_batches, y_true, y_pred, y_prob


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int,
    config: Dict,
    device: str,
    model_name: str,
    results_dir: str,
) -> Tuple[nn.Module, List[float], List[float], Dict]:
    """
    Full training loop for a model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        config: Configuration dictionary
        device: Device to train on
        model_name: Name of the model (for logging)
        results_dir: Directory to save results
    
    Returns:
        Tuple of (trained model, train_losses, val_losses, best_metrics)
    """
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Optional learning rate scheduler
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
    
    # Setup logging
    log_path = os.path.join(results_dir, "logs", f"{model_name}_training.csv")
    logger = MetricsLogger(
        log_path,
        fieldnames=["epoch", "train_loss", "val_loss", "accuracy", "auroc", "precision", "recall", "f1"]
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_metrics = {}
    best_y_true = None
    best_y_pred = None
    
    n_epochs = config["n_epochs"]
    
    print(f"\nTraining {model_name} for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, y_true, y_pred, y_prob = validate(
            model, val_loader, criterion, num_classes, device
        )
        
        # Compute metrics
        metrics = compute_classification_metrics(y_true, y_pred, y_prob, num_classes)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Log
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics
        }
        logger.log(log_entry)
        
        # Print progress
        if epoch % config.get("print_every", 10) == 0 or epoch == 1:
            auroc_str = f"{metrics.get('auroc', 0):.4f}" if not np.isnan(metrics.get('auroc', float('nan'))) else "N/A"
            print(f"  Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Acc: {metrics['accuracy']:.4f} | AUROC: {auroc_str}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics.copy()
            best_metrics["val_loss"] = val_loss
            best_y_true = y_true
            best_y_pred = y_pred
            
            checkpoint_path = os.path.join(results_dir, "checkpoints", f"{model_name}_best.pt")
            save_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_path,
                extra_info={"metrics": metrics}
            )
    
    auroc_str = f"{best_metrics.get('auroc', 0):.4f}" if not np.isnan(best_metrics.get('auroc', float('nan'))) else "N/A"
    print(f"  Best Val Loss: {best_val_loss:.6f} | Best Acc: {best_metrics['accuracy']:.4f} | AUROC: {auroc_str}")
    
    # Store predictions for later plotting
    best_metrics["_y_true"] = best_y_true
    best_metrics["_y_pred"] = best_y_pred
    
    return model, train_losses, val_losses, best_metrics


def main():
    """Main function to run the tabular classification experiment."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train KAN and MLP on tabular classification")
    parser.add_argument("--config", type=str, default="configs/tabular.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--efficient-kan", action="store_true", dest="use_efficient_kan",
                        help="Use efficient-kan instead of pykan (faster, but no interpretability features)")
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override with command line args
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    
    # Setup
    set_seed(config["seed"])
    device = config.get("device", get_device())
    print(f"Using device: {device}")
    
    # Create results directory
    exp_name = get_experiment_name("tabular")
    results_dir = os.path.join("results", exp_name)
    ensure_dirs(results_dir)
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\nLoading data...")
    dataset_name = config["dataset_name"]
    dataset_info = get_dataset_info(dataset_name)
    print(f"  Dataset: {dataset_name}")
    print(f"  Samples: {dataset_info['n_samples']}")
    print(f"  Features: {dataset_info['n_features']}")
    print(f"  Classes: {dataset_info['n_classes']}")
    
    train_loader, val_loader, n_features, n_classes = get_tabular_dataloaders(
        dataset_name=dataset_name,
        test_size=config["test_size"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        normalize=config.get("normalize", True),
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Create models
    print("\nCreating models...")
    
    # KAN model
    kan_hidden = config.get("kan_hidden", [16])
    kan_width = [n_features] + kan_hidden + [n_classes]
    kan_width_copy = list(kan_width)  # Copy because pykan modifies the list in place
    
    use_efficient = args.use_efficient_kan or config.get("use_efficient_kan", False)
    
    kan_model = KANModel(
        width=kan_width_copy,
        grid_size=config["kan_grid"],
        spline_order=config["kan_k"],
        device=device,
        use_efficient_kan=use_efficient,
    )
    kan_params = count_parameters(kan_model)
    print_model_summary(kan_model, "KAN")
    print(f"  Using pykan: {kan_model.use_pykan}")
    
    # MLP model (similar parameter count)
    # Calculate hidden size to match KAN parameters
    mlp_hidden = config.get("mlp_hidden", None)
    if mlp_hidden is None:
        # Auto-calculate to match KAN parameters
        # Rough formula: n_features * h + h + h * h + h + h * n_classes + n_classes ≈ kan_params
        target = kan_params
        best_h = 16
        for h in range(8, 256):
            p = n_features * h + h + h * h + h + h * n_classes + n_classes
            if p >= target:
                best_h = h
                break
        mlp_hidden = [best_h, best_h]
    
    mlp_model = MLP(
        input_dim=n_features,
        hidden_dims=mlp_hidden,
        output_dim=n_classes,
        dropout=config.get("dropout", 0.0),
    )
    print_model_summary(mlp_model, "MLP")
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- KAN BEFORE TRAINING PLOT ---
    if PYKAN_AVAILABLE:
        print("\nPlotting KAN structure BEFORE training...")
        # pykan requires a forward pass before plotting (to compute activations)
        with torch.no_grad():
            sample_x, _ = next(iter(train_loader))
            sample_x = sample_x.to(device)
            _ = kan_model(sample_x)
        kan_model.plot(
            save_path=os.path.join(plots_dir, "kan_before.png"),
            title="KAN Before Training"
        )
    
    # Train KAN
    print("\n" + "="*60)
    print("Training KAN")
    print("="*60)
    kan_model, kan_train_losses, kan_val_losses, kan_metrics = train_model(
        kan_model, train_loader, val_loader, n_classes, config, device, "kan", results_dir
    )
    
    # Train MLP
    print("\n" + "="*60)
    print("Training MLP")
    print("="*60)
    mlp_model, mlp_train_losses, mlp_val_losses, mlp_metrics = train_model(
        mlp_model, train_loader, val_loader, n_classes, config, device, "mlp", results_dir
    )
    
    # --- KAN INTERPRETABILITY ANALYSIS ---
    if PYKAN_AVAILABLE:
        print("\n" + "="*60)
        print("KAN INTERPRETABILITY ANALYSIS")
        print("="*60)
        
        # Plot trained KAN structure
        print("\n[1/3] Plotting trained KAN structure...")
        kan_model.plot(
            save_path=os.path.join(plots_dir, "kan_after.png"),
            title="KAN After Training (Learned Splines)"
        )
        
        # Prune the network
        print("\n[2/3] Pruning KAN network...")
        prune_threshold = config.get("prune_threshold", 0.01)
        if kan_model.prune(threshold=prune_threshold):
            kan_model.plot(
                save_path=os.path.join(plots_dir, "kan_pruned.png"),
                title="KAN After Pruning"
            )
        
        # Symbolic regression (optional for classification)
        if config.get("enable_symbolic", False):
            print("\n[3/3] Running symbolic regression (auto_symbolic)...")
            success, symbolic_result = kan_model.auto_symbolic()
            if success and symbolic_result.get("formula"):
                symbolic_path = os.path.join(results_dir, "logs", "symbolic.txt")
                with open(symbolic_path, "w") as f:
                    f.write(f"Symbolic Formula:\n{symbolic_result['formula']}\n")
                print(f"  Saved symbolic results to: {symbolic_path}")
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating comparison plots...")
    print("="*60)
    
    # Loss curves
    plot_loss_curve(
        kan_train_losses, kan_val_losses,
        save_path=os.path.join(plots_dir, "kan_loss_curve.png"),
        title="KAN Training Loss"
    )
    plot_loss_curve(
        mlp_train_losses, mlp_val_losses,
        save_path=os.path.join(plots_dir, "mlp_loss_curve.png"),
        title="MLP Training Loss"
    )
    
    # Confusion matrices
    class_names = dataset_info.get("target_names", None)
    
    plot_confusion_matrix(
        kan_metrics["_y_true"], kan_metrics["_y_pred"],
        class_names=class_names,
        save_path=os.path.join(plots_dir, "kan_confusion_matrix.png"),
        title="KAN Confusion Matrix"
    )
    plot_confusion_matrix(
        mlp_metrics["_y_true"], mlp_metrics["_y_pred"],
        class_names=class_names,
        save_path=os.path.join(plots_dir, "mlp_confusion_matrix.png"),
        title="MLP Confusion Matrix"
    )
    
    # Remove internal keys before comparison
    kan_metrics_clean = {k: v for k, v in kan_metrics.items() if not k.startswith("_")}
    mlp_metrics_clean = {k: v for k, v in mlp_metrics.items() if not k.startswith("_")}
    
    # Comparison bar chart
    plot_comparison_bar(
        kan_metrics_clean, mlp_metrics_clean,
        save_path=os.path.join(plots_dir, "comparison.png"),
        title="KAN vs MLP Classification Comparison"
    )
    
    # Print classification reports
    print("\n" + "="*60)
    print("Classification Reports")
    print("="*60)
    
    print("\nKAN Classification Report:")
    print(get_classification_report(kan_metrics["_y_true"], kan_metrics["_y_pred"], target_names=class_names))
    
    print("\nMLP Classification Report:")
    print(get_classification_report(mlp_metrics["_y_true"], mlp_metrics["_y_pred"], target_names=class_names))
    
    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print("\nFinal Metrics:")
    kan_auroc = kan_metrics.get('auroc', float('nan'))
    mlp_auroc = mlp_metrics.get('auroc', float('nan'))
    kan_auroc_str = f"{kan_auroc:.4f}" if not np.isnan(kan_auroc) else "N/A"
    mlp_auroc_str = f"{mlp_auroc:.4f}" if not np.isnan(mlp_auroc) else "N/A"
    print(f"  KAN - Accuracy: {kan_metrics['accuracy']:.4f}, AUROC: {kan_auroc_str}")
    print(f"  MLP - Accuracy: {mlp_metrics['accuracy']:.4f}, AUROC: {mlp_auroc_str}")
    print(f"\nParameter counts:")
    print(f"  KAN: {kan_params:,}")
    print(f"  MLP: {count_parameters(mlp_model):,}")
    print("\nOutput Files:")
    print(f"  plots/")
    print(f"    - kan_loss_curve.png, mlp_loss_curve.png")
    print(f"    - kan_confusion_matrix.png, mlp_confusion_matrix.png")
    print(f"    - comparison.png")
    if PYKAN_AVAILABLE:
        print(f"    - kan_before.png (initial splines)")
        print(f"    - kan_after.png (learned splines)")
        print(f"    - kan_pruned.png (pruned network)")
    print(f"  logs/")
    print(f"    - kan_training.csv, mlp_training.csv")
    print(f"  checkpoints/")
    print(f"    - kan_best.pt, mlp_best.pt")


if __name__ == "__main__":
    main()
