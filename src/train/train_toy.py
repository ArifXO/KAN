"""
Training script for the toy function regression experiment.

This script trains both KAN and MLP models on a synthetic 2D function
and compares their performance. It also demonstrates KAN interpretability
features from pykan: plot(), prune(), auto_symbolic().

Target function: y = sin(3*x1) + x2^2

Usage:
    python -m src.train.train_toy --config configs/toy.yaml
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.kan import KANModel, PYKAN_AVAILABLE
from src.models.mlp import MLP, create_mlp_matching_kan
from src.data.toy import get_toy_dataloaders, generate_toy_data, TOY_FUNCTIONS
from src.eval.metrics import compute_regression_metrics
from src.eval.plots import (
    plot_loss_curve,
    plot_predictions_scatter,
    plot_predictions_heatmap,
    plot_kan_splines,
    plot_comparison_bar,
)
from src.train.utils import (
    set_seed,
    get_device,
    load_config,
    ensure_dirs,
    save_checkpoint,
    MetricsLogger,
    print_model_summary,
    get_experiment_name,
)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    return total_loss / n_batches, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict,
    device: str,
    model_name: str,
    results_dir: str,
) -> Tuple[nn.Module, List[float], List[float], Dict]:
    """Full training loop for a model."""
    model = model.to(device)
    
    # Setup training
    criterion = nn.MSELoss()
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
        fieldnames=["epoch", "train_loss", "val_loss", "mse", "rmse", "mae", "r2"]
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_metrics = {}
    
    n_epochs = config["n_epochs"]
    
    print(f"\nTraining {model_name.upper()} for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, y_true, y_pred = validate(model, val_loader, criterion, device)
        
        # Compute metrics
        metrics = compute_regression_metrics(y_true, y_pred)
        
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
            print(f"  Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | R²: {metrics['r2']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics.copy()
            best_metrics["val_loss"] = val_loss
            
            checkpoint_path = os.path.join(results_dir, "checkpoints", f"{model_name}_best.pt")
            save_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_path,
                extra_info={"metrics": metrics}
            )
    
    print(f"  Best Val Loss: {best_val_loss:.6f} | Best R²: {best_metrics['r2']:.4f}")
    
    return model, train_losses, val_losses, best_metrics


def save_metrics(metrics_dict: Dict, results_dir: str):
    """Save metrics to CSV and JSON files."""
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(logs_dir, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved metrics to: {json_path}")
    
    # Save as CSV
    csv_path = os.path.join(logs_dir, "metrics.csv")
    with open(csv_path, 'w') as f:
        # Header
        f.write("model,mse,rmse,mae,r2,val_loss\n")
        # KAN row
        f.write(f"kan,{metrics_dict['kan']['mse']:.6f},{metrics_dict['kan']['rmse']:.6f},"
                f"{metrics_dict['kan']['mae']:.6f},{metrics_dict['kan']['r2']:.6f},"
                f"{metrics_dict['kan']['val_loss']:.6f}\n")
        # MLP row
        f.write(f"mlp,{metrics_dict['mlp']['mse']:.6f},{metrics_dict['mlp']['rmse']:.6f},"
                f"{metrics_dict['mlp']['mae']:.6f},{metrics_dict['mlp']['r2']:.6f},"
                f"{metrics_dict['mlp']['val_loss']:.6f}\n")
    print(f"  Saved metrics to: {csv_path}")


def save_symbolic_results(results: Dict, results_dir: str):
    """Save symbolic regression results to a text file."""
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    txt_path = os.path.join(logs_dir, "symbolic.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("KAN SYMBOLIC REGRESSION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Target function: y = sin(3*x1) + x2^2\n\n")
        
        if results.get("success", False):
            f.write("Discovered formula:\n")
            f.write(f"  {results.get('formula', 'N/A')}\n\n")
            f.write(f"Library used: {results.get('lib', 'N/A')}\n")
        else:
            f.write(f"Symbolic regression failed or skipped.\n")
            f.write(f"Reason: {results.get('error', 'Unknown')}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"  Saved symbolic results to: {txt_path}")


def plot_ground_truth_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
    title: str = "Ground Truth: y = sin(3*x1) + x2^2"
):
    """Plot ground truth function as a heatmap."""
    from scipy.interpolate import griddata
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create grid
    grid_size = 50
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    x1_grid = np.linspace(x1_min, x1_max, grid_size)
    x2_grid = np.linspace(x2_min, x2_max, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    y_flat = y.flatten()
    Y_grid = griddata(X, y_flat, (X1, X2), method='cubic', fill_value=np.nan)
    
    im = ax.imshow(
        Y_grid, extent=[x1_min, x1_max, x2_min, x2_max],
        origin='lower', cmap='viridis', aspect='auto'
    )
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_prediction_heatmap(
    X: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Model Predictions"
):
    """Plot model predictions as a heatmap."""
    from scipy.interpolate import griddata
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create grid
    grid_size = 50
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    x1_grid = np.linspace(x1_min, x1_max, grid_size)
    x2_grid = np.linspace(x2_min, x2_max, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    y_flat = y_pred.flatten()
    Y_grid = griddata(X, y_flat, (X1, X2), method='cubic', fill_value=np.nan)
    
    im = ax.imshow(
        Y_grid, extent=[x1_min, x1_max, x2_min, x2_max],
        origin='lower', cmap='viridis', aspect='auto'
    )
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_kan_interpretability(
    kan_model: KANModel,
    train_loader: torch.utils.data.DataLoader,
    config: Dict,
    results_dir: str,
    device: str,
):
    """
    Run KAN interpretability analysis using pykan features.
    
    This includes:
    - Plotting before training (already done in main)
    - Plotting after training
    - Pruning the network
    - Symbolic regression (auto_symbolic)
    """
    plots_dir = os.path.join(results_dir, "plots")
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    if not PYKAN_AVAILABLE:
        print("\n⚠️  pykan not installed. Skipping advanced interpretability features.")
        print("   Install with: pip install pykan")
        
        # Still try basic spline visualization
        plot_kan_splines(
            kan_model,
            save_path=os.path.join(plots_dir, "kan_splines.png"),
            title="KAN Learned Spline Functions (fallback)"
        )
        
        save_symbolic_results(
            {"success": False, "error": "pykan not installed"},
            results_dir
        )
        return
    
    print("\n" + "=" * 60)
    print("KAN INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    
    # --- Plot after training ---
    print("\n[1/3] Plotting trained KAN structure...")
    kan_model.plot(
        save_path=os.path.join(plots_dir, "kan_after.png"),
        title="KAN After Training"
    )
    
    # --- Pruning ---
    enable_prune = config.get("enable_prune", True)
    if enable_prune:
        print("\n[2/3] Pruning KAN network...")
        prune_threshold = config.get("prune_threshold", 1e-2)
        success = kan_model.prune(threshold=prune_threshold)
        
        if success:
            kan_model.plot(
                save_path=os.path.join(plots_dir, "kan_pruned.png"),
                title="KAN After Pruning"
            )
    else:
        print("\n[2/3] Pruning disabled in config (enable_prune: false)")
    
    # # --- Symbolic Regression ---
    # enable_symbolic = config.get("enable_symbolic", True)
    # if enable_symbolic:
    #     print("\n[3/3] Running symbolic regression (auto_symbolic)...")
        
    #     # Get training data for symbolic fitting
    #     X_train = train_loader.dataset.tensors[0].to(device)
    #     y_train = train_loader.dataset.tensors[1].to(device)
        
    #     # Update grid for better symbolic fitting
    #     try:
    #         kan_model.kan.update_grid_from_samples(X_train)
    #     except Exception as e:
    #         print(f"    Could not update grid: {e}")
        
    #     # Run auto_symbolic
    #     success, results = kan_model.auto_symbolic()
        
    #     results["success"] = success
    #     save_symbolic_results(results, results_dir)
        
    #     # Print to console
    #     print("\n  " + "-" * 40)
    #     print("  SYMBOLIC REGRESSION RESULT:")
    #     print("  " + "-" * 40)
    #     print(f"  Target:     y = sin(3*x1) + x2^2")
    #     if success:
    #         print(f"  Discovered: {results.get('formula', 'N/A')}")
    #     else:
    #         print(f"  Failed: {results.get('error', 'Unknown error')}")
    #     print("  " + "-" * 40)
    # else:
    #     print("\n[3/3] Symbolic regression disabled in config (enable_symbolic: false)")
    #     save_symbolic_results(
    #         {"success": False, "error": "Disabled in config"},
    #         results_dir
    #     )


def main():
    """Main function to run the toy experiment."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train KAN and MLP on toy function regression")
    parser.add_argument("--config", type=str, default="configs/toy.yaml", help="Path to config file")
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
    seed = config["seed"]
    set_seed(seed)
    device = config.get("device", get_device())
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")
    
    # Create timestamped results directory
    exp_name = get_experiment_name("toy")
    results_dir = os.path.join("results", exp_name)
    ensure_dirs(results_dir)
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    function_name = config.get("function_name", "default")
    print(f"Target function: {function_name}")
    if function_name == "default":
        print("  y = sin(3*x1) + x2^2")
    
    train_loader, val_loader = get_toy_dataloaders(
        n_train=config["n_train"],
        n_val=config["n_val"],
        function_name=function_name,
        x_range=tuple(config["x_range"]),
        noise_std=config["noise_std"],
        batch_size=config["batch_size"],
        seed=seed,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create models
    print("\n" + "=" * 60)
    print("CREATING MODELS")
    print("=" * 60)
    
    # KAN model
    kan_width = config["kan_width"]  # e.g., [2, 5, 1]
    kan_width_copy = list(kan_width)  # Copy because pykan modifies the list in place
    
    use_efficient = args.use_efficient_kan or config.get("use_efficient_kan", False)
    
    kan_model = KANModel(
        width=kan_width_copy,
        grid_size=config["kan_grid"],
        spline_order=config["kan_k"],
        device=device,
        use_efficient_kan=use_efficient,
    )
    print_model_summary(kan_model, "KAN")
    print(f"  Using pykan: {kan_model.use_pykan}")
    
    # MLP model (matching parameter count) - use original kan_width
    mlp_model = create_mlp_matching_kan(
        input_dim=kan_width[0],
        output_dim=kan_width[-1],
        kan_width=kan_width,
        kan_grid=config["kan_grid"],
        kan_k=config["kan_k"],
    )
    print_model_summary(mlp_model, "MLP")
    
    # --- KAN BEFORE TRAINING PLOT ---
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if PYKAN_AVAILABLE:
        print("\nPlotting KAN structure BEFORE training...")
        # pykan requires a forward pass before plotting (to compute activations)
        # Run one batch through the model to initialize internal states
        with torch.no_grad():
            sample_x, _ = next(iter(train_loader))
            sample_x = sample_x.to(device)
            _ = kan_model(sample_x)
        
        kan_model.plot(
            save_path=os.path.join(plots_dir, "kan_before.png"),
            title="KAN Before Training"
        )
    
    # Train KAN
    print("\n" + "=" * 60)
    print("TRAINING KAN")
    print("=" * 60)
    kan_model, kan_train_losses, kan_val_losses, kan_metrics = train_model(
        kan_model, train_loader, val_loader, config, device, "kan", results_dir
    )
    
    # Train MLP
    print("\n" + "=" * 60)
    print("TRAINING MLP")
    print("=" * 60)
    mlp_model, mlp_train_losses, mlp_val_losses, mlp_metrics = train_model(
        mlp_model, train_loader, val_loader, config, device, "mlp", results_dir
    )
    
    # Save combined metrics
    print("\n" + "=" * 60)
    print("SAVING METRICS")
    print("=" * 60)
    all_metrics = {
        "kan": kan_metrics,
        "mlp": mlp_metrics,
        "config": {
            "function": function_name,
            "n_train": config["n_train"],
            "n_val": config["n_val"],
            "kan_width": kan_width,
            "n_epochs": config["n_epochs"],
        }
    }
    save_metrics(all_metrics, results_dir)
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    # Loss curves (combined)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(kan_train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(kan_val_losses, 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('KAN Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(mlp_train_losses, 'b-', label='Train', linewidth=2)
    axes[1].plot(mlp_val_losses, 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('MLP Loss Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_curve_path = os.path.join(plots_dir, "loss_curve.png")
    fig.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {loss_curve_path}")
    
    # Get validation data for plotting
    X_val = val_loader.dataset.tensors[0]
    y_val = val_loader.dataset.tensors[1]
    X_val_np = X_val.numpy()
    y_val_np = y_val.numpy()
    
    kan_model.eval()
    mlp_model.eval()
    
    with torch.no_grad():
        kan_pred = kan_model(X_val.to(device)).cpu().numpy()
        mlp_pred = mlp_model(X_val.to(device)).cpu().numpy()
    
    # Scatter plot: predictions vs ground truth
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(y_val_np.flatten(), kan_pred.flatten(), alpha=0.5, s=20)
    axes[0].plot([y_val_np.min(), y_val_np.max()], [y_val_np.min(), y_val_np.max()], 'r--', lw=2)
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Prediction')
    axes[0].set_title(f'KAN: R² = {kan_metrics["r2"]:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_val_np.flatten(), mlp_pred.flatten(), alpha=0.5, s=20)
    axes[1].plot([y_val_np.min(), y_val_np.max()], [y_val_np.min(), y_val_np.max()], 'r--', lw=2)
    axes[1].set_xlabel('Ground Truth')
    axes[1].set_ylabel('Prediction')
    axes[1].set_title(f'MLP: R² = {mlp_metrics["r2"]:.4f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(plots_dir, "pred_vs_gt_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {scatter_path}")
    
    # Heatmaps
    plot_ground_truth_heatmap(
        X_val_np, y_val_np,
        save_path=os.path.join(plots_dir, "heatmap_gt.png"),
        title="Ground Truth: y = sin(3*x1) + x2²"
    )
    
    plot_prediction_heatmap(
        X_val_np, kan_pred,
        save_path=os.path.join(plots_dir, "heatmap_pred_kan.png"),
        title="KAN Predictions"
    )
    
    plot_prediction_heatmap(
        X_val_np, mlp_pred,
        save_path=os.path.join(plots_dir, "heatmap_pred_mlp.png"),
        title="MLP Predictions"
    )
    
    # --- KAN INTERPRETABILITY ---
    run_kan_interpretability(
        kan_model, train_loader, config, results_dir, device
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print("\nFinal Metrics (Validation Set):")
    print(f"  +{'-'*50}+")
    print(f"  | {'Model':<8} {'MSE':<12} {'RMSE':<12} {'R2':<12} |")
    print(f"  +{'-'*50}+")
    print(f"  | {'KAN':<8} {kan_metrics['mse']:<12.6f} {kan_metrics['rmse']:<12.6f} {kan_metrics['r2']:<12.4f} |")
    print(f"  | {'MLP':<8} {mlp_metrics['mse']:<12.6f} {mlp_metrics['rmse']:<12.6f} {mlp_metrics['r2']:<12.4f} |")
    print(f"  +{'-'*50}+")
    
    print("\nOutput Files:")
    print(f"  plots/")
    print(f"    - loss_curve.png")
    print(f"    - pred_vs_gt_scatter.png")
    print(f"    - heatmap_gt.png")
    print(f"    - heatmap_pred_kan.png")
    print(f"    - heatmap_pred_mlp.png")
    if PYKAN_AVAILABLE:
        print(f"    - kan_before.png")
        print(f"    - kan_after.png")
        print(f"    - kan_pruned.png")
    print(f"  logs/")
    print(f"    - metrics.json")
    print(f"    - metrics.csv")
    print(f"    - kan_training.csv")
    print(f"    - mlp_training.csv")
    print(f"    - symbolic.txt")
    print(f"  checkpoints/")
    print(f"    - kan_best.pt")
    print(f"    - mlp_best.pt")
    

if __name__ == "__main__":
    main()
