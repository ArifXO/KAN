"""
Plotting utilities for visualization of results.

This module provides functions for plotting loss curves, predictions,
confusion matrices, and KAN spline visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Optional, Dict, Tuple
import torch


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training and Validation Loss",
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the figure (None = display only)
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to log scale if losses vary significantly
    if max(train_losses) / (min(train_losses) + 1e-8) > 100:
        ax.set_yscale("log")
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved loss curve to: {save_path}")
    
    return fig


def plot_predictions_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Predictions vs Ground Truth",
) -> plt.Figure:
    """
    Create a scatter plot of predictions vs ground truth.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save the figure
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", s=30)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")
    
    ax.set_xlabel("Ground Truth", fontsize=12)
    ax.set_ylabel("Prediction", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved scatter plot to: {save_path}")
    
    return fig


def plot_predictions_heatmap(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_size: int = 50,
    save_path: Optional[str] = None,
    title: str = "Prediction Heatmaps",
) -> plt.Figure:
    """
    Create heatmaps comparing ground truth and predictions for 2D input.
    
    Args:
        X: Input features of shape (n_samples, 2)
        y_true: Ground truth values
        y_pred: Predicted values
        grid_size: Size of the grid for plotting
        save_path: Path to save the figure
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Create regular grid for visualization
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    x1_grid = np.linspace(x1_min, x1_max, grid_size)
    x2_grid = np.linspace(x2_min, x2_max, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Interpolate values onto grid (using nearest neighbor for simplicity)
    from scipy.interpolate import griddata
    
    Y_true_grid = griddata(X, y_true, (X1, X2), method="cubic", fill_value=np.nan)
    Y_pred_grid = griddata(X, y_pred, (X1, X2), method="cubic", fill_value=np.nan)
    
    # Common color scale
    vmin = min(np.nanmin(Y_true_grid), np.nanmin(Y_pred_grid))
    vmax = max(np.nanmax(Y_true_grid), np.nanmax(Y_pred_grid))
    
    # Ground truth heatmap
    im1 = axes[0].imshow(
        Y_true_grid, extent=[x1_min, x1_max, x2_min, x2_max],
        origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto"
    )
    axes[0].set_title("Ground Truth", fontsize=12)
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    plt.colorbar(im1, ax=axes[0])
    
    # Prediction heatmap
    im2 = axes[1].imshow(
        Y_pred_grid, extent=[x1_min, x1_max, x2_min, x2_max],
        origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto"
    )
    axes[1].set_title("Prediction", fontsize=12)
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.colorbar(im2, ax=axes[1])
    
    # Error heatmap
    error_grid = np.abs(Y_true_grid - Y_pred_grid)
    im3 = axes[2].imshow(
        error_grid, extent=[x1_min, x1_max, x2_min, x2_max],
        origin="lower", cmap="Reds", aspect="auto"
    )
    axes[2].set_title("Absolute Error", fontsize=12)
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional names for the classes
        save_path: Path to save the figure
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    
    # Rotate x labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12
            )
    
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to: {save_path}")
    
    return fig


def plot_kan_splines(
    model,
    save_path: Optional[str] = None,
    title: str = "KAN Learned Spline Functions",
    x_range: Tuple[float, float] = (-1, 1),
    n_points: int = 100,
) -> plt.Figure:
    """
    Visualize the learned spline functions in a KAN model.
    
    This works with our EfficientKAN implementation. For pykan,
    use the model.plot() method instead.
    
    Args:
        model: KANModel instance
        save_path: Path to save the figure
        title: Plot title
        x_range: Range of x values to plot
        n_points: Number of points to evaluate
    
    Returns:
        matplotlib Figure object
    """
    # Check if it's using pykan
    if hasattr(model, "use_pykan") and model.use_pykan:
        print("For pykan models, use model.plot() for detailed spline visualization.")
        # Try to use pykan's built-in plotting
        try:
            if save_path:
                model.plot(save_path=save_path)
            else:
                model.plot()
            return None
        except:
            pass
    
    # For our EfficientKAN, visualize the spline weights
    model.eval()
    
    # Get the internal KAN
    if hasattr(model, "kan"):
        kan = model.kan
    else:
        kan = model
    
    # Check if it has layers attribute (EfficientKAN)
    if not hasattr(kan, "layers"):
        print("Cannot visualize splines for this model type.")
        return None
    
    n_layers = len(kan.layers)
    
    # Create figure with subplots for each layer
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    x = torch.linspace(x_range[0], x_range[1], n_points)
    
    for layer_idx, layer in enumerate(kan.layers):
        ax = axes[layer_idx]
        
        in_dim = layer.in_features
        out_dim = layer.out_features
        
        # Evaluate splines for each input dimension
        colors = plt.cm.tab10(np.linspace(0, 1, in_dim))
        
        with torch.no_grad():
            for i in range(min(in_dim, 5)):  # Plot at most 5 input dimensions
                # Create input that varies only in dimension i
                x_input = torch.zeros(n_points, in_dim)
                x_input[:, i] = x
                
                # Get output (sum over output dimensions for visualization)
                y = layer(x_input).mean(dim=1)
                
                ax.plot(x.numpy(), y.numpy(), color=colors[i], 
                       label=f"Input {i}", linewidth=2, alpha=0.7)
        
        ax.set_xlabel("Input Value", fontsize=10)
        ax.set_ylabel("Spline Output (mean)", fontsize=10)
        ax.set_title(f"Layer {layer_idx + 1}\n({in_dim} → {out_dim})", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.05)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved spline plot to: {save_path}")
    
    return fig


def plot_comparison_bar(
    metrics_kan: Dict[str, float],
    metrics_mlp: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "KAN vs MLP Comparison",
) -> plt.Figure:
    """
    Create a bar chart comparing KAN and MLP metrics.
    
    Args:
        metrics_kan: Dictionary of KAN metrics
        metrics_mlp: Dictionary of MLP metrics
        save_path: Path to save the figure
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    # Find common metrics
    common_metrics = set(metrics_kan.keys()) & set(metrics_mlp.keys())
    metrics = sorted(list(common_metrics))
    
    kan_values = [metrics_kan[m] for m in metrics]
    mlp_values = [metrics_mlp[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, kan_values, width, label="KAN", color="steelblue")
    bars2 = ax.bar(x + width/2, mlp_values, width, label="MLP", color="coral")
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9
            )
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    # Test loss curve
    train_losses = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]
    val_losses = [1.1, 0.6, 0.35, 0.25, 0.2, 0.15]
    plot_loss_curve(train_losses, val_losses, save_path=None)
    
    # Test scatter plot
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.2
    plot_predictions_scatter(y_true, y_pred, save_path=None)
    
    plt.show()
