"""
Evaluation metrics for regression and classification tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with MSE, RMSE, MAE, R2
    """
    # Flatten arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUROC)
        num_classes: Number of classes
    
    Returns:
        Dictionary with accuracy, precision, recall, f1, auroc
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    
    # Compute AUROC if probabilities are provided
    if y_prob is not None:
        try:
            if num_classes == 2:
                # Binary classification
                if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    auroc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    auroc = roc_auc_score(y_true, y_prob)
            else:
                # Multi-class classification
                auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            metrics["auroc"] = float(auroc)
        except ValueError as e:
            # AUROC might fail if only one class is present
            metrics["auroc"] = float("nan")
    
    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Get a detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Optional names for the classes
    
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def evaluate_model_regression(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a regression model on a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with (X, y) pairs
        device: Device to run on
    
    Returns:
        Tuple of (metrics_dict, y_true, y_pred)
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    return metrics, y_true, y_pred


def evaluate_model_classification(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 2,
    device: str = "cpu",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a classification model on a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with (X, y) pairs
        num_classes: Number of classes
        device: Device to run on
    
    Returns:
        Tuple of (metrics_dict, y_true, y_pred, y_prob)
    """
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_targets)
    
    # Convert logits to predictions and probabilities
    if num_classes == 2 and logits.shape[1] == 1:
        # Binary with single output
        y_prob = 1 / (1 + np.exp(-logits.flatten()))  # sigmoid
        y_pred = (y_prob > 0.5).astype(int)
    else:
        # Multi-class or binary with 2 outputs
        y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
        y_pred = logits.argmax(axis=1)
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, num_classes)
    
    return metrics, y_true, y_pred, y_prob


if __name__ == "__main__":
    # Quick test
    # Regression
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    print("Regression metrics:", compute_regression_metrics(y_true_reg, y_pred_reg))
    
    # Classification
    y_true_cls = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred_cls = np.array([0, 1, 0, 0, 1, 1, 1, 1])
    y_prob_cls = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.85, 0.95])
    print("Classification metrics:", compute_classification_metrics(y_true_cls, y_pred_cls, y_prob_cls))
