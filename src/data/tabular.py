"""
Tabular dataset loaders for classification experiments.

This module provides loaders for sklearn's built-in datasets,
specifically the breast cancer dataset for binary classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Optional
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Dictionary of available datasets
DATASETS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
}


class TabularDataset(Dataset):
    """
    A simple dataset wrapper for tabular data.
    
    Args:
        X: Feature tensor of shape (n_samples, n_features)
        y: Target tensor of shape (n_samples,)
    """
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_tabular_data(
    dataset_name: str = "breast_cancer",
    test_size: float = 0.2,
    seed: int = 42,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Load and split a tabular dataset.
    
    Args:
        dataset_name: Name of the dataset ('breast_cancer', 'iris', 'wine')
        test_size: Fraction of data to use for validation
        seed: Random seed for reproducibility
        normalize: Whether to standardize features
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, n_features, n_classes)
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    # Load dataset
    data = DATASETS[dataset_name]()
    X, y = data.data, data.target
    
    # Get metadata
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_train, y_val, n_features, n_classes


def get_tabular_dataloaders(
    dataset_name: str = "breast_cancer",
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create train and validation DataLoaders for tabular classification.
    
    Args:
        dataset_name: Name of the dataset
        test_size: Fraction of data for validation
        batch_size: Batch size
        seed: Random seed
        normalize: Whether to standardize features
    
    Returns:
        Tuple of (train_loader, val_loader, n_features, n_classes)
    """
    # Load and split data
    X_train, X_val, y_train, y_val, n_features, n_classes = load_tabular_data(
        dataset_name=dataset_name,
        test_size=test_size,
        seed=seed,
        normalize=normalize,
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, n_features, n_classes


def get_dataset_info(dataset_name: str = "breast_cancer") -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with dataset information
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data = DATASETS[dataset_name]()
    
    return {
        "name": dataset_name,
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "n_classes": len(np.unique(data.target)),
        "feature_names": list(data.feature_names) if hasattr(data, "feature_names") else None,
        "target_names": list(data.target_names) if hasattr(data, "target_names") else None,
    }


if __name__ == "__main__":
    # Quick test
    info = get_dataset_info("breast_cancer")
    print(f"Dataset info: {info}")
    
    train_loader, val_loader, n_features, n_classes = get_tabular_dataloaders(
        dataset_name="breast_cancer",
        batch_size=32,
    )
    
    print(f"\nDataLoader info:")
    print(f"  n_features: {n_features}")
    print(f"  n_classes: {n_classes}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Check one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shapes: X={X_batch.shape}, y={y_batch.shape}")
