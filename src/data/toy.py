"""
Toy function dataset for 2D regression experiments.

This module provides synthetic data generators for testing function approximation.
The functions are designed to visualize how KANs learn and represent functions.
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Callable
import numpy as np


def toy_function(x: torch.Tensor) -> torch.Tensor:
    """
    Default toy function: f(x1, x2) = sin(3 * x1) + x2^2
    
    This function is designed to test KAN's symbolic regression capabilities.
    The sin(3*x1) and x2^2 terms should be discoverable via auto_symbolic().
    
    Args:
        x: Input tensor of shape (batch, 2)
    
    Returns:
        Output tensor of shape (batch, 1)
    """
    x1, x2 = x[:, 0], x[:, 1]
    y = torch.sin(3 * x1) + x2 ** 2
    return y.unsqueeze(-1)


def toy_function_sincos(x: torch.Tensor) -> torch.Tensor:
    """
    Alternative toy function: f(x1, x2) = sin(pi * x1) + cos(pi * x2)
    
    This function is smooth and periodic, good for testing spline approximation.
    """
    x1, x2 = x[:, 0], x[:, 1]
    y = torch.sin(np.pi * x1) + torch.cos(np.pi * x2)
    return y.unsqueeze(-1)


def toy_function_complex(x: torch.Tensor) -> torch.Tensor:
    """
    More complex toy function: f(x1, x2) = sin(pi * x1) * exp(-x2^2) + x1 * x2
    
    This function has multiplicative interactions, testing KAN's ability
    to decompose complex functions.
    """
    x1, x2 = x[:, 0], x[:, 1]
    y = torch.sin(np.pi * x1) * torch.exp(-x2**2) + x1 * x2
    return y.unsqueeze(-1)


def toy_function_polynomial(x: torch.Tensor) -> torch.Tensor:
    """
    Polynomial toy function: f(x1, x2) = x1^2 + x2^2 + x1*x2
    
    Simple polynomial to test basic function learning.
    """
    x1, x2 = x[:, 0], x[:, 1]
    y = x1**2 + x2**2 + x1 * x2
    return y.unsqueeze(-1)


# Dictionary of available toy functions
TOY_FUNCTIONS = {
    "default": toy_function,           # sin(3*x1) + x2^2 (for symbolic regression)
    "sincos": toy_function_sincos,     # sin(pi*x1) + cos(pi*x2)
    "complex": toy_function_complex,
    "polynomial": toy_function_polynomial,
}


def generate_toy_data(
    n_samples: int,
    function_name: str = "sincos",
    x_range: Tuple[float, float] = (-1, 1),
    noise_std: float = 0.0,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic 2D regression data.
    
    Args:
        n_samples: Number of samples to generate
        function_name: Name of the toy function ('sincos', 'complex', 'polynomial')
        x_range: Range for input features
        noise_std: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X, y) tensors
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate uniform random inputs
    X = torch.rand(n_samples, 2) * (x_range[1] - x_range[0]) + x_range[0]
    
    # Get the function
    if function_name not in TOY_FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}. Available: {list(TOY_FUNCTIONS.keys())}")
    
    func = TOY_FUNCTIONS[function_name]
    y = func(X)
    
    # Add noise if specified
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    
    return X.float(), y.float()


def generate_grid_data(
    grid_size: int = 50,
    function_name: str = "sincos",
    x_range: Tuple[float, float] = (-1, 1),
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Generate a regular grid of points for visualization.
    
    Args:
        grid_size: Number of points per dimension
        function_name: Name of the toy function
        x_range: Range for input features
    
    Returns:
        Tuple of (X, y, x1_grid, x2_grid) where x1_grid, x2_grid are meshgrid arrays
    """
    x1 = np.linspace(x_range[0], x_range[1], grid_size)
    x2 = np.linspace(x_range[0], x_range[1], grid_size)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    
    X = torch.tensor(
        np.stack([x1_grid.flatten(), x2_grid.flatten()], axis=1),
        dtype=torch.float32
    )
    
    func = TOY_FUNCTIONS.get(function_name, toy_function)
    y = func(X)
    
    return X, y, x1_grid, x2_grid


def get_toy_dataloaders(
    n_train: int = 1000,
    n_val: int = 200,
    function_name: str = "sincos",
    x_range: Tuple[float, float] = (-1, 1),
    noise_std: float = 0.0,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for the toy regression task.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        function_name: Name of the toy function
        x_range: Range for input features
        noise_std: Standard deviation of noise
        batch_size: Batch size for DataLoaders
        seed: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Generate training data
    X_train, y_train = generate_toy_data(
        n_samples=n_train,
        function_name=function_name,
        x_range=x_range,
        noise_std=noise_std,
        seed=seed,
    )
    
    # Generate validation data with different seed
    X_val, y_val = generate_toy_data(
        n_samples=n_val,
        function_name=function_name,
        x_range=x_range,
        noise_std=noise_std,
        seed=seed + 1,  # Different seed for validation
    )
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    X, y = generate_toy_data(100, function_name="sincos")
    print(f"Generated data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    
    train_loader, val_loader = get_toy_dataloaders(n_train=100, n_val=20, batch_size=16)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
