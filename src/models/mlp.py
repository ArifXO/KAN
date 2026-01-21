"""
Simple Multi-Layer Perceptron (MLP) for fair comparison with KAN.

This is a standard feedforward neural network with ReLU activations.
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron.
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions, e.g., [64, 32]
        output_dim: Number of output features (1 for regression, num_classes for classification)
        dropout: Dropout probability (default: 0.0, meaning no dropout)
    
    Example:
        >>> model = MLP(input_dim=2, hidden_dims=[64, 32], output_dim=1)
        >>> x = torch.randn(32, 2)  # batch of 32 samples, 2 features each
        >>> y = model(x)  # shape: (32, 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation - applied later based on task)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mlp_matching_kan(
    input_dim: int,
    output_dim: int,
    kan_width: List[int],
    kan_grid: int = 5,
    kan_k: int = 3,
) -> MLP:
    """
    Create an MLP with approximately the same number of parameters as a KAN.
    
    KAN parameter count (pykan implementation):
    - Each edge has learnable B-spline coefficients: (grid + k) per edge
    - Plus base weight, scale, and other learnable parameters
    - Empirically: ~14 parameters per edge for grid=5, k=3
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        kan_width: KAN architecture width list, e.g., [2, 5, 1]
        kan_grid: KAN grid size
        kan_k: KAN spline order
        match_mode: "equal" to match params, "more" to give MLP more params (default)
    
    Returns:
        MLP with similar or greater parameter count
    """
    # Estimate KAN parameters more accurately
    # pykan uses: spline_coeffs (grid+k) + base_weight + scale + other internal params
    # Empirical multiplier based on pykan 0.2.x implementation
    kan_params = 0
    for i in range(len(kan_width) - 1):
        edges = kan_width[i] * kan_width[i + 1]
        # More accurate: ~14 params per edge for default settings
        params_per_edge = (kan_grid + kan_k) + 6  # +6 for base, scale, etc.
        kan_params += edges * params_per_edge
    
    # Create MLP with >= parameter count (fair comparison favoring MLP)
    # For simplicity, use 2 hidden layers and solve for hidden size
    
    def mlp_params(h):
        # Layer 1: input_dim -> h (weights + biases)
        p1 = input_dim * h + h
        # Layer 2: h -> h (weights + biases)
        p2 = h * h + h
        # Layer 3: h -> output_dim (weights + biases)
        p3 = h * output_dim + output_dim
        return p1 + p2 + p3
    
    # Find hidden size that gives >= KAN parameter count
    best_h = 8
    for h in range(4, 256):
        if mlp_params(h) >= kan_params:
            best_h = h
            break
    
    return MLP(input_dim=input_dim, hidden_dims=[best_h, best_h], output_dim=output_dim)


if __name__ == "__main__":
    # Quick test
    model = MLP(input_dim=2, hidden_dims=[32, 16], output_dim=1)
    print(f"MLP created with {model.count_parameters()} parameters")
    
    x = torch.randn(8, 2)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
