"""
Kolmogorov-Arnold Network (KAN) implementation.

This module provides a KAN wrapper that works with the 'pykan' library.
If pykan is not available, it falls back to a simplified efficient-kan style implementation.

KANs replace linear layers with learnable spline-based activation functions,
which can provide better interpretability and sometimes better performance.
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
import warnings
import matplotlib.pyplot as plt


# Try to import pykan, fall back to our own implementation if not available
try:
    from kan import KAN
    PYKAN_AVAILABLE = True
except ImportError:
    PYKAN_AVAILABLE = False
    warnings.warn(
        "pykan not found. Using built-in efficient-kan style implementation. "
        "Install pykan with: pip install pykan"
    )


class BSplineBasis(nn.Module):
    """
    B-spline basis functions for KAN layers.
    
    This implements the learnable univariate functions in KAN using B-splines.
    """
    
    def __init__(
        self,
        num_splines: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1, 1),
    ):
        super().__init__()
        self.num_splines = num_splines
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Create grid points
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)
        
        # Learnable spline coefficients
        self.coefficients = nn.Parameter(
            torch.randn(num_splines, grid_size + spline_order) * 0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions.
        
        Args:
            x: Input tensor of shape (batch, num_splines)
        
        Returns:
            Output tensor of shape (batch, num_splines)
        """
        # Compute basis functions using Cox-de Boor recursion
        # For efficiency, we use a simplified approach
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)  # (batch, num_splines, 1)
        
        grid = self.grid  # (grid_size + 2*k + 1,)
        
        # Initialize order-0 basis
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        # Recursively compute higher-order bases
        for k in range(1, self.spline_order + 1):
            left = (x - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)] + 1e-8)
            right = (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:-k] + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        
        # Apply coefficients
        output = (bases * self.coefficients.unsqueeze(0)).sum(dim=-1)
        return output


class EfficientKANLayer(nn.Module):
    """
    A single KAN layer using efficient B-spline implementation.
    
    This implements: output_j = sum_i phi_{i,j}(input_i)
    where phi_{i,j} are learnable univariate spline functions.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create spline basis for each input-output pair
        self.num_edges = in_features * out_features
        self.splines = BSplineBasis(
            num_splines=self.num_edges,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        
        # Base linear transformation (residual connection)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Scaling factors
        self.scale_spline = nn.Parameter(torch.ones(out_features, in_features))
        self.scale_base = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN layer.
        
        Args:
            x: Input tensor of shape (batch, in_features)
        
        Returns:
            Output tensor of shape (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # Expand input for all edges
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)  # (batch, out, in)
        x_flat = x_expanded.reshape(batch_size, -1)  # (batch, out * in)
        
        # Apply splines
        spline_out = self.splines(x_flat)  # (batch, out * in)
        spline_out = spline_out.reshape(batch_size, self.out_features, self.in_features)
        
        # Apply scaling and sum over input dimension
        spline_out = (spline_out * self.scale_spline.unsqueeze(0)).sum(dim=-1)
        
        # Base linear transformation (residual)
        base_out = torch.nn.functional.linear(x, self.base_weight) * self.scale_base
        
        return spline_out + base_out + self.bias


class EfficientKAN(nn.Module):
    """
    Efficient KAN implementation using B-splines.
    
    This is used when pykan is not available.
    """
    
    def __init__(
        self,
        width: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1, 1),
    ):
        super().__init__()
        self.width = width
        
        layers = []
        for i in range(len(width) - 1):
            layers.append(
                EfficientKANLayer(
                    in_features=width[i],
                    out_features=width[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                )
            )
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KANModel(nn.Module):
    """
    KAN wrapper that provides a unified interface.
    
    Uses pykan if available, otherwise falls back to EfficientKAN.
    
    Args:
        width: List defining the network architecture, e.g., [2, 5, 1]
               means 2 inputs -> 5 hidden -> 1 output
        grid_size: Number of grid intervals for splines
        spline_order: Order of the spline (k), typically 3 for cubic
        grid_range: Range for the spline grid
        device: Device to place the model on
    
    Example:
        >>> model = KANModel(width=[2, 5, 1], grid_size=5, spline_order=3)
        >>> x = torch.randn(32, 2)
        >>> y = model(x)  # shape: (32, 1)
    """
    
    def __init__(
        self,
        width: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1, 1),
        device: str = "cpu",
        use_efficient_kan: bool = False,
    ):
        super().__init__()
        self.width = width
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.device = device
        
        # Determine which implementation to use
        # Priority: 1) Force efficient-kan if requested, 2) Use pykan if available, 3) Fallback to efficient-kan
        if use_efficient_kan:
            self.use_pykan = False
        else:
            self.use_pykan = PYKAN_AVAILABLE
        
        if self.use_pykan:
            # Use official pykan
            self.kan = KAN(
                width=width,
                grid=grid_size,
                k=spline_order,
                device=device,
            )
        else:
            # Use our efficient implementation
            self.kan = EfficientKAN(
                width=width,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the KAN."""
        return self.kan(x)
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def plot(self, save_path: Optional[str] = None, title: Optional[str] = None) -> bool:
        """
        Plot the learned spline functions (pykan only).
        
        Args:
            save_path: Path to save the plot. If None, displays interactively.
            title: Optional title for the plot.
        
        Returns:
            True if plot was successful, False otherwise.
        """
        if not self.use_pykan:
            warnings.warn(
                "Plotting is limited without pykan. "
                "Use plot_kan_splines() from src.eval.plots for basic visualization."
            )
            return False
        
        try:
            # Close any existing figures to avoid overlap
            plt.close('all')
            
            # pykan's plot() creates a figure
            fig = self.kan.plot()
            
            if title:
                plt.suptitle(title, fontsize=14, y=1.02)
            
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close('all')
                print(f"  Saved KAN plot to: {save_path}")
            
            return True
        except Exception as e:
            warnings.warn(f"Could not plot KAN: {e}")
            return False
    
    def prune(self, threshold: float = 1e-2) -> bool:
        """
        Prune the KAN network by removing insignificant edges.
        
        Args:
            threshold: Threshold for pruning (edges with importance below this are removed)
        
        Returns:
            True if pruning was successful, False otherwise.
        """
        if not self.use_pykan:
            warnings.warn("Pruning requires pykan. Skipping.")
            return False
        
        try:
            # pykan's prune method - some versions use positional arg, others use keyword
            try:
                self.kan = self.kan.prune(threshold)  # Try positional first
            except TypeError:
                self.kan = self.kan.prune(threshold=threshold)  # Fallback to keyword
            print(f"  KAN pruned with threshold={threshold}")
            return True
        except Exception as e:
            warnings.warn(f"Could not prune KAN: {e}")
            return False
    
    def auto_symbolic(self, lib: Optional[List[str]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Automatically find symbolic formulas for the learned spline functions.
        
        Args:
            lib: List of symbolic function names to try. 
                 Default: ['sin', 'cos', 'x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'tanh', 'abs']
        
        Returns:
            Tuple of (success, results_dict)
            results_dict contains 'formulas' and any other info from pykan.
        """
        if not self.use_pykan:
            warnings.warn("auto_symbolic requires pykan. Skipping.")
            return False, {"error": "pykan not available"}
        
        if lib is None:
            lib = ['x', 'x^2', 'x^3', 'x^4', 'sin', 'cos', 'exp', 'log', 'sqrt', 'tanh', 'abs']
        
        try:
            # pykan's auto_symbolic tries to fit symbolic functions
            self.kan.auto_symbolic(lib=lib)
            
            # Get the symbolic formula
            formula = self.kan.symbolic_formula()
            
            results = {
                "formula": str(formula) if formula else "No formula found",
                "lib": lib,
            }
            
            print(f"\n  Symbolic formula discovered:")
            print(f"    {results['formula']}")
            
            return True, results
        except Exception as e:
            warnings.warn(f"Could not run auto_symbolic: {e}")
            return False, {"error": str(e)}
    
    def suggest_symbolic(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Let pykan suggest symbolic functions for each edge.
        
        Returns:
            Tuple of (success, suggestions_dict)
        """
        if not self.use_pykan:
            warnings.warn("suggest_symbolic requires pykan. Skipping.")
            return False, {"error": "pykan not available"}
        
        try:
            # pykan can suggest functions
            suggestions = self.kan.suggest_symbolic()
            return True, {"suggestions": suggestions}
        except Exception as e:
            warnings.warn(f"Could not get symbolic suggestions: {e}")
            return False, {"error": str(e)}
    
    def get_spline_weights(self) -> List[torch.Tensor]:
        """
        Get the spline coefficients for visualization.
        
        Returns:
            List of coefficient tensors for each layer.
        """
        if self.use_pykan:
            # Extract from pykan
            try:
                weights = []
                for layer in self.kan.act_fun:
                    weights.append(layer.coef.data.clone())
                return weights
            except:
                return []
        else:
            # Extract from our implementation
            weights = []
            for layer in self.kan.layers:
                weights.append(layer.splines.coefficients.data.clone())
            return weights
    
    def train_pykan(
        self,
        train_input: torch.Tensor,
        train_label: torch.Tensor,
        steps: int = 100,
        lr: float = 0.01,
        lamb: float = 0.01,
    ) -> Dict[str, List[float]]:
        """
        Train using pykan's built-in training (if available).
        
        This is an alternative to the standard PyTorch training loop.
        Useful because pykan's training includes regularization that helps
        with symbolic regression.
        
        Args:
            train_input: Training inputs
            train_label: Training labels
            steps: Number of training steps
            lr: Learning rate
            lamb: Regularization strength
        
        Returns:
            Dictionary with training history
        """
        if not self.use_pykan:
            warnings.warn("train_pykan requires pykan. Use standard training loop.")
            return {"error": "pykan not available"}
        
        try:
            dataset = {
                'train_input': train_input,
                'train_label': train_label,
                'test_input': train_input,  # Use same for simplicity
                'test_label': train_label,
            }
            
            results = self.kan.fit(
                dataset,
                opt="Adam",
                steps=steps,
                lr=lr,
                lamb=lamb,
            )
            
            return {"train_loss": results['train_loss'], "test_loss": results['test_loss']}
        except Exception as e:
            warnings.warn(f"pykan training failed: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Quick test
    print(f"Using pykan: {PYKAN_AVAILABLE}")
    
    model = KANModel(width=[2, 5, 1], grid_size=5, spline_order=3)
    print(f"KAN created with {model.count_parameters()} parameters")
    
    x = torch.randn(8, 2)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
