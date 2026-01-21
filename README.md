# KAN vs MLP Baseline Experiments

Minimal, reproducible benchmarks comparing **Kolmogorov-Arnold Networks (KANs)** with **Multi-Layer Perceptrons (MLPs)**.

## Key Results

### Toy Regression: `y = sin(3*x1) + x2²`

| Implementation | Model | Parameters | MSE | R² |
|----------------|-------|------------|-----|-----|
| **Pykan** | KAN | 210 | 0.000128 | **0.9997** |
| | MLP | 235 | 0.012466 | 0.9767 |
| **Efficient-KAN** | KAN | 158 | 0.000083 | **0.9998** |
| | MLP | 235 | 0.011365 | 0.9787 |

### Tabular Classification: Breast Cancer Dataset

| Implementation | Model | Parameters | Accuracy | AUROC |
|----------------|-------|------------|----------|-------|
| **Pykan** | KAN | 7,168 | **99.1%** | 0.9954 |
| | MLP | 7,282 | 95.6% | 0.9950 |
| **Efficient-KAN** | KAN | 5,600 | **97.4%** | 0.9954 |
| | MLP | 7,282 | 95.6% | 0.9944 |

**Key Findings:**
- KAN achieves **better results with fewer parameters** in both experiments
- Efficient-KAN is faster but loses interpretability features (plot, prune, symbolic)
- KAN excels at learning smooth functions like `sin()` and `x²`

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiments
python -m src.train.train_toy --config configs/toy.yaml
python -m src.train.train_tabular --config configs/tabular.yaml
```

---

## Experiments

### 1. Toy Function Regression

Learns `y = sin(3*x1) + x2²` and visualizes the learned spline functions.

```bash
python -m src.train.train_toy --config configs/toy.yaml
```

**Outputs** in `results/toy/`:
- `plots/kan_before.png` - Initial random splines
- `plots/kan_after.png` - **Learned splines** (interpretability!)
- `plots/kan_pruned.png` - Network after pruning weak edges
- `plots/loss_curve.png` - Training dynamics
- `plots/heatmap_*.png` - Spatial predictions comparison
- `logs/symbolic.txt` - Symbolic formula from auto_symbolic()

### 2. Tabular Classification

Breast cancer dataset (569 samples, 30 features, binary classification).

```bash
python -m src.train.train_tabular --config configs/tabular.yaml
```

**Outputs** in `results/tabular_*/`:
- `plots/kan_before.png`, `kan_after.png`, `kan_pruned.png` - Spline evolution
- `plots/confusion_matrix_*.png` - Classification performance
- `plots/comparison.png` - KAN vs MLP metrics bar chart

---

## Project Structure

```
KAN EXP/
├── configs/
│   ├── toy.yaml              # Toy regression config
│   └── tabular.yaml          # Classification config
├── src/
│   ├── data/
│   │   ├── toy.py            # Synthetic function generator
│   │   └── tabular.py        # Sklearn dataset loaders
│   ├── models/
│   │   ├── kan.py            # KAN with pykan interpretability
│   │   └── mlp.py            # MLP baseline (matching params)
│   ├── train/
│   │   ├── train_toy.py      # Regression experiment
│   │   ├── train_tabular.py  # Classification experiment
│   │   └── utils.py          # Training utilities
│   └── eval/
│       ├── metrics.py        # MSE, R², Accuracy, AUROC, etc.
│       └── plots.py          # Visualization functions
├── results/                   # Generated outputs (auto-created)
├── requirements.txt
└── README.md
```

---

## Configuration

### Toy Experiment (`configs/toy.yaml`)

```yaml
# Architecture
kan_width: [2, 5, 1]      # [input_dim, hidden, output_dim]
kan_grid: 5               # B-spline grid points (higher = more expressive)
kan_k: 3                  # Spline order

# Training
n_epochs: 100
learning_rate: 0.01
batch_size: 64

# Interpretability
enable_prune: true        # Prune weak edges after training
prune_threshold: 0.01     # Edge importance threshold
enable_symbolic: true     # Run symbolic regression
```

### Tabular Experiment (`configs/tabular.yaml`)

```yaml
dataset_name: "breast_cancer"   # Options: breast_cancer, iris, wine, digits
kan_hidden: [16]                # Hidden layer widths
n_epochs: 100
learning_rate: 0.001
```

---

## Understanding KAN Interpretability

### Spline Visualization

KAN's key advantage: **learnable activation functions on edges**.

- `kan_before.png` - Random B-splines before training
- `kan_after.png` - **Learned functions** that approximate the target
- `kan_pruned.png` - Important edges retained, weak edges removed

Each curve in the plot shows the 1D transformation applied to that input.

### Symbolic Regression

Pykan's `auto_symbolic()` attempts to fit symbolic functions (sin, cos, x², etc.) to each learned spline:

```
# Example output for y = sin(3*x1) + x2²
fixing (0,0,0) with sin, r2=0.95...
fixing (0,1,0) with x^2, r2=0.99...
```

---

## Parameter Counting

### KAN Parameters

For architecture `[2, 5, 1]` with `grid=5`, `k=3`:
- Each edge has `(grid + k) + extras ≈ 14` learnable parameters
- Layer 1: `2 × 5 = 10` edges → 140 params
- Layer 2: `5 × 1 = 5` edges → 70 params
- **Total: ~210 parameters**

### MLP Parameters (matched)

The MLP is auto-configured to have **equal or more** parameters than KAN for fair comparison.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root with `python -m src.train.train_toy` |
| `pykan not found` | Install with `pip install pykan` or use the built-in fallback |
| Unicode errors on Windows | Fixed - the code uses ASCII-only console output |
| CUDA out of memory | Add `--device cpu` or reduce `batch_size` in config |
| Plots not showing | They're saved to `results/*/plots/` |

---

## Using Efficient-KAN (Alternative to Pykan)

The code includes a **built-in efficient-kan implementation** as a fallback. This is useful when:
- Pykan installation fails
- You want a lighter-weight implementation
- You don't need symbolic regression features

### Option 1: CLI Flag (Easiest)

Use the `--efficient-kan` flag to switch implementations:

```bash
# Use efficient-kan for toy experiment
python -m src.train.train_toy --config configs/toy.yaml --efficient-kan

# Use efficient-kan for tabular experiment  
python -m src.train.train_tabular --config configs/tabular.yaml --efficient-kan
```

You'll see in the output:
```
Using pykan: False
```

### Option 2: Config File

Add to your YAML config:

```yaml
use_efficient_kan: true
```

### Option 3: Uninstall Pykan

Simply uninstall pykan - the code automatically uses efficient-kan:

```bash
pip uninstall pykan
```

### Feature Comparison

| Feature | Pykan | Efficient-KAN |
|---------|-------|---------------|
| Training | Yes | Yes |
| Spline visualization (`plot()`) | Yes | No |
| Pruning (`prune()`) | Yes | No |
| Symbolic regression (`auto_symbolic()`) | Yes | No |
| Speed | Moderate | Faster |
| Dependencies | Many (sympy, tqdm, pandas) | None |

**Recommendation**: Use **pykan** for research/interpretability, **efficient-kan** for production/speed.

---

## References

- **KAN Paper**: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
- **Pykan**: https://github.com/KindXiaoming/pykan


