# DeFoG Core Module

A standalone, reusable implementation of **Discrete Flow Matching for Graph Generation** (DeFoG). This module provides clean interfaces for training graph generative models and sampling new graphs, designed for easy integration into external projects.

## Overview

DeFoG is a generative model for discrete graph-structured data. It learns to generate graphs by:
1. **Training**: Learning to predict clean graph features from noisy inputs
2. **Sampling**: Generating graphs via CTMC (Continuous-Time Markov Chain) denoising

Key features of this module:
- **PyTorch Lightning based**: Standard `LightningModule` patterns
- **PyG compatible**: Accepts PyTorch Geometric `DataLoader`, returns `List[Data]`
- **Configurable via constructor**: No external config files required
- **Flexible sampling**: Adjust parameters (eta, omega, time distortion) without retraining

## Installation

Requires Python 3.8+ with the following dependencies:

```bash
pip install torch torch_geometric pytorch_lightning
```

Then ensure the parent repository is in your Python path:

```bash
pip install -e .  # From the repository root
```

## Quick Start

### Training a Model

```python
from src.core import DeFoGModel
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

# Create dataloader from your PyG dataset
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model (auto-infers dimensions from data)
model = DeFoGModel.from_dataloader(
    train_loader,
    n_layers=6,
    hidden_dim=256,
    noise_type="marginal",
)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_dataloaders=train_loader)
```

### Generating Samples

```python
# Generate 100 graphs with default parameters
samples = model.sample(num_samples=100)

# Each sample is a PyG Data object
for graph in samples:
    print(f"Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")

# Generate with custom sampling parameters
samples = model.sample(
    num_samples=50,
    eta=10.0,              # Stochasticity (error correction)
    omega=0.05,            # Target guidance strength
    sample_steps=100,      # Number of denoising steps
    time_distortion="polydec",  # Variable step sizes
)
```

### Saving and Loading Models

```python
# Save model (weights + hyperparameters)
model.save("my_model")  # Saves to my_model.ckpt

# Load model
loaded_model = DeFoGModel.load("my_model")  # .ckpt extension auto-appended
samples = loaded_model.sample(num_samples=100)

# Load to specific device
model_gpu = DeFoGModel.load("my_model", device="cuda")
```

## Model Configuration

### Constructor Parameters

Create a model with explicit configuration:

```python
model = DeFoGModel(
    # Required: Data dimensions
    num_node_classes=4,        # Number of node feature classes
    num_edge_classes=2,        # Number of edge classes (including no-edge)

    # Architecture
    n_layers=6,                # Transformer layers
    hidden_dim=256,            # Hidden dimension
    hidden_mlp_dim=512,        # MLP hidden dimension
    n_heads=8,                 # Attention heads
    dropout=0.1,

    # Noise distribution: "uniform", "marginal", or "absorbing"
    noise_type="marginal",
    node_marginals=node_probs,  # Required for "marginal"
    edge_marginals=edge_probs,  # Required for "marginal"

    # Graph sizes
    node_counts=size_distribution,  # Tensor of size frequencies
    max_nodes=100,

    # Extra features: "none", "rrwp", or "cycles"
    extra_features_type="rrwp",
    rrwp_steps=10,

    # Training
    lr=1e-4,
    weight_decay=1e-5,
    lambda_edge=1.0,           # Edge loss weight
    train_time_distortion="identity",

    # Sampling defaults
    sample_steps=100,
    eta=0.0,
    omega=0.0,
    sample_time_distortion="identity",
)
```

### Factory Method

For convenience, create a model with dimensions inferred from data:

```python
model = DeFoGModel.from_dataloader(
    dataloader,
    n_layers=6,
    hidden_dim=256,
    noise_type="marginal",  # Will compute marginals from data
)
```

## Sampling Parameters

DeFoG's key innovation is that sampling parameters can be adjusted without retraining:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `eta` | Stochasticity / error correction | 0-100 (dataset dependent) |
| `omega` | Target guidance strength | 0-0.5 |
| `sample_steps` | Number of denoising steps | 10-1000 |
| `time_distortion` | Step size distribution | See below |

### Time Distortion Options

- `"identity"`: Uniform step sizes
- `"polydec"`: Smaller steps near t=1 (recommended for preserving global structure)
- `"polyinc"`: Smaller steps near t=0
- `"cos"`: Emphasize boundaries
- `"revcos"`: Reverse cosine

```python
# Example: Using polydec for better planarity preservation
samples = model.sample(
    num_samples=100,
    time_distortion="polydec",
    sample_steps=50,
)
```

## Data Format

### Input (Training)

The model expects a PyTorch Geometric `DataLoader` where each `Data` object has:
- `x`: Node features `(num_nodes, num_node_classes)` - one-hot encoded
- `edge_index`: Edge indices `(2, num_edges)`
- `edge_attr`: Edge features `(num_edges, num_edge_classes)` - one-hot encoded

### Output (Sampling)

Returns a `List[Data]` where each `Data` object contains:
- `x`: Node features `(num_nodes, num_node_classes)` - one-hot encoded
- `edge_index`: Edge indices `(2, num_edges)`
- `edge_attr`: Edge features `(num_edges, num_edge_classes)` - one-hot encoded

## Core Components

### Main Classes

| Class | Purpose |
|-------|---------|
| `DeFoGModel` | Main LightningModule for training and sampling |
| `LimitDistribution` | Defines noise distribution at t=0 |
| `RateMatrixDesigner` | Computes CTMC rate matrices for sampling |
| `TimeDistorter` | Applies time distortion for variable step sizes |
| `GraphTransformer` | Neural network architecture |
| `ExtraFeatures` | Computes RRWP and cycle features |

### Utility Functions

| Function | Purpose |
|----------|---------|
| `to_dense()` | Convert PyG batch to dense tensors |
| `dense_to_pyg()` | Convert dense tensors to PyG Data objects |
| `compute_dataset_statistics()` | Analyze dataloader for model configuration |
| `sample_noise()` | Sample from the limit distribution |

## Architecture Overview

```
Training Pipeline:
  PyG Batch → to_dense() → Apply Noise → Transformer → Predict Marginals → Loss

Sampling Pipeline:
  Sample Noise → [CTMC Denoising Loop] → dense_to_pyg() → List[Data]
                        ↓
              Transformer → Rate Matrix → Sample Next State
```

The **Graph Transformer** processes three feature types:
- **X**: Node features `(batch, nodes, features)`
- **E**: Edge features `(batch, nodes, nodes, features)`
- **y**: Global features `(batch, features)`

**RRWP (Relative Random Walk Probabilities)** provide structural encoding via powers of the normalized adjacency matrix, enhancing the transformer's ability to understand graph topology.

## Examples

For complete working examples, see `examples/core_usage.py` which demonstrates:

1. Creating models with explicit arguments
2. Creating models from dataloaders
3. Training with PyTorch Lightning
4. Sampling with various parameters
5. Saving and loading checkpoints
6. Data conversion utilities

Run the examples:

```bash
python examples/core_usage.py
```

## Common Patterns

### Custom Training Loop

```python
model = DeFoGModel(num_node_classes=4, num_edge_classes=2)
optimizer = model.configure_optimizers()

for batch in dataloader:
    optimizer.zero_grad()
    result = model.training_step(batch, 0)
    result["loss"].backward()
    optimizer.step()
```

### Sampling with Fixed Graph Size

```python
# All graphs with exactly 10 nodes
samples = model.sample(num_samples=50, num_nodes=10)

# Variable sizes from a tensor
sizes = torch.tensor([5, 8, 10, 12, 15])
samples = model.sample(num_samples=5, num_nodes=sizes)
```

### Accessing Internal Components

```python
# Get the limit distribution
limit_dist = model.limit_dist
print(f"Node distribution: {limit_dist.X}")
print(f"Edge distribution: {limit_dist.E}")

# Get the transformer
transformer = model.model
print(f"Layers: {transformer.n_layers}")
```

## Troubleshooting

### Empty Batches
If you encounter "Found a batch with no edges", ensure your graphs have at least one edge. The training step will skip empty batches.

### Memory Issues
For large graphs, reduce batch size or use gradient checkpointing:

```python
trainer = pl.Trainer(
    max_epochs=100,
    accumulate_grad_batches=4,  # Effective larger batch
)
```

### Sampling Quality
If generated graphs have poor quality:
- Increase `sample_steps` (more denoising iterations)
- Try `time_distortion="polydec"` (preserves global structure)
- Adjust `eta` for your dataset (start with 0, increase if needed)
- Use `noise_type="marginal"` to match training distribution

## API Reference

### DeFoGModel

```python
class DeFoGModel(pl.LightningModule):
    def __init__(self, num_node_classes, num_edge_classes, ...)
    def training_step(batch, batch_idx) -> dict
    def sample(num_samples, ...) -> List[Data]
    def configure_optimizers() -> Optimizer
    def save(path) -> str  # Save model to checkpoint

    @classmethod
    def load(path, device="cpu") -> DeFoGModel  # Load from checkpoint

    @classmethod
    def from_dataloader(dataloader, ...) -> DeFoGModel
```

### LimitDistribution

```python
class LimitDistribution:
    def __init__(noise_type, num_node_classes, num_edge_classes, ...)
    def ignore_virtual_classes(X, E, y) -> Tuple
    def add_virtual_classes(X, E, y) -> Tuple
```

### RateMatrixDesigner

```python
class RateMatrixDesigner:
    def __init__(eta, omega, limit_dist, ...)
    def compute_rate_matrices(t, node_mask, X_t, E_t, X_1_pred, E_1_pred) -> Tuple
```

For complete API details, see the docstrings in each module.

## License

See the main repository LICENSE file.
