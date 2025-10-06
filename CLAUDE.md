# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeFoG (Discrete Flow Matching for Graph Generation) is a PyTorch-based generative model for discrete graph-structured data. It uses a novel discrete flow matching formulation that decouples training from sampling, enabling flexible optimization of sampling parameters without retraining.

**Paper**: https://arxiv.org/pdf/2410.04263
**ICML 2025**: Oral presentation

## Core Commands

### Installation
```bash
# Option 1: Conda
conda env create -f environment.yaml
conda activate defog

# Compile ORCA evaluator (required for metrics)
cd src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp

# Option 2: Docker
docker build --platform=linux/amd64 -t defog-image .
```

After installation, run `pip install -e .` to make repository modules visible.

### Training
All commands run from `src/` directory using Hydra config system:

```bash
# Quick test
python main.py +experiment=debug

# Full training (examples)
python main.py +experiment=planar dataset=planar
python main.py +experiment=qm9_no_h dataset=qm9
python main.py +experiment=moses dataset=moses
```

Available datasets: `planar`, `tree`, `sbm`, `comm20`, `qm9`, `guacamol`, `moses`, `zinc`, `tls`

### Evaluation/Sampling

```bash
# Test with checkpoint
python main.py +experiment=planar dataset=planar general.test_only=<path/to/checkpoint.ckpt>

# Default sampling (uses paper's optimized parameters)
python main.py +experiment=planar dataset=planar general.test_only=<ckpt>

# Custom sampling parameters
python main.py +experiment=planar dataset=planar general.test_only=<ckpt> \
  sample.eta=0 sample.omega=0.05 sample.time_distortion=polydec

# Sampling optimization (search for best η, ω, distortion)
python main.py +experiment=planar dataset=planar general.test_only=<ckpt> sample.search=all

# Multiple runs (for mean ± std)
python main.py +experiment=planar dataset=planar general.test_only=<ckpt> general.num_sample_fold=5

# Evaluate pre-generated samples without sampling
python main.py +experiment=planar dataset=planar general.generated_path=<path/to/samples.pkl>
```

## Architecture

### High-Level Flow

```
Training: Clean Graph G₁ → Noisy G_t → Neural Network → Predicted Marginals p_θ_{1|t}
Sampling: Noise p₀ → CTMC Denoising (using R_t from predictions) → Clean Graph G₁
```

### Key Components

**1. Main Entry Point**: `src/main.py`
- Uses Hydra for configuration (configs in `configs/`)
- Dataset-specific initialization (molecular vs non-molecular)
- Creates `GraphDiscreteFlowModel` with appropriate metrics and visualization tools

**2. Core Model**: `src/graph_discrete_flow_model.py`
- `GraphDiscreteFlowModel` (PyTorch Lightning module)
- **Training**: Learns to predict clean graph marginals from noisy inputs
  - Loss: Cross-entropy between predictions and ground truth
  - Time sampling: Uses `TimeDistorter.train_ft()` with train distortion
- **Sampling**: Generates graphs via CTMC denoising
  - Constructs rate matrices at runtime using `RateMatrixDesigner`
  - Uses sampling-specific time distortion and parameters (η, ω)

**3. Rate Matrix Computation**: `src/flow_matching/rate_matrix.py`
- `RateMatrixDesigner` computes: **R_t = R*_t + R^DB_t + R^TG_t**
  - **R\***: Base flow matching rate matrix (Equation 3 in paper)
  - **R^DB**: Detailed balance stochasticity (controlled by η parameter)
  - **R^TG**: Target guidance (controlled by ω parameter)
- Rate matrices are **computed at sampling time**, not fixed during training
- This enables changing η, ω without retraining

**4. Time Distortion**: `src/flow_matching/time_distorter.py`
- Creates variable step sizes through bijective transformations
- **Training distortion**: Skews time sampling during training
- **Sampling distortion**: Controls step sizes during generation
- Key functions:
  - `identity`: f(t) = t (uniform steps)
  - `polydec`: f(t) = 2t - t² (smaller steps near t=1, critical for planarity)
  - `cos`: f(t) = (1 - cos(πt))/2 (emphasize boundaries)

**5. Graph Transformer**: `src/models/transformer_model.py`
- `XEyTransformerLayer`: Updates node (X), edge (E), and global (y) features
- Multi-head attention with edge features integrated via message passing
- Permutation equivariant by design

**6. RRWP Features**: `src/models/extra_features.py`
- Relative Random Walk Probabilities: [I, M, M², ..., M^(K-1)]
- M = degree-normalized adjacency matrix
- Provides structural information beyond standard GNN expressivity
- 10-100x faster than spectral features for large graphs

**7. Noise Distribution**: `src/flow_matching/noise_distribution.py`
- Defines initial noise distribution p₀
- Supports uniform and marginal-based distributions

**8. Flow Matching Utils**: `src/flow_matching/flow_matching_utils.py`
- `sample_discrete_feature_noise()`: Samples from limit distribution
- `sample_discrete_features()`: Samples from predicted marginals
- Handles masking and symmetry for graphs

### Dataset-Specific Components

**Molecular Datasets** (QM9, MOSES, Guacamol, ZINC):
- Dataset files: `src/datasets/{qm9,moses,guacamol,zinc}_dataset.py`
- Metrics: `src/metrics/molecular_metrics.py`, `molecular_metrics_discrete.py`
- Extra features: `src/models/extra_features_molecular.py`
- Visualization: Uses RDKit for molecular rendering

**Non-Molecular Datasets** (Planar, Tree, SBM, Comm20):
- Dataset file: `src/datasets/spectre_dataset.py`
- Metrics: `src/analysis/spectre_utils.py` (dataset-specific sampling metrics)
- Visualization: `src/analysis/visualization.py` (graph plots)

**Conditional Dataset** (TLS):
- Dataset file: `src/datasets/tls_dataset.py`
- Metrics: `src/metrics/tls_metrics.py`

## Key Theoretical Concepts

### Training-Sampling Decoupling

**Standard Flow Matching**: Rate matrices implicitly defined during training and fixed

**DeFoG Innovation**:
- Training learns only to predict: p_θ_{1|t}(·|G_t)
- Rate matrices constructed at sampling time from these predictions
- **Impact**: Can optimize η, ω, distortion without retraining (hours vs days)

### Noising Process (Training)

Linear interpolation from clean to noise:
```
p_{t|1}(G_t|G_1) = t·δ(G_t, G_1) + (1-t)·p_0(G_t)
```
- At t=0: pure noise p₀
- At t=1: clean data G₁
- t sampled with training distortion

### Denoising Process (Sampling)

CTMC with rate matrix R_t computed from network predictions:
```python
# Pseudocode for one denoising step
G_t_plus_dt = sample_from_rate_matrix(
    R_t = compute_Rstar(pred) + eta*compute_RDB(pred) + omega*compute_Rtg(pred),
    current_state = G_t,
    dt = distorted_step_size
)
```

### Critical Parameters

- **η (eta)**: Stochasticity control (default: 0, dataset-dependent)
  - Higher η → more exploration, acts as error correction
  - Too high → disrupts generation
  - Planar benefits from η≈50, SBM doesn't need it

- **ω (omega)**: Target guidance strength (default: 0, dataset-dependent)
  - Amplifies transitions to predicted clean states
  - Particularly effective with fewer sampling steps
  - Sweet spot varies: ω∈[0.05, 0.5]
  - Too high → overfits to training distribution

- **Time Distortion**: Variable step sizes
  - `polydec` → smaller steps near t=1 (preserves global properties)
  - Critical for discrete graphs with abrupt transitions
  - Planar: 99.5% validity with polydec vs 77.5% with identity

## Configuration System (Hydra)

Structure: `configs/config.yaml` imports defaults from:
- `general/`: Training settings, checkpointing, GPU configuration
- `model/`: Model architecture, RRWP steps, extra features
- `train/`: Optimizer, learning rate, EMA decay
- `sample/`: Sampling parameters (η, ω, distortion, number of steps)
- `dataset/`: Dataset-specific settings
- `experiment/`: Combined configs for specific experiments

Override with command line:
```bash
python main.py +experiment=planar sample.omega=0.1 train.n_epochs=1000
```

## Adding New Datasets

1. Create dataset file in `src/datasets/`:
   - Unattributed graphs: Follow `spectre_dataset.py`
   - Node attributes: Follow `tls_dataset.py`
   - Node + edge attributes (molecules): Follow `qm9_dataset.py` or `guacamol_dataset.py`

2. Define two classes:
   - `Dataset`: PyTorch Geometric dataset (see [PyG docs](https://pytorch-geometric.readthedocs.io/))
   - `DatasetInfos`: Metadata (num_classes, node/edge dimensions)

3. Update `src/main.py` to handle new dataset

4. Add config file in `configs/dataset/`

5. (Optional) Add custom metrics in `src/metrics/`

## Checkpoints

Shared at: https://drive.switch.ch/index.php/s/MG7y2EZoithAywE

Includes:
- `.ckpt` files: Model checkpoints
- `.pkl` files: Pre-generated samples
- Results and metrics

Performance is consistent with paper findings (minor variations due to stochasticity).

## Important Implementation Details

### Permutation Equivariance
- Graph transformer uses symmetric attention
- RRWP features are provably equivariant
- Loss function and sampling are permutation invariant

### Edge Symmetry
- Edges are upper-triangular during sampling, then symmetrized
- Rate matrices respect edge symmetry: E_ij = E_ji

### Masking
- `node_mask` used throughout to handle variable graph sizes
- Masked nodes/edges should not contribute to loss or sampling

### Discrete States
- Node features: Categorical (one-hot encoded)
- Edge features: Categorical (one-hot encoded)
- Sampling from multinomial distributions over discrete states

## Performance Expectations

**Sampling Efficiency**:
- 10-20x fewer steps than diffusion models
- Planar: 95% validity with 50 steps (vs 500-1000 for diffusion)

**State-of-the-Art Results**:
- Planar: 99.5% validity
- Tree: 96.5% validity
- SBM: 90% validity
- MOSES: 92.8% validity (previous SOTA: 90.5%)
