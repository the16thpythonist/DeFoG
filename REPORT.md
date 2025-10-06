# DeFoG: Discrete Flow Matching for Graph Generation - Technical Report

## Overview

**DeFoG** (Discrete Flow Matching for Graph Generation) is a novel graph generative framework that addresses fundamental limitations in existing graph diffusion models. Published at ICML 2025, it introduces a discrete flow matching formulation specifically designed for graph-structured data.

### What Problem Does It Solve?

Graph diffusion models, while achieving state-of-the-art performance, suffer from **tight coupling between training and sampling stages**. This means:
- Optimizing components like noise schedules or rate matrices requires expensive re-training
- Models use a one-size-fits-all approach across diverse graph datasets
- Limited flexibility in sampling stage optimization
- Inefficient sampling requiring many denoising steps

### Core Approach

DeFoG employs a **discrete flow matching** formulation with two key processes:

1. **Noising Process** (training): Linear interpolation from clean data to noise
   ```
   p_{t|1}(G_t|G_1) = t·δ(G_t, G_1) + (1-t)·p_0(G_t)
   ```

2. **Denoising Process** (sampling): Continuous-time Markov chain (CTMC) with learned rate matrices
   - Rate matrices are computed at sampling time (not fixed during training)
   - Neural network predicts clean graph marginals: p_θ_{1|t}(·|G_t)
   - Rate matrices are derived from these predictions

### Key Results

- **Near-saturated validity**: 99.5% (Planar), 96.5% (Tree), 90% (SBM)
- **Molecular generation**: 92.8% validity on MOSES (previous SOTA: 90.5%)
- **Sampling efficiency**: Achieves 95% validity on Planar with only 5-10% of diffusion model steps
- **State-of-the-art** across synthetic, molecular, and digital pathology datasets

---

## Implementation

The codebase is well-structured with modular components implementing the theoretical framework described in the paper.

### Architecture Overview

```
src/
├── graph_discrete_flow_model.py    # Main model (PyTorch Lightning)
├── flow_matching/
│   ├── rate_matrix.py              # R*, R^DB, R^TG implementations
│   ├── time_distorter.py           # Time distortion functions
│   ├── noise_distribution.py       # Initial distributions (p_0)
│   └── flow_matching_utils.py      # Noising/denoising utilities
├── models/
│   ├── transformer_model.py        # Graph Transformer architecture
│   ├── extra_features.py           # RRWP positional encodings
│   └── layers.py                   # Attention and MLP layers
└── metrics/                        # Evaluation metrics
```

### Core Components

#### 1. Main Model (`src/graph_discrete_flow_model.py`)

The `GraphDiscreteFlowModel` class orchestrates training and sampling:

```python
class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, ...):
        # Initialize graph transformer
        self.model = GraphTransformer(...)

        # Noise distribution (p_0)
        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)

        # Time distortion
        self.time_distorter = TimeDistorter(
            train_distortion, sample_distortion
        )

        # Rate matrix designer (sampling only)
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb, rdb_crit, eta, omega, limit_dist
        )
```

**Training**: Predicts clean graph marginals from noisy inputs
- Input: Noisy graph G_t at timestep t
- Output: Predicted marginals p_θ_{1|t}(·|G_t) for all nodes and edges
- Loss: Cross-entropy between predictions and ground truth

**Sampling**: Generates graphs via CTMC denoising
- Start from noise p_0
- Iteratively denoise using rate matrices
- Rate matrices computed from neural network predictions at each step

#### 2. Rate Matrix Computation (`src/flow_matching/rate_matrix.py`)

The `RateMatrixDesigner` class implements three types of rate matrices:

**a) R\* - Base Flow Matching Rate Matrix**
```python
def compute_Rstar(self, dfm_variables):
    # Equation (3) from paper
    inner_X = dt_p_vals_X - dt_p_vals_at_Xt[:, :, None]
    Rstar_t_numer_X = F.relu(inner_X)

    Rstar_t_denom_X = Z_t_X * pt_vals_at_Xt
    Rstar_t_X = Rstar_t_numer_X / Rstar_t_denom_X[:, :, None]
    return Rstar_t_X, Rstar_t_E
```

**b) R^DB - Detailed Balance Stochasticity** (parameter η)
```python
def compute_RDB(self, ...):
    # Preserves Kolmogorov equation
    # Controlled by η (stochasticity level)
    Rdb_t_X = pt_vals_X * x_mask * self.eta
    Rdb_t_E = pt_vals_E * e_mask * self.eta
    return Rdb_t_X, Rdb_t_E
```

**c) R^TG - Target Guidance** (parameter ω)
```python
def compute_R_tg(self, X_1_sampled, E_1_sampled, ...):
    # Amplify transitions to predicted clean state
    # Controlled by ω (guidance strength)
    X1_onehot = F.one_hot(X_1_sampled, num_classes=self.num_classes_X)
    Rtg_t_numer_X = X1_onehot * self.omega * mask_X
    Rtg_t_X = Rtg_t_numer_X / denom_X[:, :, None]
    return Rtg_t_X, Rtg_t_E
```

**Final Rate Matrix**:
```python
R_t = R*_t + R^DB_t + R^TG_t
```

#### 3. Time Distortion (`src/flow_matching/time_distorter.py`)

Implements variable step sizes through bijective time transformations:

```python
class TimeDistorter:
    def apply_distortion(self, t, distortion_type):
        if distortion_type == "identity":
            ft = t
        elif distortion_type == "polydec":  # Smaller steps near t=1
            ft = 2*t - t**2
        elif distortion_type == "cos":      # Emphasize boundaries
            ft = (1 - torch.cos(t * torch.pi)) / 2
        # ... more distortion types
        return ft
```

**Use cases**:
- **Training distortion**: Skews time sampling during training to focus on critical ranges
- **Sampling distortion**: Creates variable step sizes (e.g., smaller steps near t=1 for final refinement)

#### 4. Graph Transformer with RRWP Features

**Base Architecture** (`src/models/transformer_model.py`):
- Multi-head self-attention on nodes
- Edge features integrated via message passing
- Permutation equivariant by design

**RRWP Features** (`src/models/extra_features.py`):
- Relative Random Walk Probabilities encode structural information
- Computed as powers of degree-normalized adjacency: [I, M, M², ..., M^(K-1)]
- More efficient than spectral/cycle features (see paper Appendix G.4)
- Enhances expressivity beyond standard GNN limitations

---

## Key Innovations

DeFoG introduces several innovations that distinguish it from standard flow matching approaches:

### 1. Training-Sampling Decoupling

**Standard Flow Matching**: Rate matrices are implicitly defined during training and fixed

**DeFoG Innovation**:
- Training only learns to predict clean marginals p_θ_{1|t}(·|G_t)
- Rate matrices are **constructed at sampling time** from these predictions
- Enables flexible rate matrix design without retraining

**Theoretical Guarantee** (Theorem 1):
```
||R_t(G_t, G_{t+dt}) - R^θ_t(G_t, G_{t+dt})||² ≤ C₀ + C₁·E[p_{t|1}(G_t|G₁) Σ -log p^θ_{1|t}]
```
Minimizing cross-entropy loss directly minimizes rate matrix estimation error!

**Impact**:
- Try different sampling strategies without retraining (hours vs. days)
- Dataset-specific optimization (e.g., ω=0.05 for Planar, ω=0.5 for MOSES)
- Enables extensive sampling ablations (Figures 9-10 in paper)

### 2. Novel Sampling Methods

#### a) Sample Distortion (Time Warping)

**Standard approach**: Uniform time steps Δt = 1/T

**DeFoG**: Variable step sizes via bijective functions f(t)
```python
# polydec: Smaller steps near t=1 (critical for planarity)
f(t) = 2t - t²

# Effective step sizes decrease as t → 1
```

**Why it matters**:
- Graphs undergo abrupt transitions (discrete states) unlike continuous images
- Final steps critical for preserving global properties (e.g., planarity, tree structure)
- Planar dataset: 99.5% validity with polydec vs 77.5% with identity

#### b) Target Guidance (ω parameter)

**Standard R\***: Only uses gradient of noising process

**DeFoG R^TG**: Amplifies transitions to predicted clean states
```python
# Additional rate boost when transitioning to predicted x₁
R^ω_t(z_t, z_{t+dt}|z₁) = ω · δ(z_{t+dt}, z₁) / (Z^>0_t · p_{t|1}(z_t|z₁))
```

**Effect**:
- Guides generation toward high-confidence predictions
- Especially effective with fewer sampling steps
- Controlled violation of Kolmogorov equation: O(ω)

**Empirical results** (paper Figure 3b):
- Sweet spot varies by dataset: ω∈[0.05, 0.5]
- Too high → overfits to training distribution (↓novelty)
- Particularly beneficial with limited steps (50 vs 1000)

#### c) Stochasticity Control (η parameter)

**Standard R\***: Minimal expected jumps (deterministic-like)

**DeFoG R^η**: Adds controlled stochasticity via detailed balance
```python
R^η_t = R*_t + η·R^DB_t
```

**R^DB satisfies detailed balance**:
```
p_{t|1}(z_t|z₁)·R^DB_t(z_t, z_{t+dt}|z₁) = p_{t|1}(z_{t+dt}|z₁)·R^DB_t(z_{t+dt}, z_t|z₁)
```

**Why it works**:
- Enables transitions to states forbidden by R\*
- Acts as error correction mechanism
- Preserves Kolmogorov equation (provably valid)
- Optimal level exists: too much → disrupts generation

**Findings** (paper Figure 3c):
- Moderate η beneficial (e.g., η=50 for Planar)
- More steps → can tolerate higher η
- Dataset-dependent (Planar benefits, SBM doesn't)

### 3. Graph-Specific Adaptations

#### Discrete State Space Modeling

**Standard flow matching**: Designed for continuous data (images)

**DeFoG adaptations**:
- Respects discrete nature of graphs (nodes, edges ∈ categorical spaces)
- Linear interpolation noising preserves structural properties
- CTMC denoising naturally handles categorical transitions

#### Permutation Equivariance

**Guarantees** (Lemma 3):
- Loss function is permutation invariant
- Sampling probability is permutation invariant
- Model outputs are permutation equivariant

**Implementation**:
- Graph transformer with symmetric attention
- RRWP features are provably equivariant
- Node ordering doesn't affect generation

#### RRWP Expressivity

**Beyond standard GNN limitations**:
- Can approximate shortest path distances
- Encodes random walk dynamics
- Captures cycle and connectivity information
- **10-100x faster** than spectral features for large graphs (Table 13)

### 4. Theoretical Soundness

Unlike heuristic sampling tricks, DeFoG provides **formal guarantees**:

**Theorem 1** (Rate Matrix Bound):
Training loss directly bounds rate matrix estimation error

**Theorem 2** (Generation Quality):
```
||p(G₁) - p_data||_TV ≤ Ū(XN + E·N(N-1)/2) + B̄(XN + E·N(N-1)/2)²·Δt + O(Δt)
```

**Interpretation**:
- First term: Neural network approximation error (bounded by Theorem 1)
- Second term: CTMC discretization error (controlled by Δt)
- Both can be made arbitrarily small!

**Impact**: Unlike continuous-time discrete diffusion, DeFoG's sampling innovations are theoretically grounded.

---

## Comparison with Standard Flow Matching

| Aspect | Standard Discrete FM | DeFoG |
|--------|---------------------|-------|
| **Training** | Learns rate matrices implicitly | Learns clean marginals p_θ_{1\|t} |
| **Sampling** | Uses training-time rate matrices | Constructs rate matrices at sampling time |
| **Flexibility** | Fixed design after training | Adjustable ω, η, distortion without retraining |
| **Time steps** | Typically uniform | Variable (polydec, cos, etc.) |
| **Graph symmetries** | Not specifically addressed | Permutation equivariance guaranteed |
| **Theoretical guarantees** | Limited to basic FM theory | Extended bounds for graphs (Theorems 1-2) |
| **Sampling efficiency** | Standard | 10-20x fewer steps (5-10% of diffusion steps) |

---

## Conclusion

DeFoG represents a significant advance in graph generation by:

1. **Decoupling training from sampling** - enables rapid experimentation and dataset-specific optimization
2. **Novel sampling methods** - time distortion, target guidance, and stochasticity control work synergistically
3. **Theoretical rigor** - formal guarantees ensure sampling innovations are sound
4. **Practical impact** - achieves SOTA with 10-20x fewer sampling steps

The key insight is that **training should focus on prediction, not process** - by learning to predict clean graphs rather than fixing denoising dynamics, DeFoG unlocks a rich design space for sampling optimization.

---

## References

- Paper: [arXiv:2410.04263](https://arxiv.org/pdf/2410.04263)
- Code: [github.com/manuelmlmadeira/DeFoG](https://github.com/manuelmlmadeira/DeFoG)
- ICML 2025 Oral Presentation: [icml.cc/virtual/2025/oral/47238](https://icml.cc/virtual/2025/oral/47238)
