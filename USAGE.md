# DeFoG Usage Guide

Quick reference for running the DeFoG pipeline on a new dataset.

---

## 1. Installation

### Option A: Conda
```bash
conda env create -f environment.yaml
conda activate defog
pip install -e .

# Compile ORCA evaluator (required for metrics)
cd src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
cd ../../..
```

### Option B: Docker
```bash
docker build --platform=linux/amd64 -t defog-image .
pip install -e .  # Run inside container
```

---

## 2. Dataset Preparation

### Step 2.1: Create Dataset File

Create `src/datasets/your_dataset.py` with two classes:

**For unattributed graphs** (nodes/edges have no features):
- Template: `src/datasets/spectre_dataset.py`
- Implement: `YourDataset` (PyTorch Geometric `InMemoryDataset`)
- Implement: `YourDatasetInfos` (metadata: num_nodes, num_edges, etc.)

**For graphs with node attributes**:
- Template: `src/datasets/tls_dataset.py`
- Additional fields in `DatasetInfos`: node feature dimensions

**For molecular data** (node + edge attributes):
- Template: `src/datasets/qm9_dataset.py` or `src/datasets/guacamol_dataset.py`
- Additional fields: atom types, bond types, valency constraints

### Step 2.2: Create Dataset Config

Create `configs/dataset/your_dataset.yaml`:
```yaml
name: your_dataset
remove_h: null  # For molecules: true/false to remove hydrogens
datadir: 'data/your_dataset/'
```

### Step 2.3: Update Main Script

Edit `src/main.py` to handle your dataset (lines 30-198):

```python
elif dataset_config["name"] == "your_dataset":
    from datasets import your_dataset

    datamodule = your_dataset.YourDataModule(cfg)
    dataset_infos = your_dataset.YourDatasetInfos(datamodule, cfg)

    # Choose appropriate metrics/visualization
    # For non-molecular: use NonMolecularVisualization, TrainAbstractMetricsDiscrete
    # For molecular: use MolecularVisualization, TrainMolecularMetricsDiscrete
```

### Step 2.4: Create Experiment Config (Optional but Recommended)

Create `configs/experiment/your_dataset.yaml`:
```yaml
# @package _global_
general:
    name : 'your_dataset'
    gpus : 1
    wandb: 'online'
    check_val_every_n_epochs: 2000
    sample_every_val: 1
    samples_to_generate: 40
    final_model_samples_to_generate: 40
    sample_steps: 1000

train:
    n_epochs: 100000
    batch_size: 64
    save_model: True

sample:
    time_distortion: 'identity'  # Start with identity, optimize later
    omega: 0                      # Start with 0, optimize later
    eta: 0                        # Start with 0, optimize later

model:
    n_layers: 10
    hidden_mlp_dims: { 'X': 128, 'E': 64, 'y': 128 }
    hidden_dims: { 'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256 }
```

---

## 3. Training

All commands run from `src/` directory.

### Quick Test
```bash
python main.py +experiment=debug
```

### Full Training
```bash
python main.py +experiment=your_dataset dataset=your_dataset
```

**Override config parameters**:
```bash
python main.py +experiment=your_dataset dataset=your_dataset \
    train.n_epochs=50000 \
    train.batch_size=128 \
    general.gpus=2
```

**Training outputs**:
- Checkpoints: `outputs/<timestamp>/checkpoints/`
- Logs: `outputs/<timestamp>/`

---

## 4. Sampling Optimization

After training, optimize sampling hyperparameters (η, ω, time_distortion).

### Search for All Parameters
```bash
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=<path/to/checkpoint.ckpt> \
    sample.search=all
```

### Search Individual Components
```bash
# Target guidance (ω)
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=<ckpt> sample.search=target_guidance

# Time distortion
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=<ckpt> sample.search=distortion

# Stochasticity (η)
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=<ckpt> sample.search=stochasticity
```

**Output**: CSV file with metrics for each parameter combination

**Search intervals**: Configured in `configs/sample/default.yaml`. Adjust as needed.

---

## 5. Final Evaluation

Use optimal parameters from sampling optimization.

### Single Run
```bash
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=<ckpt> \
    sample.eta=<optimal_η> \
    sample.omega=<optimal_ω> \
    sample.time_distortion=<optimal_distortion>
```

### Multiple Runs (for mean ± std)
```bash
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=<ckpt> \
    sample.eta=<η> sample.omega=<ω> sample.time_distortion=<distortion> \
    general.num_sample_fold=5
```

### Evaluate Pre-Generated Samples
If you already have samples (`.pkl` file):
```bash
python main.py +experiment=your_dataset dataset=your_dataset \
    general.generated_path=<path/to/samples.pkl>
```

---

## 6. Key Parameters Reference

### Sampling Parameters
- **η (eta)**: Stochasticity control (default: 0)
  - Higher → more exploration, acts as error correction
  - Dataset-dependent: planar ≈ 50, most others = 0

- **ω (omega)**: Target guidance strength (default: 0)
  - Amplifies transitions to predicted clean states
  - Typical range: [0.05, 0.5]
  - Too high → overfits to training distribution

- **time_distortion**: Variable step sizes during sampling
  - `identity`: f(t) = t (uniform steps)
  - `polydec`: f(t) = 2t - t² (smaller steps near t=1)
  - `cos`: f(t) = (1 - cos(πt))/2

### Training Parameters
- **n_epochs**: Total epochs (default: 100000)
- **batch_size**: Batch size (default: 64)
- **sample_steps**: Number of denoising steps (default: 1000)

---

## Common Workflows

### Workflow 1: Training from Scratch
```bash
# 1. Train model
cd src
python main.py +experiment=your_dataset dataset=your_dataset

# 2. Optimize sampling
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=outputs/<timestamp>/checkpoints/last.ckpt \
    sample.search=all

# 3. Final evaluation with optimal parameters
python main.py +experiment=your_dataset dataset=your_dataset \
    general.test_only=outputs/<timestamp>/checkpoints/last.ckpt \
    sample.eta=<η> sample.omega=<ω> sample.time_distortion=<dist> \
    general.num_sample_fold=5
```

### Workflow 2: Using Pre-Trained Checkpoint
```bash
# Download checkpoint from https://drive.switch.ch/index.php/s/MG7y2EZoithAywE
cd src

# Use paper's optimized parameters (included in experiment config)
python main.py +experiment=planar dataset=planar \
    general.test_only=<path/to/checkpoint.ckpt>

# Or override with custom parameters
python main.py +experiment=planar dataset=planar \
    general.test_only=<ckpt> \
    sample.eta=50 sample.omega=0.05 sample.time_distortion=polydec
```

---

## Troubleshooting

**Import errors after installation**:
```bash
pip install -e .
```

**ORCA evaluator not found**:
```bash
cd src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
```

**Out of memory during training**:
- Reduce `train.batch_size`
- Reduce `model.hidden_dims` dimensions
- Use fewer GPUs or smaller model

**Poor sampling quality**:
- Run sampling optimization (`sample.search=all`)
- Try different time distortions (`polydec`, `cos`, `identity`)
- Increase `sample_steps` (default: 1000)

---

## Additional Resources

- **Paper**: https://arxiv.org/pdf/2410.04263
- **Checkpoints**: https://drive.switch.ch/index.php/s/MG7y2EZoithAywE
- **PyTorch Geometric Docs**: https://pytorch-geometric.readthedocs.io/
- **Hydra Docs**: https://hydra.cc/
