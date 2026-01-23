"""
DeFoG Core Module Usage Examples

This script demonstrates how to use the standalone DeFoG core module for:
1. Creating a model with explicit arguments
2. Creating a model from a dataloader (auto-inferred dimensions)
3. Training with PyTorch Lightning
4. Sampling with various parameters
5. Loading a trained model from checkpoint

Requirements:
    pip install torch torch_geometric pytorch_lightning

Usage:
    python examples/core_usage.py
"""

import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# Import the core module
from src.core import (
    DeFoGModel,
    PlaceHolder,
    to_dense,
    dense_to_pyg,
    LimitDistribution,
    compute_dataset_statistics,
)


def create_synthetic_dataset(num_graphs: int = 100, min_nodes: int = 5, max_nodes: int = 15):
    """
    Create a synthetic dataset of random graphs for demonstration.

    Each graph has:
    - Random number of nodes between min_nodes and max_nodes
    - Node features: 4 classes (one-hot)
    - Edge features: 2 classes (no-edge, edge)
    """
    dataset = []

    for _ in range(num_graphs):
        # Random number of nodes
        n = torch.randint(min_nodes, max_nodes + 1, (1,)).item()

        # Random node features (4 classes)
        x = torch.zeros(n, 4)
        x[torch.arange(n), torch.randint(0, 4, (n,))] = 1

        # Random edges (Erdos-Renyi style, ~30% edge probability)
        adj = (torch.rand(n, n) < 0.3).float()
        adj = ((adj + adj.t()) > 0).float()  # Symmetrize
        adj.fill_diagonal_(0)  # No self-loops

        # Convert to edge_index format
        edge_index = adj.nonzero(as_tuple=False).t()

        # Edge attributes (2 classes: no-edge=0, edge=1)
        # For sparse format, we only store actual edges (class 1)
        edge_attr = torch.zeros(edge_index.size(1), 2)
        edge_attr[:, 1] = 1  # All stored edges are "edge" class

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        dataset.append(data)

    return dataset


def example_1_explicit_model():
    """
    Example 1: Create a model with explicit arguments.

    This approach gives you full control over all model parameters.
    """
    print("\n" + "="*60)
    print("Example 1: Creating model with explicit arguments")
    print("="*60)

    # Define node count distribution (probability of each graph size)
    node_counts = torch.zeros(21)
    node_counts[5:16] = 1.0  # Uniform over 5-15 nodes
    node_counts = node_counts / node_counts.sum()

    # Create model with explicit configuration
    model = DeFoGModel(
        # Data dimensions
        num_node_classes=4,
        num_edge_classes=2,

        # Architecture
        n_layers=4,
        hidden_dim=128,
        hidden_mlp_dim=256,
        n_heads=4,
        dropout=0.1,

        # Noise distribution
        noise_type="uniform",  # or "marginal", "absorbing"

        # Graph size distribution
        node_counts=node_counts,
        max_nodes=20,

        # Features
        extra_features_type="rrwp",
        rrwp_steps=8,

        # Training
        lr=1e-4,
        weight_decay=1e-5,
        lambda_edge=1.0,
        train_time_distortion="identity",

        # Sampling defaults
        sample_steps=50,
        eta=0.0,
        omega=0.0,
        sample_time_distortion="identity",
    )

    print(f"Created model: {model}")
    print(f"Input dimensions: {model.input_dims}")
    print(f"Output dimensions: {model.output_dims}")

    return model


def example_2_from_dataloader():
    """
    Example 2: Create a model from a dataloader.

    This approach automatically infers dimensions from your data.
    """
    print("\n" + "="*60)
    print("Example 2: Creating model from dataloader")
    print("="*60)

    # Create synthetic dataset
    dataset = create_synthetic_dataset(num_graphs=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Analyze dataset statistics
    stats = compute_dataset_statistics(dataloader)
    print(f"Dataset statistics:")
    print(f"  - Node classes: {stats['num_node_classes']}")
    print(f"  - Edge classes: {stats['num_edge_classes']}")
    print(f"  - Max nodes: {stats['max_nodes']}")
    print(f"  - Node marginals: {stats['node_marginals']}")
    print(f"  - Edge marginals: {stats['edge_marginals']}")

    # Create model (auto-infers dimensions)
    model = DeFoGModel.from_dataloader(
        dataloader,
        n_layers=4,
        hidden_dim=128,
        noise_type="marginal",  # Uses computed marginals as prior
        extra_features_type="rrwp",
        rrwp_steps=8,
    )

    print(f"\nCreated model: {model}")

    return model, dataloader


def example_3_training():
    """
    Example 3: Train the model with PyTorch Lightning.
    """
    print("\n" + "="*60)
    print("Example 3: Training with PyTorch Lightning")
    print("="*60)

    # Create dataset and model
    dataset = create_synthetic_dataset(num_graphs=200)
    train_loader = DataLoader(dataset[:160], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[160:], batch_size=32)

    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=2,  # Small model for demo
        hidden_dim=64,
        noise_type="uniform",
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir="./checkpoints",
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader)
    print("Training complete!")

    return model, trainer


def example_4_sampling():
    """
    Example 4: Generate samples with various parameters.
    """
    print("\n" + "="*60)
    print("Example 4: Sampling with various parameters")
    print("="*60)

    # Create a small model for demo
    model = DeFoGModel(
        num_node_classes=4,
        num_edge_classes=2,
        n_layers=2,
        hidden_dim=64,
        max_nodes=15,
    )
    model.eval()

    # Basic sampling
    print("\n1. Basic sampling (5 graphs):")
    samples = model.sample(num_samples=5, show_progress=True)
    for i, s in enumerate(samples):
        print(f"   Graph {i}: {s.x.size(0)} nodes, {s.edge_index.size(1)//2} edges")

    # Fixed number of nodes
    print("\n2. Sampling with fixed node count (10 nodes each):")
    samples = model.sample(num_samples=3, num_nodes=10, show_progress=False)
    for i, s in enumerate(samples):
        print(f"   Graph {i}: {s.x.size(0)} nodes")

    # Different sampling parameters
    print("\n3. Sampling with stochasticity (eta=10):")
    samples = model.sample(
        num_samples=3,
        eta=10.0,
        sample_steps=50,
        show_progress=False
    )
    for i, s in enumerate(samples):
        print(f"   Graph {i}: {s.x.size(0)} nodes")

    # Target guidance
    print("\n4. Sampling with target guidance (omega=0.1):")
    samples = model.sample(
        num_samples=3,
        omega=0.1,
        sample_steps=50,
        show_progress=False
    )
    for i, s in enumerate(samples):
        print(f"   Graph {i}: {s.x.size(0)} nodes")

    # Time distortion
    print("\n5. Sampling with polydec time distortion:")
    samples = model.sample(
        num_samples=3,
        time_distortion="polydec",
        sample_steps=50,
        show_progress=False
    )
    for i, s in enumerate(samples):
        print(f"   Graph {i}: {s.x.size(0)} nodes")

    return samples


def example_5_checkpoint():
    """
    Example 5: Save and load model from checkpoint.
    """
    print("\n" + "="*60)
    print("Example 5: Saving and loading checkpoints")
    print("="*60)

    import tempfile
    import os

    # Create and "train" a model
    model = DeFoGModel(
        num_node_classes=4,
        num_edge_classes=2,
        n_layers=2,
        hidden_dim=64,
    )

    # Create a temporary directory for the checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model using the simple save() method
        ckpt_path = os.path.join(tmpdir, "my_model")  # .ckpt auto-appended
        saved_path = model.save(ckpt_path)
        print(f"Saved model to: {saved_path}")

        # Load model using the simple load() method
        loaded_model = DeFoGModel.load(ckpt_path)
        print(f"Loaded model: {loaded_model}")

        # Verify loaded model works
        samples = loaded_model.sample(num_samples=2, sample_steps=3, show_progress=False)
        print(f"Generated {len(samples)} samples from loaded model")

        # Load to specific device
        loaded_cpu = DeFoGModel.load(ckpt_path, device="cpu")
        print(f"Loaded to device: {next(loaded_cpu.parameters()).device}")

    return loaded_model


def example_6_data_conversion():
    """
    Example 6: Working with data conversions.
    """
    print("\n" + "="*60)
    print("Example 6: Data conversion utilities")
    print("="*60)

    # Create some sample PyG data
    dataset = create_synthetic_dataset(num_graphs=3)
    batch = Batch.from_data_list(dataset)

    print(f"PyG Batch: {batch}")
    print(f"  - batch.x shape: {batch.x.shape}")
    print(f"  - batch.edge_index shape: {batch.edge_index.shape}")
    print(f"  - batch.edge_attr shape: {batch.edge_attr.shape}")

    # Convert to dense tensors
    dense_data, node_mask = to_dense(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch
    )

    print(f"\nDense representation:")
    print(f"  - X shape: {dense_data.X.shape} (batch, max_nodes, node_classes)")
    print(f"  - E shape: {dense_data.E.shape} (batch, max_nodes, max_nodes, edge_classes)")
    print(f"  - node_mask shape: {node_mask.shape}")

    # Convert back to PyG
    n_nodes = node_mask.sum(dim=1).int()
    pyg_list = dense_to_pyg(dense_data.X, dense_data.E, dense_data.y, node_mask, n_nodes)

    print(f"\nConverted back to {len(pyg_list)} PyG Data objects")
    for i, data in enumerate(pyg_list):
        print(f"  - Graph {i}: {data.x.size(0)} nodes, {data.edge_index.size(1)} edges")


def main():
    """Run all examples."""
    print("DeFoG Core Module - Usage Examples")
    print("="*60)

    # Example 1: Explicit model creation
    model1 = example_1_explicit_model()

    # Example 2: Model from dataloader
    model2, loader = example_2_from_dataloader()

    # Example 3: Training (short demo)
    # Uncomment to run actual training:
    # model3, trainer = example_3_training()
    print("\n" + "="*60)
    print("Example 3: Training (skipped for quick demo)")
    print("  Uncomment example_3_training() to run actual training")
    print("="*60)

    # Example 4: Sampling
    samples = example_4_sampling()

    # Example 5: Checkpointing
    loaded_model = example_5_checkpoint()

    # Example 6: Data conversion
    example_6_data_conversion()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
