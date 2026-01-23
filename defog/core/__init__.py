"""
DeFoG Core Module - Clean, reusable discrete flow matching for graph generation.

This module provides the core DeFoG model functionality with explicit interfaces,
decoupled from Hydra configuration, for use in external projects.

Quick Start:
    >>> from defog.core import DeFoGModel
    >>> import pytorch_lightning as pl
    >>> from torch_geometric.loader import DataLoader
    >>>
    >>> # Create model from dataloader (auto-infers dimensions)
    >>> loader = DataLoader(dataset, batch_size=32)
    >>> model = DeFoGModel.from_dataloader(loader)
    >>>
    >>> # Train
    >>> trainer = pl.Trainer(max_epochs=100)
    >>> trainer.fit(model, train_dataloaders=loader)
    >>>
    >>> # Sample
    >>> samples = model.sample(num_samples=100)

Main Classes:
    DeFoGModel: Main LightningModule for training and sampling
    PlaceHolder: Container for graph tensors (X, E, y)
    LimitDistribution: Noise/limit distribution specification
    RateMatrixDesigner: CTMC rate matrix computation
    TimeDistorter: Time distortion for variable step sizes
    GraphTransformer: Neural network architecture
    ExtraFeatures: RRWP and cycle features

Utility Functions:
    to_dense: Convert PyG batch to dense tensors
    dense_to_pyg: Convert dense tensors to PyG Data objects
    compute_dataset_statistics: Analyze dataloader for model configuration
"""

from .model import DeFoGModel
from .data import (
    PlaceHolder,
    to_dense,
    dense_to_pyg,
    DistributionNodes,
    compute_dataset_statistics,
    encode_no_edge,
    symmetrize_edges,
)
from .noise import (
    LimitDistribution,
    sample_noise,
    sample_from_probs,
)
from .rate_matrix import RateMatrixDesigner
from .time_distortion import TimeDistorter
from .transformer import GraphTransformer
from .features import ExtraFeatures, RRWPFeatures
from .loss import TrainLoss, compute_loss_components
from .layers import XEyTransformerLayer, NodeEdgeBlock, timestep_embedding

__all__ = [
    # Main model
    "DeFoGModel",
    # Data utilities
    "PlaceHolder",
    "to_dense",
    "dense_to_pyg",
    "DistributionNodes",
    "compute_dataset_statistics",
    "encode_no_edge",
    "symmetrize_edges",
    # Noise distribution
    "LimitDistribution",
    "sample_noise",
    "sample_from_probs",
    # Rate matrix
    "RateMatrixDesigner",
    # Time distortion
    "TimeDistorter",
    # Neural network
    "GraphTransformer",
    "XEyTransformerLayer",
    "NodeEdgeBlock",
    "timestep_embedding",
    # Features
    "ExtraFeatures",
    "RRWPFeatures",
    # Loss
    "TrainLoss",
    "compute_loss_components",
]

__version__ = "0.1.0"
