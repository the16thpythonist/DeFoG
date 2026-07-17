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
from .callbacks import TrainingMonitorCallback, SampleVisualizationCallback, EMACallback
from .domain import GraphDomain, GenericGraphDomain, generation_metrics
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
from .size_distribution import (
    SizeDistribution,
    EmpiricalSizeDistribution,
    FixedSizeDistribution,
    ExplicitSizeDistribution,
    UniformSizeDistribution,
    CategoricalSizeDistribution,
    ConditionalSizeDistribution,
)
from .rate_matrix import RateMatrixDesigner
from .sampler import Sampler, InpaintingSampler, GuidedSampler, RefinementSampler, AdaptedSampler
from .adapter import (
    Modulation,
    AdaLNAdapter,
    ConditionBranch,
    AdapterComposition,
    AdapterRegistry,
    AdapterModule,
)
from .constraint import Constraint, SubgraphConstraint
from .guidance import (
    DensityRatio,
    EnergyRatio,
    ClassifierRatio,
    RewardRatio,
    MoleculePropertyEnergy,
    MultiPropertyEnergy,
    build_guidance_network,
    bregman_loss,
    ExactGuidance,
    CompositeGuidance,
    GuidanceModule,
    AmortizedPropertyGuidanceModule,
    LatentGuidanceModule,
    tanimoto_similarity,
)
from .feynman_kac import FeynmanKacSampler, JointGuidanceSampler, predict_clean
from .rl import (
    GDPOTrainer,
    RolloutSampler,
    RolloutBuffer,
    Reward,
    reward_from_energy,
    eager_logprob,
    kl_clean,
    group_advantage,
    EMA,
)
from .time_distortion import TimeDistorter
from .transformer import GraphTransformer
from .features import ExtraFeatures, RRWPFeatures
from .loss import TrainLoss, compute_loss_components
from .layers import XEyTransformerLayer, NodeEdgeBlock, timestep_embedding

__all__ = [
    # Main model
    "DeFoGModel",
    # Callbacks
    "TrainingMonitorCallback",
    "SampleVisualizationCallback",
    "EMACallback",
    # Graph-domain adapters (decode / visualize / evaluate)
    "GraphDomain",
    "GenericGraphDomain",
    "generation_metrics",
    # Data utilities
    "PlaceHolder",
    "to_dense",
    "dense_to_pyg",
    "DistributionNodes",
    "compute_dataset_statistics",
    "encode_no_edge",
    "symmetrize_edges",
    # Size distributions
    "SizeDistribution",
    "EmpiricalSizeDistribution",
    "FixedSizeDistribution",
    "ExplicitSizeDistribution",
    "UniformSizeDistribution",
    "CategoricalSizeDistribution",
    "ConditionalSizeDistribution",
    # Noise distribution
    "LimitDistribution",
    "sample_noise",
    "sample_from_probs",
    # Rate matrix
    "RateMatrixDesigner",
    # Sampling orchestration
    "Sampler",
    "InpaintingSampler",
    "GuidedSampler",
    "AdaptedSampler",
    "Modulation",
    "AdaLNAdapter",
    "ConditionBranch",
    "AdapterComposition",
    "AdapterRegistry",
    "AdapterModule",
    "RefinementSampler",
    # Constraints (inpainting)
    "Constraint",
    "SubgraphConstraint",
    # Exact discrete guidance (arXiv:2509.21912)
    "DensityRatio",
    "EnergyRatio",
    "ClassifierRatio",
    "RewardRatio",
    "MoleculePropertyEnergy",
    "build_guidance_network",
    "bregman_loss",
    "ExactGuidance",
    "CompositeGuidance",
    "MultiPropertyEnergy",
    "GuidanceModule",
    "AmortizedPropertyGuidanceModule",
    "LatentGuidanceModule",
    "tanimoto_similarity",
    "FeynmanKacSampler",
    "JointGuidanceSampler",
    "predict_clean",
    # GDPO reinforcement-learning fine-tuning (arXiv:2402.16302)
    "GDPOTrainer",
    "RolloutSampler",
    "RolloutBuffer",
    "Reward",
    "reward_from_energy",
    "eager_logprob",
    "kl_clean",
    "group_advantage",
    "EMA",
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
