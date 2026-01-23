"""
DeFoG Model - Main LightningModule for discrete flow matching graph generation.

This module provides the main model class that integrates all components:
- Training: Learn to predict clean graph marginals from noisy inputs
- Sampling: Generate graphs via CTMC denoising with configurable parameters

Example:
    >>> from src.core import DeFoGModel
    >>> model = DeFoGModel(
    ...     num_node_classes=4,
    ...     num_edge_classes=2,
    ...     n_layers=6,
    ... )
    >>> trainer = pl.Trainer(max_epochs=100)
    >>> trainer.fit(model, train_dataloader)
    >>> samples = model.sample(num_samples=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

from .data import PlaceHolder, to_dense, dense_to_pyg, DistributionNodes
from .transformer import GraphTransformer
from .features import ExtraFeatures
from .noise import LimitDistribution, sample_noise, sample_from_probs
from .rate_matrix import RateMatrixDesigner
from .time_distortion import TimeDistorter
from .loss import TrainLoss


class DeFoGModel(pl.LightningModule):
    """
    Discrete Flow Matching for Graph Generation.

    A PyTorch Lightning module implementing the DeFoG model for generating
    discrete graphs. The model learns to predict clean graph marginals from
    noisy inputs during training, and generates graphs via CTMC denoising
    at sampling time.

    Args:
        num_node_classes: Number of node feature classes
        num_edge_classes: Number of edge feature classes (including no-edge)
        n_layers: Number of transformer layers (default: 6)
        hidden_dim: Hidden dimension for transformer (default: 256)
        hidden_mlp_dim: Hidden dimension for MLPs (default: 512)
        n_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        noise_type: Type of noise distribution - "uniform", "marginal", or "absorbing"
        node_marginals: Prior marginals for nodes (required if noise_type="marginal")
        edge_marginals: Prior marginals for edges (required if noise_type="marginal")
        node_counts: Tensor of node count frequencies for sampling graph sizes
        max_nodes: Maximum number of nodes in graphs (default: 100)
        extra_features_type: Type of extra features - "none", "rrwp", or "cycles"
        rrwp_steps: Number of RRWP random walk steps (default: 10)
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay for optimizer (default: 1e-5)
        lambda_edge: Weight for edge loss (default: 1.0)
        train_time_distortion: Time distortion for training (default: "identity")
        sample_steps: Number of sampling steps (default: 100)
        eta: Stochasticity parameter for sampling (default: 0.0)
        omega: Target guidance strength for sampling (default: 0.0)
        sample_time_distortion: Time distortion for sampling (default: "identity")

    Example:
        >>> model = DeFoGModel(
        ...     num_node_classes=4,
        ...     num_edge_classes=2,
        ...     node_counts=torch.tensor([0, 0.1, 0.3, 0.4, 0.2]),  # 2-4 nodes
        ... )
        >>> # Training
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(model, train_dataloader)
        >>> # Sampling
        >>> samples = model.sample(num_samples=10)
    """

    def __init__(
        self,
        # Required dimensions
        num_node_classes: int,
        num_edge_classes: int,
        # Architecture
        n_layers: int = 6,
        hidden_dim: int = 256,
        hidden_mlp_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        # Noise distribution
        noise_type: str = "uniform",
        node_marginals: Optional[torch.Tensor] = None,
        edge_marginals: Optional[torch.Tensor] = None,
        # Graph sizes (for sampling)
        node_counts: Optional[torch.Tensor] = None,
        max_nodes: int = 100,
        # Extra features
        extra_features_type: str = "rrwp",
        rrwp_steps: int = 10,
        # Training
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_edge: float = 1.0,
        train_time_distortion: str = "identity",
        # Sampling defaults
        sample_steps: int = 100,
        eta: float = 0.0,
        omega: float = 0.0,
        sample_time_distortion: str = "identity",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store dimensions
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.max_nodes = max_nodes

        # Store training params
        self.lr = lr
        self.weight_decay = weight_decay

        # Store sampling params
        self.sample_steps = sample_steps
        self.eta = eta
        self.omega = omega
        self.sample_time_distortion = sample_time_distortion

        # Create limit distribution
        self.limit_dist = LimitDistribution(
            noise_type=noise_type,
            num_node_classes=num_node_classes,
            num_edge_classes=num_edge_classes,
            node_marginals=node_marginals,
            edge_marginals=edge_marginals,
        )

        # Get actual dimensions (may include virtual class for absorbing)
        actual_node_classes = self.limit_dist.num_node_classes
        actual_edge_classes = self.limit_dist.num_edge_classes

        # Create node distribution for sampling
        if node_counts is not None:
            self.node_dist = DistributionNodes(node_counts)
        else:
            # Default: uniform over 2-20 nodes
            counts = torch.zeros(max_nodes + 1)
            counts[2:21] = 1.0
            self.node_dist = DistributionNodes(counts)

        # Create extra features
        self.extra_features = ExtraFeatures(
            feature_type=extra_features_type,
            rrwp_steps=rrwp_steps,
            max_nodes=max_nodes,
        )
        extra_dims = self.extra_features.output_dims()

        # Compute input/output dimensions
        # Input: node/edge classes + extra features + time
        self.input_dims = {
            "X": actual_node_classes + extra_dims["X"],
            "E": actual_edge_classes + extra_dims["E"],
            "y": 1 + extra_dims["y"],  # time + extra global features
        }
        self.output_dims = {
            "X": actual_node_classes,
            "E": actual_edge_classes,
            "y": 0,
        }

        # Create transformer
        self.model = GraphTransformer(
            n_layers=n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims={"X": hidden_mlp_dim, "E": hidden_mlp_dim // 2, "y": hidden_mlp_dim},
            hidden_dims={
                "dx": hidden_dim,
                "de": hidden_dim // 4,
                "dy": hidden_dim // 4,
                "n_head": n_heads,
                "dim_ffX": hidden_dim * 2,
                "dim_ffE": hidden_dim // 2,
            },
            output_dims=self.output_dims,
            dropout=dropout,
        )

        # Create loss function
        self.train_loss = TrainLoss(lambda_edge=lambda_edge)

        # Create time distorter
        self.time_distorter = TimeDistorter(
            train_distortion=train_time_distortion,
            sample_distortion=sample_time_distortion,
        )

        # Create rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            eta=eta,
            omega=omega,
            limit_dist=self.limit_dist,
        )

    def forward(
        self,
        noisy_data: Dict[str, torch.Tensor],
        extra_data: PlaceHolder,
        node_mask: torch.Tensor,
    ) -> PlaceHolder:
        """
        Forward pass through the transformer.

        Args:
            noisy_data: Dict with "X_t", "E_t", "y_t", "t", "node_mask"
            extra_data: PlaceHolder with extra features
            node_mask: Boolean mask (bs, n)

        Returns:
            PlaceHolder with predicted marginals
        """
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for a batch of PyG graphs.

        Args:
            batch: PyG Batch object
            batch_idx: Batch index

        Returns:
            Dict with "loss" key
        """
        if batch.edge_index.numel() == 0:
            return None

        # Convert PyG batch to dense tensors
        dense_data, node_mask = to_dense(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # Get y (global features) if available
        if hasattr(batch, 'y') and batch.y is not None:
            y = batch.y
            if y.dim() == 1:
                y = y.unsqueeze(-1)
        else:
            y = torch.zeros(X.size(0), 0, device=X.device)

        # Apply noise
        noisy_data = self._apply_noise(X, E, y, node_mask)

        # Compute extra features
        extra_data = self._compute_extra_data(noisy_data)

        # Forward pass
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Compute loss
        loss = self.train_loss(
            pred_X=pred.X,
            pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=y,
            node_mask=node_mask,
        )

        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_nodes: Optional[Union[int, torch.Tensor]] = None,
        eta: Optional[float] = None,
        omega: Optional[float] = None,
        sample_steps: Optional[int] = None,
        time_distortion: Optional[str] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> List[Data]:
        """
        Generate graph samples.

        Args:
            num_samples: Number of graphs to generate
            num_nodes: Fixed number of nodes (int) or per-sample counts (Tensor).
                      If None, samples from node distribution.
            eta: Stochasticity parameter (overrides default)
            omega: Target guidance strength (overrides default)
            sample_steps: Number of sampling steps (overrides default)
            time_distortion: Time distortion type (overrides default)
            device: Device to sample on (defaults to model device)
            show_progress: Whether to show progress bar

        Returns:
            List of PyG Data objects, each containing:
            - x: Node features (n, num_node_classes) one-hot
            - edge_index: Edge indices (2, num_edges)
            - edge_attr: Edge features (num_edges, num_edge_classes) one-hot
        """
        # Use defaults if not specified
        eta = eta if eta is not None else self.eta
        omega = omega if omega is not None else self.omega
        sample_steps = sample_steps if sample_steps is not None else self.sample_steps
        time_distortion = time_distortion if time_distortion is not None else self.sample_time_distortion
        device = device if device is not None else self.device

        # Update rate matrix designer parameters
        self.rate_matrix_designer.eta = eta
        self.rate_matrix_designer.omega = omega

        # Sample number of nodes
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(num_samples, device)
        elif isinstance(num_nodes, int):
            n_nodes = num_nodes * torch.ones(num_samples, device=device, dtype=torch.int)
        else:
            n_nodes = num_nodes.to(device)
        n_max = n_nodes.max().item()

        # Build node mask
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(num_samples, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample initial noise
        z_T = sample_noise(self.limit_dist, node_mask)
        X, E = z_T.X, z_T.E
        y = torch.zeros(num_samples, 0, device=device)

        # Sampling loop
        iterator = range(sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for t_int in iterator:
            # Current and next time
            t_array = t_int * torch.ones((num_samples, 1), device=device)
            t_norm = t_array / sample_steps
            s_array = t_array + 1
            s_norm = s_array / sample_steps

            # Handle absorbing transition edge case
            if self.limit_dist.noise_type == "absorbing" and t_int == 0:
                t_norm = t_norm + 1e-6

            # Apply time distortion
            t_norm = self.time_distorter.sample_ft(t_norm, time_distortion)
            s_norm = self.time_distorter.sample_ft(s_norm, time_distortion)

            # Sample next state
            X, E, y = self._sample_step(t_norm, s_norm, X, E, y, node_mask)

        # Final cleanup: remove virtual classes if absorbing
        X, E, _ = self.limit_dist.ignore_virtual_classes(X, E)

        # Convert to PyG Data objects
        samples = dense_to_pyg(X, E, y, node_mask, n_nodes)

        return samples

    def _apply_noise(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply noise to clean data for training.

        Args:
            X: Clean node features one-hot (bs, n, dx)
            E: Clean edge features one-hot (bs, n, n, de)
            y: Global features (bs, dy)
            node_mask: Boolean mask (bs, n)
            t: Optional time tensor; if None, samples from time distorter

        Returns:
            Dict with noisy data and metadata
        """
        bs = X.size(0)
        device = X.device

        # Sample timestep
        if t is None:
            t_float = self.time_distorter.train_ft(bs, device)
        else:
            t_float = t

        # Get class indices
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)

        # Compute p(x_t | x_1) = t * one_hot(x_1) + (1-t) * limit_dist
        prob_X_t, prob_E_t = self._p_xt_given_x1(X_1_label, E_1_label, t_float)

        # Sample from the noisy distribution
        sampled_t = sample_from_probs(prob_X_t, prob_E_t, node_mask)

        # Convert to one-hot
        X_t = F.one_hot(sampled_t.X, num_classes=self.limit_dist.num_node_classes).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.limit_dist.num_edge_classes).float()

        # Create placeholder and mask
        z_t = PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def _p_xt_given_x1(
        self,
        X1: torch.Tensor,
        E1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute p(x_t | x_1) for the linear interpolation.

        p(x_t | x_1) = t * delta(x_t, x_1) + (1-t) * p_0(x_t)

        Args:
            X1: Node class indices (bs, n)
            E1: Edge class indices (bs, n, n)
            t: Time in [0, 1] (bs, 1)

        Returns:
            Tuple of (prob_X_t, prob_E_t) probability distributions
        """
        device = X1.device
        limit_X = self.limit_dist.X.to(device)
        limit_E = self.limit_dist.E.to(device)

        t_time = t.squeeze(-1)[:, None, None]
        X1_onehot = F.one_hot(X1, num_classes=len(limit_X)).float()
        E1_onehot = F.one_hot(E1, num_classes=len(limit_E)).float()

        prob_X = t_time * X1_onehot + (1 - t_time) * limit_X[None, None, :]
        prob_E = (
            t_time[:, None] * E1_onehot
            + (1 - t_time[:, None]) * limit_E[None, None, None, :]
        )

        return prob_X.clamp(min=0.0, max=1.0), prob_E.clamp(min=0.0, max=1.0)

    def _sample_step(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        y_t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample z_s given z_t (one CTMC step).

        Args:
            t: Current time (bs, 1)
            s: Next time (bs, 1)
            X_t: Current node features one-hot (bs, n, dx)
            E_t: Current edge features one-hot (bs, n, n, de)
            y_t: Current global features (bs, dy)
            node_mask: Boolean mask (bs, n)

        Returns:
            Tuple of (X_s, E_s, y_s) for next state
        """
        bs, n, dx = X_t.shape
        de = E_t.shape[-1]
        dt = (s - t)[0]
        device = X_t.device

        # Create noisy data dict
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        # Compute extra features and forward pass
        extra_data = self._compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)

        # Compute rate matrices
        R_t_X, R_t_E = self.rate_matrix_designer.compute_rate_matrices(
            t, node_mask, X_t, E_t, pred_X, pred_E
        )

        # Compute step probabilities
        prob_X, prob_E = self._compute_step_probs(R_t_X, R_t_E, X_t, E_t, dt)

        # At final step, use predicted marginals directly
        if s[0].item() >= 1.0 - 1e-6:
            prob_X, prob_E = pred_X, pred_E

        # Sample next state
        sampled_s = sample_from_probs(prob_X, prob_E, node_mask)

        limit_X = self.limit_dist.X.to(device)
        limit_E = self.limit_dist.E.to(device)

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_X)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_E)).float()

        return X_s, E_s, y_t

    def _compute_step_probs(
        self,
        R_t_X: torch.Tensor,
        R_t_E: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute transition probabilities from rate matrices.

        P(x_s | x_t) ≈ I + R_t * dt for small dt

        Args:
            R_t_X: Node rate matrix (bs, n, num_classes)
            R_t_E: Edge rate matrix (bs, n, n, num_classes)
            X_t: Current node features one-hot (bs, n, dx)
            E_t: Current edge features one-hot (bs, n, n, de)
            dt: Time step

        Returns:
            Tuple of (prob_X, prob_E) transition probabilities
        """
        step_probs_X = R_t_X * dt
        step_probs_E = R_t_E * dt

        # Zero out diagonal entries
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # Set diagonal such that rows sum to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        return step_probs_X, step_probs_E

    def _compute_extra_data(self, noisy_data: Dict[str, torch.Tensor]) -> PlaceHolder:
        """
        Compute extra features from noisy data.

        Args:
            noisy_data: Dict with "X_t", "E_t", "y_t", "t", "node_mask"

        Returns:
            PlaceHolder with extra features
        """
        extra_features = self.extra_features(noisy_data)

        # Append time to global features
        t = noisy_data["t"]
        extra_y = torch.cat((extra_features.y, t), dim=1)

        return PlaceHolder(X=extra_features.X, E=extra_features.E, y=extra_y)

    @classmethod
    def from_dataloader(
        cls,
        dataloader,
        n_layers: int = 6,
        hidden_dim: int = 256,
        noise_type: str = "marginal",
        **kwargs,
    ) -> "DeFoGModel":
        """
        Create a model with dimensions inferred from a dataloader.

        This is a convenience method that analyzes a dataloader to determine:
        - Number of node/edge classes
        - Node count distribution
        - Marginal distributions (if noise_type="marginal")
        - Maximum number of nodes

        Args:
            dataloader: PyG DataLoader
            n_layers: Number of transformer layers
            hidden_dim: Hidden dimension
            noise_type: Type of noise distribution
            **kwargs: Additional arguments passed to constructor

        Returns:
            Configured DeFoGModel instance

        Example:
            >>> from torch_geometric.loader import DataLoader
            >>> loader = DataLoader(dataset, batch_size=32)
            >>> model = DeFoGModel.from_dataloader(loader, n_layers=6)
        """
        from .data import compute_dataset_statistics

        # Compute statistics from dataloader
        stats = compute_dataset_statistics(dataloader)

        # Prepare constructor arguments
        model_kwargs = {
            "num_node_classes": stats["num_node_classes"],
            "num_edge_classes": stats["num_edge_classes"],
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "noise_type": noise_type,
            "node_counts": stats["node_counts"],
            "max_nodes": stats["max_nodes"],
        }

        # Add marginals if using marginal noise
        if noise_type == "marginal":
            model_kwargs["node_marginals"] = stats["node_marginals"]
            model_kwargs["edge_marginals"] = stats["edge_marginals"]

        # Override with any additional kwargs
        model_kwargs.update(kwargs)

        return cls(**model_kwargs)

    def save(self, path: str) -> str:
        """
        Save the model to a checkpoint file.

        Saves model weights and hyperparameters (no optimizer state).
        The model can be restored using DeFoGModel.load().

        Args:
            path: File path to save to. If no .ckpt extension is provided,
                  it will be appended automatically.

        Returns:
            The actual path the model was saved to (with extension).

        Example:
            >>> model.save("my_model")  # Saves to my_model.ckpt
            >>> model.save("checkpoints/model.ckpt")
        """
        # Ensure .ckpt extension
        if not path.endswith(".ckpt"):
            path = path + ".ckpt"

        # Save checkpoint with weights and hyperparameters only
        checkpoint = {
            "state_dict": self.state_dict(),
            "hyper_parameters": dict(self.hparams),
        }
        torch.save(checkpoint, path)

        return path

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[Union[str, torch.device]] = "cpu",
    ) -> "DeFoGModel":
        """
        Load a model from a checkpoint file.

        Args:
            path: Path to the checkpoint file. If no .ckpt extension is
                  provided, it will be appended automatically.
            device: Device to load the model to. Defaults to "cpu".
                    Can be "cpu", "cuda", "cuda:0", etc.

        Returns:
            Loaded DeFoGModel instance ready for inference or further training.

        Example:
            >>> model = DeFoGModel.load("my_model.ckpt")
            >>> model = DeFoGModel.load("my_model", device="cuda")
            >>> samples = model.sample(num_samples=10)
        """
        # Ensure .ckpt extension
        if not path.endswith(".ckpt"):
            path = path + ".ckpt"

        # Load checkpoint (weights_only=False needed for hyperparameters with tensors)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle both our format and Lightning's format
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
        elif "hparams" in checkpoint:
            hparams = checkpoint["hparams"]
        else:
            raise ValueError(
                f"Checkpoint at {path} does not contain hyperparameters. "
                "Cannot reconstruct model."
            )

        # Create model instance
        model = cls(**hparams)

        # Load state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError(
                f"Checkpoint at {path} does not contain state_dict."
            )

        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()

        return model

    def __repr__(self) -> str:
        return (
            f"DeFoGModel(node_classes={self.num_node_classes}, "
            f"edge_classes={self.num_edge_classes}, "
            f"n_layers={self.model.n_layers})"
        )
