"""
Limit distribution and noise sampling for discrete flow matching.

This module provides:
- LimitDistribution: Defines the noise distribution at t=0
- sample_noise: Sample initial noise from the limit distribution
- sample_from_probs: Sample discrete features from probability distributions
"""

import torch
import torch.nn.functional as F
from typing import Literal, Optional, Tuple

from .data import PlaceHolder


NoiseType = Literal["uniform", "marginal", "absorbing"]


class LimitDistribution:
    """
    Manages the limit (noise) distribution for discrete flow matching.

    The limit distribution defines what the noisy data looks like at t=0.
    Different noise types have different properties:

    - "uniform": Equal probability for all classes. Simple but may not match
                 data distribution well.
    - "marginal": Matches the empirical marginal distribution of the training data.
                  Generally recommended for best performance.
    - "absorbing": Adds a virtual "absorbing" class that receives all probability mass.
                   Useful for certain generation scenarios.

    Args:
        noise_type: Type of noise distribution ("uniform", "marginal", "absorbing")
        num_node_classes: Number of node feature classes
        num_edge_classes: Number of edge feature classes (including no-edge as class 0)
        node_marginals: Marginal probabilities for each node class (required for "marginal")
        edge_marginals: Marginal probabilities for each edge class (required for "marginal")

    Example:
        >>> # Uniform noise
        >>> limit_dist = LimitDistribution("uniform", num_node_classes=4, num_edge_classes=2)

        >>> # Marginal noise (recommended)
        >>> limit_dist = LimitDistribution(
        ...     "marginal",
        ...     num_node_classes=4,
        ...     num_edge_classes=2,
        ...     node_marginals=torch.tensor([0.25, 0.25, 0.25, 0.25]),
        ...     edge_marginals=torch.tensor([0.85, 0.15]),
        ... )
    """

    def __init__(
        self,
        noise_type: NoiseType,
        num_node_classes: int,
        num_edge_classes: int,
        node_marginals: Optional[torch.Tensor] = None,
        edge_marginals: Optional[torch.Tensor] = None,
    ):
        self.noise_type = noise_type
        self.num_node_classes_original = num_node_classes
        self.num_edge_classes_original = num_edge_classes
        self.x_added_classes = 0
        self.e_added_classes = 0

        if noise_type == "uniform":
            x_limit = torch.ones(num_node_classes) / num_node_classes
            e_limit = torch.ones(num_edge_classes) / num_edge_classes

        elif noise_type == "marginal":
            if node_marginals is None or edge_marginals is None:
                raise ValueError(
                    "node_marginals and edge_marginals are required for 'marginal' noise type"
                )
            # Normalize to ensure they sum to 1
            x_limit = node_marginals.float() / node_marginals.sum()
            e_limit = edge_marginals.float() / edge_marginals.sum()

        elif noise_type == "absorbing":
            # Add virtual absorbing state
            if num_node_classes > 1:
                self.x_added_classes = 1
            if num_edge_classes > 1:
                self.e_added_classes = 1

            x_limit = torch.zeros(num_node_classes + self.x_added_classes)
            x_limit[-1] = 1.0  # All mass on absorbing state

            e_limit = torch.zeros(num_edge_classes + self.e_added_classes)
            e_limit[-1] = 1.0  # All mass on absorbing state

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Store limit distributions
        self.X = x_limit
        self.E = e_limit
        self.y = torch.ones(1)  # Dummy for global features

        # Store effective number of classes (including virtual)
        self.num_node_classes = len(self.X)
        self.num_edge_classes = len(self.E)

    def to(self, device: torch.device) -> "LimitDistribution":
        """Move distributions to specified device."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device)
        return self

    def get_limit_dist(self) -> PlaceHolder:
        """Return limit distribution as a PlaceHolder."""
        return PlaceHolder(X=self.X, E=self.E, y=self.y)

    def ignore_virtual_classes(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Remove virtual absorbing state from outputs.

        Only has effect when noise_type="absorbing".

        Args:
            X: Node features with potential virtual class
            E: Edge features with potential virtual class
            y: Optional global features

        Returns:
            Tuple of (X, E, y) with virtual classes removed
        """
        if self.noise_type == "absorbing":
            if self.x_added_classes > 0:
                X = X[..., :-self.x_added_classes]
            if self.e_added_classes > 0:
                E = E[..., :-self.e_added_classes]
            if y is not None and self.e_added_classes > 0:
                y = y[..., :-self.e_added_classes]
        return X, E, y

    def add_virtual_classes(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Add virtual absorbing state to inputs.

        Only has effect when noise_type="absorbing".

        Args:
            X: Node features without virtual class
            E: Edge features without virtual class
            y: Optional global features

        Returns:
            Tuple of (X, E, y) with virtual classes added
        """
        if self.noise_type == "absorbing":
            if self.x_added_classes > 0:
                x_virtual = torch.zeros_like(X[..., :1]).repeat(
                    *([1] * (len(X.shape) - 1)), self.x_added_classes
                )
                X = torch.cat([X, x_virtual], dim=-1)

            if self.e_added_classes > 0:
                e_virtual = torch.zeros_like(E[..., :1]).repeat(
                    *([1] * (len(E.shape) - 1)), self.e_added_classes
                )
                E = torch.cat([E, e_virtual], dim=-1)

            if y is not None and self.e_added_classes > 0:
                y_virtual = torch.zeros_like(y[..., :1]).repeat(
                    *([1] * (len(y.shape) - 1)), self.e_added_classes
                )
                y = torch.cat([y, y_virtual], dim=-1)

        return X, E, y

    def __repr__(self) -> str:
        return (
            f"LimitDistribution(type={self.noise_type}, "
            f"node_classes={self.num_node_classes}, "
            f"edge_classes={self.num_edge_classes})"
        )


def sample_noise(
    limit_dist: LimitDistribution,
    node_mask: torch.Tensor
) -> PlaceHolder:
    """
    Sample from the limit (noise) distribution.

    Samples initial noise for the CTMC denoising process.

    Args:
        limit_dist: LimitDistribution object defining the noise distribution
        node_mask: Boolean tensor of shape (batch_size, max_nodes) indicating valid nodes

    Returns:
        PlaceHolder with one-hot encoded X, E, and empty y tensors
    """
    bs, n_max = node_mask.shape
    device = node_mask.device

    x_limit = limit_dist.X.to(device)
    e_limit = limit_dist.E.to(device)

    # Sample node features
    x_probs = x_limit.unsqueeze(0).unsqueeze(0).expand(bs, n_max, -1)
    U_X = x_probs.flatten(end_dim=-2).multinomial(1, replacement=True).reshape(bs, n_max)

    # Sample edge features
    e_probs = e_limit.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, n_max, n_max, -1)
    U_E = e_probs.flatten(end_dim=-2).multinomial(1, replacement=True).reshape(bs, n_max, n_max)

    # Make edges symmetric: keep upper triangular, mirror to lower
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=n_max, col=n_max, offset=1)
    upper_triangular_mask[:, indices[0], indices[1]] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    # Convert to one-hot
    X = F.one_hot(U_X, num_classes=len(x_limit)).float()
    E = F.one_hot(U_E, num_classes=len(e_limit)).float()
    y = torch.zeros(bs, 0, device=device)

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def sample_from_probs(
    prob_X: torch.Tensor,
    prob_E: torch.Tensor,
    node_mask: torch.Tensor,
) -> PlaceHolder:
    """
    Sample discrete features from probability distributions.

    Used during CTMC sampling to sample the next state from predicted probabilities.

    Args:
        prob_X: Node probabilities of shape (batch_size, n_nodes, num_classes)
        prob_E: Edge probabilities of shape (batch_size, n_nodes, n_nodes, num_classes)
        node_mask: Boolean mask of shape (batch_size, n_nodes)

    Returns:
        PlaceHolder with sampled class indices (not one-hot):
        - X: (batch_size, n_nodes) integer class indices
        - E: (batch_size, n_nodes, n_nodes) integer class indices
        - y: empty tensor
    """
    bs, n, _ = prob_X.shape
    device = prob_X.device

    # Clone to avoid modifying inputs
    prob_X = prob_X.clone()
    prob_E = prob_E.clone()

    # Set uniform distribution for masked (invalid) nodes
    prob_X[~node_mask] = 1 / prob_X.shape[-1]

    # Flatten and sample nodes
    prob_X_flat = prob_X.reshape(bs * n, -1)
    X_t = prob_X_flat.multinomial(1, replacement=True).reshape(bs, n)

    # Handle edge masking
    inverse_edge_mask = ~(node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)

    prob_E[inverse_edge_mask] = 1 / prob_E.shape[-1]
    prob_E[diag_mask] = 1 / prob_E.shape[-1]

    # Flatten and sample edges
    prob_E_flat = prob_E.reshape(bs * n * n, -1)
    E_t = prob_E_flat.multinomial(1, replacement=True).reshape(bs, n, n)

    # Make symmetric: keep upper triangular, mirror to lower
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0, device=device))
