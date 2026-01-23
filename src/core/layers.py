"""
Transformer building blocks for the graph transformer architecture.

This module provides the core attention layers used in GraphTransformer:
- XEyTransformerLayer: Main transformer layer updating nodes, edges, and global features
- NodeEdgeBlock: Self-attention with edge feature integration
- Xtoy, Etoy: Feature aggregation layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class Xtoy(nn.Module):
    """
    Aggregate node features to global features.

    Uses mean, min, max, and std pooling followed by a linear layer.
    """

    def __init__(self, dx: int, dy: int):
        """
        Args:
            dx: Input node feature dimension
            dy: Output global feature dimension
        """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: Node features of shape (batch_size, n_nodes, dx)

        Returns:
            Global features of shape (batch_size, dy)
        """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class Etoy(nn.Module):
    """
    Aggregate edge features to global features.

    Uses mean, min, max, and std pooling followed by a linear layer.
    """

    def __init__(self, de: int, dy: int):
        """
        Args:
            de: Input edge feature dimension
            dy: Output global feature dimension
        """
        super().__init__()
        self.lin = nn.Linear(4 * de, dy)

    def forward(self, E: Tensor) -> Tensor:
        """
        Args:
            E: Edge features of shape (batch_size, n_nodes, n_nodes, de)

        Returns:
            Global features of shape (batch_size, dy)
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


def masked_softmax(x: Tensor, mask: Tensor, **kwargs) -> Tensor:
    """
    Softmax with masking support.

    Args:
        x: Input tensor
        mask: Boolean or float mask (1 for valid, 0 for invalid)
        **kwargs: Additional arguments for torch.softmax (e.g., dim)

    Returns:
        Masked softmax output
    """
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class NodeEdgeBlock(nn.Module):
    """
    Self-attention block that updates node, edge, and global features.

    Implements multi-head attention with edge feature integration via FiLM
    (Feature-wise Linear Modulation).
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            dx: Node feature dimension
            de: Edge feature dimension
            dy: Global feature dimension
            n_head: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert dx % n_head == 0, f"dx ({dx}) must be divisible by n_head ({n_head})"

        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = dx // n_head
        self.n_head = n_head

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_X = nn.Dropout(dropout)
        self.dropout_E = nn.Dropout(dropout)

        # Query, Key, Value projections
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        # FiLM: Edge to attention modulation
        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)

        # FiLM: Global to edge modulation
        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)

        # FiLM: Global to node modulation
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        # Global feature processing
        self.y_y = nn.Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output projections
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(
            nn.Linear(dy, dy),
            nn.ReLU(),
            nn.Linear(dy, dy)
        )

    def forward(
        self,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        node_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the attention block.

        Args:
            X: Node features (batch_size, n_nodes, dx)
            E: Edge features (batch_size, n_nodes, n_nodes, de)
            y: Global features (batch_size, dy)
            node_mask: Boolean mask (batch_size, n_nodes)

        Returns:
            Tuple of (new_X, new_E, new_y) with same shapes as inputs
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # (bs, n, 1)
        e_mask1 = x_mask.unsqueeze(2)  # (bs, n, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)  # (bs, 1, n, 1)

        # 1. Compute queries and keys
        Q = self.q(X) * x_mask  # (bs, n, dx)
        K = self.k(X) * x_mask  # (bs, n, dx)

        # 2. Reshape for multi-head attention
        Q = Q.reshape(bs, n, self.n_head, self.df)
        K = K.reshape(bs, n, self.n_head, self.df)

        Q = Q.unsqueeze(2)  # (bs, n, 1, n_head, df)
        K = K.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # 3. Compute attention scores
        Y = Q * K  # (bs, n, n, n_head, df)
        Y = Y / math.sqrt(self.df)

        # 4. Incorporate edge features via FiLM
        E1 = self.e_mul(E) * e_mask1 * e_mask2  # (bs, n, n, dx)
        E1 = E1.reshape(bs, n, n, self.n_head, self.df)
        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = E2.reshape(bs, n, n, self.n_head, self.df)
        Y = Y * (E1 + 1) + E2

        # 5. Update edge features with global modulation
        newE = Y.flatten(start_dim=3)  # (bs, n, n, dx)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2  # (bs, n, n, de)

        # 6. Compute attention weights
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)

        # 7. Compute weighted values
        V = self.v(X) * x_mask
        V = V.reshape(bs, n, self.n_head, self.df)
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)
        weighted_V = weighted_V.flatten(start_dim=2)  # (bs, n, dx)

        # 8. Update node features with global modulation
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask

        # 9. Update global features
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)

        return newX, newE, new_y


class XEyTransformerLayer(nn.Module):
    """
    Full transformer layer for graph data.

    Combines self-attention (NodeEdgeBlock) with feedforward networks
    for node, edge, and global features.

    Args:
        dx: Node feature dimension
        de: Edge feature dimension
        dy: Global feature dimension
        n_head: Number of attention heads
        dim_ffX: Feedforward dimension for nodes
        dim_ffE: Feedforward dimension for edges
        dim_ffy: Feedforward dimension for global features
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head)

        # Node feedforward
        self.linX1 = nn.Linear(dx, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, dx)
        self.normX1 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)

        # Edge feedforward
        self.linE1 = nn.Linear(de, dim_ffE)
        self.linE2 = nn.Linear(dim_ffE, de)
        self.normE1 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)

        # Global feedforward
        self.lin_y1 = nn.Linear(dy, dim_ffy)
        self.lin_y2 = nn.Linear(dim_ffy, dy)
        self.norm_y1 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.norm_y2 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        node_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the transformer layer.

        Args:
            X: Node features (batch_size, n_nodes, dx)
            E: Edge features (batch_size, n_nodes, n_nodes, de)
            y: Global features (batch_size, dy)
            node_mask: Boolean mask (batch_size, n_nodes)

        Returns:
            Tuple of (new_X, new_E, new_y) with same shapes
        """
        # Self-attention
        newX, newE, new_y = self.self_attn(X, E, y, node_mask)

        # Node residual + norm
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        # Edge residual + norm
        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        # Global residual + norm
        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        # Node feedforward
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        # Edge feedforward
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        # Global feedforward
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1-D tensor of N indices (may be fractional)
        dim: Output embedding dimension
        max_period: Controls minimum frequency of embeddings

    Returns:
        Tensor of shape (N, dim) with positional embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
