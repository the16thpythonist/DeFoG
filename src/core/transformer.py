"""
Graph Transformer architecture for discrete flow matching.

This module provides the main neural network architecture that predicts
clean graph marginals from noisy inputs.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .data import PlaceHolder
from .layers import XEyTransformerLayer, timestep_embedding


class GraphTransformer(nn.Module):
    """
    Graph Transformer network for predicting graph marginals.

    The transformer processes node features (X), edge features (E), and
    global features (y) through multiple layers of self-attention with
    edge feature integration.

    The network includes:
    - Input MLPs to project features to hidden dimensions
    - Timestep embedding for time conditioning
    - Stack of XEyTransformerLayer blocks
    - Output MLPs to project back to original dimensions
    - Skip connections from input to output

    Args:
        n_layers: Number of transformer layers
        input_dims: Dict with input dimensions {"X": int, "E": int, "y": int}
        hidden_mlp_dims: Dict with MLP hidden dimensions {"X": int, "E": int, "y": int}
        hidden_dims: Dict with transformer dimensions:
            - "dx": Node hidden dimension
            - "de": Edge hidden dimension
            - "dy": Global hidden dimension
            - "n_head": Number of attention heads
            - "dim_ffX": Node feedforward dimension
            - "dim_ffE": Edge feedforward dimension
        output_dims: Dict with output dimensions {"X": int, "E": int, "y": int}
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> transformer = GraphTransformer(
        ...     n_layers=6,
        ...     input_dims={"X": 16, "E": 8, "y": 3},
        ...     hidden_mlp_dims={"X": 256, "E": 128, "y": 256},
        ...     hidden_dims={
        ...         "dx": 256, "de": 64, "dy": 64,
        ...         "n_head": 8, "dim_ffX": 512, "dim_ffE": 128
        ...     },
        ...     output_dims={"X": 4, "E": 2, "y": 0},
        ... )
        >>> output = transformer(X, E, y, node_mask)
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: Dict[str, int],
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims.get("y", 0)

        # Time embedding dimension
        self.time_emb_dim = 64

        # Input MLPs
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            nn.ReLU(),
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            nn.ReLU(),
        )

        # y input includes time embedding
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"] + self.time_emb_dim, hidden_mlp_dims["y"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            nn.ReLU(),
        )

        # Transformer layers
        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=hidden_dims["dx"],
                de=hidden_dims["de"],
                dy=hidden_dims["dy"],
                n_head=hidden_dims["n_head"],
                dim_ffX=hidden_dims.get("dim_ffX", hidden_dims["dx"] * 2),
                dim_ffE=hidden_dims.get("dim_ffE", hidden_dims["de"] * 2),
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output MLPs
        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        if self.out_dim_y > 0:
            self.mlp_out_y = nn.Sequential(
                nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
                nn.ReLU(),
                nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
            )
        else:
            self.mlp_out_y = None

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> PlaceHolder:
        """
        Forward pass through the graph transformer.

        Args:
            X: Node features (batch_size, n_nodes, input_dims["X"])
               Last dimension should include time as the last feature
            E: Edge features (batch_size, n_nodes, n_nodes, input_dims["E"])
            y: Global features (batch_size, input_dims["y"])
               Last element should be the time t in [0, 1]
            node_mask: Boolean mask (batch_size, n_nodes)

        Returns:
            PlaceHolder with predicted marginals:
            - X: (batch_size, n_nodes, output_dims["X"])
            - E: (batch_size, n_nodes, n_nodes, output_dims["E"])
            - y: (batch_size, output_dims["y"]) or empty
        """
        bs, n = X.shape[0], X.shape[1]

        # Create diagonal mask for edge symmetrization
        diag_mask = torch.eye(n, device=X.device, dtype=torch.bool)
        diag_mask = ~diag_mask
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # Skip connections: store input for residual
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y] if self.out_dim_y > 0 else None

        # Process edges with symmetrization
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # Add sinusoidal time embedding
        # Assume last element of y is the time
        time = y[:, -1:] if y.shape[-1] > 0 else torch.zeros(bs, 1, device=X.device)
        time_emb = timestep_embedding(time, self.time_emb_dim)
        y_with_time = torch.cat([y, time_emb], dim=-1)

        # Input projections with masking
        X = self.mlp_in_X(X)
        E = new_E
        y = self.mlp_in_y(y_with_time)

        # Apply node mask
        x_mask = node_mask.unsqueeze(-1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)
        X = X * x_mask
        E = E * e_mask

        # Transformer layers
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # Output projections
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)

        # Add skip connections
        X = X + X_to_out
        E = (E + E_to_out) * diag_mask  # Zero out diagonal

        # Symmetrize edges
        E = (E + E.transpose(1, 2)) / 2

        # Handle global output
        if self.mlp_out_y is not None and y_to_out is not None:
            y_out = self.mlp_out_y(y) + y_to_out
        else:
            y_out = torch.zeros(bs, 0, device=X.device)

        # Apply final masking
        result = PlaceHolder(X=X, E=E, y=y_out)
        return result.mask(node_mask)

    def __repr__(self) -> str:
        return (
            f"GraphTransformer(n_layers={self.n_layers}, "
            f"out_X={self.out_dim_X}, out_E={self.out_dim_E}, out_y={self.out_dim_y})"
        )
