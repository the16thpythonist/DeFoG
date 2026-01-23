"""
Extra features for graph transformer input augmentation.

This module provides structural features computed from the graph structure:
- RRWP (Relative Random Walk Probabilities): Encodes structural information via random walks
- Cycle features: Counts of k-cycles at each node and globally

These features enhance the expressivity of the graph transformer beyond
standard message passing.
"""

import torch
from typing import Dict, Literal, Optional, Tuple

from .data import PlaceHolder


FeatureType = Literal["none", "rrwp", "cycles"]


class ExtraFeatures:
    """
    Compute extra structural features from graph structure.

    Supports different feature types:
    - "none": No extra features (returns empty tensors)
    - "rrwp": Relative Random Walk Probabilities
    - "cycles": Cycle count features (3, 4, 5, 6-cycles)

    The RRWP features are particularly important as they provide structural
    information that enhances the transformer's ability to understand
    graph topology. They are 10-100x faster than spectral features.

    Args:
        feature_type: Type of features to compute
        rrwp_steps: Number of random walk steps for RRWP (default: 10)
        max_nodes: Maximum number of nodes (for normalization)

    Example:
        >>> features = ExtraFeatures("rrwp", rrwp_steps=10, max_nodes=64)
        >>> extra = features(noisy_data)
        >>> # extra.X: (bs, n, rrwp_steps) - node RRWP features
        >>> # extra.E: (bs, n, n, rrwp_steps) - edge RRWP features
        >>> # extra.y: (bs, 2+4) - normalized n + cycle counts
    """

    def __init__(
        self,
        feature_type: FeatureType = "rrwp",
        rrwp_steps: int = 10,
        max_nodes: int = 100,
    ):
        self.feature_type = feature_type
        self.rrwp_steps = rrwp_steps
        self.max_nodes = max_nodes

        self.rrwp = RRWPFeatures()
        self.cycles = NodeCycleFeatures()

    def __call__(self, noisy_data: Dict[str, torch.Tensor]) -> PlaceHolder:
        """
        Compute extra features from noisy graph data.

        Args:
            noisy_data: Dict containing:
                - "X_t": Node features (bs, n, dx)
                - "E_t": Edge features (bs, n, n, de) - one-hot
                - "y_t": Global features (bs, dy)
                - "node_mask": Boolean mask (bs, n)

        Returns:
            PlaceHolder with extra features to concatenate to inputs
        """
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        y = noisy_data["y_t"]
        node_mask = noisy_data["node_mask"]

        # Normalized node count
        n_normalized = node_mask.sum(dim=1, keepdim=True).float() / self.max_nodes

        if self.feature_type == "none":
            return self._empty_features(X, E, y)

        elif self.feature_type == "cycles":
            x_cycles, y_cycles = self.cycles(noisy_data)
            extra_edge = torch.zeros((*E.shape[:-1], 0), device=E.device, dtype=E.dtype)
            return PlaceHolder(
                X=x_cycles,
                E=extra_edge,
                y=torch.cat([n_normalized, y_cycles], dim=-1)
            )

        elif self.feature_type == "rrwp":
            # Convert one-hot edges to adjacency matrix (sum all edge types except no-edge)
            adj = E.float()[..., 1:].sum(-1)  # (bs, n, n)

            # Compute RRWP features
            rrwp_edge = self.rrwp(adj, k=self.rrwp_steps)  # (bs, n, n, k)

            # Extract diagonal for node features
            diag_idx = torch.arange(rrwp_edge.shape[1], device=rrwp_edge.device)
            rrwp_node = rrwp_edge[:, diag_idx, diag_idx, :]  # (bs, n, k)

            # Compute cycle features for global
            x_cycles, y_cycles = self.cycles(noisy_data)

            return PlaceHolder(
                X=rrwp_node,
                E=rrwp_edge,
                y=torch.cat([n_normalized, y_cycles], dim=-1)
            )

        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _empty_features(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor
    ) -> PlaceHolder:
        """Return empty feature tensors."""
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return PlaceHolder(X=empty_x, E=empty_e, y=empty_y)

    def output_dims(self) -> Dict[str, int]:
        """
        Return the output dimensions of extra features.

        Returns:
            Dict with "X", "E", "y" dimensions
        """
        if self.feature_type == "none":
            return {"X": 0, "E": 0, "y": 0}
        elif self.feature_type == "cycles":
            return {"X": 3, "E": 0, "y": 5}  # 1 (n) + 4 (cycles)
        elif self.feature_type == "rrwp":
            return {"X": self.rrwp_steps, "E": self.rrwp_steps, "y": 5}
        else:
            return {"X": 0, "E": 0, "y": 0}

    def __repr__(self) -> str:
        return f"ExtraFeatures(type={self.feature_type}, rrwp_steps={self.rrwp_steps})"


class RRWPFeatures:
    """
    Compute Relative Random Walk Probabilities.

    RRWP encodes structural information via powers of the normalized adjacency:
    [I, M, M^2, ..., M^(k-1)] where M = D^(-1) @ A (degree-normalized adjacency).

    This provides information about:
    - Local structure (low powers)
    - Global connectivity (higher powers)

    Args:
        normalize: Whether to normalize by degree (default: True)
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, adj: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Compute RRWP features.

        Args:
            adj: Adjacency matrix (bs, n, n)
            k: Number of random walk steps

        Returns:
            RRWP tensor of shape (bs, n, n, k)
        """
        bs, n, _ = adj.shape
        device = adj.device

        if self.normalize:
            # Compute degree-normalized adjacency: D^(-1) @ A
            degree = adj.sum(dim=-1)  # (bs, n)
            degree_inv = torch.zeros(bs, n, n, device=device)
            safe_degree = degree.clone()
            safe_degree[degree == 0] = 1  # Avoid division by zero
            degree_inv = torch.diag_embed(1.0 / safe_degree)
            degree_inv[degree.unsqueeze(-1).expand(-1, -1, n) == 0] = 0
            adj = degree_inv @ adj

        # Compute powers: I, A, A^2, ..., A^(k-1)
        identity = torch.eye(n, device=device).unsqueeze(0).expand(bs, -1, -1)
        rrwp_list = [identity]

        current = identity
        for _ in range(k - 1):
            current = current @ adj
            rrwp_list.append(current)

        return torch.stack(rrwp_list, dim=-1)  # (bs, n, n, k)


class NodeCycleFeatures:
    """
    Compute cycle count features at each node.

    Counts the number of k-cycles (k=3,4,5,6) passing through each node
    and globally in the graph.
    """

    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cycle features.

        Args:
            noisy_data: Dict with "E_t" (one-hot edges) and "node_mask"

        Returns:
            Tuple of (x_cycles, y_cycles):
            - x_cycles: (bs, n, 3) - node-level 3,4,5-cycle counts
            - y_cycles: (bs, 4) - global 3,4,5,6-cycle counts
        """
        # Convert one-hot to adjacency
        adj = noisy_data["E_t"][..., 1:].sum(dim=-1).float()
        node_mask = noisy_data["node_mask"]

        x_cycles, y_cycles = self.kcycles.k_cycles(adj)

        # Apply mask and normalize
        x_cycles = x_cycles * node_mask.unsqueeze(-1)
        x_cycles = x_cycles / 10  # Avoid large values
        y_cycles = y_cycles / 10

        # Clamp to reasonable range
        x_cycles = x_cycles.clamp(max=1)
        y_cycles = y_cycles.clamp(max=1)

        return x_cycles, y_cycles


class KNodeCycles:
    """
    Compute k-cycle counts using matrix powers.

    Uses the fact that tr(A^k) counts k-cycles (with appropriate corrections).
    """

    def k_cycles(self, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cycle counts.

        Args:
            adj_matrix: Adjacency matrix (bs, n, n)

        Returns:
            Tuple of (x_cycles, y_cycles)
        """
        self.adj = adj_matrix.float()
        self._compute_powers()

        k3x, k3y = self._k3_cycle()
        k4x, k4y = self._k4_cycle()
        k5x, k5y = self._k5_cycle()
        _, k6y = self._k6_cycle()

        x_cycles = torch.cat([k3x, k4x, k5x], dim=-1)
        y_cycles = torch.cat([k3y, k4y, k5y, k6y], dim=-1)

        return x_cycles, y_cycles

    def _compute_powers(self):
        """Compute matrix powers up to A^6."""
        self.d = self.adj.sum(dim=-1)  # Degree
        self.A2 = self.adj @ self.adj
        self.A3 = self.A2 @ self.adj
        self.A4 = self.A3 @ self.adj
        self.A5 = self.A4 @ self.adj
        self.A6 = self.A5 @ self.adj

    def _diag(self, M: torch.Tensor) -> torch.Tensor:
        """Extract diagonal."""
        return torch.diagonal(M, dim1=-2, dim2=-1)

    def _trace(self, M: torch.Tensor) -> torch.Tensor:
        """Compute trace."""
        return self._diag(M).sum(dim=-1)

    def _k3_cycle(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Count 3-cycles (triangles)."""
        c3 = self._diag(self.A3)
        x = (c3 / 2).unsqueeze(-1)
        y = (c3.sum(dim=-1) / 6).unsqueeze(-1)
        return x, y

    def _k4_cycle(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Count 4-cycles."""
        diag_a4 = self._diag(self.A4)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.adj @ self.d.unsqueeze(-1)).squeeze(-1)
        )
        x = (c4 / 2).unsqueeze(-1)
        y = (c4.sum(dim=-1) / 8).unsqueeze(-1)
        return x, y

    def _k5_cycle(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Count 5-cycles."""
        diag_a5 = self._diag(self.A5)
        triangles = self._diag(self.A3)
        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.adj @ triangles.unsqueeze(-1)).squeeze(-1)
            + triangles
        )
        x = (c5 / 2).unsqueeze(-1)
        y = (c5.sum(dim=-1) / 10).unsqueeze(-1)
        return x, y

    def _k6_cycle(self) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Count 6-cycles (global only)."""
        term1 = self._trace(self.A6)
        term2 = self._trace(self.A3 ** 2)
        term3 = (self.adj * self.A2.pow(2)).sum(dim=[-2, -1])
        d_2 = self._diag(self.A2)
        a_4 = self._diag(self.A4)
        term4 = (d_2 * a_4).sum(dim=-1)
        term5 = self._trace(self.A4)
        term6 = self._trace(self.A3)
        term7 = d_2.pow(3).sum(-1)
        term8 = self.A3.sum(dim=[-2, -1])
        term9 = d_2.pow(2).sum(-1)
        term10 = self._trace(self.A2)

        c6 = (
            term1 - 3 * term2 + 9 * term3 - 6 * term4
            + 6 * term5 - 4 * term6 + 4 * term7
            + 3 * term8 - 12 * term9 + 4 * term10
        )
        y = (c6 / 12).unsqueeze(-1)
        return None, y
