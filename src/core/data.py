"""
Data structures and conversion utilities for DeFoG core module.

This module provides:
- PlaceHolder: Container for graph tensors (X: nodes, E: edges, y: global)
- to_dense: Convert PyG Batch to dense tensors
- dense_to_pyg: Convert dense tensors back to list of PyG Data objects
- DistributionNodes: Sample graph sizes from a distribution
"""

import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass


class PlaceHolder:
    """
    Container for batched graph tensors in dense format.

    Attributes:
        X: Node features tensor of shape (batch_size, max_nodes, num_node_features)
        E: Edge features tensor of shape (batch_size, max_nodes, max_nodes, num_edge_features)
        y: Global features tensor of shape (batch_size, num_global_features) or None
    """

    def __init__(self, X: torch.Tensor, E: torch.Tensor, y: Optional[torch.Tensor] = None):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor) -> "PlaceHolder":
        """Cast all tensors to same type and device as x."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.y is not None:
            self.y = self.y.type_as(x)
        return self

    def to(self, device: torch.device) -> "PlaceHolder":
        """Move all tensors to specified device."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self

    def mask(self, node_mask: torch.Tensor, collapse: bool = False) -> "PlaceHolder":
        """
        Apply node mask to tensors, zeroing out invalid positions.

        Args:
            node_mask: Boolean tensor of shape (batch_size, max_nodes) indicating valid nodes
            collapse: If True, convert from one-hot to class indices

        Returns:
            Self with masked tensors
        """
        x_mask = node_mask.unsqueeze(-1)  # (bs, n, 1)
        e_mask1 = x_mask.unsqueeze(2)  # (bs, n, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)  # (bs, 1, n, 1)

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
        return self

    def copy(self) -> "PlaceHolder":
        """Create a copy of this PlaceHolder."""
        return PlaceHolder(
            X=self.X.clone(),
            E=self.E.clone(),
            y=self.y.clone() if self.y is not None else None
        )

    def split(self, node_mask: torch.Tensor) -> List["PlaceHolder"]:
        """
        Split a batched PlaceHolder into a list of individual graph PlaceHolders.

        Args:
            node_mask: Boolean tensor of shape (batch_size, max_nodes)

        Returns:
            List of PlaceHolder objects, one per graph
        """
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i]).item()
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list

    def __repr__(self) -> str:
        x_shape = self.X.shape if isinstance(self.X, torch.Tensor) else self.X
        e_shape = self.E.shape if isinstance(self.E, torch.Tensor) else self.E
        y_shape = self.y.shape if isinstance(self.y, torch.Tensor) else self.y
        return f"PlaceHolder(X: {x_shape}, E: {e_shape}, y: {y_shape})"


def to_dense(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> Tuple[PlaceHolder, torch.Tensor]:
    """
    Convert PyG sparse tensors to dense tensor representation.

    Args:
        x: Node features tensor (total_nodes, num_features)
        edge_index: Edge indices tensor (2, num_edges)
        edge_attr: Edge attributes tensor (num_edges, num_edge_features)
        batch: Batch assignment tensor (total_nodes,)
        y: Optional global features tensor

    Returns:
        Tuple of (PlaceHolder, node_mask):
        - PlaceHolder with X (bs, n_max, dx), E (bs, n_max, n_max, de), y
        - node_mask: Boolean tensor (bs, n_max) indicating valid nodes
    """
    X, node_mask = to_dense_batch(x=x, batch=batch)

    # Remove self-loops
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )

    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=y), node_mask


def dense_to_pyg(
    X: torch.Tensor,
    E: torch.Tensor,
    y: Optional[torch.Tensor],
    node_mask: torch.Tensor,
    n_nodes: Optional[torch.Tensor] = None,
) -> List[Data]:
    """
    Convert dense tensors back to list of PyG Data objects.

    Args:
        X: Node features (bs, n_max, num_node_classes) - one-hot or class indices
        E: Edge features (bs, n_max, n_max, num_edge_classes) - one-hot or class indices
        y: Global features (bs, dy) or None
        node_mask: Boolean mask (bs, n_max) indicating valid nodes
        n_nodes: Optional tensor of node counts per graph (bs,)

    Returns:
        List of PyG Data objects, one per graph in the batch
    """
    bs = X.size(0)
    device = X.device
    graphs = []

    # Determine if inputs are one-hot or class indices
    x_is_onehot = len(X.shape) == 3
    e_is_onehot = len(E.shape) == 4

    # Get class indices
    if x_is_onehot:
        X_idx = torch.argmax(X, dim=-1)  # (bs, n_max)
        num_node_classes = X.size(-1)
    else:
        X_idx = X
        num_node_classes = int(X.max().item()) + 1

    if e_is_onehot:
        E_idx = torch.argmax(E, dim=-1)  # (bs, n_max, n_max)
        num_edge_classes = E.size(-1)
    else:
        E_idx = E
        num_edge_classes = int(E.max().item()) + 1

    for i in range(bs):
        # Get number of nodes for this graph
        if n_nodes is not None:
            n = int(n_nodes[i].item())
        else:
            n = int(node_mask[i].sum().item())

        # Extract node features for this graph
        x_i = X_idx[i, :n]

        # Convert to one-hot for output
        x_onehot = F.one_hot(x_i.long(), num_classes=num_node_classes).float()

        # Extract adjacency matrix
        adj_i = E_idx[i, :n, :n]

        # Find edges (where edge class > 0, i.e., not "no-edge")
        edge_mask = adj_i > 0
        edge_indices = edge_mask.nonzero(as_tuple=False)

        if edge_indices.numel() > 0:
            edge_index = edge_indices.t().contiguous()

            # Get edge attributes
            if e_is_onehot:
                edge_attr = E[i, edge_index[0], edge_index[1], :]
            else:
                edge_classes = adj_i[edge_index[0], edge_index[1]]
                edge_attr = F.one_hot(edge_classes.long(), num_classes=num_edge_classes).float()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, num_edge_classes), dtype=torch.float, device=device)

        # Create Data object
        data = Data(
            x=x_onehot,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Add global features if present
        if y is not None and y.numel() > 0:
            data.y = y[i:i+1]

        graphs.append(data)

    return graphs


def encode_no_edge(E: torch.Tensor) -> torch.Tensor:
    """
    Encode missing edges by setting the first class (no-edge) to 1.
    Also zeros out the diagonal.

    Args:
        E: Edge tensor of shape (bs, n, n, num_classes)

    Returns:
        Modified edge tensor with no-edge encoded
    """
    if E.shape[-1] == 0:
        return E

    # Find positions with no edge (all zeros)
    no_edge = torch.sum(E, dim=3) == 0

    # Set first class to 1 for no-edge positions
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt

    # Zero out diagonal (self-loops)
    diag = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return E


def symmetrize_edges(E: torch.Tensor) -> torch.Tensor:
    """
    Symmetrize edge tensor and zero out diagonal.

    Args:
        E: Edge tensor of shape (bs, n, n) or (bs, n, n, d)

    Returns:
        Symmetrized edge tensor
    """
    # Create upper triangular mask
    upper_triangular_mask = torch.zeros_like(E)
    indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1)

    if len(E.shape) == 4:
        upper_triangular_mask[:, indices[0], indices[1], :] = 1
    else:
        upper_triangular_mask[:, indices[0], indices[1]] = 1

    # Keep only upper triangular and mirror to lower
    E = E * upper_triangular_mask
    E = E + torch.transpose(E, 1, 2)

    # Zero out diagonal
    diag = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return E


class DistributionNodes:
    """
    Distribution over graph sizes for sampling.

    Samples the number of nodes for each generated graph based on
    the empirical distribution from training data.
    """

    def __init__(self, histogram: Union[torch.Tensor, dict]):
        """
        Initialize the node distribution.

        Args:
            histogram: Either a tensor where histogram[i] = count/probability of graphs
                      with i nodes, or a dict mapping node count to count/probability.
        """
        if isinstance(histogram, dict):
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram.float()

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(self.prob)

    def sample_n(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample n graph sizes.

        Args:
            n_samples: Number of sizes to sample
            device: Device to place the tensor on

        Returns:
            Tensor of shape (n_samples,) with sampled node counts
        """
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given node counts.

        Args:
            batch_n_nodes: Tensor of node counts

        Returns:
            Log probabilities
        """
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)
        probas = p[batch_n_nodes]
        return torch.log(probas + 1e-30)

    @property
    def max_nodes(self) -> int:
        """Maximum number of nodes in the distribution."""
        return len(self.prob) - 1


def compute_dataset_statistics(dataloader) -> dict:
    """
    Compute statistics from a PyG DataLoader for model initialization.

    Args:
        dataloader: PyG DataLoader

    Returns:
        Dict with keys:
        - node_marginals: Tensor of node type probabilities
        - edge_marginals: Tensor of edge type probabilities
        - node_counts: Tensor of graph size frequencies
        - num_node_classes: Number of node classes
        - num_edge_classes: Number of edge classes
        - max_nodes: Maximum number of nodes in any graph
    """
    num_node_classes = None
    num_edge_classes = None
    max_nodes = 0

    node_counts_dict = {}
    node_type_counts = None
    edge_type_counts = None

    for batch in dataloader:
        # Determine number of classes from first batch
        if num_node_classes is None:
            num_node_classes = batch.x.shape[1] if len(batch.x.shape) > 1 else int(batch.x.max().item()) + 1
        if num_edge_classes is None:
            num_edge_classes = batch.edge_attr.shape[1] if len(batch.edge_attr.shape) > 1 else int(batch.edge_attr.max().item()) + 1

        # Initialize count tensors
        if node_type_counts is None:
            node_type_counts = torch.zeros(num_node_classes)
            edge_type_counts = torch.zeros(num_edge_classes)

        # Count node types
        if len(batch.x.shape) > 1:
            node_type_counts += batch.x.sum(dim=0).cpu()
        else:
            for c in range(num_node_classes):
                node_type_counts[c] += (batch.x == c).sum().item()

        # Count edge types
        if len(batch.edge_attr.shape) > 1:
            edge_type_counts[1:] += batch.edge_attr[:, 1:].sum(dim=0).cpu()
        else:
            for c in range(1, num_edge_classes):
                edge_type_counts[c] += (batch.edge_attr == c).sum().item()

        # Count graph sizes
        unique, counts = torch.unique(batch.batch, return_counts=True)
        for count in counts:
            n = count.item()
            node_counts_dict[n] = node_counts_dict.get(n, 0) + 1
            max_nodes = max(max_nodes, n)

        # Count no-edges (need to compute from total possible - actual edges)
        for count in counts:
            n = count.item()
            all_pairs = n * (n - 1)  # Exclude self-loops
            # Each edge appears twice in edge_index (both directions)
            # so divide by 2 to get actual number of edges
            edge_type_counts[0] += all_pairs

    # Subtract actual edges from no-edge count
    actual_edges = edge_type_counts[1:].sum()
    edge_type_counts[0] -= actual_edges

    # Normalize to get marginals
    node_marginals = node_type_counts / node_type_counts.sum()
    edge_marginals = edge_type_counts / edge_type_counts.sum()

    # Convert node counts to tensor
    node_counts = torch.zeros(max_nodes + 1)
    for n, count in node_counts_dict.items():
        node_counts[n] = count

    return {
        "node_marginals": node_marginals,
        "edge_marginals": edge_marginals,
        "node_counts": node_counts,
        "num_node_classes": num_node_classes,
        "num_edge_classes": num_edge_classes,
        "max_nodes": max_nodes,
    }
