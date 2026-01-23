"""
Shared pytest fixtures for DeFoG core module tests.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


# ============================================================================
# Configuration fixtures
# ============================================================================

@pytest.fixture
def device():
    """Default device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_model_config():
    """Configuration for a small model suitable for testing."""
    return {
        "num_node_classes": 4,
        "num_edge_classes": 2,
        "n_layers": 2,
        "hidden_dim": 32,
        "hidden_mlp_dim": 64,
        "n_heads": 2,
        "dropout": 0.0,
        "max_nodes": 15,
        "sample_steps": 5,
    }


# ============================================================================
# Single graph fixtures
# ============================================================================

@pytest.fixture
def single_graph():
    """Create a single synthetic PyG Data object."""
    n = 6
    # Node features (4 classes, one-hot)
    x = torch.zeros(n, 4)
    x[torch.arange(n), torch.tensor([0, 1, 2, 3, 0, 1])] = 1

    # Create adjacency (simple chain + extra edge)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
    ], dtype=torch.long)

    # Edge attributes (2 classes: no-edge=0, edge=1)
    edge_attr = torch.zeros(edge_index.size(1), 2)
    edge_attr[:, 1] = 1  # All stored edges are "edge" class

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def single_node_graph():
    """Create a graph with only one node (edge case)."""
    x = torch.zeros(1, 4)
    x[0, 0] = 1
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 2))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def empty_edges_graph():
    """Create a graph with nodes but no edges."""
    n = 4
    x = torch.zeros(n, 4)
    x[torch.arange(n), torch.randint(0, 4, (n,))] = 1
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 2))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================================
# Batch fixtures
# ============================================================================

@pytest.fixture
def graph_batch(single_graph):
    """Create a batch of graphs."""
    # Create slightly different graphs
    graphs = []
    for i in range(3):
        n = 4 + i * 2  # 4, 6, 8 nodes
        x = torch.zeros(n, 4)
        x[torch.arange(n), torch.randint(0, 4, (n,))] = 1

        # Random edges
        adj = (torch.rand(n, n) < 0.3).float()
        adj = ((adj + adj.t()) > 0).float()
        adj.fill_diagonal_(0)
        edge_index = adj.nonzero(as_tuple=False).t()

        edge_attr = torch.zeros(edge_index.size(1), 2)
        edge_attr[:, 1] = 1

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    return Batch.from_data_list(graphs)


@pytest.fixture
def small_dataset():
    """Create a small dataset of synthetic graphs."""
    dataset = []
    for _ in range(20):
        n = torch.randint(3, 10, (1,)).item()
        x = torch.zeros(n, 4)
        x[torch.arange(n), torch.randint(0, 4, (n,))] = 1

        adj = (torch.rand(n, n) < 0.3).float()
        adj = ((adj + adj.t()) > 0).float()
        adj.fill_diagonal_(0)
        edge_index = adj.nonzero(as_tuple=False).t()

        edge_attr = torch.zeros(edge_index.size(1), 2)
        edge_attr[:, 1] = 1

        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    return dataset


@pytest.fixture
def small_dataloader(small_dataset):
    """Create a DataLoader from the small dataset."""
    return DataLoader(small_dataset, batch_size=8, shuffle=False)


# ============================================================================
# Dense tensor fixtures
# ============================================================================

@pytest.fixture
def dense_tensors():
    """Create dense tensor representation of graphs."""
    bs, n, dx, de = 2, 5, 4, 2

    # Node features (one-hot)
    X = torch.zeros(bs, n, dx)
    for i in range(bs):
        for j in range(n):
            X[i, j, torch.randint(0, dx, (1,))] = 1

    # Edge features (one-hot, symmetric)
    E = torch.zeros(bs, n, n, de)
    for i in range(bs):
        for j in range(n):
            for k in range(j + 1, n):
                if torch.rand(1) < 0.3:
                    E[i, j, k, 1] = 1
                    E[i, k, j, 1] = 1
                else:
                    E[i, j, k, 0] = 1
                    E[i, k, j, 0] = 1

    # Node mask
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[0, 4] = False  # First graph has 4 nodes
    node_mask[1, 3:] = False  # Second graph has 3 nodes

    # Node counts
    n_nodes = torch.tensor([4, 3])

    return X, E, node_mask, n_nodes


# ============================================================================
# Distribution fixtures
# ============================================================================

@pytest.fixture
def node_counts_distribution():
    """Distribution of graph sizes."""
    counts = torch.zeros(15)
    counts[3:10] = torch.tensor([1, 2, 3, 4, 3, 2, 1], dtype=torch.float)
    return counts


@pytest.fixture
def node_marginals():
    """Marginal distribution over node types."""
    return torch.tensor([0.3, 0.3, 0.2, 0.2])


@pytest.fixture
def edge_marginals():
    """Marginal distribution over edge types (including no-edge)."""
    return torch.tensor([0.85, 0.15])


# ============================================================================
# Model fixtures
# ============================================================================

@pytest.fixture
def limit_distribution(node_marginals, edge_marginals):
    """Create a LimitDistribution for testing."""
    from src.core.noise import LimitDistribution
    return LimitDistribution(
        noise_type="marginal",
        num_node_classes=4,
        num_edge_classes=2,
        node_marginals=node_marginals,
        edge_marginals=edge_marginals,
    )


@pytest.fixture
def uniform_limit_distribution():
    """Create a uniform LimitDistribution for testing."""
    from src.core.noise import LimitDistribution
    return LimitDistribution(
        noise_type="uniform",
        num_node_classes=4,
        num_edge_classes=2,
    )


@pytest.fixture
def small_model(small_model_config, node_counts_distribution):
    """Create a small DeFoGModel for testing."""
    from src.core import DeFoGModel
    return DeFoGModel(
        **small_model_config,
        noise_type="uniform",
        node_counts=node_counts_distribution,
    )
