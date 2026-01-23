"""
Tests for src/core/data.py

Tests cover:
- PlaceHolder class operations
- to_dense conversion from PyG to dense tensors
- dense_to_pyg conversion back to PyG Data objects
- compute_dataset_statistics
- DistributionNodes sampling
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from defog.core.data import (
    PlaceHolder,
    to_dense,
    dense_to_pyg,
    encode_no_edge,
    symmetrize_edges,
    DistributionNodes,
    compute_dataset_statistics,
)


class TestPlaceHolder:
    """Tests for PlaceHolder class."""

    def test_creation(self):
        """Test PlaceHolder creation with basic tensors."""
        X = torch.randn(2, 5, 4)
        E = torch.randn(2, 5, 5, 2)
        y = torch.randn(2, 3)

        ph = PlaceHolder(X=X, E=E, y=y)

        assert ph.X.shape == (2, 5, 4)
        assert ph.E.shape == (2, 5, 5, 2)
        assert ph.y.shape == (2, 3)

    def test_creation_no_y(self):
        """Test PlaceHolder creation without global features."""
        X = torch.randn(2, 5, 4)
        E = torch.randn(2, 5, 5, 2)

        ph = PlaceHolder(X=X, E=E)

        assert ph.X is not None
        assert ph.E is not None
        assert ph.y is None

    def test_type_as(self):
        """Test type_as casts tensors correctly."""
        X = torch.randn(2, 5, 4)
        E = torch.randn(2, 5, 5, 2)
        y = torch.randn(2, 3)
        ph = PlaceHolder(X=X, E=E, y=y)

        target = torch.zeros(1, dtype=torch.float64)
        ph = ph.type_as(target)

        assert ph.X.dtype == torch.float64
        assert ph.E.dtype == torch.float64
        assert ph.y.dtype == torch.float64

    def test_mask(self):
        """Test mask zeros out invalid positions."""
        bs, n, dx, de = 2, 4, 3, 2
        X = torch.ones(bs, n, dx)
        E = torch.ones(bs, n, n, de)
        y = torch.ones(bs, 2)

        node_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ])

        ph = PlaceHolder(X=X, E=E, y=y)
        ph = ph.mask(node_mask)

        # Check masked positions are zero
        assert (ph.X[0, 3, :] == 0).all()
        assert (ph.X[1, 2:, :] == 0).all()
        assert (ph.E[0, 3, :, :] == 0).all()
        assert (ph.E[0, :, 3, :] == 0).all()

    def test_mask_collapse(self):
        """Test mask with collapse converts to indices."""
        bs, n, dx = 2, 3, 4
        X = torch.zeros(bs, n, dx)
        X[0, 0, 0] = 1
        X[0, 1, 1] = 1
        X[0, 2, 2] = 1
        X[1, 0, 3] = 1
        X[1, 1, 0] = 1
        X[1, 2, 1] = 1

        E = torch.zeros(bs, n, n, 2)
        E[:, :, :, 0] = 1  # All no-edges

        node_mask = torch.ones(bs, n, dtype=torch.bool)

        ph = PlaceHolder(X=X, E=E, y=torch.zeros(bs, 0))
        ph = ph.mask(node_mask, collapse=True)

        assert ph.X[0, 0] == 0
        assert ph.X[0, 1] == 1
        assert ph.X[0, 2] == 2

    def test_copy(self):
        """Test copy creates independent tensors."""
        X = torch.randn(2, 5, 4)
        E = torch.randn(2, 5, 5, 2)
        ph = PlaceHolder(X=X, E=E)

        ph_copy = ph.copy()
        ph_copy.X[0, 0, 0] = 999

        assert ph.X[0, 0, 0] != 999


class TestToDense:
    """Tests for to_dense function."""

    def test_single_graph(self, single_graph):
        """Test conversion of a single graph."""
        batch = Batch.from_data_list([single_graph])
        ph, node_mask = to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        assert ph.X.shape[0] == 1  # batch size
        assert ph.X.shape[1] == single_graph.x.shape[0]  # num nodes
        assert ph.X.shape[2] == single_graph.x.shape[1]  # node features
        assert node_mask.shape == (1, single_graph.x.shape[0])
        assert node_mask.all()  # All nodes valid for single graph

    def test_batch_shapes(self, graph_batch):
        """Test conversion of a batch produces correct shapes."""
        ph, node_mask = to_dense(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch,
        )

        assert ph.X.dim() == 3  # (bs, n, dx)
        assert ph.E.dim() == 4  # (bs, n, n, de)
        assert node_mask.dim() == 2  # (bs, n)
        assert ph.X.shape[0] == 3  # 3 graphs in batch

    def test_node_mask_correctness(self, graph_batch):
        """Test node mask correctly identifies valid nodes."""
        ph, node_mask = to_dense(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch,
        )

        # Count nodes per graph
        unique, counts = torch.unique(graph_batch.batch, return_counts=True)
        max_nodes = ph.X.shape[1]

        for i, count in enumerate(counts):
            assert node_mask[i, :count].all()
            if count < max_nodes:
                assert not node_mask[i, count:].any()

    def test_edge_symmetry(self, graph_batch):
        """Test dense edges are symmetric."""
        ph, _ = to_dense(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch,
        )

        E_transposed = ph.E.transpose(1, 2)
        assert torch.allclose(ph.E, E_transposed)


class TestDenseToPyg:
    """Tests for dense_to_pyg function."""

    def test_basic_conversion(self, dense_tensors):
        """Test basic conversion from dense to PyG."""
        X, E, node_mask, n_nodes = dense_tensors
        y = torch.zeros(X.size(0), 0)

        graphs = dense_to_pyg(X, E, y, node_mask, n_nodes)

        assert len(graphs) == X.size(0)
        for i, g in enumerate(graphs):
            assert isinstance(g, Data)
            assert g.x.shape[0] == n_nodes[i].item()

    def test_preserves_node_features(self, dense_tensors):
        """Test node features are preserved in conversion."""
        X, E, node_mask, n_nodes = dense_tensors
        y = torch.zeros(X.size(0), 0)

        graphs = dense_to_pyg(X, E, y, node_mask, n_nodes)

        for i, g in enumerate(graphs):
            n = n_nodes[i].item()
            # Check node features match (argmax should be same)
            original_classes = X[i, :n].argmax(dim=-1)
            recovered_classes = g.x.argmax(dim=-1)
            assert torch.equal(original_classes, recovered_classes)

    def test_edge_recovery(self, dense_tensors):
        """Test edges are correctly recovered."""
        X, E, node_mask, n_nodes = dense_tensors
        y = torch.zeros(X.size(0), 0)

        graphs = dense_to_pyg(X, E, y, node_mask, n_nodes)

        for i, g in enumerate(graphs):
            n = n_nodes[i].item()
            # Check that edge_index only contains valid node indices
            if g.edge_index.numel() > 0:
                assert g.edge_index.max() < n
                assert g.edge_index.min() >= 0


class TestEncodeNoEdge:
    """Tests for encode_no_edge function."""

    def test_sets_no_edge_class(self):
        """Test that missing edges get class 0."""
        E = torch.zeros(2, 4, 4, 3)
        # Set some actual edges
        E[0, 0, 1, 1] = 1
        E[0, 1, 0, 1] = 1

        E_encoded = encode_no_edge(E.clone())

        # Positions without edges should have class 0 = 1
        assert E_encoded[0, 0, 2, 0] == 1
        assert E_encoded[0, 2, 3, 0] == 1
        # Diagonal should be zero
        assert (E_encoded[0, torch.arange(4), torch.arange(4), :] == 0).all()

    def test_preserves_existing_edges(self):
        """Test that existing edges are preserved."""
        E = torch.zeros(1, 3, 3, 2)
        E[0, 0, 1, 1] = 1
        E[0, 1, 0, 1] = 1

        E_encoded = encode_no_edge(E.clone())

        assert E_encoded[0, 0, 1, 1] == 1
        assert E_encoded[0, 1, 0, 1] == 1


class TestSymmetrizeEdges:
    """Tests for symmetrize_edges function."""

    def test_symmetrizes_4d(self):
        """Test symmetrization of 4D edge tensor."""
        E = torch.zeros(2, 4, 4, 2)
        E[0, 0, 1, 1] = 1  # Only upper triangle

        E_sym = symmetrize_edges(E.clone())

        assert E_sym[0, 0, 1, 1] == 1
        assert E_sym[0, 1, 0, 1] == 1  # Should be mirrored

    def test_diagonal_is_zero(self):
        """Test diagonal is zeroed out."""
        E = torch.ones(2, 4, 4, 2)

        E_sym = symmetrize_edges(E.clone())

        for i in range(4):
            assert (E_sym[:, i, i, :] == 0).all()


class TestDistributionNodes:
    """Tests for DistributionNodes class."""

    def test_creation_from_tensor(self, node_counts_distribution):
        """Test creation from a tensor histogram."""
        dist = DistributionNodes(node_counts_distribution)

        assert dist.prob.sum().item() == pytest.approx(1.0)
        assert dist.max_nodes == len(node_counts_distribution) - 1

    def test_creation_from_dict(self):
        """Test creation from a dictionary."""
        histogram = {3: 10, 4: 20, 5: 30, 6: 20, 7: 10}
        dist = DistributionNodes(histogram)

        assert dist.prob.sum().item() == pytest.approx(1.0)
        assert dist.max_nodes == 7

    def test_sample_n(self, node_counts_distribution):
        """Test sampling produces valid sizes."""
        dist = DistributionNodes(node_counts_distribution)

        samples = dist.sample_n(100, torch.device("cpu"))

        assert samples.shape == (100,)
        # All samples should be within valid range
        assert samples.min() >= 0
        assert samples.max() <= dist.max_nodes
        # Should only sample from non-zero probability indices
        valid_indices = torch.where(node_counts_distribution > 0)[0]
        for s in samples:
            assert s in valid_indices

    def test_log_prob(self, node_counts_distribution):
        """Test log probability computation."""
        dist = DistributionNodes(node_counts_distribution)

        # Test on indices with known probability
        batch = torch.tensor([5, 6, 7])
        log_probs = dist.log_prob(batch)

        assert log_probs.shape == (3,)
        # Log prob should be negative for valid probabilities < 1
        assert (log_probs < 0).all()


class TestComputeDatasetStatistics:
    """Tests for compute_dataset_statistics function."""

    def test_returns_dict(self, small_dataloader):
        """Test that function returns expected dict structure."""
        stats = compute_dataset_statistics(small_dataloader)

        assert isinstance(stats, dict)
        assert "node_marginals" in stats
        assert "edge_marginals" in stats
        assert "node_counts" in stats
        assert "num_node_classes" in stats
        assert "num_edge_classes" in stats
        assert "max_nodes" in stats

    def test_marginals_sum_to_one(self, small_dataloader):
        """Test marginal distributions sum to 1."""
        stats = compute_dataset_statistics(small_dataloader)

        assert stats["node_marginals"].sum().item() == pytest.approx(1.0, rel=1e-5)
        assert stats["edge_marginals"].sum().item() == pytest.approx(1.0, rel=1e-5)

    def test_detects_classes(self, small_dataloader):
        """Test correct detection of class counts."""
        stats = compute_dataset_statistics(small_dataloader)

        assert stats["num_node_classes"] == 4
        assert stats["num_edge_classes"] == 2

    def test_max_nodes(self, small_dataloader):
        """Test max_nodes is correctly computed."""
        stats = compute_dataset_statistics(small_dataloader)

        # Max nodes should be at least as large as any graph
        for batch in small_dataloader:
            _, counts = torch.unique(batch.batch, return_counts=True)
            assert stats["max_nodes"] >= counts.max().item()
