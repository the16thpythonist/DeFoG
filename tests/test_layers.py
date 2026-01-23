"""
Tests for src/core/layers.py

Tests cover:
- XEyTransformerLayer forward pass
- NodeEdgeBlock operations
- Helper functions
"""

import pytest
import torch

from src.core.layers import (
    XEyTransformerLayer,
    NodeEdgeBlock,
    Xtoy,
    Etoy,
    masked_softmax,
    timestep_embedding,
)


class TestXEyTransformerLayer:
    """Tests for XEyTransformerLayer class."""

    @pytest.fixture
    def layer_config(self):
        """Configuration for a transformer layer."""
        return {
            "dx": 16,
            "de": 8,
            "dy": 8,
            "n_head": 2,
            "dim_ffX": 32,
            "dim_ffE": 16,
        }

    @pytest.fixture
    def layer(self, layer_config):
        """Create a transformer layer."""
        return XEyTransformerLayer(**layer_config)

    @pytest.fixture
    def layer_inputs(self, layer_config):
        """Create inputs for the layer."""
        bs, n = 2, 6
        X = torch.randn(bs, n, layer_config["dx"])
        E = torch.randn(bs, n, n, layer_config["de"])
        E = (E + E.transpose(1, 2)) / 2  # Symmetric
        y = torch.randn(bs, layer_config["dy"])
        node_mask = torch.ones(bs, n, dtype=torch.bool)
        return X, E, y, node_mask

    def test_forward_shapes(self, layer, layer_inputs, layer_config):
        """Test forward produces correct shapes."""
        X, E, y, node_mask = layer_inputs
        bs, n = X.shape[:2]

        X_out, E_out, y_out = layer(X, E, y, node_mask)

        assert X_out.shape == (bs, n, layer_config["dx"])
        assert E_out.shape == (bs, n, n, layer_config["de"])
        assert y_out.shape == (bs, layer_config["dy"])

    def test_forward_finite(self, layer, layer_inputs):
        """Test forward produces finite outputs."""
        X, E, y, node_mask = layer_inputs

        X_out, E_out, y_out = layer(X, E, y, node_mask)

        assert torch.isfinite(X_out).all()
        assert torch.isfinite(E_out).all()
        assert torch.isfinite(y_out).all()

    def test_forward_edge_shape_preserved(self, layer, layer_inputs):
        """Test output edge shape is preserved."""
        X, E, y, node_mask = layer_inputs

        X_out, E_out, y_out = layer(X, E, y, node_mask)

        # Shape should be preserved
        assert E_out.shape == E.shape

    def test_gradient_flow(self, layer, layer_inputs):
        """Test gradients flow through the layer."""
        X, E, y, node_mask = layer_inputs
        X = X.clone().requires_grad_(True)

        X_out, E_out, y_out = layer(X, E, y, node_mask)
        loss = X_out.sum() + E_out.sum() + y_out.sum()
        loss.backward()

        assert X.grad is not None


class TestNodeEdgeBlock:
    """Tests for NodeEdgeBlock class."""

    @pytest.fixture
    def block(self):
        """Create a NodeEdgeBlock."""
        return NodeEdgeBlock(dx=16, de=8, dy=8, n_head=2)

    @pytest.fixture
    def block_inputs(self):
        """Create inputs for the block."""
        bs, n = 2, 5
        X = torch.randn(bs, n, 16)
        E = torch.randn(bs, n, n, 8)
        y = torch.randn(bs, 8)
        node_mask = torch.ones(bs, n, dtype=torch.bool)
        return X, E, y, node_mask

    def test_forward_shapes(self, block, block_inputs):
        """Test forward produces correct shapes."""
        X, E, y, node_mask = block_inputs

        X_out, E_out, y_out = block(X, E, y, node_mask)

        assert X_out.shape == X.shape
        assert E_out.shape == E.shape
        assert y_out.shape == y.shape

    def test_forward_finite(self, block, block_inputs):
        """Test forward produces finite outputs."""
        X, E, y, node_mask = block_inputs

        X_out, E_out, y_out = block(X, E, y, node_mask)

        assert torch.isfinite(X_out).all()
        assert torch.isfinite(E_out).all()
        assert torch.isfinite(y_out).all()


class TestXtoy:
    """Tests for Xtoy aggregation."""

    def test_forward_shape(self):
        """Test Xtoy produces correct shape."""
        xtoy = Xtoy(dx=16, dy=8)
        bs, n = 2, 5
        X = torch.randn(bs, n, 16)

        y = xtoy(X)

        assert y.shape == (bs, 8)

    def test_forward_finite(self):
        """Test Xtoy produces finite outputs."""
        xtoy = Xtoy(dx=16, dy=8)
        X = torch.randn(2, 5, 16)

        y = xtoy(X)

        assert torch.isfinite(y).all()


class TestEtoy:
    """Tests for Etoy aggregation."""

    def test_forward_shape(self):
        """Test Etoy produces correct shape."""
        etoy = Etoy(de=8, dy=8)
        bs, n = 2, 5
        E = torch.randn(bs, n, n, 8)

        y = etoy(E)

        assert y.shape == (bs, 8)

    def test_forward_finite(self):
        """Test Etoy produces finite outputs."""
        etoy = Etoy(de=8, dy=8)
        E = torch.randn(2, 5, 5, 8)

        y = etoy(E)

        assert torch.isfinite(y).all()


class TestMaskedSoftmax:
    """Tests for masked_softmax function."""

    def test_basic_softmax(self):
        """Test basic softmax without mask."""
        X = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5, dtype=torch.bool)

        result = masked_softmax(X, mask, dim=-1)

        # Should sum to 1 along last dimension
        assert torch.allclose(result.sum(dim=-1), torch.ones(2, 5))

    def test_with_mask(self):
        """Test softmax respects mask."""
        X = torch.randn(2, 5, 10)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ])

        result = masked_softmax(X, mask, dim=-1)

        # Valid positions should still sum to 1
        assert torch.allclose(result[0, :3].sum(dim=-1), torch.ones(3))
        assert torch.allclose(result[1, :2].sum(dim=-1), torch.ones(2))

    def test_output_range(self):
        """Test softmax outputs are in [0, 1]."""
        X = torch.randn(2, 5, 10)
        mask = torch.ones(2, 5, dtype=torch.bool)

        result = masked_softmax(X, mask, dim=-1)

        assert (result >= 0).all()
        assert (result <= 1).all()


class TestTimestepEmbedding:
    """Tests for timestep_embedding function."""

    def test_shape(self):
        """Test embedding has correct shape."""
        # timestep_embedding expects 2-D tensor of shape (N, 1)
        timesteps = torch.tensor([[0.0], [0.5], [1.0]])
        dim = 32

        emb = timestep_embedding(timesteps, dim)

        assert emb.shape == (3, dim)

    def test_finite(self):
        """Test embedding values are finite."""
        timesteps = torch.arange(10).float().unsqueeze(1)  # Shape: (10, 1)
        dim = 64

        emb = timestep_embedding(timesteps, dim)

        assert torch.isfinite(emb).all()

    def test_different_timesteps_different_embeddings(self):
        """Test different timesteps give different embeddings."""
        t1 = torch.tensor([[0.1]])
        t2 = torch.tensor([[0.9]])
        dim = 32

        emb1 = timestep_embedding(t1, dim)
        emb2 = timestep_embedding(t2, dim)

        assert not torch.allclose(emb1, emb2)

    def test_batch_consistency(self):
        """Test batch processing is consistent with individual."""
        timesteps = torch.tensor([[0.2], [0.5], [0.8]])
        dim = 32

        # Batch
        emb_batch = timestep_embedding(timesteps, dim)

        # Individual
        emb_0 = timestep_embedding(timesteps[0:1], dim)
        emb_1 = timestep_embedding(timesteps[1:2], dim)
        emb_2 = timestep_embedding(timesteps[2:3], dim)

        assert torch.allclose(emb_batch[0], emb_0[0])
        assert torch.allclose(emb_batch[1], emb_1[0])
        assert torch.allclose(emb_batch[2], emb_2[0])
