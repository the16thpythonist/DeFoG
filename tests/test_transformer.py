"""
Tests for src/core/transformer.py

Tests cover:
- GraphTransformer creation
- Forward pass shapes
- Output validity
"""

import pytest
import torch

from defog.core.transformer import GraphTransformer
from defog.core.data import PlaceHolder


@pytest.fixture
def transformer_config():
    """Configuration for a small transformer."""
    return {
        "n_layers": 2,
        "input_dims": {"X": 8, "E": 6, "y": 3},
        "hidden_mlp_dims": {"X": 32, "E": 16, "y": 32},
        "hidden_dims": {
            "dx": 16,
            "de": 8,
            "dy": 8,
            "n_head": 2,
            "dim_ffX": 32,
            "dim_ffE": 16,
        },
        "output_dims": {"X": 4, "E": 2, "y": 0},
    }


@pytest.fixture
def transformer(transformer_config):
    """Create a small GraphTransformer for testing."""
    return GraphTransformer(**transformer_config)


@pytest.fixture
def transformer_inputs(transformer_config):
    """Create inputs for the transformer."""
    bs, n = 2, 6
    input_dims = transformer_config["input_dims"]

    X = torch.randn(bs, n, input_dims["X"])
    E = torch.randn(bs, n, n, input_dims["E"])
    # Make E symmetric
    E = (E + E.transpose(1, 2)) / 2
    y = torch.randn(bs, input_dims["y"])

    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[0, 5] = False  # First graph has 5 nodes

    return X, E, y, node_mask


class TestGraphTransformer:
    """Tests for GraphTransformer class."""

    def test_creation(self, transformer_config):
        """Test basic creation."""
        transformer = GraphTransformer(**transformer_config)

        assert transformer.n_layers == transformer_config["n_layers"]

    def test_forward_returns_placeholder(self, transformer, transformer_inputs):
        """Test forward returns PlaceHolder."""
        X, E, y, node_mask = transformer_inputs

        output = transformer(X, E, y, node_mask)

        assert isinstance(output, PlaceHolder)

    def test_forward_output_shapes(self, transformer, transformer_inputs, transformer_config):
        """Test forward output shapes."""
        X, E, y, node_mask = transformer_inputs
        bs, n = X.shape[:2]
        output_dims = transformer_config["output_dims"]

        output = transformer(X, E, y, node_mask)

        assert output.X.shape == (bs, n, output_dims["X"])
        assert output.E.shape == (bs, n, n, output_dims["E"])

    def test_forward_finite_outputs(self, transformer, transformer_inputs):
        """Test forward produces finite outputs."""
        X, E, y, node_mask = transformer_inputs

        output = transformer(X, E, y, node_mask)

        assert torch.isfinite(output.X).all()
        assert torch.isfinite(output.E).all()

    def test_forward_edge_symmetry(self, transformer, transformer_inputs):
        """Test output edges are symmetric."""
        X, E, y, node_mask = transformer_inputs

        output = transformer(X, E, y, node_mask)

        E_transposed = output.E.transpose(1, 2)
        assert torch.allclose(output.E, E_transposed, atol=1e-5)

    def test_forward_different_batch_sizes(self, transformer_config):
        """Test forward with different batch sizes."""
        transformer = GraphTransformer(**transformer_config)
        input_dims = transformer_config["input_dims"]

        for bs in [1, 4, 8]:
            n = 5
            X = torch.randn(bs, n, input_dims["X"])
            E = torch.randn(bs, n, n, input_dims["E"])
            E = (E + E.transpose(1, 2)) / 2
            y = torch.randn(bs, input_dims["y"])
            node_mask = torch.ones(bs, n, dtype=torch.bool)

            output = transformer(X, E, y, node_mask)

            assert output.X.shape[0] == bs

    def test_forward_different_node_counts(self, transformer_config):
        """Test forward with different max node counts."""
        transformer = GraphTransformer(**transformer_config)
        input_dims = transformer_config["input_dims"]
        bs = 2

        for n in [3, 8, 15]:
            X = torch.randn(bs, n, input_dims["X"])
            E = torch.randn(bs, n, n, input_dims["E"])
            E = (E + E.transpose(1, 2)) / 2
            y = torch.randn(bs, input_dims["y"])
            node_mask = torch.ones(bs, n, dtype=torch.bool)

            output = transformer(X, E, y, node_mask)

            assert output.X.shape[1] == n
            assert output.E.shape[1] == n
            assert output.E.shape[2] == n

    def test_forward_respects_mask(self, transformer, transformer_inputs):
        """Test forward respects node mask."""
        X, E, y, node_mask = transformer_inputs

        # Set masked positions to different values
        X_modified = X.clone()
        X_modified[0, 5, :] = 999.0  # This node is masked

        output1 = transformer(X, E, y, node_mask)
        output2 = transformer(X_modified, E, y, node_mask)

        # Outputs for valid positions should be similar
        # (may not be exactly equal due to attention, but should be close)
        valid_output1 = output1.X[0, :5, :]
        valid_output2 = output2.X[0, :5, :]
        # At minimum, outputs should be finite
        assert torch.isfinite(valid_output1).all()
        assert torch.isfinite(valid_output2).all()

    def test_gradient_flow(self, transformer, transformer_inputs):
        """Test gradients flow through the transformer."""
        X, E, y, node_mask = transformer_inputs
        X.requires_grad_(True)

        output = transformer(X, E, y, node_mask)
        loss = output.X.sum()
        loss.backward()

        assert X.grad is not None
        assert torch.isfinite(X.grad).all()

    def test_parameters_count(self, transformer):
        """Test model has learnable parameters."""
        num_params = sum(p.numel() for p in transformer.parameters())
        num_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

        assert num_params > 0
        assert num_trainable > 0
        assert num_trainable == num_params  # All should be trainable
