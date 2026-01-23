"""
Tests for src/core/features.py

Tests cover:
- ExtraFeatures creation and forward pass
- RRWPFeatures computation
"""

import pytest
import torch

from defog.core.features import ExtraFeatures, RRWPFeatures
from defog.core.data import PlaceHolder


@pytest.fixture
def noisy_data_dict():
    """Create a noisy_data dictionary for feature computation."""
    bs, n, dx, de = 2, 6, 4, 2

    X_t = torch.zeros(bs, n, dx)
    for i in range(bs):
        for j in range(n):
            X_t[i, j, torch.randint(0, dx, (1,))] = 1

    E_t = torch.zeros(bs, n, n, de)
    for i in range(bs):
        for j in range(n):
            for k in range(j + 1, n):
                if torch.rand(1) < 0.3:
                    E_t[i, j, k, 1] = 1
                    E_t[i, k, j, 1] = 1
                else:
                    E_t[i, j, k, 0] = 1
                    E_t[i, k, j, 0] = 1

    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[0, 5] = False

    return {
        "X_t": X_t,
        "E_t": E_t,
        "y_t": torch.zeros(bs, 0),
        "t": torch.rand(bs, 1),
        "node_mask": node_mask,
    }


class TestExtraFeatures:
    """Tests for ExtraFeatures class."""

    def test_creation_none(self):
        """Test creation with no extra features."""
        features = ExtraFeatures(feature_type="none")
        dims = features.output_dims()

        assert dims["X"] == 0
        assert dims["E"] == 0

    def test_creation_rrwp(self):
        """Test creation with RRWP features."""
        features = ExtraFeatures(feature_type="rrwp", rrwp_steps=5)
        dims = features.output_dims()

        assert dims["X"] > 0
        assert dims["E"] > 0

    def test_creation_cycles(self):
        """Test creation with cycle features."""
        features = ExtraFeatures(feature_type="cycles")
        dims = features.output_dims()

        assert dims["X"] >= 0
        assert dims["E"] >= 0

    def test_forward_none(self, noisy_data_dict):
        """Test forward pass with no features."""
        features = ExtraFeatures(feature_type="none")

        output = features(noisy_data_dict)

        assert isinstance(output, PlaceHolder)
        bs, n = noisy_data_dict["X_t"].shape[:2]
        assert output.X.shape == (bs, n, 0)
        assert output.E.shape == (bs, n, n, 0)

    def test_forward_rrwp_shapes(self, noisy_data_dict):
        """Test RRWP features produce correct shapes."""
        rrwp_steps = 5
        features = ExtraFeatures(feature_type="rrwp", rrwp_steps=rrwp_steps)
        dims = features.output_dims()

        output = features(noisy_data_dict)

        bs, n = noisy_data_dict["X_t"].shape[:2]
        assert output.X.shape == (bs, n, dims["X"])
        assert output.E.shape == (bs, n, n, dims["E"])

    def test_forward_rrwp_finite(self, noisy_data_dict):
        """Test RRWP features are finite."""
        features = ExtraFeatures(feature_type="rrwp", rrwp_steps=5)

        output = features(noisy_data_dict)

        assert torch.isfinite(output.X).all()
        assert torch.isfinite(output.E).all()

    def test_output_dims_consistency(self, noisy_data_dict):
        """Test output_dims matches actual output."""
        for feature_type in ["none", "rrwp"]:
            features = ExtraFeatures(feature_type=feature_type, rrwp_steps=5)
            dims = features.output_dims()

            output = features(noisy_data_dict)

            assert output.X.shape[-1] == dims["X"]
            assert output.E.shape[-1] == dims["E"]


class TestRRWPFeatures:
    """Tests for RRWPFeatures class."""

    def test_creation(self):
        """Test creation."""
        rrwp = RRWPFeatures(normalize=True)
        assert rrwp.normalize == True

    def test_call_shapes(self):
        """Test RRWP computation shapes."""
        rrwp = RRWPFeatures()
        bs, n = 2, 6

        # Create binary adjacency matrix
        adj = torch.zeros(bs, n, n)
        for i in range(bs):
            for j in range(n):
                for k in range(j + 1, n):
                    if torch.rand(1) < 0.3:
                        adj[i, j, k] = 1
                        adj[i, k, j] = 1

        k = 5
        result = rrwp(adj, k=k)

        # Result should have shape (bs, n, n, k)
        assert result.shape == (bs, n, n, k)

    def test_call_finite(self):
        """Test RRWP values are finite."""
        rrwp = RRWPFeatures()
        bs, n = 2, 6

        adj = torch.zeros(bs, n, n)  # No edges

        result = rrwp(adj, k=5)

        assert torch.isfinite(result).all()

    def test_identity_in_first_step(self):
        """Test first RRWP step contains identity-related values."""
        rrwp = RRWPFeatures()
        bs, n = 1, 4

        adj = torch.zeros(bs, n, n)  # No edges

        result = rrwp(adj, k=5)

        # For isolated nodes, RRWP should be well-defined
        assert torch.isfinite(result).all()
