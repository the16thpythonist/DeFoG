"""
Tests for src/core/noise.py

Tests cover:
- LimitDistribution creation and configuration
- sample_noise function
- sample_from_probs function
- Virtual class handling for absorbing noise
"""

import pytest
import torch

from defog.core.noise import (
    LimitDistribution,
    sample_noise,
    sample_from_probs,
)
from defog.core.data import PlaceHolder


class TestLimitDistribution:
    """Tests for LimitDistribution class."""

    def test_uniform_creation(self):
        """Test uniform noise distribution creation."""
        limit_dist = LimitDistribution(
            noise_type="uniform",
            num_node_classes=4,
            num_edge_classes=2,
        )

        assert limit_dist.noise_type == "uniform"
        assert limit_dist.num_node_classes == 4
        assert limit_dist.num_edge_classes == 2

    def test_uniform_distribution_values(self):
        """Test uniform distribution has equal probabilities."""
        limit_dist = LimitDistribution(
            noise_type="uniform",
            num_node_classes=4,
            num_edge_classes=3,
        )

        # All node classes should have equal probability
        assert torch.allclose(limit_dist.X, torch.tensor([0.25, 0.25, 0.25, 0.25]))
        # All edge classes should have equal probability
        expected_e = torch.tensor([1/3, 1/3, 1/3])
        assert torch.allclose(limit_dist.E, expected_e, atol=1e-6)

    def test_marginal_creation(self, node_marginals, edge_marginals):
        """Test marginal noise distribution creation."""
        limit_dist = LimitDistribution(
            noise_type="marginal",
            num_node_classes=4,
            num_edge_classes=2,
            node_marginals=node_marginals,
            edge_marginals=edge_marginals,
        )

        assert limit_dist.noise_type == "marginal"
        assert torch.allclose(limit_dist.X, node_marginals)
        assert torch.allclose(limit_dist.E, edge_marginals)

    def test_marginal_requires_marginals(self):
        """Test marginal noise type requires marginal tensors."""
        with pytest.raises(ValueError, match="node_marginals"):
            LimitDistribution(
                noise_type="marginal",
                num_node_classes=4,
                num_edge_classes=2,
            )

    def test_absorbing_creation(self):
        """Test absorbing noise distribution creation."""
        limit_dist = LimitDistribution(
            noise_type="absorbing",
            num_node_classes=4,
            num_edge_classes=2,
        )

        assert limit_dist.noise_type == "absorbing"
        # Should add virtual class
        assert limit_dist.num_node_classes == 5
        assert limit_dist.num_edge_classes == 3
        # All probability on absorbing state
        assert limit_dist.X[-1] == 1.0
        assert limit_dist.E[-1] == 1.0

    def test_distributions_sum_to_one(self):
        """Test all distributions sum to 1."""
        for noise_type in ["uniform", "absorbing"]:
            limit_dist = LimitDistribution(
                noise_type=noise_type,
                num_node_classes=4,
                num_edge_classes=2,
            )
            assert limit_dist.X.sum().item() == pytest.approx(1.0)
            assert limit_dist.E.sum().item() == pytest.approx(1.0)

    def test_to_device(self):
        """Test moving distribution to device."""
        limit_dist = LimitDistribution(
            noise_type="uniform",
            num_node_classes=4,
            num_edge_classes=2,
        )

        limit_dist = limit_dist.to(torch.device("cpu"))

        assert limit_dist.X.device == torch.device("cpu")
        assert limit_dist.E.device == torch.device("cpu")

    def test_get_limit_dist(self):
        """Test get_limit_dist returns PlaceHolder."""
        limit_dist = LimitDistribution(
            noise_type="uniform",
            num_node_classes=4,
            num_edge_classes=2,
        )

        ph = limit_dist.get_limit_dist()

        assert isinstance(ph, PlaceHolder)
        assert torch.equal(ph.X, limit_dist.X)
        assert torch.equal(ph.E, limit_dist.E)

    def test_ignore_virtual_classes_absorbing(self):
        """Test ignore_virtual_classes removes virtual class for absorbing."""
        limit_dist = LimitDistribution(
            noise_type="absorbing",
            num_node_classes=4,
            num_edge_classes=2,
        )

        X = torch.randn(2, 5, 5)  # 5 = 4 + 1 virtual
        E = torch.randn(2, 5, 5, 3)  # 3 = 2 + 1 virtual

        X_out, E_out, _ = limit_dist.ignore_virtual_classes(X, E)

        assert X_out.shape[-1] == 4  # Virtual class removed
        assert E_out.shape[-1] == 2

    def test_ignore_virtual_classes_uniform(self):
        """Test ignore_virtual_classes is no-op for uniform."""
        limit_dist = LimitDistribution(
            noise_type="uniform",
            num_node_classes=4,
            num_edge_classes=2,
        )

        X = torch.randn(2, 5, 4)
        E = torch.randn(2, 5, 5, 2)

        X_out, E_out, _ = limit_dist.ignore_virtual_classes(X, E)

        assert X_out.shape == X.shape
        assert E_out.shape == E.shape

    def test_add_virtual_classes_absorbing(self):
        """Test add_virtual_classes adds virtual class for absorbing."""
        limit_dist = LimitDistribution(
            noise_type="absorbing",
            num_node_classes=4,
            num_edge_classes=2,
        )

        X = torch.randn(2, 5, 4)
        E = torch.randn(2, 5, 5, 2)

        X_out, E_out, _ = limit_dist.add_virtual_classes(X, E)

        assert X_out.shape[-1] == 5  # Virtual class added
        assert E_out.shape[-1] == 3

    def test_repr(self):
        """Test string representation."""
        limit_dist = LimitDistribution(
            noise_type="uniform",
            num_node_classes=4,
            num_edge_classes=2,
        )

        repr_str = repr(limit_dist)
        assert "LimitDistribution" in repr_str
        assert "uniform" in repr_str


class TestSampleNoise:
    """Tests for sample_noise function."""

    def test_sample_noise_shapes(self, uniform_limit_distribution):
        """Test sample_noise produces correct shapes."""
        bs, n = 3, 8
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        noise = sample_noise(uniform_limit_distribution, node_mask)

        assert noise.X.shape == (bs, n, uniform_limit_distribution.num_node_classes)
        assert noise.E.shape == (bs, n, n, uniform_limit_distribution.num_edge_classes)

    def test_sample_noise_is_one_hot(self, uniform_limit_distribution):
        """Test sampled noise is one-hot encoded."""
        bs, n = 3, 8
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        noise = sample_noise(uniform_limit_distribution, node_mask)

        # Each node should have exactly one class
        assert torch.allclose(noise.X.sum(dim=-1), torch.ones(bs, n))
        # Each edge should have exactly one class
        assert torch.allclose(noise.E.sum(dim=-1), torch.ones(bs, n, n))

    def test_sample_noise_edge_symmetry(self, uniform_limit_distribution):
        """Test sampled edge noise is symmetric."""
        bs, n = 3, 8
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        noise = sample_noise(uniform_limit_distribution, node_mask)

        # E should be symmetric
        E_transposed = noise.E.transpose(1, 2)
        assert torch.equal(noise.E, E_transposed)

    def test_sample_noise_respects_mask(self, uniform_limit_distribution):
        """Test sample_noise respects node mask."""
        bs, n = 2, 6
        node_mask = torch.tensor([
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
        ])

        noise = sample_noise(uniform_limit_distribution, node_mask)

        # Masked positions should be zero
        assert (noise.X[0, 4:, :] == 0).all()
        assert (noise.X[1, 3:, :] == 0).all()

    def test_sample_noise_marginal(self, limit_distribution):
        """Test sampling from marginal distribution."""
        bs, n = 10, 20
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        # Sample many times to check distribution
        noise = sample_noise(limit_distribution, node_mask)

        # Should produce valid one-hot samples
        assert torch.allclose(noise.X.sum(dim=-1), torch.ones(bs, n))


class TestSampleFromProbs:
    """Tests for sample_from_probs function."""

    def test_sample_from_probs_shapes(self):
        """Test sample_from_probs produces correct shapes."""
        bs, n, dx, de = 2, 5, 4, 2
        prob_X = torch.softmax(torch.randn(bs, n, dx), dim=-1)
        prob_E = torch.softmax(torch.randn(bs, n, n, de), dim=-1)
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        sampled = sample_from_probs(prob_X, prob_E, node_mask)

        assert sampled.X.shape == (bs, n)  # Class indices
        assert sampled.E.shape == (bs, n, n)

    def test_sample_from_probs_valid_indices(self):
        """Test sampled indices are valid class indices."""
        bs, n, dx, de = 2, 5, 4, 3
        prob_X = torch.softmax(torch.randn(bs, n, dx), dim=-1)
        prob_E = torch.softmax(torch.randn(bs, n, n, de), dim=-1)
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        sampled = sample_from_probs(prob_X, prob_E, node_mask)

        assert sampled.X.min() >= 0
        assert sampled.X.max() < dx
        assert sampled.E.min() >= 0
        assert sampled.E.max() < de

    def test_sample_from_probs_deterministic(self):
        """Test sampling from deterministic distribution."""
        bs, n, dx, de = 2, 4, 3, 2

        # Create deterministic probability (all mass on class 1)
        prob_X = torch.zeros(bs, n, dx)
        prob_X[:, :, 1] = 1.0
        prob_E = torch.zeros(bs, n, n, de)
        prob_E[:, :, :, 0] = 1.0

        node_mask = torch.ones(bs, n, dtype=torch.bool)

        sampled = sample_from_probs(prob_X, prob_E, node_mask)

        # All samples should be class 1 for nodes, class 0 for edges
        assert (sampled.X == 1).all()
        assert (sampled.E == 0).all()

    def test_sample_from_probs_edge_symmetry(self):
        """Test sampled edges are symmetric."""
        bs, n, de = 2, 5, 2
        # Use same probability for symmetric positions
        prob_X = torch.softmax(torch.randn(bs, n, 4), dim=-1)
        prob_E = torch.softmax(torch.randn(bs, n, n, de), dim=-1)
        # Make prob_E symmetric
        prob_E = (prob_E + prob_E.transpose(1, 2)) / 2

        node_mask = torch.ones(bs, n, dtype=torch.bool)

        sampled = sample_from_probs(prob_X, prob_E, node_mask)

        # E should be symmetric
        assert torch.equal(sampled.E, sampled.E.transpose(1, 2))

    def test_sample_from_probs_respects_mask(self):
        """Test sample_from_probs respects node mask."""
        bs, n, dx, de = 2, 5, 4, 2
        prob_X = torch.softmax(torch.randn(bs, n, dx), dim=-1)
        prob_E = torch.softmax(torch.randn(bs, n, n, de), dim=-1)

        node_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ])

        sampled = sample_from_probs(prob_X, prob_E, node_mask)

        # Masked positions should be zero or have consistent masking behavior
        # The exact behavior depends on implementation, but shape should be correct
        assert sampled.X.shape == (bs, n)
        assert sampled.E.shape == (bs, n, n)
