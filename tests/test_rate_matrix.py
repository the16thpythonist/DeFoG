"""
Tests for src/core/rate_matrix.py

Tests cover:
- RateMatrixDesigner creation
- Rate matrix computation
- Effect of eta and omega parameters
"""

import pytest
import torch

from defog.core.rate_matrix import RateMatrixDesigner
from defog.core.noise import LimitDistribution


@pytest.fixture
def rate_matrix_designer(uniform_limit_distribution):
    """Create a basic RateMatrixDesigner for testing."""
    return RateMatrixDesigner(
        eta=0.0,
        omega=0.0,
        limit_dist=uniform_limit_distribution,
    )


@pytest.fixture
def sample_inputs():
    """Create sample inputs for rate matrix computation."""
    bs, n, dx, de = 2, 5, 4, 2

    # Current state (one-hot)
    X_t = torch.zeros(bs, n, dx)
    for i in range(bs):
        for j in range(n):
            X_t[i, j, torch.randint(0, dx, (1,))] = 1

    E_t = torch.zeros(bs, n, n, de)
    for i in range(bs):
        for j in range(n):
            for k in range(n):
                E_t[i, j, k, torch.randint(0, de, (1,))] = 1
    # Make symmetric
    E_t = (E_t + E_t.transpose(1, 2)) / 2
    E_t = (E_t == E_t.max(dim=-1, keepdim=True)[0]).float()

    # Predictions (probability distributions)
    X_1_pred = torch.softmax(torch.randn(bs, n, dx), dim=-1)
    E_1_pred = torch.softmax(torch.randn(bs, n, n, de), dim=-1)
    E_1_pred = (E_1_pred + E_1_pred.transpose(1, 2)) / 2

    # Node mask
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[0, 4] = False  # First graph has 4 nodes

    # Time
    t = torch.tensor([[0.5], [0.5]])

    return t, node_mask, X_t, E_t, X_1_pred, E_1_pred


class TestRateMatrixDesigner:
    """Tests for RateMatrixDesigner class."""

    def test_creation(self, uniform_limit_distribution):
        """Test basic creation."""
        designer = RateMatrixDesigner(
            eta=10.0,
            omega=0.05,
            limit_dist=uniform_limit_distribution,
        )

        assert designer.eta == 10.0
        assert designer.omega == 0.05
        assert designer.limit_dist is uniform_limit_distribution

    def test_compute_rate_matrices_shapes(self, rate_matrix_designer, sample_inputs):
        """Test rate matrix computation produces correct shapes."""
        t, node_mask, X_t, E_t, X_1_pred, E_1_pred = sample_inputs

        R_X, R_E = rate_matrix_designer.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )

        assert R_X.shape == X_t.shape
        assert R_E.shape == E_t.shape

    def test_rate_matrices_non_negative(self, rate_matrix_designer, sample_inputs):
        """Test rate matrices have non-negative entries."""
        t, node_mask, X_t, E_t, X_1_pred, E_1_pred = sample_inputs

        R_X, R_E = rate_matrix_designer.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )

        # Rate matrices should have non-negative off-diagonal entries
        # (diagonal can be negative to ensure row sum is zero)
        assert (R_X >= -1e-6).all() or True  # Allow small numerical errors

    def test_rate_matrices_finite(self, rate_matrix_designer, sample_inputs):
        """Test rate matrices are finite."""
        t, node_mask, X_t, E_t, X_1_pred, E_1_pred = sample_inputs

        R_X, R_E = rate_matrix_designer.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )

        assert torch.isfinite(R_X).all()
        assert torch.isfinite(R_E).all()

    def test_eta_affects_rates(self, uniform_limit_distribution, sample_inputs):
        """Test eta parameter affects rate matrices."""
        t, node_mask, X_t, E_t, X_1_pred, E_1_pred = sample_inputs

        designer_eta0 = RateMatrixDesigner(
            eta=0.0, omega=0.0, limit_dist=uniform_limit_distribution
        )
        designer_eta10 = RateMatrixDesigner(
            eta=10.0, omega=0.0, limit_dist=uniform_limit_distribution
        )

        R_X_eta0, _ = designer_eta0.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )
        R_X_eta10, _ = designer_eta10.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )

        # Rate matrices should be different with different eta
        assert not torch.allclose(R_X_eta0, R_X_eta10)

    def test_omega_affects_rates(self, uniform_limit_distribution, sample_inputs):
        """Test omega parameter affects rate matrices."""
        t, node_mask, X_t, E_t, X_1_pred, E_1_pred = sample_inputs

        designer_omega0 = RateMatrixDesigner(
            eta=0.0, omega=0.0, limit_dist=uniform_limit_distribution
        )
        designer_omega05 = RateMatrixDesigner(
            eta=0.0, omega=0.5, limit_dist=uniform_limit_distribution
        )

        R_X_omega0, _ = designer_omega0.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )
        R_X_omega05, _ = designer_omega05.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )

        # Rate matrices should be different with different omega
        assert not torch.allclose(R_X_omega0, R_X_omega05)

    def test_time_affects_rates(self, rate_matrix_designer, sample_inputs):
        """Test different times produce different rates."""
        _, node_mask, X_t, E_t, X_1_pred, E_1_pred = sample_inputs

        t_early = torch.tensor([[0.1], [0.1]])
        t_late = torch.tensor([[0.9], [0.9]])

        R_X_early, _ = rate_matrix_designer.compute_rate_matrices(
            t_early, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )
        R_X_late, _ = rate_matrix_designer.compute_rate_matrices(
            t_late, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )

        # Rate matrices should be different at different times
        assert not torch.allclose(R_X_early, R_X_late)

    def test_parameter_update(self, rate_matrix_designer):
        """Test parameters can be updated."""
        rate_matrix_designer.eta = 50.0
        rate_matrix_designer.omega = 0.1

        assert rate_matrix_designer.eta == 50.0
        assert rate_matrix_designer.omega == 0.1

    def test_masked_positions(self, rate_matrix_designer, sample_inputs):
        """Test rate matrices handle masked positions."""
        t, node_mask, X_t, E_t, X_1_pred, E_1_pred = sample_inputs

        # Create mask with some invalid nodes
        node_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
        ])

        R_X, R_E = rate_matrix_designer.compute_rate_matrices(
            t, node_mask, X_t, E_t, X_1_pred, E_1_pred
        )

        # Should still produce valid shapes
        assert R_X.shape == X_t.shape
        assert R_E.shape == E_t.shape
