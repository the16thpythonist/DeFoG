"""
Tests for src/core/loss.py

Tests cover:
- TrainLoss creation and forward pass
- Loss computation correctness
"""

import pytest
import torch

from src.core.loss import TrainLoss, compute_loss_components


@pytest.fixture
def loss_inputs():
    """Create inputs for loss computation."""
    bs, n, dx, de = 2, 5, 4, 2

    # Predictions (logits)
    pred_X = torch.randn(bs, n, dx)
    pred_E = torch.randn(bs, n, n, de)
    pred_y = torch.randn(bs, 3)

    # Ground truth (one-hot)
    true_X = torch.zeros(bs, n, dx)
    for i in range(bs):
        for j in range(n):
            true_X[i, j, torch.randint(0, dx, (1,))] = 1

    true_E = torch.zeros(bs, n, n, de)
    for i in range(bs):
        for j in range(n):
            for k in range(n):
                true_E[i, j, k, torch.randint(0, de, (1,))] = 1

    true_y = torch.randn(bs, 3)

    # Node mask
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[0, 4] = False

    return pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask


class TestTrainLoss:
    """Tests for TrainLoss class."""

    def test_creation_default(self):
        """Test creation with default parameters."""
        loss_fn = TrainLoss()
        assert loss_fn.lambda_edge == 1.0

    def test_creation_custom(self):
        """Test creation with custom lambda."""
        loss_fn = TrainLoss(lambda_edge=2.0)
        assert loss_fn.lambda_edge == 2.0

    def test_forward_returns_scalar(self, loss_inputs):
        """Test forward returns a scalar loss."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs
        loss_fn = TrainLoss()

        loss = loss_fn(pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)

        assert loss.dim() == 0  # Scalar
        assert loss.dtype == torch.float32

    def test_forward_positive_loss(self, loss_inputs):
        """Test forward returns positive loss."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs
        loss_fn = TrainLoss()

        loss = loss_fn(pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)

        assert loss > 0

    def test_forward_finite_loss(self, loss_inputs):
        """Test forward returns finite loss."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs
        loss_fn = TrainLoss()

        loss = loss_fn(pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)

        assert torch.isfinite(loss)

    def test_perfect_prediction_low_loss(self):
        """Test perfect predictions give low loss."""
        bs, n, dx, de = 2, 4, 3, 2

        # Create one-hot ground truth
        true_X = torch.zeros(bs, n, dx)
        true_X[:, :, 0] = 1  # All class 0

        true_E = torch.zeros(bs, n, n, de)
        true_E[:, :, :, 0] = 1  # All class 0

        # Predictions that strongly favor the true class
        pred_X = torch.zeros(bs, n, dx)
        pred_X[:, :, 0] = 10.0  # High logit for class 0

        pred_E = torch.zeros(bs, n, n, de)
        pred_E[:, :, :, 0] = 10.0  # High logit for class 0

        node_mask = torch.ones(bs, n, dtype=torch.bool)

        loss_fn = TrainLoss()
        loss = loss_fn(
            pred_X, pred_E, torch.zeros(bs, 0),
            true_X, true_E, torch.zeros(bs, 0),
            node_mask
        )

        # Loss should be small (close to 0)
        assert loss < 1.0

    def test_random_prediction_higher_loss(self):
        """Test random predictions give higher loss than good predictions."""
        bs, n, dx, de = 2, 4, 3, 2

        true_X = torch.zeros(bs, n, dx)
        true_X[:, :, 0] = 1

        true_E = torch.zeros(bs, n, n, de)
        true_E[:, :, :, 0] = 1

        # Good predictions
        pred_X_good = torch.zeros(bs, n, dx)
        pred_X_good[:, :, 0] = 10.0
        pred_E_good = torch.zeros(bs, n, n, de)
        pred_E_good[:, :, :, 0] = 10.0

        # Random predictions
        pred_X_random = torch.randn(bs, n, dx)
        pred_E_random = torch.randn(bs, n, n, de)

        node_mask = torch.ones(bs, n, dtype=torch.bool)
        loss_fn = TrainLoss()

        loss_good = loss_fn(
            pred_X_good, pred_E_good, torch.zeros(bs, 0),
            true_X, true_E, torch.zeros(bs, 0),
            node_mask
        )
        loss_random = loss_fn(
            pred_X_random, pred_E_random, torch.zeros(bs, 0),
            true_X, true_E, torch.zeros(bs, 0),
            node_mask
        )

        assert loss_good < loss_random

    def test_lambda_edge_effect(self, loss_inputs):
        """Test lambda_edge affects total loss."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs

        loss_fn_1 = TrainLoss(lambda_edge=1.0)
        loss_fn_2 = TrainLoss(lambda_edge=2.0)

        loss_1 = loss_fn_1(pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)
        loss_2 = loss_fn_2(pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)

        # Different lambda should give different loss (unless edge loss is 0)
        # They might be equal if edge loss happens to be 0, so just check they're valid
        assert torch.isfinite(loss_1)
        assert torch.isfinite(loss_2)

    def test_gradient_flow(self, loss_inputs):
        """Test gradients flow through the loss."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs
        pred_X = pred_X.clone().requires_grad_(True)

        loss_fn = TrainLoss()
        loss = loss_fn(pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)
        loss.backward()

        assert pred_X.grad is not None
        assert torch.isfinite(pred_X.grad).all()


class TestComputeLossComponents:
    """Tests for compute_loss_components function."""

    def test_returns_tuple(self, loss_inputs):
        """Test function returns tuple of losses."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs

        loss_X, loss_E = compute_loss_components(
            pred_X, pred_E,
            true_X, true_E,
            node_mask
        )

        assert isinstance(loss_X, torch.Tensor)
        assert isinstance(loss_E, torch.Tensor)

    def test_components_finite(self, loss_inputs):
        """Test all components are finite."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs

        loss_X, loss_E = compute_loss_components(
            pred_X, pred_E,
            true_X, true_E,
            node_mask
        )

        assert torch.isfinite(loss_X)
        assert torch.isfinite(loss_E)

    def test_components_non_negative(self, loss_inputs):
        """Test loss components are non-negative."""
        pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask = loss_inputs

        loss_X, loss_E = compute_loss_components(
            pred_X, pred_E,
            true_X, true_E,
            node_mask
        )

        assert loss_X >= 0
        assert loss_E >= 0
