"""
Training loss computation for discrete flow matching.

The loss is a cross-entropy between the predicted marginals and the
ground truth clean graph features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TrainLoss(nn.Module):
    """
    Cross-entropy loss for training the discrete flow matching model.

    The loss computes cross-entropy between predicted marginals and the
    true one-hot encoded features, with masking for padded nodes/edges.

    Args:
        lambda_edge: Weight for edge loss relative to node loss (default: 1.0)
        lambda_y: Weight for global feature loss (default: 0.0)

    Example:
        >>> loss_fn = TrainLoss(lambda_edge=5.0)
        >>> loss = loss_fn(pred_X, pred_E, pred_y, true_X, true_E, true_y, node_mask)
    """

    def __init__(
        self,
        lambda_edge: float = 1.0,
        lambda_y: float = 0.0,
    ):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_y = lambda_y

    def forward(
        self,
        pred_X: torch.Tensor,
        pred_E: torch.Tensor,
        pred_y: Optional[torch.Tensor],
        true_X: torch.Tensor,
        true_E: torch.Tensor,
        true_y: Optional[torch.Tensor],
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            pred_X: Predicted node logits (bs, n, dx)
            pred_E: Predicted edge logits (bs, n, n, de)
            pred_y: Predicted global features (bs, dy) or None
            true_X: True node features one-hot (bs, n, dx)
            true_E: True edge features one-hot (bs, n, n, de)
            true_y: True global features (bs, dy) or None
            node_mask: Boolean mask (bs, n) for valid nodes

        Returns:
            Scalar loss tensor
        """
        # Flatten for cross-entropy
        bs, n, dx = pred_X.shape
        de = pred_E.shape[-1]

        pred_X_flat = pred_X.reshape(-1, dx)  # (bs*n, dx)
        pred_E_flat = pred_E.reshape(-1, de)  # (bs*n*n, de)
        true_X_flat = true_X.reshape(-1, dx)  # (bs*n, dx)
        true_E_flat = true_E.reshape(-1, de)  # (bs*n*n, de)

        # Create masks from true values (non-zero rows are valid)
        mask_X = (true_X_flat.abs().sum(dim=-1) > 0)
        mask_E = (true_E_flat.abs().sum(dim=-1) > 0)

        # Compute node loss
        if mask_X.any():
            valid_pred_X = pred_X_flat[mask_X]
            valid_true_X = true_X_flat[mask_X]
            # Cross-entropy with soft targets
            loss_X = self._cross_entropy(valid_pred_X, valid_true_X)
        else:
            loss_X = torch.tensor(0.0, device=pred_X.device)

        # Compute edge loss
        if mask_E.any():
            valid_pred_E = pred_E_flat[mask_E]
            valid_true_E = true_E_flat[mask_E]
            loss_E = self._cross_entropy(valid_pred_E, valid_true_E)
        else:
            loss_E = torch.tensor(0.0, device=pred_E.device)

        # Compute global loss
        if pred_y is not None and true_y is not None and pred_y.numel() > 0 and true_y.numel() > 0:
            loss_y = self._cross_entropy(pred_y, true_y)
        else:
            loss_y = torch.tensor(0.0, device=pred_X.device)

        # Combine losses
        total_loss = loss_X + self.lambda_edge * loss_E + self.lambda_y * loss_y

        return total_loss

    def _cross_entropy(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with soft targets.

        Args:
            pred: Predicted logits (N, C)
            target: Target probabilities (N, C) - can be one-hot or soft

        Returns:
            Scalar cross-entropy loss
        """
        # Log softmax of predictions
        log_probs = F.log_softmax(pred, dim=-1)

        # Cross entropy: -sum(target * log(pred))
        loss = -(target * log_probs).sum(dim=-1)

        return loss.mean()

    def __repr__(self) -> str:
        return f"TrainLoss(lambda_edge={self.lambda_edge}, lambda_y={self.lambda_y})"


def compute_loss_components(
    pred_X: torch.Tensor,
    pred_E: torch.Tensor,
    true_X: torch.Tensor,
    true_E: torch.Tensor,
    node_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute separate node and edge losses for logging.

    Args:
        pred_X: Predicted node logits (bs, n, dx)
        pred_E: Predicted edge logits (bs, n, n, de)
        true_X: True node features (bs, n, dx)
        true_E: True edge features (bs, n, n, de)
        node_mask: Boolean mask (bs, n)

    Returns:
        Tuple of (node_loss, edge_loss)
    """
    dx = pred_X.shape[-1]
    de = pred_E.shape[-1]

    # Flatten
    pred_X_flat = pred_X.reshape(-1, dx)
    pred_E_flat = pred_E.reshape(-1, de)
    true_X_flat = true_X.reshape(-1, dx)
    true_E_flat = true_E.reshape(-1, de)

    # Masks
    mask_X = (true_X_flat.abs().sum(dim=-1) > 0)
    mask_E = (true_E_flat.abs().sum(dim=-1) > 0)

    # Node loss
    if mask_X.any():
        log_probs_X = F.log_softmax(pred_X_flat[mask_X], dim=-1)
        node_loss = -(true_X_flat[mask_X] * log_probs_X).sum(dim=-1).mean()
    else:
        node_loss = torch.tensor(0.0, device=pred_X.device)

    # Edge loss
    if mask_E.any():
        log_probs_E = F.log_softmax(pred_E_flat[mask_E], dim=-1)
        edge_loss = -(true_E_flat[mask_E] * log_probs_E).sum(dim=-1).mean()
    else:
        edge_loss = torch.tensor(0.0, device=pred_E.device)

    return node_loss, edge_loss
