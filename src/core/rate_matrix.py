"""
Rate matrix computation for discrete flow matching.

The rate matrix R_t defines the CTMC (Continuous-Time Markov Chain) dynamics
used during sampling. It consists of three components:

R_t = R*_t + R^DB_t + R^TG_t

Where:
- R*_t: Base flow matching rate matrix (from Equation 3 in paper)
- R^DB_t: Detailed balance stochasticity (controlled by eta parameter)
- R^TG_t: Target guidance (controlled by omega parameter)

The key insight of DeFoG is that these rate matrices are computed at sampling
time using the network's predictions, not fixed during training. This enables
changing eta, omega without retraining.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Literal, Tuple

from .noise import LimitDistribution, sample_from_probs


RDBType = Literal["general", "marginal", "column", "entry"]


class RateMatrixDesigner:
    """
    Computes rate matrices for CTMC sampling.

    The rate matrix R_t determines transition probabilities in the denoising
    process. This class is decoupled from training - parameters can be changed
    at sampling time without retraining the model.

    Args:
        rdb: Detailed balance design type ("general", "marginal", "column", "entry")
        rdb_crit: Criterion for column selection in "column" mode
        eta: Stochasticity parameter - controls R^DB contribution
        omega: Target guidance parameter - controls R^TG contribution
        limit_dist: LimitDistribution defining noise distribution

    Example:
        >>> designer = RateMatrixDesigner(
        ...     rdb="column", rdb_crit="x_1",
        ...     eta=50.0, omega=0.05,
        ...     limit_dist=limit_dist
        ... )
        >>> R_X, R_E = designer.compute_rate_matrices(t, node_mask, G_t, G_1_pred)
    """

    def __init__(
        self,
        rdb: RDBType = "column",
        rdb_crit: str = "x_1",
        eta: float = 0.0,
        omega: float = 0.0,
        limit_dist: LimitDistribution = None,
    ):
        self.rdb = rdb
        self.rdb_crit = rdb_crit
        self.eta = eta
        self.omega = omega
        self.limit_dist = limit_dist

        if limit_dist is not None:
            self.num_classes_X = limit_dist.num_node_classes
            self.num_classes_E = limit_dist.num_edge_classes
        else:
            self.num_classes_X = 0
            self.num_classes_E = 0

    def compute_rate_matrices(
        self,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        X_1_pred: torch.Tensor,
        E_1_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rate matrices R_t_X and R_t_E.

        Args:
            t: Current time (bs, 1) in [0, 1]
            node_mask: Boolean mask (bs, n) indicating valid nodes
            X_t: Current node features (bs, n, dx) - one-hot
            E_t: Current edge features (bs, n, n, de) - one-hot
            X_1_pred: Predicted clean node probabilities (bs, n, dx)
            E_1_pred: Predicted clean edge probabilities (bs, n, n, de)

        Returns:
            Tuple of (R_t_X, R_t_E):
            - R_t_X: (bs, n, dx) rate matrix for nodes
            - R_t_E: (bs, n, n, de) rate matrix for edges
        """
        # Get current state labels
        X_t_label = X_t.argmax(-1, keepdim=True)  # (bs, n, 1)
        E_t_label = E_t.argmax(-1, keepdim=True)  # (bs, n, n, 1)

        # Sample x_1 from predicted distribution
        sampled = sample_from_probs(X_1_pred, E_1_pred, node_mask)
        X_1_sampled = sampled.X  # (bs, n) class indices
        E_1_sampled = sampled.E  # (bs, n, n) class indices

        # Compute intermediate variables needed for rate matrix computation
        dfm_vars = self._compute_dfm_variables(
            t, X_t_label, E_t_label, X_1_sampled, E_1_sampled
        )

        # Compute R* (base flow matching rate)
        Rstar_X, Rstar_E = self._compute_Rstar(dfm_vars)

        # Compute R^DB (detailed balance stochasticity)
        Rdb_X, Rdb_E = self._compute_RDB(
            X_t_label, E_t_label,
            X_1_pred, E_1_pred,
            X_1_sampled, E_1_sampled,
            node_mask, t, dfm_vars
        )

        # Compute R^TG (target guidance)
        Rtg_X, Rtg_E = self._compute_Rtg(
            X_1_sampled, E_1_sampled,
            X_t_label, E_t_label,
            dfm_vars
        )

        # Combine: R_t = R* + R^DB + R^TG
        R_t_X = Rstar_X + Rdb_X + Rtg_X
        R_t_E = Rstar_E + Rdb_E + Rtg_E

        # Stabilize for numerical safety
        R_t_X, R_t_E = self._stabilize(R_t_X, R_t_E, dfm_vars)

        return R_t_X, R_t_E

    def _compute_dfm_variables(
        self,
        t: torch.Tensor,
        X_t_label: torch.Tensor,
        E_t_label: torch.Tensor,
        X_1_sampled: torch.Tensor,
        E_1_sampled: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute intermediate variables for rate matrix calculation."""
        device = X_t_label.device

        # Time derivative: d/dt p(x_t | x_1)
        dt_p_X, dt_p_E = self._dt_p_xt_g_x1(X_1_sampled, E_1_sampled, device)

        # Gather at current state
        dt_p_at_Xt = dt_p_X.gather(-1, X_t_label).squeeze(-1)
        dt_p_at_Et = dt_p_E.gather(-1, E_t_label).squeeze(-1)

        # p(x_t | x_1)
        pt_X, pt_E = self._p_xt_g_x1(X_1_sampled, E_1_sampled, t, device)

        # Gather at current state
        pt_at_Xt = pt_X.gather(-1, X_t_label).squeeze(-1)
        pt_at_Et = pt_E.gather(-1, E_t_label).squeeze(-1)

        # Count non-zero probabilities (for normalization)
        Z_t_X = (pt_X > 0).sum(dim=-1).float()
        Z_t_E = (pt_E > 0).sum(dim=-1).float()

        return {
            "pt_vals_X": pt_X,
            "pt_vals_E": pt_E,
            "pt_vals_at_Xt": pt_at_Xt,
            "pt_vals_at_Et": pt_at_Et,
            "dt_p_vals_X": dt_p_X,
            "dt_p_vals_E": dt_p_E,
            "dt_p_vals_at_Xt": dt_p_at_Xt,
            "dt_p_vals_at_Et": dt_p_at_Et,
            "Z_t_X": Z_t_X,
            "Z_t_E": Z_t_E,
        }

    def _p_xt_g_x1(
        self,
        X_1: torch.Tensor,
        E_1: torch.Tensor,
        t: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute p(x_t | x_1) = t * delta(x_1) + (1-t) * p_0.

        Linear interpolation from clean state to noise distribution.
        """
        limit_X = self.limit_dist.X.to(device)
        limit_E = self.limit_dist.E.to(device)

        t_time = t.squeeze(-1)[:, None, None]

        X1_onehot = F.one_hot(X_1, num_classes=len(limit_X)).float()
        E1_onehot = F.one_hot(E_1, num_classes=len(limit_E)).float()

        pt_X = t_time * X1_onehot + (1 - t_time) * limit_X[None, None, :]
        pt_E = t_time[:, None] * E1_onehot + (1 - t_time[:, None]) * limit_E[None, None, None, :]

        return pt_X.clamp(0, 1), pt_E.clamp(0, 1)

    def _dt_p_xt_g_x1(
        self,
        X_1: torch.Tensor,
        E_1: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute d/dt p(x_t | x_1) = delta(x_1) - p_0.

        Time derivative of the interpolation.
        """
        limit_X = self.limit_dist.X.to(device)
        limit_E = self.limit_dist.E.to(device)

        X1_onehot = F.one_hot(X_1, num_classes=len(limit_X)).float()
        E1_onehot = F.one_hot(E_1, num_classes=len(limit_E)).float()

        dt_X = X1_onehot - limit_X[None, None, :]
        dt_E = E1_onehot - limit_E[None, None, None, :]

        return dt_X, dt_E

    def _compute_Rstar(
        self,
        dfm_vars: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute R* (base flow matching rate matrix).

        R*_{ij} = [d/dt p(j|x_1) - d/dt p(x_t|x_1)]+ / (Z_t * p(x_t|x_1))
        """
        dt_p_X = dfm_vars["dt_p_vals_X"]
        dt_p_E = dfm_vars["dt_p_vals_E"]
        dt_p_at_Xt = dfm_vars["dt_p_vals_at_Xt"]
        dt_p_at_Et = dfm_vars["dt_p_vals_at_Et"]
        pt_at_Xt = dfm_vars["pt_vals_at_Xt"]
        pt_at_Et = dfm_vars["pt_vals_at_Et"]
        Z_t_X = dfm_vars["Z_t_X"]
        Z_t_E = dfm_vars["Z_t_E"]

        # Numerator: [d/dt p(j|x_1) - d/dt p(x_t|x_1)]+
        inner_X = dt_p_X - dt_p_at_Xt[:, :, None]
        inner_E = dt_p_E - dt_p_at_Et[:, :, :, None]
        numer_X = F.relu(inner_X)
        numer_E = F.relu(inner_E)

        # Denominator: Z_t * p(x_t|x_1)
        denom_X = Z_t_X * pt_at_Xt
        denom_E = Z_t_E * pt_at_Et

        # Final R*
        Rstar_X = numer_X / (denom_X[:, :, None] + 1e-8)
        Rstar_E = numer_E / (denom_E[:, :, :, None] + 1e-8)

        return Rstar_X, Rstar_E

    def _compute_RDB(
        self,
        X_t_label: torch.Tensor,
        E_t_label: torch.Tensor,
        X_1_pred: torch.Tensor,
        E_1_pred: torch.Tensor,
        X_1_sampled: torch.Tensor,
        E_1_sampled: torch.Tensor,
        node_mask: torch.Tensor,
        t: torch.Tensor,
        dfm_vars: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute R^DB (detailed balance stochasticity component).

        This adds exploration/error correction to the sampling process.
        Controlled by eta parameter.
        """
        if self.eta == 0:
            return (
                torch.zeros_like(dfm_vars["pt_vals_X"]),
                torch.zeros_like(dfm_vars["pt_vals_E"])
            )

        pt_X = dfm_vars["pt_vals_X"]
        pt_E = dfm_vars["pt_vals_E"]
        dx = pt_X.shape[-1]
        de = pt_E.shape[-1]

        # Build mask based on rdb type
        if self.rdb == "general":
            x_mask = torch.ones_like(pt_X)
            e_mask = torch.ones_like(pt_E)

        elif self.rdb == "marginal":
            x_limit = self.limit_dist.X.to(pt_X.device)
            e_limit = self.limit_dist.E.to(pt_E.device)
            Xt_marginal = x_limit[X_t_label]
            Et_marginal = e_limit[E_t_label]
            x_mask = x_limit.unsqueeze(0).unsqueeze(0).expand_as(pt_X) > Xt_marginal
            e_mask = e_limit.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pt_E) > Et_marginal

        elif self.rdb == "column":
            # Select column based on criterion
            if self.rdb_crit == "max_marginal":
                x_col = self.limit_dist.X.argmax().expand_as(X_t_label)
                e_col = self.limit_dist.E.argmax().expand_as(E_t_label)
            elif self.rdb_crit == "x_t":
                x_col = X_t_label.squeeze(-1)
                e_col = E_t_label.squeeze(-1)
            elif self.rdb_crit == "x_1":
                x_col = X_1_sampled
                e_col = E_1_sampled
            elif self.rdb_crit == "p_x1_g_xt":
                x_col = X_1_pred.argmax(dim=-1)
                e_col = E_1_pred.argmax(dim=-1)
            else:
                x_col = X_1_sampled
                e_col = E_1_sampled

            # Create one-hot mask
            x_mask = F.one_hot(x_col, num_classes=dx).float()
            e_mask = F.one_hot(e_col, num_classes=de).float()

            # Also allow transitions where we're already at the target
            x_at_target = (x_col.unsqueeze(-1) == X_t_label).squeeze(-1)
            e_at_target = (e_col.unsqueeze(-1) == E_t_label).squeeze(-1)
            x_mask[x_at_target] = 1.0
            e_mask[e_at_target] = 1.0

        else:  # "entry" or default
            x_mask = torch.ones_like(pt_X)
            e_mask = torch.ones_like(pt_E)

        # R^DB = eta * p(x_t|x_1) * mask
        Rdb_X = pt_X * x_mask * self.eta
        Rdb_E = pt_E * e_mask * self.eta

        return Rdb_X, Rdb_E

    def _compute_Rtg(
        self,
        X_1_sampled: torch.Tensor,
        E_1_sampled: torch.Tensor,
        X_t_label: torch.Tensor,
        E_t_label: torch.Tensor,
        dfm_vars: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute R^TG (target guidance component).

        Amplifies transitions toward the predicted clean state.
        Controlled by omega parameter.
        """
        if self.omega == 0:
            return (
                torch.zeros_like(dfm_vars["pt_vals_X"]),
                torch.zeros_like(dfm_vars["pt_vals_E"])
            )

        pt_at_Xt = dfm_vars["pt_vals_at_Xt"]
        pt_at_Et = dfm_vars["pt_vals_at_Et"]
        Z_t_X = dfm_vars["Z_t_X"]
        Z_t_E = dfm_vars["Z_t_E"]

        # One-hot of sampled x_1
        X1_onehot = F.one_hot(X_1_sampled, num_classes=self.num_classes_X).float()
        E1_onehot = F.one_hot(E_1_sampled, num_classes=self.num_classes_E).float()

        # Mask: only add guidance when x_1 != x_t
        mask_X = (X_1_sampled.unsqueeze(-1) != X_t_label).float()
        mask_E = (E_1_sampled.unsqueeze(-1) != E_t_label).float()

        # Numerator
        numer_X = X1_onehot * self.omega * mask_X
        numer_E = E1_onehot * self.omega * mask_E

        # Denominator
        denom_X = Z_t_X * pt_at_Xt
        denom_E = Z_t_E * pt_at_Et

        # Final R^TG
        Rtg_X = numer_X / (denom_X[:, :, None] + 1e-8)
        Rtg_E = numer_E / (denom_E[:, :, :, None] + 1e-8)

        return Rtg_X, Rtg_E

    def _stabilize(
        self,
        R_t_X: torch.Tensor,
        R_t_E: torch.Tensor,
        dfm_vars: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stabilize rate matrices for numerical safety."""
        pt_X = dfm_vars["pt_vals_X"]
        pt_E = dfm_vars["pt_vals_E"]
        pt_at_Xt = dfm_vars["pt_vals_at_Xt"]
        pt_at_Et = dfm_vars["pt_vals_at_Et"]

        # Replace NaN and Inf
        R_t_X = torch.nan_to_num(R_t_X, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_E = torch.nan_to_num(R_t_E, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp large values
        R_t_X = R_t_X.clamp(max=1e5)
        R_t_E = R_t_E.clamp(max=1e5)

        # Zero out where p(x_t|x_1) = 0
        dx = R_t_X.shape[-1]
        de = R_t_E.shape[-1]
        R_t_X[(pt_at_Xt == 0)[:, :, None].expand(-1, -1, dx)] = 0.0
        R_t_E[(pt_at_Et == 0)[:, :, :, None].expand(-1, -1, -1, de)] = 0.0

        # Zero out where p(j|x_1) = 0
        R_t_X[pt_X == 0] = 0.0
        R_t_E[pt_E == 0] = 0.0

        return R_t_X, R_t_E

    def __repr__(self) -> str:
        return (
            f"RateMatrixDesigner(rdb={self.rdb}, rdb_crit={self.rdb_crit}, "
            f"eta={self.eta}, omega={self.omega})"
        )
