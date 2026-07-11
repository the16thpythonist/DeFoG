"""
Generation constraints for DeFoG.

A :class:`Constraint` is a projection of a partially-generated graph onto a
feasible set, applied by an ``InpaintingSampler`` inside its loop hooks. It is a
pure orchestration concern: the model never sees it.

The constraint is deliberately batch- and size-agnostic. It carries only the
*intrinsic* description of what is fixed (for :class:`SubgraphConstraint`, the
k-node core), and :meth:`Constraint.project` reads the batch size and node count
from the running tensors at call time, with ``limit_dist`` injected by the
sampler. So the same constraint object is reusable across ``num_samples`` /
``n_free`` without rebuilding.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Constraint(ABC):
    """A projection of a (batched, dense) graph state onto a feasible set."""

    @abstractmethod
    def project(self, X, E, tau, node_mask, limit_dist):
        """
        Project the running state onto the constraint at (normalized) time tau.

        Args:
            X: node one-hot (bs, n, dx)
            E: edge one-hot (bs, n, n, de)
            tau: scalar or (bs, 1) time in [0, 1]. tau=1 installs the exact target.
            node_mask: (bs, n) bool
            limit_dist: the model's LimitDistribution (for re-noising toward p_0).

        Returns:
            (X, E) with the constrained region overwritten; free region untouched.
        """
        raise NotImplementedError


class SubgraphConstraint(Constraint):
    """
    Freeze a k-node subgraph (rigid scaffold): the core atom classes AND the full
    k x k internal connectivity (bonded pairs stay bonded, non-bonded pairs stay
    non-bonded). New nodes and every bond touching them (core<->new, new<->new)
    are generated freely.

    RePaint semantics: at time tau the core is re-noised to the SAME level as the
    rest of the graph via the model's own interpolation
    ``p(x_tau | x_1) = tau * onehot(core) + (1 - tau) * limit`` -- so the network
    input stays on-distribution. At tau=1 the exact clean core is installed.

    Args:
        X_core: (k, dx) one-hot node features of the core.
        E_core: (k, k, de) one-hot edge features of the core (symmetric, class 0
            = no bond on the diagonal and between non-bonded core atoms).
        mode: "repaint" (re-noise to tau, default) or "clamp" (always install the
            exact core). "clamp" manufactures a more extreme off-distribution
            input at every step and is intended as a stress-test ablation, not a
            co-equal alternative to "repaint".
    """

    def __init__(self, X_core: torch.Tensor, E_core: torch.Tensor, mode: str = "repaint"):
        if X_core.dim() != 2:
            raise ValueError(f"X_core must be (k, dx); got {tuple(X_core.shape)}")
        if E_core.dim() != 3 or E_core.shape[0] != E_core.shape[1]:
            raise ValueError(f"E_core must be (k, k, de); got {tuple(E_core.shape)}")
        if E_core.shape[0] != X_core.shape[0]:
            raise ValueError("X_core and E_core disagree on k")
        if mode not in ("repaint", "clamp"):
            raise ValueError(f"mode must be 'repaint' or 'clamp'; got {mode!r}")
        self.X_core = X_core.float()
        self.E_core = E_core.float()
        self.mode = mode

    @property
    def k(self) -> int:
        return self.X_core.shape[0]

    @staticmethod
    def _align(onehot: torch.Tensor, target_classes: int) -> torch.Tensor:
        """Pad the last dim of a one-hot tensor up to target_classes with zeros.

        The core is built against the base vocabulary; limit_dist may carry an
        extra virtual class (absorbing noise). Padding keeps the write into the
        running tensors dimension-correct. No-op for marginal/uniform noise.
        """
        c = onehot.shape[-1]
        if c == target_classes:
            return onehot
        if c > target_classes:
            raise ValueError(
                f"core has {c} classes but limit_dist expects {target_classes}"
            )
        pad = list(onehot.shape[:-1]) + [target_classes - c]
        return torch.cat([onehot, onehot.new_zeros(pad)], dim=-1)

    def project(self, X, E, tau, node_mask, limit_dist):
        bs, n, dx = X.shape
        de = E.shape[-1]
        k = self.k
        device = X.device

        # Resolve a scalar tau (the whole batch shares one denoising time).
        if self.mode == "clamp":
            tau_val = 1.0
        elif isinstance(tau, (int, float)):
            tau_val = float(tau)
        else:
            tau_val = float(tau.reshape(-1)[0].item())

        Xc = self._align(self.X_core.to(device), dx)          # (k, dx)
        Ec = self._align(self.E_core.to(device), de)          # (k, k, de)

        X = X.clone()
        E = E.clone()

        # --- exact install at (or numerically at) tau = 1 -------------------
        if tau_val >= 1.0 - 1e-6:
            X[:, :k, :] = Xc.unsqueeze(0).expand(bs, k, dx)
            E[:, :k, :k, :] = Ec.unsqueeze(0).expand(bs, k, k, de)
            return X, E

        # --- otherwise re-noise the core to level tau ----------------------
        limX = limit_dist.X.to(device).float()                # (dx,)
        limE = limit_dist.E.to(device).float()                # (de,)

        # Nodes: prob = tau*onehot(core) + (1-tau)*limit, sampled per-sample.
        probX = tau_val * Xc + (1.0 - tau_val) * limX.unsqueeze(0)      # (k, dx)
        probX = probX.clamp_min(0).unsqueeze(0).expand(bs, k, dx).reshape(bs * k, dx)
        idxX = probX.multinomial(1).reshape(bs, k)
        X[:, :k, :] = F.one_hot(idxX, dx).float()

        # Edges: same interpolation, sampled on the upper triangle then mirrored
        # to keep the block symmetric (diagonal stays no-bond, class 0).
        probE = tau_val * Ec + (1.0 - tau_val) * limE.view(1, 1, de)    # (k, k, de)
        probE = probE.clamp_min(0).unsqueeze(0).expand(bs, k, k, de).reshape(bs * k * k, de)
        idxE = probE.multinomial(1).reshape(bs, k, k)

        iu = torch.triu(torch.ones(k, k, device=device), diagonal=1).bool()
        idxE_sym = torch.zeros(bs, k, k, dtype=idxE.dtype, device=device)
        idxE_sym[:, iu] = idxE[:, iu]
        idxE_sym = idxE_sym + idxE_sym.transpose(1, 2)         # mirror; diag stays 0
        E[:, :k, :k, :] = F.one_hot(idxE_sym.long(), de).float()

        return X, E
