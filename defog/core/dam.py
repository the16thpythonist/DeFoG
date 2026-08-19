"""
Discrete Adjoint Matching (DAM, arXiv:2602.07132) for DeFoG.

DAM extends Adjoint Matching to CTMCs. Its optimal rate is

    u*_t(y, x) = u^base_t(y, x) * exp(-V_t(y) + V_t(x))

so the reward enters as an exponential of a value DIFFERENCE, never as a gradient --
which is why it works on a discrete state space and on non-differentiable rewards.
The correction is multiplicative on the base rate, and the loss is a generalized-KL
Bregman divergence between rates.

This module supplies the piece that makes that implementable against a DeFoG
checkpoint without touching the sampler: DeFoG's rate is an exact LINEAR functional
of the clean-graph head.

    Rbar_i(x_t -> j) = sum_c p_theta(x_1^i = c | x_t) * R_i(x_t -> j | c, t)

The conditional path p_{t|1}(z|x_1) = t*delta(z,x_1) + (1-t)*p_0(z) factorises over
coordinates and a CTMC jump changes exactly one coordinate, so R_i depends on x_1 only
through x_1^i. R_i(.|c, t) is therefore a network-free basis, obtained by evaluating
the existing RateMatrixDesigner with a one-hot prediction at class c.

Two properties of that identity, both measured (see tests/test_dam_rate.py):

* It is EXACT, not approximate -- relative error 0.0 across noise types, rdb modes
  and t including the regimes where the 1e-8 denominator floor and the >1e5 zeroing
  in _stabilize fire. The reason is structural: the head enters
  compute_rate_matrices only through sample_from_probs, so every downstream
  nonlinearity is applied per class by construction, and the expectation over that
  single draw IS the contraction.
* It is therefore also LOWER VARIANCE than the sampler, which estimates the same
  expectation with one draw.

See docs/dam_design.md sections 3 and 4.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "assert_valid_base_rate",
    "rate_basis",
    "marginal_rate",
    "REVERSIBLE_RDB",
]

#: detailed-balance designs whose R^DB is reversible w.r.t. p_t, hence admissible as
#: part of a base rate. Measured violation exactly 0.0 for these; 1.0x the full flow
#: scale for rdb="marginal", whose mask 1[limit(j) > limit(x_t)] is one-directional.
REVERSIBLE_RDB = ("general", "column", "entry")


def assert_valid_base_rate(designer, *, check_omega: bool = True) -> None:
    """Refuse rate configurations that break DAM's preconditions.

    DAM needs ``u^base`` to actually generate ``p^base``. Two of DeFoG's sampling
    knobs quietly break that, and both fail in the same way -- a one-directional flow
    that is not reversible w.r.t. ``p_t``, so it does not preserve the marginals:

    * ``omega > 0``. R^TG is an unnormalised additive push toward the sampled clean
      state with no reverse flow. Measured detailed-balance violation equals the FULL
      flow scale. With omega > 0 the rollout endpoints are not draws from p^theta_1
      and the (G1, G_t) joint DAM reconstructs is wrong.
    * ``rdb='marginal'``. Its mask ``1[limit(j) > limit(x_t)]`` is strictly
      one-directional: whenever it fires for x->y it is zero for y->x. Same failure
      mode, same measured violation. ``rdb`` is a free string on
      ``DeFoGModel.__init__`` persisted into every checkpoint, so a foreign
      checkpoint can select it without anyone noticing.

    A third configuration breaks the rate BASIS rather than the base rate:

    * ``rdb='column'`` with ``rdb_crit='p_x1_g_xt'`` reads ``X_1_pred.argmax(-1)``,
      the one place the head enters ``compute_rate_matrices`` without going through
      ``sample_from_probs``. A one-hot probe moves that argmax to class c, so the
      basis no longer describes the sampler (measured Monte-Carlo mean z = 86.5
      against 4000 draws, versus 0.054 for every other criterion).

    This is a DENYLIST on the criterion, not an allowlist: ``rdb_crit='dummy'`` is
    what ``configs/sample/sample_default.yaml`` sets, falls through to the branch
    using ``X_1_sampled``, and measures 0.0 error -- an allowlist would reject the
    repo's own default.
    """
    if check_omega and getattr(designer, "omega", 0.0) != 0:
        raise ValueError(
            f"DAM requires omega == 0, got {designer.omega}. R^TG is not "
            "marginal-preserving (measured detailed-balance violation = the full flow "
            "scale), so with omega > 0 the rate is not a base rate that generates "
            "p^base and the rollout endpoints are not draws from p^theta_1."
        )
    if designer.rdb not in REVERSIBLE_RDB:
        raise ValueError(
            f"DAM requires rdb in {REVERSIBLE_RDB}, got {designer.rdb!r}. "
            "rdb='marginal' uses a one-directional mask 1[limit(j) > limit(x_t)] "
            "whose R^DB is not reversible w.r.t. p_t -- the same failure mode that "
            "forbids omega > 0."
        )
    if designer.rdb == "column" and designer.rdb_crit == "p_x1_g_xt":
        raise ValueError(
            "DAM cannot use rdb='column' with rdb_crit='p_x1_g_xt': that path reads "
            "X_1_pred.argmax(-1) directly instead of going through sample_from_probs, "
            "so the one-hot rate basis does not describe the sampler and the linear "
            "identity behind marginal_rate() no longer holds."
        )


@torch.no_grad()
def rate_basis(
    model,
    X_t: torch.Tensor,
    E_t: torch.Tensor,
    t: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The network-free rate basis ``R_i(x_t -> j | c, t)``.

    Returns ``(BX, BE)`` of shape ``(dx, bs, n, dx)`` and ``(de, bs, n, n, de)``,
    where ``B[c]`` is the rate matrix the sampler would produce if the clean-graph
    prediction were one-hot at class ``c`` everywhere.

    Reuses ``model.rate_matrix_designer`` rather than constructing one, so the basis
    is built from exactly the configuration the sampler uses. ``eta`` is set the same
    way ``denoise_step`` sets it (a mutable attribute, restored afterwards);
    ``omega`` is forced to 0 and checked, per :func:`assert_valid_base_rate`.

    Cost is ``dx + de`` calls -- 13 on the real ZINC base -- measured at 32.9 ms
    against 488.7 ms for one network forward at bs=128, n=38.
    """
    designer = model.rate_matrix_designer
    # Validate BEFORE mutating, so a refusal cannot leave the designer disturbed.
    # omega is exempt here because rate_basis sets it itself: the run-level omega
    # check belongs in the trainer's __init__, where the run is configured.
    assert_valid_base_rate(designer, check_omega=False)
    eta_before, omega_before = designer.eta, designer.omega
    designer.eta, designer.omega = float(eta), 0.0
    try:
        dx = model.limit_dist.num_node_classes
        de = model.limit_dist.num_edge_classes
        bs, n = node_mask.shape
        dev = X_t.device

        BX, BE = [], []
        for c in range(dx):
            pX = F.one_hot(torch.full((bs, n), c, device=dev), dx).float()
            pE = F.one_hot(torch.zeros(bs, n, n, dtype=torch.long, device=dev), de).float()
            BX.append(model.rate_matrix_designer.compute_rate_matrices(
                t, node_mask, X_t, E_t, pX, pE)[0])
        for c in range(de):
            pX = F.one_hot(torch.zeros(bs, n, dtype=torch.long, device=dev), dx).float()
            pE = F.one_hot(torch.full((bs, n, n), c, device=dev), de).float()
            BE.append(model.rate_matrix_designer.compute_rate_matrices(
                t, node_mask, X_t, E_t, pX, pE)[1])
    finally:
        designer.eta, designer.omega = eta_before, omega_before

    BX = torch.stack(BX)                                   # (dx, bs, n, dx)
    BE = torch.stack(BE)                                   # (de, bs, n, n, de)

    # Padded entries are garbage by construction: sample_from_probs fills masked
    # nodes and the edge diagonal with a uniform distribution, so their sampled class
    # -- hence their basis row -- is random per call. Zero them so the basis is
    # deterministic everywhere a caller might look.
    nm = node_mask[None, :, :, None]
    em = (node_mask[:, :, None] & node_mask[:, None, :])[None, :, :, :, None]
    return BX * nm, BE * em


def marginal_rate(
    pX: torch.Tensor,
    pE: torch.Tensor,
    BX: torch.Tensor,
    BE: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contract a clean-graph head against the rate basis: the exact marginal rate.

    ``pX`` ``(bs, n, dx)`` and ``pE`` ``(bs, n, n, de)`` are the network's clean-graph
    probabilities (softmax outputs, NOT logits). Gradient flows through them, which is
    the whole point -- ``RateMatrixDesigner.compute_rate_matrices`` samples the clean
    graph internally and therefore carries no gradient to the head at all.
    """
    return (
        torch.einsum("cbnj,bnc->bnj", BX, pX),
        torch.einsum("cbnmj,bnmc->bnmj", BE, pE),
    )
