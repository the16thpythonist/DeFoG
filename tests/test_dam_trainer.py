"""
Structural tests for AdapterDAMTrainer (docs/dam_design.md step 6).

These pin the invariants that are checkable without a trained model. The BEHAVIOURAL
gate -- reward rises without collapsing diversity -- is deliberately not here: the
plan requires it to run at sample_steps >= 50, because at the fixture's default of 5
the memorylessness precondition the whole re-noising step rests on is violated by
~27 points, so a pass or a fail there would mean nothing.

The sharpest test in this file is test_zero_loss_at_init_with_constant_reward. At
construction the adapter EQUALS its frozen reference, so u^theta == u^base; and with a
constant reward the adjoint is exactly 1 because e^{-g} cancels between the two factors
of Eq. (13). The loss must therefore be exactly 0.0 -- not small, zero. Anything that
breaks the anchor, the ratio or the contraction breaks that.
"""

import pytest
import torch

from defog.core import DeFoGModel
from defog.core.adapter import AdaLNAdapter
from defog.core.dam import AdapterDAMTrainer
from defog.core.guidance import _edge_upper_mask


SEED = 11


@pytest.fixture
def base_model(small_model_config, node_counts_distribution):
    torch.manual_seed(SEED)
    return DeFoGModel(**small_model_config, noise_type="uniform",
                      node_counts=node_counts_distribution)


@pytest.fixture
def adapter(base_model):
    torch.manual_seed(SEED + 1)
    a = AdaLNAdapter.for_base(base_model, cond_dim=2, time_conditioned=True)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return a


def cond_sampler():
    return torch.randn(8, 2), torch.zeros(8, dtype=torch.long)


def constant_reward(X1, E1, node_mask, cond):
    return torch.ones(X1.shape[0])


def edge_reward(X1, E1, node_mask, cond):
    return ((E1.argmax(-1) == 1).float() * _edge_upper_mask(node_mask)).sum(dim=(1, 2))


def _trainer(base, adapter, reward=edge_reward, **kw):
    for k, v in dict(dam_k=4, renoise_draws=2, rollout_size=8, sample_steps=5,
                     minibatch_size=8, eta=1.0, omega=0.0,
                     time_distortion="identity", lr=1e-2, ema_decay=None,
                     seed=SEED).items():
        kw.setdefault(k, v)
    return AdapterDAMTrainer(base, adapter, reward,
                             condition_sampler=cond_sampler, **kw)


# ================================================================ the anchor
def test_zero_loss_at_init_with_constant_reward(base_model, adapter):
    """u^theta == u^base at init, and a constant reward makes the adjoint exactly 1
    because e^{-g} cancels between the two factors of Eq. (13). So gKL(u, u) = 0."""
    tr = _trainer(base_model, adapter, reward=constant_reward)
    m = tr.step()
    assert m["loss"] == 0.0, f"loss {m['loss']!r} is not exactly zero at the fixed point"
    # The gKL cancellation is exact, so the loss is a hard 0.0. The log-adjoint is not:
    # it comes out of a logsumexp whose terms cancel, so it carries one ulp of float32
    # (2**-24 = 5.96e-08). Anything above ~1e-6 is a real deviation.
    assert abs(m["log_adjoint"]) < 1e-6, f"adjoint {m['log_adjoint']} != 1 for a flat reward"
    assert m["grad_norm"] < 1e-6, f"gradient {m['grad_norm']} is not ~0 at the fixed point"


def test_nonconstant_reward_moves_the_adjoint_off_one(base_model, adapter):
    """The counterpart: a reward that varies must produce an adjoint that is not 1,
    otherwise nothing would ever be learned."""
    tr = _trainer(base_model, adapter, reward=edge_reward, dam_lambda=1.0)
    seen = [abs(tr.step()["log_adjoint"]) for _ in range(3)]
    assert max(seen) > 1e-6, f"adjoint stayed at 1 for a varying reward: {seen}"


# ================================================================ p^base identity
def test_ref_adapter_is_built_unconditionally(base_model, adapter):
    """For DAM the frozen reference is not a regulariser, it IS the distribution being
    tilted -- so unlike GDPO (which builds one only when kl_coef > 0) it must always
    exist. Anchoring to the unconditional base instead would target p_uncond e^{-g}
    and pull the adapter's conditioning out."""
    tr = _trainer(base_model, adapter)
    assert tr.ref_adapter is not None
    assert tr.kl_coef == 0.0, "DAM should carry no kl_coef; the anchor is structural"


def test_ref_adapter_is_frozen_and_distinct(base_model, adapter):
    tr = _trainer(base_model, adapter)
    before = {k: v.detach().clone() for k, v in tr.ref_adapter.state_dict().items()}
    assert all(not p.requires_grad for p in tr.ref_adapter.parameters())
    for _ in range(2):
        tr.step()
    after = tr.ref_adapter.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "reference moved"
    assert any(not torch.equal(before[k], v.detach())
               for k, v in tr.adapter.state_dict().items()), "adapter did not move"


# ================================================================ isolation
def test_gradient_reaches_only_the_adapter(base_model, adapter):
    tr = _trainer(base_model, adapter)
    tr.step()
    assert all(not p.requires_grad for p in base_model.parameters())
    assert all(p.grad is None for p in base_model.parameters()), "base accumulated grad"
    assert all(p.grad is None for p in tr.ref_adapter.parameters()), "reference accumulated grad"


def test_does_not_record_the_trace(base_model, adapter):
    """DAM scores at re-noised states, so paying to stash the trajectory would be
    waste -- but the endpoint stash and the CRN start must survive."""
    tr = _trainer(base_model, adapter)
    assert tr.record_trace is False
    buf = tr.rollout()
    assert buf.states == [], "DAM rollout recorded trajectory states"
    assert buf.X1 is not None and buf.node_mask is not None, "lost the endpoint stash"


# ================================================================ guards
def test_refuses_nonzero_omega(base_model, adapter):
    with pytest.raises(ValueError, match="omega == 0"):
        _trainer(base_model, adapter, omega=0.05)


def test_refuses_bad_debias(base_model, adapter):
    with pytest.raises(ValueError, match="debias must be"):
        _trainer(base_model, adapter, debias="nope")


def test_refuses_marginal_rdb(small_model_config, node_counts_distribution):
    torch.manual_seed(SEED)
    cfg = dict(small_model_config); cfg["rdb"] = "marginal"
    base = DeFoGModel(**cfg, noise_type="uniform", node_counts=node_counts_distribution)
    torch.manual_seed(SEED + 1)
    ad = AdaLNAdapter.for_base(base, cond_dim=2, time_conditioned=True)
    with pytest.raises(ValueError, match="one-directional mask"):
        _trainer(base, ad)


# ================================================================ reporting
def test_reports_the_health_metrics(base_model, adapter):
    """Section 8.5 needs these logged every run: the adjoint's mean, the clamp
    fraction (inert by construction when lambda*span <= clamp) and the residual gKL
    ratio, which is the only in-run signal for the projection gap."""
    tr = _trainer(base_model, adapter)
    m = tr.step()
    for k in ("log_adjoint", "adjoint_clamp_frac", "resid_gkl_ratio"):
        assert k in m, f"missing health metric {k}"
        assert m[k] == m[k], f"{k} is NaN"
    assert 0.0 <= m["adjoint_clamp_frac"] <= 1.0


@pytest.mark.parametrize("debias", ["snis", "raw"])
def test_both_debias_modes_run_and_stay_finite(base_model, adapter, debias):
    tr = _trainer(base_model, adapter, debias=debias)
    m = tr.step()
    assert all(v == v for v in m.values()), f"non-finite metric under debias={debias}"
