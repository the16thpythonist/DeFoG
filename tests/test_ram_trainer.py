"""
Tests for AdapterRAMTrainer (docs/dam_design.md step 7) -- the ablation arm.

RAM is GDPO with one substitution: the scored states are redrawn from
p_{t|1}(.|G1) instead of taken from the rollout trajectory. These tests pin that it
really is one substitution -- so a GDPO-vs-RAM difference is attributable -- plus the
two guards the review made mandatory.
"""

import pytest
import torch

from defog.core import DeFoGModel
from defog.core.adapter import AdaLNAdapter
from defog.core.guidance import _edge_upper_mask
from defog.core.ram import AdapterRAMTrainer
from defog.core.rl import AdapterGDPOTrainer


SEED = 21


@pytest.fixture
def base_model(small_model_config, node_counts_distribution):
    torch.manual_seed(SEED)
    return DeFoGModel(**small_model_config, noise_type="uniform",
                      node_counts=node_counts_distribution)


def _adapter(base, seed=SEED + 1):
    torch.manual_seed(seed)
    a = AdaLNAdapter.for_base(base, cond_dim=2, time_conditioned=True)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return a


def cond_sampler():
    return torch.randn(8, 2), torch.zeros(8, dtype=torch.long)


def edge_reward(X1, E1, node_mask, cond):
    return ((E1.argmax(-1) == 1).float() * _edge_upper_mask(node_mask)).sum(dim=(1, 2))


def _kw(**over):
    kw = dict(rollout_size=8, sample_steps=5, subsample_steps=2, minibatch_size=8,
              eta=1.0, omega=0.0, time_distortion="identity", lr=1e-2,
              ema_decay=None, seed=SEED, condition_sampler=cond_sampler)
    kw.update(over)
    return kw


# ================================================================ guards
def test_refuses_kl_coef_zero(base_model):
    with pytest.raises(ValueError, match="kl_coef > 0"):
        AdapterRAMTrainer(base_model, _adapter(base_model), edge_reward,
                          kl_coef=0.0, **_kw())


def test_refuses_grpo_advantages(base_model):
    with pytest.raises(ValueError, match="advantage_mode='grpo'"):
        AdapterRAMTrainer(base_model, _adapter(base_model), edge_reward,
                          kl_coef=0.1, **_kw(advantage_mode="grpo"))


@pytest.mark.parametrize("mode", ["mean", "none"])
def test_accepts_the_scale_preserving_advantage_modes(base_model, mode):
    AdapterRAMTrainer(base_model, _adapter(base_model), edge_reward,
                      kl_coef=0.1, **_kw(advantage_mode=mode))


def test_renoise_draws_is_a_literal_not_a_mirror(base_model):
    """Mirroring subsample_steps while the trace is off would give zero re-noised
    states -- an update that silently does nothing in exactly the configuration this
    arm needs. renoise_draws must override it, and must be the knob the experiment
    plan's RENOISE_DRAWS sweep actually moves."""
    tr = AdapterRAMTrainer(base_model, _adapter(base_model), edge_reward,
                           kl_coef=0.1, **_kw(subsample_steps=0))
    assert tr.renoise_draws == 16, "the default is not a literal 16"
    # In "match" mode the levels are distinct grid indices, so the count is capped at
    # sample_steps -- 5 on this fixture, 250 in production. What matters here is that
    # subsample_steps=0 did NOT silence the arm.
    n = len(tr._renoised_states(tr.rollout()))
    assert n == min(16, tr.sample_steps) == 5, f"got {n} states"

    deep = AdapterRAMTrainer(base_model, _adapter(base_model), edge_reward,
                             kl_coef=0.1, **_kw(subsample_steps=0, sample_steps=20))
    assert len(deep._renoised_states(deep.rollout())) == 16, \
        "renoise_draws is not driving the count once the grid is long enough"


def test_renoise_draws_moves_the_count_in_every_mode(base_model):
    """The RENOISE_DRAWS sweep must not be a no-op in the DEFAULT mode, which is the
    trap this nearly walked into: draw_times takes its count from step_indices in
    'match' mode, so renoise_draws was being ignored there."""
    for mode in ("match", "train", "ram", "uniform"):
        for n in (4, 9):
            tr = AdapterRAMTrainer(base_model, _adapter(base_model), edge_reward,
                                   kl_coef=0.1, renoise_draws=n, t_sampler=mode,
                                   **_kw(sample_steps=20))
            got = len(tr._renoised_states(tr.rollout()))
            assert got == n, f"mode={mode}, renoise_draws={n} gave {got} states"


# ================================================================ the substitution
def test_scores_at_renoised_states_not_trajectory_states(base_model):
    tr = AdapterRAMTrainer(base_model, _adapter(base_model), edge_reward,
                           kl_coef=0.1, renoise_draws=3, **_kw())
    assert tr.record_trace is False
    buf = tr.rollout()
    assert buf.states == [], "RAM rollout recorded trajectory states"
    states = tr._renoised_states(buf)
    assert len(states) == 3
    for X_t, E_t, t in states:
        assert X_t.shape == buf.X1.shape and E_t.shape == buf.E1.shape
        assert torch.equal(E_t, E_t.transpose(1, 2)), "re-noised E is not symmetric"


def test_is_gdpo_given_the_same_states(base_model):
    """The load-bearing claim of this arm: RAM's update IS GDPO's update. Feed both
    the identical states and the gradients must match exactly -- otherwise a
    GDPO-vs-RAM difference is not attributable to the state source."""
    from defog.core.rl import RolloutBuffer

    ad_a, ad_b = _adapter(base_model), _adapter(base_model)
    common = dict(kl_coef=0.1, **_kw())
    ram = AdapterRAMTrainer(base_model, ad_a, edge_reward, renoise_draws=2, **common)
    gdpo = AdapterGDPOTrainer(base_model, ad_b, edge_reward, **common)

    buf = ram.rollout()
    states = ram._renoised_states(buf)
    fixed = RolloutBuffer(states, buf.X1, buf.E1, buf.y, buf.node_mask,
                          buf.reward, buf.advantage)

    ram.opt.zero_grad(); gdpo.opt.zero_grad()
    m_ram = AdapterGDPOTrainer.update(ram, fixed)      # what RAM's update reduces to
    g_ram = [p.grad.detach().clone() for p in ad_a.parameters() if p.grad is not None]
    m_gdpo = gdpo.update(fixed)
    g_gdpo = [p.grad.detach().clone() for p in ad_b.parameters() if p.grad is not None]

    assert len(g_ram) == len(g_gdpo) > 0
    assert all(torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(g_ram, g_gdpo)), \
        "RAM and GDPO disagree on identical states -- the arm is not a clean ablation"
    assert abs(m_ram["loss"] - m_gdpo["loss"]) < 1e-5


# ================================================================ end to end
def test_step_runs_moves_the_adapter_and_leaves_the_base_alone(base_model):
    ad = _adapter(base_model)
    tr = AdapterRAMTrainer(base_model, ad, edge_reward, kl_coef=0.1,
                           renoise_draws=2, **_kw())
    before = [p.detach().clone() for p in ad.parameters()]
    hist = [tr.step() for _ in range(3)]
    assert all(v == v for m in hist for v in m.values()), "non-finite metric"
    # The reference is a frozen COPY of the adapter, so the KL is exactly 0 on the
    # first step by construction, and must become positive once the adapter moves --
    # that is the check that the term actually reaches the loss.
    assert hist[0]["kl"] == 0.0, f"KL {hist[0]['kl']} != 0 against an identical reference"
    assert max(m["kl"] for m in hist[1:]) > 0, "the KL term never reaches the loss"
    assert any(not torch.equal(b, p) for b, p in zip(before, ad.parameters()))
    assert all(p.grad is None for p in base_model.parameters()), "base accumulated grad"
