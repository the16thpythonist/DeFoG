"""
The three-arm switch (docs/dam_design.md step 8 / D8).

Two things are checked here that the per-arm test modules cannot:

1. The DISPATCH CONTRACT -- all three trainers accept the same shared kwargs dict.
   The experiment builds one `_shared` dict and passes it to whichever arm is
   selected, so a signature drift in any arm breaks a cluster run at construction
   time, after the data is loaded.
2. The experiment's parameter block carries the knobs the plan's Run A needs, with
   the defaults the plan justifies.
"""

import pytest
import torch

from defog.core import (AdapterDAMTrainer, AdapterGDPOTrainer, AdapterRAMTrainer,
                        DeFoGModel)
from defog.core.adapter import AdaLNAdapter
from defog.core.guidance import _edge_upper_mask


SEED = 31


def _shared(**over):
    kw = dict(lr=1e-2, ema_decay=None, rollout_size=8, sample_steps=5, eta=1.0,
              omega=0.0, time_distortion="identity", subsample_steps=2,
              minibatch_size=8, lambda_edge=1.0, crn=True, size_dist=None,
              grad_clip=1.0, seed=SEED, device=torch.device("cpu"))
    kw.update(over)
    return kw


def _fixture(small_model_config, node_counts_distribution):
    torch.manual_seed(SEED)
    base = DeFoGModel(**small_model_config, noise_type="uniform",
                      node_counts=node_counts_distribution)
    torch.manual_seed(SEED + 1)
    ad = AdaLNAdapter.for_base(base, cond_dim=2, time_conditioned=True)
    with torch.no_grad():
        for p in ad.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return base, ad


def cond_sampler():
    return torch.randn(8, 2), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])


def edge_reward(X1, E1, node_mask, cond):
    return ((E1.argmax(-1) == 1).float() * _edge_upper_mask(node_mask)).sum(dim=(1, 2))


def _build(arm, base, ad):
    """Mirrors the dispatch in experiments/adapter_rl_finetune__zinc.py."""
    sh = _shared(condition_sampler=cond_sampler)
    if arm == "gdpo":
        return AdapterGDPOTrainer(base, ad, edge_reward, kl_coef=0.1, **sh)
    if arm == "ram":
        return AdapterRAMTrainer(base, ad, edge_reward, kl_coef=0.1,
                                 renoise_draws=2, t_sampler="match", **sh)
    return AdapterDAMTrainer(base, ad, edge_reward, dam_k=4, dam_lambda=0.3,
                             debias="snis", renoise_draws=2, t_sampler="match", **sh)


# ================================================================ dispatch
@pytest.mark.parametrize("arm", ["gdpo", "ram", "dam"])
def test_every_arm_accepts_the_shared_kwargs(small_model_config,
                                             node_counts_distribution, arm):
    base, ad = _fixture(small_model_config, node_counts_distribution)
    tr = _build(arm, base, ad)
    assert tr.rollout_size == 8 and tr.sample_steps == 5


@pytest.mark.parametrize("arm", ["gdpo", "ram", "dam"])
def test_every_arm_runs_one_iteration_end_to_end(small_model_config,
                                                 node_counts_distribution, arm):
    base, ad = _fixture(small_model_config, node_counts_distribution)
    before = [p.detach().clone() for p in ad.parameters()]
    tr = _build(arm, base, ad)
    m = tr.step()
    assert all(v == v for v in m.values()), f"{arm}: non-finite metric in {m}"
    for k in ("loss", "grad_norm", "reward_mean"):
        assert k in m, f"{arm}: missing {k}"
    assert all(p.grad is None for p in base.parameters()), f"{arm}: base accumulated grad"
    assert any(not torch.equal(b, p) for b, p in zip(before, ad.parameters())), \
        f"{arm}: adapter did not move"


def test_only_the_renoising_arms_skip_the_trace(small_model_config,
                                                node_counts_distribution):
    base, ad = _fixture(small_model_config, node_counts_distribution)
    assert _build("gdpo", base, ad).record_trace is True
    assert _build("ram", base, ad).record_trace is False
    assert _build("dam", base, ad).record_trace is False


def test_dam_carries_no_kl_coef_and_ram_requires_one(small_model_config,
                                                     node_counts_distribution):
    base, ad = _fixture(small_model_config, node_counts_distribution)
    assert _build("dam", base, ad).kl_coef == 0.0
    with pytest.raises(ValueError, match="kl_coef > 0"):
        AdapterRAMTrainer(base, ad, edge_reward, kl_coef=0.0,
                          **_shared(condition_sampler=cond_sampler))


# ================================================================ D8 parameters
def test_experiment_exposes_the_run_a_knobs():
    import experiments.adapter_rl_finetune__zinc as ex
    assert ex.ESTIMATOR == "gdpo", "the default arm must stay gdpo"
    assert ex.RL_ITERS == 0, "RL_ITERS must default off so existing runs are unchanged"
    assert ex.RENOISE_DRAWS == 16, "RENOISE_DRAWS must be a literal, not a mirror"
    assert ex.T_SAMPLER == "match"
    assert ex.DAM_K == 12
    # 0.3, not 1.0: PropertyMatchReward spans [-10, 0], and the tabular gate measured
    # KL to the tilted optimum at 0.0001 (lambda=0.3) against 0.0156 (lambda=1.0).
    assert ex.DAM_LAMBDA == 0.3
    assert ex.DAM_DEBIAS == "snis"
    assert ex.ROLLOUT_OMEGA == 0.0, "DAM requires omega == 0"


def test_rl_iters_must_divide_probe_every():
    """A pinned count that is not a multiple of PROBE_EVERY silently drops the last
    partial block from early-stop selection -- and that selection picks the snapshot
    that ships."""
    import experiments.adapter_rl_finetune__zinc as ex
    assert ex.PROBE_EVERY == 40
    for good in (40, 120, 200):
        assert good % ex.PROBE_EVERY == 0
    assert 130 % ex.PROBE_EVERY != 0, "the check would be vacuous"
