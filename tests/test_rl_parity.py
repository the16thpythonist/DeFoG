"""
Refactor parity gate for the GDPO trainer stack (docs/dam_design.md section 7).

This test exists to make the ``RLTrainerBase`` extraction safe. It compares the
FROZEN pre-refactor trainers in ``tests/_gdpo_frozen.py`` against the live ones in
``defog.core.rl``, and requires bit-identical metrics and weights.

Two things it deliberately does NOT do, both for measured reasons:

* It does not use the ``small_model`` fixture. That fixture constructs the model
  BEFORE the test body runs, so its weights come from the global RNG seeded at
  process start; the same protocol measured three different weight hashes on three
  runs. Here the model is seeded and then constructed inside the test body, which
  measured bit-identical ``fit(3)`` across separate processes.
* It does not commit a ``state_dict`` hash. Such a hash is machine-locked: measured,
  it differs across 1/4/8/12 CPU threads (29/132 tensors differ) and again when
  ``ATEN_CPU_CAPABILITY`` changes (114/132 tensors), and
  ``torch.use_deterministic_algorithms(True)`` is a no-op on this path. Comparing
  frozen against live IN THE SAME PROCESS removes the machine from the comparison
  entirely.

If this test fails after the refactor, the refactor is wrong. There is no other
reading.
"""

import pytest
import torch
import torch.nn.functional as F

from defog.core import DeFoGModel
from defog.core import rl as live
from defog.core.guidance import _edge_upper_mask

import _gdpo_frozen as frozen   # tests/ has no __init__.py; pytest prepends this dir


SEED = 1234


# ------------------------------------------------------------------ helpers
def _build_model(cfg, node_counts):
    """Seed THEN construct, so the weights are reproducible across processes."""
    torch.manual_seed(SEED)
    return DeFoGModel(**cfg, noise_type="uniform", node_counts=node_counts)


def _build_adapter(base):
    from defog.core.adapter import AdaLNAdapter
    torch.manual_seed(SEED + 1)
    a = AdaLNAdapter.for_base(base, cond_dim=2, time_conditioned=True)
    with torch.no_grad():                 # off the zero-init no-op
        for p in a.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return a


def edge_count_reward(X1, E1, node_mask):
    emask = _edge_upper_mask(node_mask)
    return ((E1.argmax(-1) == 1).float() * emask).sum(dim=(1, 2))


def cond_edge_reward(X1, E1, node_mask, cond):
    return edge_count_reward(X1, E1, node_mask)


def cond_sampler():
    return torch.randn(8, 2), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])


def _snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _assert_identical_state(a, b, what):
    assert a.keys() == b.keys(), f"{what}: different state_dict keys"
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    assert not bad, (
        f"{what}: {len(bad)}/{len(a)} tensors differ after the refactor, "
        f"first offenders {bad[:5]}"
    )


def _assert_identical_history(h1, h2, what):
    assert len(h1) == len(h2), f"{what}: different iteration counts"
    for i, (m1, m2) in enumerate(zip(h1, h2)):
        assert m1.keys() == m2.keys(), f"{what}: iter {i} metric keys differ"
        for k in m1:
            assert m1[k] == m2[k], (
                f"{what}: iter {i} metric {k!r} differs: {m1[k]!r} vs {m2[k]!r}"
            )


# ------------------------------------------------------------------ GDPO
# (kl on/off, ema on/off, positive_only, adaptive kl, crn -- all exercised)
GDPO_CONFIGS = [
    pytest.param(
        dict(rollout_size=8, sample_steps=5, subsample_steps=2, minibatch_size=4,
             eta=0.0, time_distortion="identity", advantage_mode="grpo",
             kl_coef=0.0, lr=1e-3, ema_decay=None),
        id="minimal",
    ),
    pytest.param(
        dict(rollout_size=8, sample_steps=5, subsample_steps=2, minibatch_size=4,
             eta=1.0, time_distortion="polydec", advantage_mode="mean",
             positive_only=True, kl_coef=0.3, kl_target=0.05, lr=1e-3,
             ema_decay=0.99, subsample="late"),
        id="everything-on",
    ),
]


@pytest.mark.parametrize("kw", GDPO_CONFIGS)
def test_gdpo_trainer_bit_identical_across_refactor(small_model_config,
                                                    node_counts_distribution, kw):
    results = []
    for mod in (frozen, live):
        model = _build_model(small_model_config, node_counts_distribution)
        torch.manual_seed(SEED)
        trainer = mod.GDPOTrainer(model, edge_count_reward, seed=SEED, **kw)
        history = trainer.fit(3)
        results.append((history, _snapshot(model)))

    _assert_identical_history(results[0][0], results[1][0], "GDPOTrainer history")
    _assert_identical_state(results[0][1], results[1][1], "GDPOTrainer weights")


def test_gdpo_trainer_actually_moved_the_weights(small_model_config,
                                                 node_counts_distribution):
    """Guard against a vacuous parity pass: if fit() were a no-op the comparison
    above would succeed trivially."""
    model = _build_model(small_model_config, node_counts_distribution)
    before = _snapshot(model)
    torch.manual_seed(SEED)
    live.GDPOTrainer(model, edge_count_reward, seed=SEED, **GDPO_CONFIGS[0].values[0]).fit(2)
    after = _snapshot(model)
    assert any(not torch.equal(before[k], after[k]) for k in before), \
        "fit() did not change any weight -- the parity test would be vacuous"


# ------------------------------------------------------------------ adapter
ADAPTER_CONFIGS = [
    pytest.param(
        dict(rollout_weight=1.0, kl_coef=0.0, rollout_size=8, sample_steps=5,
             subsample_steps=2, minibatch_size=4, eta=0.0,
             time_distortion="identity", lr=1e-2, ema_decay=None, crn=False),
        id="w1-nokl",
    ),
    pytest.param(
        dict(rollout_weight=2.0, kl_coef=0.1, kl_target=0.02, rollout_size=8,
             sample_steps=5, subsample_steps=2, minibatch_size=4, eta=1.0,
             time_distortion="polydec", lr=1e-2, ema_decay=0.99, crn=True),
        id="w2-kl-crn",
    ),
]


@pytest.mark.parametrize("kw", ADAPTER_CONFIGS)
def test_adapter_gdpo_trainer_bit_identical_across_refactor(small_model_config,
                                                            node_counts_distribution, kw):
    results = []
    for mod in (frozen, live):
        base = _build_model(small_model_config, node_counts_distribution)
        adapter = _build_adapter(base)
        torch.manual_seed(SEED)
        trainer = mod.AdapterGDPOTrainer(
            base, adapter, cond_edge_reward, condition_sampler=cond_sampler,
            seed=SEED, **kw,
        )
        history = trainer.fit(3)
        results.append((history, _snapshot(adapter), _snapshot(base)))

    _assert_identical_history(results[0][0], results[1][0], "AdapterGDPOTrainer history")
    _assert_identical_state(results[0][1], results[1][1], "AdapterGDPOTrainer adapter")
    _assert_identical_state(results[0][2], results[1][2], "AdapterGDPOTrainer base")


def test_adapter_trainer_actually_moved_the_adapter(small_model_config,
                                                    node_counts_distribution):
    base = _build_model(small_model_config, node_counts_distribution)
    adapter = _build_adapter(base)
    before = _snapshot(adapter)
    torch.manual_seed(SEED)
    live.AdapterGDPOTrainer(
        base, adapter, cond_edge_reward, condition_sampler=cond_sampler,
        seed=SEED, **ADAPTER_CONFIGS[0].values[0],
    ).fit(2)
    assert any(not torch.equal(before[k], v) for k, v in _snapshot(adapter).items()), \
        "adapter did not move -- the parity test would be vacuous"


# ------------------------------------------------------------------ determinism
def test_seed_then_construct_is_reproducible(small_model_config, node_counts_distribution):
    """The protocol this whole file rests on: seeding before construction makes the
    model reproducible. (Cross-PROCESS reproducibility is checked separately in CI /
    by running this file twice; within a process this is the necessary condition.)"""
    a = _snapshot(_build_model(small_model_config, node_counts_distribution))
    b = _snapshot(_build_model(small_model_config, node_counts_distribution))
    _assert_identical_state(a, b, "seed-then-construct")
