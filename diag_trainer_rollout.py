#!/usr/bin/env python
"""Call the REAL AdapterGDPOTrainer.rollout() once and tally its reward.

diag_rollout_reward.py rebuilt the rollout by hand and got a healthy mean of -1.003 with
1/128 at the floor. Job 43175, on the same adapter/base/config, reports ~-9.5 from iter 0 --
before any weight update, so the difference cannot be the adapter. Every documented rollout
parameter matches. So construct the actual trainer the experiment constructs and ask it,
rather than approximating it again.
"""
from __future__ import annotations

import collections
import importlib.util
import sys

import torch

sys.path.insert(0, ".")


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "rl_exp", "experiments/adapter_rl_finetune__zinc.py")
    rl_exp = importlib.util.module_from_spec(spec)
    sys.modules["rl_exp"] = rl_exp
    spec.loader.exec_module(rl_exp)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from defog.core import (AdaLNAdapter, AdapterGDPOTrainer, DeFoGModel,
                            HeadPropertyMatchReward, LearnedSizeDistribution, PropertyHead)
    from defog.core.rl import make_condition_sampler
    from defog.domains.molecule import build_encoders

    dev = "cuda"
    atoms, bonds, kek, _ = rl_exp._vocabulary("e1_kekulized")
    ae, adec, be, bdec = build_encoders(atoms, bonds)

    base = DeFoGModel.load("ckpts/zinc_rl2_seed42/best_model", device=dev).eval()
    adapter = AdaLNAdapter.load("ckpts/qed_adapter_pre_rl.ckpt", device=dev)
    head = PropertyHead.load("ckpts/heads/qed_head.ckpt", device=dev)
    size_dist = LearnedSizeDistribution.load("ckpts/heads/qed_head_size.ckpt")
    prop_std = 0.1339

    reward = HeadPropertyMatchReward(head, ae, be, adec, bdec, dev, scale=prop_std,
                                     invalid_reward=-10.0, disconnect_reward=-4.0,
                                     prop_clamp=3.0)
    cond_sampler = make_condition_sampler(0.4, 0.9, 128, 8, seed=42)

    trainer = AdapterGDPOTrainer(
        base, adapter, reward, kl_coef=0.05, lr=1e-4, ema_decay=0.99,
        rollout_size=128, sample_steps=250, eta=1.0, omega=0.0,
        time_distortion="polydec", condition_sampler=cond_sampler,
        subsample_steps=16, minibatch_size=16, crn=True, size_dist=size_dist,
        seed=42, device=dev,
    )
    print(f"trainer: rollout_weight={trainer.rollout_weight} mode={trainer.rollout_mode} "
          f"num_nodes={getattr(trainer, 'num_nodes', 'unset')!r} "
          f"eta={trainer.eta} steps={trainer.sample_steps} td={trainer.time_distortion}")
    print(f"reward object on trainer: {type(getattr(trainer, 'cond_reward', None)).__name__}")

    with torch.no_grad():
        buf = trainer.rollout()
    r = buf.reward.float()
    tally = collections.Counter()
    for v in r.tolist():
        tally["floor (-10)" if v <= -9.99 else
              "disconnect (-4)" if abs(v + 4) < 1e-6 else "scored"] += 1
    print(f"\nREAL trainer.rollout(): mean {float(r.mean()):.3f}  min {float(r.min()):.3f}  "
          f"max {float(r.max()):.3f}")
    for k in sorted(tally):
        print(f"  {k:18s} {tally[k]:4d}  ({100*tally[k]/len(r):5.1f}%)")
    print("\nEXPECTED from the hand-built rollout: mean ~-1.00, 1/128 at floor")
    print("JOB 43175 REPORTS:                    mean ~-9.50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
