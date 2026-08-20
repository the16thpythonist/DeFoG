#!/usr/bin/env python
"""Is the -9.500 an artifact of the global seed interacting with CRN?

Everything else is excluded: same adapter, base, size model, reward, rollout parameters,
step()-vs-rollout(), and the pre-RL eval. The last unreplicated difference is that the
experiment calls pytorch_lightning.seed_everything(SEED) at startup and my diagnostics
never did.

That matters because crn=True shares the start noise AND the size draw within each of the
8 target groups, so a single bad shared draw takes all 16 members of that group with it.
Failures would then cluster in multiples of 16 rather than scattering -- which is testable,
so the per-group tally is printed rather than just the mean.
"""
from __future__ import annotations

import collections
import importlib.util
import sys

import torch

sys.path.insert(0, ".")


def build(dev="cuda"):
    from defog.core import (AdaLNAdapter, AdapterGDPOTrainer, DeFoGModel,
                            HeadPropertyMatchReward, LearnedSizeDistribution, PropertyHead)
    from defog.core.rl import make_condition_sampler
    from defog.domains.molecule import build_encoders
    rl_exp = sys.modules["rl_exp"]
    atoms, bonds, _, _ = rl_exp._vocabulary("e1_kekulized")
    ae, adec, be, bdec = build_encoders(atoms, bonds)
    base = DeFoGModel.load("ckpts/zinc_rl2_seed42/best_model", device="cpu").to(dev).eval()
    adapter = AdaLNAdapter.load("ckpts/qed_adapter_pre_rl.ckpt", device=dev)
    head = PropertyHead.load("ckpts/heads/qed_head.ckpt", device=dev)
    sd = LearnedSizeDistribution.load("ckpts/heads/qed_head_size.ckpt", device=dev)
    reward = HeadPropertyMatchReward(head, ae, be, adec, bdec, dev, scale=0.1339,
                                     invalid_reward=-10.0, disconnect_reward=-4.0,
                                     prop_clamp=3.0)
    return AdapterGDPOTrainer(
        base, adapter, reward, kl_coef=0.05, lr=1e-4, ema_decay=0.99,
        rollout_size=128, sample_steps=250, eta=1.0, omega=0.0, time_distortion="polydec",
        condition_sampler=make_condition_sampler(0.4, 0.9, 128, 8, seed=42),
        subsample_steps=16, minibatch_size=16, crn=True, size_dist=sd,
        seed=42, device=dev)


def report(tag, tr):
    with torch.no_grad():
        buf = tr.rollout()
    r = buf.reward.float()
    n = buf.node_mask.sum(-1).float()
    t = collections.Counter("floor" if v <= -9.99 else
                            "disc" if abs(v + 4) < 1e-6 else "scored" for v in r.tolist())
    print(f"  {tag:34s} reward {float(r.mean()):+8.3f}  floor {t['floor']:3d}/128  "
          f"scored {t['scored']:3d}  nodes {float(n.mean()):.1f} [{int(n.min())}-{int(n.max())}]")
    # CRN shares noise+size within a group of 16; clustered failure shows up here.
    per = [int(sum(1 for v in r[g * 16:(g + 1) * 16].tolist() if v <= -9.99)) for g in range(8)]
    sizes = [float(n[g * 16:(g + 1) * 16].mean()) for g in range(8)]
    print(f"       floor per group of 16: {per}")
    print(f"       mean nodes per group:  {[round(s, 1) for s in sizes]}")


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "rl_exp", "experiments/adapter_rl_finetune__zinc.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["rl_exp"] = m
    spec.loader.exec_module(m)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    print("without seed_everything (what my diagnostics did):\n")
    report("A no global seed", build())

    print("\nwith seed_everything(42, workers=True) (what the experiment does):\n")
    from pytorch_lightning import seed_everything
    seed_everything(42, workers=True)
    report("B seed_everything(42)", build())

    print("\n  experiment reports -9.500 (adv_std 1.38) at iter 0, seed 42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
