#!/usr/bin/env python
"""Bisect the -0.860 vs -9.500 gap: size-model device x base-load route.

Direct construction gives a healthy rollout (-0.860, 98.4% scored). The experiment's own
path gives -9.500 (~94% at the invalid floor) with the same seed, adapter, base and every
documented rollout parameter. Two construction details differ, and both are the kind of
thing that changes behaviour without raising:

  size model: LearnedSizeDistribution.load(path)              [mine, CPU]
              LearnedSizeDistribution.load(path, device=cuda) [experiment]
  base:       DeFoGModel.load(path, device=cuda).eval()       [mine]
              DeFoGModel.load(path, device="cpu").to(cuda).eval()  [experiment]

A condition-aware size distribution silently falls back to the marginal when the condition
does not reach it (rl.py:735 documents exactly this costing a run), so a device mismatch
between model and condition is a live candidate for drawing the wrong node counts.
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
    head = PropertyHead.load("ckpts/heads/qed_head.ckpt", device=dev)

    def run(tag, size_on_cuda, base_via_cpu):
        base = (DeFoGModel.load("ckpts/zinc_rl2_seed42/best_model", device="cpu").to(dev).eval()
                if base_via_cpu else
                DeFoGModel.load("ckpts/zinc_rl2_seed42/best_model", device=dev).eval())
        adapter = AdaLNAdapter.load("ckpts/qed_adapter_pre_rl.ckpt", device=dev)
        sd = (LearnedSizeDistribution.load("ckpts/heads/qed_head_size.ckpt", device=dev)
              if size_on_cuda else
              LearnedSizeDistribution.load("ckpts/heads/qed_head_size.ckpt"))
        reward = HeadPropertyMatchReward(head, ae, be, adec, bdec, dev, scale=0.1339,
                                         invalid_reward=-10.0, disconnect_reward=-4.0,
                                         prop_clamp=3.0)
        trainer = AdapterGDPOTrainer(
            base, adapter, reward, kl_coef=0.05, lr=1e-4, ema_decay=0.99,
            rollout_size=128, sample_steps=250, eta=1.0, omega=0.0,
            time_distortion="polydec",
            condition_sampler=make_condition_sampler(0.4, 0.9, 128, 8, seed=42),
            subsample_steps=16, minibatch_size=16, crn=True, size_dist=sd,
            seed=42, device=dev)
        with torch.no_grad():
            buf = trainer.rollout()
        r = buf.reward.float()
        n = buf.node_mask.sum(-1).float()
        t = collections.Counter("floor" if v <= -9.99 else
                                "disc" if abs(v + 4) < 1e-6 else "scored" for v in r.tolist())
        print(f"  {tag:34s} reward {float(r.mean()):+8.3f}   floor {t['floor']:3d}/128   "
              f"scored {t['scored']:3d}   nodes mean {float(n.mean()):.1f} "
              f"min {int(n.min())} max {int(n.max())}")

    print("size-model device x base-load route, one rollout each (seed 42):\n")
    run("A size=cpu   base=direct  [mine]", False, False)
    run("B size=cuda  base=direct", True, False)
    run("C size=cpu   base=cpu->cuda", False, True)
    run("D size=cuda  base=cpu->cuda [experiment]", True, True)
    print("\n  mine reproduced -0.860 ; the experiment reports -9.500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
