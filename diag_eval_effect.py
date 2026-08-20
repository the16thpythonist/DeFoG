#!/usr/bin/env python
"""Does running the pre-RL eval before the first rollout collapse the reward?

Everything else is now excluded. Constructing the trainer directly and calling rollout()
gives -0.86 to -1.23 across every size-model/base-load combination. The experiment, with
the same seed and arguments, reports -9.500. The one structural difference left is that the
experiment runs steer_eval() BEFORE constructing the trainer, and my diagnostics never did.

steer_eval saves and restores the CPU rng, but torch.manual_seed also seeds CUDA and
torch.set_rng_state restores only the CPU generator -- so the CUDA stream is left wherever
the eval put it. This measures whether that matters, by taking the same rollout twice with
the eval in between.
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
    from rdkit.Chem import QED
    RDLogger.DisableLog("rdApp.*")

    from defog.core import (AdaLNAdapter, AdapterGDPOTrainer, DeFoGModel,
                            HeadPropertyMatchReward, LearnedSizeDistribution, PropertyHead)
    from defog.core.rl import make_condition_sampler
    from defog.domains.molecule import build_encoders

    dev = "cuda"
    atoms, bonds, kek, _ = rl_exp._vocabulary("e1_kekulized")
    ae, adec, be, bdec = build_encoders(atoms, bonds)

    base = DeFoGModel.load("ckpts/zinc_rl2_seed42/best_model", device="cpu").to(dev).eval()
    adapter = AdaLNAdapter.load("ckpts/qed_adapter_pre_rl.ckpt", device=dev)
    head = PropertyHead.load("ckpts/heads/qed_head.ckpt", device=dev)
    sd = LearnedSizeDistribution.load("ckpts/heads/qed_head_size.ckpt", device=dev)
    reward = HeadPropertyMatchReward(head, ae, be, adec, bdec, dev, scale=0.1339,
                                     invalid_reward=-10.0, disconnect_reward=-4.0,
                                     prop_clamp=3.0)

    def new_trainer():
        return AdapterGDPOTrainer(
            base, adapter, reward, kl_coef=0.05, lr=1e-4, ema_decay=0.99,
            rollout_size=128, sample_steps=250, eta=1.0, omega=0.0,
            time_distortion="polydec",
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
        print(f"  {tag:28s} reward {float(r.mean()):+8.3f}  floor {t['floor']:3d}/128  "
              f"scored {t['scored']:3d}  nodes {float(n.mean()):.1f} "
              f"[{int(n.min())}-{int(n.max())}]")

    print("rollout, then the pre-RL eval, then the same rollout again:\n")
    report("1. before any eval", new_trainer())

    head_fn = lambda mols: rl_exp.head_predict_batch(mols, head, ae, be, dev) \
        if hasattr(rl_exp, "head_predict_batch") else None
    from defog.core import head_predict_batch as _hpb
    head_fn = lambda mols: _hpb(mols, head, ae, be, dev)

    print("\n  ...running steer_eval (small) ...")
    rl_exp.steer_eval(base, adapter, adec, bdec, lambda m: float(QED.qed(m)),
                      {"mid": 0.7776876259891845}, [1.0], 500, 25.0, 0.0, "polydec",
                      16, 16, 16, dev, seed=1234, head_fn=head_fn)

    print()
    report("2. after the eval", new_trainer())
    print("\n  experiment reports -9.500 at iter 0 (eval runs before the trainer exists)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
