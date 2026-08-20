#!/usr/bin/env python
"""Where do rollout molecules die in the tiered head reward?

Job 43175's reward sits at ~-9.5 with the floor at -10, i.e. ~94% of rollouts land in the
invalid tier and it is not climbing out. That is not explained by:
  - the head being broken: 64/64 REAL molecules score normally through the same reward object
  - bad generation: the rdkit-reward run used the SAME adapter, base and rollout config and
    averaged +3.0 out of ~4 under the weighted shape, whose invalid tier is 0.0
So the two rewards disagree about the same rollouts, and this counts exactly where.

Reproduces AdapterGDPOTrainer.rollout (rl.py:724) rather than approximating it, then walks
the reward's decision tree branch by branch instead of reading the aggregate.
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

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    from defog.core import (AdaLNAdapter, DeFoGModel, HeadPropertyMatchReward,
                            LearnedSizeDistribution, PropertyHead, head_predict_batch)
    from defog.core.adapter import AdapterComposition, ConditionBranch
    from defog.core.data import dense_to_pyg
    from defog.core.rl import RolloutSampler, make_condition_sampler
    from defog.domains.molecule import build_encoders, mol_to_smiles, pyg_data_to_mol

    dev = "cuda"
    atoms, bonds, kek, _ = rl_exp._vocabulary("e1_kekulized")
    ae, adec, be, bdec = build_encoders(atoms, bonds)

    base = DeFoGModel.load("ckpts/zinc_rl2_seed42/best_model", device=dev).eval()
    adapter = AdaLNAdapter.load("ckpts/qed_adapter_pre_rl.ckpt", device=dev).eval()
    head = PropertyHead.load("ckpts/heads/qed_head.ckpt", device=dev)
    size_dist = LearnedSizeDistribution.load("ckpts/heads/qed_head_size.ckpt")

    K = 128
    cond_sampler = make_condition_sampler(0.4, 0.9, K, 8, seed=42)
    cond, groups = cond_sampler()
    cond = cond.to(dev).float()

    comp = AdapterComposition([ConditionBranch(adapter, cond, 1.0)], base=base, mode="product")
    sampler = RolloutSampler(base, subsample_idx=None, eta=1.0, omega=0.0,
                             sample_steps=250, time_distortion="polydec",
                             group_ids=groups, crn=True, guidance_scale=1.0)
    sampler.composition = comp
    with torch.no_grad():
        sampler.sample(K, size_dist=size_dist, condition=cond, device=dev, show_progress=False)
    X1, E1 = sampler.endpoint
    node_mask = sampler.end_node_mask
    Xr, Er, _ = base.limit_dist.ignore_virtual_classes(X1.clone(), E1.clone())

    # ---- walk the reward's decision tree, counting each branch -----------------------
    n = node_mask.sum(-1)
    datas = dense_to_pyg(Xr, Er, None, node_mask, n)
    tally = collections.Counter()
    conn_mols, conn_idx = [], []
    for i, d in enumerate(datas):
        mol = pyg_data_to_mol(d, adec, bdec)
        if mol is None:
            tally["1. pyg_data_to_mol -> None"] += 1
            continue
        smi = mol_to_smiles(mol)
        if smi is None:
            tally["2. mol_to_smiles -> None"] += 1
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            tally["3. MolFromSmiles -> None"] += 1
            continue
        if "." in smi:
            tally["4. disconnected (-4)"] += 1
            continue
        conn_mols.append(m)
        conn_idx.append(i)
    if conn_mols:
        preds = head_predict_batch(conn_mols, head, ae, be, dev)
        tally["5. head re-encode -> None"] += sum(p is None for p in preds)
        tally["6. SCORED"] += sum(p is not None for p in preds)

    print(f"rollout of {K} at eta=1.0, 250 steps, pre-RL adapter\n")
    for k in sorted(tally):
        print(f"  {k:34s} {tally[k]:4d}  ({100*tally[k]/K:5.1f}%)")

    reward = HeadPropertyMatchReward(head, ae, be, adec, bdec, dev, scale=0.1328)
    r = reward(Xr, Er, node_mask, cond)
    print(f"\n  tiered head reward: mean {float(r.mean()):.3f}  "
          f"min {float(r.min()):.3f}  max {float(r.max()):.3f}")
    print(f"  at floor (-10): {int((r <= -9.99).sum())}/{K}")

    if hasattr(rl_exp, "WeightedSanityPropertyReward"):
        from rdkit.Chem import QED
        wr = rl_exp.WeightedSanityPropertyReward(
            adec, bdec, lambda m: float(QED.qed(m)), scale=0.1328,
            w_prop=3.0, w_sanity=1.0, prop_span=3.0)
        w = wr(Xr, Er, node_mask, cond)
        print(f"  weighted rdkit reward on the SAME rollouts: mean {float(w.mean()):.3f}  "
              f"min {float(w.min()):.3f}  max {float(w.max()):.3f}")
        print(f"  at its floor (0.0): {int((w <= 1e-6).sum())}/{K}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
