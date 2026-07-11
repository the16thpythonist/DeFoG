"""
Final AqSolDB logP-guidance generation: reuse the ALREADY-TRAINED amortized
guidance network, and (crucially) draw graph SIZE from a target-conditioned size
distribution -- logP is strongly size-dependent, so steering atom/bond types at a
target-independent size caps the reachable logP (esp. the high end).

Pipeline (no retraining):
  * load frozen base + trained guidance h (guided_logp_amortized.ckpt)
  * build ConditionalSizeDistribution from training (logP -> n_nodes)
  * per target (p10/p50/p90): draw sizes conditioned on the target, guided-sample,
    decode, recompute logP
Outputs mirror the earlier demo (3x 5x5 grids + distribution figure + metrics).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from defog.core import DeFoGModel, GuidedSampler, ExactGuidance, ConditionalSizeDistribution
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from rdkit import RDLogger
from rdkit.Chem import Crippen

# reuse the demo's data + plotting helpers
from experiments.guided_logp_demo import (
    derive_atom_types, build_dataset, save_grid, plot_distributions, BOND_TYPES, LEVELS,
)

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def guided_sample(base, guidance, target, size_dist, n, chunk, steps, eta, omega,
                  distortion, device, atom_decoder, bond_decoder, cond_size):
    guidance.set_target(target)
    sampler = GuidedSampler(base, guidance, eta=eta, omega=omega,
                            sample_steps=steps, time_distortion=distortion)
    mols, logps = [], []
    remaining = n
    while remaining > 0:
        cur = min(chunk, remaining)
        kw = {}
        if cond_size:
            kw["size_dist"] = size_dist
            kw["condition"] = torch.full((cur, 1), float(target))  # raw logP -> size only
        for s in sampler.sample(cur, device=device, show_progress=False, **kw):
            mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
            if mol is not None and mol_to_smiles(mol) is not None:
                try:
                    logps.append(float(Crippen.MolLogP(mol)))
                    mols.append(mol)
                except Exception:
                    pass
        remaining -= cur
    return mols, np.array(logps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--guidance", default="experiments/_guided_logp_out/guided_logp_amortized.ckpt")
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_guided_logp_final")
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--eta", type=float, default=5.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--sample-steps", type=int, default=350)
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--size-method", default="kernel", choices=["kernel", "regression"])
    ap.add_argument("--size-bandwidth", default="median",
                    help="kernel bandwidth in logP units ('median' auto, or a float like 0.5 for tighter size conditioning)")
    ap.add_argument("--no-cond-size", action="store_true", help="disable conditional size (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_size = not args.no_cond_size

    df = pd.read_csv(args.data)
    atom_types = derive_atom_types(df["smiles"])
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    logp_all = df["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    targets = dict(zip(LEVELS, np.percentile(logp_all, [10, 50, 90])))

    # (logP -> n_nodes) size distribution from the dataset graphs.
    graphs, _, _ = build_dataset(df, atom_encoder, bond_encoder)
    conds = torch.tensor([[float(g.prop_val)] for g in graphs])
    sizes = torch.tensor([int(g.x.size(0)) for g in graphs])
    bw = args.size_bandwidth
    bw = float(bw) if bw not in ("median",) else bw
    size_dist = ConditionalSizeDistribution(conds, sizes, method=args.size_method, bandwidth=bw)
    print(f"[final] device={device} cond_size={cond_size} method={args.size_method} "
          f"w={args.weight} eta={args.eta} omega={args.omega} targets="
          f"{ {k: round(v,2) for k,v in targets.items()} }", flush=True)
    # illustrate the size->target link the fix exploits:
    for lvl in LEVELS:
        n = size_dist.sample(2000, condition=torch.full((2000, 1), float(targets[lvl])))
        print(f"[final] target {lvl}={targets[lvl]:.2f} -> mean sampled size {n.float().mean():.1f}", flush=True)

    base = DeFoGModel.load(args.ckpt, device="cpu").to(device).eval()
    h = DeFoGModel.load(args.guidance, device="cpu").to(device).eval()
    guidance = ExactGuidance(h, prop_mean=prop_mean, prop_std=prop_std, weight=args.weight)

    metrics, per_target = {}, {}
    for lvl in LEVELS:
        tgt = float(targets[lvl])
        mols, logps = guided_sample(
            base, guidance, tgt, size_dist, args.num_eval, args.chunk,
            args.sample_steps, args.eta, args.omega, args.time_distortion,
            device, atom_decoder, bond_decoder, cond_size,
        )
        per_target[lvl] = logps
        achieved = float(np.mean(logps)) if len(logps) else float("nan")
        mae = float(np.mean(np.abs(logps - tgt))) if len(logps) else float("nan")
        metrics[lvl] = {"target": tgt, "n_valid": len(logps), "validity": len(logps) / args.num_eval,
                        "achieved_mean_logp": achieved, "logp_mae": mae}
        print(f"[final] {lvl}: target={tgt:.2f} valid={len(logps)}/{args.num_eval} "
              f"achieved_mean={achieved:.2f} MAE={mae:.2f}", flush=True)
        if mols:
            save_grid(mols, logps, tgt, os.path.join(args.outdir, f"guided_logp_grid_{lvl}.png"))

    plot_distributions(logp_all, per_target, targets,
                       os.path.join(args.outdir, "guided_logp_distributions.png"))
    with open(os.path.join(args.outdir, "guided_logp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[final] DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
