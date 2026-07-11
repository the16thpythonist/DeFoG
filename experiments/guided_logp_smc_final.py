"""
Full FK-SMC (guided proposal) deliverable at K=128 particles: low/med/high logP.

Produces the same outputs as guided_logp_final.py but generated with
FeynmanKacSampler (guided proposal + reward resampling) instead of single-
trajectory GuidedSampler:
  * guided_logp_grid_{low,med,high}.png  -- 5x5 grids of valid molecules
  * guided_logp_distributions.png        -- distributions over dataset backdrop
  * guided_logp_metrics.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem import Crippen

from defog.core import (
    DeFoGModel, ExactGuidance, FeynmanKacSampler,
    MoleculePropertyEnergy, ConditionalSizeDistribution,
)
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import (
    derive_atom_types, build_dataset, save_grid, plot_distributions, BOND_TYPES, LEVELS,
)

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def collect_fk(fk, runs, K, size_dist, target, device, ad, bd):
    mols, logps = [], []
    for _ in range(runs):
        cond = torch.full((K, 1), float(target))
        for s in fk.sample(K, size_dist=size_dist, condition=cond,
                           device=device, show_progress=False):
            mol = pyg_data_to_mol(s, ad, bd)
            if mol is not None and mol_to_smiles(mol) is not None:
                try:
                    logps.append(float(Crippen.MolLogP(mol)))
                    mols.append(mol)
                except Exception:
                    pass
    return mols, np.array(logps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--guidance", default="experiments/_guided_logp_out/guided_logp_amortized.ckpt")
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_guided_logp_smc_final")
    ap.add_argument("--particles", type=int, default=128)
    ap.add_argument("--runs", type=int, default=2, help="independent SMC ensembles per target")
    ap.add_argument("--sample-steps", type=int, default=250)
    ap.add_argument("--eta", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--resample-interval", type=int, default=25)
    ap.add_argument("--ess-frac", type=float, default=None,
                    help="adaptive resampling: only resample when ESS < ess_frac*K (e.g. 0.5). None = always.")
    ap.add_argument("--size-bandwidth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    atom_types = derive_atom_types(df["smiles"])
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    logp_all = df["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    targets = dict(zip(LEVELS, np.percentile(logp_all, [10, 50, 90])))

    graphs, _, _ = build_dataset(df, ae, be)
    conds = torch.tensor([[float(g.prop_val)] for g in graphs])
    sizes = torch.tensor([int(g.x.size(0)) for g in graphs])
    size_dist = ConditionalSizeDistribution(conds, sizes, method="kernel", bandwidth=args.size_bandwidth)

    base = DeFoGModel.load(args.ckpt, device="cpu").to(device).eval()
    h = DeFoGModel.load(args.guidance, device="cpu").to(device).eval()
    domain = MoleculeDomain(ad, bd)
    print(f"[smc-final] K={args.particles} runs={args.runs} beta={args.beta} "
          f"resample_every={args.resample_interval} steps={args.sample_steps} "
          f"targets={ {k: round(v,2) for k,v in targets.items()} }", flush=True)

    metrics, per_target = {}, {}
    for lvl in LEVELS:
        tgt = float(targets[lvl])
        energy_fn = MoleculePropertyEnergy(domain, Crippen.MolLogP, tgt)
        guidance = ExactGuidance(h, prop_mean=prop_mean, prop_std=prop_std,
                                 weight=args.weight).set_target(tgt)
        fk = FeynmanKacSampler(
            base, energy_fn, proposal_transform=guidance.reweight, beta=args.beta,
            resample_interval=args.resample_interval, ess_frac=args.ess_frac,
            eta=args.eta, omega=args.omega,
            sample_steps=args.sample_steps, time_distortion="polydec",
        )
        mols, logps = collect_fk(fk, args.runs, args.particles, size_dist, tgt, device, ad, bd)
        per_target[lvl] = logps
        n = len(logps)
        achieved = float(np.mean(logps)) if n else float("nan")
        med = float(np.median(logps)) if n else float("nan")
        mae = float(np.mean(np.abs(logps - tgt))) if n else float("nan")
        n_atoms = np.array([m.GetNumHeavyAtoms() for m in mols]) if mols else np.array([])
        size_std = float(n_atoms.std()) if n else float("nan")
        n_size = int(len(set(n_atoms.tolist()))) if n else 0
        total = args.runs * args.particles
        metrics[lvl] = {"target": tgt, "n_valid": n, "n_total": total,
                        "validity": n / total, "achieved_mean_logp": achieved,
                        "median_logp": med, "logp_mae": mae,
                        "size_std": size_std, "n_distinct_sizes": n_size}
        print(f"[smc-final] {lvl}: target={tgt:.2f} valid={n}/{total} "
              f"mean={achieved:.2f} median={med:.2f} MAE={mae:.2f} "
              f"size_std={size_std:.1f} distinct_sizes={n_size}", flush=True)
        if mols:
            save_grid(mols, logps, tgt, os.path.join(args.outdir, f"guided_logp_grid_{lvl}.png"))

    plot_distributions(logp_all, per_target, targets,
                       os.path.join(args.outdir, "guided_logp_distributions.png"))
    with open(os.path.join(args.outdir, "guided_logp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[smc-final] DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
