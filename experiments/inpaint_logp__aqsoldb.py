"""
FK-SMC logP steering with a FIXED inpainting core (benzene).

Freezes a benzene ring (c1ccccc1) via SubgraphConstraint and grows n_free new
atoms around it, while FK-steering the whole molecule's logP toward the dataset
p10 (low) and p90 (high). Uses the best FK config (beta=4.0, warmup=0.8) with the
constraint wired into FeynmanKacSampler.

Note: benzene alone has logP ~1.7, so it acts as a logP floor -- the low case is
pulled down but cannot go strongly negative while a benzene ring is mandatory.

Outputs (to --outdir):
  inpaint_logp_distributions.png       -- low/high logP over dataset backdrop
  grid_{low,high}.png                  -- 5x5 grids, benzene core highlighted
  inpaint_logp_metrics.json, generated.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Draw

from defog.core import (
    DeFoGModel, ExactGuidance, FeynmanKacSampler,
    MoleculePropertyEnergy, SubgraphConstraint,
)
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import derive_atom_types, BOND_TYPES

RDLogger.DisableLog("rdApp.*")
CASES = {"low": 10, "high": 90}          # percentile per case
COLORS = {"low": "#2c7fb8", "high": "#d95f0e"}


@torch.no_grad()
def collect(sampler, runs, K, core_mol, device, ad, bd):
    mols, lps = [], []
    n_core = 0
    for _ in range(runs):
        seen = set()
        for s in sampler.sample(K, device=device, show_progress=False):
            mol = pyg_data_to_mol(s, ad, bd)
            if mol is None:
                continue
            smi = mol_to_smiles(mol)
            if smi is None or smi in seen:
                continue
            seen.add(smi)
            try:
                lps.append(float(Crippen.MolLogP(mol)))
                mols.append(mol)
                if mol.HasSubstructMatch(core_mol):
                    n_core += 1
            except Exception:
                pass
    return mols, np.array(lps), n_core


def grid_with_core(mols, lps, core_mol, path):
    m = mols[:25]
    highlights = [list(mol.GetSubstructMatch(core_mol)) for mol in m]
    img = Draw.MolsToGridImage(m, molsPerRow=5, subImgSize=(240, 240),
                               legends=[f"logP={v:.2f}" for v in lps[:25]],
                               highlightAtomLists=highlights)
    img.save(path)


def dist_plot(per_case, targets, dataset_lp, path):
    from scipy.stats import gaussian_kde
    lo, hi = -4.0, 9.0
    grid = np.linspace(lo, hi, 300)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    d = dataset_lp[(dataset_lp >= lo) & (dataset_lp <= hi)]
    ax.hist(d, bins=60, range=(lo, hi), density=True, color="0.85", label="AqSolDB", zorder=1)
    for case, lp in per_case.items():
        c = COLORS[case]
        if len(lp) > 5 and np.std(lp) > 1e-3:
            ax.plot(grid, gaussian_kde(lp)(grid), color=c, lw=2.2,
                    label=f"{case} (target={targets[case]:.2f}, mean={lp.mean():.2f})", zorder=3)
        ax.axvline(targets[case], color=c, ls="--", lw=1.6)
    ax.axvline(Crippen.MolLogP(Chem.MolFromSmiles("c1ccccc1")), color="k", ls=":", lw=1.2,
               label="benzene core logP")
    ax.set_xlim(lo, hi); ax.set_xlabel("logP (Crippen)"); ax.set_ylabel("density")
    ax.set_title("logP steering with a fixed benzene core (dashed = target)")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--logp-guidance", default="experiments/_guided_logp_out/guided_logp_amortized.ckpt")
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_inpaint_logp")
    ap.add_argument("--core", default="c1ccccc1", help="fixed inpainting core (benzene)")
    ap.add_argument("--n-free", type=int, default=10)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--warmup-frac", type=float, default=0.8)
    ap.add_argument("--particles", type=int, default=128)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=200)
    ap.add_argument("--eta", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--resample-interval", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    atom_types = derive_atom_types(df["smiles"])
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    lp_all = df["logp"].values.astype(float)
    lp_mean, lp_std = float(lp_all.mean()), float(lp_all.std())
    targets = {c: float(np.percentile(lp_all, p)) for c, p in CASES.items()}

    domain = MoleculeDomain(ad, bd)
    core_mol = Chem.MolFromSmiles(args.core)
    Xc, Ec = domain.encode(args.core)
    constraint = SubgraphConstraint(Xc, Ec)
    k = Xc.shape[0]
    print(f"[inpaint] core={args.core!r} k={k} n_free={args.n_free} total={k + args.n_free} "
          f"targets={targets} benzene_logP={Crippen.MolLogP(core_mol):.2f}", flush=True)

    base = DeFoGModel.load(args.ckpt, device="cpu").to(device).eval()
    h = DeFoGModel.load(args.logp_guidance, device="cpu").to(device).eval()
    g = ExactGuidance(h, prop_mean=lp_mean, prop_std=lp_std, weight=args.weight)

    per_case, metrics, generated = {}, {}, {}
    for case in CASES:
        t = targets[case]
        g.set_target(t)
        energy = MoleculePropertyEnergy(domain, Crippen.MolLogP, t)
        sampler = FeynmanKacSampler(
            base, energy, proposal_transform=g.reweight, constraint=constraint,
            n_free=args.n_free, beta=args.beta, warmup_frac=args.warmup_frac,
            resample_interval=args.resample_interval, eta=args.eta, omega=args.omega,
            sample_steps=args.sample_steps, time_distortion="polydec",
        )
        mols, lps, n_core = collect(sampler, args.runs, args.particles, core_mol, device, ad, bd)
        per_case[case] = lps
        n = len(lps)
        metrics[case] = {
            "target": t, "n_valid": n, "core_preserved": n_core,
            "core_rate": (n_core / n) if n else 0.0,
            "logp_mean": float(np.mean(lps)) if n else None,
            "logp_mae": float(np.mean(np.abs(lps - t))) if n else None,
        }
        generated[case] = [{"smiles": mol_to_smiles(m), "logp": float(v)} for m, v in zip(mols, lps)]
        print(f"[inpaint] {case}: target={t:.2f} n={n} logP mean={metrics[case]['logp_mean']:.2f} "
              f"MAE={metrics[case]['logp_mae']:.2f} core_preserved={n_core}/{n} "
              f"({metrics[case]['core_rate']:.0%})", flush=True)
        if mols:
            grid_with_core(mols, lps, core_mol, os.path.join(args.outdir, f"grid_{case}.png"))

    dist_plot(per_case, targets, lp_all, os.path.join(args.outdir, "inpaint_logp_distributions.png"))
    with open(os.path.join(args.outdir, "inpaint_logp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.outdir, "generated.json"), "w") as f:
        json.dump(generated, f, indent=2)
    print(f"[inpaint] DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
