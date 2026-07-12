"""
Joint 2-property FK-SMC steering on AqSolDB: logP x TPSA, all four low/high corners.

Uses JointGuidanceSampler with a composite (product-of-experts) proposal over the
two trained amortized guidance nets (h_logP, h_TPSA) + a joint MultiPropertyEnergy
FK reward, at the best FK config (beta=4.0, warmup=0.8). Graph size is drawn from a
2D-conditional size distribution P(n | logP, TPSA) (z-scored conditions).

Targets: low = dataset p10, high = p90 for each property. Note logP and TPSA are
anti-correlated, so (high logP, high TPSA) is an antagonistic corner.

Outputs (to --outdir):
  joint_scatter.png          -- 2D logP-vs-TPSA scatter, 4 corners + targets + dataset
  logp_marginals.png / tpsa_marginals.png
  grid_{lolo,lohi,hilo,hihi}.png
  joint_metrics.json, generated.json
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
from rdkit.Chem import Crippen, Descriptors, Draw

from defog.core import (
    DeFoGModel, ExactGuidance, JointGuidanceSampler,
    MultiPropertyEnergy, ConditionalSizeDistribution,
)
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import derive_atom_types, BOND_TYPES

RDLogger.DisableLog("rdApp.*")

# (label, logP level, TPSA level)
CORNERS = [("lolo", "low", "low"), ("lohi", "low", "high"),
           ("hilo", "high", "low"), ("hihi", "high", "high")]
COLORS = {"lolo": "#2c7fb8", "lohi": "#31a354", "hilo": "#d95f0e", "hihi": "#d7191c"}


def build_dataset(df, ae, be):
    """Aligned per-molecule (graph, logP, TPSA, size) for molecules that encode."""
    graphs, logp, tpsa, size = [], [], [], []
    for smi, lp in zip(df["smiles"], df["logp"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            d = smiles_to_pyg_data(smi, ae, be)
        except Exception:
            d = None
        if d is None or d.edge_index.numel() == 0:
            continue
        try:
            tp = float(Descriptors.TPSA(mol))
        except Exception:
            continue
        graphs.append(d); logp.append(float(lp)); tpsa.append(tp); size.append(int(d.x.size(0)))
    return graphs, np.array(logp), np.array(tpsa), np.array(size)


@torch.no_grad()
def collect(sampler, runs, K, size_dist, cond2d, device, ad, bd):
    mols, lps, tps = [], [], []
    for _ in range(runs):
        seen = set()
        for s in sampler.sample(K, size_dist=size_dist, condition=cond2d, device=device, show_progress=False):
            mol = pyg_data_to_mol(s, ad, bd)
            if mol is None:
                continue
            smi = mol_to_smiles(mol)
            if smi is None or smi in seen:
                continue
            seen.add(smi)
            try:
                lps.append(float(Crippen.MolLogP(mol))); tps.append(float(Descriptors.TPSA(mol))); mols.append(mol)
            except Exception:
                pass
    return mols, np.array(lps), np.array(tps)


def scatter_plot(per_corner, targets, dataset_lp, dataset_tp, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(dataset_lp, dataset_tp, s=4, c="0.8", alpha=0.4, label="AqSolDB", zorder=1)
    for lbl, lvl_lp, lvl_tp in CORNERS:
        lp, tp = per_corner[lbl]
        c = COLORS[lbl]
        if len(lp):
            ax.scatter(lp, tp, s=10, c=c, alpha=0.6,
                       label=f"{lvl_lp} logP / {lvl_tp} TPSA (n={len(lp)})", zorder=3)
        tlp, ttp = targets["logp"][lvl_lp], targets["tpsa"][lvl_tp]
        ax.plot(tlp, ttp, marker="X", ms=16, c=c, mec="k", mew=1.5, zorder=5)
    ax.set_xlabel("logP (Crippen)"); ax.set_ylabel("TPSA")
    ax.set_title("Joint logP x TPSA steering (X = target corner)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def marginal_plot(per_corner, targets, dataset_vals, prop, idx, path):
    from scipy.stats import gaussian_kde
    lo, hi = np.percentile(dataset_vals, [0.5, 99.5])
    grid = np.linspace(lo, hi, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dataset_vals, bins=50, range=(lo, hi), density=True, color="0.85", label="dataset", zorder=1)
    for lbl, lvl_lp, lvl_tp in CORNERS:
        v = per_corner[lbl][idx]; c = COLORS[lbl]
        lvl = lvl_lp if prop == "logp" else lvl_tp
        if len(v) > 5 and np.std(v) > 1e-6:
            ax.plot(grid, gaussian_kde(v)(grid), color=c, lw=2, label=f"{lbl}", zorder=3)
        ax.axvline(targets[prop][lvl], color=c, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel(prop.upper()); ax.set_ylabel("density")
    ax.set_title(f"{prop.upper()} marginal per corner (dashed = target)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def grid_plot(mols, lps, tps, path):
    m = mols[:25]
    legends = [f"logP={a:.1f} TPSA={b:.0f}" for a, b in zip(lps[:25], tps[:25])]
    img = Draw.MolsToGridImage(m, molsPerRow=5, subImgSize=(230, 230), legends=legends)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--logp-guidance", default="experiments/_guided_logp_out/guided_logp_amortized.ckpt")
    ap.add_argument("--tpsa-guidance", default="experiments/_tpsa_guidance/tpsa_guidance.ckpt")
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_joint_logp_tpsa")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--warmup-frac", type=float, default=0.8)
    ap.add_argument("--particles", type=int, default=128)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=200)
    ap.add_argument("--eta", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--resample-interval", type=int, default=15)
    ap.add_argument("--mode", default="product", choices=["product", "mean", "none"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    atom_types = derive_atom_types(df["smiles"])
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    graphs, lp_all, tp_all, sz_all = build_dataset(df, ae, be)
    lp_mean, lp_std = float(lp_all.mean()), float(lp_all.std())
    tp_mean, tp_std = float(tp_all.mean()), float(tp_all.std())
    targets = {
        "logp": {"low": float(np.percentile(lp_all, 10)), "high": float(np.percentile(lp_all, 90))},
        "tpsa": {"low": float(np.percentile(tp_all, 10)), "high": float(np.percentile(tp_all, 90))},
    }
    print(f"[joint] logP std={lp_std:.2f} p10/90={targets['logp']} | "
          f"TPSA std={tp_std:.1f} p10/90={targets['tpsa']}", flush=True)

    # 2D conditional size prior P(n | logP, TPSA), z-scored so both properties count.
    conds = np.stack([(lp_all - lp_mean) / lp_std, (tp_all - tp_mean) / tp_std], axis=1)
    size_dist = ConditionalSizeDistribution(torch.tensor(conds, dtype=torch.float),
                                            torch.tensor(sz_all), method="kernel")

    base = DeFoGModel.load(args.ckpt, device="cpu").to(device).eval()
    h_lp = DeFoGModel.load(args.logp_guidance, device="cpu").to(device).eval()
    h_tp = DeFoGModel.load(args.tpsa_guidance, device="cpu").to(device).eval()
    domain = MoleculeDomain(ad, bd)
    g_lp = ExactGuidance(h_lp, prop_mean=lp_mean, prop_std=lp_std, weight=args.weight)
    g_tp = ExactGuidance(h_tp, prop_mean=tp_mean, prop_std=tp_std, weight=args.weight)

    per_corner, metrics, generated = {}, {}, {}
    for lbl, lvl_lp, lvl_tp in CORNERS:
        t_lp = targets["logp"][lvl_lp]; t_tp = targets["tpsa"][lvl_tp]
        g_lp.set_target(t_lp); g_tp.set_target(t_tp)
        energy = MultiPropertyEnergy(domain, [
            (Crippen.MolLogP, t_lp, lp_std, 1.0),
            (Descriptors.TPSA, t_tp, tp_std, 1.0),
        ])
        sampler = JointGuidanceSampler(
            base, [g_lp, g_tp], energy, mode=args.mode, beta=args.beta,
            warmup_frac=args.warmup_frac, resample_interval=args.resample_interval,
            eta=args.eta, omega=args.omega, sample_steps=args.sample_steps, time_distortion="polydec",
        )
        cond2d = torch.tensor([[(t_lp - lp_mean) / lp_std, (t_tp - tp_mean) / tp_std]],
                              dtype=torch.float).repeat(args.particles, 1)
        mols, lps, tps = collect(sampler, args.runs, args.particles, size_dist, cond2d, device, ad, bd)
        per_corner[lbl] = (lps, tps)
        n = len(lps)
        metrics[lbl] = {
            "target_logp": t_lp, "target_tpsa": t_tp, "n": n,
            "logp_mean": float(np.mean(lps)) if n else None, "logp_mae": float(np.mean(np.abs(lps - t_lp))) if n else None,
            "tpsa_mean": float(np.mean(tps)) if n else None, "tpsa_mae": float(np.mean(np.abs(tps - t_tp))) if n else None,
        }
        generated[lbl] = [{"smiles": mol_to_smiles(m), "logp": float(a), "tpsa": float(b)}
                          for m, a, b in zip(mols, lps, tps)]
        print(f"[joint] {lbl}: target(logP={t_lp:.2f},TPSA={t_tp:.1f}) n={n} "
              f"logP mean={metrics[lbl]['logp_mean']:.2f} TPSA mean={metrics[lbl]['tpsa_mean']:.1f}", flush=True)
        if mols:
            grid_plot(mols, lps, tps, os.path.join(args.outdir, f"grid_{lbl}.png"))

    scatter_plot(per_corner, targets, lp_all, tp_all, os.path.join(args.outdir, "joint_scatter.png"))
    marginal_plot(per_corner, targets, lp_all, "logp", 0, os.path.join(args.outdir, "logp_marginals.png"))
    marginal_plot(per_corner, targets, tp_all, "tpsa", 1, os.path.join(args.outdir, "tpsa_marginals.png"))
    with open(os.path.join(args.outdir, "joint_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.outdir, "generated.json"), "w") as f:
        json.dump(generated, f, indent=2)
    print(f"[joint] DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
