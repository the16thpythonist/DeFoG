"""
Amortized exact-guidance demo on AqSolDB: steer a FROZEN unconditional DeFoG base
toward target logP values with a single target-conditioned guidance network.

Pipeline
--------
1. Load the frozen unconditional AqSolDB base (``--ckpt``).
2. Build training graphs from data/aqsoldb_conditional.csv, attaching each
   molecule's precomputed logP as ``prop_val``.
3. Train ONE AmortizedPropertyGuidanceModule (h conditioned on the target logP;
   targets sampled from the empirical logP distribution) via the Bregman objective.
4. For low/med/high targets (dataset p10/p50/p90), guided-sample with
   GuidedSampler, decode to RDKit molecules, and recompute logP.

Outputs (to --outdir):
  * guided_logp_grid_{low,med,high}.png  -- one 5x5 grid of valid guided molecules.
  * guided_logp_distributions.png        -- generated logP distributions over a grey
                                            dataset-logP backdrop, with target lines.
  * guided_logp_amortized.ckpt           -- the trained guidance network (h only).
  * guided_logp_metrics.json             -- validity + achieved-logP mean/MAE per target.

Run:
  PYTHONPATH=/media/ssd2/Programming/DeFoG PYTHONUNBUFFERED=1 .venv/bin/python \
    experiments/guided_logp_demo.py --ckpt ~/Downloads/aqsoldb_4e-4_best_model.ckpt
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Draw

from defog.core import DeFoGModel, GuidedSampler, AmortizedPropertyGuidanceModule
from defog.domains.molecule import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles

RDLogger.DisableLog("rdApp.*")
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
LEVELS = ["low", "med", "high"]


def derive_atom_types(smiles_list):
    """Atom vocabulary observed in the dataset, most-common-first (matches the
    unconditional training/eval scripts)."""
    counts = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1
    return [sym for sym, _ in counts.most_common()]


def build_dataset(df, atom_encoder, bond_encoder):
    """SMILES -> PyG graphs, attaching precomputed logP as ``prop_val``."""
    graphs, logps = [], []
    skipped = 0
    for smi, lp in zip(df["smiles"], df["logp"]):
        try:
            data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        except Exception:
            skipped += 1
            continue
        if data is None or data.edge_index.numel() == 0:
            skipped += 1
            continue
        data.prop_val = torch.tensor([float(lp)], dtype=torch.float)
        graphs.append(data)
        logps.append(float(lp))
    return graphs, np.array(logps), skipped


@torch.no_grad()
def guided_sample(base, guidance, target, n, chunk, steps, eta, omega, distortion, device):
    """Guided-sample ``n`` molecules for one target; return (mols, logps, n_valid)."""
    guidance.set_target(target)
    sampler = GuidedSampler(
        base, guidance, eta=eta, omega=omega, sample_steps=steps, time_distortion=distortion
    )
    mols, logps = [], []
    remaining = n
    while remaining > 0:
        cur = min(chunk, remaining)
        samples = sampler.sample(cur, device=device, show_progress=False)
        for s in samples:
            mol = pyg_data_to_mol(s, guidance_atom_decoder, guidance_bond_decoder)
            if mol is not None and mol_to_smiles(mol) is not None:
                try:
                    logps.append(float(Crippen.MolLogP(mol)))
                    mols.append(mol)
                except Exception:
                    pass
        remaining -= cur
    return mols, np.array(logps), len(mols)


def save_grid(mols, logps, target, path):
    """5x5 grid of up to 25 valid molecules, legended with their logP."""
    m = mols[:25]
    legends = [f"logP={lp:.2f}" for lp in logps[:25]]
    img = Draw.MolsToGridImage(
        m, molsPerRow=5, subImgSize=(220, 220), legends=legends
    )
    img.save(path)


def plot_distributions(dataset_logp, per_target, targets, path):
    """Grey dataset-logP backdrop + one generated-logP curve per target + target lines."""
    from scipy.stats import gaussian_kde

    lo, hi = -4.0, 9.0
    grid = np.linspace(lo, hi, 400)
    colors = {"low": "#2c7fb8", "med": "#31a354", "high": "#d95f0e"}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    # Dataset backdrop (clipped to the plotting window).
    d = dataset_logp[(dataset_logp >= lo) & (dataset_logp <= hi)]
    ax.hist(d, bins=60, range=(lo, hi), density=True, color="0.8",
            label="AqSolDB dataset", zorder=1)

    for lvl in LEVELS:
        lp = per_target[lvl]
        c = colors[lvl]
        if len(lp) > 5 and np.std(lp) > 1e-3:
            kde = gaussian_kde(lp)
            ax.plot(grid, kde(grid), color=c, lw=2.2,
                    label=f"generated ({lvl}, target={targets[lvl]:.2f})", zorder=3)
        ax.axvline(targets[lvl], color=c, ls="--", lw=1.6, alpha=0.9, zorder=2)
        if len(lp):
            ax.axvline(float(np.mean(lp)), color=c, ls="-", lw=1.0, alpha=0.5, zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("logP (Crippen)")
    ax.set_ylabel("density")
    ax.set_title("Amortized exact guidance on AqSolDB: logP steering\n"
                 "(dashed = target, thin solid = achieved mean)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_generated_json(path, per_target_mols, per_target_logps, targets):
    """Persist the generated molecules per target: SMILES + logP + heavy-atom count."""
    data = {}
    for lvl, mols in per_target_mols.items():
        logps = per_target_logps.get(lvl, [])
        rows = [{"smiles": mol_to_smiles(m), "logp": float(lp), "n_atoms": int(m.GetNumHeavyAtoms())}
                for m, lp in zip(mols, logps)]
        sizes = [r["n_atoms"] for r in rows]
        data[lvl] = {
            "target": float(targets[lvl]), "n": len(rows),
            "logp_mean": float(np.mean([r["logp"] for r in rows])) if rows else None,
            "size_mean": float(np.mean(sizes)) if sizes else None,
            "size_std": float(np.std(sizes)) if sizes else None,
            "molecules": rows,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def plot_size_distributions(per_target_mols, targets, path, proposed=None):
    """Generated-molecule size distributions per target (solid), optionally over the
    conditional size-prior the sampler was given (dashed) so mismatches are visible."""
    colors = {"low": "#2c7fb8", "med": "#31a354", "high": "#d95f0e"}
    all_sizes = [m.GetNumHeavyAtoms() for mm in per_target_mols.values() for m in mm]
    smax = max(all_sizes) if all_sizes else 30
    xs = np.arange(1, smax + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for lvl in LEVELS:
        c = colors.get(lvl, "purple")
        sz = np.array([m.GetNumHeavyAtoms() for m in per_target_mols.get(lvl, [])])
        if len(sz):
            pmf = np.bincount(sz, minlength=smax + 1)[1:smax + 1] / len(sz)
            ax.plot(xs, pmf, marker="o", ms=3, lw=2, color=c,
                    label=f"generated {lvl} (mean {sz.mean():.1f})")
        if proposed is not None and lvl in proposed and len(proposed[lvl]):
            psz = np.asarray(proposed[lvl])
            ppmf = np.bincount(psz, minlength=smax + 1)[1:smax + 1] / len(psz)
            ax.plot(xs, ppmf, ls="--", lw=1.2, color=c, alpha=0.55,
                    label=f"size prior {lvl} (mean {psz.mean():.1f})")
    ax.set_xlabel("graph size (heavy atoms)")
    ax.set_ylabel("probability")
    ax.set_title("Generated molecule sizes (solid) vs conditional size prior (dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_guided_logp_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.7, help="energy sharpness r=exp(-gamma*(logp-c)^2)")
    ap.add_argument("--h-layers", type=int, default=6)
    ap.add_argument("--h-hidden", type=int, default=256)
    ap.add_argument("--num-eval", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--sample-steps", type=int, default=300)
    ap.add_argument("--eta", type=float, default=10.0)
    ap.add_argument("--omega", type=float, default=0.2)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--limit", type=int, default=0, help="subsample dataset for a quick smoke (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device={device} ckpt={args.ckpt}", flush=True)

    # --- data + vocab -------------------------------------------------------
    # Vocabulary + logP stats/targets always come from the FULL dataset so they
    # match the base model's training vocabulary; --limit only subsamples the
    # graphs used for guidance training (for a quick smoke).
    df_full = pd.read_csv(args.data)
    atom_types = derive_atom_types(df_full["smiles"])
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    global guidance_atom_decoder, guidance_bond_decoder
    guidance_atom_decoder, guidance_bond_decoder = atom_decoder, bond_decoder

    logp_all = df_full["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    lo_c, hi_c = np.percentile(logp_all, [1, 99])  # clip tails for target sampling
    prop_values = logp_all[(logp_all >= lo_c) & (logp_all <= hi_c)]
    targets = dict(zip(LEVELS, np.percentile(logp_all, [10, 50, 90])))

    df_build = df_full
    if args.limit:
        df_build = df_full.sample(n=min(args.limit, len(df_full)), random_state=args.seed).reset_index(drop=True)
    graphs, _, skipped = build_dataset(df_build, atom_encoder, bond_encoder)
    print(f"[demo] vocab={len(atom_decoder)} atom types; built {len(graphs)} graphs "
          f"({skipped} skipped); targets {targets}", flush=True)

    # --- base (frozen) + amortized guidance module --------------------------
    base = DeFoGModel.load(args.ckpt, device="cpu")
    assert len(atom_decoder) == base.num_node_classes, (
        f"atom vocab ({len(atom_decoder)}) != base.num_node_classes "
        f"({base.num_node_classes}); vocabulary mismatch would corrupt sampling."
    )
    module = AmortizedPropertyGuidanceModule(
        base, prop_values=prop_values, prop_mean=prop_mean, prop_std=prop_std,
        gamma=args.gamma, prop_attr="prop_val", lr=args.lr,
        n_layers=args.h_layers, hidden_dim=args.h_hidden,
    )
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1, logger=False, enable_checkpointing=False,
        enable_model_summary=False, gradient_clip_val=1.0,
        log_every_n_steps=25,
    )
    print(f"[demo] training amortized guidance: epochs={args.epochs} "
          f"h={args.h_layers}x{args.h_hidden} gamma={args.gamma}", flush=True)
    trainer.fit(module, loader)

    guidance = module.guidance()
    guidance.save(os.path.join(args.outdir, "guided_logp_amortized"))
    base.to(device).eval()
    guidance.h.to(device).eval()

    # --- guided sampling per target + outputs -------------------------------
    metrics, per_target = {}, {}
    for lvl in LEVELS:
        tgt = float(targets[lvl])
        mols, logps, n_valid = guided_sample(
            base, guidance, tgt, args.num_eval, args.chunk, args.sample_steps,
            args.eta, args.omega, args.time_distortion, device,
        )
        per_target[lvl] = logps
        achieved = float(np.mean(logps)) if len(logps) else float("nan")
        mae = float(np.mean(np.abs(logps - tgt))) if len(logps) else float("nan")
        metrics[lvl] = {
            "target": tgt, "n_valid": n_valid, "n_requested": args.num_eval,
            "validity": n_valid / args.num_eval, "achieved_mean_logp": achieved,
            "logp_mae": mae,
        }
        print(f"[demo] {lvl}: target={tgt:.2f} valid={n_valid}/{args.num_eval} "
              f"achieved_mean={achieved:.2f} MAE={mae:.2f}", flush=True)
        if mols:
            save_grid(mols, logps, tgt, os.path.join(args.outdir, f"guided_logp_grid_{lvl}.png"))

    plot_distributions(logp_all, per_target, targets,
                       os.path.join(args.outdir, "guided_logp_distributions.png"))
    with open(os.path.join(args.outdir, "guided_logp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[demo] DONE. outputs in {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
