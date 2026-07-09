"""
Standalone evaluation for a trained AqSolDB conditional model.

Loads a saved DeFoG checkpoint and, for a 3x3 grid of joint (logP, SAS) targets,
generates molecules and plots dataset (gray) vs generated (red) property
distributions with the target as a black line.

Sampling is done in small CHUNKS on the GPU -- generating all samples in one
giant batch makes the ~32-node edge tensors enormous and memory-bound.

Usage:
    python experiments/eval_aqsoldb.py \
        --ckpt experiments/results/conditional_training__aqsoldb/debug/model \
        --csv data/aqsoldb_conditional.csv \
        --outdir experiments/results/aqsoldb_eval \
        --num-samples 500 --chunk 32
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, RDConfig

from experiments.utils import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.conditional_generation import build_normalization_stats
from defog.core import DeFoGModel, ConditionalSizeDistribution

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROPERTIES = {
    "logp": {"type": "regression", "callback": lambda m: float(Crippen.MolLogP(m))},
    "sas": {"type": "regression", "callback": lambda m: float(sascorer.calculateScore(m))},
}
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
LEVELS = ["low", "med", "high"]
PERCENTILES = [10, 50, 90]


def derive_atom_types(smiles_list):
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def plot_target(df, generated, targets, names, title):
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        data_vals = df[name].values
        gen_vals = np.array(generated.get(name, []))
        bins = np.linspace(float(np.min(data_vals)), float(np.max(data_vals)), 40)
        ax.hist(data_vals, bins=bins, density=True, color="0.7", label="dataset", alpha=0.9)
        if gen_vals.size:
            ax.hist(gen_vals, bins=bins, density=True, color="red", label="generated", alpha=0.55)
        ax.axvline(targets[name], color="black", linestyle="--", linewidth=2,
                   label=f"target = {targets[name]:.2f}")
        ax.set_xlabel(name); ax.set_ylabel("density"); ax.legend()
    fig.suptitle(title); fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--sample-steps", type=int, default=1000)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--time-distortion", type=str, default="polydec")
    ap.add_argument("--eta", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    names = sorted(PROPERTIES.keys())
    atom_types = derive_atom_types(df["smiles"])
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    norm_stats = build_normalization_stats(PROPERTIES, df)
    print(f"atoms={atom_types}\nnorm={norm_stats}", flush=True)

    model = DeFoGModel.load(args.ckpt)
    model = model.to(device)
    model.eval()
    print(f"model on {next(model.parameters()).device}, cond_dim={model.cond_dim}", flush=True)

    # Conditional size distribution from (normalized condition, heavy-atom size) pairs.
    conds, sizes = [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        z = [(float(row[n]) - norm_stats[n]["mean"]) / norm_stats[n]["std"] for n in names]
        conds.append(z)
        sizes.append(mol.GetNumHeavyAtoms())
    size_dist = ConditionalSizeDistribution(
        torch.tensor(conds), torch.tensor(sizes), method="kernel"
    )

    levels = {n: dict(zip(LEVELS, np.percentile(df[n].values, PERCENTILES))) for n in names}

    grid = []
    for la in LEVELS:            # logp
        for lb in LEVELS:        # sas
            targets = {names[0]: float(levels[names[0]][la]),
                       names[1]: float(levels[names[1]][lb])}
            zvec = torch.tensor(
                [(targets[n] - norm_stats[n]["mean"]) / norm_stats[n]["std"] for n in names],
                dtype=torch.float,
            )

            # chunked sampling
            samples = []
            remaining = args.num_samples
            while remaining > 0:
                cur = min(args.chunk, remaining)
                cond = zvec.unsqueeze(0).expand(cur, -1)
                samples += model.sample(
                    num_samples=cur, condition=cond, guidance_scale=args.guidance_scale,
                    sample_steps=args.sample_steps, size_dist=size_dist,
                    eta=args.eta, time_distortion=args.time_distortion,
                    device=device, show_progress=False,
                )
                remaining -= cur

            generated = {n: [] for n in names}
            n_valid = 0
            for s in samples:
                mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
                smi = mol_to_smiles(mol) if mol is not None else None
                if smi is None:
                    continue
                emol = Chem.MolFromSmiles(smi)
                if emol is None:
                    continue
                n_valid += 1
                for n in names:
                    try:
                        generated[n].append(PROPERTIES[n]["callback"](emol))
                    except Exception:
                        pass

            tag = f"logp-{la}_sas-{lb}"
            title = (f"logP {la}={targets['logp']:.2f}, SAS {lb}={targets['sas']:.2f}  "
                     f"(valid {n_valid}/{args.num_samples})")
            fig = plot_target(df, generated, targets, names, title)
            fig.savefig(os.path.join(args.outdir, f"target_{tag}.png"), dpi=110)
            plt.close(fig)

            row = {"tag": tag, "target_logp": targets["logp"], "target_sas": targets["sas"],
                   "n_valid": n_valid, "validity": n_valid / args.num_samples}
            for n in names:
                if generated[n]:
                    row[f"{n}_mean"] = float(np.mean(generated[n]))
                    row[f"{n}_mae"] = float(np.mean(np.abs(np.array(generated[n]) - targets[n])))
            grid.append(row)
            print(f"{tag}: validity={row['validity']:.1%} ({n_valid}/{args.num_samples})  "
                  + "  ".join(f"{n}_mae={row.get(f'{n}_mae', float('nan')):.3f}" for n in names),
                  flush=True)

    with open(os.path.join(args.outdir, "grid_metrics.json"), "w") as f:
        json.dump(grid, f, indent=2)
    overall = np.mean([r["validity"] for r in grid])
    print(f"\nOVERALL mean validity across targets: {overall:.1%}", flush=True)


if __name__ == "__main__":
    main()
