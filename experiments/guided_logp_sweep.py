"""
Sweep the guidance-strength knobs on an ALREADY-TRAINED amortized logP guidance
network (no retraining) to see how far we can push generated logP toward the
target. Reuses guided_logp_amortized.ckpt + the frozen base.

Knobs swept:
  * weight w:  q ∝ h^w · p = softmax(w·g + log p)   (w=1 exact; w>1 stronger)
  * eta:       CTMC stochasticity (lower -> follows guidance more directly)
  * omega:     target-guidance strength (higher -> pushes toward guided prediction)

Output: experiments/_guided_logp_sweep/
  * sweep_achieved_vs_target.png  -- achieved mean logP vs target, one line per config
  * sweep_metrics.json
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

from defog.core import DeFoGModel, GuidedSampler, ExactGuidance
from defog.domains.molecule import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles

RDLogger.DisableLog("rdApp.*")
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
LEVELS = ["low", "med", "high"]


def derive_atom_types(smiles_list):
    counts = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] += 1
    return [s for s, _ in counts.most_common()]


@torch.no_grad()
def sample_logps(base, guidance, target, n, chunk, steps, eta, omega, distortion,
                 device, atom_decoder, bond_decoder):
    guidance.set_target(target)
    sampler = GuidedSampler(base, guidance, eta=eta, omega=omega,
                            sample_steps=steps, time_distortion=distortion)
    logps = []
    remaining = n
    while remaining > 0:
        cur = min(chunk, remaining)
        for s in sampler.sample(cur, device=device, show_progress=False):
            mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
            if mol is not None and mol_to_smiles(mol) is not None:
                try:
                    logps.append(float(Crippen.MolLogP(mol)))
                except Exception:
                    pass
        remaining -= cur
    return np.array(logps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--guidance", default="experiments/_guided_logp_out/guided_logp_amortized.ckpt")
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_guided_logp_sweep")
    ap.add_argument("--num-eval", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--sample-steps", type=int, default=300)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    atom_types = derive_atom_types(df["smiles"])
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    logp_all = df["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    targets = dict(zip(LEVELS, np.percentile(logp_all, [10, 50, 90])))
    print(f"[sweep] device={device} targets={ {k: round(v,2) for k,v in targets.items()} }", flush=True)

    base = DeFoGModel.load(args.ckpt, device="cpu").to(device).eval()
    h = DeFoGModel.load(args.guidance, device="cpu").to(device).eval()
    guidance = ExactGuidance(h, prop_mean=prop_mean, prop_std=prop_std)

    # (label, weight, eta, omega)
    configs = [
        ("w1_eta10_om0.2", 1.0, 10.0, 0.2),   # original (baseline)
        ("w2_eta10_om0.2", 2.0, 10.0, 0.2),   # crank guidance weight
        ("w3_eta10_om0.2", 3.0, 10.0, 0.2),
        ("w3_eta3_om0.4",  3.0, 3.0, 0.4),    # + lower stochasticity, more target guidance
        ("w4_eta1_om0.5",  4.0, 1.0, 0.5),    # aggressive
    ]

    results = {}
    for label, w, eta, omega in configs:
        guidance.set_weight(w)
        row = {"weight": w, "eta": eta, "omega": omega, "per_target": {}}
        line = []
        for lvl in LEVELS:
            tgt = float(targets[lvl])
            lps = sample_logps(base, guidance, tgt, args.num_eval, args.chunk,
                               args.sample_steps, eta, omega, args.time_distortion,
                               device, atom_decoder, bond_decoder)
            achieved = float(np.mean(lps)) if len(lps) else float("nan")
            mae = float(np.mean(np.abs(lps - tgt))) if len(lps) else float("nan")
            row["per_target"][lvl] = {
                "target": tgt, "achieved_mean": achieved, "mae": mae,
                "validity": len(lps) / args.num_eval,
            }
            line.append(f"{lvl}:{achieved:.2f}(v{len(lps)/args.num_eval:.0%})")
        results[label] = row
        print(f"[sweep] {label:18s}  " + "  ".join(line), flush=True)

    # Plot achieved mean vs target for each config.
    fig, ax = plt.subplots(figsize=(7.5, 6))
    tx = [targets[l] for l in LEVELS]
    ax.plot(tx, tx, "k--", lw=1, label="ideal (achieved = target)")
    for label, row in results.items():
        ay = [row["per_target"][l]["achieved_mean"] for l in LEVELS]
        ax.plot(tx, ay, marker="o", label=label)
    ax.set_xlabel("target logP"); ax.set_ylabel("achieved mean logP")
    ax.set_title("Guidance-strength sweep: achieved vs requested logP\n(reusing one trained h, no retraining)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sweep_achieved_vs_target.png"), dpi=140)

    with open(os.path.join(args.outdir, "sweep_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[sweep] DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
