"""
Render a 5x5 grid of random valid molecules for each LR-sweep arm.

The sweep (job 26144) persisted only metrics + checkpoints, not the generated eval
molecules -- so this regenerates a fresh random sample from each arm's
best-validity checkpoint (`best_model.ckpt`, the one the final eval used) with the
model's own eval sampling defaults (eta=100, omega=0.3, polydec, 1000 steps),
which is statistically identical to the reported eval. Each grid is drawn with the
RDKit `MoleculeDomain` backend. Writes one PNG per arm into the repo root.

Run on a GPU node: sbatch experiments/run_make_grids_kcist.sh
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from defog.core import DeFoGModel
from defog.domains import MoleculeDomain
from experiments.utils import build_encoders

RDLogger.DisableLog("rdApp.*")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "data", "aqsoldb_conditional.csv")
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
JOB = "26144"
LRS = ["5e-5", "1e-4", "2e-4", "4e-4"]
POOL = 48          # sample this many, keep valid, take 25
N_GRID = 25        # 5x5
SAMPLE_STEPS = 1000
SEED = 20260710


def derive_atom_types(smiles_list):
    counts = {}
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for a in m.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def find_archive(lr):
    log = os.path.join(REPO, f"aqsoldb_lr_{lr}_{JOB}.out")
    with open(log) as f:
        for line in f:
            if "archive path" in line:
                return line.split("archive path:")[1].strip()
    raise RuntimeError(f"archive path not found for LR={lr}")


def main():
    df = pd.read_csv(CSV)
    atom_types = derive_atom_types(df["smiles"])
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    dom = MoleculeDomain(atom_decoder, bond_decoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "| atom_types", atom_types)

    random.seed(SEED)
    torch.manual_seed(SEED)

    for lr in LRS:
        ap = find_archive(lr)
        model = DeFoGModel.load(os.path.join(ap, "best_model.ckpt"), device=device)
        model.eval()
        samples = model.sample(num_samples=POOL, sample_steps=SAMPLE_STEPS,
                                device=device, show_progress=False)
        valid = [s for s in samples if dom.is_valid(s)]
        random.shuffle(valid)
        grid = valid[:N_GRID]

        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        flat = axes.flatten()
        for i, ax in enumerate(flat):
            if i < len(grid):
                dom.render(ax, grid[i])
                cap = dom.caption(grid[i])
                if cap:
                    ax.set_title(cap, fontsize=6)
            else:
                ax.axis("off")
        fig.suptitle(
            f"AqSolDB unconditional -- LR {lr} -- 25 random valid samples "
            f"(best-validity ckpt; {len(valid)}/{len(samples)} valid)",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(REPO, f"sample_grid_lr_{lr}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"LR={lr}: sampled {len(samples)}, valid {len(valid)}, saved {out}")

    print("DONE")


if __name__ == "__main__":
    main()
