"""
Backfill generated_smiles.json for the completed KCIST LR sweep (job 26144).

The sweep predates the SMILES-saving change, so its generated molecules weren't
persisted. This regenerates NUM_EVAL_SAMPLES from each arm's best-validity
checkpoint (the eval checkpoint) with the model's eval sampling defaults --
statistically identical to the reported eval -- and writes the tagged SMILES
(all generated, valid/unique/novel) into each arm's archive.

Novelty reference = the full AqSolDB set (the sweep had no fixed seed, so the exact
train split isn't recoverable; the full set is a conservative superset).

Run on a GPU node: sbatch experiments/run_backfill_kcist.sh
"""
import os
import json

import pandas as pd
import torch
from rdkit import Chem, RDLogger

from defog.core import DeFoGModel
from experiments.utils import build_encoders, tag_generated_smiles

RDLogger.DisableLog("rdApp.*")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "data", "aqsoldb_conditional.csv")
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
JOB = "26144"
LRS = ["5e-5", "1e-4", "2e-4", "4e-4"]
N = 1000
STEPS = 1000
CHUNK = 32
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
    with open(os.path.join(REPO, f"aqsoldb_lr_{lr}_{JOB}.out")) as f:
        for line in f:
            if "archive path" in line:
                return line.split("archive path:")[1].strip()
    raise RuntimeError(f"archive path not found for LR={lr}")


def main():
    df = pd.read_csv(CSV)
    atom_types = derive_atom_types(df["smiles"])
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    ref_smiles = list(df["smiles"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    print("device", device)

    for lr in LRS:
        ap = find_archive(lr)
        model = DeFoGModel.load(os.path.join(ap, "best_model.ckpt"), device=device)
        model.eval()
        samples = []
        remaining = N
        while remaining > 0:
            cur = min(CHUNK, remaining)
            samples += model.sample(num_samples=cur, sample_steps=STEPS,
                                     device=device, show_progress=False)
            remaining -= cur
        recs = tag_generated_smiles(samples, atom_decoder, bond_decoder, ref_smiles)
        out = os.path.join(ap, "generated_smiles.json")
        with open(out, "w") as f:
            json.dump(recs, f)
        n_valid = sum(r["valid"] for r in recs)
        n_unique = sum(1 for r in recs if r["unique"])
        print(f"LR={lr}: {len(recs)} saved, valid={n_valid}, unique={n_unique} -> {out}")

    print("DONE")


if __name__ == "__main__":
    main()
