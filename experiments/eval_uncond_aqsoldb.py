"""
Unconditional generation from a trained AqSolDB conditional checkpoint.

Samples with condition=None (the learned null embedding) and the model's default
marginal size distribution, then reports validity and uniqueness. Sampling is
chunked on the GPU to avoid memory-bound giant-batch sampling.

Usage:
    python experiments/eval_uncond_aqsoldb.py \
        --ckpt experiments/results/conditional_training__aqsoldb/debug/model \
        --csv data/aqsoldb_conditional.csv \
        --num-samples 1000 --chunk 32
"""
import argparse

import pandas as pd
import torch
from rdkit import Chem, RDLogger

from experiments.utils import build_encoders, pyg_data_to_mol, mol_to_smiles
from defog.core import DeFoGModel

RDLogger.DisableLog("rdApp.*")
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def derive_atom_types(smiles_list):
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--sample-steps", type=int, default=1000)
    ap.add_argument("--time-distortion", type=str, default="polydec")
    ap.add_argument("--eta", type=float, default=100.0)   # authors' AqSolDB value
    ap.add_argument("--omega", type=float, default=0.3)   # authors' AqSolDB value
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(args.csv)
    atom_types = derive_atom_types(df["smiles"])
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, BOND_TYPES)

    model = DeFoGModel.load(args.ckpt)
    model = model.to(device)
    model.eval()
    print(f"model on {next(model.parameters()).device}, cond_dim={model.cond_dim}", flush=True)
    print(f"sampling: steps={args.sample_steps} distortion={args.time_distortion} "
          f"eta={args.eta}", flush=True)

    # Unconditional sampling in chunks (condition=None -> learned null embedding;
    # size_dist=None -> the model's default marginal P(n)).
    samples = []
    remaining = args.num_samples
    while remaining > 0:
        cur = min(args.chunk, remaining)
        samples += model.sample(
            num_samples=cur, condition=None, sample_steps=args.sample_steps,
            eta=args.eta, omega=args.omega, time_distortion=args.time_distortion,
            device=device, show_progress=False,
        )
        remaining -= cur
        print(f"  sampled {len(samples)}/{args.num_samples}", flush=True)

    valid_smiles = []
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            valid_smiles.append(smi)

    n = args.num_samples
    n_valid = len(valid_smiles)
    n_unique = len(set(valid_smiles))
    print(f"\n=== UNCONDITIONAL GENERATION ({n} molecules) ===", flush=True)
    print(f"validity:   {n_valid}/{n} = {n_valid / n:.1%}", flush=True)
    print(f"uniqueness: {n_unique}/{n_valid} = "
          f"{(n_unique / n_valid) if n_valid else 0:.1%} (of valid)", flush=True)


if __name__ == "__main__":
    main()
