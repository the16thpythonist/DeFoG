"""
Train an amortized guidance network for an arbitrary RDKit molecular property on
AqSolDB (same recipe as the logP guidance). Property values are computed on the
fly, so no CSV column is needed.

    python experiments/train_property_guidance.py --property tpsa --outdir experiments/_tpsa_guidance

Saves <property>_guidance.ckpt (the trained h) + <property>_stats.json
(mean/std/p10/p50/p90 + gamma/scale), which the joint-sampling driver loads.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from torch_geometric.loader import DataLoader

from defog.core import DeFoGModel, AmortizedPropertyGuidanceModule
from defog.domains.molecule import build_encoders, smiles_to_pyg_data
from experiments.guided_logp_demo import derive_atom_types, BOND_TYPES

RDLogger.DisableLog("rdApp.*")

# SA score (synthetic accessibility) ships in RDKit's Contrib dir — add to path.
import sys
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

PROP_FNS = {
    "logp": Crippen.MolLogP,
    "tpsa": Descriptors.TPSA,
    "mw": Descriptors.MolWt,
    "nhoh": Descriptors.NHOHCount,
    "sascore": sascorer.calculateScore,
}


def build_property_dataset(df, atom_encoder, bond_encoder, prop_fn):
    graphs, vals = [], []
    skipped = 0
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue
        try:
            data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        except Exception:
            data = None
        if data is None or data.edge_index.numel() == 0:
            skipped += 1
            continue
        try:
            v = float(prop_fn(mol))
        except Exception:
            skipped += 1
            continue
        data.prop_val = torch.tensor([v], dtype=torch.float)
        graphs.append(data)
        vals.append(v)
    return graphs, np.array(vals), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--property", required=True, choices=sorted(PROP_FNS))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--h-layers", type=int, default=6)
    ap.add_argument("--h-hidden", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=3.0,
                    help="normalized energy sharpness r=exp(-gamma*((p-c)/std)^2)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atom-decoder", default=None,
                    help="comma-separated atom order matching the BASE MODEL's node classes "
                         "(e.g. ZINC: C,N,O,F,P,S,Cl,Br,I). Overrides the data-derived "
                         "frequency order, which need not match the checkpoint.")
    args = ap.parse_args()

    outdir = args.outdir or f"experiments/_{args.property}_guidance"
    os.makedirs(outdir, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    atom_types = (
        [s.strip() for s in args.atom_decoder.split(",")]
        if args.atom_decoder
        else derive_atom_types(df["smiles"])
    )
    print(f"[train] atom vocab ({len(atom_types)}): {atom_types}", flush=True)
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    prop_fn = PROP_FNS[args.property]

    graphs, vals, skipped = build_property_dataset(df, atom_encoder, bond_encoder, prop_fn)
    prop_mean, prop_std = float(vals.mean()), float(vals.std())
    lo, hi = np.percentile(vals, [1, 99])
    prop_values = vals[(vals >= lo) & (vals <= hi)]
    p10, p50, p90 = [float(x) for x in np.percentile(vals, [10, 50, 90])]
    print(f"[train] property={args.property} n={len(graphs)} ({skipped} skipped) "
          f"mean={prop_mean:.2f} std={prop_std:.2f} p10/50/90={p10:.1f}/{p50:.1f}/{p90:.1f}", flush=True)

    base = DeFoGModel.load(args.ckpt, device="cpu")
    module = AmortizedPropertyGuidanceModule(
        base, prop_values=prop_values, prop_mean=prop_mean, prop_std=prop_std,
        gamma=args.gamma, prop_scale=prop_std, prop_attr="prop_val", lr=args.lr,
        n_layers=args.h_layers, hidden_dim=args.h_hidden,
    )
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1, logger=False, enable_checkpointing=False, enable_model_summary=False,
        gradient_clip_val=1.0, log_every_n_steps=50,
    )
    print(f"[train] fitting {args.property} guidance: epochs={args.epochs} "
          f"h={args.h_layers}x{args.h_hidden} gamma={args.gamma} scale={prop_std:.2f}", flush=True)
    trainer.fit(module, loader)

    ckpt = os.path.join(outdir, f"{args.property}_guidance")
    module.guidance().save(ckpt)
    stats = {"property": args.property, "mean": prop_mean, "std": prop_std,
             "p10": p10, "p50": p50, "p90": p90, "gamma": args.gamma, "scale": prop_std,
             "n": len(graphs)}
    with open(os.path.join(outdir, f"{args.property}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[train] DONE -> {ckpt}.ckpt + stats; {stats}", flush=True)


if __name__ == "__main__":
    main()
