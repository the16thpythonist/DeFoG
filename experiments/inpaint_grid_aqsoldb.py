"""
RePaint-style inpainting demo on the AqSolDB checkpoint.

Freezes a small core fragment (default: benzene) and grows new atoms around it
via ``InpaintingSampler``, then renders a 25x25 grid of completions with the
frozen core highlighted in every molecule.

Usage:
    python experiments/inpaint_grid_aqsoldb.py \
        --ckpt /home/jonas/Downloads/aqsoldb_4e-4_best_model.ckpt \
        --csv data/aqsoldb_conditional.csv \
        --core "c1ccccc1" --out inpaint_benzene_grid.png
"""
import argparse
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from defog.core import DeFoGModel, SubgraphConstraint, InpaintingSampler
from defog.domains.molecule import (
    MoleculeDomain, build_encoders, pyg_data_to_mol, mol_to_smiles,
)

RDLogger.DisableLog("rdApp.*")
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
GRID = 25                      # 25 x 25
N_CELLS = GRID * GRID          # 625
CELL = 200                     # px per cell


def derive_atom_types(smiles_list):
    counts = {}
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for a in m.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def draw_cell(smiles, core_mol, size=CELL):
    """RGB array of `smiles` with the core substructure highlighted (or a blank cell)."""
    if smiles is None:
        return np.full((size, size, 3), 255, np.uint8)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.full((size, size, 3), 255, np.uint8)
    match = m.GetSubstructMatch(core_mol)
    hl_atoms = list(match)
    hl_bonds = []
    if match:
        ms = set(match)
        for b in m.GetBonds():
            if b.GetBeginAtomIdx() in ms and b.GetEndAtomIdx() in ms:
                hl_bonds.append(b.GetIdx())
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = d.drawOptions()
    opts.padding = 0.08
    color = (1.0, 0.78, 0.30)  # warm amber highlight for the frozen core
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, m,
        highlightAtoms=hl_atoms, highlightBonds=hl_bonds,
        highlightAtomColors={a: color for a in hl_atoms},
        highlightBondColors={b: color for b in hl_bonds},
    )
    d.FinishDrawing()
    img = Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/jonas/Downloads/aqsoldb_4e-4_best_model.ckpt")
    ap.add_argument("--csv", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--core", default="c1ccccc1", help="SMILES of the frozen core fragment")
    ap.add_argument("--out", default="inpaint_benzene_grid.png")
    ap.add_argument("--sample-steps", type=int, default=250)
    ap.add_argument("--eta", type=float, default=50.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--n-free", type=int, nargs="+", default=[4, 6, 8, 10],
                    help="new-atom counts cycled for size diversity")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-attempts", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"device: {device}", flush=True)

    atom_types = derive_atom_types(pd.read_csv(args.csv)["smiles"])
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    dom = MoleculeDomain(atom_decoder, bond_decoder)

    model = DeFoGModel.load(args.ckpt, device=device)
    model.eval()

    core_mol = Chem.MolFromSmiles(args.core)
    Xc, Ec = dom.encode(args.core)
    k = Xc.shape[0]
    print(f"core = {args.core!r} (k={k} atoms). free-valence report:", flush=True)
    for r in dom.core_valence_report(Xc, Ec):
        print("   ", r, flush=True)

    constraint = SubgraphConstraint(Xc, Ec)

    # Generate a pool, keeping valid completions that contain the core, until we
    # have enough to fill the grid. Cycle n_free for size diversity.
    kept = []                       # canonical SMILES (valid + contains core)
    n_gen = n_valid = n_core = 0
    nf_cycle = args.n_free
    attempt = 0
    while len(kept) < N_CELLS and n_gen < args.max_attempts:
        n_free = nf_cycle[attempt % len(nf_cycle)]
        attempt += 1
        sampler = InpaintingSampler(
            model, constraint,
            eta=args.eta, omega=args.omega,
            sample_steps=args.sample_steps, time_distortion=args.time_distortion,
        )
        samples = sampler.sample(num_samples=args.chunk, n_free=n_free,
                                 device=device, show_progress=False)
        for s in samples:
            n_gen += 1
            mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
            smi = mol_to_smiles(mol) if mol is not None else None
            if smi is None or Chem.MolFromSmiles(smi) is None:
                continue
            n_valid += 1
            if Chem.MolFromSmiles(smi).HasSubstructMatch(core_mol):
                n_core += 1
                if len(kept) < N_CELLS:
                    kept.append(smi)
        print(f"  gen={n_gen} valid={n_valid} core={n_core} kept={len(kept)}/{N_CELLS}",
              flush=True)

    validity = n_valid / n_gen if n_gen else 0.0
    core_rate = n_core / n_valid if n_valid else 0.0
    print(f"\nvalidity={validity:.1%}  core-preservation(of valid)={core_rate:.1%}  "
          f"generated={n_gen}", flush=True)

    # Render the grid as one big canvas (fast, crisp).
    print("rendering grid ...", flush=True)
    canvas = np.full((GRID * CELL, GRID * CELL, 3), 255, np.uint8)
    for idx in range(N_CELLS):
        r, c = divmod(idx, GRID)
        smi = kept[idx] if idx < len(kept) else None
        canvas[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = draw_cell(smi, core_mol)
    # thin grid lines
    for i in range(GRID + 1):
        canvas[min(i * CELL, GRID * CELL - 1), :] = 225
        canvas[:, min(i * CELL, GRID * CELL - 1)] = 225

    fig = plt.figure(figsize=(20, 21))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.955])
    ax.imshow(canvas)
    ax.axis("off")
    core_smi = Chem.MolToSmiles(core_mol)
    fig.suptitle(
        f"DeFoG RePaint inpainting  |  frozen core = {core_smi} (highlighted)  |  "
        f"{GRID}×{GRID} = {N_CELLS} completions\n"
        f"AqSolDB checkpoint  ·  η={args.eta} ω={args.omega} steps={args.sample_steps} "
        f"{args.time_distortion}  ·  n_free∈{args.n_free}  ·  "
        f"validity {validity:.0%} · core preserved {core_rate:.0%} of valid",
        fontsize=15, y=0.988,
    )
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
