"""
A/B comparison of RePaint time-travel resampling for DeFoG inpainting.

Runs two 5x5 raw (unfiltered) inpainting grids from the SAME seed on the same
core -- one with time-travel resampling off, one on -- so the effect of boundary
harmonization can be eyeballed side by side. Cells are shown as generated
(invalid completions are greyed), and cell i of both grids starts from identical
initial noise (same seed), so they are loosely paired.
"""
import argparse
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem, RDLogger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from defog.core import DeFoGModel, SubgraphConstraint, InpaintingSampler
from defog.domains.molecule import (
    MoleculeDomain, build_encoders, pyg_data_to_mol, mol_to_smiles,
)
from experiments.inpaint_grid_aqsoldb import derive_atom_types, draw_cell

RDLogger.DisableLog("rdApp.*")
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
GRID = 5
CELL = 220
GREY = np.full((CELL, CELL, 3), 238, np.uint8)  # invalid-cell fill


def decode(samples, atom_decoder, bond_decoder, core_mol):
    """Return (list of smi-or-None, n_valid, n_core)."""
    smis, n_valid, n_core = [], 0, 0
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        ok = smi is not None and Chem.MolFromSmiles(smi) is not None
        if ok:
            n_valid += 1
            if Chem.MolFromSmiles(smi).HasSubstructMatch(core_mol):
                n_core += 1
            smis.append(smi)
        else:
            smis.append(None)
    return smis, n_valid, n_core


def render_grid(smis, core_mol, title, out):
    canvas = np.full((GRID * CELL, GRID * CELL, 3), 255, np.uint8)
    for idx in range(GRID * GRID):
        r, c = divmod(idx, GRID)
        smi = smis[idx] if idx < len(smis) else None
        cell = GREY.copy() if smi is None else draw_cell(smi, core_mol, size=CELL)
        canvas[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = cell
    for i in range(GRID + 1):
        canvas[min(i * CELL, GRID * CELL - 1), :] = 210
        canvas[:, min(i * CELL, GRID * CELL - 1)] = 210
    fig = plt.figure(figsize=(11, 11.7))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.93])
    ax.imshow(canvas); ax.axis("off")
    fig.suptitle(title, fontsize=12, y=0.985)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}", flush=True)


def run_condition(model, constraint, core_mol, atom_decoder, bond_decoder, *,
                  resample, args, device):
    torch.manual_seed(args.seed)  # SAME seed for both conditions
    sampler = InpaintingSampler(
        model, constraint,
        eta=args.eta, omega=args.omega,
        sample_steps=args.sample_steps, time_distortion=args.time_distortion,
        resample=resample, n_resample=args.n_resample,
    )
    samples = sampler.sample(num_samples=GRID * GRID, n_free=args.n_free,
                             device=device, show_progress=True)
    smis, n_valid, n_core = decode(samples, atom_decoder, bond_decoder, core_mol)
    tag = "time-travel ON" if resample else "time-travel OFF"
    print(f"[{tag}] valid={n_valid}/{GRID*GRID} core={n_core}/{n_valid}", flush=True)
    return smis, n_valid, n_core, sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/jonas/Downloads/aqsoldb_4e-4_best_model.ckpt")
    ap.add_argument("--csv", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--core", default="Nc1ccccc1", help="aniline: benzene + NH2")
    ap.add_argument("--n-free", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=250)
    ap.add_argument("--eta", type=float, default=50.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--n-resample", type=int, default=8)     # "strong", RePaint-like
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-off", default="inpaint_ab_noTT.png")
    ap.add_argument("--out-on", default="inpaint_ab_TT.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    atom_types = derive_atom_types(pd.read_csv(args.csv)["smiles"])
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, BOND_TYPES)
    dom = MoleculeDomain(atom_decoder, bond_decoder)
    model = DeFoGModel.load(args.ckpt, device=device); model.eval()

    core_mol = Chem.MolFromSmiles(args.core)
    Xc, Ec = dom.encode(args.core)
    k = Xc.shape[0]
    print(f"core={args.core!r} (k={k}). free-valence report:", flush=True)
    for r in dom.core_valence_report(Xc, Ec):
        print("   ", r, flush=True)
    constraint = SubgraphConstraint(Xc, Ec)

    common = f"core={Chem.MolToSmiles(core_mol)} (highlighted) · n_free={args.n_free} · " \
             f"η={args.eta} ω={args.omega} steps={args.sample_steps} {args.time_distortion} · seed={args.seed}"

    # OFF
    smis_off, v_off, c_off, _ = run_condition(
        model, constraint, core_mol, atom_decoder, bond_decoder,
        resample=False, args=args, device=device)
    render_grid(
        smis_off, core_mol,
        f"Inpainting WITHOUT time-travel (plain replacement)\n{common} · "
        f"valid {v_off}/25 · core {c_off}/{v_off if v_off else 0}",
        args.out_off)

    # ON
    smis_on, v_on, c_on, samp_on = run_condition(
        model, constraint, core_mol, atom_decoder, bond_decoder,
        resample=True, args=args, device=device)
    render_grid(
        smis_on, core_mol,
        f"Inpainting WITH time-travel (RePaint resampling, r={args.n_resample}, "
        f"jump={samp_on.jump_length})\n{common} · valid {v_on}/25 · "
        f"core {c_on}/{v_on if v_on else 0}",
        args.out_on)

    print(f"\nSUMMARY  off: valid={v_off}/25 core={c_off}  |  "
          f"on: valid={v_on}/25 core={c_on}", flush=True)


if __name__ == "__main__":
    main()
