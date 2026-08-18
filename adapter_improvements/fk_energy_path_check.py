#!/usr/bin/env python
"""Does LearnedPropertyEnergy actually score real molecules, or does it return 1e3?

LearnedPropertyEnergy re-encodes each decoded molecule with
``smiles_to_pyg_data(Chem.MolToSmiles(mol), ae, be)`` -- no ``kekulize=True``
(property_head.py:159). For the zinc-kek base the bond vocabulary is
{SINGLE, DOUBLE, TRIPLE} with no AROMATIC class, and ``Chem.MolToSmiles``
returns aromatic SMILES. If that combination rejects the molecule, the energy
falls through to ``invalid_energy=1e3`` and the head is never consulted.

Feed it real ZINC molecules, encoded the way the model emits them (kekulized
dense one-hots), and look at what comes back.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/media/ssd2/Programming/DeFoG")
sys.path.insert(0, "/media/ssd2/Programming/defog-web")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from defog.core.data import to_dense  # noqa: E402
from defog.core.property_head import LearnedPropertyEnergy, PropertyHead  # noqa: E402
from defog.domains.molecule import (MoleculeDomain, build_encoders,  # noqa: E402
                                    smiles_to_pyg_data)

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE"]
HEAD = "/home/jonas/.molsmith/packages/e939dee7f22d3fd6/weights/head.safetensors"
CFG = dict(na=9, nb=4, hid=128, layers=3,
           prop_mean=2.8247015476226807, prop_std=1.1579052209854126)


def main() -> int:
    from torch_geometric.data import Batch

    from molsmith.weights.convert import read_safetensors

    ae, adec, be, bdec = build_encoders(ATOM_TYPES, BOND_TYPES)
    domain = MoleculeDomain(adec, bdec)
    sd, _ = read_safetensors(HEAD, device="cpu")
    head = PropertyHead.from_config(CFG, sd, device="cpu")

    from defog.data import zinc_reference as zref
    smis = zref.load_reference_split().val_smiles[:64]

    # Encode the way the MODEL emits graphs: kekulized one-hots.
    datas, kept = [], []
    for s in smis:
        d = smiles_to_pyg_data(s, ae, be, kekulize=True)
        if d is not None:
            datas.append(d)
            kept.append(s)
    print(f"kekulized encode of {len(smis)} validation SMILES -> {len(datas)} usable\n")

    batch = Batch.from_data_list(datas)
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)

    target = 2.8
    energy = LearnedPropertyEnergy(head, target, domain, ae, be)
    with torch.no_grad():
        E = energy(dense.X, dense.E, mask).cpu().numpy()

    n_invalid = int((E >= 1e3 - 1).sum())
    print(f"energies returned for {len(E)} real molecules:")
    print(f"  invalid_energy (1e3)   {n_invalid:4d} / {len(E)}   = {n_invalid/len(E):.1%}")
    print(f"  real energies          {len(E)-n_invalid:4d} / {len(E)}")
    if n_invalid < len(E):
        good = E[E < 1e3 - 1]
        print(f"  real-energy range      {good.min():.4f} .. {good.max():.4f}")

    # Ground truth for the few that survive, to confirm the head is being consulted at all.
    true = np.array([(Crippen.MolLogP(Chem.MolFromSmiles(s)) - target) ** 2 for s in kept])
    live = E < 1e3 - 1
    if live.any():
        print(f"  corr(head energy, true energy) on survivors: "
              f"{np.corrcoef(E[live], true[live])[0,1]:.3f}")

    print()
    if n_invalid == len(E):
        print("VERDICT: the head is NEVER consulted. Every particle scores 1e3, weights are "
              "uniform, and FK cannot steer on the property at all.")
    elif n_invalid > 0.5 * len(E):
        print(f"VERDICT: {n_invalid/len(E):.0%} of REAL, VALID molecules score as invalid. "
              f"FK's dominant signal is 'did the re-encode happen to succeed', not the "
              f"property.")
    else:
        print("VERDICT: the re-encode path works; the energy is a real property signal.")

    # Isolate the cause: same molecule, both encodings.
    print("\n--- isolating the cause on one molecule ---")
    s = kept[0]
    m = Chem.MolFromSmiles(s)
    round_trip = Chem.MolToSmiles(m)
    print(f"  decode -> MolToSmiles:  {round_trip}")
    print(f"  smiles_to_pyg_data(kekulize=False) -> "
          f"{'None  <-- REJECTED' if smiles_to_pyg_data(round_trip, ae, be) is None else 'ok'}")
    print(f"  smiles_to_pyg_data(kekulize=True)  -> "
          f"{'None' if smiles_to_pyg_data(round_trip, ae, be, kekulize=True) is None else 'ok'}")
    print(f"  molecule is aromatic:   {any(b.GetIsAromatic() for b in m.GetBonds())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
