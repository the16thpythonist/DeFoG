#!/usr/bin/env python
"""In-job gate: refuse to spend the sweep unless BOTH fixes are actually live.

Run this as the first thing the job does. Two defects are being corrected at once and each
has a silent-failure mode that has already cost a job on this project:

  1. The FK energy re-encoded decoded molecules without kekulizing, so on the kekulized
     zinc-kek vocabulary ~94% of real molecules scored ``invalid_energy=1e3`` and the head
     was never consulted.
  2. The energy was not divided by the property's std, so beta meant ~76x less pressure on
     QED than on logP -- and the scale field it needs was only filled by ``run()``, which
     this harness does not call, so the fix to (2) is itself easy to land as a no-op.

Checking that the code contains the fix is not enough for either: what matters is the value
that reaches the energy on the config that actually samples. So this measures.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ADAPTER = "molsmith/clogp@1.2.0"
EXPECTED_SCALE = 1.1581127643585205


def fail(msg: str) -> None:
    print(f"GATE FAILED: {msg}")
    raise SystemExit(1)


def main() -> int:
    from molsmith import sample as ms
    from molsmith.sample import _energy
    from torch_geometric.data import Batch

    from defog.core.data import to_dense
    from defog.core.property_head import LearnedPropertyEnergy
    from defog.data import zinc_reference as zref
    from defog.domains.molecule import smiles_to_pyg_data

    def cfg_for(target, seed):
        return ms.SamplingConfig(
            base="molsmith/zinc-kek", n=4, seed=seed, steps=10,
            adapters=[ms.AdapterTarget(package=ADAPTER, target=target, weight=2.0)],
            blend_space="prob", method="fk")

    probe = cfg_for(2.5, 42)
    loaded = ms.load(probe)
    scale = float(probe.adapters[0].scale)
    print(f"[1] load fills scale: {scale!r}  property={probe.adapters[0].property!r}")
    if abs(scale - EXPECTED_SCALE) > 1e-9:
        fail(f"load() left scale={scale}, expected {EXPECTED_SCALE}")

    ae, be = loaded.atom_encoder, loaded.bond_encoder
    smis = zref.load_reference_split().val_smiles[:64]
    datas = [d for d in (smiles_to_pyg_data(s, ae, be, kekulize=True) for s in smis)
             if d is not None]
    batch = Batch.from_data_list(datas)
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)

    head = loaded.heads[ADAPTER]
    raw_energy = LearnedPropertyEnergy(head, 2.5, loaded.domain, ae, be)
    with torch.no_grad():
        e_raw = raw_energy(dense.X, dense.E, mask).cpu().numpy()
        e_norm = _energy(probe, loaded)(dense.X, dense.E, mask).cpu().numpy()

    n_invalid = int((e_raw >= 1e3 - 1).sum())
    print(f"[2] kekulize fix: {n_invalid}/{len(e_raw)} real molecules score invalid_energy")
    if n_invalid > 0:
        fail(f"{n_invalid}/{len(e_raw)} real molecules still rejected -- kekulize fix absent")

    want = 1.0 / EXPECTED_SCALE ** 2
    ratio = e_norm / np.maximum(e_raw, 1e-12)
    print(f"[3] energy z-scored: ratio {ratio.min():.6f}..{ratio.max():.6f}, want {want:.6f}")
    if not np.allclose(ratio, want, rtol=1e-5):
        fail("energy is not divided by scale^2 -- normalisation absent")

    fresh = cfg_for(3.7, 99)
    ms.sample(fresh, loaded)
    fresh_scale = float(fresh.adapters[0].scale)
    print(f"[4] sample() fills a FRESH config: scale={fresh_scale!r}")
    if abs(fresh_scale - EXPECTED_SCALE) > 1e-9:
        fail("sample() does not fill scale; 99 of 100 targets would run un-normalised")

    print("\nGATE PASSED -- both fixes are live on the config that actually samples.")
    print(f"  dimensionless beta B  ==  raw-units beta B/{EXPECTED_SCALE ** 2:.4f} for logP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
