#!/usr/bin/env python
"""Prove the FK energy is now z-scored, and that a per-target config carries the scale.

Two ways this fix could land and still do nothing, both of which have precedent in this
project:

  1. ``_energy`` applies the scale but ``spec.scale`` is 1.0, so the multiply is a no-op.
  2. ``load`` fills the scale but ``sample`` does not, so only the FIRST of 100 target
     configs is normalised and the other 99 silently run in raw units.

So this checks the value that reaches the energy, not just that the code path exists.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/media/ssd2/Programming/DeFoG")
sys.path.insert(0, "/media/ssd2/Programming/defog-web")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ADAPTER = "molsmith/clogp@1.2.0"          # the one installed locally
EXPECTED_SCALE = 1.1581127643585205        # prop_std from the package metadata


def main() -> int:
    from molsmith import sample as ms
    from molsmith.sample import _energy

    from defog.core.data import to_dense
    from defog.core.property_head import LearnedPropertyEnergy
    from defog.domains.molecule import smiles_to_pyg_data

    def cfg_for(target, seed):
        return ms.SamplingConfig(
            base="molsmith/zinc-kek", n=4, seed=seed, steps=10,
            adapters=[ms.AdapterTarget(package=ADAPTER, target=target, weight=2.0)],
            blend_space="prob", method="fk")

    probe = cfg_for(2.5, 42)
    loaded = ms.load(probe)
    print(f"after ms.load        scale = {probe.adapters[0].scale!r}  "
          f"property = {probe.adapters[0].property!r}")
    assert abs(probe.adapters[0].scale - EXPECTED_SCALE) < 1e-9, "load did not fill the scale"

    # (2) the trap: a FRESH config, the shape a target sweep builds 100 times.
    fresh = cfg_for(3.7, 99)
    print(f"fresh config         scale = {fresh.adapters[0].scale!r}   (1.0 before sampling)")

    # Real molecules to score.
    from defog.data import zinc_reference as zref
    from torch_geometric.data import Batch
    ae, be = loaded.atom_encoder, loaded.bond_encoder
    datas = [d for d in (smiles_to_pyg_data(s, ae, be, kekulize=True)
                         for s in zref.load_reference_split().val_smiles[:32]) if d is not None]
    batch = Batch.from_data_list(datas)
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)

    head = loaded.heads[ADAPTER]
    raw = LearnedPropertyEnergy(head, 2.5, loaded.domain, ae, be)
    with torch.no_grad():
        e_raw = raw(dense.X, dense.E, mask).cpu().numpy()
        e_norm = _energy(probe, loaded)(dense.X, dense.E, mask).cpu().numpy()

    ratio = e_norm / np.maximum(e_raw, 1e-12)
    want = 1.0 / EXPECTED_SCALE ** 2
    print(f"\nraw energy   mean {e_raw.mean():.4f}")
    print(f"norm energy  mean {e_norm.mean():.4f}")
    print(f"ratio        {ratio.min():.6f} .. {ratio.max():.6f}   expected {want:.6f}")
    ok_scale = bool(np.allclose(ratio, want, rtol=1e-5))
    print("ENERGY IS Z-SCORED" if ok_scale else "NOT NORMALISED -- fix did not take")

    # (2) again, end to end: does sample() fill a fresh config?
    ms.sample(fresh, loaded)
    print(f"\nafter ms.sample      fresh scale = {fresh.adapters[0].scale!r}")
    ok_fill = abs(fresh.adapters[0].scale - EXPECTED_SCALE) < 1e-9
    print("sample() FILLS the scale -- per-target configs are safe" if ok_fill
          else "sample() does NOT fill -- 99 of 100 targets would run un-normalised")

    # What beta now means, so the ladder is chosen in the right units.
    print(f"\nbeta bookkeeping: dimensionless beta B == raw-units beta B/{EXPECTED_SCALE**2:.4f}")
    print(f"  the known-good logP point (raw beta 25) == dimensionless "
          f"{25 * EXPECTED_SCALE ** 2:.1f}")
    return 0 if (ok_scale and ok_fill) else 1


if __name__ == "__main__":
    raise SystemExit(main())
