#!/usr/bin/env python
"""Can the property head RANK ten particles? The gate before spending a job on beta~900.

The FK beta sweep showed a large logP gain (MAE 0.5420 -> 0.4308 at beta=25) and nothing
at all on QED. The mechanical reason is scale: the energy is an UN-NORMALISED squared error
in the property's own units (property_head.py:172), FK turns it straight into weights
(feynman_kac.py:213,262,265), and molsmith deliberately skips the scale normalisation for a
single adapter (sample.py:610-613). logP squared errors are ~37x larger than QED's, so at a
shared beta the QED weights stay within a few percent of uniform and the ESS gate never fires.

Raising QED's beta to ~900 restores the pressure. Whether that HELPS is a second question,
and this script answers it cheaply: FK does not need calibrated accuracy, it needs to pick
which of ten particles is closest to the target. If the head cannot do that among molecules
whose true values differ by the observed within-target spread, then more pressure just
resamples on noise and MAE gets worse.

logP is the positive control, not a curiosity: it is the one property where FK is KNOWN to
work, so its selection efficiency is the reference level that "good enough" means.

UPPER BOUND, not an estimate. The head here scores CLEAN validation molecules. In a real run
it scores partially-denoised MAP-decoded graphs, which is strictly harder. A property that
fails here fails in FK; passing here does not guarantee passing there.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/media/ssd2/Programming/DeFoG")
sys.path.insert(0, "/media/ssd2/Programming/defog-web")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, QED  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from defog.core.property_head import PropertyHead  # noqa: E402
from defog.domains.molecule import build_encoders, smiles_to_pyg_data  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-media-ssd2-Programming-DeFoG/"
               "6dfca3f5-92f5-4143-b270-cb575ca66202/scratchpad")
HOME = Path.home()

# zinc-kek schema, from the base package metadata.
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE"]        # 'none' is class 0, not an encoder entry

PROPS = {
    "logp": dict(
        fn=lambda m: float(Crippen.MolLogP(m)),
        head=HOME / ".molsmith/packages/e939dee7f22d3fd6/weights/head.safetensors",
        cfg=dict(na=9, nb=4, hid=128, layers=3,
                 prop_mean=2.8247015476226807, prop_std=1.1579052209854126),
        # within-target sd and |bias| measured on the adapter-only arm of job 43072
        sd=0.5478, bias=0.2573, adapter_mae=0.5420,
    ),
    "qed": dict(
        fn=lambda m: float(QED.qed(m)),
        head=SCRATCH / "qed310_head.safetensors",
        cfg=dict(na=9, nb=4, hid=128, layers=3,
                 prop_mean=0.7466652393341064, prop_std=0.13275757431983948),
        sd=0.0924, bias=0.0317, adapter_mae=0.0920,
    ),
}


def load_head(path, cfg, device):
    from molsmith.weights.convert import read_safetensors
    state_dict, _ = read_safetensors(path, device="cpu")
    return PropertyHead.from_config(cfg, state_dict, device=device)


def encode(smiles_list, ae, be, device):
    """Encode the way the head was TRAINED: kekulized, because the zinc-kek bond vocabulary
    is {SINGLE, DOUBLE, TRIPLE} with no AROMATIC class.

    This deliberately differs from what LearnedPropertyEnergy does at sampling time, which
    omits ``kekulize`` and therefore rejects ~94% of real molecules outright (measured in
    fk_energy_path_check.py). That is a separate defect; measuring the head's ranking ability
    on the 6% that slip through would measure the defect, not the head."""
    from torch_geometric.data import Batch

    from defog.core.data import to_dense
    ok, datas = [], []
    for i, smi in enumerate(smiles_list):
        try:
            d = smiles_to_pyg_data(smi, ae, be, kekulize=True)
        except Exception:                                            # noqa: BLE001
            d = None
        if d is not None and getattr(d, "x", None) is not None:
            ok.append(i)
            datas.append(d)
    if not datas:
        return np.array([], dtype=int), None
    return np.array(ok), datas


def predict_all(head, datas, device, chunk=256):
    """Chunked: the dense (B, N, N, 4) edge tensor is what blows up, not the head."""
    from torch_geometric.data import Batch

    from defog.core.data import to_dense
    out = []
    for i in range(0, len(datas), chunk):
        batch = Batch.from_data_list(datas[i:i + chunk]).to(device)
        dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        dense = dense.mask(mask)
        with torch.no_grad():
            out.append(head.predict(dense.X, dense.E, mask).reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, _, be, _ = build_encoders(ATOM_TYPES, BOND_TYPES)

    from defog.data import zinc_reference as zref
    pool = zref.load_reference_split().val_smiles
    rng = np.random.default_rng(0)
    smis = [pool[int(i)] for i in rng.permutation(len(pool))[:4000]]
    print(f"device={device}  validation molecules drawn: {len(smis)}\n")

    for name, P in PROPS.items():
        head = load_head(P["head"], P["cfg"], device)
        truth = np.array([P["fn"](Chem.MolFromSmiles(s)) for s in smis])

        # --- encoding gate: does the FK path even accept these molecules? -------------
        idx, datas = encode(smis, ae, be, device)
        rate = len(idx) / len(smis)
        if datas is None:
            print(f"### {name.upper()}: ENCODING TOTALLY FAILS.")
            continue
        pred = predict_all(head, datas, device)
        truth = truth[idx]

        err = pred - truth
        rmse = float(np.sqrt((err ** 2).mean()))
        from scipy.stats import spearmanr
        rho = float(spearmanr(pred, truth).statistic)

        print(f"### {name.upper()}   (head from the shipped bundle)")
        print(f"  encoded OK                     {rate:6.1%}"
              + ("   <-- aromatic rejection would show up here" if rate < 0.99 else ""))
        print(f"  head RMSE                      {rmse:.4f}")
        print(f"  head Spearman rho              {rho:.4f}")
        print(f"  within-target sd it must beat  {P['sd']:.4f}")
        print(f"  RMSE / sd  (>1 = ranking noise){rmse / P['sd']:>7.2f}")

        # --- the decision FK actually makes ------------------------------------------
        # Build a realistic particle cloud: ten REAL molecules whose true values sit where
        # the adapter's ten actually sit (mean offset = bias, spread = sd), then ask the head
        # to pick the closest to the target. Selecting by head vs by truth vs not at all.
        order = np.argsort(truth)
        sorted_truth = truth[order]
        n_trials, K = 3000, 10
        no_sel, head_sel, oracle_sel, top1 = [], [], [], []
        for _ in range(n_trials):
            t = float(rng.choice(truth))
            want = rng.normal(t + rng.choice([-1, 1]) * P["bias"], P["sd"], size=K)
            pick = order[np.clip(np.searchsorted(sorted_truth, want), 0, len(order) - 1)]
            tv, pv = truth[pick], pred[pick]
            no_sel.append(np.abs(tv - t).mean())
            head_sel.append(abs(tv[np.argmin(np.abs(pv - t))] - t))
            oracle_sel.append(np.abs(tv - t).min())
            top1.append(int(np.argmin(np.abs(pv - t)) == np.argmin(np.abs(tv - t))))
        no_sel, head_sel, oracle_sel = map(np.mean, (no_sel, head_sel, oracle_sel))
        eff = (no_sel - head_sel) / (no_sel - oracle_sel)

        print(f"  --- the pick-1-of-10 decision FK makes ---")
        print(f"  no selection      |err|        {no_sel:.4f}")
        print(f"  select by HEAD    |err|        {head_sel:.4f}")
        print(f"  select by TRUTH   |err|        {oracle_sel:.4f}   (ceiling)")
        print(f"  top-1 hit rate                 {np.mean(top1):6.1%}   (chance 10.0%)")
        print(f"  SELECTION EFFICIENCY           {eff:6.1%}   "
              f"(0% = head is useless, 100% = perfect)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
