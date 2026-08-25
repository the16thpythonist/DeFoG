"""Gate 3: does guided sampling produce better molecules?

The only measurement in this whole investigation that speaks to whether any of it is
useful. Everything else -- resid, coherence, reliability -- is a training-internal
diagnostic.

Arms are a guidance-strength sweep with a shared seed per arm, so the comparison is
paired. `scale=0` is the control and is BIT-IDENTICAL to unguided sampling (asserted in
tests/test_credit.py), which is what makes it a control rather than an approximation.

Reported per arm: logP MAE against the conditioning target on decodable molecules,
validity, and uniqueness. Validity is reported alongside because a guidance strength
that improves MAE by wrecking chemistry is not a result.

Baseline on record (pre-RL base, 6-iteration DAM run): MAE 0.6730, 54/64 decode.
"""
import argparse, json, statistics as st, torch

from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import AdapterComposition, ConditionBranch
from defog.core.credit import CreditHead, CreditGuidance
from defog.core.data import dense_to_pyg
from defog.core.rl import RolloutSampler
from defog.domains.molecule import build_encoders, mol_to_smiles, pyg_data_to_mol
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

ATOMS = ["C","N","O","F","P","S","Cl","Br","I"]; BONDS = ["SINGLE","DOUBLE","TRIPLE"]


def evaluate(base, adapter, head, cond, scale, args, dev, adec, bdec, seed):
    torch.manual_seed(seed)
    comp = AdapterComposition([ConditionBranch(adapter, cond, 1.0)], base=base,
                              mode="product")
    guide = None if head is None else CreditGuidance(head, cond, scale=scale)
    # RolloutSampler, not Sampler: it stashes the terminal one-hot in the NETWORK's
    # class space via _post_loop, before ignore_virtual_classes strips it -- which is
    # the space dense_to_pyg/pyg_data_to_mol decode from. It forwards **kwargs to
    # Sampler, so posterior_transform reaches the denoise loop unchanged.
    s = RolloutSampler(base, eta=args.eta, omega=0.0, sample_steps=args.steps,
                       time_distortion="polydec", posterior_transform=guide,
                       record_trace=False)
    s.composition = comp
    s.sample(cond.shape[0], condition=cond, device=dev, show_progress=False)
    X1, E1 = s.endpoint; nm = s.end_node_mask
    tgt = cond[:, 0].tolist()
    errs, smis = [], []
    for i, d in enumerate(dense_to_pyg(X1, E1, None, nm, nm.sum(-1))):
        m = pyg_data_to_mol(d, adec, bdec)
        smi = mol_to_smiles(m) if m else None
        mm = Chem.MolFromSmiles(smi) if smi else None
        if mm is not None and "." not in smi:
            errs.append(abs(float(Crippen.MolLogP(mm)) - tgt[i])); smis.append(smi)
    n = cond.shape[0]
    return {"scale": scale, "seed": seed, "n": n, "valid": len(errs),
            "validity": len(errs)/n, "unique": len(set(smis))/max(len(smis),1),
            "logp_mae": (st.mean(errs) if errs else float("nan"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="")
    p.add_argument("--base", default="ckpts/zinc_e1_seed42_kek.ckpt")
    p.add_argument("--adapter", default="ckpts/clogp_v11/clogp_adapter.ckpt")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eta", type=float, default=30.0)
    p.add_argument("--scales", default="0,0.5,1.0,2.0")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = DeFoGModel.load(a.base, device="cpu").to(dev).eval()
    adapter = AdaLNAdapter.load(a.adapter, device=dev)
    head = None
    if a.head:
        head = CreditHead.load(a.head, base, device=dev,
                               cond_mean=[1.0], cond_std=[1.2]).eval()
    _, adec, _, bdec = build_encoders(ATOMS, BONDS)

    scales = [float(x) for x in a.scales.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"Gate 3  |  n={a.n}/arm, eta={a.eta:g}, {a.steps} steps, "
          f"seeds={seeds}\n")
    print(f"  {'scale':>6s} {'seed':>5s} | {'MAE':>7s} {'validity':>9s} {'unique':>7s}")
    rows = []
    for scale in scales:
        for seed in seeds:
            torch.manual_seed(1000 + seed)              # SAME conditions across arms
            cond = (torch.rand(a.n, 1, device=dev) * 4.0 - 1.0)
            r = evaluate(base, adapter, head, cond, scale, a, dev, adec, bdec, seed)
            rows.append(r)
            print(f"  {scale:6.2f} {seed:5d} | {r['logp_mae']:7.4f} "
                  f"{r['validity']:9.3f} {r['unique']:7.3f}", flush=True)
        sel = [x for x in rows if x["scale"] == scale]
        m = [x["logp_mae"] for x in sel]
        v = [x["validity"] for x in sel]
        print(f"  {scale:6.2f} {'MEAN':>5s} | {st.mean(m):7.4f} {st.mean(v):9.3f}"
              f"   sd(MAE) {st.stdev(m) if len(m)>1 else 0:.4f}", flush=True)
    ctrl = st.mean([x["logp_mae"] for x in rows if x["scale"] == 0.0])
    best = min((st.mean([x["logp_mae"] for x in rows if x["scale"] == s]), s)
               for s in scales if s != 0.0) if len(scales) > 1 else (ctrl, 0.0)
    print(f"\n  control (scale=0) MAE {ctrl:.4f}   best guided {best[0]:.4f} "
          f"at scale {best[1]:g}   delta {best[0]-ctrl:+.4f}")
    if a.out:
        json.dump({"rows": rows, "control": ctrl, "best": best, "args": vars(a)},
                  open(a.out, "w"), indent=1)
    print("GATE3-DONE", flush=True)


if __name__ == "__main__":
    main()
