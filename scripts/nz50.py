"""Does averaging more continuations per edit amplify the adjoint's signal?

snr.py measured the shortfall: resolving ONE edit's effect on the final score takes
~21 continuations at t=0.978 and ~38 at t=0.75; the training config sits at t=0.938,
so ~21-25. The estimator uses ONE. This runs n_z = 1 (baseline), 10 (under the
estimate) and 50 (over it), each against its matched null.

Prediction being tested: a slight improvement at 10 and a clear one at 50, with the
paired difference growing roughly as sqrt(n_z) until it saturates.

n_z=1 reproduces the cell whose three-seed baseline is edges -0.026 +- 0.010.
"""
import json, statistics as st, time, torch
from defog.core import DeFoGModel, AdaLNAdapter, AdapterDAMTrainer
from defog.core.data import dense_to_pyg
from defog.core.rl import PropertyMatchReward
from defog.domains.molecule import build_encoders, mol_to_smiles, pyg_data_to_mol
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

dev = torch.device("cuda")
_, adec, _, bdec = build_encoders(["C","N","O","F","P","S","Cl","Br","I"],
                                  ["SINGLE","DOUBLE","TRIPLE"])
rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)
NCOND = [16]
def cs(): return (torch.rand(NCOND[0],1)*4.0-1.0), torch.zeros(NCOND[0], dtype=torch.long)

CELLS = [(50, 6)]                              # n_z=1,10 already in nz_sweep.json
BURN = 2
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()

def build(n_z, null):
    torch.manual_seed(7)
    ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
    return AdapterDAMTrainer(base, ad, rew, condition_sampler=cs, dam_k=8,
                             dam_lambda=0.3, n_jumps=4, renoise_draws=2,
                             t_sampler="match", subsample="late", rollout_size=16,
                             sample_steps=100, minibatch_size=16, eta=1.0, omega=0.0,
                             time_distortion="polydec", lr=3e-4, ema_decay=None,
                             seed=42, device=dev, coupled=True, null_adjoint=null,
                             candidate_mode="simulate", n_z=n_z, sub_chunk_rows=128)

def logp_eval(tr, tag, n=64):
    """Outcome measure: reward and true-logP error on molecules the policy generates."""
    NCOND[0] = n; tr.rollout_size = n
    try:
        buf = tr.rollout()
    finally:
        NCOND[0] = 16; tr.rollout_size = 16
    errs = []
    tgt = buf.y.reshape(n, -1)[:, 0].tolist()
    for i, d in enumerate(dense_to_pyg(buf.X1, buf.E1, None, buf.node_mask,
                                       buf.node_mask.sum(-1))):
        m = pyg_data_to_mol(d, adec, bdec)
        smi = mol_to_smiles(m) if m else None
        mm = Chem.MolFromSmiles(smi) if smi else None
        if mm is not None and "." not in smi:
            errs.append(abs(float(Crippen.MolLogP(mm)) - tgt[i]))
    out = {"tag": tag, "reward": float(buf.reward.mean()), "n_valid": len(errs),
           "logp_mae": (st.mean(errs) if errs else float("nan"))}
    print(f"  LOGP[{tag:11s}] reward {out['reward']:+.4f} | valid {len(errs)}/{n} "
          f"| logP MAE {out['logp_mae']:.4f}", flush=True)
    return out

print("pre-RL base, coupled, lambda=0.3, lr=3e-4, K=8, minibatch 16\n"
      f"  {'n_z':>4s} {'it':>3s} {'arm':5s} | {'drift':>6s} {'E[a]':>6s} {'sd(a)':>6s} "
      f"| {'resid_n':>7s} {'resid_e':>7s} {'noop':>7s} | {'s/it':>6s}", flush=True)
results = json.load(open("nz_sweep.json"))["diffs"]; evals = []
for n_z, iters in CELLS:
    out = {}
    for null in (False, True):
        tr = build(n_z, null)
        if n_z == CELLS[-1][0] and not null:
            evals.append(logp_eval(tr, "untrained"))
        acc, t0 = {}, time.time()
        for i in range(iters):
            m = tr.step()
            if i >= BURN:
                for k in ("drift","a_mean","a_sd","resid_gkl_nodes","resid_gkl_edges","noop_mag"):
                    acc.setdefault(k, []).append(m[k])
        v = out['null' if null else 'real'] = {k: st.median(x) for k, x in acc.items()}
        print(f"  {n_z:4d} {iters:3d} {'null' if null else 'real':5s} | {v['drift']:6.3f} "
              f"{v['a_mean']:6.3f} {v['a_sd']:6.3f} | {v['resid_gkl_nodes']:7.3f} "
              f"{v['resid_gkl_edges']:7.3f} {v['noop_mag']:7.4f} | {(time.time()-t0)/iters:6.0f}", flush=True)
        if n_z == CELLS[-1][0] and not null:
            evals.append(logp_eval(tr, f"n_z={n_z} real"))
    dn = out['real']['resid_gkl_nodes'] - out['null']['resid_gkl_nodes']
    de = out['real']['resid_gkl_edges'] - out['null']['resid_gkl_edges']
    results[n_z] = {"nodes": dn, "edges": de}
    print(f"  {'':4s} {'':3s} {'DIFF':5s} |{'':22s}| {dn:+7.3f} {de:+7.3f} |"
          f"   <- negative = the adjoint helps", flush=True)
    json.dump({"diffs": results, "evals": evals}, open("nz_sweep50.json","w"), indent=1)

print("\n  trend (edges):  " + "  ".join(f"n_z={k}: {v['edges']:+.3f}"
                                          for k, v in results.items()))
print("  trend (nodes):  " + "  ".join(f"n_z={k}: {v['nodes']:+.3f}"
                                        for k, v in results.items()))
print("\nNZ-SWEEP-DONE", flush=True)
