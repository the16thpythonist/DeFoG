"""Paired real/null at matched LR, plus the drift->resid curve.

power.py showed resid < 1 at lr=1e-4 (0.998/0.979) but its null arm was pinned at
lr=1e-3, so the one cell that mattered had no control. This pairs every LR.

Read: at each LR, `real` and `null` differ ONLY in whether the adjoint carries the
value-function difference (null draws Z at X_t, so the true adjoint is 1). Same seed,
same RNG consumption, same picks. resid(real) - resid(null) at matched drift is the
signal, and it is the only number in this project that has ever isolated it.
"""
import statistics as st, torch
from defog.core import DeFoGModel, AdaLNAdapter, AdapterDAMTrainer
from defog.core.rl import PropertyMatchReward
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

dev = torch.device("cuda")
_, adec, _, bdec = build_encoders(["C","N","O","F","P","S","Cl","Br","I"],
                                  ["SINGLE","DOUBLE","TRIPLE"])
rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)
def cs(): return (torch.rand(16,1)*4.0-1.0), torch.zeros(16, dtype=torch.long)

ITERS, BURN = 12, 4
LRS = [3e-5, 1e-4, 3e-4]
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()

def run(lr, null):
    torch.manual_seed(7)
    ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
    t = AdapterDAMTrainer(base, ad, rew, condition_sampler=cs, dam_k=12,
                          dam_lambda=0.3, n_jumps=8, renoise_draws=4,
                          t_sampler="match", subsample="late", rollout_size=16,
                          sample_steps=100, minibatch_size=16, eta=1.0, omega=0.0,
                          time_distortion="polydec", lr=lr, ema_decay=None,
                          seed=42, device=dev, null_adjoint=null)
    acc = {}
    for i in range(ITERS):
        m = t.step()
        if i >= BURN:
            for k in ("drift","a_mean","resid_gkl_nodes","resid_gkl_edges",
                      "orc_flat","orc_state"):
                acc.setdefault(k, []).append(m[k])
    return {k: st.median(v) for k, v in acc.items()}

print(f"pre-RL base, {ITERS} iters, median over last {ITERS-BURN}\n"
      f"  {'lr':>6s} {'arm':5s} | {'drift':>6s} {'E[a]':>6s} | {'resid_n':>7s} "
      f"{'resid_e':>7s} | {'orc_fl':>6s} {'orc_st':>6s}", flush=True)
for lr in LRS:
    out = {}
    for null in (False, True):
        v = out['null' if null else 'real'] = run(lr, null)
        print(f"  {lr:6.0e} {'null' if null else 'real':5s} | {v['drift']:6.3f} "
              f"{v['a_mean']:6.3f} | {v['resid_gkl_nodes']:7.3f} "
              f"{v['resid_gkl_edges']:7.3f} | {v['orc_flat']:6.3f} "
              f"{v['orc_state']:6.3f}", flush=True)
    dn = out['real']['resid_gkl_nodes'] - out['null']['resid_gkl_nodes']
    de = out['real']['resid_gkl_edges'] - out['null']['resid_gkl_edges']
    dd = out['real']['drift'] / max(out['null']['drift'], 1e-9)
    print(f"  {'':6s} {'DIFF':5s} | grad x{dd:5.2f} |  {dn:+7.3f} {de:+7.3f} |"
          f"   <- real minus null; negative = the adjoint helps", flush=True)
print("\nLRPAIR-DONE", flush=True)
