"""Coupled DAM (Alg. 1 line 7) at restored temperature, against the matched null.

With (Y, Z) taken from one trajectory, E[a_hat] = 1 identically, so `lambda` is free
to carry temperature again instead of being pinned at 0.3 to suppress an estimator
bias. This asks whether the signal that survived at lambda=0.3 (edges -0.021/-0.051
at lr 1e-4/3e-4, uncoupled) gets larger once both suppressors are removed.

real vs null differ in ONE thing: whether Z's trajectory is the one that passed
through Y. Both satisfy E[a_hat] = 1. resid(real) - resid(null) is the signal.
"""
import statistics as st, time, torch
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
CELLS = [(1.0, 1e-4), (1.0, 3e-4), (0.3, 3e-4)]
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()

def run(lam, lr, null):
    torch.manual_seed(7)
    ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
    t = AdapterDAMTrainer(base, ad, rew, condition_sampler=cs, dam_k=8,
                          dam_lambda=lam, n_jumps=4, renoise_draws=2,
                          t_sampler="match", subsample="late", rollout_size=16,
                          sample_steps=100, minibatch_size=16, eta=1.0, omega=0.0,
                          time_distortion="polydec", lr=lr, ema_decay=None,
                          seed=42, device=dev, coupled=True, null_adjoint=null,
                          candidate_mode="simulate")
    acc = {}
    for i in range(ITERS):
        m = t.step()
        if i >= BURN:
            for k in ("drift","a_mean","a_sd","g_spread","resid_gkl_nodes",
                      "resid_gkl_edges","orc_flat","orc_state"):
                acc.setdefault(k, []).append(m[k])
    return {k: st.median(v) for k, v in acc.items()}

print(f"pre-RL base, COUPLED (Alg. 1 line 7), {ITERS} iters, median over last "
      f"{ITERS-BURN}\n  {'lam':>4s} {'lr':>6s} {'arm':5s} | {'drift':>6s} {'E[a]':>6s} "
      f"{'sd(a)':>6s} {'g_spr':>6s} | {'resid_n':>7s} {'resid_e':>7s} | "
      f"{'orc_st':>6s}", flush=True)
for lam, lr in CELLS:
    out = {}
    for null in (False, True):
        t0 = time.time(); v = out['null' if null else 'real'] = run(lam, lr, null)
        print(f"  {lam:4.1f} {lr:6.0e} {'null' if null else 'real':5s} | "
              f"{v['drift']:6.3f} {v['a_mean']:6.3f} {v['a_sd']:6.3f} "
              f"{v['g_spread']:6.3f} | {v['resid_gkl_nodes']:7.3f} "
              f"{v['resid_gkl_edges']:7.3f} | {v['orc_state']:6.3f}"
              f"   [{(time.time()-t0)/ITERS:.0f}s/it]", flush=True)
    dn = out['real']['resid_gkl_nodes'] - out['null']['resid_gkl_nodes']
    de = out['real']['resid_gkl_edges'] - out['null']['resid_gkl_edges']
    print(f"  {'':4s} {'':6s} {'DIFF':5s} |{'':29s}| {dn:+7.3f} {de:+7.3f} |"
          f"   <- negative = the adjoint helps", flush=True)
print("\nCOUPLED-SWEEP-DONE", flush=True)
