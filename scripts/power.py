"""Does `resid` measure direction, or step size?

Four arms per base, identical in every respect except the two knobs that separate
the two readings:

  lr=0    real adjoint  -- the policy cannot move. If resid is a drift meter this
                           must print EXACTLY 1.000.
  lr=1e-4 real adjoint  -- production LR (adapter_rl_finetune__zinc.py:185)
  lr=1e-3 real adjoint  -- the LR every probe to date actually used
  lr=1e-3 NULL adjoint  -- Z drawn at X_t instead of X_Y, so the true adjoint is
                           identically 1. Any signal reported here is noise.

If arm 4 is indistinguishable from arm 3, the target carries no learnable signal
and every resid number recorded so far is uninformative.
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

ARMS = [("lr=0     real", 0.0,  False),
        ("lr=1e-4  real", 1e-4, False),
        ("lr=1e-3  real", 1e-3, False),
        ("lr=1e-3  NULL", 1e-3, True)]
BASES = [("pre-RL ", "ckpts/zinc_e1_seed42_kek.ckpt"),
         ("shipped", "ckpts/zinc_kek_base.ckpt")]

HDR = (f"  {'arm':14s} | {'drift':>6s} {'E[a]':>6s} {'sd(a)':>6s} | "
       f"{'resid_n':>7s} {'resid_e':>7s} | {'orc_fl':>6s} {'orc_st':>6s} | "
       f"{'g_spr':>6s} {'log_a':>6s}")

for tag, bck in BASES:
    base = DeFoGModel.load(bck, device="cpu").to(dev).eval()
    print(f"\n=== {tag} ===\n{HDR}", flush=True)
    for label, lr, null in ARMS:
        torch.manual_seed(7)
        ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
        t = AdapterDAMTrainer(base, ad, rew, condition_sampler=cs, dam_k=12,
                              dam_lambda=0.3, n_jumps=8, renoise_draws=4,
                              t_sampler="match", subsample="late", rollout_size=16,
                              sample_steps=100, minibatch_size=16, eta=1.0, omega=0.0,
                              time_distortion="polydec", lr=lr, ema_decay=None,
                              seed=42, device=dev, null_adjoint=null)
        acc = {}
        for i in range(8):
            m = t.step()
            if i >= 2:
                for k in ("drift","a_mean","a_sd","resid_gkl_nodes","resid_gkl_edges",
                          "orc_flat","orc_state","g_spread","log_adjoint"):
                    acc.setdefault(k, []).append(m[k])
        v = {k: st.median(x) for k, x in acc.items()}
        print(f"  {label:14s} | {v['drift']:6.3f} {v['a_mean']:6.3f} {v['a_sd']:6.3f} | "
              f"{v['resid_gkl_nodes']:7.3f} {v['resid_gkl_edges']:7.3f} | "
              f"{v['orc_flat']:6.3f} {v['orc_state']:6.3f} | "
              f"{v['g_spread']:6.3f} {v['log_adjoint']:+6.3f}", flush=True)
print("\nPOWER-DONE", flush=True)
