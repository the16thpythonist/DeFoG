"""Does the -0.039 node difference replicate across seeds?

The coupled sweep's strongest cell (lambda=0.3, lr=3e-4, simulate) rests on ONE seed,
and the channel pattern across the six cells was inconsistent -- nodes best here,
edges best in the uncoupled head-mode run. Three conclusions in this project have
already been drawn from too few points and overturned. This runs the same cell at two
further seeds before anything is concluded from it.
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
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()

def run(seed, null):
    torch.manual_seed(seed)
    ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
    t = AdapterDAMTrainer(base, ad, rew, condition_sampler=cs, dam_k=8,
                          dam_lambda=0.3, n_jumps=4, renoise_draws=2,
                          t_sampler="match", subsample="late", rollout_size=16,
                          sample_steps=100, minibatch_size=16, eta=1.0, omega=0.0,
                          time_distortion="polydec", lr=3e-4, ema_decay=None,
                          seed=seed, device=dev, coupled=True, null_adjoint=null,
                          candidate_mode="simulate")
    acc = {}
    for i in range(ITERS):
        m = t.step()
        if i >= BURN:
            for k in ("drift","a_mean","resid_gkl_nodes","resid_gkl_edges"):
                acc.setdefault(k, []).append(m[k])
    return {k: st.median(v) for k, v in acc.items()}

print("lambda=0.3, lr=3e-4, coupled, simulate -- seed replicates\n"
      f"  {'seed':>4s} {'arm':5s} | {'drift':>6s} {'E[a]':>6s} | {'resid_n':>7s} "
      f"{'resid_e':>7s}", flush=True)
dns, des = [], []
for seed in (7, 43, 91):        # 7 reproduces the sweep cell exactly
    out = {}
    for null in (False, True):
        v = out['null' if null else 'real'] = run(seed, null)
        print(f"  {seed:4d} {'null' if null else 'real':5s} | {v['drift']:6.3f} "
              f"{v['a_mean']:6.3f} | {v['resid_gkl_nodes']:7.3f} "
              f"{v['resid_gkl_edges']:7.3f}", flush=True)
    dn = out['real']['resid_gkl_nodes'] - out['null']['resid_gkl_nodes']
    de = out['real']['resid_gkl_edges'] - out['null']['resid_gkl_edges']
    dns.append(dn); des.append(de)
    print(f"  {'':4s} {'DIFF':5s} |{'':15s}| {dn:+7.3f} {de:+7.3f}", flush=True)
print(f"\n  nodes: mean {st.mean(dns):+.4f}  sd {st.stdev(dns):.4f}  -> "
      f"{'REPLICATES' if max(dns) < 0 else 'DOES NOT REPLICATE (sign flips)'}")
print(f"  edges: mean {st.mean(des):+.4f}  sd {st.stdev(des):.4f}  -> "
      f"{'REPLICATES' if max(des) < 0 else 'DOES NOT REPLICATE (sign flips)'}")
print("\nSEEDREP-DONE", flush=True)
