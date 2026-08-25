"""Is it the PROPERTY that blocks DAM, or the SHAPE of the reward?

Same 256 completions from the same states, scored three ways:

  logp-match   -|logP - target|      a target on a SUM over the molecule  -> combinations
  oxy-max      #oxygens              exp(lam*r) FACTORISES exactly        -> positions
  oxy-match    -|#oxygens - mean|    a target on a SUM again              -> combinations

oxy-max vs oxy-match is the controlled contrast: the SAME chemical quantity, once as a
per-atom preference and once as a constraint on its total. If the shape is what matters,
oxy-max moves the marginals hard and oxy-match barely moves them at all -- even though
both change the joint by a comparable amount.

Every reward is standardised to unit sd across the 256 samples, so lambda means "tilt by
one standard deviation" for all three and the tilt strengths are comparable (check ESS).
Endpoints are simulated ONCE; scoring is free.
"""
import torch
from defog.core import DeFoGModel, AdaLNAdapter, AdapterDAMTrainer
from defog.core.data import dense_to_pyg
from defog.core.renoise import draw_times, renoise_states
from defog.core.rl import PropertyMatchReward, RolloutSampler
from defog.domains.molecule import build_encoders, mol_to_smiles, pyg_data_to_mol
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

dev = torch.device("cuda")
ATOMS = ["C","N","O","F","P","S","Cl","Br","I"]
OXY = ATOMS.index("O")
_, adec, _, bdec = build_encoders(ATOMS, ["SINGLE","DOUBLE","TRIPLE"])
rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)
BS, K, CHUNK = 4, 256, 128
def cs(): return (torch.rand(BS,1)*4.0-1.0), torch.zeros(BS, dtype=torch.long)
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()
ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
tr = AdapterDAMTrainer(base, ad, rew, condition_sampler=cs, dam_k=8, dam_lambda=1.0,
                       n_jumps=4, renoise_draws=2, t_sampler="match", subsample="late",
                       rollout_size=BS, sample_steps=100, minibatch_size=BS, eta=1.0,
                       omega=0.0, time_distortion="polydec", lr=0.0, ema_decay=None,
                       seed=42, device=dev, coupled=True, candidate_mode="simulate")
torch.manual_seed(11)
s = RolloutSampler(base, eta=1.0, omega=0.0, sample_steps=100,
                   time_distortion="polydec", record_trace=False)
s.sample(BS, device=dev, show_progress=False)
X1, E1 = s.endpoint; nm = s.end_node_mask
y0 = torch.zeros(BS,0,device=dev); cond = torch.zeros(BS,1,device=dev)
t_int = 75
times = draw_times(base, BS, dev, mode="match", n_draws=1, step_indices=[t_int],
                   sample_steps=100, time_distortion="polydec")
X_t, E_t, t = renoise_states(base, X1, E1, y0, nm, times)[0]

zx, ze, gl = [], [], []
with torch.no_grad():
    done = 0
    while done < K:
        r = min(CHUNK // BS, K - done)
        aX, aE = tr._simulate_endpoints(X_t, E_t, nm, cond, t_int, r)
        zx.append(aX.view(r, BS, *aX.shape[1:])); ze.append(aE.view(r, BS, *aE.shape[1:]))
        gl.append(tr._terminal_loss(aX, aE, nm.repeat(r,1), cond.repeat(r,1)).view(r, BS))
        done += r
ZX = torch.cat(zx,0); ZE = torch.cat(ze,0); G = torch.cat(gl,0)      # (K,BS,...)
noxy = (ZX[..., OXY] * nm[None]).sum(-1)                             # (K,BS)
REW = {"logp-match": -G,                                             # lam=1 in trainer
       "oxy-max":    noxy,
       "oxy-match":  -(noxy - noxy.mean(0, keepdim=True)).abs()}

iu = torch.triu(torch.ones(nm.shape[1], nm.shape[1], device=dev, dtype=torch.bool), 1)
em = (nm[:,:,None] & nm[:,None,:]) & iu[None]
print(f"t={float(t[0,0]):.3f}, K={K}, {BS} states.  base #oxygens "
      f"{float(noxy.mean()):.2f} +- {float(noxy.std()):.2f}\n")
print(f"  {'reward':11s} {'lam':>4s} {'ESS':>6s} | {'gain(sd)':>8s} | {'node dTV':>9s} "
      f"{'shuf':>7s} {'ratio':>6s} | {'d p(O)':>8s} | {'edge dTV':>9s} {'shuf':>7s} {'ratio':>6s}")
for name, raw in REW.items():
    r_s = (raw - raw.mean(0, keepdim=True)) / raw.std(0, keepdim=True).clamp_min(1e-8)
    for lam in (1.0, 3.0):
        w = torch.softmax(lam * r_s, dim=0)
        ess = float(((w.sum(0)**2)/(w**2).sum(0)).mean())
        gain = float((w * r_s).sum(0).mean())
        vals = []
        for wt in (w, w[torch.randperm(K, device=dev)]):
            mbX, mtX = ZX.mean(0), (wt[:,:,None,None]*ZX).sum(0)
            mbE, mtE = ZE.mean(0), (wt[:,:,None,None,None]*ZE).sum(0)
            vals.append((float((0.5*(mtX-mbX).abs().sum(-1))[nm].mean()),
                         float((0.5*(mtE-mbE).abs().sum(-1))[em].mean()),
                         float((mtX-mbX)[..., OXY][nm].mean())))
        (dX, dE, dO), (sX, sE, _) = vals
        print(f"  {name:11s} {lam:4.1f} {ess:6.1f} | {gain:+8.3f} | {dX:9.4f} {sX:7.4f} "
              f"{dX/max(sX,1e-9):5.2f}x | {dO:+8.4f} | {dE:9.4f} {sE:7.4f} "
              f"{dE/max(sE,1e-9):5.2f}x", flush=True)
print("\nDECOMP-DONE", flush=True)
