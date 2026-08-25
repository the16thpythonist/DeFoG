"""How much of the reward tilt survives into the only channel DeFoG has?

DAM's target posterior is  p*(x1|xt) ~ p_base(x1|xt) * exp(-g(x1)).  But DeFoG's rate
reads ONE coordinate's MARGINAL at a time, so the entire instruction the model can
ever receive is the difference between the base marginals and the tilted marginals.

Draw K endpoints from x_t, weight them by exp(-g), and compare:

  E[r] base vs tilted   how much the reweighting improves the molecules (the JOINT)
  marginal shift        how much that moves per-coordinate marginals (the CHANNEL)
  shuffled control      the same weights permuted across samples, so they carry the
                        same spread but no relationship to the molecules -- the noise
                        floor for a shift of this weight spread

If the tilt clearly improves the reward while the marginal shift is at the shuffled
floor, the reward's information does not fit through the channel and no estimator,
temperature or sample count can help. Weights are recomputed from the same endpoints
for each lambda, so the lambda sweep is free.
"""
import statistics as st, torch
from defog.core import DeFoGModel, AdaLNAdapter, AdapterDAMTrainer
from defog.core.renoise import draw_times, renoise_states
from defog.core.rl import PropertyMatchReward, RolloutSampler
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

dev = torch.device("cuda")
_, adec, _, bdec = build_encoders(["C","N","O","F","P","S","Cl","Br","I"],
                                  ["SINGLE","DOUBLE","TRIPLE"])
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

for t_int in (75, 50):
    times = draw_times(base, BS, dev, mode="match", n_draws=1, step_indices=[t_int],
                       sample_steps=100, time_distortion="polydec")
    X_t, E_t, t = renoise_states(base, X1, E1, y0, nm, times)[0]
    zx, ze, gs = [], [], []
    with torch.no_grad():
        done = 0
        while done < K:
            r = min(CHUNK // BS, K - done)
            aX, aE = tr._simulate_endpoints(X_t, E_t, nm, cond, t_int, r)
            zx.append(aX.view(r, BS, *aX.shape[1:])); ze.append(aE.view(r, BS, *aE.shape[1:]))
            gs.append(tr._terminal_loss(aX, aE, nm.repeat(r,1), cond.repeat(r,1)).view(r, BS))
            done += r
    ZX = torch.cat(zx, 0); ZE = torch.cat(ze, 0); G1 = torch.cat(gs, 0)   # (K,BS,...)
    r_all = -G1 / 1.0                                          # lambda=1.0 in the trainer

    iu = torch.triu(torch.ones(nm.shape[1], nm.shape[1], device=dev, dtype=torch.bool), 1)
    em = (nm[:,:,None] & nm[:,None,:]) & iu[None]
    print(f"\n=== t_int={t_int}  (t={float(t[0,0]):.3f}) ===")
    print(f"  {'lam':>4s} {'ESS':>6s} | {'E[r]base':>9s} {'E[r]tilt':>9s} {'gain':>7s} "
          f"| {'node dTV':>9s} {'shuf':>7s} | {'edge dTV':>9s} {'shuf':>7s}")
    for lam in (0.3, 1.0, 3.0):
        w = torch.softmax(lam * r_all, dim=0)                  # (K,BS); exp(-g)=exp(lam*r)
        ess = float(((w.sum(0)**2) / (w**2).sum(0)).mean())
        gain = float((w * r_all).sum(0).mean() - r_all.mean(0).mean())
        row = []
        for wt in (w, w[torch.randperm(K, device=dev)]):       # real, then shuffled
            mbX = ZX.mean(0); mtX = (wt[:,:,None,None] * ZX).sum(0)
            mbE = ZE.mean(0); mtE = (wt[:,:,None,None,None] * ZE).sum(0)
            dX = 0.5 * (mtX - mbX).abs().sum(-1)[nm]
            dE = 0.5 * (mtE - mbE).abs().sum(-1)[em]
            row += [float(dX.mean()), float(dE.mean())]
        print(f"  {lam:4.1f} {ess:6.1f} | {float(r_all.mean()):+9.4f} "
              f"{float((w*r_all).sum(0).mean()):+9.4f} {gain:+7.4f} | "
              f"{row[0]:9.4f} {row[2]:7.4f} | {row[1]:9.4f} {row[3]:7.4f}", flush=True)
print("\nMARGINAL-DONE", flush=True)
