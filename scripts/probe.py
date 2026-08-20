"""Sub-rollout throughput vs batch rows, to size the n_z run before committing 12h."""
import time, torch
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
BS = 16
def cs(): return (torch.rand(BS,1)*4.0-1.0), torch.zeros(BS, dtype=torch.long)
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()
ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
tr = AdapterDAMTrainer(base, ad, rew, condition_sampler=cs, dam_k=8, dam_lambda=0.3,
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
idx = tr._choose_subsample() or [61]
t_int = idx[0]
times = draw_times(base, BS, dev, mode="match", n_draws=1, step_indices=[t_int],
                   sample_steps=100, time_distortion="polydec")
X_t, E_t, t = renoise_states(base, X1, E1, y0, nm, times)[0]
print(f"subsample='late' -> step {t_int}, t={float(t[0,0]):.3f}, "
      f"{100-t_int} steps per sub-rollout\n")
K = 8
Xk, Ek = X_t.repeat(K,1,1), E_t.repeat(K,1,1,1)
nmk, ck = nm.repeat(K,1), cond.repeat(K,1)
print(f"  {'rows':>6s} {'sec':>7s} {'sec/row':>8s}")
base_rows = K*BS
for reps in (1, 2, 4):
    torch.cuda.synchronize(); t0 = time.time()
    try:
        sX, sE = tr._simulate_endpoints(Xk, Ek, nmk, ck, t_int, reps)
        torch.cuda.synchronize(); el = time.time()-t0
        n = base_rows*reps
        print(f"  {n:6d} {el:7.1f} {el/n*1000:8.2f} ms", flush=True)
    except RuntimeError as e:
        print(f"  {base_rows*reps:6d}  FAILED: {str(e)[:60]}", flush=True); break
print("\nPROBE-DONE", flush=True)
