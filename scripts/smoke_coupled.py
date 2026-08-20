import time, torch
from defog.core import DeFoGModel, AdaLNAdapter, AdapterDAMTrainer
from defog.core.rl import PropertyMatchReward
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen
dev=torch.device("cuda")
_,adec,_,bdec = build_encoders(["C","N","O","F","P","S","Cl","Br","I"],["SINGLE","DOUBLE","TRIPLE"])
rew = PropertyMatchReward(adec,bdec,lambda m: float(Crippen.MolLogP(m)),scale=1.0)
def cs(): return (torch.rand(16,1)*4.0-1.0), torch.zeros(16,dtype=torch.long)
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt",device="cpu").to(dev).eval()
for coup, lam in ((False,0.3),(True,0.3),(True,1.0),(True,3.0)):
    torch.manual_seed(7)
    ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt",device=dev)
    t = AdapterDAMTrainer(base,ad,rew,condition_sampler=cs,dam_k=8,dam_lambda=lam,
        n_jumps=4,renoise_draws=2,t_sampler="match",subsample="late",rollout_size=16,
        sample_steps=100,minibatch_size=16,eta=1.0,omega=0.0,time_distortion="polydec",
        lr=1e-4,ema_decay=None,seed=42,device=dev,coupled=coup,
        candidate_mode="simulate")
    t0=time.time(); ms=[t.step() for _ in range(3)]; el=(time.time()-t0)/3
    a=lambda k: sum(m[k] for m in ms)/3
    print(f"coupled={str(coup):5s} lam={lam}: E[a]={a('a_mean'):6.3f} sd(a)={a('a_sd'):6.3f} "
          f"g_spread={a('g_spread'):6.3f} resid_n={a('resid_gkl_nodes'):6.3f} "
          f"resid_e={a('resid_gkl_edges'):6.3f} | {el:.1f}s/it", flush=True)
print("SMOKE-DONE", flush=True)
