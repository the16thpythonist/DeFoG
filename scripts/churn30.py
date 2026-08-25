"""Churn at REALISTIC sampling settings (eta=30, 500 steps) + a GIF to check by eye.

The earlier run used eta=1.0 / 100 steps and concluded edits were near-permanent. Eta
is the detailed-balance stochasticity term -- the knob that controls back-and-forth
jumping -- so that measurement said little about real sampling.

agree[k]   = fraction of coordinates already holding their FINAL value at step k
settled[k] = fraction holding it at k AND every step after
The gap is churn. Masked diffusion's gap is 0 by construction.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio, numpy as np, networkx as nx, torch
from defog.core import DeFoGModel
from defog.core.rl import RolloutSampler

dev = torch.device("cuda")
BS, STEPS, ETA = 8, 500, 30.0
ATOMS = ["C","N","O","F","P","S","Cl","Br","I"]
COL = {"C":"#909090","N":"#3050F8","O":"#FF0D0D","F":"#90E050","P":"#FF8000",
       "S":"#D0D030","Cl":"#1FF01F","Br":"#A62929","I":"#940094"}
base = DeFoGModel.load("/media/ssd2/Programming/DeFoG/ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()
torch.manual_seed(3)
s = RolloutSampler(base, eta=ETA, omega=0.0, sample_steps=STEPS,
                   time_distortion="polydec", record_trace=True, subsample_idx=None)
s.sample(BS, device=dev, show_progress=False)
X1, E1 = s.endpoint; nm = s.end_node_mask
n = nm.shape[1]
iu = torch.triu(torch.ones(n, n, device=dev, dtype=torch.bool), 1)
em = (nm[:, :, None] & nm[:, None, :]) & iu[None]
xs = [t.argmax(-1).cpu() for t in s.trace_X] + [X1.argmax(-1).cpu()]
es = [t.argmax(-1).cpu() for t in s.trace_E] + [E1.argmax(-1).cpu()]
ts = [float(t.reshape(-1)[0]) for t in s.trace_t] + [1.0]
S = len(xs); nmc, emc = nm.cpu(), em.cpu()
fx, fe = xs[-1], es[-1]

agX=[(xs[k]==fx)[nmc].float().mean().item() for k in range(S)]
agE=[(es[k]==fe)[emc].float().mean().item() for k in range(S)]
stX,stE=[0]*S,[0]*S; okX=torch.ones_like(nmc); okE=torch.ones_like(emc)
for k in range(S-1,-1,-1):
    okX &= (xs[k]==fx); okE &= (es[k]==fe)
    stX[k]=okX[nmc].float().mean().item(); stE[k]=okE[emc].float().mean().item()
flipX=sum((xs[k]!=xs[k+1]).float() for k in range(S-1))[nmc].mean().item()
flipE=sum((es[k]!=es[k+1]).float() for k in range(S-1))[emc].mean().item()
print(f"{BS} molecules, {STEPS} steps, polydec, ETA={ETA}\n")
print(f"  {'step':>4s} {'t':>6s} | {'agree_X':>8s} {'settled_X':>9s} {'gap':>6s} "
      f"| {'agree_E':>8s} {'settled_E':>9s} {'gap':>6s}")
for k in list(range(0,S-1,50))+[S-20,S-10,S-3,S-1]:
    print(f"  {k:4d} {ts[k]:6.3f} | {agX[k]:8.3f} {stX[k]:9.3f} {agX[k]-stX[k]:6.3f} "
          f"| {agE[k]:8.3f} {stE[k]:9.3f} {agE[k]-stE[k]:6.3f}")
print(f"\n  mean flips per coordinate:  nodes {flipX:.2f}   edges {flipE:.2f}")
print(f"  total changes per molecule per step: "
      f"{(flipX*int(nmc[0].sum())+flipE*int(emc[0].sum()))/STEPS:.2f}", flush=True)

# ---- GIF -----------------------------------------------------------------
G = nx.Graph(); G.add_nodes_from(range(n))
for i in range(n):
    for j in range(i+1, n):
        if emc[0,i,j] and fe[0,i,j] > 0: G.add_edge(i,j)
pos = nx.spring_layout(G, seed=7, k=0.55, iterations=250)
frames, idx = [], list(range(0, S, 5)) + [S-1]*8
for k in idx:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.6), dpi=88)
    for b, ax in enumerate(axes.ravel()):
        live = [i for i in range(n) if nmc[b,i]]
        for i in live:
            for j in live:
                if j <= i: continue
                o = int(es[k][b,i,j])
                if o == 0: continue
                (x1,y1),(x2,y2) = pos[i], pos[j]
                ax.plot([x1,x2],[y1,y2], lw=[0,1.6,3.4,5.0][o], color="#333",
                        solid_capstyle="round", zorder=1)
        ax.scatter([pos[i][0] for i in live], [pos[i][1] for i in live],
                   c=[COL[ATOMS[int(xs[k][b,i])]] for i in live], s=170,
                   edgecolors="#222", linewidths=0.8, zorder=2)
        ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25); ax.axis("off")
    fig.suptitle(f"DeFoG sampling   eta={ETA:g}, {STEPS} steps   "
                 f"step {k}/{S-1}   t = {ts[k]:.3f}", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
imageio.mimsave("/tmp/claude-1000/-media-ssd2-Programming-DeFoG/7776b2f4-26e7-4b70-8ca5-2414801d1a1c/scratchpad/defog_sampling_eta30.gif", frames, duration=0.09, loop=0)
print(f"\n  wrote defog_sampling_eta30.gif  ({len(frames)} frames)")
print("\nCHURN30-DONE", flush=True)
