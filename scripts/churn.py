"""How permanent is an edit during DeFoG generation?

Masked diffusion's edits are irreversible by construction: a token is either masked, or
unmasked AND final. So for it, "holds its final value at step k" and "never changes
again after step k" are the SAME curve. DeFoG uses a marginal (not absorbing) noise
distribution, so a bond may go single -> double -> single. Nothing forbids it.

Two curves per step k, over real coordinates only:

  agree[k]    fraction already holding their FINAL value at step k
  settled[k]  fraction holding their final value at k AND at every step after

The GAP between them is churn -- coordinates that are momentarily right and will move
again. Masked diffusion's gap is exactly 0. A large gap that closes abruptly near t=1
is the "churns, then suddenly solidifies" picture; a gap that closes gradually is not.
"""
import torch
from defog.core import DeFoGModel
from defog.core.rl import RolloutSampler

dev = torch.device("cuda")
BS, STEPS = 16, 100
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()
torch.manual_seed(3)
s = RolloutSampler(base, eta=1.0, omega=0.0, sample_steps=STEPS,
                   time_distortion="polydec", record_trace=True, subsample_idx=None)
s.sample(BS, device=dev, show_progress=False)
X1, E1 = s.endpoint; nm = s.end_node_mask
n = nm.shape[1]
iu = torch.triu(torch.ones(n, n, device=dev, dtype=torch.bool), 1)
em = (nm[:, :, None] & nm[:, None, :]) & iu[None]

xs = [t.argmax(-1) for t in s.trace_X] + [X1.argmax(-1)]
es = [t.argmax(-1) for t in s.trace_E] + [E1.argmax(-1)]
ts = [float(t.reshape(-1)[0]) for t in s.trace_t] + [1.0]
S = len(xs)
fx, fe = xs[-1], es[-1]

agX = [ (xs[k] == fx)[nm].float().mean().item() for k in range(S) ]
agE = [ (es[k] == fe)[em].float().mean().item() for k in range(S) ]
stX, stE = [None]*S, [None]*S
okX = torch.ones_like(nm); okE = torch.ones_like(em)
for k in range(S-1, -1, -1):                      # backward: settled = agrees from k on
    okX = okX & (xs[k] == fx); okE = okE & (es[k] == fe)
    stX[k] = okX[nm].float().mean().item(); stE[k] = okE[em].float().mean().item()

flipX = sum((xs[k] != xs[k+1]).float() for k in range(S-1))[nm].mean().item()
flipE = sum((es[k] != es[k+1]).float() for k in range(S-1))[em].mean().item()
# t at which each coordinate changes for the LAST time
lastX = torch.zeros_like(fx, dtype=torch.float); lastE = torch.zeros_like(fe, dtype=torch.float)
for k in range(S-1):
    lastX = torch.where(xs[k] != xs[k+1], torch.full_like(lastX, ts[k+1]), lastX)
    lastE = torch.where(es[k] != es[k+1], torch.full_like(lastE, ts[k+1]), lastE)

print(f"{BS} molecules, {STEPS} steps, polydec, eta=1.0\n")
print(f"  {'step':>4s} {'t':>6s} | {'agree_X':>8s} {'settled_X':>9s} {'gap':>6s} "
      f"| {'agree_E':>8s} {'settled_E':>9s} {'gap':>6s}")
for k in list(range(0, S-1, 10)) + [S-6, S-4, S-2, S-1]:
    print(f"  {k:4d} {ts[k]:6.3f} | {agX[k]:8.3f} {stX[k]:9.3f} {agX[k]-stX[k]:6.3f} "
          f"| {agE[k]:8.3f} {stE[k]:9.3f} {agE[k]-stE[k]:6.3f}")
print(f"\n  mean flips per coordinate over the run:  nodes {flipX:.2f}   edges {flipE:.2f}")
for tag, v in (("nodes", lastX[nm]), ("edges", lastE[em])):
    q = torch.quantile(v, torch.tensor([0.25,0.5,0.75,0.9,0.99], device=dev))
    print(f"  t of LAST change, {tag}: p25 {q[0]:.3f}  p50 {q[1]:.3f}  p75 {q[2]:.3f}  "
          f"p90 {q[3]:.3f}  p99 {q[4]:.3f}")
print("\nCHURN-DONE", flush=True)
