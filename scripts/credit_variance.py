"""Did the credit head learn per-COORDINATE credit, or just per-STATE value?

Gate 1's references are a global constant and a per-class scalar. Neither is a
per-STATE predictor, so a head that learned only "this state leads to good molecules"
passes Gate 1 while carrying nothing coordinate-specific -- and Gate 2, which removes
per-state means, then reads as null. That is the observed pattern.

Decompose the variance of the head's own output:

  between-state  variance of each state's mean log m      -> state value
  within-state   mean variance across coordinates         -> per-coordinate credit

within / total near 0 means the head is a value function wearing a credit head's shape.
"""
import argparse, torch

from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import AdapterComposition, ConditionBranch
from defog.core.credit import CreditHead, edge_mask_of
from defog.core.renoise import draw_times, renoise_states
from defog.core.rl import RolloutSampler

p = argparse.ArgumentParser()
p.add_argument("--head", required=True)
p.add_argument("--base", default="ckpts/zinc_e1_seed42_kek.ckpt")
p.add_argument("--adapter", default="ckpts/clogp_v11/clogp_adapter.ckpt")
p.add_argument("--states", type=int, default=32)
p.add_argument("--steps", type=int, default=500)
p.add_argument("--t-ints", default="125,250,375,450")
p.add_argument("--eta", type=float, default=30.0)
a = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(5)

base = DeFoGModel.load(a.base, device="cpu").to(dev).eval()
adapter = AdaLNAdapter.load(a.adapter, device=dev)
head = CreditHead.load(a.head, base, device=dev, cond_mean=[1.0], cond_std=[1.2]).eval()
cond = (torch.rand(a.states, 1, device=dev) * 4.0 - 1.0)
s = RolloutSampler(base, eta=a.eta, omega=0.0, sample_steps=a.steps,
                   time_distortion="polydec", record_trace=False)
s.composition = AdapterComposition([ConditionBranch(adapter, cond, 1.0)], base=base,
                                   mode="product")
s.sample(a.states, condition=cond, device=dev, show_progress=False)
X1, E1 = s.endpoint; nm = s.end_node_mask
em = edge_mask_of(nm)

print(f"{a.states} states, eta={a.eta:g}\n")
print(f"  {'t_int':>6s} {'t':>6s} | {'total sd':>10s} {'per-class sd':>12s} "
      f"{'beyond-class sd':>14s} {'beyond/total':>13s}")
for t_int in [int(x) for x in a.t_ints.split(",")]:
    times = draw_times(base, a.states, dev, mode="match", n_draws=1,
                       step_indices=[t_int], sample_steps=a.steps,
                       time_distortion="polydec")
    X_t, E_t, t = renoise_states(base, X1, E1,
                                 torch.zeros(a.states, 0, device=dev), nm, times)[0]
    with torch.no_grad():
        lmX, _ = head(X_t, E_t, t, nm, cond)
    # Decompose log m over real (state, coordinate, class) entries into the part a
    # per-class vector explains and the part it cannot.
    #
    # NOTE: do NOT average over the class axis to get a "state mean". The readout
    # centres pred.X on exactly that axis, so such a mean is state-independent BY
    # ALGEBRA and reports 0.00000 whatever the head learned -- a degenerate diagnostic,
    # the same shape of mistake as an A - B test where A and B are the same quantity.
    v = lmX[nm]                                   # (n_real, dx)
    per_class = v.mean(0, keepdim=True)           # the per-class component
    resid = v - per_class                         # everything beyond it
    tot_sd = float(v.std())
    pc_sd = float(per_class.squeeze(0).std())
    rs_sd = float(resid.std())
    print(f"  {t_int:6d} {float(t[0,0]):6.3f} | {tot_sd:10.5f} {pc_sd:12.5f} "
          f"{rs_sd:14.5f} {rs_sd/max(tot_sd,1e-12):13.3f}")
print("\nVAR-DONE", flush=True)
