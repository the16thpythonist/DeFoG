"""How big is one edit's effect, against the noise of everything that follows?

DAM's adjoint asks "does edit Y lead to better molecules?" and answers it by running
the generation forward and scoring the result. But each continuation makes hundreds
more random choices, so the measurement is

    (effect of edit Y)  +  (accumulated randomness of the rest of the trajectory)

This measures both terms directly, and then measures what COMMON RANDOM NUMBERS do to
the second one: the continuations from x and from y are run with the SAME random
stream (identical seed, identical batch shape, so row i of each side draws the same
uniforms), which cancels shared downstream randomness in the difference.

Reports, per (state, edit):
  effect        mean g(from y) - mean g(from x)
  sd_unpaired   spread of a single unpaired comparison
  sd_paired     spread of the matched difference
  n             samples needed to see the effect at 1 sigma, each way

If n_unpaired is in the hundreds and n_paired in the single digits, the diagnosis is
variance and DAM is fine. If sd_paired stays large, the edit genuinely does not matter
and no estimator rescues it.
"""
import statistics as st, torch
from defog.core import DeFoGModel, AdaLNAdapter, AdapterDAMTrainer
from defog.core.dam import (_base_uncond_softmax, rate_basis, marginal_rate,
                            sample_jump)
from defog.core.renoise import draw_times, renoise_states
from defog.core.rl import PropertyMatchReward, RolloutSampler
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

dev = torch.device("cuda")
_, adec, _, bdec = build_encoders(["C","N","O","F","P","S","Cl","Br","I"],
                                  ["SINGLE","DOUBLE","TRIPLE"])
rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)
BS, N, EDITS = 4, 24, 3
base = DeFoGModel.load("ckpts/zinc_e1_seed42_kek.ckpt", device="cpu").to(dev).eval()
ad = AdaLNAdapter.load("ckpts/clogp_v11/clogp_adapter.ckpt", device=dev)
def cs(): return (torch.rand(BS,1)*4.0-1.0), torch.zeros(BS, dtype=torch.long)
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
cond = torch.zeros(BS, 1, device=dev)          # fixed target, so g varies only with G
y0 = torch.zeros(BS, 0, device=dev)

print(f"{BS} states x {EDITS} edits, {N} continuations per side\n")
for t_int in (85, 50):
    times = draw_times(base, BS, dev, mode="match", n_draws=1, step_indices=[t_int],
                       sample_steps=100, time_distortion="polydec")
    X_t, E_t, t = renoise_states(base, X1, E1, y0, nm, times)[0]
    with torch.no_grad():
        puX, puE, noisy, extra = _base_uncond_softmax(base, X_t, E_t, t, nm)
        polX, polE = tr._composed(tr.adapter, noisy, extra, nm, cond, puX, puE)
        BX, BE = rate_basis(base, X_t, E_t, t, nm, eta=1.0)
        uX, uE = marginal_rate(polX.exp(), polE.exp(), BX, BE)

    print(f"=== t_int={t_int}  (t={float(t[0,0]):.3f}, {100-t_int} steps remain) ===")
    print(f"  {'edit':>4s} | {'effect':>7s} {'sd_unpair':>9s} {'sd_paired':>9s} "
          f"{'var.red':>7s} | {'n_unpair':>8s} {'n_paired':>8s} {'same_end':>8s}")
    ru, rp = [], []
    for e in range(EDITS):
        with torch.no_grad():
            _, pick, X_Y, E_Y = sample_jump(uX, uE, X_t, E_t, nm)
            torch.manual_seed(500 + e)
            aX, aE = tr._simulate_endpoints(X_t, E_t, nm, cond, t_int, N)
            torch.manual_seed(500 + e)                       # <-- common random numbers
            bX, bE = tr._simulate_endpoints(X_Y, E_Y, nm, cond, t_int, N)
            nmr, cr = nm.repeat(N, 1), cond.repeat(N, 1)
            g_x = tr._terminal_loss(aX, aE, nmr, cr).view(N, BS)
            g_y = tr._terminal_loss(bX, bE, nmr, cr).view(N, BS)
            same = float(((aX == bX).all(-1).all(-1) &
                          (aE == bE).all(-1).all(-1).all(-1)).float().mean())
        for j in range(BS):
            d = (g_y[:, j] - g_x[:, j])
            eff = float(d.mean())
            su = float((g_x[:, j].var() + g_y[:, j].var()).sqrt())
            sp = float(d.std())
            if abs(eff) < 1e-9:
                continue
            nu, npd = (su / abs(eff)) ** 2, (sp / abs(eff)) ** 2
            ru.append(nu); rp.append(npd)
        print(f"  {e:4d} | {eff:+7.3f} {su:9.3f} {sp:9.3f} {(su/max(sp,1e-9))**2:7.1f} |"
              f" {nu:8.0f} {npd:8.0f} {same:8.1%}", flush=True)
    print(f"   median over {len(ru)} (state, edit) pairs:  "
          f"n_unpaired {st.median(ru):8.0f}   n_paired {st.median(rp):8.0f}   "
          f"variance reduction {st.median(ru)/max(st.median(rp),1e-9):.1f}x\n", flush=True)
print("SNR-DONE", flush=True)
