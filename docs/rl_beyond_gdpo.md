# Beyond GDPO: what the literature offers for our ratchet-collapse problem

**Status: literature sweep complete, nothing built yet.** Produced by a 15-agent
workflow (5 literature scouts → triage → 4 deep-reads → 4 adversarial fit-checks →
synthesis). Companion to [`gdpo_design.md`](gdpo_design.md) and
[`RL_FINETUNING.md`](RL_FINETUNING.md). The question asked: *is there a more
promising method or extension than the GDPO we run today, given that round 1
gains, round 2 gains less, and round 3 is noise?*

## The answer in one paragraph

**Only marginally, and not from the papers.** Four methods survived triage
(Graph-GRPO, RTB, VIDD, DoMinO). **None of them fixes reward hacking**, and three
of them report a reward-vs-fidelity frontier trade of exactly the same character
as our descriptor-MMD point at β≈14. What the sweep actually produced is one
mechanism-level finding about *our own code*, which three independent fit-checks
converged on from three different papers:

> **Our KL trust region is computed in the direction that is structurally blind to
> the failure we observe, at the states where it is weakest, against a reference
> that ratchets along with the drift.**

That finding is verified, not inferred — see below. It is a ~30-line change plus a
~150-line data-anchored state pool, costs +2–3% per round, and is the only
proposal in the packet whose mechanism matches the observed failure rather than
merely constraining aggregate movement.

**But before building anything**, four diagnostics that cost zero cluster jobs can
invalidate the entire framing — and one of them raises a *third* hypothesis for
round-3 death that was not on our list of two.

---

## 1. The verified finding: the KL cannot see a mode drop

`defog/core/rl.py:141` (and the adapter twin at `:696`):

```python
klX = (pX.exp() * (pX - qX)).sum(-1)     # p = policy, q = frozen reference
```

That is `KL(policy ‖ ref)` — the **mode-seeking** direction. Wherever the policy
has driven a class probability to zero, `p·log(p/q) → 0`: the term contributes
**exactly zero**. Dropping mass the reference has is free.

Our MOSES hack *is* a mode drop. The policy discovered that aromatic
N-heterocycles are where valence failures happen and stopped making them:
aromatic rings 1.88 → 1.63 (−0.31 σ), N 2.88 → 2.60, sp3 0.35 → 0.40, while the
sanity reward rose 0.885 → 0.939 on 4/4 seeds and FCD degraded 0.863 → 1.706 on
4/4. **The trust region cannot see that by construction.** The mass-covering
direction `KL(ref ‖ policy)` can: it is unbounded exactly where the policy zeroes
a class the reference likes.

Two aggravating factors on top of the direction:

- **The states.** The term is evaluated at the policy's *own* rollout states. At
  high `t` the rollout state already carries the aliphatic composition, so a
  reference conditioned on that same partial graph also predicts an aliphatic
  continuation. The term weakens exactly as it becomes needed.
- **The reference.** The ratchet re-anchors to the previous round, so each
  re-anchor *legitimises the previous round's mode drop*. By round 3 we are
  constraining the policy to stay near a model that already contains the hack.

Note `kl_clean`'s docstring calls this "Forward KL", which in the
variational-inference convention is the opposite direction — plausibly why the
choice was never examined. (Standard PPO-based RLHF does use this direction, so
this is not a universal bug; it is a mismatch with *our specific* failure mode.)

**One caveat the fit-checks did not fully price.** This KL is on *factorized*
per-node / per-edge clean marginals. The composition part of the hack (N count,
sp3 fraction) is a genuine marginal shift and should be visible. The ring-topology
part is a joint-structure shift, and per-coordinate quantities are weak at joint
structure — which is the DAM lesson restated. Expect the flip to catch part of the
hack, not all of it. Diagnostic 0a measures how much.

---

## 2. Rank 0 — four diagnostics, today, zero rollouts

Each can kill a week of work. Run these before writing trainer code.

### 0a. The 2×2 KL diagnostic — decides whether §1 is real

Three MOSES checkpoints we already own: the frozen pre-RL base (ref), the round-1
policy (gains, FCD roughly intact), the hacked policy. Four scalars per policy
against the same frozen ref:

| | at policy rollout states | at forward-noised real training graphs |
|---|---|---|
| `KL(policy‖ref)` — today's term | **(a)** control | (d) |
| `KL(ref‖policy)` — mass-covering | (b) | **(c)** the proposal |

Reuse `renoise.draw_times` + `renoise_states` with a training pool in place of
`buf.X1/E1` (`ram.py:90` is the template). One no-grad forward pair per state.
~10 min GPU, no optimizer, no reward, no sampling for (c)/(d).

> **Availability check (done):** all three checkpoints are present locally —
> `ckpts/zinc_kek_base.ckpt`, `ckpts/zinc_rl2_seed42/`, `ckpts/moses_kek_seed44/`.
> This diagnostic is runnable today on this machine with no cluster job.

Read as a 2×2:

- **(a) small on the hacked model AND (c) large** → blindness confirmed, direction
  *and* states load-bearing → build Rank 2 in full.
- **(b) also large** → direction alone suffices → ship the one-line flip, skip the
  state pool.
- **(a) already large on the hacked model** → the trust region was never blind,
  just underweighted at `KL_COEF=0.3`. Raise it and re-run. *This outcome is live
  and would save the entire implementation.*
- **nothing separates hacked from round-1** → the hack is invisible to any
  clean-marginal divergence against this reference. The whole family dies,
  including the flip. Go back to distribution penalties on decoded samples.

### 0b. Advantage-collapse rate across rounds 1/2/3 — the third hypothesis

**The metric is already emitted**: `adv_std` and `pos_frac` at `rl.py:392-393`.
If `adv_std` is near zero at round 3, round 3 is noise *because there is no
gradient left*, and **nothing in this packet helps** — not a divergence term, not
a better estimator, not a new objective. The fix would be reward *resolution*, not
regularisation.

> **Availability check (done):** no local file contains `adv_std` — the ratchet
> runs were on JUPITER/KCIST, so this is *not* purely-local arithmetic the way 0a
> and 0c are. Either pull the round-1/2/3 stdout off the clusters, or re-derive it
> from one short re-run per round. Still cheap, but budget for the retrieval.

With Dr. GRPO we dodge the 0/0 blow-up, but the numerator still vanishes when
group rewards equalise — and the MOSES sanity reward is already 0.939 by round 2.
[He et al., *Advantage Collapse in GRPO*, arXiv:2605.21125](https://arxiv.org/abs/2605.21125)
report early-stage collapse rate strongly predicting training stagnation and final
performance across 0.5B–14B models. **Read this off existing logs first.**

### 0c. Fit Gao's overoptimisation law

`R(d) = d(α − β log d)` on the round-0/1/2 checkpoints; locate `d*`. Tells us
whether round 3 was ever recoverable. Needs the per-round reward/KL pairs, so it
shares 0b's retrieval caveat.

### 0d. Policy entropy per round

One plot. Discriminates "exploration collapsed" from "estimator noise".

**Plus, from triage's rejected pile, also free:** the WARP-style checkpoint
arithmetic — sweep the interpolation coefficient between the frozen base and the
saved round-2 weights against FCD. It may show the useful model was somewhere on
the line segment between round 0 and round 2 all along. No paper reports weight
averaging on a discrete-diffusion model, so a positive result here is a small
novel finding rather than a reproduction — which is a reason to run it and a
reason not to assume it works.

---

## 3. Rank 1 — the free protocol arm (one script change, 1.0× rollouts)

Run this as the control that every method arm below must beat. Three changes, no
new machinery:

1. **Restart from the frozen base each round**, training on accumulated rounds-1..n
   filtered data plus a real-data slice, rather than continuing from round-(n−1)
   weights. Three converging prescriptions: ReST-EM ([arXiv:2312.06585](https://arxiv.org/abs/2312.06585)),
   accumulate-don't-replace ([arXiv:2404.01413](https://arxiv.org/abs/2404.01413)),
   iterated RLHF ([arXiv:2505.18126](https://arxiv.org/abs/2505.18126)).
2. **Stop re-anchoring the KL reference to the previous round.** BOND
   ([arXiv:2407.14622](https://arxiv.org/abs/2407.14622)) measured per-round
   re-anchoring as strictly worse than an EMA anchor. Mechanistically it is worse
   than "worse" — see §1.
3. **Fix `kl_coef`; turn off the adaptive controller** (`rl.py:384-386`). Gao et
   al. §3.6: *"The KL penalty only causes the gold RM score to converge earlier,
   but does not affect the KL_RL–gold reward frontier, and so the effect of the
   penalty on the gold score is akin to early stopping."* Deleu et al.
   ([arXiv:2509.01632](https://arxiv.org/abs/2509.01632)) reach the same place
   from the RTB↔Trust-PCL equivalence: an adaptive KL coefficient is an adaptive
   *temperature*, i.e. it moves the target during training — the exact failure we
   already diagnosed for the moving reference.

**If this alone restores a usable round 3, most of the rest of this document is
unnecessary.** One 70-minute round.

---

## 4. Rank 2 — anchored, mass-covering trust region (~1.5–2 days, +2–3% compute)

The one worth building, conditional on 0a. Two edits to the KL term; the policy
gradient is untouched.

1. **Flip the direction.** `rl.py:141-142` and `:690-691`:
   `(pX.exp() * (pX - qX))` → `(qX.exp() * (qX - pX))`. Reference forward stays
   `no_grad` (only the `−Σ q log p` half carries gradient). **Add a log-prob floor**
   (clamp `log_softmax` to ≈ −20): the mass-covering direction is unbounded above
   when the policy zeroes a class, and one saturated coordinate can dominate a
   batch gradient. Log both directions side by side and re-measure `kl_target`
   before re-enabling the controller.
2. **Move the states off the rollout onto a fixed data anchor.** Evaluate at
   forward-noised *real training graphs* — a fixed pool of dense
   `(X1, E1, node_mask)` re-noised each iteration through `p_{t|1}(·|G1)`.
   Structurally this is `AdapterRAMTrainer._renoised_states` with `buf.X1/E1`
   replaced by a fixed pool. Add `anchor_loss()` to `RLTrainerBase`, one call per
   `update()`. The PG term stays at rollout states.
3. **Anchor to the pretraining data, not the previous round**, held fixed across
   all rounds. Start with the hard-label variant (CE against the true one-hot G1
   of the noised training graph — no reference forward at all). Be clear-eyed:
   that variant is literally the pretraining loss, i.e. replay / anti-forgetting.
   That is not a reason to skip it; it is a reason not to pay for the ref forward
   first. The soft-target variant (DoMinO proper) is one no-grad forward more.

**Why this ranks above the papers it came from.** Three independent fit-checks —
DoMinO's, VIDD's, and a rejected candidate's — converged on the same edit from
different derivations. DoMinO itself evaluates at *policy rollout states*, which
for our failure mode defeats the point (§1). **Take the regulariser, reject the
framework, change the states.**

### Silent-failure risks, in order of nastiness

1. **Gradient-inert term that still logs a plausible number.** Reverse KL has a
   constant `Σ q log q` piece; detach the wrong branch or `.exp()` the wrong side
   and the reported `kl` stays healthy while the term contributes nothing. Same
   class of bug as the void η-sweep. **Mandatory guard:** unit test asserting
   nonzero grad norm attributable to the anchor term alone, plus a weight-diff
   check with the PG term zeroed.
2. **Adaptive controller retarget** (`rl.py:384`) — same `kl_target`, different
   measured scale, `kl_coef` walks somewhere unintended. Freeze it first.
3. **Reduction mismatch.** The anchor is a second place to get `"mean"` wrong, and
   the two terms now live on different state sets with different node-count
   distributions. Match the anchor pool's size distribution to the rollout's, or
   the anchor will quietly fight our learned `P(n|target)` size conditioning. Log
   mean heavy-atom count on both sides.
4. **Pool leakage.** Keep the anchor pool disjoint from the FCD/scaffold eval
   reference split, or the fidelity gain is partly memorisation.

**Honest expected value:** ~60% it sits on a better reward-vs-FCD frontier than
descriptor-MMD (it acts on model marginals rather than decoded descriptors, so the
hack cannot ride the dominant descriptor variance axes the way it defeated our
Mahalanobis attempt), 25% a wash, 15% worse. Separately ~30% that it extends the
useful ratchet past round 2. Plausibly **stackable** with MMD, since the two
penalise different projections. Not a free lunch: DoMinO's own Table 2 costs 5–9
points of the held-out functional metric in all four regularised arms, and one arm
made the fidelity metric *worse*.

---

## 5. Rank 3–5 — the papers, in order of what they actually buy us

### Rank 3 — Graph-GRPO's exact kernel, as a data-reuse primitive

[Zhu, Bo, Zhang, Wang, *Graph-GRPO: Training Graph Flow Models with Reinforcement
Learning*, arXiv:2603.10395, ICML 2026](https://arxiv.org/abs/2603.10395) — the
only methodological descendant of GDPO, and it targets DeFoG itself.

Buys an exact per-step transition log-likelihood and a valid importance ratio,
which we do not currently have. **Stage A** is a prerequisite, not a deliverable
(reuse `dam.py::rate_basis`/`marginal_rate`, add `allow_omega`, gradient-safe
`step_kernel` avoiding the autograd-killing in-place `scatter_`). **Stage B is the
actual reason to do it:** a PPO ratio over a windowed step subset with
`num_inner_epochs` 2–4, reusing one rollout batch for several gradient epochs.
That is the compute lever the paper leaves entirely untested (`num_inner_epochs:
1` in every shipped config) and the only one that matches a rollout-dominated
70-minute budget.

**Do not build Stage A as a trust region.** `R` is affine in `p_θ` and the kernel
is `O(dt)` from a point mass, so kernel KL is likely an attenuated
reparameterisation of the marginal KL we already penalise. Test first: if
`kl_kernel/kl_clean` is roughly constant across the good ZINC policy and the
hacked MOSES policy, it is the same constraint in different units.

**Hard gate before any kernel work:** `_compute_step_probs` fills the diagonal
with `(1 − rowsum).clamp(min=0.0)` and does **not** renormalise. Our MOSES config
is η=200/ω=0.5, far outside the η=25 regime measured as safe. Refuse at every
`(t, dt)` if max off-diagonal row-sum of `R·dt` > 0.9, in the style of the FK
threshold gate we already shipped. Also assert `composition.blend_space == "prob"`
(`model.py:1002`) — under `"rate"` the closed form is simply wrong and produces
plausible numbers, breaking the `behavior == scored` invariant.

The paper's own evidence reproduces our frontier trade rather than escaping it:
against DeFoG's published numbers (their Appendix Table 9, not the de-tuned
50-step main-text baseline with η and ω deleted), Tree gains 1.0 V.U.N. while the
distribution Ratio worsens 1.6 → 2.2, and **Planar loses 4.5 V.U.N.**

### Rank 4 — RTB: run the 25-minute audit, do not build the trainer until it passes

[Venkatraman et al., *Amortizing intractable inference in diffusion models*,
NeurIPS 2024, arXiv:2405.20971](https://arxiv.org/abs/2405.20971).

The only candidate whose *fixed point* differs: `p_base(G)·exp(R(G)/β)/Z`, a
normalisable distribution, so a round that finds nothing further is terminal
behaviour rather than degradation. The ratchet stops being a concept.

**A correction in our favour.** Exact per-step transition log-probs are *not*
available as the extraction claimed — `RateMatrixDesigner.compute_rate_matrices`
(`rate_matrix.py:104`) draws `sample_from_probs(X_1_pred, E_1_pred)` internally and
builds `R_t` from that hard draw. But under our shipped `rdb='general'`, every
coordinate's rate depends on φ **only through that coordinate's own x1 draw**, so
on the augmented path the CTMC factor is φ-independent and **cancels exactly**:

```
Σ_t [log p_φ(G_{t+dt}|G_t) − log p_b(G_{t+dt}|G_t)] == Σ_t [log q_φ(x1_t|G_t) − log p_b(x1_t|G_t)]
```

That reduces the RTB path log-ratio to a per-step version of `_score_logprob`
(`rl.py:679`) — no kernel machinery, no second forward pass, and in the adapter
setting the prior branch is already computed by `_base_uncond_softmax`
(`rl.py:649`).

**The audit.** One rollout batch, ~20–25 min, recorder only. K=128 at the deployed
config (T=500, η=25, polydec); per trajectory compute `L = A − B` (full 500-step
x1-draw log-ratio) and `R`. Report `mean(L)`, `sd(L)`, `β·sd(R)`,
`corr(L, n_nodes)`. Decision rule **stated in advance**:

- `sd(L) ≤ ~2·β·sd(R)` → residual carries real reward signal at our length. Proceed.
- `sd(L) > ~5·β·sd(R)` → the reward is below the noise floor of the trust-region
  term; every gradient coefficient is driven by KL estimation noise. **Stop.**
- `|corr(L, n_nodes)| > ~0.5` → a scalar `log Z` is disqualified; size-conditioned
  `log Z(n)` is mandatory.

**I expect it to fail.** Δ accumulates 500 steps × ~820 coordinates ≈ 4×10⁵
stochastic log-ratio terms. At a per-coordinate log-ratio sd of 0.01 nats the
per-trajectory spread is ≈ 6 nats; at 0.05 it is ≈ 32 nats. `β·R` spans O(1).
Nobody in this literature has run RTB above T≈300 and there is no length ablation
anywhere. Δ sits inside a square, so the step subsampling that makes our eager PG
cheap is **not valid** for RTB — it biases the residual itself, not just the
gradient.

**Severity note that separates RTB from everything else here.** Under GDPO a
behaviour/scored mismatch rescales an advantage — the run gets worse but still
points roughly the right way, and weight-diff and paired-eval probes catch it.
Under RTB the same mismatch **biases Δ, and a biased Δ moves the fixed point**:
the run converges cleanly and confidently to the wrong distribution with a healthy
loss curve. No metric in our current harness would flag it.

**And it does not fix reward hacking.** If R is hackable, an unbiased tilted
posterior *puts mass on the hack in proportion to the hack's probability under the
base*. Our MOSES hack is simultaneously high-reward and entirely typical under the
base — it sits squarely inside the tilted posterior. RTB finds it *more*
faithfully, not less. Their own single discrete experiment hacked its reward model
(SEDD repetitions) despite the unbiased fixed point.

### Rank 5 — VIDD's per-step value baseline: probe first, predicted dead

[Su, Li, Uehara et al., *Iterative Distillation for Reward-Guided Fine-Tuning of
Diffusion Models in Biomolecular Design*, arXiv:2507.00445](https://arxiv.org/abs/2507.00445).

Half-day offline probe, no gradient steps: on one existing rollout batch, for each
subsampled trace state, argmax-decode `x̂₀(x_t)`, strip virtual classes, call the
same reward. Record per `t`: (i) fraction decoding to a connected valid molecule,
(ii) Spearman corr across k between `r(x̂₀(x_t))` and `r(x₀)`, (iii) across-k std
of `v̂`.

**Predicted outcome:** at low `t` the clean-marginal argmax is the dataset marginal
mode — carbon nodes, no-bond edges — decoding to isolated carbons, disconnected, a
flat **−4 for every rollout**. So `v̂` is constant across k over most of the
trajectory, the naive advantage `r_k − v̂_{k,t}` is *uncentered* ≈ `r_k + 4`, and
that destroys exactly the Dr. GRPO centering we deliberately use. At high `t`,
`v̂ → r(x₀)` and `A → 0`. The informative band is a narrow middle window whose
existence is an empirical question.

If the probe *does* show a band with corr > ~0.3, the correct form is **not**
`r_k − v̂_{k,t}` but `A_{k,t} = (r_k − v̂_{k,t}) − mean_k(...)` within the CRN group
at that `t` — a strict generalisation collapsing to today's Dr. GRPO wherever `v̂`
is constant across k, which is what makes it low-risk. Guard: assert
`mean_k A_{k,t} ≈ 0` for every `t`.

---

## 6. What we should NOT do

**Do not port Graph-GRPO's adaptive prior.** It is a deliberate, unregularised
drift generator — their Figure 6 is our MOSES hack engineered on purpose (carbon
0.74 → 0.84, N 0.125 → 0.08, node counts collapsing to a spike at 25–27). Shifting
`p₀` away from the data marginal *is* a global composition shift. They report no
FCD, no scaffold diversity, no drift metric; the string "FCD" does not occur in
the paper.

**Do not port Graph-GRPO's refinement loop.** Inference-only, never touches a
gradient, and we already have `renoise.py` + `feynman_kac.py` in that family.

**Do not port VIDD as configured.** `gkd_lmbda=1.0` in 3 of 4 headline scripts
means zero frozen-base roll-in — the teacher is a lagged student refreshed every K
steps, which is structurally the moving KL reference we already measured as inert,
plus an iterated tilt whose fixed point is `exp((S/K)·r/α)`: unbounded sharpening
with no anchor. Their own fidelity column agrees: 3-mer Corr 0.162, **last among
every fine-tuning baseline**, diversity 0.90 → 0.52. Every reported number is at
the argmax-over-training checkpoint selected on the *optimised* reward, with the
held-out-oracle selection line commented out, so a late fidelity collapse cannot
appear in any table. Their one graph result is GDSS on ZINC with base FCD 12.979 —
13 units of fidelity headroom, so reward and fidelity move together. Our MOSES base
is FCD/TestSF 0.8158.

**Do not implement VIDD's Eq-(7) transition-level KL or DoMinO's inner-MDP
estimator.** Both need gradient through `R* + ηR^DB + ωR^TG`, including
`_stabilize`'s hard zeroing of rates > 1e5 (`rate_matrix.py:390-407`). New
plumbing, and it makes the RL result a function of η/ω. UDM-GRPO's own ablation
says the clean-endpoint action beats the step-wise one (0.89 vs 0.84) in a
non-masked model — our current design is already the right one.

**Do not use DoMinO's gKL variant.** DeFoG's rate field has structural zeros from
the ReLU and `_stabilize`, so `D_gKL` is +∞ without an epsilon floor the paper
never mentions, and it drags η/ω into the penalty for nothing.

**Never set λ from DoMinO's Theorems 6.2/6.3.** The TV bound is vacuous at our
scale — at 500 steps the linear path's `1/(1−t)` alone gives `exp(M_u) ~ 1e217`
before η=25 multiplies it by `e²⁵` — and the tractable factorized version assumes
the terminal law factorizes across coordinates, i.e. no molecular structure.
Consistency statement only.

**Do not touch the masked-dLLM RL line at all** — d1/diffu-GRPO, wd1, SPG, MDPO,
DDPP, TraFL, DRAKES, D2-DPO, Diffusion-DPO. The entire methodological content of
that literature is inventing surrogates for an intractable per-step
log-likelihood. **We have it in closed form.** Porting those methods is a strict
downgrade. wd1's headline contribution ("reformulate the RL objective as a
weighted log-likelihood") is a *description of our current code*. This is the
mistake we have already paid for twice.

**Do not do reward ensembles / WARM / pessimism on the sanity reward.** The
mechanism is epistemic uncertainty in a *learned* reward; ours is a deterministic
RDKit oracle with zero epistemic uncertainty, so worst-case-over-ensemble and
mean−kσ both degenerate to the single reward. Our "error" is that
valid+connected+rings_ok genuinely does not penalise dropping aromatic
N-heterocycles — no ensemble of that reward would disagree. **WARM does apply,
cheaply, on the learned-`PropertyHead` path**: train 3–5 heads varying seed and
data order from the same trunk, average the *weights* (inference cost unchanged).
A few hours. It will not help with the MOSES hack.

**Do not expect any of the four papers to fix reward hacking.** Graph-GRPO
mentions it once, asserts the KL prevents it, never measures it. VIDD's Appendix G
shows 10% multiplicative reward noise halves their gains. DoMinO has one task, one
dataset, no seeds, no λ sweep, no curves, no iteration count, no code. RTB's own
discrete experiment hacked its reward model. **Reward hacking here is a
reward-specification problem, and nothing in the shortlist is a
reward-specification contribution.**

**Two traps already paid for and live in every port here:** verify against
`defog/core/rate_matrix.py`, not `src/flow_matching/rate_matrix.py` (one extractor
checked the legacy tree); and assert `blend_space == "prob"` explicitly rather
than assuming the default, since `"rate"` silently voids the closed-form and
cancellation arguments in Ranks 3 and 4 while producing plausible numbers.

---

## 7. The honest uncertainty

### What the literature did not answer — four gaps, all ours to fill

1. **Nobody publishes a multi-round ratchet with round-3 diagnostics.** Graph-GRPO
   does single-round plus refinement, reporting only monotone improvement. VIDD
   reports iterations but no saturation curve. DoMinO never states its iteration
   count. Our round-3 observation appears to be unpublished territory in this
   subfield.
2. **Nobody measures FCD, scaffold diversity, or any composition-drift metric
   while reporting RL reward gains on molecular graphs.** Graph-GRPO reports V.U.N.
   and docking only, and would very likely reproduce our MOSES failure if anyone
   measured FCD on its outputs. **Our instrumentation is ahead of the published
   work.**
3. **Nobody has calibrated any trajectory-ratio method at T=500.** RTB tops out
   around T=300 with no length ablation; dFlowGRPO trains at K=8 steps; UDM-GRPO
   optimises ~3 high-noise steps. The field handles trajectory length by not using
   the full trajectory.
4. **No RL on graph discrete flow matching exists beyond GDPO itself.** The
   Semantic Scholar citation graph for arXiv:2402.16302 returns 24 citing papers,
   overwhelmingly wireless-networking topology applications. Graph-GRPO is the only
   methodological descendant.

### Where we are extrapolating

DoMinO is a v1 April-2026 preprint with no code, validated on regulatory DNA
sequences only. **The mass-covering argument in §1 is mechanism, not evidence** —
the code direction is verified and the mechanism follows, but nobody has run it on
a molecular graph. The competing explanation: our low-`t` KL already bites (at very
low `t` the states are near-pure noise and hence policy-independent, so the term
*is* already state-anchored there) and is simply being outvoted on coefficient — in
which case Rank 2 is a state-reweighting and descriptor-MMD at β≈14 already
occupies that frontier. Diagnostic 0a distinguishes these. That is why it comes
first.

### Is the ratchet ceiling our reward or our base model, not the RL algorithm?

The right question — and there are **three** hypotheses, not two. Distinguishing
tests, cheapest first:

- **H1: reward specification.** The MOSES evidence already points here — the
  optimiser was working correctly and found the composition shift. Test: hold out a
  reward we never optimise (FCD, or a second property head trained on disjoint
  data) and plot it against the optimised reward per round. Monotone fall from
  round 1 while the proxy rises ⇒ specification, and no estimator, leash, or
  objective in this packet is the fix. Combine with 0c. *If H1 confirms, the
  correct next investment is the constraint / tail-shaping family — dual ascent on
  an FCD or composition constraint replacing the hand-tuned β≈14
  ([arXiv:2310.04373](https://arxiv.org/abs/2310.04373)). Those are reward-side
  wrappers that compose with whatever inner loop we run, which is why triage
  deferred them; if H1 confirms, they stop being deferrable.*
- **H2: advantage exhaustion.** Test 0b, from logs on disk **today**. If `adv_std`
  → 0 at round 3, the entire shortlist is the wrong tool and the fix is reward
  *resolution*, not regularisation.
- **H3: base-model / adapter capacity.** We already have one supporting data point
  — the prob-space blending work found the low-logP end to be a capacity wall, not
  an optimisation failure. Test: run round 3 with a *fresh, unhacked* reward on the
  same base. If it also dies at round 3 while `adv_std` is healthy, the ceiling is
  the frozen base or the adapter rank. Second discriminator: round 3 on the adapter
  vs on the full model — if the full model keeps moving and the adapter does not,
  it is adapter capacity, and the fix is a wider adapter, not a new objective.

**H2 is checkable from logs on disk today, and if true, the entire shortlist is
the wrong tool.** That is the single strongest reason to run Rank 0 before
committing to anything.

---

## Citation status

All five load-bearing arXiv IDs were fetched and verified after the workflow
returned: 2603.10395 (Graph-GRPO, ICML 2026), 2604.06491 (DoMinO, 7 Apr 2026),
2405.20971 (RTB, NeurIPS 2024), 2509.01632 (RTB≡Trust-PCL), 2605.21125 (Advantage
Collapse / AVSPO). Two corrections to the workflow's own output: VIDD's authors are
**Xingyu Su, Xiner Li, Masatoshi Uehara et al.** (the scout's list was wrong and
flagged itself for checking), and Deleu et al. show KL-regularized RL performs
*comparably* on RTB's illustrative example — they do not call the earlier result a
reward-design error.

See [[dam-postmortem]] for the previous method that failed, and the pattern it
established: *a broken measurement does not look broken, it looks like a result.*
Every Rank-0 diagnostic above exists to falsify a mechanism before we build on it.
