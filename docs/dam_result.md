> **CLOSED.** This line of work is finished -- see [`dam_postmortem.md`](dam_postmortem.md) for the verdict and a
> plain-terms summary of every measurement. This file is the detailed record of the DAM arm
> and is kept as the appendix.

# DAM on DeFoG: measured outcome

**Status:** the earlier negative result is **withdrawn**. It was produced by a metric
whose best attainable value is 1.0 and a learning rate 10x production. Re-measured
against a matched null control, the DAM update carries a small but real directional
signal. Whether it *steers* is still untested.
**Plan:** [`dam_design.md`](dam_design.md) · **Code:** `defog/core/dam.py`, branch `feat/dam-rl`

## What went wrong with the first conclusion

Three faults, compounding. None is in the method; all three are in how it was measured.

### 1. `resid` cannot go below 1 unless the target has learnable signal

`dam.py:689-710` builds `target = u_base * exp(log_a)` from an adjoint redrawn every
iteration, then scores the policy and the base against that same draw and takes the
ratio. `u_base` cancels, so `resid` is a function of policy drift and target noise
only. Verified in closed form and on the real model:

| adjoint | resid at `u_theta = u_base` | best achievable |
|---|---|---|
| pure noise, mean 1, sd 0.2-1.0 | 1.0000 | **1.0000** |

When the adjoint carries no state-dependent signal, **staying at the base is optimal**
and `resid = 1.000` is the CEILING. The pass mark was set at "below 1". It was
unreachable by construction, and drift enters quadratically:

| drift sd | 0.05 | 0.10 | 0.20 | 0.30 | 0.45 |
|---|---|---|---|---|---|
| resid (signal-free target) | 1.006 | 1.023 | 1.094 | 1.214 | 1.482 |

Every previously reported cell -- 1.057, 1.099, 1.200, 1.379, 1.615, 2.113, 2.387,
2.455 -- lies on that curve.

### 2. Every probe ran at 10x the production learning rate

`accept.py`, `sim.py`, `lowt.py` and `early.py` all set `lr=1e-3`. Production is
`LR = 1e-4` (`experiments/adapter_rl_finetune__zinc.py:185`). At `1e-3` drift alone
contributes +0.13 to +0.20 and buries everything else.

### 3. The `y = x` control returns 0.0 for any input

`tests/test_dam_estimator.py:215` -- the one test meant to separate "lambda is too hot"
from "the reward is doing work", and whose docstring claims it "is not zero by
construction" -- computes `A - A`:

```
A = 0.3961801528930664   second factor = 0.3961801528930664
log_a = 0.0    IDENTICALLY ZERO? True
```

It passes for any reward, any lambda, any model. The failure mode it existed to detect
went undetected because the test was inert.

## The measurement that replaces it

`AdapterDAMTrainer(..., null_adjoint=True)` draws `Z` from `X_t` instead of `X_Y`, so
numerator and denominator sample the same law and **the true adjoint is identically 1**
-- while picks, rates, gKL support, seed and RNG consumption stay bit-for-bit the real
arm's. `resid(real) - resid(null)` at matched step size isolates the signal. This is the
control `test_adjoint_is_one_at_y_equals_x` was supposed to be.

Pre-RL zinc-kek base, RDKit logP, head candidates, `n_jumps=8`, `K=12`, `lambda=0.3`,
12 iterations, median over the last 8 (`scratchpad/lrpair.py`):

| lr | arm | drift | E[a] | resid nodes | resid edges |
|---|---|---|---|---|---|
| 3e-5 | real | 0.010 | 1.169 | 1.003 | 0.992 |
| 3e-5 | null | 0.005 | 1.051 | 0.999 | 1.000 |
| | **diff** | x1.95 | | **+0.004** | **-0.007** |
| 1e-4 | real | 0.029 | 1.163 | 0.995 | 0.980 |
| 1e-4 | null | 0.016 | 1.065 | 1.001 | 1.000 |
| | **diff** | x1.76 | | **-0.006** | **-0.021** |
| 3e-4 | real | 0.058 | 1.161 | 1.000 | 0.965 |
| 3e-4 | null | 0.045 | 1.063 | 1.017 | 1.017 |
| | **diff** | x1.29 | | **-0.017** | **-0.051** |

The paired difference **grows monotonically with step size** in both channels. That
dose-response is the result: noise gives no reason for the real-minus-null gap to widen
systematically as the policy moves further. The null degrades as a signal-free target
should (1.000 -> 1.000 -> 1.017); the real arm improves (0.992 -> 0.980 -> 0.965).

Four-arm run on both bases (`scratchpad/power.py`, 8 iterations):

| base | arm | drift | E[a] | resid nodes | resid edges | orc_flat | orc_state |
|---|---|---|---|---|---|---|---|
| pre-RL | lr=0 real | 0.000 | 1.209 | **1.000** | **1.000** | 0.943 | 0.608 |
| pre-RL | lr=1e-4 real | 0.021 | 1.209 | 0.998 | 0.979 | 0.940 | 0.551 |
| pre-RL | lr=1e-3 real | 0.197 | 1.128 | 1.176 | 1.157 | 0.993 | 0.700 |
| pre-RL | lr=1e-3 null | 0.092 | 1.068 | 1.119 | 1.095 | 0.972 | 0.777 |
| shipped | lr=0 real | 0.000 | 1.125 | **1.000** | **1.000** | 0.982 | 0.743 |
| shipped | lr=1e-4 real | 0.019 | 1.082 | 1.001 | 1.002 | 0.992 | 0.768 |
| shipped | lr=1e-3 real | 0.129 | 1.070 | 1.238 | 1.130 | 1.041 | 0.781 |
| shipped | lr=1e-3 null | 0.153 | 1.065 | 1.273 | 1.455 | 1.044 | 0.722 |

`lr=0` pins `resid` at exactly 1.000 on both bases, confirming it is a step-size meter.
`orc_flat` -- the best a *uniform rescale of the rate tensor with no state information*
can do -- is 0.94-1.04, i.e. as good as or better than every number in the withdrawn
table. On the shipped base it is >= 1: even that buys nothing there.

## Coupling and temperature: the hypothesis is falsified

Alg. 1 line 7 was implemented (`_coupled_draws`); `E[a_hat]` on the real model moved
from 1.042 to 0.989 at lambda=0.3 and holds at 0.976 / 0.967 for lambda = 1.0 / 3.0,
where the uncoupled estimator diverges. Coupling is also cheaper -- K sub-rollouts
against K + m, 83 -> 72 s/it. With the bias gone, `lambda` is free to carry
temperature: `g_spread` scales 0.428 / 1.449 / 4.130 across lambda = 0.3 / 1.0 / 3.0.

The hypothesis was that the bias and the resulting cold temperature were jointly
suppressing the signal. They were not. Pre-RL base, coupled, simulate candidates,
K=8, `n_jumps=4`, 12 iterations, median over the last 8, `resid(real) - resid(null)`:

| lambda | lr | diff nodes | diff edges |
|---|---|---|---|
| 1.0 | 1e-4 | -0.000 | +0.015 |
| 1.0 | 3e-4 | -0.010 | -0.010 |
| 0.3 | 3e-4 | -0.039 | -0.005 |

lambda=1.0 is the WEAKEST configuration measured, despite being the only one with an
unbiased estimator. Restoring the temperature made the update worse, not better.

## The channel split

The strongest cell was replicated at three trainer seeds (`scripts/seedrep.py`,
lambda=0.3, lr=3e-4, coupled):

| seed | diff nodes | diff edges |
|---|---|---|
| 7 | +0.172 | -0.016 |
| 43 | -0.013 | -0.026 |
| 91 | -0.531 | -0.036 |
| | mean -0.124, sd 0.364, t=-0.59 | **mean -0.026, sd 0.010, t=-4.50** |

**Edges replicate; nodes are noise.** Across all nine paired cells measured -- two
candidate modes, two temperatures, coupled and uncoupled, three seeds -- the edge
difference is negative in 8 of 9 and the node difference in 6 of 9 with sign flips and
36x the spread.

That is a sharper statement than the withdrawn verdict and partly rehabilitates the
intuition behind it: the x1-parameterisation does bind, but specifically **on the node
channel**, while edges remain steerable. The original error was pooling the two on a
contaminated support (see below) and concluding the whole rate family was the obstacle.

### Absolute `resid` has run-to-run scatter of ~0.8

Same nominal cell, trainer seed 42 -> 7: `resid_nodes` goes 1.031 -> 1.837. Every
absolute figure in the withdrawn table -- 1.057, 1.099, 1.200, 2.113 -- sits inside
that. The paired differences meanwhile stay within +-0.010 across seeds on edges. The
null is what absorbs the shared variance; without it nothing at this scale is
resolvable, which is why no configuration before it produced an interpretable number.

## Why it does not work: the estimator is short by 20-40x

`scripts/snr.py` measures the thing the adjoint actually needs -- the effect of ONE
edit on the final score -- against the noise of everything the trajectory does after
it. 4 states x 3 edits, 24 continuations per side, matched seeds:

| t | steps left | n needed (unpaired) | n needed (paired) | CRN variance reduction |
|---|---|---|---|---|
| 0.978 | 15 | **21** | 14 | 1.5x |
| 0.750 | 50 | **38** | 33 | 1.1x |

**DAM uses ONE continuation per estimate. It needs 20-40.** Per-edit effects are real
(0.01-0.20) but sit under a spread of 0.1-1.3 from downstream randomness, so every
correction the method computes is several sigma of noise around a one-sigma signal.
The training config (`subsample='late'`) sits at t=0.938, so ~21-25.

That is the answer to "it is a published method, why does it not work here". Two
differences from DAM's own experiments, which are masked diffusion LANGUAGE models:

* **Their edits are permanent; ours are at realistic eta, but were not in these
  experiments.** At `eta=30` DeFoG churns hard (9.15 changes per atom); at the `eta=1.0`
  every DAM run here used, it does not (1.08). See the permanence section below.
* **Their reward is decisive per edit.** A wrong digit in Sudoku takes the reward from
  1 to 0. logP averages over ~40 atoms, so no single bond moves it much.

One edit is nearly the whole outcome there; it is a rounding error here. Same
estimator, different noise regime.

### Common random numbers do not fix it

Running the continuations from `x` and from `y` on an identical random stream gives
only 1.1-1.5x. The `same_end` column says why: 17-54% of matched pairs reach the same
molecule. The two states differ by one coordinate, but the network conditions on the
whole graph, so its output shifts everywhere, coordinates flip on the first step and
the trajectories decorrelate immediately. **Any variance reduction that relies on
coupling two trajectories will hit this same wall.**

### Correction: signal does NOT rise as t falls

The t-band sweep read rising `g_spread` (0.413 -> 0.541) as more signal at low t. It
is not. This shows the spread grows while the per-edit effect SHRINKS -- `n` goes from
21 to 38 -- so what rises at low t is noise. The late band was the better one.

## Averaging continuations: the shortfall was real but NOT the constraint

`n_z` continuations averaged inside each of the K trajectories, lambda=0.3, lr=3e-4,
K=8, minibatch 16, pre-RL base (`scripts/nz_sweep.py`):

| n_z | iters | sd(a_hat) | diff nodes | diff edges | drift real/null | noop real/null |
|---|---|---|---|---|---|---|
| 1 | 12 | 0.568 | -0.039 | -0.005 | 0.044 / 0.030 | -- |
| 10 | 12 | **0.116** | +0.121 | +0.046 | 0.031 / 0.032 | -- |
| 50 | 6 | **0.116** | -0.605 | +1.382 | 0.034 / 0.052 | 0.027 / 0.092 |

**1. The averaging worked, and it saturates at n_z=10.** `sd(a_hat)` falls 0.568 ->
0.116 and then does NOT move at n_z=50. So estimator noise is gone by n_z=10 and the
remaining 0.116 is genuine between-jump and between-state variation, which averaging
continuations cannot remove. Back out the noise at n_z=1:
`sqrt(0.568^2 - 0.116^2) = 0.556`, a noise/signal ratio of 4.8, implying ~23 samples
-- an independent confirmation of `snr.py`'s 21-25 by a completely different route.

**2. Removing the noise does not make the update helpful.** The n_z=10 comparison is
clean -- drift 0.031 vs 0.032, `sd(a_hat)` 0.116 vs 0.111 -- and the real arm is WORSE
than its null in both channels. The 20-40x sample shortfall is real and was not the
binding constraint.

**3. The n_z=50 row is confounded; do not read it.** Drift differs 1.5x (0.034 vs
0.052) and the target scale `noop` differs 3.4x (0.027 vs 0.092), so real and null are
not on a common footing. `-0.605 / +1.382` means nothing.

**4. First outcome measurement.** Untrained: reward -1.4741, 54/64 decode, logP MAE
0.6730. After 6 iterations at n_z=50: reward -1.6229, 51/64, MAE 0.6640. No change in
either direction -- the MAE difference is nothing at n=52 and the reward drop tracks
the validity drop, itself within binomial noise. Six iterations is far too little
training for this to rule anything out, but it establishes the harness and baseline.

### Retraction: the edge effect does not survive noise removal

`docs/dam_result.md` previously reported edges as a reproducible signal, -0.026 +-
0.010 across three seeds at n_z=1. It flips to **+0.046 at n_z=10 with matched drift**.
A genuine directional signal should get CLEARER when estimator noise is removed, not
reverse. The likeliest reading is that the effect was an artifact of noise interacting
with the ratio, not the adjoint pointing anywhere useful. **The "edges replicate,
nodes are noise" finding is withdrawn.** What replicated was a property of the metric.

### `resid` has now failed in three distinct ways

1. Its ceiling is 1.0 when the target carries no learnable signal, so the pass mark
   was unreachable (`d30062c`).
2. Its absolute value has ~0.8 run-to-run scatter, swallowing every reported cell.
3. Its denominator `noop` varies between arms and shrinks as the adjoint sharpens, so
   it is comparable neither across `n_z` nor, at n_z=50, between real and null.

It should not be used again. Any further DAM work here has to be judged on generated
outcomes at realistic training length.

## Why it fails here: per-coordinate credit assignment is thousands of times harder

DAM's target posterior is `p*(x1|xt) ~ p_base(x1|xt) * exp(-g(x1))`. But DeFoG's rate
is `R_i(xt->j) = sum_c p_theta(x1^i=c|xt) * R_i(xt->j|c,t)` -- it reads ONE
coordinate's MARGINAL at a time. So the entire instruction DAM can deliver is the
difference between the base marginals and the tilted marginals.

`scripts/marginal.py` draws K=256 endpoints from a state, weights them by `exp(-g)`,
and measures what the tilt does to the JOINT (mean reward) against what it does to the
CHANNEL (per-coordinate total variation). The control permutes the same weights across
samples: identical spread, no relationship to the molecules, so its shift is the
finite-sample noise floor.

| t | lambda | ESS | E[r] base | E[r] tilt | gain | node dTV | shuffled | ratio |
|---|---|---|---|---|---|---|---|---|
| 0.938 | 0.3 | 231.3 | -2.2090 | -1.6975 | +0.5115 | 0.0056 | 0.0034 | 1.65x |
| 0.938 | 1.0 | 153.2 | -2.2090 | -1.2399 | +0.9691 | 0.0160 | 0.0091 | 1.76x |
| 0.938 | 3.0 | 36.8 | -2.2090 | -0.6700 | +1.5390 | 0.0416 | 0.0207 | 2.01x |
| 0.750 | 0.3 | 231.0 | -1.5828 | -0.9733 | +0.6094 | 0.0075 | 0.0066 | 1.14x |
| 0.750 | 1.0 | 176.3 | -1.5828 | -0.6316 | +0.9511 | 0.0155 | 0.0129 | 1.20x |
| 0.750 | 3.0 | 85.5 | -1.5828 | -0.3240 | +1.2588 | 0.0347 | 0.0309 | 1.12x |

**The tilt is excellent, and the per-state instruction is tiny.** Reweighting cuts the
logP error threefold at lambda=3 using only the base model's own samples -- the target
is not the problem. But the marginals move 1.1-2.0x above a shuffled-weight floor, and
at t=0.75 the edge channel at lambda=3 is 0.0134 against a 0.0137 floor, i.e. BELOW
noise. Raising lambda does not change this: 0.3 -> 3.0 grows the shift 7x while the
ratio to the floor moves 1.65 -> 2.01 and ESS collapses 231 -> 37.

### What this does NOT show (corrected)

An earlier revision of this document concluded "the information does not fit through the
channel" and "DeFoG can only move marginals". **Both are wrong and are retracted.**

* **DeFoG represents combinations perfectly well.** The head predicts
  `p(x1^i|x_t)` conditioned on the WHOLE graph through many transformer layers, and
  sampling is sequential, so correlations are built across steps exactly as in any
  autoregressive or diffusion model. Validity is the proof: a valid molecule is nothing
  but correlations, and DeFoG produces them.
* **Decisive counter-evidence is in this repo.** GDPO policy-gradient RL steers logP on
  this model; so does the CFG adapter (MAE 0.6453 -> 0.5420); so does Feynman-Kac
  particle steering. Same model, same property, three methods that work. logP is not
  unreachable for this architecture.
* **A tiny per-step instruction is arithmetically expected.** The whole tilt is worth
  ~1 nat at the path level; spread over 250 steps x ~3000 coordinates that is ~1e-6
  nats per coordinate-step. Smallness is not evidence of an obstruction, and the
  earlier revision treated it as though it were. What the shuffled control actually
  establishes is narrower: *with 256 samples the instruction is hard to distinguish
  from noise* -- a statement about the ESTIMATOR, not the model.

### The real difference: DAM needs per-coordinate credit, GDPO does not

GDPO takes one scalar reward per finished molecule and pushes the whole trajectory's
log-probability. It never works out which atom was responsible. DAM must estimate,
separately for every coordinate at every state, how much that specific edit changed the
expected outcome -- strictly more information, and strictly harder to obtain.

How hard depends on whether the instruction is the SAME across coordinates. When it is,
an estimator pools across all ~3000 of them, an enormous effective-sample-size gain.
When each coordinate's instruction differs and partly cancels, it must be resolved
coordinate by coordinate. That is the pooling deficit, and it is of order thousands --
against which the bias fix, the temperature and the 20-40x sample shortfall are
corrections of order 1-50x. **That is why none of them helped.**

This is a COST argument, not an impossibility one.

### Confirmed by controlled contrast: it is the SHAPE of the reward

`scripts/decomp.py` scores the SAME 256 completions from the SAME states three ways --
`logp-match` = -|logP - target|, `oxy-max` = #oxygens, `oxy-match` = -|#oxygens - mean|.
`oxy-max` vs `oxy-match` holds the chemical quantity fixed and varies only the shape:
`exp(lam * #O)` factorises over atoms exactly, while a target on the COUNT is a
constraint on a sum. lambda is bisected per reward to a COMMON effective sample size,
because standardising to unit sd does NOT equalise tilt strength (oxygen count takes
few discrete values, so the same lambda concentrates its weights far harder).

| reward | lambda | ESS | d p(O) | shuffled | node dTV | shuf | coherence |
|---|---|---|---|---|---|---|---|
| logp-match | 3.24 | 64.0 | +0.0036 | -0.0007 | 0.0436 | 0.0155 | 0.577 |
| oxy-max | 0.82 | 64.0 | **+0.0313** | -0.0001 | 0.0440 | 0.0175 | **1.000** |
| oxy-match | 200.00 | 170.2 | -0.0028 | -0.0001 | 0.0213 | 0.0088 | 0.215 |

`coherence` = |mean signed shift| / mean |shift|: 1.0 means every slot moved the same
way, 0 means the movements cancel.

At **identical ESS and identical gross movement** (dTV 0.0436 vs 0.0440, within 1%),
`oxy-max` delivers **8.7x the net directional signal** with coherence exactly 1.000,
while `logp-match` moves each slot a fifth as far at 58% coherence so much of it
cancels. Same model, same states, same completions -- only the reward's shape differs.

Coherence 1.000 is exactly the poolability above: every coordinate receiving the same
instruction is what lets an estimator average across coordinates instead of resolving
each one. This contrast holds the PROCESS fixed and varies only the reward, so it
demonstrates the reward-shape effect independently of any claim about permanence.

`oxy-match` is sharper still: it **could not reach ESS 64** at any lambda up to the
bisection ceiling of 200. A target on a count, when the base is already centred there,
offers almost nothing to select on, and what it selects gives `d p(O)` = -0.0028, i.e.
zero. The identical quantity is a strong coherent instruction as "more of this" and no
instruction at all as "exactly this many".

### What was wrong along the way

* "Correlations are lost because the head is factorised" -- WRONG as stated. DeFoG
  builds correlations across steps; per-step factorisation is not the issue.
* "The information does not fit through the channel" / "DeFoG can only move marginals"
  -- WRONG, retracted above. GDPO, the CFG adapter and FK steering all move logP on
  this model. The barrier is the cost of per-coordinate credit assignment, not
  expressiveness.
* "The dTV ratio separates decomposable from aggregate rewards" -- FALSIFIED. dTV is
  unsigned and cannot tell coherent movement from jitter; all three rewards sit at
  2.4-2.8x. The signed `d p(O)` and `coherence` are what separate them.
* The channel claim itself -- HOLDS, against a shuffled-weight floor throughout.

### Permanence: measured at the WRONG eta first, then corrected

The natural rival explanation is that masked diffusion's edits are irreversible by
construction while DeFoG's are not, so an edit's effect on the endpoint is diluted here.
`scripts/churn.py` measures it. For each step: `agree` = fraction of coordinates already
holding their FINAL value, `settled` = fraction holding it at that step AND every step
after. Masked diffusion's gap between the two is exactly 0 by construction.

| t | agree_X | settled_X | gap | agree_E | settled_E | gap |
|---|---|---|---|---|---|---|
| 0.000 | 0.638 | 0.503 | 0.135 | 0.850 | 0.774 | 0.077 |
| 0.510 | 0.728 | 0.691 | 0.037 | 0.904 | 0.878 | 0.025 |
| 0.750 | 0.829 | 0.809 | 0.020 | 0.940 | 0.931 | 0.009 |
| 0.910 | 0.907 | 0.904 | **0.003** | 0.972 | 0.968 | **0.004** |
| 0.990 | 0.992 | 0.992 | 0.000 | 0.995 | 0.995 | 0.000 |

Mean flips per coordinate over the whole run: **nodes 1.08, edges 0.39**. Median t of
last change is 0.000 for both, i.e. **over half of atoms and over three quarters of
bonds never change at all**. At t=0.938 -- the operating point of every DAM run here --
the reversion rate is ~0.3%.

**That table is at `eta=1.0`, and eta is the stochasticity knob itself.** Re-measured at
realistic settings (`eta=30`, 500 steps, `scripts/churn30.py`) the picture inverts:

| t | agree_X | settled_X | gap | agree_E | settled_E | gap |
|---|---|---|---|---|---|---|
| 0.000 | 0.564 | **0.006** | 0.558 | 0.821 | 0.124 | 0.697 |
| 0.360 | 0.547 | 0.058 | 0.488 | 0.819 | 0.345 | 0.474 |
| 0.510 | 0.529 | 0.128 | **0.401** | 0.814 | 0.485 | 0.329 |
| 0.750 | 0.698 | 0.535 | 0.163 | 0.877 | 0.784 | 0.092 |
| 0.910 | 0.913 | 0.878 | 0.035 | 0.961 | 0.945 | 0.016 |
| 0.990 | 0.977 | 0.977 | 0.000 | 0.995 | 0.995 | 0.000 |

Mean flips per coordinate: **nodes 9.15, edges 3.80** -- against 1.08 and 0.39 at
eta=1. At t=0 essentially nothing is in its final state; through the middle ~40% of
atoms are momentarily correct and will move again; solidification is abrupt between
t~0.64 and t~0.91. The "churns wildly, then suddenly solidifies" picture is CORRECT at
realistic eta, and the eta=1 conclusion above was an artifact of measuring one
unrepresentative configuration.

**But it does not rescue permanence as the explanation of what we observed.** Every DAM
run in this document used `eta=1.0` -- the regime where edits ARE near-permanent
(1.08 flips/atom) -- and DAM failed there, on the axis most favourable to it. So
permanence is excluded for the observed failures, not because DeFoG's edits are
permanent in general (they are not) but because they were permanent at the operating
point that was tested.

(Caveat: much of the raw `agree` figure is coincidence -- the noise distribution is the
data marginal and carbon is 73.7% of atoms, so chance agreement is ~57%. The GAP and the
flip count are free of that base rate and carry the argument.)

This also corrects the earlier reading of the CRN result (only 17-54% of seed-matched
pairs reach the same molecule). That is not the edit being reverted: it is the edit
changing which OTHER coordinates move, because one altered coordinate shifts the
network's predictions across the whole graph. Downstream divergence, not instability in
the edited coordinate.

### Open, and bigger than DAM: the eta regime gap

`experiments/adapter_rl_finetune__zinc.py:180` sets `ROLLOUT_ETA = 1.0`, documented as a
sweep winner (job 1006501) because under CRN eta is the sole within-group diversity
source. Evaluation harnesses use `eta=25` (27 occurrences). For GDPO that is defensible
-- it optimises trajectory log-probabilities and the sweep validates it.

**For DAM it is not.** DAM fits a RATE, and the rate is `R* + eta*R^DB + omega*R^TG`.
At eta=25 the detailed-balance term is 25x larger, so `rate_basis` -- and therefore the
map from head marginals into rate space -- is a substantially different object from the
one deployed. Every DAM number here was fitted at eta=1.0. Even a successful fit would
not obviously transfer to eta=25.

This is untested and applies to any rate-space method, not just DAM.

## The instruction IS learnable: split-half reliability 0.89

`scripts/splithalf.py`. 24 states x 384 completions (568 node, 6709 edge coordinates).
Split each state's completions in half, compute the signed per-coordinate shift
independently in each half, correlate. Three levels: `raw`; `resid-1` with the per-class
global mean removed; `resid-2` with the per-state per-class mean removed -- the last
asks whether, GIVEN the state and the class, knowing which coordinate it is carries
reproducible information. Null = weights shuffled within each half.

| eta | lambda | channel | level | r_half | null | r_full (Spearman-Brown) |
|---|---|---|---|---|---|---|
| 1 | 1.0 | node | raw | +0.816 | -0.025 | **+0.899** |
| 1 | 1.0 | node | resid-2 | +0.802 | -0.032 | **+0.890** |
| 1 | 1.0 | edge | raw | +0.810 | +0.081 | +0.895 |
| 1 | 1.0 | edge | resid-2 | **+0.810** | -0.058 | **+0.895** |
| 1 | 3.0 | node | resid-2 | +0.673 | -0.061 | +0.804 |
| 30 | 1.0 | node | resid-2 | +0.634 | +0.062 | +0.776 |
| 30 | 1.0 | edge | resid-2 | +0.396 | -0.012 | +0.567 |

**Two findings, both positive.**

1. **The instruction is highly reproducible.** Both halves condition on the SAME `x_t`,
   so whatever reproduces across them is by definition a function of
   `(x_t, coordinate, class)` -- precisely the object a regressor would fit. It is
   learnable in principle, not merely present.
2. **It survives residualisation essentially untouched.** `resid-2` 0.802 against `raw`
   0.816 for nodes, and 0.810 vs 0.810 for edges. Removing the per-state per-class mean
   removes almost NONE of the signal, so this is not a per-element preference in
   disguise. Nine numbers per element would capture nearly nothing of it.

### This does not contradict the channel measurement

They measure different things. `dTV` against a shuffled floor asks how BIG the shift is
(1.8x, small). Split-half correlation asks how REPRODUCIBLE its pattern is (0.89, very
high). **Small but extremely consistent** is exactly what a regressor exploits and what
per-state Monte Carlo cannot.

### The pooling deficit as one number

Extrapolating the same reliability down by Spearman-Brown:

| samples per instruction | reliability |
|---|---|
| **8 -- what DAM used** | **0.144** |
| 12 | 0.202 |
| 50 | 0.513 |
| 100 | 0.678 |
| 384 | 0.890 |

DAM estimated each instruction at reliability 0.14-0.20, i.e. mostly noise. The same
instruction is 0.89 reliable at 384 samples. A regressor needs neither, because every
coordinate of every sample at every state supervises one shared function.

At eta=30 the structure is ~30% weaker but alive (nodes 0.634, edges 0.396, nulls ~0),
so it does not evaporate in the regime where it would be deployed.

**This is the first clearly positive result in this investigation, and it is the
justification for [`credit_head_design.md`](credit_head_design.md).**

### Still not run

The training confirmation: DAM on `oxy-max`. The prediction is that it steers, where
every logP run did not. It now also SEPARATES two explanations that both fit the
evidence: reward shape (poolability) and process permanence (masked diffusion's edits
are irreversible; DeFoG's are not). `oxy-max` fixes the reward shape while leaving the
process untouched. With permanence now excluded by measurement, it is a cleaner test
than it was: the process is not a confound. That would complete the boundary: **DAM transfers to graph
generation for rewards that decompose over coordinates, and not for rewards defined on
aggregate properties** -- with `scripts/decomp.py` as a minutes-long diagnostic that
says which case you are in before spending days on training.

## Established / not established

**Established.** DAM's per-coordinate credit assignment is far more expensive for an
aggregate reward than a decomposable one: at matched tilt strength the decomposable
reward gives 8.7x the net signal at coherence 1.000 against 0.577. `resid` is unusable
(three independent failure modes above). The
estimator's noise/signal ratio is ~4.8 at one continuation per estimate, needing ~23
to resolve -- measured two independent ways -- and `n_z=10` removes it entirely.
Alg. 1's coupling fixes `E[a_hat]` at 1 as specified. **None of the three suppressors
was the binding constraint**: not the estimator bias (coupling), not the temperature
(lambda=1.0), not the sample shortfall (n_z). Each was removed and measured; none
produced a usable update.

**Not established.** That DAM steers DeFoG. A 2.6% residual reduction on one channel is
not a property shift; no molecules were generated and no logP was measured in any of
this. The effect is absent on the shipped base (1.001/1.002 at production LR). The
`grad x` ratio does not replicate across bases (pre-RL 1.3-2.0x, shipped 0.84x) and is
not a signal detector on its own.

## The suppressor that was removed (and did not help)

DAM Alg. 1 line 7 (PDF p.5): *"Set (Y, Z) as the first and last jumps of one of the
trajectory X^(k)"* -- `Z` is a **member of the K-candidate denominator set**, which
forces `E[a_hat] = 1` identically. `dam.py:662` draws `Y` independently and `:674`
simulates a fresh `Z`, so the coupling is gone and `E[a_hat]` runs at 1.06-1.21.

`dam_design.md:244` records the same bias offline (1.06 / 12.21 / 93202 at
lambda = 0.3 / 1.0 / 2.0) and **`lambda` was pinned at 0.3 to suppress it**. But
`lambda` is the only knob that puts spread into `g`. So the temperature was cut 3x to
buy off a bias the paper's own coupling removes for free, and every measurement to date
was taken at one third the intended signal.

## What was wrong in the withdrawn verdict

The old conclusion named the "reach ceiling of the x1-parameterisation, 0.6-0.9" as the
surviving cause. That number is wrong: `projection_gap` (`dam_calibrate.py:187`) sums
gKL over the **full** rate tensor including the `j == x_t` entry, which DAM Eq. (14)
excludes (`Sum_{y != x}`) and which DeFoG's sampler overwrites at `model.py:1121-1123`.
At eta=1 with `rdb='general'` (`rate_matrix.py:293-294`, `x_mask = ones_like`) that
entry carries full weight. The correct-support numbers were already in
`dam_design.md:161` -- **0.281 / 0.246 / 0.566** -- under the conclusion *"the family is
not the binding constraint"*. The verdict also eliminated reachability and then named
it as the cause.

## Next

1. ~~Implement Alg. 1 line-7 coupling~~ -- done, `E[a_hat]` = 0.97-1.01 at every lambda.
2. ~~Raise `lambda` toward 1.0~~ -- done, and it made the update worse.
3. Measure something real -- generated logP against the conditioning target,
   the way the GDPO and adapter arms are scored. `resid` is a debugging instrument, not
   an outcome, and should never again be the acceptance criterion.
4. Fix `test_adjoint_is_one_at_y_equals_x` to use the `null_adjoint` path.
5. Fix `projection_gap` to sum over `y != x`.

## What is worth keeping

* `dam.rate_basis` / `dam.marginal_rate` -- DeFoG's rate as an EXACT linear functional
  of the clean-graph head, differentiable where `compute_rate_matrices` is not.
  Independent of DAM and the reusable result.
* `defog/core/renoise.py`, verified against the kernel for all three noise types.
* `RLTrainerBase` and the frozen-copy parity gate (`tests/test_rl_parity.py`).
* `null_adjoint` and the six drift/adjoint/oracle diagnostics -- the reason any of this
  was diagnosable.
* `AdapterRAMTrainer` -- untouched by all of the above; it uses GDPO's loss and never
  enters rate space. GDPO-vs-RAM remains open and cheap.
* The kek vocabulary guard: kek is de=4 with a different atom order.

## Method note

Three conclusions in this investigation were drawn from two data points and each was
overturned by the third: "reward degeneracy", "the candidate surrogate", and "more
signal, worse update". The fix that worked was not more thinking, it was building the
matched null -- a control that pins the metric at its known value when the quantity
under test is absent. Nothing here was diagnosable until that existed.
