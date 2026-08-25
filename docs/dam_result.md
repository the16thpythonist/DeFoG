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

* **Their edits are permanent.** Unmasking a token can never be undone, so the first
  jump's effect reaches the endpoint intact. DeFoG is a general CTMC -- a bond can go
  single -> double -> single, so an edit at t=0.9 may simply be reverted.
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

## THE ANSWER: the reward's information does not fit through the channel

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

**The tilt is excellent and the channel is nearly closed.** Reweighting cuts the logP
error threefold at lambda=3 using only the base model's own samples -- the target is
not the problem. But the marginals move 1.1-2.0x above a shuffled-weight floor, and at
t=0.75 the edge channel at lambda=3 is 0.0134 against a 0.0137 floor, i.e. BELOW noise.

**Temperature does not open it.** From lambda=0.3 to 3.0 the absolute shift grows 7x
while the ratio to the floor moves 1.65 -> 2.01 and ESS collapses 231 -> 37. Shift and
noise are bought in equal measure.

### Why: the reward is about combinations, the channel carries positions

Hitting a target total logP is a constraint on WHICH ATOMS AND BONDS OCCUR TOGETHER.
Reweighting reshuffles the joint dramatically while leaving each individual coordinate's
marginal essentially unmoved. The adjoint asks for an ~11.6% rate correction
(`sd(a_hat)`=0.116) where the marginals support ~1.6%: it is mostly asking for
something the parameterisation cannot deliver.

This closes every thread at once. Fixing the estimator bias, raising lambda, removing
the estimator noise with `n_z`, common random numbers -- **none was the constraint,
because the information never enters the channel.** The "edge signal" was spurious for
the same reason: the true edge shift is 0.1-0.3%, at the noise floor.

And it is why the paper's results are real. GSM8K, Countdown and Sudoku rewards ARE
per-token statements -- "cell (3,4) must be 7" moves that token's marginal from 1/9 to
1. Their reward is native to the channel; a target on a sum over the molecule is
orthogonal to it.

### The confirming test, not yet run

Swap logP for a DECOMPOSABLE reward -- e.g. the count of oxygen atoms, which is a
direct per-node preference. The prediction is a large dTV/floor ratio and a DAM update
that steers. If that holds, the diagnosis is nailed and the boundary is drawn: DAM
transfers to graph generation for rewards that decompose over coordinates, and not for
rewards defined on aggregate properties.

## Established / not established

**Established.** `resid` is unusable (three independent failure modes above). The
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
