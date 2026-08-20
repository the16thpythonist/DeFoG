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

## Established / not established

**Established.** `resid`'s ceiling is 1.0 and its pass mark was unreachable. Every
historical cell is reproduced by drift against a near-signal-free target. Against a
matched null, the adjoint carries directional signal, edge-dominated, with a monotone
dose-response across three step sizes. `E[a_hat]` is 1.06-1.21 and never 1.

**Not established.** That DAM steers DeFoG. A 5% residual reduction is not a property
shift; no molecules were generated and no logP was measured in any of this. The effect
is absent on the shipped base (1.001/1.002 at production LR). The `grad x` ratio does
not replicate across bases (pre-RL 1.3-2.0x, shipped 0.84x) and should not be read as a
signal detector on its own.

## The suppressor worth removing first

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

1. Implement Alg. 1 line-7 coupling: draw `Y` and `Z` from one of the K trajectories.
   Verify `E[a_hat] = 1.00` in the null arm before anything else.
2. Raise `lambda` toward 1.0 once the bias is gone, and re-run the paired sweep.
3. Only then measure something real -- generated logP against the conditioning target,
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
