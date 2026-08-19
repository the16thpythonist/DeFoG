# DAM on DeFoG: measured outcome

**Status:** negative result. Recorded so it is not re-derived.
**Plan:** [`dam_design.md`](dam_design.md) · **Code:** `defog/core/dam.py`, branch `feat/dam-rl`

Discrete Adjoint Matching (arXiv:2602.07132) was implemented against DeFoG and does
not move the policy toward its own target on either zinc-kek base. No cluster
experiment was run beyond one 2-GPU head-fitting job; every number below is local or
from that job.

## The measurement that decides it

`resid` is the gKL distance from the fitted rate to DAM's target, divided by the same
distance for the untouched base rate. **Below 1 = the update helped. Above 1 = it
moved away.** Medians over iterations 3-8, `n_jumps=8`, `lambda=0.3`, `K=12`:

| base | terminal loss | g_spread | log_a | resid nodes | resid edges |
|---|---|---|---|---|---|
| shipped (2x sanity-RL) | RDKit | 0.487 | -0.187 | 1.200 | 0.978 |
| shipped | scoring head | 0.013 | -0.011 | 2.209 | 1.081 |
| pre-RL (E1 seed42) | RDKit | 0.478 | -0.165 | 1.057 | 1.117 |
| pre-RL | scoring head | 0.016 | -0.011 | 2.455 | 1.500 |

Nothing reaches below 1. The best cell in the table is the pre-RL base with the
ordinary RDKit reward, at 1.057.

## What was ruled out, in order

1. **The algebra.** `test_dam_reaches_kl_optimum` converges to `p_ref * e^-g / Z`
   (KL 0.0001 at lambda=0.3) and goes red under three deliberate mistranscriptions
   and under breaking the shipped function in memory. Eq. (11) and Eq. (13) were
   read off the typeset PDF, not a text extraction that was dropping a superscript.
2. **Estimator variance.** Eq. (11) is a one-sample estimate of a sum over ~3.2k
   reachable jumps. Averaging m=8 narrowed the node residual from [0.87, 7.03] to
   [1.01, 2.42] and did not move the median (1.295 -> 1.284).
3. **Reachability.** The pre-RL base expresses far more of an arbitrary tilt than the
   shipped one -- node projection gap 0.09-0.58 against 0.65-0.87, same vocabulary,
   differing only by two sanity-RL rounds. Its residual is still 1.057. Better reach
   did not convert into a better update, so reachability was never the binding
   constraint. (The RL rounds do measurably reduce steerability, which is worth
   knowing independently.)
4. **Reward degeneracy.** Tested and rejected -- see below. The RDKit floor does not
   collapse `g` (spread 0.44-0.51, not 0), and a head fitted to grade the floored
   region made both the spread and the residual markedly worse.
5. **The candidate surrogate.** The one-shot head draw was replaced with the paper's
   actual procedure -- simulated model trajectories. The residual does not improve;
   see below for what it did reveal instead.

## Two wrong diagnoses, in order

### First: reward degeneracy (wrong)

84-98% of one-shot clean-graph draws fail to decode, and RDKit floors every failure
at the same value. The hypothesis was that this makes `g(Z) == g(X1_k)`, so the
adjoint is 1 and there is nothing to learn.

Falsified twice over. The floor does **not** collapse `g` -- measured `g_spread` under
RDKit is 0.44-0.51, not 0, because the 5-15% of draws that decode still vary. And
grading the floored region made things **worse**: a PropertyHead fitted on 30k draws
(~50% own-labelled, ~50% surrogate) drove `g_spread` DOWN ~30x to 0.013 and pushed the
residual from 1.20 to 2.21 (shipped) and 1.06 to 2.46 (pre-RL).

The head's training target was also ill-posed, which is the transferable lesson. A
broken graph has no logP, so it was labelled with the logP of the endpoint it was
corrupted from -- **not a property of the graph being scored**. The same broken graph
reached from two ancestors gets two labels, so it is not a function of the input at
all. Mixed with own-logP labels on valid draws, the set asks two contradictory
questions; the heads land at MAE 1.57 and 1.61 against truth SDs of 1.04 and 1.15,
i.e. worse than predicting the mean.

The floor is arguably correct: under the real objective invalid structures **are** all
equally worthless, and inventing a graded score for them manufactures signal the
objective does not contain.

### Second: the candidate surrogate (real, but not the cause either)

DAM Alg. 1 line 6 draws candidates by simulating **model trajectories** from `X_t`.
This implementation substituted a one-shot draw from the factorised clean-graph head
-- ~25x cheaper, and not the same object. The head is a good marginal and a poor
joint, so sampling ~740 coordinates independently produces almost nothing valid. That
substitution was listed as a risk in `dam_design.md` section 11.3 and then built upon
without being tested.

It never bites in DAM's own experiments because those are masked diffusion **language**
models: every completion is a valid token sequence and no invalid state exists.
Validity is a hard constraint here and independent per-slot sampling destroys it.

`simulate_to_end` implements the paper's version. It does not fix the residual:

| base | candidates | g_spread | log_a | resid nodes | resid edges | s/it |
|---|---|---|---|---|---|---|
| shipped | one-shot head | 0.439 | -0.239 | 1.379 | 1.077 | 12.4 |
| shipped | **simulated** | 0.282 | -0.033 | 1.380 | 1.040 | 147.3 |
| pre-RL | one-shot head | 0.507 | -0.196 | 1.474 | 1.208 | 12.4 |
| pre-RL | **simulated** | 0.399 | -0.051 | 1.615 | 1.058 | 136.0 |

## What the simulated candidates actually revealed

`g_spread` went **down** with real candidates (0.44 -> 0.28, 0.51 -> 0.40) and `log_a`
moved **toward** 0 (-0.24 -> -0.03, -0.20 -> -0.05). The prediction was the opposite.

The explanation reframes the whole investigation. Candidates simulated from the same
`X_t` are valid molecules with **similar properties**: at the `t` these runs scored at,
the endpoint is already nearly determined, so every continuation lands on much the
same molecule and there is genuinely little to compare. The one-shot draws only looked
more varied because they were varied *garbage* -- spread of noise, read as signal.

That exposes a squeeze:

* **high t** -- simulation is cheap (short sub-rollouts) but the endpoint is settled,
  so the adjoint has nothing to say;
* **low t** -- the endpoint is genuinely uncertain, so the adjoint should have real
  signal, but only a real sub-rollout can score it and those are long.

Every measurement before this sat in the first corner, because runs used
`subsample='late'` on the strength of an earlier "no signal below t~0.7" finding --
which is itself suspect, having been derived from the head's ability to produce varied
draws and therefore conflating "the endpoint is uncertain" with "the head produces
garbage". A `subsample='early'` mode was added to probe the other corner; results
below when available.

## What is worth keeping

* `RLTrainerBase` and the frozen-copy parity gate (`tests/test_rl_parity.py`).
* `defog/core/renoise.py` -- the re-noising step, verified against the kernel for all
  three noise types, with `mode="match"` reproducing a recorded rollout at atol=0.
* `dam.rate_basis` / `dam.marginal_rate` -- DeFoG's rate as an EXACT linear functional
  of the clean-graph head, differentiable where `compute_rate_matrices` is not. This
  is independent of DAM and is the reusable result.
* `AdapterRAMTrainer` -- unaffected by any of the above, since it uses GDPO's loss and
  never touches rate space. The GDPO-vs-RAM comparison ("do re-noised states beat
  trajectory states?") remains open and cheap.
* The kek vocabulary guard: kek is de=4 with a different atom order, and mismatches
  previously mis-decoded silently.

## Not tested

A reward-shaped rather than white-noise tilt in the projection-gap measurement; a
graded validity term in place of the floor; and whether the reach ceiling moves once
an adapter has trained away from its starting point.
