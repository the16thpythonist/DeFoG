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
4. **Reward degeneracy.** See below. Tested and rejected.

## The reward-degeneracy hypothesis, and why it was wrong

84-98% of the head's one-shot clean-graph draws fail to decode, and RDKit floors
every failure at the same value. The hypothesis was that this collapses `g(Z)` and
`g(X1_k)` to the same number, making the adjoint 1 and leaving nothing to learn.

Two things falsify it:

* **The floor does not collapse `g`.** Measured `g_spread` under RDKit is 0.487, not
  0. The 5-15% of draws that do decode supply real variation. The earlier claim that
  the adjoint "collapses to 1" was overstated -- inferred from `log_a ~ -0.15` without
  measuring the spread directly.
* **Grading the floored region made it worse.** A PropertyHead fitted on 30k draws
  (~50% own-labelled, ~50% surrogate) drove `g_spread` DOWN by ~30x, to 0.013. It
  predicts nearly the same value for every broken graph, because they look alike to
  it. The residual rose from 1.20 to 2.21 (shipped) and 1.06 to 2.46 (pre-RL).

The training target was also ill-posed, which is the more useful lesson. A broken
graph has no logP, so it was labelled with the logP of the endpoint it was corrupted
from -- **not a property of the graph being scored**. The same broken graph reached
from two different molecules gets two different labels, so it is not a function of
the input; the best attainable fit is a conditional mean. Worse, valid draws were
labelled with their OWN logP, so the set asks two contradictory questions at once.
The heads land at MAE 1.57 and 1.61 against truth standard deviations of 1.04 and
1.15 -- both worse than predicting the mean.

**The floor is arguably correct.** Under the real objective, invalid structures are
all equally worthless. Inventing a graded score for them manufactures signal the
objective does not contain. If a graded term is wanted it should measure a genuine
property of the graph -- valence violations, fragment count, ring sanity -- not a
property it does not have.

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
