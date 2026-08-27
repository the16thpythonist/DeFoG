# DAM on DeFoG: postmortem and verdict

**Status: CLOSED. Do not restart this line of work.**

**Verdict.** Per-coordinate credit assignment -- DAM and the amortised credit head that
replaced it -- is a **dead end for the rewards we actually want**, which are targets on
an aggregate molecular property ("make logP equal 2.5"). It is dead for a measured
reason, not a suspected one: for such rewards the per-coordinate instruction is not
merely small, its *estimability does not improve with more computation*.

**The nuance that must not be lost.** The same machinery demonstrably WORKS for a
**decomposable** reward ("more oxygen"): Gate 2 reached 0.68 and 0.74 against a ceiling
of 0.89 and a null of -0.15, on two seeds. So the failure is about the reward's shape,
not about DeFoG, not about DAM's mathematics, and not about the implementation. Anyone
tempted to retry this should read section 4 before spending a GPU-hour.

**What to use instead for property targeting:** GDPO policy-gradient RL, the CFG
adapter, or Feynman-Kac particle steering. All three already work on this model for
logP. See section 7.

Detailed records: [`dam_result.md`](dam_result.md) (the DAM arm),
[`credit_head_design.md`](credit_head_design.md) (the rescue). This document is the
summary; those are the appendix.

---

## 1. What we were trying to do

DeFoG builds a molecule by starting from noise and repeatedly editing a graph. We want
to steer it toward a property target without retraining the base model.

Methods that already work here (GDPO, CFG adapters, Feynman-Kac) all use **one number
per finished molecule**: generate, score, nudge. None of them ever asks *which atom
deserved the credit*.

DAM (Discrete Adjoint Matching, arXiv:2602.07132, ICLR 2026) promised something richer:
for each individual edit during generation, work out how much *that edit* changed the
final outcome, and regress the model's transition rate onto the corrected one. Strictly
more information than a single scalar. On the paper's own benchmarks it is worth a great
deal -- Sudoku 11.5% -> 89.2% where a coarse method reaches 23.8%.

**One line from the DAM paper matters more than anything else here**, and we found it
only after a week of work (p.11, Conclusion and Limitations):

> "Applying DAM to non-masked CTMC's presents an interesting future work."

DeFoG *is* a non-masked CTMC. We were doing the paper's future work, not applying a
validated method. Read a paper's limitations section before implementing it.

---

## 2. Every measurement we ran, in plain terms

Ordered roughly as they happened. "Ruled out" means the measurement eliminated a
candidate explanation.

### The DAM arm

| # | What we measured | In plain terms | Result |
|---|---|---|---|
| 1 | Tabular convergence gate | Does the algebra reach the right answer on a toy problem where we know it? | **Yes.** Converges to `p_base * e^-g / Z`, and goes red under three deliberate mis-transcriptions. The implementation is faithful. |
| 2 | `resid`, our acceptance metric | Is the number we were judging everything by actually meaningful? | **No, three separate ways.** Its best achievable value is 1.0, and we set the pass mark at "below 1" -- unreachable by construction. Its run-to-run scatter is 0.8, larger than every difference we were discussing. And its denominator changes between arms. |
| 3 | Learning-rate sweep with the policy frozen | If the model cannot move at all, what does `resid` say? | **Exactly 1.000**, on both base models. Confirms `resid` measures *how far you moved*, not *whether you moved correctly*. Every number in the original results table lies on a pure-drift curve. |
| 4 | The matched null control | Run everything identically but with a correction we KNOW carries no information. | The single most useful thing built in the DAM arm. Any difference from it is real signal; without it no absolute number is interpretable. |
| 5 | Alg. 1's line-7 coupling | The paper draws the edit and the outcome from the SAME simulated trajectory; we drew them separately. | Fixing it pinned `E[a_hat]` at 1.00 (was 1.04-1.21) and was **cheaper**. It did not improve steering. |
| 6 | Temperature (lambda) sweep | The bias above was why lambda had been pinned at 0.3. With the bias gone, does a hotter tilt help? | **No, worse.** Ruled out temperature. |
| 7 | Averaging more continuations (`n_z`) | The estimate is noisy; does averaging it fix things? | Noise fell from 0.568 to 0.116 and then **stopped falling** -- so the estimator noise was gone by `n_z=10`. Still no steering. Ruled out estimator noise. |
| 8 | Signal-to-noise per edit | How many simulations does it take to see one edit's effect on the final molecule? | **21-25. DAM uses one.** Every correction it computes is several sigma of noise around a one-sigma signal. |
| 9 | Common random numbers | Can we cancel the downstream randomness by running both continuations on the same random stream? | **No, 1.1-1.5x.** Only 17-54% of matched pairs reach the same molecule: one changed coordinate shifts the network's predictions everywhere and the trajectories decorrelate immediately. The process is chaotic; any coupling-based variance reduction hits this wall. |
| 10 | Edit permanence | Are edits permanent (like the paper's setting) or do they get undone? | Measured at eta=1 first and got the wrong answer; at realistic eta=30 the process churns hard (9.15 type-changes per atom vs 1.08). But every DAM run used eta=1, where edits ARE near-permanent -- and it failed there anyway. Permanence ruled out. |
| 11 | The channel measurement | Reweighting toward good molecules -- how much does it move what the model can actually be told? | The tilt cuts logP error **threefold**, and moves per-coordinate marginals by **1.6%** against a 0.9% noise floor. The reward's information is real and almost none of it reaches the channel. |
| 12 | The reward-shape contrast | Same molecules, same tilt strength, three rewards: hit a target logP, maximise oxygen, hit a target oxygen count. | The decomposable reward gives **8.7x** the usable signal at *identical* gross movement, with coherence 1.000 vs 0.577. And the same quantity phrased as "exactly this many" gives **zero**. The shape of the reward decides it, not the property. |
| 13 | Reliability of the instruction | Given a state, is the per-coordinate instruction reproducible, or is it noise? | **r = 0.89** with 384 simulations. It is a real, stable, coordinate-specific quantity. |
| 14 | The eta regime gap | Training rollouts run at eta=1; evaluation runs at eta=25. | DAM fits a *rate*, and the rate's composition changes 25-fold between those. Every DAM number was fitted in a regime it would not be deployed in. Untested, and it applies to any rate-space method. |

### The credit-head arm (the rescue attempt)

Since 23 simulations per instruction is unaffordable but the instruction is reproducible
(#13), the idea was to **train a network to predict it once** and look it up for free.

| # | What we measured | In plain terms | Result |
|---|---|---|---|
| 15 | Gate 1 | Does the network beat trivial baselines on held-out data? | Passed in every round -- including a round where the network was provably throttled to 0.8% of its capacity. **A gate that always passes is measuring the wrong thing.** |
| 16 | Gate 2 | Does the network reproduce the instruction we measured directly? Built so a per-element shortcut cannot pass. | The honest gate. Sat at 0.10-0.23 (null 0.10) across four rounds and eight networks, then jumped to **0.68-0.74** once the target stopped being noise. |
| 17 | Gate 3 | Do the molecules get better? | Null throughout. Every "best" arm is inside the range you get by picking the luckiest of five noisy measurements. Baseline: MAE 0.6519, validity 0.945. |
| 18 | Variance decomposition | Is the network's output actually state-dependent, or is it a lookup table? | Caught a design error: a zero-initialised gate had settled at 0.008, so the network's contribution was throttled to near-nothing. |
| 19 | The lambda=0 control | Remove the reward entirely. If the improvement stays, it was never about the reward. | **It stayed, and got slightly better.** The whole 6-9% gain was the network fixing DeFoG's own miscalibrated marginals. Zero reward content. |
| 20 | Target reliability vs K | Is the thing we are asking the network to fit even estimable? | **The measurement that should have come first.** At 8 completions per state the target's split-half reliability is ~0.01 -- two independent halves give uncorrelated answers. Five rounds of conclusions were drawn from a target that was pure noise. |
| 21 | The K dose-response | Raise completions per state to 8, 16, 32, 64 on identical states. | For oxygen, reliability rises 0.058 -> 0.159 and the network goes worse -> equal -> **better than no guidance**, monotonically, on both seeds. For logP, reliability is **flat across an 8x range**. |

---

## 3. The result, in one table

Round 6, K=64, identical states, identical pipeline, two seeds. **Only the reward's
shape varies.**

| | oxygen (decomposable) | logP (aggregate) |
|---|---|---|
| target reliability, K = 8 -> 64 | 0.058 -> **0.159** | 0.015 -> **0.015** |
| Gate 1 vs no guidance | **beats it**, t = -6.7 / -7.7 | worse, t = +22.8 |
| **Gate 2** (ceiling 0.890) | **+0.679 / +0.737** | **+0.023 / +0.054** |
| its null | -0.154 | +0.101 |

**The sharpest single fact is logP's flatness.** More completions do not help *at all*.
That is not a budget problem you can spend your way out of -- for a target on a sum, the
per-coordinate instruction is essentially zero, and no amount of sampling resolves zero.

**Three independent instruments agree on the boundary:**

| measurement | oxygen | logP | ratio |
|---|---|---|---|
| net directional signal at matched tilt strength | coherence 1.000 | coherence 0.577 | **8.7x** |
| target reliability at K=32 | 0.140 | 0.015 | **9.1x** |
| Gate 2 at K=64 | 0.68-0.74 | 0.02-0.05 | -- |

The first two share no methodology -- one measures marginal shifts inside a single
state, the other split-half reliability of a training target across 1024 states -- and
they agree to within 5%.

---

## 4. Why this is a dead end for us, specifically

The reward we want is **"hit a target value of an aggregate property."** That is a
constraint on a *sum over the whole molecule*, and it is therefore a statement about
which pieces occur *together*, not about any single position.

Concretely: suppose target logP is 2.5 and the model can produce (many polar groups +
short chain) or (few polar groups + long chain), both at 2.5, plus the two crossed
combinations at 0.5 and 4.5. Reweighting toward good molecules kills the crossed pair.
That is a large change. But "many polar groups" was 50% before and is 50% after; "long
chain" likewise. **Every per-position preference is unchanged.** All the information is
in the pairing, and per-coordinate credit is the wrong language for a pairing.

That is why:

* the tilt moves marginals by 1.6% while cutting logP error threefold (#11);
* "maximise oxygen" carries 8.7x the signal of "hit a target logP" (#12);
* "hit a target oxygen count" -- the SAME quantity, phrased as a sum constraint --
  carries **zero** (#12);
* logP's target reliability does not improve with any number of simulations (#21).

Four measurements, one cause.

---

## 5. What we got wrong along the way

Recorded because the pattern is more useful than any individual error.

**Every serious mistake was in a measurement, not in the method.** The DAM
implementation was audited by six independent reviewers and found faithful. What
repeatedly broke was the thing we judged it *by*:

* `resid` had an unreachable pass mark, 0.8 run-to-run scatter, and a shifting
  denominator -- three independent failures in one metric.
* Gate 1's baselines were fitted on the validation labels, making them 12% too strong.
* Gate 1 compared two aggregate means, hiding a `t = -13.6` effect as "+0.22%".
* Gate 2's reference predictor was computed from the wrong population and was 12x too
  weak.
* A variance diagnostic was degenerate by algebra -- it reported 0.00000 regardless of
  input.
* A "y = x control" computed `A - A` and returned 0.0 for any input, while its own
  docstring claimed it was "not zero by construction".

**A broken measurement does not look broken. It looks like a result.** That is why
`resid` survived three separate failures and why five rounds of credit-head work rested
on a target with 0.01 reliability.

**The corrective that actually worked, every time, was a control** -- an arm rigged so
the quantity under test is known to be absent. The matched null (#4), the lambda=0
control (#19), and the target-reliability check (#20) each overturned a confident
conclusion in minutes. Each was cheap. Each should have been built first.

**Three conclusions were stated confidently and later retracted:** "reward degeneracy",
"the candidate surrogate", and "credit is not amortisable" (asserted four times). All
three came from reading two data points without a control.

---

## 6. What is worth keeping

* **`dam.rate_basis` / `dam.marginal_rate`** -- DeFoG's transition rate expressed as an
  *exact* linear function of the clean-graph head, differentiable where the sampler is
  not. Independent of DAM and the most reusable thing produced.
* **`defog/core/renoise.py`** -- re-noising verified against the kernel for all three
  noise types.
* **`scripts/decomp.py` / `decomp2.py`** -- the reward-shape diagnostic. **Run this
  before committing to any per-coordinate steering method.** Three minutes tells you
  whether your reward is in the regime where such methods can work.
* **`scripts/splithalf.py`** -- measures whether a training target is estimable at all.
  Two minutes. Had we run it first, four rounds would not have happened.
* **The credit head as a calibration tool** (#19) -- it improves DeFoG's own clean-graph
  marginals by 6-9% cross-entropy against simulated ground truth, with no reward
  involved. That is a real, separable result about the model. Never tested on generated
  molecules.
* **The eta regime gap** (#14) -- rollouts at eta=1, evaluation at eta=25. Unresolved,
  and it affects the GDPO and adapter work too, not just DAM.

---

## 7. What to do instead

For property targeting on DeFoG, three methods already work and are already in this
repo:

* **GDPO policy-gradient RL** (`defog/core/rl.py`) -- one scalar per molecule, never
  asks which atom was responsible, which is exactly why it sidesteps everything above.
* **The CFG adapter** -- logP MAE 0.6453 -> 0.5420 with prob-space blending.
* **Feynman-Kac particle steering** -- and note this is the honest answer to the whole
  effort: if you need ~23 simulations per state to get the instruction, FK simply
  *does* that at sampling time rather than trying to avoid it.

**Do not retry per-coordinate credit for an aggregate-property target.** The one route
that could reopen the question -- a reward that is decomposable AND non-uniform across
coordinates -- was never run, and is noted here only so that nobody has to rediscover
that it is the gap. It is not a recommendation.
