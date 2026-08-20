# Plan: improving adapter conditioning

Companion to `RESEARCH.md` (the evidence). This is the ordering and the reasoning behind it.
Serves two things at once — **shipping tighter steering** in molsmith/defog-web, and **the E2
table in the paper**. Where those pull apart is called out explicitly.

---

## The one-paragraph version

We are at logP MAE ~0.52; FreeGress is at 0.16. Four mechanisms could be responsible, and
they are not equally cheap to test. Do the cheap, near-certain things first (finish the
node-count work, fix how branches are blended, fix the guidance null), because two of them
are free at inference and one of them targets a *specific diagnosed discrepancy* with
FreeGress. Then spend one week deciding — with two experiments, not arguments — whether the
remaining gap is **control** or **capacity**. Only build the closed-loop adapter if that week
says control. If it says capacity, the honest paper contribution changes shape, and that is
worth knowing before, not after, the frozen test pass.

---

## The single most important scheduling fact

**The E2 test pass is a one-shot resource.** Protocol discipline says: tune on validation,
run test exactly once with the configuration frozen. So every wave added *before* the test
pass delays the paper, and every wave added *after* it either forces a re-run or ships as
unvalidated. Decide the freeze point deliberately. My recommendation is marked below.

---

## Ranked levers

| # | Lever | Effort | Retrain? | Odds it helps | If it helps, how much | Serves |
|---|---|---|---|---|---|---|
| 1 | Node-count ablation (**built, unrun**) | hours | no | it's a measurement | measures a real effect | paper + ship |
| 2 | Blend in probability space | 1–2 d | no | ~60% | small | paper + ship |
| 3 | **Learned null token** | 3–5 d | yes (cheap) | ~45% | **potentially large** | paper + ship |
| 4a | Adaptive-`w` probe | 1 d | no | ~50% | diagnostic only | paper (science) |
| 4b | Capacity oracle | 2–3 d | yes | n/a | **decides everything** | paper (science) |
| 5 | L3 closed-loop + Reinforced Conditioning | 2–3 wk | yes, ~2× | ~50% given 4a | 0.52 → ~0.42 | ship |
| 6 | Feedback bandwidth (ControlNet-XS lever) | weeks | yes | unknown | unknown | future work |

---

## Wave 1 — DONE (2026-08-13). Null pooled; a real, pre-registered win at the low end.

Run on the frozen Wave-2 config (prob blend, w=2.0): `{marginal, learned} × {seed 42, 43}`,
KCIST job 43080. Size model fitted against `clogp@1.2.0`'s own labels — the decoded
pipeline reproduced its `cond_mean`/`cond_std` to **six decimals** (2.825129 / 1.158113),
so the label convention is confirmed rather than assumed. Fit gain **+0.0939 nats**,
shrink 0.911.

| arm | MAE | low | mid | high | validity |
|---|---|---|---|---|---|
| marginal seed42 | 0.5420 | 0.696 | 0.470 | 0.455 | 0.982 |
| marginal seed43 | 0.5338 | 0.681 | 0.473 | 0.443 | 0.989 |
| learned seed42 | 0.5400 | 0.657 | 0.495 | 0.465 | 0.979 |
| learned seed43 | 0.5358 | 0.606 | 0.508 | 0.492 | 0.983 |

**Pooled: nothing.** Mean paired difference +0.0002, learned better on exactly 100/200,
p≈0.80. Reported first because it is the headline number and it is null.

**By target third, pooled over both seeds:**

| third | n | target range | marginal | learned | Δ | p | validity |
|---|---|---|---|---|---|---|---|
| **low** | 68 | [−3.95, 1.83] | 0.6888 | **0.6314** | **−0.0575** | **0.035** | 0.984 → 0.991 |
| mid | 66 | [1.78, 3.29] | 0.4715 | 0.5015 | +0.0300 | 0.33 | 0.986 → 0.980 |
| high | 66 | [3.24, 5.34] | 0.4486 | 0.4783 | +0.0297 | 0.31 | 0.986 → 0.971 |

Learned size **helps the low-logP third by 8.3% and improves its validity**, while costing
a non-significant amount at mid and high. It is a redistribution, not a gain.

**How much to believe it.** In favour: the prediction was made *before* the run (Wave 1 was
justified on "size conditioning should help disproportionately at the low end, which is
where prob-space blending did nothing"), and the sign replicates independently in both
seeds across all three thirds. Against: it is one subgroup of three, and p=0.035 becomes
0.105 under Bonferroni. Treat as **suggestive and worth one confirmatory low-end run**,
not as established.

This partly walks back the "low end is a pure capacity wall" reading from Wave 2. Some of
that failure was the base being handed the wrong atom count — and note validity *rises* at
the low end with the correct size, while falling at the high end, where the learned model
asks for bigger molecules that break more easily under strong guidance.

**Also established here:** the pipeline is **exactly deterministic** (marginal/seed42
reproduced the earlier identical-config run to 1e-9), and the **seed spread is 0.0082** —
so Wave 2's −16% headline (Δ0.103) is ~12× run-to-run noise.

## Wave 1 (original text) — finish what is already built

**Run the node-count ablation pair** on `clogp@1.2.0`, validation split: `--size-mode
marginal` vs `--size-mode learned`. The code, the fitted models and the harness flag all
exist; this is compute, not development.

Report MAE, validity **and distinct-molecules-per-target** side by side. A sharpened size
distribution is exactly the kind of change that buys MAE with diversity, and the paired rows
are what stop the improvement being read as "the adapter got better."

Why first: zero risk, zero new code, and it is simultaneously a paper row and a shipped
feature. It also settles how much of the remaining gap size conditioning already closed,
which changes how much the later waves have left to win.

---

## Wave 2 — DONE (2026-08-13). Result below; it changes Wave 3.

Implemented as `AdapterComposition.blend_space` (default `"rate"`, so prior numbers
reproduce). Branch `feat/prob-space-blending`, KCIST jobs 43069 / 43072.
E2 logP, validation, 100 targets × 10, adapter-only, seed 42, 500 steps, η=25, size marginal.

| blend | w | MAE | low | mid | high | validity |
|---|---|---|---|---|---|---|
| rate *(historical)* | 1.0 | 0.6453 | 0.712 | 0.593 | 0.629 | 0.991 |
| rate | 2.0 | **5.5948** | 8.385 | 6.326 | 2.049 | **0.526** |
| prob | 1.0 | 0.6410 | 0.711 | 0.599 | 0.614 | 0.997 |
| **prob** | **2.0** | **0.5420** | 0.696 | 0.470 | 0.455 | 0.982 |
| prob | 2.5 | 0.5818 | 0.813 | 0.479 | 0.480 | 0.957 |
| prob | 3.0 | 0.5943 | 0.858 | 0.459 | 0.491 | 0.898 |
| prob | 4.0 | 0.6717 | 1.237 | 0.613 | 0.584 | 0.466 |
| prob | 6.0 | 5.9468 | — | — | — | 0.031 |

**At w=1 the change is a no-op**, exactly as theory says (paired mean −0.0034, 51/100,
Wilcoxon p=0.72) — the guided distribution *is* `p_cond` there. **At w=2 rate-space is
catastrophically broken** (paired mean −5.05, prob better on 99/99, p=5.7e-18) and prob-space
is not. The rate arm runs the *unchanged* code path, and `model.py:1053` documents the
mechanism in its own docstring: "w>1 with a small R_uncond can make the log-ratio deviation
explode". The `1e5` clamp does not rescue the distribution.

**Headline: 0.6453 → 0.5420, −16.0%, for a 0.9pt validity cost.** Also ~6% faster
(1047s vs 1111s per arm: one `compute_rate_matrices` call per step instead of two). For
scale that nearly matches FK at K=64 (0.5194) without the 64-particle cost.

### The asymmetry, which is the more interesting finding

Validity **by target third** at increasing w:

| w | low | mid | high |
|---|---|---|---|
| 1.0 | 0.991 | 1.000 | 1.000 |
| 2.0 | 0.982 | 0.982 | 0.982 |
| 3.0 | 0.835 | 0.930 | 0.930 |
| 4.0 | **0.232** | 0.518 | 0.655 |

Two distinct low-end problems, not one. **(a)** Even at w=2, where validity is uniform, the
low third barely responds to guidance (0.711 → 0.696, −2%) while mid and high move −22% and
−26%. **(b)** Past w=2 the low third breaks structurally *far* faster than the high third.
Target logP spans [−3.95, 5.34] with the base centred near 2.46, so the low third is asking
the model to go below its own centre — which needs polar/heteroatom content the frozen base
evidently cannot add without wrecking the molecule.

This is evidence for the **capacity** side of the Wave 4 question, at least at the low end:
the base cannot make these molecules, and pushing harder destroys them rather than steering
them. It also matches the RL notes ("tightens HIGH-logP in all 4 seeds but not LOW-logP").

### Consequence: Wave 3's expected value just dropped

Wave 3 was ranked highest because `w=1` always won and a miscalibrated null seemed the likely
cause. **That symptom is now explained and fixed by the blend space instead.** With prob-space
we sit at an optimum of w=2, which is the same regime FreeGress's learned null reaches (their
best logP is at s=2–3). A learned null may still help, but the specific evidence that
motivated it is gone. **Demote Wave 3 below Wave 4** and re-derive the case for it before
spending 3–5 days.

## Wave 2 (original text) — the free inference-time fix

**Blend branches in probability space, not rate space.** Today `denoise_step` builds a full
rate matrix per branch — each drawing *its own* `X₁` sample (`model.py:999`) — and blends
those. FreeGress Eq. 10 combines the predicted clean-graph probabilities and derives one
transition. Ours adds sampling noise the other does not have, and is inconsistent with our
own terminal step, which already PoE-blends log-probs.

**Every already-trained adapter benefits retroactively**, because nothing is retrained. That
combination — cheap, no retraining, fixes a real inconsistency — is why it comes before
anything more interesting.

Expect a small gain. Take it anyway; it is nearly free.

---

## Wave 3 — the highest expected-value untried lever

**Replace gate-zero with a learned null token.**

This is the one I would bet on, and the argument is specific rather than general:

- FreeGress's logP MAE at guidance strength `s=1` is **0.22**. At their best `s` it is
  **0.16** — guidance strength alone is worth **27%** to them.
- In every one of our sweeps, **`w=1.0` wins and `w>1` does not help**.
- CFG extrapolates along `(cond − uncond)`. Our unconditional branch is the *frozen base* —
  a separately trained, well-calibrated model — so the difference is dominated by whatever
  the small adapter changed, and extrapolating along it mostly amplifies noise. FreeGress's
  two branches come from the same weights with a *learned* null, so the direction is
  calibrated and `s=3` is a sensible thing to do.

So we have a quantified benefit, a matching symptom on our side, and a mechanism that
explains it. That is a much sharper case than "closed loop might help."

**The one design wrinkle**: a per-adapter learned null changes what the PoE composition
anchors on. The formula becomes `log p_base + Σᵢ wᵢ [log pᵢ(·|cᵢ) − log pᵢ(·|∅ᵢ)]` — each
ratio is self-contained against its own null, with the frozen base still the additive anchor.
That should be fine, but it must be *tested*, not assumed: re-run the 4-quadrant logP×QED
composition check before shipping. It also breaks the `zero_gate_verified` exact-no-op
property, so the metadata flag and its meaning need revisiting.

Success criterion: **`w > 1` starts winning on validation.** If `w=1.0` still wins after
this, the null-calibration story is wrong and we learn something either way.

---

## Wave 4b — DONE (2026-08-15). The adapter is NOT the bottleneck.

Capacity ladder on QED, KCIST job 43124, four arms on four GPUs. Metric is the SLOPE of
achieved-vs-requested QED (1.0 perfect, 0 ignores the target), not MAE — a QED adapter
emitting the dataset mean already scores MAE ~0.15. Evaluated at w=1.0, where rate- and
prob-space blending are provably identical for a single adapter, so these numbers are
comparable with everything measured before the blend fix.

**Final eval** (128 molecules, 500 steps), targets at the 5/25/50/75/95th percentiles
(span 0.432):

| arm | params | slope | span | mean MAE | uncond |
|---|---|---|---|---|---|
| A_base h256 20ep | 4.97M | **0.369** | 0.154 | 0.1252 | 0.748 |
| B_wide h1024 20ep | 20.6M | **0.323** | 0.154 | 0.1284 | 0.720 |

| hypothesis | verdict | evidence |
|---|---|---|
| trunk too small | **no** | 4x width, identical 0.154 span; slope 0.369 -> 0.323 (final evals) |
| undertrained | **NOT RESOLVED** | 10 probes, trend +0.0043/epoch, **p=0.118** — underpowered |
| modulation too shallow | **no** | matched-epoch arm spread 0.051 < probe noise 0.063 |

**Correction.** This table first read "undertrained: no — flat". That was wrong, and the
way it was wrong is worth keeping: the analysis script judged the trend against a hard
threshold (`m > 0.002`) with no error bar, so when a tenth probe arrived at epoch 49 the
verdict flipped from FLAT to RISING on one observation. Tested properly, C_long's trend is
+0.00425/epoch with se 0.00242, **p=0.118**, R2=0.28, and drops to +0.00164 (p=0.51) if the
last point is removed. That is not evidence of flatness — it is too little power to tell.
The point estimate, if real, would be worth +0.19 slope over 45 epochs, which is large.
D_attn over the same range gives +0.00127, p=0.367.

So: **width and depth are ruled out on their own evidence; training length is still open**,
and it is open specifically on the arm that got killed.

Going from 4.97M to 27.7M trainable parameters — from half the frozen base's size to
nearly 3x it — and from 20 to 44 epochs moves QED range recovery not at all. It sits at
~36% of the requested range throughout. **This is the measurement the capacity argument
was missing, and it says the adapter is not the constraint.**

### What this result cannot support

`C_long` and `D_attn` were killed by the SLURM wall mid-training with **no final eval and
no saved adapter**, so their conclusions rest on probe data (31 molecules, 100 steps).
The calibration check earned its place: on the two arms with both, the probe-vs-final
offsets go in OPPOSITE directions (A_base +0.112, B_wide −0.131), a 0.24 disagreement on
a quantity of ~0.35. So a single probe cannot stand in for a final eval, and no point
estimate is reported for C/D — only the 9-point trend, which averages the noise down.

The depth verdict is "null below noise", not "interior attention cannot help". Single
seed throughout.

**Cause of the loss, for next time:** `MAX_TIME_HOURS=9.5` against a 12h SLURM wall left
no margin, Lightning's cap did not fire, and the adapter is only saved after training
completes. Set the training cap well under the wall (7h against 12h) AND checkpoint
periodically, so a kill costs the tail rather than everything.

### Consequence

Wave 4a (the adaptive-`w` probe) is now the only untested control lever, and **Step 2 —
the unfrozen "cheating model" oracle — is the clear next move**: it measures what the
frozen-base design actually costs, which is the number the paper is missing.

## Wave 4 (original text) — one week to decide: control or capacity?

Two experiments, run together. They answer the question the rest of the plan hinges on, and
neither produces a shippable artifact — which is exactly why they are easy to skip and
shouldn't be.

**4a. Adaptive-`w` probe** (1 day, no training). Hand-coded closed-loop controller using
machinery that already exists: at each step, read the property off the argmax'd predicted
clean graph with the existing head, and adjust `w` by the residual — `w_t = clip(w₀ + k·e)`.
This is FK with K=1 and a deterministic rule instead of resampling. CFG-Ctrl formalises the
framing (vanilla CFG *is* a fixed-gain proportional controller) and gives a sign-switching law
to try after plain P-control.

*Read it by target third.* If the diagnosis is right, **the tails improve and the middle does
not**. A uniform improvement across all three thirds means it was extra capacity, not
feedback — agree that reading in advance, because a uniform gain is easy to over-read as
confirmation.

**4b. Capacity oracle** (2–3 days). Train one adapter with much more capacity and much longer
— hidden 256 → 1024, 20 → 100 epochs. Not to ship: to find the ceiling of the frozen-base
design. If MAE barely moves, capacity is *not* the binding constraint at the adapter level
and control fixes have room to work. If it moves a lot, we have been optimising the wrong
thing and the frozen base is the wall.

**Decision rule.** 4a positive → build Wave 5. 4a flat but 4b moves → stop pursuing control,
put the effort into capacity. Both flat → the frozen base is the ceiling; see the strategy
note below.

---

## Wave 5 — the closed-loop adapter, only on a green Wave 4

Build in this order, and do not skip the middle step.

1. **Architecture.** `AdaLNAdapter.forward(c, t, ctx)` with `ctx = [target, target −
   head(argmax(Ĝ₁))]` from the previous step, detached. `cond_in += 2`, config field,
   package-format bump. The readout **must** be the trained head on a discretised graph —
   `RESEARCH.md` §3.1 measured that a closed-form linear readout gives ±0.72 on logP, useless
   at the precision we need, and §3.2 records that soft-input coupling already failed once.
2. **50% dropout of `ctx`** during training (Analog Bits). Non-negotiable: it is what gives
   you both a with-feedback and a without-feedback agent from one set of weights, which the
   next step requires.
3. **Reinforced Conditioning** (TReC). Paired advantage `A = clip(R(with) − R(without), ±ε)`,
   REINFORCE, added to the denoising loss, with `R = −|c − property(decode(argmax(pred)))|`
   from RDKit. Our GDPO work already uses common random numbers and paired evaluation —
   precisely the variance reduction this needs.

**Do not ship step 1 alone.** TReC's Figure 1a shows the advantage of feedback conditioning
rising and then falling *below zero* with more training: without the RC term, the model learns
to copy the fed-back value and marginalise the actual state. Supervised-only L3 is ~15% and
likely to *degrade*.

Cost: ~2× training time (an extra frozen forward per batch), and every adapter now needs a
head trained first.

---

## Wave 6 — only if Wave 5 underdelivers

ControlNet-XS gains 14–29% relative on FID by **rewiring for high feedback bandwidth**, not
by adding a scalar. Our L3 loop is one scalar per denoising step, the lowest bandwidth the
design admits. If L3 works but disappoints, their result says the next lever is *more
injection sites per step*, not a better scalar.

---

## Where shipping and the paper pull apart

They agree through Wave 3 and diverge after.

- **For shipping**, Waves 4–5 are straightforwardly worth trying: a 0.52 → 0.42 improvement is
  real to users, and there is no deadline forcing a freeze.
- **For the paper**, Wave 5 is a risk. A half-finished mechanism is worse in a table than no
  mechanism, and it delays the one-shot test pass by weeks for a ~50% shot at a modest gain.

**Recommended freeze point: after Wave 3, gated on Wave 4a/4b being run but not acted on.**
Waves 1–3 are near-certain and cheap; freeze the configuration there, run the test pass, and
write the table. Wave 4 costs one week, produces no artifact needing validation, and its
*findings* can go in the paper as analysis without touching the frozen config. Wave 5 becomes
either future work or a second paper.

### A strategic note worth taking seriously

**We probably cannot beat 0.16, and the paper does not need to.** FreeGress trains 16M
parameters for 1000 epochs with the guide present; we train a 5.4M adapter for 20 epochs over
a base whose representation was never shaped to make property information accessible. The
capacity argument likely dominates, and no amount of control fixes it.

The more defensible contribution is the one this investigation actually produced: a **clean
characterisation of where and why frozen-base adapter conditioning plateaus on property
targeting** — the open-loop diagnosis, the measurements behind it, the node-count result, and
whichever of Waves 2–3 pays off — set against a full-conditional baseline that we do not
pretend to match. That is honest, it is useful to anyone else building swappable adapters, and
it does not depend on a ~50% bet landing.

Wave 4b is what turns that from an argument into a result. It is three days.

---

## Immediate next actions

1. Run the Wave 1 ablation pair on `clogp@1.2.0`, validation. (compute only)
2. Implement Wave 2 probability-space blending; re-score existing adapters. (no retraining)
3. Implement Wave 3 learned null token; sweep `w ∈ {1, 2, 3}` and re-run the composition check.
4. Run 4a and 4b in parallel; apply the decision rule above.
5. Freeze, test pass, write the table.
