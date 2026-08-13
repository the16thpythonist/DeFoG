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

## Wave 1 — finish what is already built

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

## Wave 2 — the free inference-time fix

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

## Wave 4 — one week to decide: control or capacity?

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
