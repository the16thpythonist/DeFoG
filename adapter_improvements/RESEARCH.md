# Adapter improvements: why property targeting plateaus, and what to do about it

Working research note, 2026-08-13. Branch `feat/kekulized-foundation`.

**The question.** FreeGress reports logP MAE **0.16** on ZINC-250k. Our best E2 validation
number is **~0.52** (FK, K=64). This document works out where that 3x gap comes from,
rules out one explanation I initially believed, and lays out what to build.

Related memory: `learned-size-conditioning`, `e2-property-targeting`, `freegress-paper`,
`fingerprint-steering-ceiling`. Papers in `./papers/`.

---

## Part 1 — What FreeGress actually does

**Ninniri, Podda & Bacciu, ECML PKDD 2024** — `papers/2312.17397-*.pdf`

DiGress with classifier-free instead of classifier-based guidance. The target `y` is fed to
the denoiser during training with conditional-dropout rate `ρ`; a guidance strength `s`
mixes conditional and unconditional predictions at inference. No auxiliary regressor, so
half DiGress's parameters (16M vs 32M on ZINC).

**Mechanism (§3, Fig. 2a).** `y` is concatenated onto all three streams — `X`, `E`, and the
global vector `u` — before the input projection, then processed by 12 graph-transformer
layers. The null token `ȳ` is a **learned parameter**, not zeros. Guidance is applied to the
predicted clean-graph probabilities (Eq. 10):

```
p_θ(x⁰|G^t) + s·( p_θ(x⁰|G^t, y) − p_θ(x⁰|G^t) )        s ∈ {1,2,3}
```

Eq. 11 is a log-space variant, which switches the loss from CE to NLL.

**Protocol (§4.2, verbatim).** "we randomly sampled 100 molecules from the dataset ... we
used the vectors 10 times each ... for a total of 1000 generated molecules."
MAE = `(1/1000) Σᵢ¹⁰⁰ Σⱼ¹⁰ |yᵢ − ŷᵢⱼ|`. **No split is specified** — our harness drawing from
validation/test with a recorded seed is stricter than the source.

### ZINC-250k Table 2

| | logP MAE | Val. | QED MAE | Val. | logP+QED | Val. |
|---|---|---|---|---|---|---|
| Unconditional | 1.52 | 86.1% | 0.15 | 86.1% | 0.83 | 86.1% |
| DiGress best | 0.74 (λ=400) | 74.6% | 0.14 (λ=600) | 84.5% | 0.35 | 75.5% |
| FreeGress best | **0.16** (ρ=.2,s=3) | 81.2% | **0.04** (ρ=.1,s=2) | 84.9% | **0.12** | 80.7% |

**Not apples-to-apples.** Their ZINC strips stereochemistry and all charged atoms except
N+/O- (kept as distinct types), leaving **228k** molecules. Their regime sits at ~86%
validity where ours is ~99%. Some of the gap is bought with validity we defend and they
don't — but 0.16 vs 0.52 is far too large for preprocessing to explain.

### §3.1 — node-count conditioning

`n ~ p_ξ(n|y)`, a 2×512 ReLU MLP with softmax output, replacing the dataset marginal.
Table 3: lifts FreeGress validity by up to +57% on MW targeting, and *destroys* DiGress.
**We have now built this** — see Part 6.

---

## Part 2 — Diagnosis

### 2.1 A claim I made and then retracted

I first argued: *our modulation is graph-agnostic, FreeGress's is position-resolved.*
**This is false.** `layers.py:230-232` and `249-251`, inside the base's own attention:

```python
ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
newE = ye1 + (ye2 + 1) * newE          # broadcast over BOTH (i,j)

yx1 = self.y_x_add(y).unsqueeze(1)
yx2 = self.y_x_mul(y).unsqueeze(1)
newX = yx1 + (yx2 + 1) * weighted_V    # broadcast over every node i
```

That is DiGress's native global→node/edge FiLM, and it is **exactly the channel FreeGress
uses** to distribute its guide. Their conditioning is as node-independent as ours.

Their input concatenation is not position-resolved either: `h_i = W_x·x_i + W_y·y + b`, and
`W_y·y` is identical for every `i` — a uniform translation. Ours is
`X_i ← (1 + g⊙s)⊙X_i + g⊙b`: a uniform diagonal scale **plus** a uniform translation. At the
injection point **our adapter is the richer of the two.**

### 2.2 The real difference: open loop vs closed loop

`adapter.py:350`:

```python
def forward(self, c: torch.Tensor, t: Optional[torch.Tensor] = None) -> Modulation:
```

It never sees `X`, `E`, or `node_mask`. Every scale, shift and gate at every layer is
computed **before looking at the graph**, and the same graph-blind table is re-injected at
all 9 layers. **The adapter is an open-loop controller.**

FreeGress's per-layer signal is also a uniform shift at layer 0 — but by layer 1 it has
passed through attention whose `Q·K` products contain cross-terms between the y-component
and node content, with weights **trained with `y` present**. By layer 3 the condition-derived
signal at node `i` genuinely differs from the one at node `j`, as a *learned* function of
local structure.

**Why this binds for logP.** Wildman–Crippen logP is `Σᵢ c(atom type, environment)` — a
global sum over local contributions. The correct action depends on the running total:
partial graph at 4.2 with target 3.5 → add polar atoms; at 2.8 → add lipophilic ones.
Opposite actions, identical condition vector. An open-loop controller emits the same push.

**This predicts the measured failure signature.** Open-loop shifts a distribution's centre
without tracking a target — bias without tracking. Which is what E2 shows: QED achieved
spread ~0.12 of a requested 0.432, unchanged from epoch 4 to epoch 19; adapter holds the
middle (mid 0.104) and fails the ends (low 0.191). More epochs cannot turn an open-loop
controller into a tracking one.

**Honest caveat.** DiT's adaLN-Zero is equally state-blind and is state of the art. Open-loop
conditioning is not fatal in general. "Produce something from category k" is a distributional
constraint an open-loop bias satisfies; "produce something whose scalar functional equals v"
needs the running value. Same mechanism, different demands.

### 2.3 Why we cannot simply do what FreeGress does

DiGress's global stream **already is** a closed loop — it accumulates pooled `X` and pooled
`E` every layer (`layers.py:255-258`) and broadcasts back via `y_x_mul`/`y_e_mul`. FreeGress
gets feedback for free by putting its target where the loop already runs, and training the
routing.

That routing is frozen for us, and it is not small:

```
layers=12  dx=384  de=96  dy=96
total model params      25,922,897
y-stream params          9,086,208   = 35.1%
adapter params           5,412,352
```

The disqualifier is not cost. Unfreezing shared weights gives **a different base per
property** — nothing left to product-of-experts over, so composition, stacking and the
4-quadrant result all go away. An external readout into the adapter trunk is the only way to
close the loop while keeping the base shared and the adapters composable.

Note the adapter **already modulates the `y` stream** (`_STREAMS = ("X","E","y")`), applying a
graph-blind affine to a vector that carries the graph summary. It can rescale the running
summary; it cannot make its coefficients depend on it. That is the one-line gap L3 closes.

---

## Part 3 — Measurements

### 3.1 Can the property be read off soft marginals? (mostly no)

Linear regression of property on 9 atom-type counts + 3 bond-type counts + n, 60k ZINC
molecules. This bounds any closed-form `E[property]` readout, since `E[Σ] = Σ E[·]` only
holds for a linear map.

| property | R² | MAE | MAE/std |
|---|---|---|---|
| TPSA | 0.852 | 6.76 | 0.30 |
| **logP** | **0.536** | **0.72** | 0.53 |
| QED | 0.201 | 0.099 | 0.71 |

**A logP thermometer with ±0.72 error is useless** when the target precision is 0.16–0.5.
The cheap readout works for TPSA, fails for logP and QED.

**Consequence:** the readout must be the trained `PropertyHead` (logP MAE 0.084, corr 0.998)
applied to the **argmax'd** predicted-clean graph — exactly what FK's `_predict_clean` already
does. Non-differentiable, which does not matter: the reading is an *input*, so `detach()` it.

### 3.2 Prior failure worth remembering

Soft-input self-consistency coupling **failed**. FK works only because it feeds the head a
discrete argmax'd graph, keeping it in distribution. Evaluating a nonlinear GNN at the mean
of a distribution is not the mean of the GNN. Any closed-loop design must preserve the
discretisation.

---

## Part 4 — Prior art

### 4.1 TReC — the recipe for the training problem
`papers/2402.14843-*.pdf` · AAAI 2024 · **the most important paper here**

Diagnosed our exact training problem in text diffusion and named it **"degradation of
self-conditioning."** Definition 1: the denoiser is *degraded* if it **marginalises `z_t`** —
learns to copy the fed-back prediction and ignore the current state. Why:

> the self-condition denoising step `f_θ(z_t, ẑ₀, x, t)` could easily achieve a low loss by
> simply copying `ẑ₀` as its output, as reconstruction from `ẑ₀` to `z₀^SC` becomes
> substantially easier when `ẑ₀`'s quality increases progressively

**Worse than expected: Fig. 1a shows the advantage of self-conditioning rising, then falling
below zero during training.** Left alone, feedback conditioning becomes actively harmful.
Fig. 1b confirms it: outputs given the same `ẑ₀` overlap regardless of `z_t`.

**Reinforced Conditioning.** Two agents **sharing the same parameters**, differing only in
whether the feedback is fed:

```
A(z₀^SC, ẑ₀) = clip( R(z₀^SC) − R(ẑ₀), −ε, +ε )      # Eq. 10, paired advantage
L_RL = −E[A]                                          # Eq. 11
∇L_RL ≈ −(1/N) Σ Aᵢ · ∇ log p_θ(y | z₀^SC)            # Eq. 12, REINFORCE
L_total = L_diffusion + L_RL                          # 50% SC rate, from Analog Bits
```

**This answers the objection that killed supervised L3.** The concern was that denoising
provides no course-correction signal, since every training trajectory ends at the right
answer. RC's reward is not about the endpoint — it is a paired comparison *at the same noisy
state*: did feeding the feedback beat not feeding it? Available in every batch.

For us `R = −|c − property(decode(argmax(pred)))|` via RDKit — easier than their BLEU.

Ablation (QQP, Table 2): RC alone contributes **+1.24 BLEU** (33.19 vs 31.95).

### 4.2 ControlNet-XS — the architecture precedent
`papers/2312.06573-*.pdf`

Literally *"Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control
Systems."* Frozen base, trainable control module: **20M params against the 2.6B SD-XL base**,
*"i.e. less than 1% of parameters of the generative model"* (§4.5). On SD-1.5 the headline
comparison is **55M against ControlNet's 361M** on an 860M base (Table 1).

Their diagnosis is the lesson: ControlNet's feedback enters only at the decoder, so
corrections arrive late — *"delay is the most unwanted aspect of any control system."* Fixed
by bandwidth: encoder→controller **and** controller→encoder connections. Better FID than
ControlNet with far fewer parameters, ~2× faster.

Caveat: their feedback is the base's *internal features* (an L1-style state readout), not a
task error. Our loop is very low-bandwidth by their standard — one scalar per denoising step.

### 4.3 CFG-Ctrl — formalises the cheap first experiment
`papers/2603.03281-*.pdf`

With error signal `e(t) = v_θ(x_t,t,c) − v_θ(x_t,t,∅)`, **vanilla CFG is a proportional
controller with fixed gain `w`**. They replace it with sliding-mode control:
`s(t) = ė(t) + λe(t)`, `Δe = −K·sign(s(t))`. Inference-time only, nothing learned.
FID 21.42 → 20.04 on SD3.5.

Two notes. It **improves robustness at high guidance scales** — our exact symptom (`w=1.0`
always wins). But their error is `cond − uncond`, an *internal* discrepancy, still open-loop
w.r.t. the objective. Ours would be a genuine task error: the interesting part, and a reason
their gains may not transfer directly.

### 4.4 Supporting
- **Analog Bits** (`2208.04202`) — origin of self-conditioning; the **50% dropout** of the
  fed-back input during training. Non-negotiable: it is what yields both agents from one set
  of weights.
- **Reward-Conditioned Policies** (`1912.13465`) — return-to-go is the same structure
  (condition on the remaining target). **Do not cite this paper for an OOD-target failure
  mode: it claims the opposite**, that conditioning on the reward value yields "a policy
  that generalizes to larger returns" (abstract). It only ever trains and evaluates
  in-support — §4.3.1 trains "only for target return values Z that have actually been
  observed", §4.2 evaluates at the single value `μ_Z + σ_Z` — so our extreme-target regime is
  simply untested here. Its nearest caveat is §6: the methods "rely on generalization and
  random chance ... Sometimes the reward-conditioned policies might generalize successfully,
  and sometimes they might not." The OOD-target failure mode is documented by later work
  (Brandfonbrener et al., *When does return-conditioned supervised learning work?*).
- **DDPO** (`2305.13301`) — RL for diffusion as a multi-step MDP; the family our GDPO work
  sits in.

### 4.5 What does not appear to exist

Nobody combines **frozen base + swappable adapter + conditioning on a measured task-property
error + composable across adapters**. Molecular work is either full-model conditional
(FreeGress, DriftingMol) or inference-time guidance (SILVR, retrieval refinement). Every
component is established; the combination is not.

---

## Part 5 — Proposed work

### The ladder

`build_modulation(bs, t)` is already called **inside** the denoise step (`model.py:987`), so a
per-step readout needs no change to the call cadence — only `forward(c, t)` →
`forward(c, t, ctx)` and a wider `cond_in`.

| | `ctx` | Cost | Notes |
|---|---|---|---|
| **L1** | pooled `X_t`/`E_t`, or the base's `y_t` | free | `G_t` in hand before any forward. |
| **L2** | readout of `Ĝ₁` from the uncond branch | free at sampling | CFG already computes that pass. This is self-conditioning. |
| **L3** | `target − Ê[property]` — the **error signal** | free on top of L2 | 1–2 extra dims. |

L3 is the one worth building: L1/L2 hand the adapter raw state and hope it extracts the
scalar; L3 hands it what a controller needs.

### Staged recipe

**Stage 0 — adaptive `w` (one afternoon, no training).** CFG-Ctrl's framing with
`e = target − head(argmax(x̂₀))`; `w_t = clip(w₀ + k·e)`. P-control first, then their
sign-switching law. Essentially FK with K=1 and a deterministic rule instead of resampling.
**This is the go/no-go gate.** If a hand-coded thermostat improves the tails, build the
learned version; if not, L3 almost certainly won't either.

**Stage 1 — architecture.** `AdaLNAdapter.forward(c, t, ctx)`, `cond_in += 2`, config field,
package-format bump. Carry last step's reading forward; zeros at step 0. Analog Bits' **50%
dropout of `ctx`** during training. Base stays frozen; existing adapters stay valid.

**Stage 2 — training.** TReC's Reinforced Conditioning verbatim: paired advantage, clipped,
REINFORCE, added to the denoising loss. Our GDPO work already uses common random numbers and
paired eval — precisely the variance reduction this needs.

Training cost: one extra frozen forward per batch for the uncond prediction, ≈**2× training
time**. Every adapter now requires a head trained first.

### Falsification

If the open-loop diagnosis is right, **the low and high thirds should improve and the middle
should not.** `e2_targeting.py` already reports MAE by target third. A uniform improvement
across all three means it was extra capacity, not feedback — agree this in advance, because a
uniform gain is easy to over-read as confirmation.

---

## Part 6 — Already shipped: learned node-count conditioning

FreeGress §3.1, built 2026-08-13. `LearnedSizeDistribution`, `SizeBranch`,
`ComposedSizeDistribution` in `defog/core/size_distribution.py`; `fit_size_model` in
`property_head.py`; `--with-size-model` in `scripts/train_property_head.py`; `--size-mode` in
`scripts/e2_targeting.py`; full `size_model` package path in molsmith.

Every E2 number before this drew node counts from the **unconditional marginal**.

| property | corr(y, n) | gain_nats | shrink |
|---|---|---|---|
| logP | +0.398 | 0.098 | 0.910 |
| QED | −0.321 | 0.165 | 0.945 |
| TPSA | +0.457 | 0.127 | 0.886 |

`E[n | logP]` runs 20.2 → 26.7 heavy atoms across deciles against a marginal of 23.2. But
`shrink` stays ~0.9: **a bias correction on the size draw, not a variance reduction** — the
same signature as the open-loop adapter.

**Composition: use `product`, not `mean`.** I predicted `mean` (double-counting a shared
size signal) and was wrong twice. Matched-resolution bucketing: joint gap 0.236 nats vs
sum-of-singles 0.266 vs best-single 0.169 → **89% additive**, because logP and QED pull size
in *opposite* directions. Direct oracle: `KL(joint‖product) 0.0371` beats
`KL(joint‖mean) 0.0660`; held-out NLL 2.7045 / 2.7309 / 2.6716 (joint) / 2.9219 (marginal).
Per-adapter models composed by product recover **87% of a joint model's gain**.

Still to run: the E2 ablation pair against `clogp@1.2.0`.

---

## Part 6b — Correction: what rate-space blending actually does

Written after implementing Wave 2. The argument in Part 5 called rate-space blending
"extra estimator noise". That understates it.

`rate_matrix.py:104` builds the rate matrix from a **discrete sample** of the clean graph,
not from the predicted marginal:

```python
# Sample x_1 from predicted distribution
sampled = sample_from_probs(X_1_pred, E_1_pred, node_mask)
```

So the two placements are:

| | what it does |
|---|---|
| `rate` | draw `X₁` from `p_uncond`, draw `X₁` from `p_cond`, build a rate matrix from each, take their geometric mean |
| `prob` | blend the marginals into the guided `q`, draw `X₁` **once from `q`**, build one rate matrix |

Only the second is what classifier-free guidance means. The first averages two rate
matrices derived from two different discrete samples of two different distributions.

**This also makes the change hard to measure, in a way worth recording.** The two paths
consume the random stream a different number of times (N+1 draws vs 1), so naive
comparisons measure RNG alignment rather than the math — three of my first attempts did
exactly that and disagreed with each other. Pinning the draw makes them agree *exactly* at
every `w`, which is equally misleading: with one uniform draw fixed the sampled `X₁` stops
moving with `p`, so `R` is locally constant. The difference is distributional and has to be
measured as one. Two hypotheses died here and are pinned as tests: prob-space does **not**
re-open transitions the base vetoes (those zeros are structural), and the paths are **not**
pointwise equal at `w=1`.

## Part 7 — Calibration

| | odds | note |
|---|---|---|
| Stage 0 adaptive-`w` improves tails | ~50% | crude, but decisive and cheap |
| L3, supervised only | **~15%** | TReC says it will likely *degrade* with training |
| L3 + Reinforced Conditioning | **~50%** | proven mechanism in another modality |

**Expect a modest win at best.** The two closest analogues gain **~4–6% relative** — TReC
+1.24 BLEU (33.19 vs 31.95, Table 2), CFG-Ctrl FID 21.421 → 20.044. Both add a feedback
*term* to an otherwise unchanged pipeline, which is what we would be doing.

ControlNet-XS is the apparent counterexample and gains far more — depth FID 19.01 → 16.36
(13.9%), canny 21.18 → 15.13 (28.6%), with the paper calling it *"the tremendous gain in
FID"*. But it earns that by **rewiring the control architecture for high feedback bandwidth**,
not by adding a scalar. Read the right way this argues *against* optimism for L3 rather than
for it: our loop is one scalar per denoising step, the lowest bandwidth the design admits,
while their result says bandwidth is where the large gains live. If the tail improvement from
L3 turns out to be small, ControlNet-XS points at the next lever — inject the feedback at
more sites per step, not a better scalar.

If L3 works here, expect logP MAE ≈ 0.52 → 0.42, **not** → 0.16.

**The remaining gap to FreeGress is very likely capacity, not control.** They train 16M
parameters for 1000 epochs with the guide present; we train a 5.4M adapter for 20 epochs over
a base whose representation was never shaped to make property information accessible. No
amount of feedback fixes that. The honest ceiling for the frozen-base design should be
established before more is invested in closing the loop.

Two other levers from the same analysis, not yet pursued:
- **A learned null token** instead of gate-zero, so both CFG branches come from the adapter.
  Currently the unconditional branch is the frozen base, so `(cond − uncond)` is dominated by
  whatever the small adapter changed. That `w=1.0` always wins is a symptom of a
  poorly-calibrated guidance direction, not a property of the task.
- **Blend in probability space** rather than rate space, matching FreeGress Eq. 10/11.
  **Implemented and under test — see `PLAN.md` Wave 2 and the note below.**

---

## Papers

| file | why |
|---|---|
| `2312.17397-FreeGress-*` | the baseline; Table 2, the protocol, §3.1 node counts |
| `2402.14843-TReC-*` | **the training recipe**; self-conditioning degradation + the fix |
| `2312.06573-ControlNet-XS-*` | frozen-base feedback-control adapter; the delay/bandwidth lesson |
| `2603.03281-CFG-Ctrl-*` | CFG as a P-controller; the adaptive-`w` experiment |
| `2208.04202-Analog-Bits-*` | self-conditioning origin; the 50% dropout trick |
| `1912.13465-Reward-Conditioned-Policies` | return-to-go conditioning; in-support targets only |
| `2305.13301-DDPO-*` | RL for diffusion; the family GDPO sits in |
