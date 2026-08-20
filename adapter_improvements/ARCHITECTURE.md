# Architectural and training-side upgrades for the frozen-base adapter

Literature survey + design, 2026-08-20. Branch `feat/dam-rl`.
Companion to [`RESEARCH.md`](RESEARCH.md) (the diagnosis) and [`PLAN.md`](PLAN.md) (the ordering).

> **Nothing in this document is measured on our stack.** Every number quoted is from the cited
> paper, on that paper's own task. The predictions marked *odds* are my estimates, written down
> before any run so they can be scored afterwards.

**Constraint envelope, agreed before the search:**

| | |
|---|---|
| Base | stays **bit-identical and shared**. No re-pretraining, no per-property base weights. |
| Composition | PoE stacking and hot-swapping must survive exactly as today. |
| Timing | must be buildable **and validated before the E2 test freeze**. Rank by odds × cheapness. |
| Success bar | any solid, replicable improvement. 0.52 → ~0.42 on logP counts as a win. |

Every lead below is adapter-side. Anything that needed to touch the base was dropped during the
search rather than reported, except where it appears in the post-freeze appendix and is explicitly
marked as inadmissible-as-published.

---

## The one-paragraph version

Six leads survive the constraints, and the highest-value one is **not architectural**: we own a
conditional model that beats the adapter, and we are not using it as a teacher. Distilling it into
the adapter attacks the gap directly, and because the residual distillation error *is* the adapter
family's own expressiveness floor, it answers PLAN.md's Wave-4b capacity-vs-control question as a
by-product — the separate "cheating model" oracle does not need to be built. Two cheap architectural
fixes should ride along regardless, because both are near-free and both are things we would be
embarrassed to have left undone: the property enters the trunk as **one raw scalar** while flow-time
gets a 64-dimensional sinusoidal embedding, which is precisely the spectral-bias setup Fourier
features exist to fix; and our modulation is **diagonal**, which a per-sample low-rank term
generalises for ~+8% adapter parameters with the zero-init exact-no-op preserved. Two further leads
are post-freeze: node-resolved decoupled cross-attention, and the Wave-5 closed loop — for which the
2026 discrete-diffusion literature **contradicts one item our current spec marks "non-negotiable."**

---

## Part 0 — Two things I need before this is actionable

1. **The in-house conditional number.** Which property, which split, how many targets, and the
   validity alongside it. Everything below is ranked against "0.52 adapter vs 0.16 FreeGress";
   if our own conditional model sits at, say, 0.30, the gap being explained is half as large and
   L4's ceiling moves with it.
2. **Whether the teacher shares the base's representation.** `experiments/conditional_training__zinc.py`
   matches the base architecture (`N_LAYERS=9`, `HIDDEN_DIM=256`, `COND_DROP_PROB=0.1`) but shows no
   `KEKULIZE` / `VOCABULARY` binding, which suggests it predates the kekulized lineage. Distillation
   requires the teacher and `zinc_kek_base` to agree on atom/bond class **order**, not just on
   chemistry — the same trap `adapter_training__zinc.py:120-126` already guards against. If the
   teacher is aromatic-representation, it must be retrained on the kekulized vocabulary first, which
   moves L4 from ~1 week to ~2 and probably out of the pre-freeze window.

Measured facts this note relies on, read out of `ckpts/logp_adapter_preRL_dBe2.ckpt`:
`cond_dim=1, n_layers=9, dims={dx:256, de:64, dy:64}, hidden=256, time_emb_dim=64`, **2,747,778
parameters**. (RESEARCH.md §2.3's `layers=12 dx=384` block is DiGress/FreeGress's configuration, not
ours.)

---

## Part 1 — Prior art

### 1.1 AGD — adapters that carry a guidance-shaped correction over a frozen base
`arXiv:2503.07274`, Perez Jensen & Sadat, **TMLR 10/2025**

Distils classifier-free guidance into lightweight adapters so CFG runs in one forward pass instead
of two. Base frozen; adapters are **0.2–3.2M parameters, ~2% of the model**; two designs tried,
*cross-attention adapters* and *offset adapters*. Loss is plain **L2 against cached CFG
predictions**. Reaches comparable or better FID than CFG at half the NFEs, distils a 2.6B model on
one 24GB GPU, and composes with IP-Adapter and ControlNet.

**The finding that matters to us is not the speedup.** It is their diagnosis of prior work: the
adapter must be trained on **CFG-guided trajectories, not standard diffusion trajectories**, because
training on one state distribution and deploying on another is a train/inference mismatch. That is
exactly our situation — `AdapterModule.training_step` (`adapter.py:608-624`) draws states from
`base._apply_noise(X1, ...)`, the *forward* kernel on real data, while at deployment the adapter sees
states produced by its own w=2 PoE-blended sampler.

*Honest repurposing note:* AGD distils a model's own CFG into an adapter over that same model. We
would distil a **different, stronger** conditional model into an adapter over a frozen unconditional
base. The transferable claims are (a) ~2% extra parameters can carry a guidance-shaped correction,
and (b) the state distribution you train on is a first-order design choice. The claim that *our*
adapter family can absorb *our* teacher is not established by their result.

### 1.2 TC-LoRA — the case against activation-space conditioning
`arXiv:2510.09561`

A hypernetwork emits LoRA weights as an explicit function of timestep **and** condition, rather than
learning one static adapter. Compared against ControlNet-style static conditioning, FiLM/adaLN
activation-space conditioning, and non-temporal LoRA. Their argument, in their words: weight-space
adaptation lets the model *restructure its computational pathways*, whereas activation-space methods
*only modulate feature magnitudes*.

That is a precise description of what our adapter is allowed to do and what it isn't. It is also the
one architectural critique in the literature that lands squarely on our design without requiring the
adapter to see the graph.

### 1.3 IP-Adapter and Att-Adapter — decoupled cross-attention for a frozen backbone
`arXiv:2308.06721` (IP-Adapter) · `arXiv:2503.11937` (Att-Adapter)

IP-Adapter adds a **separate** cross-attention layer per existing attention layer, trains only the
new K/V projections, keeps the backbone frozen, and zero-initialises the output projection so the
model *begins as the exact pretrained backbone*. 22M parameters matches a fully fine-tuned image-prompt
model.

Att-Adapter is the more relevant of the two: it targets **multiple continuous attributes** — our
problem, not a discrete-class problem — using decoupled cross-attention plus a conditional VAE to
absorb the visual variation that a single attribute value maps onto. It reports beating **LoRA
baselines** at continuous attribute control, handles several attributes in one model, and trains from
unpaired data.

Two things transfer. First, the zero-init-output-projection convention is *identical in spirit* to
our zero-init gate, so a cross-attention branch would preserve the exact-no-op property the PoE
composition depends on. Second, cross-attention is the cheapest genuinely **node-resolved** mechanism
available: each node's own query decides how much of the condition it pulls in, which is precisely
what RESEARCH.md §2.2 says our broadcast FiLM cannot do.

### 1.4 SCMDM — self-conditioning as *post-training* adaptation, in a discrete state space
`arXiv:2604.26985v2` (6 Jun 2026), Cardei, Ta & Fioretto, UVA

The closest prior art to our Wave 5, and the only one in a discrete state space. Feeds the denoiser
its own previous clean-state prediction. During post-training the previous-step estimate isn't
available, so it is approximated with **two forward passes at the same timestep**, with
**stop-gradient** on the first — explicitly because "without stop-gradient, the model could reduce
the loss by shaping the first-pass output to assist the second pass, rather than providing a faithful
clean-state estimate." Sampling costs **no extra forwards**, since step *t+1*'s prediction is reused
at step *t*. Reported gains: OWT generative perplexity 42.89 → 23.72; QM9 molecular validity
594.2±9.5 → 618.0±13.2 (+4.01%); CIFAR-10 FID +9.12%.

**The result that contradicts our plan.** PLAN.md Wave 5 step 2 says "50% dropout of `ctx` during
training (Analog Bits). **Non-negotiable.**" SCMDM's stated second contribution is that this is
wrong *in the post-training regime*: once a converged backbone's self-generated clean-state estimates
are informative, **full self-conditioning consistently outperforms partial**, and stochastic removal
of the signal "can be detrimental" because it forces training to mix refinement updates with less
informative unconditional ones. They introduce λ ∈ [0,1] as the self-conditioning rate precisely to
test this, and default to λ=1.

Our justification for the 50% rate was that it yields both a with-feedback and a without-feedback
agent from one set of weights, which TReC's Reinforced Conditioning then needs. That justification
survives — but it is now a *cost we are paying for TReC*, not a free best practice, and it should be
ablated rather than assumed.

**Caveats before importing it.** SCMDM is for **absorbing-mask** MDMs, where the specific pathology
is that a still-masked position discards its previous clean-state estimate. DeFoG is a marginal-noise
CTMC with no absorbing mask, so the "discarded prior belief" mechanism does not transfer literally;
what transfers is the general cross-step-reuse argument and the two regime findings (λ=1, stop-grad).
And Algorithm 1 line 8 **updates θ** — it fine-tunes the pretrained denoiser. As published it is
inadmissible under our envelope. The admissible version routes the self-conditioning signal into the
**adapter** instead of the base, which is Wave 5.

### 1.5 MolGuidance — what to use as the negative branch
`arXiv:2512.12198`

Compares CFG, **autoguidance** (a deliberately degraded guide model — reduced capacity or
undertrained — replaces the unconditional branch, so generation is pushed *away from the weaker
model*) and model guidance, on SE(3)-equivariant flow matching for molecules. CFG wins property
alignment (polarizability MAE 1.27 Bohr³ vs 1.97 for GCDM, beating JODO on four of six properties,
~10% relative over vanilla) at a 2–3.4% stability cost. **Autoguidance is the balanced arm and
actually improves structural validity.**

Our group-0 branch is the frozen unconditional base. Swapping it for a degraded *adapter* — an early
checkpoint of the same run — is an autoguidance arm, and we already write those checkpoints
(`CKPT_EVERY_K`, `adapter_training__zinc.py:185`).

### 1.6 Fourier features — why a raw scalar condition is the wrong input
`arXiv:2006.10739`, Tancik et al., NeurIPS 2020

A standard MLP has *impractically slow* convergence to the high-frequency components of its target
function; passing the input through a Fourier feature mapping turns the effective NTK into a
stationary kernel with tunable bandwidth and removes the pathology. This is textbook, and it is the
reason every diffusion model in existence embeds its timestep sinusoidally rather than feeding the
raw float.

We do embed `t` that way — `timestep_embedding(t, 64)` at `adapter.py:356`. We do **not** do it for
the property. With `cond_dim=1`, the entire conditioning signal is one normalised scalar entering
`Linear(1+64, 256)`.

### 1.7 adaLN-Zero mechanics (minor)
`arXiv:2608.09438`

Finds adaLN-Zero's benefit comes from the zero-init aligning weights with their eventual Gaussian-like
trained distribution, and proposes **adaLN-Gaussian** (init N(0, 1e-3), ~46% less training time at
comparable quality) and **SE-adaLN-Zero** (a squeeze-and-excitation bottleneck, r=2, ~14% fewer
parameters). Relevant to training cost, not to the ceiling. Listed for completeness; not ranked.

---

## Part 2 — The leads, mapped onto our code

### L1 — Fourier-feature the condition *(cheapest thing in this document)*

**Change.** In `AdaLNAdapter.forward` (`adapter.py:350`), replace the raw normalised `c` with
`[c, sin(2πBc), cos(2πBc)]` for a fixed random or geometric frequency bank `B`, keeping the raw
scalar concatenated so nothing is lost. `cond_in` grows from 65 to ~129; the trunk's first layer
grows by ~16k parameters. Store `B` as a buffer, add the bandwidth to `_config()` and `_CONFIG_KEYS`
so old adapters keep loading with the feature off.

**Why it should work.** Our required map is `c → 27 × {scale, shift, gate}` triples. Under spectral
bias, an MLP over a 1-D input learns the low-frequency part of that map first and the fine structure
impractically slowly — which produces a modulation that shifts the distribution's centre without
resolving *where in the range* the target sits. That is the measured failure signature in
RESEARCH.md §2.2: bias without tracking, "adapter holds the middle and fails the ends".

**Honest counter-argument.** If the true `c → modulation` map is genuinely smooth and monotone,
Fourier features buy little, and an over-wide bandwidth can alias on a 1-D input. Start with a modest
bank and keep the raw channel. This is also the one lead whose mechanism *overlaps* the open-loop
diagnosis rather than contradicting it — the low-frequency prior is one plausible reason an
open-loop controller looks like it can't track.

**Effort** ~1 day + one training run per property. **Odds** 40%. **If it helps** small-to-moderate,
concentrated at the tails.

### L2 — Conditional low-rank modulation (diagonal → rank-r)

**Change.** Our modulation is `h ← h + gate ⊙ (scale ⊙ h + shift)`: a **diagonal** affine, so channel
*i* of the delta depends only on channel *i* of `h`. Generalise to

```
Δh = B · diag(γ(c,t)) · A · h + shift          B zero-initialised
```

with `A ∈ R^{r×d}` and `B ∈ R^{d×r}` **learned static bases** and only the r-dimensional core
`γ(c,t)` emitted by the trunk. This is TC-LoRA's weight-space conditioning, factorised so that the
hypernetwork output stays tiny.

**Why this form and not per-sample full LoRA.** Emitting a full `d×r` pair per (layer, stream) from a
256-wide trunk would cost ~1.6M parameters *per site*. With static bases the trunk emits only `r`
numbers per site. At r=16: static bases 110,592 + γ heads 111,024 ≈ **+222k parameters, +8% on
2.75M.**

**Why it preserves everything.** `B` zero-init ⇒ exact no-op at init ⇒ group-0 bypass is still
bit-exact ⇒ PoE composition, `bypass_rows`, and `stack_groups` are untouched. The delta is still a
per-sample tensor, so it slots into `Modulation.apply` (`adapter.py:68`) and the batched `(N+1)·B`
forward without changing the contract.

**What it does and does not buy.** It makes the response **content-dependent in a non-diagonal way** —
different nodes get different deltas as a learned function of their own features, and channels can
mix. It does **not** make the response depend on graph structure or on neighbouring nodes. It is a
strict expressiveness upgrade within the open-loop family, not an escape from it.

**Effort** ~4–5 days including the `_config`/`from_config`/package-format bump. **Odds** 35%.
**If it helps** moderate.

### L3 — Decoupled cross-attention (the genuinely node-resolved option)

**Change.** Per layer, an adapter-owned cross-attention branch: nodes query a small set of condition
tokens, `ΔX = W_o · Attn(Q = X W_q, K = c_tok W_k, V = c_tok W_v)`, `W_o` zero-initialised, added to
the frozen self-attention output at `layers.py:360`.

**Cost.** X-stream only, d=256, ~1.77M parameters over 9 layers → adapter grows to ~4.5M (+65%).
Cross-attention over a handful of tokens is negligible against the n² edge attention at sampling time.

**The one real engineering obstacle.** `Modulation` currently carries **tensors only**, which is why
the batched `(N+1)·B` forward works. A cross-attention branch needs the adapter's own *weights* to run
inside the layer, and different groups in that batch belong to different adapters. The fix is
contained: have `Modulation.layers[i]` optionally carry a per-group callable, and apply it to the
batch slice `[g·bs : (g+1)·bs]`, with group 0 receiving zero. Exact, cheap, and it keeps one forward
per step — but it *is* a change to the contract that `stack_groups` currently guarantees, and it
needs its own test.

**Effort** ~2 weeks. **Odds** 45% — the highest of the architectural options, and the only one that
addresses the diagnosis head-on. **Post-freeze.**

### L4 — Distil our own conditional model into the adapter ★

**Change.** Replace (or mix into) `AdapterModule`'s CE-against-one-hot with a KL against the
**teacher's predicted clean-graph marginals** at the same `(G_t, t, c)`:

```
L = KL( p_teacher(·|G_t,c) ‖ p_adapter(·|G_t,c) )      on X and on E, masked
```

The teacher is frozen and runs no-grad, so the step cost is one extra forward — the same overhead
`GroundedAdapterModule` already pays.

**Why this is the headline lead.** Every other item on this list tries to make the adapter *discover*
the conditional structure from a cross-entropy signal against a single sampled molecule. We already
have a model that has discovered it. CE-against-one-hot is a high-variance, low-information target;
the teacher's full conditional distribution is dense supervision at every node and edge, and it
carries exactly the thing the adapter needs to learn — *how the conditional differs from the
unconditional* — as a per-state signal rather than something recovered from averages over the dataset.
For a capacity-limited student this is the standard reason distillation beats training on the same
data.

**Guided or unguided target — this matters for composition.** AGD distils the *guided* output, which
bakes a single guidance scale into the adapter. For us that would break multi-property PoE, which
needs an un-guided conditional per branch. So:
- **Main arm:** distil the teacher's **bare conditional** and keep our w=2 prob-space PoE blend on top.
  Composition survives.
- **Side arm (single-property only):** distil the teacher's CFG output at its best `s`, deploy at
  w=1. Cannot be stacked; run it only as a ceiling probe.

**The by-product that makes this worth doing even if it fails.** The residual — how far the distilled
adapter stays from its teacher on held-out targets — **is** the expressiveness floor of open-loop
diagonal FiLM over this base, measured directly. That is Wave 4b's question, answered without building
Wave 4b's separate over-capacity oracle.

**Effort** ~1 week if the teacher's vocabulary matches, ~2 if it must be retrained (see Part 0).
**Odds** 60%. **If it helps** potentially large — this is the only lead whose ceiling is set by the
teacher rather than by our own architecture.

### L5 — Train on the states the adapter actually visits

**Change.** Fine-tune the adapter on `(G_t, t, c)` drawn from its **own** guided rollouts instead of
from `base._apply_noise` on real data. Requires a target that is defined at arbitrary states — which
is exactly what L4's teacher provides, so **L5 only makes sense stacked on L4**. With a CE-against-data
target it is not even well-posed: at a self-generated state there is no true clean graph to score
against.

**We already own the machinery.** `defog/core/renoise.py` draws states at chosen noise levels, `rl.py`
has the rollout loop, and RAM was built precisely to score at re-noised rather than trajectory states.
The cheapest form is a second phase: train with L4 on forward-noised states, then fine-tune on cached
rollout states with the same loss.

**Effort** ~3 days on top of L4. **Odds** 45% conditional on L4 being built. **If it helps** small
to moderate; AGD frames this as the difference between their method working and prior work not.

### L6 — Autoguidance negative branch *(one eval run, no training)*

**Change.** In `AdapterComposition`, allow group 0 to be a *degraded adapter* rather than the frozen
unconditional base — an early `CKPT_EVERY_K` checkpoint of the same run. `_blend_logp`
(`model.py:1068`) is unchanged; only what fills group 0 changes.

**Why.** MolGuidance found autoguidance improves structural validity where CFG costs it, which is our
exact pain point at w>2 (validity 0.982 at w=2, 0.898 at w=3, 0.466 at w=4). Pushing away from a
weak *conditional* rather than from the unconditional keeps the guidance direction inside the
conditional manifold.

**Effort** hours — the checkpoints exist. **Odds** 30%. **If it helps** small, but it may buy back
validity headroom that lets a larger `w` pay off, which compounds with everything else.

---

## Part 3 — Ranking and decision rule

| # | Lead | Effort | Retrain adapter? | Odds | Ceiling | Pre-freeze? |
|---|---|---|---|---|---|---|
| **L4** | **Distil the in-house conditional teacher** | 1–2 wk | yes | **60%** | **large** | yes, if the vocabulary matches |
| L1 | Fourier-feature the condition | 1 d | yes (cheap) | 40% | small–moderate | yes |
| L6 | Autoguidance negative branch | hours | **no** | 30% | small | yes |
| L5 | On-policy states (needs L4) | +3 d | yes | 45% | small–moderate | yes, if L4 lands |
| L2 | Conditional low-rank modulation | 4–5 d | yes | 35% | moderate | borderline |
| L3 | Decoupled cross-attention | ~2 wk | yes | 45% | moderate–large | **no** |

**Run order.** L6 today (it is an eval, not a build). L1 and L4 in parallel — they touch different
files and their effects are independent, so a 2×2 is affordable and tells us whether they compose.
L5 only if L4 lands. L2 only if the schedule holds after that. L3 and Wave 5 are post-freeze.

**Decision rule on L4 — this is the one that changes the story.**

- **Distilled adapter lands close to the teacher.** The adapter family was never the binding
  constraint; the *training signal* was. Ship it, and Wave 5 becomes unnecessary rather than merely
  risky. PLAN.md's strategic note ("we probably cannot beat 0.16 and the paper does not need to")
  gets weaker in a good way.
- **Distilled adapter plateaus well short of the teacher.** That residual is the measured
  expressiveness floor of open-loop diagonal FiLM over a frozen base. The honest paper contribution
  is then the characterisation, exactly as PLAN.md argues — but now with a *number* attached instead
  of an argument, and L2 → L3 become the indicated post-freeze fixes in that order.
- **Either way**, Wave 4b's over-capacity oracle does not need to be built, which frees the three days
  PLAN.md budgeted for it.

---

## Part 4 — Post-freeze, and one correction to the existing plan

**Wave 5 stands, with an amendment.** SCMDM (§1.4) is the closest prior art and it is in a discrete
state space. Two of its findings should be folded into the spec:

1. **Drop "50% dropout of `ctx` is non-negotiable."** In the post-training regime, full
   self-conditioning beat partial, and stochastic removal was harmful. Our reason for the 50% rate is
   that TReC's paired advantage needs a without-feedback agent — that reason is real, but it is a cost
   we pay for TReC, not a free best practice. Ablate λ ∈ {0.5, 1.0}.
2. **Keep the stop-gradient, and say why.** PLAN.md already specifies `detach()`. SCMDM gives the
   mechanism: without it, the model shapes the first-pass output to make the second pass easy instead
   of producing a faithful estimate. That is the same failure family as TReC's "degradation of
   self-conditioning" and worth stating in those terms.
3. **Inference is nearly free.** SCMDM reuses step *t+1*'s clean-state prediction at step *t* and adds
   no forwards. Our loop already computes the predicted clean graph every step; the only added cost is
   one `PropertyHead` forward. The "~2× cost" in PLAN.md is a *training* cost only.

**Also noted, out of the chosen scope.** *Feedback Guidance of Diffusion Models* (`arXiv:2506.06085`)
derives a state-dependent guidance coefficient that self-regulates by how much guidance the current
prediction needs. We excluded inference-time steering from this search, but our per-molecule weight
path — `AdapterComposition.weights` returning `(N, bs)`, `adapter.py:524-541` — already exists to carry
exactly this, and PLAN.md's Wave 4a adaptive-`w` probe is the same idea with a hand-tuned gain. If 4a
gets run, that paper is the principled version of its update rule.

---

## Part 5 — What none of this fixes

L1, L2, L4, L5 and L6 all leave the adapter **open-loop**. If RESEARCH.md §2.2 is right that logP
targeting specifically needs the *running value* — "partial graph at 4.2 with target 3.5 → add polar
atoms; at 2.8 → add lipophilic ones; opposite actions, identical condition vector" — then the ceiling
for all of them is bounded, and only L3 (node-resolved) and Wave 5 (closed-loop) attack it.

The reason to run the open-loop leads first anyway is that they are cheap, they are pre-freeze, and
**L4 measures how much of the plateau the diagnosis actually explains.** Right now "open loop" is a
mechanism story supported by a failure signature. After L4 it is a number.
