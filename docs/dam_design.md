# Three-arm RL fine-tuning for DeFoG: GDPO / RAM / DAM

**Status:** plan, not implemented. Revision 2 — amended after adversarial review (11 agents, all 6 checked findings survived).
**Supersedes:** `ram_design.md`
**Companion docs:** [`gdpo_design.md`](gdpo_design.md), [`RL_FINETUNING.md`](RL_FINETUNING.md), [`reward-finetuning-soc.html`](reward-finetuning-soc.html)

---

## 0. Provenance

Every "measured" number below was produced by running code against this repo, not asserted. Revision 2
folded in nine amendments; the substantive ones were:

- the §4 adjoint had an undefined symbol and prose pointing at a reading measured **360–520× wrong** (A1);
- $u^{\text{base}}$ for the adapter arm was the **wrong distribution** and would have dissolved the
  adapter's conditioning (A5);
- $\lambda{=}1, K{=}12$ leaves E$[\hat a] = 12.06$ against a true value of 1.0, and the clamp guard
  specified in revision 1 provably could not fire (A4);
- $R^{DB}$ is **not** reversible for `rdb='marginal'` (A8);
- Run A as designed could not resolve the effect it was looking for (A2);
- the connectivity pilot was an unfunded gate and has been **dropped** (A3).

---

## 1. Goal and non-goals

**Goal.** Add two RL fine-tuning estimators to `defog.core` alongside `GDPOTrainer` /
`AdapterGDPOTrainer`, sharing their reward / rollout / checkpoint machinery, and run a three-arm
comparison on the ZINC logP adapter.

**Non-goals.** Not replacing GDPO (existing runs stay bit-reproducible, §7). Not changing the sampler,
the deployed checkpoint format, or anything at generation time. Not implementing time-dependent
$\eta(t)$ (§12).

---

## 2. The three arms

| arm | class | loss | states scored at | weight | target |
|---|---|---|---|---|---|
| **GDPO** (today) | `AdapterGDPOTrainer` | advantage-weighted CE | the rollout trajectory | group advantage $A_k$ | none — reward max with optional KL guard |
| **RAM** (ablation) | `AdapterRAMTrainer` | advantage-weighted CE | re-noised from $G_1$ | group advantage $A_k$ | $p^{\text{base}}e^{r}$, asserted not derived |
| **DAM** (headline) | `AdapterDAMTrainer` | generalized-KL on **rates** | re-noised from $G_1$ | importance-weighted discrete adjoint $\hat a$ | $p^{\text{base}}e^{-g}$, derived |

**These are not a clean ablation ladder.** GDPO → RAM is a one-variable change (where $G_t$ comes from).
RAM → DAM changes the loss (CE → gKL), the space (clean marginals → rates), and the weighting
(advantage → discrete adjoint). So:

- **GDPO vs RAM**: do re-noised states beat trajectory states?
- **RAM vs DAM**: is the derived adjoint worth ~2× the forwards?
- **GDPO vs DAM**: the headline, mirroring the paper's D1-vs-DAM.

### What DAM is

$$u^\star_t(y,x) = u^{\text{base}}_t(y,x)\,e^{-V_t(y)+V_t(x)},\qquad
V_t(x) = -\log\sum_z p^{\text{base}}_{1|t}(z|x)\,e^{-g(z)}$$

The discrete adjoint (DAM Thm 2.2) estimates that exponential value difference with terminal condition
$\tilde a_1(y;X) = e^{-g(y)+g(X_1)}$ — the exponential of a reward **difference**, where the continuous
version has $\nabla g$. Non-differentiable rewards are native. The correction is **multiplicative**.

Verified against the paper: Thm 2.2, Prop 2.3 and Prop 2.4 are stated for **general CTMC with arbitrary
base rate**; only Prop 2.5 assumes the masked class, and its proof leans entirely on the exactly-$N$-jumps
unmasking structure. The state-space blowup does not bite — the intractable sum is the *inner*
expectation over terminal states, and Alg. 1 lines 6–9 replace it by sampling one jump target and
debiasing.

### The precondition

Both AM and DAM require $p^{\text{base}}(X_1|X_0)=p^{\text{base}}(X_1)$. Masked models get this free;
DeFoG does not. Measured with `RolloutSampler` at `ROLLOUT_STEPS=250`, 512 samples: excess
$P(X_0{=}X_1)$ over a fresh $p_0$ draw is **+0.010/+0.012 (uniform, $\eta$=0/1), −0.004/−0.001
(marginal)**, all within ~0.006 SE. DeFoG is approximately memoryless at cluster settings.

It fails badly at small step counts (~27 points excess at `sample_steps=5`), which is why §8 forbids the
toy gates from running there. It was measured on the **base** model; the coupling can tighten as the
policy fine-tunes, so §8 requires re-measuring it mid-run.

---

## 3. The derivation: DeFoG's rate is linear in the clean-graph head

### 3.1 Factorisation

DeFoG's conditional path $p_{t|1}(z|x_1) = t\,\delta(z,x_1) + (1-t)p_0(z)$ factorises over coordinates
and a CTMC jump changes exactly one coordinate, so $R_i(x_t \to j \mid x_1)$ depends on $x_1$ **only
through $x_1^i$**.

> Measured: `R_X` bit-identical across all edge classes, `R_E` across all node classes, `max|Δ| = 0.0`.

### 3.2 The identity

$$\boxed{\;\bar R_i(x_t \to j) \;=\; \sum_{c=1}^{d_x} p_\theta\!\left(x_1^i = c \mid x_t\right)\cdot R_i(x_t \to j \mid c,\,t)\;}$$

with $R_i(\cdot \mid c,t)$ a **network-free** basis obtained by evaluating `RateMatrixDesigner` with a
one-hot prediction at class $c$.

**The identity is exact, not approximate.** Measured relative error `0.000e+00` across
`noise_type ∈ {marginal, uniform, absorbing}` × `rdb ∈ {general, marginal, column, entry}` ×
`rdb_crit ∈ {x_1, x_t, dummy}` × $\eta \in \{0, 25\}$ × $t$ from 0 to 1, including the regimes where the
1e-8 denominator floor and the >1e5 zeroing fire.

**Why it is that robust:** the head enters `compute_rate_matrices` only through `sample_from_probs`
(`rate_matrix.py:104-106`), so every downstream nonlinearity — the clamps, the 1e-8 floor, `_stabilize` —
is applied *per class* by construction, and the expectation over that single draw **is** the contraction.

Basis cost on the real base ($d_x = 9$, $d_e = 4$, so 13 calls): **32.9 ms at bs=128, n=38 against
488.7 ms for one network forward — 6.7%.**

### 3.3 What was verified

| claim | result |
|---|---|
| current `compute_rate_matrices` is differentiable in the head | **No** — `RuntimeError` from the multinomial at `rate_matrix.py:104` |
| the exact marginalisation is differentiable | **Yes** — grad L1 811.33 reaches the head |
| the sampler's one-sample rate is unbiased for it | **Yes** — mean $z=-0.040$, median $\lvert z\rvert=0.694$ vs theoretical 0.674 at N=4000 |
| $R^{DB}$ reversible w.r.t. $p_t$ | **Only for `rdb ∈ {general, column, entry}`** — violation exactly 0.0 there, **1.0 × flow scale for `rdb='marginal'`** |
| $R^{TG}$ reversible | **No** — violation / flow scale = 1.0000 exactly |

The exact marginalisation is not merely differentiable, it is **lower variance than the sampler**: it
takes analytically the expectation the sampler estimates with a single draw.

### 3.4 What this forces — four `__init__` guards

1. **$\omega = 0$.** $R^{TG}$ is not marginal-preserving, so with $\omega>0$ the rate you call
   $u^{\text{base}}$ does not generate $p^{\text{base}}$ and rollout endpoints are not draws from
   $p^\theta_1$. (`ROLLOUT_OMEGA=0.0` today at `adapter_rl_finetune__zinc.py:181`.) Note the *identity*
   still holds at $\omega>0$ (measured 0.000e+00 at $\omega{=}0.5$) — the restriction is about marginal
   preservation, not linearity.
2. **`rdb ∈ {general, column, entry}`.** For `rdb='marginal'` the mask
   `1[limit(j) > limit(x_t)]` (`rate_matrix.py:302-303`) is strictly one-directional, so detailed balance
   fails by the **full flow scale** — the identical failure mode that forbids $\omega>0$. `rdb` is a free
   string on `DeFoGModel.__init__` (`model.py:141`) persisted by `save_hyperparameters` into every
   checkpoint, so a foreign checkpoint can select it silently. Same refusal message as $\omega$.
3. **Refuse `rdb == 'column' AND rdb_crit == 'p_x1_g_xt'`** — a **denylist, not an allowlist**. That path
   reads `X_1_pred.argmax(-1)` (`rate_matrix.py:317-318`), the only non-sampled head dependence in the
   file, which the one-hot basis moves to class $c$ (measured max relative error 4.0e-1; Monte-Carlo mean
   $z$ = 86.5, max $|z|$ = 2427, versus mean $z$ = 0.054 for every other criterion). It is unreachable
   today — `rdb_crit` is not exposed on `DeFoGModel.__init__` and has zero assignments in `defog/`,
   `experiments/`, `tests/`, `scripts/` — so this is a guard rail. **Do not write
   `assert rdb_crit in {"x_1","x_t"}`**: that rejects `rdb_crit='dummy'`, which
   `configs/sample/sample_default.yaml:12` sets, which falls to the `else` branch using `X_1_sampled`, and
   which measures 0.0 relative error.
4. **$\eta$ is free and belongs inside $u^{\text{base}}$** — and it is better than free: the $c = x_t$
   basis column is exactly zero at $\eta{=}0$, 2.2e-1 at $\eta{=}1$, 5.5e0 at $\eta{=}25$, so $\eta$
   strictly **enlarges** the reachable rate family. Same $\eta$ in target and policy rate.

**The two-sided bridge collapses.** DAM samples training states from $p^{\text{base}}_{t|0,1}$; for
DeFoG's linear mixture path the conditional references $X_1$ only, so this reduces to $p_{t|1}(\cdot|X_1)$
— i.e. `DeFoGModel._apply_noise`, already exercised at `guidance.py:443,504,636` and `adapter.py:615`.

### 3.5 The honest caveat: a feasibility gap, not an inconsistency

The policy rate is constrained to the **image of the head** under §3.2's linear map. The obstruction is
**not** that the linear system is overdetermined — the off-diagonal targets number $d_x-1$, not $d_x$
(`_compute_step_probs` at `model.py:1121-1135` discards $R$'s diagonal and rebuilds it as $1-\text{rowsum}$,
and DAM Eq. 14 sums over $y \neq x$), and the basis matrix has **full column rank** (measured 8/8 at
$d_x{=}9$ for every $t \in \{0.2,0.5,0.9\}$, $\eta \in \{0,1\}$).

The real obstruction is **nonnegativity**: the reachable set is $\mathrm{conv}\{B[c,\cdot]\}$, a bounded
polytope, and a target is attainable iff its unique basis coordinates are nonnegative and sum to $\le 1$.

So this is DAM as **gKL projection onto DeFoG's deployable rate family**, not DAM attaining its fixed
point. Measured with real heads from `zinc_kek_1.0.0.ckpt` on 64 noised ZINC molecules and an
$\exp(\mathcal{N}(0,0.5))$ directional tilt, the median residual/no-op gKL is **0.281 ($t{=}0.2,\eta{=}0$),
0.246 ($t{=}0.5,\eta{=}1$), 0.566 ($t{=}0.9,\eta{=}1$)** — the gap is real and directional.

Do **not** add a network-free geometry gate: the polytope's magnitude ceiling is
$1/(1 - p_\theta(x_1{=}x_t|x_t))$, a *network* quantity, and on the real base the median head mass on the
current class is 0.776/0.849/0.999 (nodes) and 0.922/0.955/1.000 (edges) at $t = 0.2/0.5/0.9$, giving
4.5×–10⁴× headroom. The family is not the binding constraint.

---

## 4. The DAM objective for DeFoG

Write $y_{ij}$ for state $x = G_t$ with coordinate $i$ changed to class $j$.

**Rates** (both via §3.2; $R = R^\star + \eta R^{DB}$, $\omega = 0$):

$$u^{\text{base}}(y_{ij},x) = \textstyle\sum_c p^{\text{base}}(x_1^i{=}c|x)\,R_i(x{\to}j|c,t),\qquad
  u^{\theta}(y_{ij},x) = \textstyle\sum_c p_\theta(x_1^i{=}c|x)\,R_i(x{\to}j|c,t)$$

> **$p^{\text{base}}$ for the adapter arm is the PRE-RL COMPOSED POLICY** — frozen base **+ frozen pre-RL
> adapter — not the unconditional base.** `_base_uncond_softmax` (`rl.py:590`) returns group 0 of the PoE
> blend; using it would retarget the optimum to $p_{\text{uncond}}e^{-g}/Z$, so the structural anchor
> would actively pull the adapter's learned conditioning *out* — exactly what `KL_COEF=0.1` exists to
> prevent. `AdapterDAMTrainer.__init__` must therefore build the frozen ref-adapter **unconditionally**,
> not only when `kl_coef > 0` as `rl.py:714-716` does today.

**Adjoint** (DAM Prop 2.4 / Eq. 13), with DeFoG's clean-graph head standing in for $p_{1|t}$:

$$\hat a_t(y,x) \;=\; \underbrace{\frac{p^{\text{base}}_{1|t}(Z|y)}{p^{\theta}_{1|t}(Z|y)}\,e^{-g(Z)}}_{Z\sim p^\theta_{1|t}(\cdot|y),\ \text{single sample}}
\;\Bigg/\;\underbrace{\frac{1}{K}\sum_{k=1}^{K} w_k}_{X_1^{(k)}\sim p^\theta_{1|t}(\cdot|x)},
\qquad
w_k := \frac{p^{\text{base}}_{1|t}(X_1^{(k)}|x)}{p^{\theta}_{1|t}(X_1^{(k)}|x)}\,e^{-g(X_1^{(k)})}$$

**Read the weights carefully.** $w_k$ is the closed-form per-coordinate categorical ratio **times
$e^{-g}$** — the $e^{-g}$ is not optional. Dropping it (using the bare ratio) was measured **363×** too
large on a 40-state tabular problem with this repo's reward tiering, and **517×** with
$p^\theta = p^{\text{base}}$; normalising the weights to sum to 1 instead of mean 1 is off by exactly
$1/K$. Against a true $e^{V_t(x)} = 3.5722$: Eq-13 form 4.239, SNIS-normalised-to-mean-1 3.811 (equivalent,
marginally lower bias), sum-to-1 0.3176, bare ratio 1299.

**Both factors are computed in log-space via `logsumexp` — mandatory, not "where possible."** The ratio
is a product over ~740 coordinates at $n{=}38$.

**Loss** (DAM Eq. 14, with the $y$-sampling debias of Eq. 11):

$$\mathcal{L}_{\text{DAM}} = \frac{1}{p^\theta_t(y|x)}\,D_{\text{gKL}}\!\Big(u^\theta(y,x),\;u^{\text{base}}(y,x)\,\hat a_t(y,x)\Big),
\qquad D_{\text{gKL}}(u,w) = u - w + w\log\frac{w}{u}$$

Verified against the paper: the gKL expression matches Eq. 14 verbatim including argument order, its
minimiser in $u$ is $u = w$, and the $y$-debias divides by $p^u_t(y|x)$ as the prose above Eq. 11
instructs.

### 4.1 Temperature, and the health metric that actually works

DAM has **no `kl_coef`**. Problem (1) fixes the KL weight at 1 and folds the temperature into the terminal
loss: $g = -\lambda\cdot\text{reward}$, so $\lambda$ **is** the inverse temperature with no extra factor.
The anchor is structural — $u^{\text{base}}$ multiplies the target in Thm 2.2 / Eq. 10 / Eq. 11 — so it
cannot be switched off. Revision 1's "the anchor must be mandatory" amendment is satisfied by
construction here, and binds only on the RAM arm.

**The reward span is $[-10, 0]$**, not $[-10,3]$: `rl.py:834-838` gives `invalid=-10`, `disconnect=-4`, and
`rl.py:859` gives $-\min(|\Delta p|/\text{scale}, 3.0) \in [-3,0]$.

**Consequence for the clamp.** Keep $\pm 10$ as a *saturating envelope* — the true adjoint genuinely
reaches $e^{10}$ at low $t$, so a tighter clamp corrupts the target — but its fraction is **0 by
construction** whenever $\lambda\cdot\text{span} \le \text{clamp}$ (measured clamp fraction 0.0000 at
$\lambda{=}1$). **It is not a health metric.**

**The real bias, and the real metric.** gKL's minimiser matches $\mathbb{E}[\hat a]$, so a heavy-tailed
adjoint biases the fitted *rate* by its mean. The tail is driven by $K$-sets where every adjoint sample
sits at the invalid floor — and §4.3 makes that likely, because $Z$ and $X_1^{(k)}$ are one-shot clean
predictions, measured at **94–97% invalid at low $t$** on the real base. On a $y = x$ control where the
true adjoint is exactly 1.0:

| | 10% invalid | 94% invalid | 94%, K=64 |
|---|---|---|---|
| $\lambda = 0.3$ | 1.019 | **1.31** | — |
| $\lambda = 1$ | 1.063 | **410.7** | 18.3 |
| $\lambda = 2$ | 1.136 | 7.1e6 | 2.8e5 |

End-to-end on the real base with the plan's re-noising density: $\mathbb{E}[\hat a] = $ **12.06** at
$(\lambda{=}1, K{=}12)$, 2.06 at $K{=}64$, 1.06 at $\lambda{=}0.3$.

**So: the $y = x$ control is the health metric.** Reuse the $K$ samples already drawn, true value exactly
1.0, gate on $|\log \mathbb{E}[\hat a]|$. It is the only signal that separates "$\lambda$ is too hot" from
"the reward is doing work". Defaults: **`DAM_LAMBDA = 0.3`** for this tiered reward, and sweep
**`DAM_K ∈ {12, 64}`**. The two knobs are not interchangeable — at high invalid rates $\lambda$ is the
stronger lever (1.31 at $\lambda{=}0.3,K{=}12$ versus 18.3 at $\lambda{=}1,K{=}64$).

### 4.2 What DAM does not use

`group_advantage`, `advantage_mode`, `advantage_clip`, `positive_only` — the whole advantage layer is a
GDPO/RAM concept. `condition_sampler`, CRN, `size_dist` and the reward classes are shared unchanged.

### 4.3 Rewards on predicted-clean graphs — not benign

$Z$ and $X_1^{(k)}$ are one-shot clean predictions, so a larger fraction is invalid: **94–97% at low $t$**
on the real base. Invalid graphs do get $e^{-g}\approx 0$ and near-zero weight, but **that is not
self-correcting**: above roughly 50% it is the leading source of target-rate bias, via the mechanism in
§4.1. Log the invalid fraction among adjoint samples separately from rollout validity, and treat a high
reading as the trigger to **lower $\lambda$ or raise $K$**, not as a curiosity.

---

## 5. Cost

Per scored state, against GDPO's 1 forward + 1 backward:

| | forwards | backward | reward evals |
|---|---|---|---|
| GDPO | 1 (grad) | 1 | — (1 per *trajectory*) |
| RAM | 1 (grad) | 1 | — (1 per *trajectory*) |
| DAM, full model | 1 grad @ $x$ + 1 base @ $x$ + 1 policy @ $y$ + 1 base @ $y$ | 1 | $K{+}1$ |
| DAM, adapter | as above; only the shared **unconditional** forward is saved | 1 | $K{+}1$ |

Anchors: one scored state's forward+backward is **1299 ms** at $K{=}128, n{=}38$; 16 `_apply_noise` draws
are **199.6 ms**; the §3.2 basis is **32.9 ms** against **488.7 ms** for one forward; reward evaluation is
**0.198 ms/mol**, so $16\times 13 = 208$ evals per iteration is **~40 ms** — negligible.

**DAM is ~2× GDPO per scored state on the full-model path and 2.28× on the adapter path** against GDPO
with `kl_coef=0`. The revision-1 figure of 1.7× assumed the unconditional forward was $p^{\text{base}}$;
§4 corrects that, and the pre-RL composed policy needs its own forwards at both $x$ and $y$.

---

## 6. Module layout

```
defog/core/rl.py                       (refactored in D1; GDPO behaviour frozen)
├── ... existing ...
├── RLTrainerBase                      ← NEW (extracted; verified extractable, §7)
├── GDPOTrainer(RLTrainerBase)         ← signature + behaviour unchanged
└── AdapterGDPOTrainer(GDPOTrainer)    ← unchanged EXCEPT a record_trace flag (below)

defog/core/renoise.py   → renoise_states(...)
defog/core/ram.py       → RAMTrainer(RLTrainerBase), AdapterRAMTrainer(AdapterGDPOTrainer)
defog/core/dam.py       → rate_basis, marginal_rate, discrete_adjoint, gkl,
                          DAMTrainer(RLTrainerBase), AdapterDAMTrainer(AdapterGDPOTrainer)
```

Both adapter trainers subclass `AdapterGDPOTrainer` and override **`__init__` and `update()`**.
`__init__` must be overridden regardless: the four §3.4 guards live there, `DAM_K` / `DAM_LAMBDA` / the
clamp range must be stored, and the frozen ref-adapter (§4) must be built.

Rationale for subclassing rather than duplicating: `AdapterGDPOTrainer.rollout` is the one method in
`rl.py` carrying a comment recording a bug that already cost a run — omitting `condition=cond` on
`sampler.sample()` does not raise, it silently draws every rollout graph from the wrong size distribution
(rollouts averaged 22.5 heavy atoms against 18.1 at evaluation). The rollout is identical across all three
arms; only `update()` differs.

**Suppressing trace recording needs a real flag.** Revision 1 proposed `subsample_idx=[]`; that is
unreachable from an update-only subclass, because the inherited `rollout` builds the sampler from
`self._choose_subsample()` (`rl.py:435`, used at `rl.py:731-733`). The only route is
`subsample_steps=0`, which returns `[]` for `'stratified'` and `'uniform'` but **raises `RuntimeError`
for `'late'`**. So: add an explicit `record_trace: bool = True` to the rollout path (or override
`_choose_subsample`). The empty-set short-circuit at `rl.py:266-267` itself is verified —
`set([])` is not `None`, so recording stops while the endpoint stash, `end_node_mask` and the CRN
`_init_state` all still fire.

### 6.1 `dam.rate_basis` / `dam.marginal_rate` / `dam.discrete_adjoint`

`rate_basis` builds $R_i(\cdot|c,t)$ for $c = 1..d_x$ (and $d_e$) by calling the **existing**
`RateMatrixDesigner.compute_rate_matrices` under `no_grad` with a one-hot prediction. `marginal_rate`
contracts it against the head with `einsum`. `RateMatrixDesigner` is **not modified**; the sampler path is
untouched.

Stabilise **per basis class before contracting**. Note this is currently untested and unobservable in the
configurations §8 runs in: under any strictly positive noise support the `pt == 0` clauses of `_stabilize`
never fire and the `> 1e5` clause does not fire below $t \approx 0.9999$, so the two orders are
bit-identical. §8 adds the two regimes where it is observable.

**`discrete_adjoint` and `gkl` must take plain probability tensors** — not a trainer, not a `DeFoGModel`.
Otherwise §8's tabular gate silently degrades into testing a re-implementation rather than shipped code.

---

## 7. Refactor safety (D1/D2)

`RLTrainerBase` takes the optimiser / EMA / device / seed construction, `_frozen_reference()`, `step()`,
`fit()`, `save()`, and the adaptive-KL controller. `rollout()` and `update(buf)` stay abstract.
**`GDPOTrainer.__init__` keeps its exact current signature.**

> Verified by performing the extraction on a copy: `inspect.signature(GDPOTrainer.__init__)`
> byte-identical, `AdapterGDPOTrainer`'s `super().__init__(...)` still works, `fit(3)` bit-identical for
> both trainers with `kl_coef`, `kl_target`, `ema_decay`, `positive_only` and `crn` exercised.

**No hash is committed.** Two measured obstacles: `small_model` is function-scoped and constructs the
model *before* the test body runs (three runs → three different hashes); and a `state_dict` sha256 is
machine-locked (four hashes at 1/4/8/12 CPU threads; `ATEN_CPU_CAPABILITY` changes 114/132 tensors;
`torch.use_deterministic_algorithms(True)` is a measured no-op here).

`tests/test_rl_parity.py` therefore (1) seeds **then** constructs the model inside the test body from
`small_model_config`; (2) freezes a verbatim copy of today's `GDPOTrainer` as `tests/_gdpo_frozen.py` —
which must import nothing from `rl.py` that the refactor moves, i.e. it carries its own copies of
`eager_logprob`/`kl_clean` or imports them by value at module load; (3) asserts frozen vs refactored **in
the same process**, regenerating both sides at test time.

---

## 8. Tests

Baseline: `pytest tests/test_rl.py` collects **22** tests, all passing, 14–58 s.

### Re-noising

`test_renoise_marginal_matches_kernel` (reference: 20k draws, max deviation 0.0090 vs a 3σ band of
0.0106) · `test_renoise_symmetric_and_masked` (**the diagonal is one-hot(class 0), not zero** —
`sample_from_probs` does `triu(E,1)` then mirrors) · `test_renoise_t1_is_identity` (**on
`_edge_upper_mask(node_mask)` only**, for the same reason) · `test_renoise_t0_is_prior`.

### Rate identity

`test_marginal_rate_matches_sampler_in_expectation` — **parametrised over `noise_type='absorbing'`**, and
a companion `test_stabilise_order_matters` at $t \ge 0.99999$ under marginal noise. Those are the only
regimes where §6.1's stabilise-order instruction is observable; `tests/conftest.py:235` builds
`small_model` with `noise_type="uniform"`, where it is not.
Plus `test_marginal_rate_is_differentiable` and `test_rate_basis_factorises` (expect exact equality).

### Guards

`test_omega_nonzero_refused` · `test_rdb_marginal_refused` · `test_column_p_x1_g_xt_refused` ·
`test_rdb_crit_dummy_accepted` (the value `sample_default.yaml:12` sets).

### The estimator gate — step 5

`test_dam_reaches_kl_optimum`. An independent replica converged to $p^{\text{ref}}e^{-g}/Z$ (KL 0.0006
against 0.576 for the un-tilted base) and goes red under both wrong adjoint readings — a ~500× or a $1/K$
mis-scale cannot reach that fixed point. **It is ~60–80 lines, not 40**: with no head it must supply
$p^\theta_{1|t}$ itself, i.e. a small time-dependent CTMC propagator plus a terminal-law integrator.

**Write the fixture at the production reward span.** The replica loses three orders of KL accuracy across
it — KL 0.0006 / 0.047 / 2.89 at span 3 / 9 / 30 — so a fixture written at a comfortable span passes while
the production configuration is biased.

Also add `test_adjoint_reproduces_value_function`: the estimator recovers $e^{V_t(x)}$ on a small known
$p^{\text{base}}$ and $g$. Step 5 does catch a mistranscribed adjoint, but §11.1 primes the reader to
blame the projection gap when it goes red.

### Toy training

`test_dam_increases_toy_reward_without_collapse` — reward rises **and** a node-class-entropy /
unique-endpoint floor holds, at `sample_steps>=50` via a `small_model_dam` fixture, because
memorylessness is violated at the fixture's default 5. Add the same floor to the **existing**
`test_gdpo_increases_toy_reward`, which currently passes most emphatically when the policy has collapsed
(measured: node entropy −72%).

### RAM arm

Mandatory `kl_coef>0` (`ValueError`), `advantage_mode='grpo'` refused (per-group std whitening deletes
the temperature), a tabular fixed-point test at $c{=}1$ with a collapse counterpart at $c{=}0$, the same
diversity floor. Revision 1's `test_ram_matches_gdpo_when_states_are_forced` is **downgraded** to
`test_ram_pg_term_normalisation`; with the anchor added it is no longer a tautology and is not a gate.

---

## 9. Experiment plan — ZINC logP adapter

**The connectivity pilot is dropped.** It needed an `ESTIMATOR` switch in `gdpo_connectivity.py` (which
constructs `GDPOTrainer` directly at :271 with `kl_coef`/`kl_target`, which DAM does not accept), a
freshly written full-model DAM rollout, a $\lambda$ sweep before any null was attributable, and ≥4 seeds
before its gate meant anything — and it runs at `ETA=0.0` (:68), the setting where the reachable rate
family is smallest and least representative of the ZINC arm. That budget goes to Run A instead.

### D8 — the switch, in `experiments/adapter_rl_finetune__zinc.py`

```python
ESTIMATOR: str   = "gdpo"    # "gdpo" | "ram" | "dam"
RL_ITERS: int    = 120       # pinned; MUST be a multiple of PROBE_EVERY
RENOISE_DRAWS: int = 16      # literal, NOT a mirror of SUBSAMPLE_STEPS
T_SAMPLER: str   = "match"   # "match" | "train" | "ram" | "uniform"
INNER_STEPS: int = 1
DAM_K: int       = 12        # swept {12, 64}
DAM_LAMBDA: float = 0.3      # see §4.1
BASE_TRAIN_DISTORTION: str = None   # default: checkpoint hparam; declare if absent
```

**`RENOISE_DRAWS` must be a literal 16.** Mirroring `SUBSAMPLE_STEPS` combined with `subsample_steps=0`
(the trace-suppression route) yields zero re-noised states — an update that does nothing, in exactly the
configuration the new arms need.

**Pinning the iteration count requires touching the deadline too.** The loop at :605 is
`while it < e.MAX_ITERS and time.time() < deadline` — an **AND**. Raise `MAX_TIME_HOURS` per arm so the
deadline never binds, and **assert `it == RL_ITERS` at loop exit — raise, do not warn** — so a truncated
arm fails loudly instead of being reported. `RL_ITERS` must be a multiple of `PROBE_EVERY` (:207, =40),
or the last partial block is silently dropped from early-stop selection. At 120 pinned iterations DAM at
$M{=}64$ is roughly **8.7 h/run**; state the GPU-hour total before submitting.

**`BASE_TRAIN_DISTORTION` is a declaration, not an assert.** `DeFoGModel.load` defaults
`train_distortion` to `'identity'` for checkpoints whose hparams omit it, and the shipped `zinc_kek` base
reports exactly that while every arm runs `polydec` — so an assert would fire on all three Run A arms and
could not be satisfied, since the true distortion is not recoverable from that checkpoint. Default to the
hparam when present, require a declaration otherwise, **warn** on mismatch. Optionally re-save the shipped
checkpoint with the field recorded.

### Readout — the 100-target paired protocol

**Not the 2-target probe.** `TARGET_PERCENTILES = [5, 95]` (:219) and `N_PER_TARGET = 128` (:221) give one
target and 128 samples per band — a target-specific number with no control over target-to-target
idiosyncrasy. Reconstructed SE ≈ 0.03/band from sampling alone; at 4 seeds the two-arm MDE is ≈ **0.06**,
which is the entire effect cited as the prior.

Use the existing E2 protocol instead: `adapter_improvements/*/e2_*.json` are **100-target paired runs**,
measured across-target SD 0.183/0.187/0.179/0.155 → **SE 0.0183/0.0187/0.0179/0.0155**, with target lists
verified identical across configs, so a paired per-target difference is available (measured paired SE
0.0205). Analyse as a paired per-target difference between arms, $n = 100$ paired units within a run, seed
as a block and arm as a fixed effect. Keep the 2-target probe for early stopping only.

**Pre-register before submitting:** the HIGH band carries the confirmatory claim (it is the less noisy of
the two, and the −0.06 prior is a HIGH-band effect); paired two-sided $t$ on the per-target differences;
$\alpha = 0.05$.

### Run A

Held identical: seed, `LR`, `EMA_DECAY`, `ROLLOUT_SIZE`, `SAMPLE_STEPS`, `ETA`, `RL_ITERS`,
`SUBSAMPLE_STEPS`, condition sampler, reward, CRN grouping, eval protocol. **`OMEGA` is not in this
list** — it is a GDPO-only degree of freedom and must be 0 for DAM.

Varied: `ESTIMATOR ∈ {gdpo, ram, dam}` at `INNER_STEPS=1`, `T_SAMPLER="match"`, `RENOISE_DRAWS=16`,
`DAM_K=12`. **3 arms × 4 seeds = 12 runs.**

Then **Run A′, conditional on Run A not being a regression**: sweep the second axis on GDPO and the better
re-noising arm only — `RENOISE_DRAWS ∈ {16,64}` for RAM, `DAM_K ∈ {12,64}` for DAM. 8 runs.
Note these are *different* axes: RAM's $M$ is the number of re-noised states, DAM's $K$ is the number of
adjoint samples per state. Revision 1 conflated them.

Re-noising only decorrelates draws *within* a trajectory; the between-trajectory variance is identical
across arms because they consume the same rollouts. So **a tie at the low setting is uninformative;
evidence is a gain that grows with the axis.**

Secondary readouts: validity, uniqueness, scaffold diversity, node-class entropy, adapter weight drift,
$\mathbb{E}[\hat a]$ at $y{=}x$, adjoint-sample invalid fraction, residual/no-op gKL per coordinate type
and $t$-band.

### Run B — throughput, only if Run A is not a regression

`INNER_STEPS ∈ {4, 8}`, `T_SAMPLER="train"`, matched wall clock.

`"match"` and `"train"` are the **same t-density** on a polydec-pretrained base (exact sup-CDF distance
$1/S = 0.004$). What differs is the **coupling**: `"match"` shares one stratified grid value across all
$K$ trajectories while `"train"` draws $K$ independent values, as DAM/RAM Alg. 1 does — measured
gradient-covariance ratio 0.58, bootstrap 95% CI [0.487, 0.694]. Add `t_sampler="ram"` drawing
$t = 1-\sqrt{U}$ as one extra arm; without it a null cannot be reported as "the weighting doesn't help in
discrete".

---

## 10. Sequencing

| step | work | gate |
|---|---|---|
| 1 | `tests/_gdpo_frozen.py` + `tests/test_rl_parity.py` | passes on unmodified `rl.py` |
| 2 | D1 refactor `RLTrainerBase` + `record_trace` flag | parity + all 22 of `test_rl.py` green |
| 3 | `renoise.py` + its 4 tests | green |
| 4 | `dam.rate_basis` / `marginal_rate` + rate tests + the four guards | green |
| 5 | `dam.discrete_adjoint` + `gkl` + **`test_dam_reaches_kl_optimum`** + `test_adjoint_reproduces_value_function` | green — **estimator algebra go/no-go** |
| 6 | `DAMTrainer`, `AdapterDAMTrainer` | toy + diversity-floor tests green |
| 7 | `ram.py` | its tests green |
| 8 | exports, `ESTIMATOR` switch, local smoke of all three arms | one iteration of each end to end |
| **8.5** | **calibration on the real base at the run's own $\lambda$, $K$** | **model-level go/no-go — see below** |
| 9 | ZINC Run A (12 runs) | pre-registered paired test |
| 10 | Run A′ (8 runs), then Run B | conditional on 9 |

### Step 8.5 — the gate step 5 cannot be

Step 5 pins the estimator's *algebra* and nothing else. It is blind by construction to the three things
most likely to kill DAM on the real model: the projection gap (§3.5 — unconstrained tabular policy), the
head-surrogate substitution (§11.3 — no head in the fixture), and $\lambda/K$ mis-tempering (§4.1 — the
fixture's reward span is a free choice). Before any cluster time, on the real ZINC base:

**(a) The $y = x$ control.** The true adjoint there is exactly 1.0. Reuse the $K$ samples already drawn —
free. Gate on $|\log \mathbb{E}[\hat a]|$ small. This is the only signal separating "$\lambda$ is too hot"
from "the reward is doing work", and it is what selects $\lambda$ and $K$ in minutes.

**(b) The residual/no-op gKL ratio** from §3.5, logged per coordinate type (node vs edge) and per
$t$-band. A ratio that plateaus high is the projection gap biting.

---

## 11. Risks

1. **The projection gap may bite.** §3.5 measures the residual/no-op gKL at 0.24–0.57 with real heads, so
   the gap is real and directional. If it eats the benefit, DAM degenerates toward RAM. Only detectable on
   the real model — hence step 8.5(b).
2. **$\lambda$ and $K$ are under-determined.** §4.1 gives defaults from a $y{=}x$ control, not from a run.
   Step 8.5(a) resolves it cheaply; the FK `beta` sweeps are the closest existing prior, since FK targets
   the same tilt.
3. **The factorised head is a good marginal but a poor joint.** On the real base, $P(\text{connected})$ of
   the true rollout continuation is **0.975 versus 0.247** for graphs drawn from the factorised head at
   the same $X_t$ ($t{=}0.64$), while bond counts agree to 0.9%. The *ordering* survives (Spearman +0.710,
   $p = 1.0\times10^{-4}$) and the level bias largely cancels in the adjoint's ratio, so this is probably
   absorbed by the $\lambda$ sweep — but the effective tilt on the connectivity/validity tier is ~3× the
   nominal $\lambda$. Log $\mathbb{E}_{\text{head}}[e^{-g}]$ vs $\mathbb{E}_{\text{rollout}}[e^{-g}]$ and
   their rank correlation on a handful of states; **alert on the rank correlation, not the ratio.**
4. **`INNER_STEPS>1` is off-policy in the endpoint law** and there is no cheap correction: the needed
   ratio is $p^{\theta_{\text{new}}}_0(G_1)/p^{\theta_{\text{old}}}_0(G_1)$, the intractable marginal
   likelihood. A PPO-style ratio on $\log p_\theta(G_1|\tilde G)$ corrects nothing, because the noising
   kernel is analytic and $\theta$-independent. Controls: a small cap plus a drift early-stop. Compute
   drift by reusing the `(noisy, extra, puX, puE)` already built in the update loop — one extra
   `_compose_logmarginals` against a snapshot, **not** `adapter_kl_clean`, which redoes
   `_base_uncond_softmax` and both composition calls. Off for timed Run B arms.
5. **Memorylessness was measured on the base model**, but the buffer holds endpoints from the fine-tuned
   policy, where the $(X_0, X_1)$ coupling can tighten. Re-measure the excess $P(X_0{=}X_1)$ at least once
   mid-run; it is the precondition the whole re-noising step rests on.
6. **Novelty is narrow.** DAM on graph CTMCs with marginal (not masked) noise, in the $x_1$-parameterisation,
   as a gKL projection onto a restricted rate family, plus the measured memorylessness result. Not a novel
   discrete port.

## 12. Out of scope, recorded here

- **Time-dependent $\eta(t)$.** §3.4 establishes $R^{DB}$ is exactly reversible for any $\eta$ under
  `rdb ∈ {general, column, entry}`, including a time-varying one, so `eta_t = eta * (1-t)/t` is legitimate
  and a one-line change. The continuous memoryless schedule is $\sigma^2(t) = 2(1-t)/t$, suggesting the
  same shape. §2's measurement says DeFoG is already approximately memoryless at 250 steps, so the
  expected gain is small.
- **A discrete two-sided bridge.** §3.4 shows it is degenerate for DeFoG's mixture path; only interesting
  if the memorylessness measurement fails on some other base model.
- **The connectivity experiment.** Dropped from this plan (§9), not from the roadmap — it remains the
  reward shape where DAM's published gains were largest (Sudoku 11.5 → 23.8 (D1) → 89.2 (DAM)), and worth
  revisiting once a full-model DAM trainer exists for its own sake.
