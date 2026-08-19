> **SUPERSEDED by [`dam_design.md`](dam_design.md).** Kept for the record. Two things
> invalidated this plan: Discrete Adjoint Matching (arXiv:2602.07132, ICLR 2026) already
> extends Adjoint Matching to CTMCs, and a derivation pass showed DAM is implementable
> here. RAM survives in the new plan as an ablation arm. Sections 2 and 11 below are
> known-wrong on which half of RAM's derivation ports to discrete state.

# RAM: Reinforce-Adjoint-Matching RL fine-tuning for DeFoG

**Status:** plan, not implemented
**Author:** design doc, 2026-08-19
**Companion docs:** [`gdpo_design.md`](gdpo_design.md), [`RL_FINETUNING.md`](RL_FINETUNING.md), [`reward-finetuning-soc.html`](reward-finetuning-soc.html)

---

## 1. Goal and non-goals

**Goal.** Add a second, independent RL fine-tuning estimator to `defog.core` —
Reinforce Adjoint Matching ([arXiv:2605.10759](https://arxiv.org/abs/2605.10759)) —
as new classes that sit *alongside* `GDPOTrainer` / `AdapterGDPOTrainer`, share their
reward / advantage / rollout / checkpoint machinery, and can be A/B'd against them
under a matched budget.

**Non-goals.**

- Not replacing GDPO. Every existing run, script and checkpoint must remain
  bit-reproducible. This is enforced by a golden test (§7), not by good intentions.
- Not changing the sampler, the rate matrix, `eta`, or anything at generation time.
  The output of both trainers is a plain DeFoG checkpoint / adapter.
- Not implementing the time-dependent `eta(t)` "memoryless schedule" idea. That is a
  separate, independent investigation; see §11.

---

## 2. What RAM is, in one table

RAM's claim is that under KL-regularised reward maximisation, the optimal fine-tuned
process **tilts the distribution over clean endpoints toward high reward and leaves
the noising law unchanged**. So fine-tuning can reuse the pretraining loop verbatim:
only the data source and a scalar weight change.

| | pretraining (`_apply_noise` + CE loss) | GDPO (today) | RAM (proposed) |
|---|---|---|---|
| where `G1` comes from | dataset | model rollout | model rollout |
| where `G_t` comes from | `_apply_noise(G1, t)` | **the rollout trajectory** | **`_apply_noise(G1, t)`** |
| `t` | `time_distorter.train_ft` | the sampler's distorted grid | free (see §6.3) |
| loss | `-log p(G1 | G_t)` | `-A · log p(G1 | G_t)` | `-A · log p(G1 | G_t)` |

The single substantive difference between the two RL columns is the second row.

### Why this is low-risk in this codebase

`DeFoGModel._apply_noise(X, E, y, node_mask, t=None)` already takes an explicit `t`,
and is already used in exactly this "re-noise a known clean graph at a chosen `t`"
mode by `guidance.py:443,504,636` and `adapter.py:615`. Edge symmetry and
node/edge masking are handled inside `sample_from_probs`. There is no new math to
write and no new correctness surface in the noising path.

Likewise, the entire *scoring* path is already parameterised on `X_t`:

```python
eager_logprob(model, X_t, E_t, y_t, t, X1, E1, node_mask, ...)
adapter_eager_logprob(base, adapter, X_t, E_t, cond, t, X1, E1, node_mask, ...)
kl_clean(policy, ref, X_t, E_t, y_t, t, node_mask, ...)
adapter_kl_clean(base, adapter, ref_adapter, X_t, E_t, cond, t, node_mask, ...)
```

None of these care where `X_t` came from. **All four are reused unchanged.**

### Honest framing

This is a **different estimator with a different derivation**, not a variance-reduction
tweak to GDPO. GDPO's eager policy gradient is justified by an MDP argument in which
the scored states must be states the trajectory actually visited; RAM is justified by
the adjoint-matching optimality condition plus a REINFORCE identity, which is what
licenses the pretraining-kernel states. Swapping the states inside GDPO's derivation
is *not* automatically valid. Hence: a new class, and an A/B — never a silent swap.

Additional caveat: RAM's derivation is for continuous-state flow/diffusion. DeFoG's
conditional path is built the same way (a mixture interpolating noise and data), so it
should carry over, but neither the paper nor this document proves it for the discrete
case. §9's toy test is the empirical guard.

---

## 3. Expected benefits

1. **Endpoint reuse.** A rollout costs `sample_steps` network passes plus an RDKit
   decode. Today that buys `subsample_steps` (default 16) scored states, once. With
   re-noising, one `(G1, A)` pair can be re-noised at fresh `t` indefinitely — the
   replay-buffer structure from Adjoint Sampling. Exposed as `inner_steps`.
2. **Lower gradient variance.** The 16 trajectory states are all from one trajectory,
   so they are correlated across `t`. Re-noised draws are conditionally independent
   given `G1`.
3. **Full-resolution `t` coverage.** `_choose_subsample()` picks indices off a fixed
   `sample_steps` grid, shared across all K trajectories in the iteration. RAM can
   draw a different continuous `t` per trajectory per draw.
4. **Less machinery.** No `trace_X` / `trace_E` / `trace_t`, no `_pre_step` hook, lower
   rollout memory.

Only (2) is measurable at matched gradient steps; (1) and (3) require `inner_steps>1`
or a different `t` policy, which is why the experiment is staged (§8).

---

## 4. Deliverables

| # | Deliverable | File |
|---|---|---|
| D1 | Non-behavioural refactor: extract `RLTrainerBase` from `GDPOTrainer` | `defog/core/rl.py` |
| D2 | Golden parity test locking GDPO behaviour across the refactor | `tests/test_rl_parity.py` |
| D3 | `renoise_states()` helper | `defog/core/ram.py` |
| D4 | `RAMBuffer`, `RAMTrainer` | `defog/core/ram.py` |
| D5 | `AdapterRAMTrainer` | `defog/core/ram.py` |
| D6 | Unit tests | `tests/test_ram.py` |
| D7 | Package exports | `defog/core/__init__.py` |
| D8 | `ESTIMATOR` switch in the ZINC logP adapter experiment | `experiments/adapter_rl_finetune__zinc.py` |
| D9 | Cluster script for the matched-budget A/B | `run_zinc_ram_ab_<cluster>.sh` |

---

## 5. Module layout and class hierarchy

```
defog/core/rl.py                       (existing; refactored in D1, behaviour frozen)
├── Reward, reward_from_energy
├── eager_logprob, kl_clean, group_advantage, EMA          ← reused by RAM
├── RolloutSampler                                          ← reused by RAM
├── RolloutBuffer
├── RLTrainerBase                      ← NEW (extracted)
├── GDPOTrainer(RLTrainerBase)         ← signature + behaviour unchanged
├── AdapterGDPOTrainer(GDPOTrainer)    ← unchanged
├── _base_uncond_softmax, _compose_logmarginals, _score_logprob,
│   _kl_from_logmarginals, adapter_eager_logprob, adapter_kl_clean   ← reused by RAM
└── PropertyMatchReward, HeadPropertyMatchReward, make_condition_sampler

defog/core/ram.py                      (NEW)
├── renoise_states(model, X1, E1, y, node_mask, n_draws, t_sampler, ...)
├── RAMBuffer
├── RAMTrainer(RLTrainerBase)
└── AdapterRAMTrainer(RAMTrainer)
```

### D1 — what `RLTrainerBase` owns

Extracted verbatim from `GDPOTrainer`, no logic changes:

- optimiser / EMA / device / seed / `_iter` construction
- `_frozen_reference()`
- `step()`, `fit()`, `save()`
- the adaptive-KL controller inside `step()`
- the metrics dict assembled in `step()`

Left abstract: `rollout()` and `update(buf)`.

**Constraint:** `GDPOTrainer.__init__` keeps its exact current signature and keyword
defaults, because `AdapterGDPOTrainer.__init__` calls
`super().__init__(base, reward_fn=None, kl_coef=0.0, ema_decay=None, lr=..., **gdpo_kw)`
and dozens of experiment modules pass keywords positionally-by-name. The refactor is
*internal* — `GDPOTrainer.__init__` may delegate to `RLTrainerBase.__init__`, but its
own signature does not move.

---

## 6. The RAM estimator, precisely

### 6.1 GDPO's update (for reference)

With `K` trajectories and `S = len(buf.states)` scored states:

```
L_GDPO = -(1 / (K·S)) · Σ_{s=1..S} Σ_{k=1..K}  A_k · log p_θ(G1_k | G_{t_s,k})
```

where `G_{t_s,k}` is the state trajectory `k` actually occupied at grid index `t_s`.

### 6.2 RAM's update

With `M = renoise_draws` fresh draws per endpoint:

```
for m in 1..M:
    t_m           ~ t_sampler                              # (K,1)
    G̃_{m,k}       ~ p_{t_m|1}(· | G1_k)   via _apply_noise  # (K,n,dx),(K,n,n,de)

L_RAM  = -(1 / (K·M)) · Σ_{m=1..M} Σ_{k=1..K}  A_k · log p_θ(G1_k | G̃_{m,k})
```

plus, when `kl_coef > 0`, the same `kl_clean` term evaluated at the same `G̃_{m,k}`,
scaled identically to GDPO's (`kl_coef / M · kl · nb/K`) so `kl_coef` transfers
between the two trainers without retuning.

Everything upstream of the loss — reward, `group_advantage`, `positive_only`,
`advantage_mode`, CRN grouping, `condition_sampler` — is **unchanged and shared**.

### 6.3 Where `t` comes from — `t_sampler`

This is the one genuinely free choice, and it decides what the A/B measures.

| value | draw | use |
|---|---|---|
| `"match"` **(default)** | the same distorted grid values `_choose_subsample()` would have produced, `M = subsample_steps` | **Run A.** Marginal over `t` is identical to GDPO's, so the *only* difference is trajectory-state vs re-noised-state. Maximum isolation. |
| `"train"` | `model.time_distorter.train_ft(K, device)` | Theory-faithful: "noise it as in pretraining". Use for Run B. |
| `"uniform"` | `U(0,1)` per trajectory per draw | Ablation. |

Implementation note for `"match"`: `_choose_subsample()` returns integer indices into
`range(sample_steps)`; convert to normalised `t` with the same
`Sampler._step_times()` path the rollout uses, so the values are exactly the ones
GDPO would have scored at. Factor that conversion into a small helper rather than
duplicating the distortion arithmetic.

### 6.4 `inner_steps` and staleness

`inner_steps: int = 1`.

- `inner_steps == 1` → one `opt.step()` per rollout. Same rollout count, same
  optimiser-step count, same reward-evaluation count as GDPO. This is Run A.
- `inner_steps > 1` → `n` optimiser steps per rollout, each with a *fresh* re-noise
  of the same buffered endpoints. The advantages `A_k` are **not** recomputed
  (the reward is a function of `G1`, which is fixed), but they become stale as
  on-policy quantities because θ moves.

v1 accepts the staleness with two guards rather than an importance ratio:

1. A hard cap (`inner_steps <= 16` by assertion, revisit with data).
2. Log `policy_drift` each inner step: `kl_clean(model, θ_rollout_snapshot, ...)` at
   the current re-noised states. This reuses existing machinery and gives an
   empirical stopping signal. Cheap: one extra frozen forward.

If drift turns out to matter, the follow-up is a PPO-style ratio on
`log p_θ(G1|G̃) − log p_{θ_old}(G1|G̃)`; explicitly out of scope for v1.

### 6.5 Class-space and masking contract

- `RolloutSampler` stashes `X1, E1` in the **network output class space**, before
  `ignore_virtual_classes`. `_apply_noise` also operates in that space and uses
  `self.limit_dist`. So re-noising consumes the stashed endpoint directly — no
  conversion, and absorbing noise (which adds a virtual class) works unchanged.
- The reward is still evaluated on the **stripped** endpoint
  (`limit_dist.ignore_virtual_classes(...)`), exactly as `GDPOTrainer.rollout()` does.
- `_apply_noise` returns `y_t` passed straight through from its `y` argument, so the
  full-model trainer passes `buf.y` and the adapter trainer passes the zero-width
  `y0` (conditioning enters through `cond`, not `y`) — mirroring `adapter.py:615`.

---

## 7. Refactor safety (D2)

The refactor of `rl.py` is the only part of this work that can break existing results,
so it gets an explicit protocol:

1. **Before touching `rl.py`**, add `tests/test_rl_parity.py`, which:
   - builds `small_model` from the existing `tests/conftest.py` fixture,
   - seeds torch / numpy / python,
   - runs `GDPOTrainer.fit(3)` and `AdapterGDPOTrainer.fit(3)` with fixed hyperparameters,
   - asserts the returned metrics and a hash of the final `state_dict` against
     **hardcoded reference values captured on current `HEAD`**.
2. Commit that test on its own. It must pass on unmodified `rl.py`.
3. Then perform the refactor. The test must still pass, unchanged.

Any diff in that test after the refactor is a bug in the refactor, full stop.

Additionally: `pytest tests/test_rl.py` (28 existing tests) must stay green throughout,
and `defog.core.__init__` must keep exporting the same names.

---

## 8. Experiment plan — ZINC logP adapter

Target: `experiments/adapter_rl_finetune__zinc.py`.

### D8 — the switch

Add one pycomex parameter:

```python
ESTIMATOR: str = "gdpo"     # "gdpo" | "ram"
RENOISE_DRAWS: int = None   # RAM only; None -> mirror SUBSAMPLE_STEPS
T_SAMPLER: str = "match"    # RAM only
INNER_STEPS: int = 1        # RAM only
```

and a single dispatch at the construction site (currently line ~556):

```python
Trainer = AdapterRAMTrainer if e.ESTIMATOR == "ram" else AdapterGDPOTrainer
trainer = Trainer(base, adapter, reward, kl_coef=e.KL_COEF, lr=e.LR, ...)
```

Everything else in the module — reward, condition sampler, eval, checkpointing,
summary JSON — is untouched. The summary gains `"estimator"` so downstream analysis
can group by it.

### Run A — matched gradient steps (the primary result)

Held identical across both arms: seed, `KL_COEF`, `LR`, `EMA_DECAY`, `ROLLOUT_SIZE`,
`SAMPLE_STEPS`, `ETA`, `OMEGA`, `SUBSAMPLE_STEPS`, iteration count, condition sampler,
reward, CRN grouping, evaluation protocol.

Varied: `ESTIMATOR ∈ {gdpo, ram}`, with RAM at `T_SAMPLER="match"`,
`RENOISE_DRAWS = SUBSAMPLE_STEPS`, `INNER_STEPS = 1`.

4 seeds per arm (matching the existing K=128 protocol), 8 runs total.

**Primary readout:** per-band logP MAE (low / high target bands reported separately —
the existing definitive run tightened HIGH by −0.06 and left LOW flat, so a
band-aggregated number would hide the effect).
**Secondary:** validity, uniqueness, scaffold diversity, adapter weight drift, the KL
trace, and the reward curve.

**Success criterion:** RAM ≥ GDPO on per-band logP MAE in ≥3 of 4 seeds, with no
regression in validity/uniqueness beyond noise. A tie is a *good* outcome for Run A —
it means the estimator is sound and Run B's throughput win is bankable.

### Run B — throughput (only if Run A is not a regression)

`INNER_STEPS ∈ {4, 8}`, `T_SAMPLER="train"`, matched **wall-clock** or matched
**reward-evaluation count** against the GDPO arm. Readout: same metrics per unit of
rollout budget, plus the `policy_drift` trace to see where staleness bites.

---

## 9. Tests (D6)

`tests/test_ram.py`, mirroring the structure of `test_rl.py`:

| test | asserts |
|---|---|
| `test_renoise_marginal_matches_kernel` | over many draws at fixed `t`, the empirical class frequencies of `G̃_t` match `t·δ(G1) + (1−t)·p_0` within tolerance |
| `test_renoise_symmetric_and_masked` | `E` symmetric, diagonal empty, padded nodes/edges masked — same invariants `sample_from_probs` guarantees |
| `test_renoise_t1_is_identity` | at `t=1`, `G̃_t == G1` exactly |
| `test_renoise_t0_is_prior` | at `t=0`, `G̃_t` is independent of `G1` |
| `test_ram_increases_toy_reward` | the RAM analogue of `test_gdpo_increases_toy_reward` — **the load-bearing test**, since it is the only direct evidence the estimator works in the discrete setting |
| `test_ram_matches_gdpo_when_states_are_forced` | inject GDPO's trajectory states in place of the re-noised ones and assert the two trainers produce identical gradients — isolates "the estimator plumbing is correct" from "the states are different" |
| `test_t_sampler_match_reproduces_gdpo_grid` | `t_sampler="match"` yields exactly the `t` values GDPO would have scored at |
| `test_inner_steps_takes_n_optimizer_steps` | `inner_steps=n` performs `n` `opt.step()` calls per rollout and re-noises freshly each time |
| `test_adapter_ram_grad_only_to_adapter` | base params keep `grad is None`; mirrors `test_adapter_scoring_grad_only_to_adapter` |
| `test_adapter_ram_scoring_matches_rollout_blend` | the composed scored policy equals the behaviour policy at any `rollout_weight`, mirroring the existing GDPO invariant |
| `test_ram_save_load_roundtrip` | checkpoint loads with `DeFoGModel.load` / adapter loader and samples with the ordinary `Sampler` |
| `test_kl_off_allocates_no_reference` | parity with `test_no_reference_allocated_when_kl_off` |

`test_ram_matches_gdpo_when_states_are_forced` is worth the effort: it splits any Run A
difference into "bug" vs "genuine estimator effect" before any cluster time is spent.

---

## 10. Sequencing

| step | work | gate |
|---|---|---|
| 1 | D2 golden parity test, captured on current HEAD | passes on unmodified `rl.py` |
| 2 | D1 refactor `RLTrainerBase` | D2 + all of `test_rl.py` green |
| 3 | D3 `renoise_states` + its 4 unit tests | tests green |
| 4 | D4 `RAMTrainer` + `RAMBuffer` | toy-reward and forced-states tests green |
| 5 | D5 `AdapterRAMTrainer` | adapter tests green |
| 6 | D7 exports | `from defog.core import RAMTrainer, AdapterRAMTrainer` |
| 7 | D8 `ESTIMATOR` switch + local smoke on `small_model` | one iteration of each arm runs end to end |
| 8 | D9 cluster script, verified on the cluster's own copy first | Run A submitted |

Steps 1–7 are local and cheap. Step 8 is the only one that consumes cluster time, and
it is gated on the forced-states test proving the plumbing is correct.

---

## 11. Risks and open questions

1. **The estimator may simply not transfer to discrete state spaces.** RAM is derived
   for continuous flow/diffusion. Mitigation: `test_ram_increases_toy_reward` is the
   cheap early falsifier; Run A is the real one. If RAM loses at matched gradient
   steps, that is a publishable negative result about the discrete case, and the code
   still stands as a comparison baseline.
2. **`t`-distribution confound.** Addressed by `t_sampler="match"` in Run A, but note
   that `"match"` is *not* what RAM's theory prescribes (`"train"` is). Run A therefore
   tests "re-noised states vs trajectory states"; Run B tests the full recipe. Both
   numbers are needed to interpret either.
3. **Staleness at `inner_steps>1`** — deliberately unhandled in v1 beyond logging.
4. **Refactor risk to shipped runs** — mitigated by D2, which is why it is step 1.
5. **CRN interaction.** `AdapterGDPOTrainer` defaults `crn=True`. CRN acts on the
   rollout's initial state, which RAM does not touch, so it carries over unchanged —
   but the within-group endpoint diversity it produces is also, incidentally, a direct
   measurement of the `G_0 → G_1` dependency discussed in the SOC explainer. Worth
   logging while we are in here; **not** in scope to act on.
6. **Naming.** "RAM" collides with memory in log output. Class names are explicit
   (`RAMTrainer`); the experiment parameter value is the lowercase string `"ram"`;
   log lines should say `estimator=ram (reinforce-adjoint-matching)` on first use.
