# DeFoG GDPO Fine-Tuning — Design & Implementation Plan

> Training-time, reward-aligned fine-tuning of a pretrained `DeFoGModel` via GDPO's
> **eager policy gradient** (Graph Diffusion Policy Optimization, arXiv:2402.16302).
>
> Status: **Phase 0 + Phase 1 implemented and tested** (`defog/core/rl.py`,
> `tests/test_rl.py` 10/10 green); Phases 2–4 designed, not yet built. Produced by a
> multi-agent design workflow (2 Opus drafts → 4 Sonnet audits → Opus consolidation
> → 2 Sonnet validations, both "ready-with-noted-fixes"); the four noted fixes are
> folded into the implementation below.

---

## 1. The idea in one paragraph

Frame denoising as an MDP (state = noisy graph `G_t`, action = next graph, reward
`r(G1)` paid only at the terminal clean graph) and maximize `E[r(G1)]`. Naive
REINFORCE pushes `∇log p_θ(G_{t-1}|G_t)` per transition, which for the clean-graph
parameterization expands into a reward-agnostic sum over all clean graphs → high
variance (fails past ~4 nodes). GDPO's **eager** estimator replaces the transition
term with the log-prob of the trajectory's own realized endpoint:

```
g(θ) = (1/K) Σ_k  A_k · Σ_{t∈T_k}  ∇_θ log p_θ(G1_k | G_{t,k})
```

**Why DeFoG fits for free:** the network already emits the clean-graph marginals
`p_θ(G1|G_t) = softmax(pred.X), softmax(pred.E)`. So `log p_θ(G1|G_t)` is exactly
the masking- and symmetry-aware cross-entropy the training loss computes, with the
rollout's sampled endpoint as the one-hot target. **The whole method reduces to an
advantage-weighted cross-entropy of each rollout's endpoint** against the network's
clean prediction at subsampled noisy states along that rollout. GDPO's intractable-
transition-likelihood problem never arises.

Convention: DeFoG's clean graph is `G1` at `t=1` (flow); GDPO/DiGress call it `G0`
at `t=0` (diffusion).

Relationship to the existing guidance / FK-SMC work: those steer a **frozen** model
at inference time; GDPO **bakes the reward into the weights**, so afterward you
sample normally with no reward in the loop. They compose (RL-tune, then still guide).

## 2. File-by-file

| File | Change | Status |
|---|---|---|
| `defog/core/sampler.py` | Extract `Sampler._step_times()` from `_advance` (behavior-preserving; the rollout recorder reuses the exact schedule). | ✅ done |
| `defog/core/rl.py` | The whole feature (below). | ✅ done |
| `defog/core/__init__.py` | Export `GDPOTrainer`, `RolloutSampler`, `RolloutBuffer`, `Reward`, `reward_from_energy`, `eager_logprob`, `kl_clean`, `group_advantage`, `EMA`. | ✅ done |
| `tests/test_rl.py` | 10 correctness tests. | ✅ done |
| `experiments/gdpo_connectivity__aqsoldb.py` | First target: train out disconnected fragments. | ✅ done |
| `experiments/gdpo_finetune__aqsoldb.py` + SLURM launcher | pycomex property-target experiment. | ⏳ Phase 3 |

No changes to `model.py`, `guidance.py`, `feynman_kac.py`, `loss.py`, `molecule.py`.

## 3. The eager gradient (implemented)

`eager_logprob(model, X_t, E_t, y_t, t, X1, E1, node_mask, lambda_edge, reduction)`
returns per-graph `log p_θ(G1|G_t)`:

- one forward → `log_softmax(pred.X)`, `log_softmax(pred.E)`;
- node CE `(X1 * logpX).sum(-1)` masked by `node_mask`; edge CE `(E1 * logpE).sum(-1)`
  masked by `_edge_upper_mask` (upper triangle of real-node pairs — `pred.E` is
  already symmetric via `transformer.py`, so the triangle is to count each undirected
  edge once, **not** for symmetry safety; no `symmetrize_edges` flag);
- `reduction="mean"` (default) divides node/edge sums by real counts → **size-
  invariant** (large molecules don't dominate the gradient by node/edge count on
  variable-size datasets); `"sum"` = true joint log-likelihood;
- `X1/E1` stay in the network's **augmented** output class space (never stripped of a
  virtual/absorbing class); a **stripped copy** goes only to the reward decoder
  (no-op for `marginal` noise, correct for `absorbing`).

Loss: `L = −(1/(K·|T|)) Σ_k A_k Σ_{t∈T_k} log p_θ(G1_k|G_{t,k})`. In `update()` this is
one `backward()` per subsampled timestep (gradient accumulation), so only one `(K,·)`
autograd graph is resident at a time — memory ≈ a training step.

`predict_clean` from `feynman_kac.py` is **not** reused: it is `@torch.no_grad` and
returns an argmax MAP one-hot, so it would kill gradients. We hand-roll the forward.

## 4. Variance reduction & KL

- `group_advantage(r, groups, mode, eps, clip)` — default **GRPO** `(r−μ)/(σ+ε)`
  (removes reward scale *and* offset, so one `lr` transfers across arbitrary rewards);
  `mean` and `none` options; `±clip`; optional per-group (per-target) whitening.
  Invalids are floored to a finite reward **before** whitening.
- `kl_clean(policy, ref, …)` — forward KL on clean marginals, toggled by `kl_coef`
  (0 → no reference allocated). The frozen reference is rebuilt from
  `DeFoGModel(**hparams) + load_state_dict` (**not** `deepcopy` of a live
  LightningModule).

## 5. Public API (implemented)

`GDPOTrainer(model, reward_fn, *, rollout_size, sample_steps, eta, omega,
time_distortion, size_dist, num_nodes, condition_sampler, subsample_steps,
subsample, lambda_edge, reduction, advantage_mode, advantage_clip, kl_coef,
ref_model, lr, weight_decay, grad_clip, ema_decay, device, seed)`.

- Plain loop (on-policy rollout doesn't fit Lightning's `training_step`/`DataLoader`).
- `reward_fn: (X1, E1, node_mask) -> (K,)`, higher = better. Reuse the guidance energy
  classes via `reward_from_energy`, or any callable.
- `.rollout()` → `RolloutBuffer`; `.update(buf)` → metrics; `.step()`; `.fit(iters, on_iter)`;
  `.save(path, use_ema=True)` → a plain DeFoG checkpoint (loads with `DeFoGModel.load`,
  samples with the ordinary `Sampler`).
- Defaults reproduce **faithful single-epoch eager REINFORCE** (`kl_coef=0`, no PPO
  surrogate). Conservative fine-tuning: `lr=1e-5`, EMA `0.999`.

`RolloutSampler(Sampler)` records passively on top of the inherited `Sampler.sample()`
(which owns `eval()`/`no_grad`/CFG/`t=0`-nudge): `_advance` stashes the constant `y`;
`_pre_step` records `(X_t, E_t, t_norm)` at pre-selected subsample indices; `_post_loop`
stashes the terminal one-hot before `ignore_virtual_classes`.

## 6. First target — connectivity (validated on two models)

`experiments/gdpo_connectivity__aqsoldb.py` trains the **disconnected-fragment** mode
out. Reward: connected+valid `1.0`, **everything else `0.0`** (disconnected `.`-in-SMILES
AND invalid are equal — an earlier disconnected=0/invalid=−0.5 reward let the optimizer
turn invalid graphs into valid-but-disconnected ones, *raising* disconnection). The
experiment is **model-agnostic**: it reconstructs the atom decoder from the checkpoint's
`atom_weights` (no dataset CSV), so it runs on AqSolDB/ZINC/GuacaMol unchanged.

Results (512 samples, eta=0 matched train/eval; GRPO, `reduction=sum`, `lr=2e-5`,
`kl_coef=0.3`, `ema_decay=0.9`, 100–120 iters):

| model | disconnected (of valid) | validity | uniqueness |
|---|---|---|---|
| ZINC uncond | 11.9% → **6.8%** (−43%) | 87.1% → 92.0% | 100% (no collapse) |
| AqSolDB uncond | 19.6% → **11.9%** (−39%) | 87.9% → 90.0% | 100% (no collapse) |

The uniqueness metric (added to `evaluate()`) is the collapse guard — a headline win with
uniqueness intact means a genuine fix, not reward-hacking.

**Robustness knobs learned empirically:** `reduction="sum"` (mean starves the bond
gradient ~n/2 — fatal for a *structural* reward); match rollout & eval sampling policy;
`lr=2e-5`/`kl_coef=0.3` is stable where `lr=3e-5` overtrains (late gnorm blowup). A binary
reward optimizes P(connected) *exactly* — fragment-size shaping optimizes the wrong
quantity and is not used. The remaining ~5–7% floor is **KL-set**, not batch- or
reward-limited: lowering `kl_coef` is the lever to push lower (trading off drift), while
larger rollout batches only reach the existing floor with lower variance.

**Snapshots / recovery** (`--ckpt-every N`, default 20): saves the EMA/deployment weights
to `<outdir>/ckpts/iter{N}.ckpt` and flushes `history.json` (per-iter reward + connected/
disconnected/invalid fractions + gnorm) each snapshot and at the end. To recover from a
collapse: find the last good iter in `history.json`, load the matching snapshot.

**Batch-size sweep** (`experiments/run_gdpo_zinc_batchsweep_kcist.sh`): 4-way KCIST sweep
(K=100/200/300/400, one per RTX 4090), fixed iterations, continuing from the round-1 ZINC
checkpoint — tests whether bigger rollout batches push below the KL-set floor.

## 7. Phased rollout

- **Phase 0** ✅ `_step_times` extraction + regression coverage.
- **Phase 1** ✅ `eager_logprob`, `group_advantage`, `RolloutSampler`, `reward_from_energy`,
  `EMA`, `GDPOTrainer` (plain eager REINFORCE), 10 tests, connectivity experiment.
- **Phase 2** ⏳ KL + frozen ref (`kl_clean` written, default off), validity bonus,
  SMILES reward cache, `condition_sampler`/groups.
- **Phase 3** ⏳ pycomex `gdpo_finetune__aqsoldb.py` (logP-target) + aslurmx launcher,
  full artifact export.
- **Phase 4** ⏳ (opt-in) PPO multi-epoch + clip, adaptive KL, multiprocessing reward decode.

## 8. Risks

Reward hacking / mode collapse → KL-to-ref + GRPO keeps gradients live + monitor
uniqueness/novelty/validity; size-driven gradient bias → `reduction="mean"`; rollout &
RDKit-decode cost → no-grad rollout, subsample only `m` states, SMILES cache; terminal-
only credit → accepted GDPO bias, `subsample="late"` up-weights informative states.
