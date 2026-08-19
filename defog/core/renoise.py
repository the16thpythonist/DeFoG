"""
Re-noising a known clean graph, for the RAM and DAM estimators.

GDPO's eager policy gradient scores ``log p_theta(G1 | G_t)`` at the states the
rollout trajectory actually occupied. The re-noising estimators score at states drawn
afresh from the *pretraining* kernel

    p_{t|1}(z | G1) = t * delta(z, G1) + (1 - t) * p_0(z)

conditioned on the rollout's endpoint. Two reasons this is the right kernel here and
not an approximation of something else:

* It is exactly ``DeFoGModel._apply_noise``, the function pretraining already uses,
  already exercised in this mode by ``guidance.py`` and ``adapter.py``. Edge symmetry,
  the empty diagonal and node/edge masking all come free from ``sample_from_probs``.
* DAM draws its training states from the two-sided bridge
  ``p^base_{t|0,1}(. | X_0, X_1)``. For DeFoG's linear mixture path that conditional
  references ``X_1`` only, so the bridge collapses to ``p_{t|1}(. | X_1)`` and no
  separate bridge kernel is needed. See docs/dam_design.md section 3.4.

Deliberately NOT built on ``Sampler._renoise_toward_current``, which looks like the
same primitive: it does not set padded rows to a uniform distribution, so an all-zero
padded row makes ``torch.multinomial`` raise at tau=1.

The two halves are separate so they can be tested separately: :func:`draw_times`
decides *which* noise levels, :func:`renoise_states` re-noises at them.
"""

from typing import List, Optional, Sequence, Tuple

import torch

from .sampler import Sampler

__all__ = ["draw_times", "renoise_states", "TIME_MODES"]


#: ``t`` distributions. They differ in COUPLING as much as in density -- see below.
TIME_MODES = ("match", "train", "ram", "uniform")


def draw_times(
    model,
    bs: int,
    device,
    *,
    mode: str = "train",
    n_draws: Optional[int] = None,
    step_indices: Optional[Sequence[int]] = None,
    sample_steps: Optional[int] = None,
    time_distortion: str = "identity",
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """Return ``n_draws`` noise levels, each a ``(bs, 1)`` tensor in [0, 1].

    ``mode``:

    * ``"match"`` -- the exact distorted grid values the GDPO rollout would have
      scored at. ``step_indices`` (from the trainer's own ``_choose_subsample()``)
      are converted through ``Sampler._distorted_time``, the single source of truth
      for the per-step schedule, so no arithmetic is duplicated. One value is SHARED
      across the batch, exactly as the rollout recorder does.
    * ``"train"`` -- ``model.time_distorter.train_ft``: the pretraining density, and
      ``bs`` INDEPENDENT draws per level, which is what RAM/DAM Alg. 1 does.
    * ``"ram"`` -- ``t = 1 - sqrt(U)``, i.e. ``p(t) = 2(1 - t)``, RAM's own timestep
      recipe expressed in DeFoG's convention (t=1 is data).
    * ``"uniform"`` -- ``U(0, 1)``.

    Note ``"match"`` and ``"train"`` have nearly the SAME density on a
    polydec-pretrained base (measured exact sup-CDF distance 1/sample_steps). What
    actually differs is the coupling: ``"match"`` shares one value across all
    trajectories, ``"train"`` draws them independently.
    """
    if mode not in TIME_MODES:
        raise ValueError(f"unknown time mode {mode!r}; expected one of {TIME_MODES}")

    if mode == "match":
        if step_indices is None:
            raise ValueError(
                "mode='match' needs step_indices -- pass the trainer's own "
                "_choose_subsample() output so the noise levels are exactly the ones "
                "the GDPO arm would have scored at."
            )
        if sample_steps is None:
            raise ValueError("mode='match' needs sample_steps")
        probe = Sampler(model, sample_steps=sample_steps, time_distortion=time_distortion)
        return [
            torch.full((bs, 1), probe._distorted_time(int(i)), device=device)
            for i in step_indices
        ]

    if n_draws is None:
        raise ValueError(f"mode={mode!r} needs n_draws")

    out = []
    for _ in range(n_draws):
        if mode == "train":
            t = model.time_distorter.train_ft(bs, device)
        else:
            u = torch.rand(bs, 1, device=device, generator=generator)
            t = 1.0 - torch.sqrt(u) if mode == "ram" else u
        out.append(t.reshape(bs, 1).to(device))
    return out


@torch.no_grad()
def renoise_states(
    model,
    X1: torch.Tensor,
    E1: torch.Tensor,
    y: torch.Tensor,
    node_mask: torch.Tensor,
    times: Sequence[torch.Tensor],
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Re-noise the endpoint ``(X1, E1)`` at each level in ``times``.

    Returns a list of ``(X_t, E_t, t)`` in the same shape as ``RolloutBuffer.states``,
    so it drops straight into the existing update loops.

    ``X1`` / ``E1`` must be one-hot in the network's OUTPUT class space -- i.e. as
    ``RolloutSampler`` stashes them, BEFORE ``ignore_virtual_classes``. That is the
    space ``_apply_noise`` operates in, so absorbing noise (which adds a virtual
    class) works without conversion.

    ``y`` is passed through unchanged and reappears as ``y_t``; a zero-width ``(K, 0)``
    tensor is fine and is what the adapter path uses, since conditioning enters
    through ``cond``, not through ``y``.
    """
    return [
        (nd["X_t"], nd["E_t"], nd["t"])
        for nd in (
            model._apply_noise(X1, E1, y, node_mask, t=t.to(X1.device)) for t in times
        )
    ]
