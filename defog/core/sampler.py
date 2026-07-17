"""
Sampling orchestration for DeFoG.

The model owns the *learned per-step dynamics* (``DeFoGModel.denoise_step``); a
``Sampler`` owns *orchestration*: resolving the sampling policy (eta, omega, step
count, time distortion, guidance), building the initial state, running the
denoising loop, and post-processing into PyG graphs.

Separating the two makes constrained generation (e.g. RePaint-style inpainting)
a pure orchestration concern -- an ``InpaintingSampler`` projects the running
state onto a :class:`~defog.core.constraint.Constraint` inside the two loop
hooks, and the model never has to know anything about constraints.

``DeFoGModel.sample`` is a thin facade that constructs a default ``Sampler`` and
delegates, so this loop is the single source of truth for generation.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.data import Batch

from .data import PlaceHolder, dense_to_pyg, to_dense


class Sampler:
    """
    Orchestrates CTMC denoising for a DeFoG model.

    The sampling policy (eta, omega, sample_steps, time_distortion,
    guidance_scale) is resolved once at construction: an explicit value wins,
    otherwise the model's own default is used. This mirrors the per-call
    defaulting that ``model.sample`` used to do inline, so a fresh Sampler built
    per call is behavior-preserving for existing callers.

    Subclasses customize generation purely through two hooks, both no-ops here:

    - ``_pre_step(X, E, t_norm, node_mask)`` runs before each ``denoise_step``.
    - ``_post_loop(X, E, node_mask)`` runs once after the loop.
    """

    def __init__(
        self,
        model,
        *,
        eta: Optional[float] = None,
        omega: Optional[float] = None,
        sample_steps: Optional[int] = None,
        time_distortion: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        posterior_transform: Optional[Callable] = None,
    ):
        self.model = model
        self.eta = eta if eta is not None else model.eta
        self.omega = omega if omega is not None else model.omega
        self.sample_steps = sample_steps if sample_steps is not None else model.sample_steps
        self.time_distortion = (
            time_distortion if time_distortion is not None else model.sample_time_distortion
        )
        self.guidance_scale = (
            guidance_scale if guidance_scale is not None else model.guidance_scale
        )
        # Generic per-step rectifier of the predicted marginals (external guidance
        # rides on this; None -> unmodified, behavior-preserving for all callers).
        self.posterior_transform = posterior_transform
        # Frozen-base adapter composition (product-of-experts CFG-adapters). Only
        # AdaptedSampler sets this; None -> byte-identical legacy denoise path.
        self.composition = None
        # Schedule offset: the denoising loop runs steps ``start_step..sample_steps``.
        # 0 (the default) is full generation from noise; a refinement sampler that
        # seeds from a partially-noised guess sets this > 0 to skip the early,
        # high-noise steps. Behavior-preserving for every existing caller.
        self.start_step = 0

    # ------------------------------------------------------------------ hooks
    def _pre_step(self, X, E, t_norm, node_mask):
        """Project/modify the running state before a denoise step. No-op in base."""
        return X, E

    def _post_loop(self, X, E, node_mask):
        """Project/modify the final state after the loop. No-op in base."""
        return X, E

    def _desc(self) -> str:
        return "Sampling"

    # ------------------------------------------------------------- step / loop
    def _step_times(self, t_int, bs, device):
        """Distorted normalized ``(t_norm, s_norm)`` for outer step ``t_int``.

        Single source of truth for the sampling-time schedule: a behavior-
        preserving extraction of the former inline body of ``_advance`` so that
        callers which need to reproduce the exact per-step time (e.g. the RL
        rollout recorder in ``defog.core.rl``) share this schedule verbatim
        instead of duplicating it.
        """
        model = self.model
        t_array = t_int * torch.ones((bs, 1), device=device)
        t_norm = t_array / self.sample_steps
        s_norm = (t_array + 1) / self.sample_steps

        if model.limit_dist.noise_type == "absorbing" and t_int == 0:
            t_norm = t_norm + 1e-6

        t_norm = model.time_distorter.sample_ft(t_norm, self.time_distortion)
        s_norm = model.time_distorter.sample_ft(s_norm, self.time_distortion)
        return t_norm, s_norm

    def _advance(self, t_int, X, E, y, node_mask, use_cfg):
        """One CTMC step from time t_int -> t_int+1 (normalized, distorted).

        Includes the ``_pre_step`` hook and the absorbing-noise t=0 nudge, so it
        reproduces the original monotone loop body exactly.
        """
        model = self.model
        t_norm, s_norm = self._step_times(t_int, X.shape[0], X.device)

        X, E = self._pre_step(X, E, t_norm, node_mask)
        X, E, y = model.denoise_step(
            t_norm, s_norm, X, E, y, node_mask,
            guidance_scale=self.guidance_scale if use_cfg else None,
            eta=self.eta, omega=self.omega,
            posterior_transform=self.posterior_transform,
            composition=self.composition,
        )
        return X, E, y

    def _run_loop(self, X, E, y, node_mask, use_cfg, show_progress, on_step=None):
        """Monotone denoising loop t=start_step..sample_steps-1. Returns (X, E, y).

        ``on_step(step, total, phase)`` (optional) is invoked once per OUTER step
        with a 1-based, monotonic step index — added for external progress
        reporting; no behavior change when ``on_step`` is None. ``start_step`` is 0
        for full generation; a refinement sampler sets it > 0 to denoise only the
        tail of the schedule from a partially-noised guess.
        """
        iterator = range(self.start_step, self.sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc=self._desc())
        for t_int in iterator:
            X, E, y = self._advance(t_int, X, E, y, node_mask, use_cfg)
            if on_step is not None:
                on_step(t_int + 1, self.sample_steps, "sampling")
        return X, E, y

    # ------------------------------------------------- re-noising / time utils
    def _renoise_toward_current(self, X, E, tau, node_mask, limit_dist):
        """Re-noise the whole graph toward its own current one-hot at level tau:
        sample from ``tau*onehot(current) + (1-tau)*limit``. Edges sampled on the
        upper triangle then mirrored; padding re-masked.

        This is DeFoG's own forward interpolation ``p(x_tau | x_1=current)``. It is
        the shared primitive behind both RePaint-style time-travel (re-noise the
        running estimate mid-loop) and SDEdit-style refinement (re-noise a guess
        graph once, before denoising its tail), so it lives on the base sampler.
        """
        bs, n, dx = X.shape
        de = E.shape[-1]
        device = X.device
        limX = limit_dist.X.to(device).float()
        limE = limit_dist.E.to(device).float()

        probX = tau * X + (1.0 - tau) * limX.view(1, 1, dx)
        idxX = probX.clamp_min(0).reshape(bs * n, dx).multinomial(1).reshape(bs, n)
        Xn = F.one_hot(idxX, dx).float()

        probE = tau * E + (1.0 - tau) * limE.view(1, 1, 1, de)
        idxE = probE.clamp_min(0).reshape(bs * n * n, de).multinomial(1).reshape(bs, n, n)
        iu = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        idxE_sym = torch.zeros(bs, n, n, dtype=idxE.dtype, device=device)
        idxE_sym[:, iu] = idxE[:, iu]
        idxE_sym = idxE_sym + idxE_sym.transpose(1, 2)
        En = F.one_hot(idxE_sym.long(), de).float()

        masked = PlaceHolder(X=Xn, E=En, y=None).mask(node_mask)
        return masked.X, masked.E

    def _distorted_time(self, t_int):
        """Distorted normalized time (scalar float) for integer step ``t_int`` --
        the noise level the loop assumes the running state sits at when it enters
        step ``t_int``. Shared by time-travel jumps and refinement seeding."""
        model = self.model
        t_norm = t_int / self.sample_steps
        if model.limit_dist.noise_type == "absorbing" and t_int == 0:
            t_norm = t_norm + 1e-6
        t_norm = model.time_distorter.sample_ft(
            torch.tensor([[t_norm]], dtype=torch.float32), self.time_distortion
        )
        return float(t_norm.reshape(-1)[0].item())

    # ----------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_nodes: Optional[Union[int, torch.Tensor]] = None,
        size_dist=None,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
        on_step: Optional[Callable] = None,
    ):
        """
        Generate graph samples. Returns a list of PyG ``Data`` objects.

        Reproduces the original ``model.sample`` behavior (initial noise, time
        distortion, CFG, virtual-class cleanup, dense->PyG) with the two hooks
        inserted around the loop.
        """
        model = self.model

        # Force eval mode so dropout is inert during denoising; restore on exit.
        was_training = model.training
        model.eval()

        device = device if device is not None else model.device

        node_mask, n_nodes, X, E, y, y_raw, use_cfg = model._prepare_generation(
            num_samples, num_nodes, size_dist, condition, self.guidance_scale, device
        )

        X, E, y = self._run_loop(X, E, y, node_mask, use_cfg, show_progress, on_step=on_step)

        # Hook: constrained samplers install the exact target here (e.g. tau=1
        # core), after the model's own final-step MAP decode inside denoise_step.
        X, E = self._post_loop(X, E, node_mask)

        X, E, _ = model.limit_dist.ignore_virtual_classes(X, E)
        samples = dense_to_pyg(X, E, y_raw, node_mask, n_nodes)

        if was_training:
            model.train()
        return samples


class InpaintingSampler(Sampler):
    """
    RePaint-style constrained generation: freeze a substructure (the constraint's
    core) and grow ``n_free`` new nodes around it.

    Each step the running state is projected onto the constraint at the current
    (distorted) time so the frozen region stays at the same noise level as the
    free region -- keeping the network input on-distribution. After the loop the
    state is projected at tau=1, installing the exact core.

    Optional RePaint "time-travel" resampling (``resample=True``) periodically
    jumps back to a noisier time and re-denoises, giving the free region repeated
    chances to harmonize with the core at the boundary. Because discrete flow has
    no closed-form backward kernel, the jump re-noises the *current* sample toward
    itself via the model's own interpolation p(x_tau | x_1=current) -- the direct
    analogue of RePaint adding forward noise to the current estimate -- then
    re-pins the core. Costs extra forward passes; helps mainly on hard cores.

    Resampling knobs:
        resample: turn time-travel on (default off -> plain replacement).
        n_resample: resamples per jump point (RePaint's r).
        jump_length: how many steps to jump back (defaults to ~10% of steps).
        jump_interval: jump every this many forward steps (defaults to jump_length).
    """

    def __init__(
        self,
        model,
        constraint,
        *,
        resample: bool = False,
        n_resample: int = 4,
        jump_length: Optional[int] = None,
        jump_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.constraint = constraint
        self.resample = resample
        self.n_resample = n_resample
        self.jump_length = jump_length if jump_length is not None else max(1, self.sample_steps // 10)
        self.jump_interval = jump_interval if jump_interval is not None else self.jump_length

    def _desc(self) -> str:
        return "Inpainting+TT" if self.resample else "Inpainting"

    def _pre_step(self, X, E, t_norm, node_mask):
        return self.constraint.project(X, E, t_norm, node_mask, self.model.limit_dist)

    def _post_loop(self, X, E, node_mask):
        return self.constraint.project(X, E, 1.0, node_mask, self.model.limit_dist)

    # ------------------------------------------------------- time-travel loop
    def _run_loop(self, X, E, y, node_mask, use_cfg, show_progress, on_step=None):
        if not self.resample:
            return super()._run_loop(X, E, y, node_mask, use_cfg, show_progress, on_step=on_step)

        model = self.model
        N = self.sample_steps
        j = self.jump_length
        limit = model.limit_dist

        pbar = tqdm(total=N, desc=self._desc()) if show_progress else None
        t_int = 0
        while t_int < N:
            X, E, y = self._advance(t_int, X, E, y, node_mask, use_cfg)
            t_int += 1
            if pbar is not None:
                pbar.update(1)
            if on_step is not None:
                on_step(t_int, N, "inpaint")

            # Time-travel: at each jump point, re-noise back j steps and
            # re-denoise, n_resample times, harmonizing the core<->free boundary.
            if (t_int % self.jump_interval == 0) and (t_int < N):
                t_back = max(0, t_int - j)
                for _ in range(self.n_resample):
                    tau = self._distorted_time(t_back)
                    X, E = self._renoise_toward_current(X, E, tau, node_mask, limit)
                    # re-pin the core at the jumped-back noise level
                    X, E = self.constraint.project(
                        X, E, torch.full((X.shape[0], 1), tau, device=X.device),
                        node_mask, limit,
                    )
                    for tt in range(t_back, t_int):
                        X, E, y = self._advance(tt, X, E, y, node_mask, use_cfg)
                        if on_step is not None:
                            on_step(t_int, N, "inpaint")
        if pbar is not None:
            pbar.close()
        return X, E, y

    # ----------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        n_free: int,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
        on_step: Optional[Callable] = None,
    ):
        """
        Generate ``num_samples`` completions, each with ``core_size + n_free``
        total nodes (core in slots ``[0, k)``; new nodes in ``[k, k + n_free)``).
        """
        k = self.constraint.k
        total = k + int(n_free)
        if total > self.model.max_nodes:
            raise ValueError(
                f"core size k={k} + n_free={n_free} = {total} exceeds the model's "
                f"max_nodes={self.model.max_nodes}. Reduce n_free (or the core)."
            )
        num_nodes = torch.full((num_samples,), total, dtype=torch.long)
        return super().sample(
            num_samples,
            num_nodes=num_nodes,
            condition=condition,
            device=device,
            show_progress=show_progress,
            on_step=on_step,
        )


class GuidedSampler(Sampler):
    """
    Exact posterior-based discrete guidance at sampling time.

    Thin sugar over :class:`Sampler`: it wires a trained
    :class:`~defog.core.guidance.ExactGuidance` in as the ``posterior_transform``
    so every denoise step reweights the predicted clean-graph marginals toward the
    guided target (one extra forward pass of the small guidance network per step).
    Everything else -- rate matrix, eta/omega, time distortion, final MAP decode --
    is inherited unchanged. Because the hook lives on the base ``Sampler``, guidance
    also composes with inpainting via
    ``InpaintingSampler(model, constraint, posterior_transform=guidance.reweight)``.
    """

    def __init__(self, model, guidance, **kwargs):
        super().__init__(model, posterior_transform=guidance.reweight, **kwargs)
        self.guidance = guidance

    def _desc(self) -> str:
        return "Guided sampling"


class AdaptedSampler(Sampler):
    """Sampling under a frozen-base adapter composition (product-of-experts CFG).

    Thin sugar over :class:`Sampler`, analogous to :class:`GuidedSampler`: it holds
    an :class:`~defog.core.adapter.AdapterComposition` and threads it into every
    ``denoise_step`` unconditionally (NOT gated by ``use_cfg``, which is always
    ``False`` for the ``cond_dim=0`` frozen bases this targets). Everything else --
    eta/omega, time distortion, posterior_transform, size distribution -- is
    inherited unchanged, so it also composes with inpainting/refinement.
    """

    def __init__(self, model, composition, **kwargs):
        super().__init__(model, **kwargs)
        for br in composition.branches:
            br.adapter.check_compatible(model)   # fail fast on dim/n_layers/base mismatch
        self.composition = composition

    def _desc(self) -> str:
        return f"Adapted[{len(self.composition)}x,{self.composition.mode}]"


class RefinementSampler(Sampler):
    """
    SDEdit-style refinement: seed the denoiser with a *guess* graph instead of
    pure noise, so the model only walks it the last few steps onto its learned
    (valid) data manifold -- "the last few meters toward validity".

    Given rough guess graphs (e.g. from a deterministic first-guess algorithm),
    each guess is treated as a clean graph ``x_1``, partially re-noised to level
    ``t_start`` via the model's own forward interpolation ``p(x_tau | x_1=guess)``
    (the SAME primitive RePaint time-travel uses -- ``_renoise_toward_current``),
    then denoised over only the *tail* of the schedule (steps
    ``start_step .. sample_steps-1``, where ``start_step = round(t_start *
    sample_steps)``).

    ``t_start`` in (0, 1) is the faithfulness<->validity knob (SDEdit's noise
    ratio):

    - near 1  -> inject little noise -> few steps -> stay close to the guess
                 (light polish, keeps its structure and its remaining flaws);
    - lower   -> inject more noise -> more steps -> more freedom to restructure
                 (heavier repair, drifts further from the guess);
    - -> 0    -> discards the guess; ~equivalent to full generation from noise.

    The node *count* of each guess is preserved (this is refinement, not the
    node-growing inpainting -- use :class:`InpaintingSampler` to add nodes around
    a frozen core). Node types and edges are both free to change.

    Because the reverse process is a stochastic CTMC, each call refines the same
    guess differently -- draw several and keep the best under your own validity
    check (a cheap ensemble of repairs). Composes with target guidance (``omega``),
    stochasticity (``eta``), and per-step guidance (``posterior_transform``) as
    usual; all inherited.

    Guess format: a list of PyG ``Data`` with one-hot ``x`` of width
    ``num_node_classes`` and one-hot ``edge_attr`` of width ``num_edge_classes``
    (exactly what the model emits and what ``dense_to_pyg`` produces), undirected
    (both directions in ``edge_index``). Each guess must have <= ``max_nodes``.
    """

    def __init__(self, model, *, t_start: float = 0.7, **kwargs):
        super().__init__(model, **kwargs)
        if not (0.0 < t_start < 1.0):
            raise ValueError(f"t_start must be in (0, 1); got {t_start}")
        self.t_start = float(t_start)
        # High trust -> start late in the schedule -> denoise few steps. Clamp to
        # [1, sample_steps-1]: always run >=1 step, and never step 0 (whose t=0
        # absorbing-noise nudge would desync from the seeded noise level).
        k = round(self.t_start * self.sample_steps)
        self.start_step = int(min(max(k, 1), self.sample_steps - 1))

    def _desc(self) -> str:
        return "Refining"

    def _encode_guesses(self, guesses, device):
        """List of PyG ``Data`` guesses -> dense one-hot ``(X, E, node_mask,
        n_nodes)`` in the model's class space. Validates class widths and sizes."""
        if not isinstance(guesses, (list, tuple)) or len(guesses) == 0:
            raise TypeError("guesses must be a non-empty list of PyG Data objects")
        batch = Batch.from_data_list(list(guesses)).to(device)
        dense, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        X, E = dense.X.float(), dense.E.float()

        dx, de = self.model.num_node_classes, self.model.num_edge_classes
        if X.shape[-1] != dx:
            raise ValueError(
                f"guess node features have {X.shape[-1]} classes but the model "
                f"expects num_node_classes={dx} (one-hot). Encode guesses in the "
                f"model's class space."
            )
        if E.shape[-1] != de:
            raise ValueError(
                f"guess edge features have {E.shape[-1]} classes but the model "
                f"expects num_edge_classes={de} (one-hot, incl. the no-edge class)."
            )
        n_nodes = node_mask.sum(-1).long()
        if int(n_nodes.max().item()) > self.model.max_nodes:
            raise ValueError(
                f"a guess has {int(n_nodes.max().item())} nodes, exceeding the "
                f"model's max_nodes={self.model.max_nodes}."
            )
        return X, E, node_mask, n_nodes

    @torch.no_grad()
    def refine(
        self,
        guesses,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
        on_step: Optional[Callable] = None,
    ):
        """Refine ``guesses`` (list of PyG ``Data``) toward the model's manifold.

        Returns a list of refined PyG ``Data``, one per guess, each with the same
        node count as its guess. ``condition`` (shape ``(len(guesses), cond_dim)``)
        is used for conditional models exactly as in ``model.sample``.
        """
        model = self.model

        # Force eval mode so dropout is inert during denoising; restore on exit.
        was_training = model.training
        model.eval()
        device = device if device is not None else model.device

        # Guess graphs -> dense one-hot seed state.
        X1, E1, node_mask, n_nodes = self._encode_guesses(guesses, device)
        bs = X1.shape[0]

        # Partially re-noise the guess to the start-step noise level:
        # p(x_tau | x_1 = guess). tau matches what step `start_step` assumes.
        tau = self._distorted_time(self.start_step)
        X, E = self._renoise_toward_current(X1, E1, tau, node_mask, model.limit_dist)

        # Conditioning (shared with the noise-seeded generation path).
        y, y_raw, use_cfg = model._prepare_condition(
            bs, condition, self.guidance_scale, device
        )

        X, E, y = self._run_loop(X, E, y, node_mask, use_cfg, show_progress, on_step=on_step)
        X, E = self._post_loop(X, E, node_mask)

        X, E, _ = model.limit_dist.ignore_virtual_classes(X, E)
        samples = dense_to_pyg(X, E, y_raw, node_mask, n_nodes)

        if was_training:
            model.train()
        return samples
