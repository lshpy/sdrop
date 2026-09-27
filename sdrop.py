"""
Suppressive Dropout (SDrop)
----------------------------
Score variants:

  SDrop        : full EGPG score   s_c = E_c * (1 - P_c)
  SDropEnergy  : energy only       s_c = E_c
  SDropPeak    : peakedness only   s_c = (1 - P_c)          [ablation]
  SDropRandom  : uniform score     s_c = 1                  [ablation control]
  SGridLC      : EGPG computed per G x G spatial grid cell (spatially-aware)

All variants:
  - During training : stochastically drop channels proportional to their score
  - During eval     : pass-through (no dropout, no rescaling needed)

Optional score refinements:

  peakedness='entropy'  : resolution-invariant peakedness
  norm='mean'           : mean-preserving drop probability (realises the nominal rate)
  gamma=None            : self-normalising energy scale

Reference:
  Lee, S. "Suppressive Dropout" (xAI 2026 Late-Breaking Work / Demo).
  An extended journal version is in preparation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Score primitives
# ---------------------------------------------------------------------------

def channel_energy(x: torch.Tensor, gamma: float = 1.0, delta: float = 1.0,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    E_c = 1 - (1 + gamma * Sigma_c)^{-delta},   Sigma_c = (1/HW) sum_hw x_{c,h,w}^2

    gamma=None selects the self-normalising scale gamma = 1 / median_c(Sigma_c),
    computed per sample. This keeps the score in the non-saturated region of the
    map for any activation magnitude, removing the need to hand-tune gamma per
    architecture (CNN gamma=1 vs ViT gamma=100).

    Args:
        x: (B, C, H, W)
    Returns:
        energy: (B, C)  in (0, 1)
    """
    mean_sq = x.pow(2).mean(dim=(2, 3))                       # (B, C)
    if gamma is None:
        med = mean_sq.median(dim=1, keepdim=True).values      # (B, 1)
        gamma_t = 1.0 / med.clamp(min=eps)
    else:
        gamma_t = gamma
    return 1.0 - (1.0 + gamma_t * mean_sq).pow(-delta)


def spatial_peakedness(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    P_c = max_{h,w} |x_{c,h,w}| / (sum_{h,w} |x_{c,h,w}| + eps)

    High value  -> activation is spatially localized  (informative, spare)
    Low value   -> activation is diffuse              (over-inhibitory)

    Note: the range of P_c is [1/(HW), 1], so its numerical scale depends on the
    spatial resolution of the layer. Within-layer ranking is unaffected, but
    cross-layer comparison of raw values is not meaningful. See
    entropy_peakedness for a resolution-invariant alternative.

    Args:
        x: (B, C, H, W)
    Returns:
        peakedness: (B, C)  in (0, 1]
    """
    abs_x = x.abs()
    peak = abs_x.amax(dim=(2, 3))                # (B, C)
    total = abs_x.sum(dim=(2, 3))                # (B, C)
    return peak / (total + eps)


def entropy_peakedness(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    P~_c = 1 - H(pi_c) / log(HW),   pi_{c,h,w} = |x_{c,h,w}| / sum |x_{c,.,.}|

    Resolution-invariant: 0 for a uniform channel, 1 for a single-position
    impulse, independent of H*W. Drop-in replacement for spatial_peakedness
    when transferring SDrop across architectures or input modalities.

    Args:
        x: (B, C, H, W)
    Returns:
        peakedness: (B, C)  in [0, 1]
    """
    B, C, H, W = x.shape
    abs_x = x.abs().reshape(B, C, H * W)
    pi = abs_x / (abs_x.sum(dim=2, keepdim=True) + eps)
    ent = -(pi * (pi + eps).log()).sum(dim=2)                 # (B, C)
    return 1.0 - ent / torch.log(torch.tensor(float(H * W), device=x.device))


def _peakedness(x: torch.Tensor, kind: str = 'max', eps: float = 1e-6) -> torch.Tensor:
    if kind == 'max':
        return spatial_peakedness(x, eps)
    if kind == 'entropy':
        return entropy_peakedness(x, eps)
    raise ValueError(f"Unknown peakedness kind: '{kind}'. Choose 'max' or 'entropy'.")


def compute_score(x: torch.Tensor, score: str = 'egpg', gamma: float = 1.0,
                  delta: float = 1.0, peakedness: str = 'max',
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Unified score front-end. Returns (B, C).

      'egpg'    : s_c = E_c * (1 - P_c)     full score
      'energy'  : s_c = E_c                 energy only
      'peak'    : s_c = (1 - P_c)           peakedness only   [ablation]
      'random'  : s_c = 1                   uniform            [ablation control]
      'gap'     : s_c = mean_hw |x_c|       pure activation magnitude [ablation]

    'gap' is the channel version of the score used by WCD (AAAI'19) / ProbDropBlock (ACML'23).
    'energy' already contains the saturating nonlinearity E_c = 1-(1+gamma*Sigma_c)^-delta,
    so it is not a "plain activation magnitude" control. To separate the contribution of EGPG
    from that prior work, this pure GAP control is required.
    """
    if score == 'gap':
        return x.abs().mean(dim=(2, 3))
    if score == 'random':
        return torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
    if score == 'energy':
        return channel_energy(x, gamma, delta, eps)
    if score == 'peak':
        return 1.0 - _peakedness(x, peakedness, eps)
    if score == 'egpg':
        E = channel_energy(x, gamma, delta, eps)
        P = _peakedness(x, peakedness, eps)
        return E * (1.0 - P)
    raise ValueError(f"Unknown score: '{score}'. "
                     "Choose from: egpg, energy, peak, random, gap")


def drop_mask_from_scores(scores: torch.Tensor, drop_rate: float,
                          norm: str = 'max', eps: float = 1e-6) -> torch.Tensor:
    """
    norm='max'  : p_drop,c = drop_rate * s_c / max_c(s_c)          (original)
    norm='mean' : p_drop,c = min(1, drop_rate * C * s_c / sum_c s_c)

    With 'max' normalisation only the top-scoring channel is dropped at exactly
    drop_rate, so the expected dropped fraction is
        p_bar = drop_rate * mean_c(s_c) / max_c(s_c) <= drop_rate.
    'mean' fixes the expected fraction at drop_rate exactly and is robust to a
    single outlying channel score.

    mask[b, c] = Bernoulli(1 - p_drop,c)   (1 = keep, 0 = drop)

    Args:
        scores: (B, C)
    Returns:
        mask: (B, C, 1, 1)  float
    """
    return _bernoulli_keep(
        _drop_probs_from_scores(scores, drop_rate, norm, eps))


def _drop_probs_from_scores(scores: torch.Tensor, drop_rate: float,
                            norm: str = 'max', eps: float = 1e-6):
    """(B, C) scores -> (B, C) per-channel drop probabilities."""
    if norm == 'max':
        denom = scores.amax(dim=1, keepdim=True).clamp(min=eps)
        return drop_rate * scores / denom
    if norm == 'mean':
        C = scores.shape[1]
        denom = scores.sum(dim=1, keepdim=True).clamp(min=eps)
        return (drop_rate * C * scores / denom).clamp(max=1.0)
    raise ValueError(f"Unknown norm: '{norm}'. Choose 'max' or 'mean'.")


def _bernoulli_keep(drop_probs: torch.Tensor) -> torch.Tensor:
    """(B, C) drop probabilities -> (B, C, 1, 1) keep mask."""
    keep = (torch.rand_like(drop_probs) > drop_probs).float()
    return keep.unsqueeze(-1).unsqueeze(-1)           # (B, C, 1, 1)


# ---------------------------------------------------------------------------
# SDrop  (single module covering every score variant)
# ---------------------------------------------------------------------------

class DropRateMonitor:
    """
    Mixin that accumulates the *realised* fraction of dropped units.

    p_base is only an upper bound on the average drop rate under 'max'
    normalisation, so the nominal rate is not directly
    comparable against unstructured dropout at the same nominal p. This
    monitor records what actually happened, at negligible cost, so every run
    can report its measured effective rate p_bar alongside p_base.
    """
    def _init_monitor(self):
        self._drop_sum = 0.0
        self._drop_batches = 0
        self.last_drop_frac = float('nan')

    # -- drop-rate schedule / participation ------------------------------
    # DropBlock (NeurIPS'18): linear ramp-up to the target rate. Strong dropout from the
    #   start destabilises training; ramping it up improves both accuracy and robustness.
    # FocusedDropout (2021): 'participation', applying the drop to only part of the batch.
    #   An orthogonal axis that controls "how often" instead of "how hard".
    def _init_schedule(self, schedule: str = 'const', participation: float = 1.0,
                       warmup_frac: float = 0.0):
        self.schedule = schedule
        self.participation = float(participation)
        # No dropping during this initial fraction of training. Guided Dropout has to learn
        # its strength parameters and used 40/200 = 0.2. SDrop has nothing to learn, but
        # whether intervening before early representations settle is harmful is a separate
        # question, so it is exposed as an axis.
        self.warmup_frac = float(warmup_frac)
        self._progress = 0.0          # 0.0 -> 1.0, training progress

    def set_progress(self, progress: float):
        """Set training progress (0-1). Called by train.py every epoch."""
        self._progress = float(min(max(progress, 0.0), 1.0))

    def _current_rate(self) -> float:
        base = self.drop_rate
        # With warmup_frac=0 there must be no warm-up. Comparing with <= would return 0
        # on the first call (progress 0) and disable even the const schedule.
        _wu = getattr(self, 'warmup_frac', 0.0)
        if _wu > 0.0 and self._progress <= _wu:
            return 0.0
        sched = getattr(self, 'schedule', 'const')
        if sched == 'const':
            return base
        if sched == 'linear':
            # DropBlock (NeurIPS'18): linear ramp-up 0 -> target
            return base * self._progress
        if sched == 'early':
            # Drop only in the first half of training (no drop in the second). --warmup_frac 0.5 is its mirror (second half only).
            return base if self._progress <= 0.5 else 0.0
        if sched == 'gd':
            # Guided Dropout (AAAI'19) profile. Original paper (200 epochs):
            #   40 epochs of strength learning (no drop) -> p for 60 epochs -> 0.75p for 50 -> 0.5p for 50
            #   with p_base=0.2 this gives 0.2 / 0.15 / 0.1.
            # Expressed as fractions to be independent of the epoch count: 0.20 / 0.30 / 0.25 / 0.25.
            # Opposite direction to DropBlock (decreasing). The two papers use opposite schedules,
            # so which one is right is an empirical question.
            t = self._progress
            if t <= 0.20:
                return 0.0
            if t <= 0.50:
                return base
            if t <= 0.75:
                return base * 0.75
            return base * 0.5
        if sched == 'gdtopk':
            # Guided Dropout top-k profile (original paper, 200 epochs):
            #   p 10ep / no drop 40 / 0.75p 10 / no drop 40 / 0.5p 10 / no drop 50
            #   Deterministic selection applied continuously breaks training, hence the pauses.
            # Fractions: 0.05 / 0.20 / 0.05 / 0.20 / 0.05 / 0.25  (the first 0.20 is warm-up)
            t = self._progress
            if t <= 0.20:      return 0.0          # warm-up
            if t <= 0.25:      return base
            if t <= 0.45:      return 0.0
            if t <= 0.50:      return base * 0.75
            if t <= 0.70:      return 0.0
            if t <= 0.75:      return base * 0.5
            return 0.0
        raise ValueError(f"Unknown schedule: '{sched}'. "
                         "Choose 'const', 'linear', 'early', 'gd' or 'gdtopk'.")

    def _record(self, mask: torch.Tensor):
        # mask is 1 = keep, 0 = drop
        frac = (1.0 - mask).mean().item()
        self.last_drop_frac = frac
        self._drop_sum += frac
        self._drop_batches += 1

    def effective_drop_rate(self) -> float:
        """Mean realised drop fraction since the last reset (NaN if unused)."""
        if self._drop_batches == 0:
            return float('nan')
        return self._drop_sum / self._drop_batches

    def reset_monitor(self):
        self._drop_sum = 0.0
        self._drop_batches = 0


class SDrop(nn.Module, DropRateMonitor):
    """
    Suppressive Dropout.

    Args:
        drop_rate  : base drop probability p_base
        score      : 'egpg' | 'energy' | 'peak' | 'random'
        gamma      : LRN energy scale; None selects the self-normalising scale
        delta      : LRN energy exponent
        peakedness : 'max' (paper default) | 'entropy' (resolution-invariant)
        norm       : 'max' (paper default) | 'mean' (mean-preserving)
        grad_mode  : 'off' (paper default)
                     'suppress' — s_c <- s_c / (1 + |dL/da_c|), i.e. drop units
                                  that are loud and diffuse but *task-irrelevant*
                     'amplify'  — s_c <- s_c * (1 + |dL/da_c|), i.e. drop units
                                  that are loud, diffuse and *loss-dominant*
        grad_ema   : EMA momentum for the per-channel gradient magnitude
        eps        : numerical stability constant

    The forward pass alone cannot know whether a high-energy channel is doing
    useful work. The gradient magnitude |dL/da_c| supplies exactly that missing
    signal, at no extra cost: it is captured by a hook during the backward pass
    and reused on the next step (one-step-stale, as in gradient-based pruning).
    Which direction is correct is an empirical question -- 'suppress' follows the
    "drop loud but useless units" reading, 'amplify' the "drop the dominant
    shortcut" reading -- so both are provided as an ablation.
    """
    def __init__(self, drop_rate: float = 0.1, score: str = 'egpg',
                 gamma: float = 1.0, delta: float = 1.0,
                 peakedness: str = 'max', norm: str = 'max',
                 grad_mode: str = 'off', grad_ema: float = 0.9,
                 schedule: str = 'const', participation: float = 1.0,
                 warmup_frac: float = 0.0,
                 rescale: bool = False, rescale_floor: float = 0.1,
                 rescale_mode: str = 'prob',
                 select: str = 'bernoulli',
                 assign: str = 'score',
                 eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.score = score
        # assign: how the computed set of probabilities is assigned to channels.
        #   'score'   - in score order (the original SDrop)
        #   'shuffle' - randomly permute the probability vector along the channel axis, per sample.
        #               The set of probabilities, expected number of drops and mask entropy are unchanged;
        #               only the link between score and channel is broken.
        #   'reverse' - keep the set of probabilities but assign the lowest ones to the highest-scoring channels.
        #   'cross'   - roll the probability vector along the **batch axis** (random offset>=1): sample i
        #               receives the whole probability vector of another sample j. Channel order and budget
        #               are unchanged; only the link to its own input is broken (cross-sample control).
        assert assign in ('score', 'shuffle', 'reverse', 'cross'), assign
        self.assign = assign
        self.gamma = gamma
        self.delta = delta
        self.peakedness = peakedness
        self.norm = norm
        self.grad_mode = grad_mode
        self.grad_ema = grad_ema
        self.eps = eps
        self._grad_mag = None          # (C,) EMA of |dL/da_c|, normalised to mean 1
        self.rescale = bool(rescale)
        self.rescale_floor = float(rescale_floor)   # rescale factor capped at 1/floor
        assert rescale_mode in ('prob', 'count'), rescale_mode
        self.rescale_mode = rescale_mode
        assert select in ('bernoulli', 'topk'), select
        self.select = select
        self._init_monitor()
        self._init_schedule(schedule, participation, warmup_frac)

    # -- gradient capture -------------------------------------------------
    def _capture_grad(self, grad: torch.Tensor):
        # grad: (B, C, H, W) -> per-channel magnitude, normalised to mean 1
        g = grad.detach().abs().mean(dim=(0, 2, 3))
        g = g / (g.mean() + self.eps)
        if self._grad_mag is None or self._grad_mag.shape != g.shape:
            self._grad_mag = g
        else:
            m = self.grad_ema
            self._grad_mag = m * self._grad_mag.to(g.device) + (1.0 - m) * g

    def _apply_grad_guidance(self, scores: torch.Tensor) -> torch.Tensor:
        if self.grad_mode == 'off' or self._grad_mag is None:
            return scores
        g = self._grad_mag.to(scores.device).unsqueeze(0)      # (1, C)
        if self.grad_mode == 'suppress':
            return scores / (1.0 + g)
        if self.grad_mode == 'amplify':
            return scores * (1.0 + g)
        raise ValueError(f"Unknown grad_mode: '{self.grad_mode}'. "
                         "Choose 'off', 'suppress' or 'amplify'.")

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x
        rate = self._current_rate()
        if rate <= 0.0:
            return x
        # participation: decide whether to skip this batch (FocusedDropout style)
        part = getattr(self, 'participation', 1.0)
        if part < 1.0 and float(torch.rand(())) >= part:
            self._record(torch.ones(1, 1, 1, 1, device=x.device))
            return x
        scores = compute_score(x, self.score, self.gamma, self.delta,
                               self.peakedness, self.eps)          # (B, C)
        if self.grad_mode != 'off':
            scores = self._apply_grad_guidance(scores)
            if x.requires_grad:
                x.register_hook(self._capture_grad)
        if self.assign == 'cross' and x.shape[0] > 1:
            # Batch-axis derangement: with a roll of offset>=1 nobody receives its own vector.
            scores = scores.roll(int(torch.randint(1, x.shape[0], ())), dims=0)
        if self.select == 'topk':
            # Top-k as in Guided Dropout (AAAI'19): drop the k highest-scoring channels
            # **deterministically**, instead of tilting probabilities.
            #   The original paper found that applying this continuously breaks training and
            #   alternated drop / no drop (10 epochs drop / 40 epochs no drop). schedule='gdtopk'
            #   reproduces that alternation.
            C = scores.shape[1]
            k = int(round(C * rate))
            if k <= 0:
                return x
            # Controls also apply to top-k: shuffle = the same k channels chosen
            # at random, reverse = the k lowest-scoring channels.
            if self.assign == 'shuffle':
                idx = torch.rand_like(scores).topk(k, dim=1).indices
            elif self.assign == 'reverse':
                idx = (-scores).topk(k, dim=1).indices
            else:
                idx = scores.topk(k, dim=1).indices               # (B, k)
            mask = torch.ones_like(scores).scatter_(1, idx, 0.0)
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            self._record(mask)
            if self.rescale:
                return x * mask * (C / max(C - k, 1))
            return x * mask
        drop_probs = _drop_probs_from_scores(scores, rate, self.norm, self.eps)
        if self.rescale and self.rescale_mode == 'prob':
            # norm='mean' gives p_c = clamp(p*C*s_c/sum(s), max=1), so once the score
            # distribution becomes skewed during training (exactly what the SDrop hypothesis predicts)
            # some channel's p_c approaches 1. When such a channel occasionally survives,
            # the 1/(1-p_c) rescale jumps to 1e6 and training collapses
            # (observed once: train accuracy fell from 91.2% to 1.3% mid-training).
            # Clip the probabilities before building the mask so mask and rescale always agree.
            drop_probs = drop_probs.clamp(max=1.0 - self.rescale_floor)
        if self.assign == 'shuffle':
            # Permute after probability computation and capping so the set of probabilities is preserved.
            perm = torch.rand_like(drop_probs).argsort(dim=1)
            drop_probs = drop_probs.gather(1, perm)
        elif self.assign == 'reverse':
            # Scatter ascending probabilities onto channels in descending score order -> highest score gets lowest probability.
            order = scores.argsort(dim=1, descending=True)
            asc = drop_probs.sort(dim=1).values
            drop_probs = torch.zeros_like(drop_probs).scatter_(1, order, asc)
        mask = _bernoulli_keep(drop_probs)
        self._record(mask)
        if self.rescale:
            # inverted dropout: match train/eval expectations. All compared
            # baselines (nn.Dropout2d, DropBlock) do this.
            if self.rescale_mode == 'count':
                # count-based (WCD style): correct by the number of channels actually kept.
                # Unlike the probability-based version it needs no clamp, so the budget does not
                # leak at high drop rates (prob mode is off by 0.12 in effective rate at p>=0.5).
                C = mask.shape[1]
                kept = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                return x * mask * (C / kept)
            # probability-based: divide by the per-channel keep probability. Exact for p<=0.3.
            keep_p = 1.0 - drop_probs          # guaranteed >= rescale_floor
            return x * mask / keep_p.unsqueeze(-1).unsqueeze(-1)
        return x * mask

    def extra_repr(self) -> str:
        s = (f"drop_rate={self.drop_rate}, score={self.score}, "
             f"gamma={self.gamma}, delta={self.delta}, "
             f"peakedness={self.peakedness}, norm={self.norm}")
        if self.grad_mode != 'off':
            s += f", grad_mode={self.grad_mode}, grad_ema={self.grad_ema}"
        if getattr(self, 'schedule', 'const') != 'const':
            s += f", schedule={self.schedule}"
        if getattr(self, 'participation', 1.0) < 1.0:
            s += f", participation={self.participation}"
        if getattr(self, 'assign', 'score') != 'score':
            s += f", assign={self.assign}"
        return s


class SDropEnergy(SDrop):
    """Energy-only variant s_c = E_c (kept for backward compatibility)."""
    def __init__(self, drop_rate: float = 0.1, gamma: float = 1.0,
                 delta: float = 1.0, **kw):
        super().__init__(drop_rate, score='energy', gamma=gamma, delta=delta, **kw)


# ---------------------------------------------------------------------------
# SGridLC  (spatially-aware)
# ---------------------------------------------------------------------------

class SDropClassAware(SDrop, DropRateMonitor):
    """Class-aware SDrop:  s_c = E_c * (1 - Sel_c).

    The spatial peakedness P_c in the original EGPG score is a *within-sample*
    statistic: it says how concentrated a channel is over space, and knows
    nothing about classes. The monopolisation hypothesis, however, is a
    statement about classes -- a few channels are recruited by many classes at
    once, leaving rare classes with nothing to claim. This variant measures
    that directly.

    A running (C, K) table accumulates how strongly each class k drives each
    channel c. Normalising a channel's row into a distribution pi_c over
    classes, its *class selectivity* is

        Sel_c = 1 - H(pi_c) / log K        in [0, 1]

    so Sel_c ~ 1 for a channel used by a single class (a rare class's lifeline,
    which must be protected) and Sel_c ~ 0 for a channel spread evenly over
    every class (a monopolist, which is what we want to drop). Pairing it with
    the same energy term keeps Proposition 1 intact: E_c is unchanged, and only
    the second factor is replaced.

    Labels are supplied per batch by the training loop via `set_targets`;
    with none provided the module falls back to the spatial score, so it stays
    usable in eval-only or label-free settings.
    """

    def __init__(self, drop_rate: float = 0.1, num_classes: int = 100,
                 usage_ema: float = 0.99, warmup_batches: int = 50, **kw):
        kw.setdefault('score', 'egpg')
        super().__init__(drop_rate=drop_rate, **kw)
        self.num_classes = num_classes
        self.usage_ema = usage_ema
        self.warmup_batches = warmup_batches
        self._usage = None            # (C, K) running class-channel usage
        self._seen = 0
        self._targets = None

    def set_targets(self, y: torch.Tensor):
        """Called by the training loop just before the forward pass."""
        self._targets = y

    @torch.no_grad()
    def _update_usage(self, x: torch.Tensor, y: torch.Tensor):
        # per-sample channel strength, then scatter-mean into the class rows
        a = x.detach().abs().mean(dim=(2, 3))                  # (B, C)
        C, K = a.shape[1], self.num_classes
        if self._usage is None or self._usage.shape != (C, K):
            self._usage = torch.zeros(C, K, device=a.device, dtype=a.dtype)
        batch = torch.zeros(C, K, device=a.device, dtype=a.dtype)
        count = torch.zeros(K, device=a.device, dtype=a.dtype)
        batch.index_add_(1, y, a.t())
        count.index_add_(0, y, torch.ones_like(y, dtype=a.dtype))
        present = count > 0
        batch[:, present] /= count[present]
        m = self.usage_ema
        self._usage = self._usage.to(a.device)
        self._usage[:, present] = (m * self._usage[:, present]
                                   + (1.0 - m) * batch[:, present])
        self._seen += 1

    def _selectivity(self, device, dtype) -> torch.Tensor:
        """Sel_c in [0, 1]; 1 = used by one class only."""
        u = self._usage.to(device=device, dtype=dtype).clamp_min(0)
        pi = u / (u.sum(dim=1, keepdim=True) + self.eps)        # (C, K)
        H = -(pi * (pi + self.eps).log()).sum(dim=1)            # (C,)
        return (1.0 - H / math.log(self.num_classes)).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x

        y = self._targets
        if y is not None:
            self._update_usage(x, y.to(x.device))

        energy = channel_energy(x, self.gamma, self.delta, self.eps)   # (B, C)

        if self._usage is None or self._seen < self.warmup_batches:
            # not enough class statistics yet -- fall back to the spatial score
            scores = energy * (1.0 - _peakedness(x, self.peakedness, self.eps))
        else:
            sel = self._selectivity(x.device, x.dtype).unsqueeze(0)    # (1, C)
            scores = energy * (1.0 - sel)

        if self.grad_mode != 'off':
            scores = self._apply_grad_guidance(scores)
            if x.requires_grad:
                x.register_hook(self._capture_grad)

        mask = drop_mask_from_scores(scores, self.drop_rate, self.norm, self.eps)
        self._record(mask)
        return x * mask

    def extra_repr(self) -> str:
        return (f"drop_rate={self.drop_rate}, score=class-aware, "
                f"num_classes={self.num_classes}, usage_ema={self.usage_ema}, "
                f"warmup_batches={self.warmup_batches}, norm={self.norm}")


class SGridLC(nn.Module, DropRateMonitor):
    """
    Spatially-Aware Local Feature Density Control.

    Partitions the feature map X in R^{C x H x W} into a G x G grid.
    For each cell (i,j): local E_c^{(i,j)} and P_c^{(i,j)} are computed;
    an independent drop mask is applied per cell, separating object and
    background treatment.

    Args:
        drop_rate  : base drop probability p_base
        grid_size  : G  (number of cells per side; total G^2 cells)
        score      : score variant, as in SDrop
        gamma      : LRN energy scale parameter
        delta      : LRN energy exponent parameter
        peakedness : 'max' | 'entropy'
        norm       : 'max' | 'mean'
        eps        : numerical stability constant

    Note:
        Low drop_rate with large G (e.g. rate=0.1, G=4) can cause training
        instability on early epochs — start with rate=0.3 or G=2 when
        fine-tuning on a new dataset.
    """
    def __init__(self, drop_rate: float = 0.1, grid_size: int = 2,
                 score: str = 'egpg', gamma: float = 1.0, delta: float = 1.0,
                 peakedness: str = 'max', norm: str = 'max', eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.grid_size = grid_size
        self.score = score
        self.gamma = gamma
        self.delta = delta
        self.peakedness = peakedness
        self.norm = norm
        self.eps = eps
        self.rescale = False
        self._init_monitor()
        self._init_schedule('const', 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x

        B, C, H, W = x.shape
        G = self.grid_size

        # pad so H, W are divisible by G
        pad_h = (G - H % G) % G
        pad_w = (G - W % G) % G
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape
        ch, cw = Hp // G, Wp // G

        mask = torch.ones_like(x)

        for i in range(G):
            for j in range(G):
                h0, h1 = i * ch, (i + 1) * ch
                w0, w1 = j * cw, (j + 1) * cw
                cell = x[:, :, h0:h1, w0:w1]              # (B, C, ch, cw)

                scores = compute_score(cell, self.score, self.gamma, self.delta,
                                       self.peakedness, self.eps)
                cell_mask = drop_mask_from_scores(scores, self.drop_rate,
                                                  self.norm, self.eps)
                mask[:, :, h0:h1, w0:w1] = cell_mask

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            mask = mask[:, :, :H, :W]

        self._record(mask)
        return x * mask

    def extra_repr(self) -> str:
        return (f"drop_rate={self.drop_rate}, grid_size={self.grid_size}, "
                f"score={self.score}, gamma={self.gamma}, delta={self.delta}, "
                f"peakedness={self.peakedness}, norm={self.norm}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCORE_OF_METHOD = {
    'sdrop':        'egpg',
    'sdrop_energy': 'energy',
    'sdrop_peak':   'peak',     # ablation: peakedness only
    'sdrop_random': 'random',   # ablation: uniform score control
    'sdrop_gap':    'gap',      # ablation: pure GAP score (channel version of WCD/ProbDropBlock)
    'sdrop_shuffle': 'egpg',    # control: permute the EGPG probability set along the channel axis
    'sdrop_reverse': 'egpg',    # control: assign EGPG probabilities in reverse score order
    'sdrop_cross':   'energy',  # control: roll energy probability vectors along the batch axis (receive another sample's)
}
_ASSIGN_OF_METHOD = {
    'sdrop_shuffle': 'shuffle',
    'sdrop_reverse': 'reverse',
    'sdrop_cross':   'cross',
}


def build_sdrop(method: str, drop_rate: float = 0.1, grid_size: int = 2,
                num_classes: int = 100,
                gamma: float = 1.0, delta: float = 1.0,
                peakedness: str = 'max', norm: str = 'max',
                grad_mode: str = 'off',
                schedule: str = 'const', participation: float = 1.0,
                rescale: bool = False, rescale_mode: str = 'prob',
                select: str = 'bernoulli', warmup_frac: float = 0.0) -> nn.Module:
    """
    method: 'none' | 'dropout' | 'sdrop' | 'sdrop_energy'
          | 'sdrop_peak' | 'sdrop_random' | 'sdrop_class' | 'sgridlc'
    """
    if method == 'none':
        return nn.Identity()
    if method == 'dropout':
        # NOTE: nn.Dropout2d zeroes whole channels -> this is SpatialDropout
        # (Tompson et al., 2015), not element-wise dropout. Kept under the
        # historical name for reproducibility of previously reported runs.
        return nn.Dropout2d(p=drop_rate)
    if method in ('spatialdropout',):
        return nn.Dropout2d(p=drop_rate)
    if method == 'dropout_std':
        # element-wise dropout (Srivastava et al., 2014)
        return nn.Dropout(p=drop_rate)
    if method == 'sdrop_class':
        return SDropClassAware(drop_rate, num_classes=num_classes, gamma=gamma,
                               delta=delta,
                               peakedness=peakedness, norm=norm,
                               grad_mode=grad_mode,
                               schedule=schedule, participation=participation,
                               rescale=rescale)
    if method in _SCORE_OF_METHOD:
        return SDrop(drop_rate, score=_SCORE_OF_METHOD[method], gamma=gamma,
                     delta=delta, peakedness=peakedness, norm=norm,
                     grad_mode=grad_mode,
                     schedule=schedule, participation=participation,
                     rescale=rescale, rescale_mode=rescale_mode,
                     select=select, warmup_frac=warmup_frac,
                     assign=_ASSIGN_OF_METHOD.get(method, 'score'))
    if method == 'sgridlc':
        if (schedule != 'const' or participation < 1.0 or rescale
                or select != 'bernoulli' or warmup_frac > 0.0):
            # Silently ignoring these would let run_id append tags, creating a phantom ablation.
            raise ValueError("sgridlc does not yet support schedule/participation/rescale/select/"
                             "warmup.")
        return SGridLC(drop_rate, grid_size, score='egpg', gamma=gamma,
                       delta=delta, peakedness=peakedness, norm=norm)
    raise ValueError(f"Unknown SDrop method: '{method}'. Choose from: none, dropout "
                     "(=SpatialDropout), dropout_std, spatialdropout, sdrop, "
                     "sdrop_energy, sdrop_peak, sdrop_random, sgridlc")


# ---------------------------------------------------------------------------
# Effective-drop-rate reporting
# ---------------------------------------------------------------------------

def collect_drop_rates(model: nn.Module, reset: bool = True) -> dict:
    """
    Walk a model and return {module_name: measured mean drop fraction} for every
    SDrop/SGridLC layer, then optionally reset the accumulators.

    Use once per epoch in the training loop, e.g.

        rates = collect_drop_rates(model)
        if rates:
            print("  effective drop rate: " +
                  ", ".join(f"{k}={v:.4f}" for k, v in rates.items()))

    Report the resulting p_bar next to the nominal p_base: under
    'max' normalisation p_bar < p_base, so nominal-rate comparisons against
    unstructured dropout are not like-for-like.
    """
    rates = {}
    for name, m in model.named_modules():
        if isinstance(m, (SDrop, SGridLC)):
            r = m.effective_drop_rate()
            if r == r:                       # skip NaN (module never active)
                rates[name or type(m).__name__] = r
            if reset:
                m.reset_monitor()
    return rates
