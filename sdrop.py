"""
Suppressive Dropout (SDrop)
----------------------------
Score variants implemented from the paper:

  SDrop        : full EGPG score   s_c = E_c * (1 - P_c)
  SDropEnergy  : energy only       s_c = E_c
  SDropPeak    : peakedness only   s_c = (1 - P_c)          [ablation]
  SDropRandom  : uniform score     s_c = 1                  [ablation control]
  SGridLC      : EGPG computed per G x G spatial grid cell (spatially-aware)

All variants:
  - During training : stochastically drop channels proportional to their score
  - During eval     : pass-through (no dropout, no rescaling needed)

Two score refinements introduced with the Springer ML extension, both
off by default so that previously reported numbers remain reproducible:

  peakedness='entropy'  : resolution-invariant peakedness (Eq. 4b)
  norm='mean'           : mean-preserving drop probability (Eq. 6b)
  gamma=None            : self-normalising energy scale (Eq. 3b)

Reference:
  Lee, S. & Longo, L. "Suppressive Dropout: A Bio-Inspired and Explainable
  Channel-Selective Regularization Framework." (Springer Machine Learning,
  in preparation, 2026). Short version: Neurocomputing, under review.
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
    """
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
                     "Choose from: egpg, energy, peak, random")


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
    if norm == 'max':
        denom = scores.amax(dim=1, keepdim=True).clamp(min=eps)
        drop_probs = drop_rate * scores / denom
    elif norm == 'mean':
        C = scores.shape[1]
        denom = scores.sum(dim=1, keepdim=True).clamp(min=eps)
        drop_probs = (drop_rate * C * scores / denom).clamp(max=1.0)
    else:
        raise ValueError(f"Unknown norm: '{norm}'. Choose 'max' or 'mean'.")
    keep = (torch.rand_like(drop_probs) > drop_probs).float()
    return keep.unsqueeze(-1).unsqueeze(-1)           # (B, C, 1, 1)


# ---------------------------------------------------------------------------
# SDrop  (single module covering every score variant)
# ---------------------------------------------------------------------------

class DropRateMonitor:
    """
    Mixin that accumulates the *realised* fraction of dropped units.

    p_base is only an upper bound on the average drop rate under 'max'
    normalisation (Eq. 7 of the paper), so the nominal rate is not directly
    comparable against unstructured dropout at the same nominal p. This
    monitor records what actually happened, at negligible cost, so every run
    can report its measured effective rate p_bar alongside p_base.
    """
    def _init_monitor(self):
        self._drop_sum = 0.0
        self._drop_batches = 0
        self.last_drop_frac = float('nan')

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
                 eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.score = score
        self.gamma = gamma
        self.delta = delta
        self.peakedness = peakedness
        self.norm = norm
        self.grad_mode = grad_mode
        self.grad_ema = grad_ema
        self.eps = eps
        self._grad_mag = None          # (C,) EMA of |dL/da_c|, normalised to mean 1
        self._init_monitor()

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
        scores = compute_score(x, self.score, self.gamma, self.delta,
                               self.peakedness, self.eps)          # (B, C)
        if self.grad_mode != 'off':
            scores = self._apply_grad_guidance(scores)
            if x.requires_grad:
                x.register_hook(self._capture_grad)
        mask = drop_mask_from_scores(scores, self.drop_rate, self.norm, self.eps)
        self._record(mask)
        return x * mask

    def extra_repr(self) -> str:
        s = (f"drop_rate={self.drop_rate}, score={self.score}, "
             f"gamma={self.gamma}, delta={self.delta}, "
             f"peakedness={self.peakedness}, norm={self.norm}")
        if self.grad_mode != 'off':
            s += f", grad_mode={self.grad_mode}, grad_ema={self.grad_ema}"
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
        self._init_monitor()

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
}


def build_sdrop(method: str, drop_rate: float = 0.1, grid_size: int = 2,
                num_classes: int = 100,
                gamma: float = 1.0, delta: float = 1.0,
                peakedness: str = 'max', norm: str = 'max',
                grad_mode: str = 'off') -> nn.Module:
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
                               peakedness=peakedness, norm=norm,
                               grad_mode=grad_mode)
    if method in _SCORE_OF_METHOD:
        return SDrop(drop_rate, score=_SCORE_OF_METHOD[method], gamma=gamma,
                     delta=delta, peakedness=peakedness, norm=norm,
                     grad_mode=grad_mode)
    if method == 'sgridlc':
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

    Report the resulting p_bar next to the nominal p_base in the paper: under
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
