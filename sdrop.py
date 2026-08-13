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
    if H * W == 1:
        # 공간 범위가 없으면 "퍼짐"이 정의되지 않는다. log(1)=0 으로 나누면
        # NaN 이 나므로 0(=완전 확산)으로 두어 점수를 에너지 단독으로 축퇴시킨다.
        # SGridLC 에서 grid_size 가 특징맵 한 변과 같을 때 이 경로를 탄다.
        return torch.zeros(B, C, device=x.device, dtype=x.dtype)
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


def _unit_rank(v: torch.Tensor) -> torch.Tensor:
    """(B, C) -> 채널 순위를 [0, 1] 로 정규화. 값의 척도를 지우고 순서만 남긴다."""
    C = v.shape[1]
    if C < 2:
        return torch.zeros_like(v)
    return v.argsort(dim=1).argsort(dim=1).to(v.dtype) / (C - 1)


def compute_score(x: torch.Tensor, score: str = 'egpg', gamma: float = 1.0,
                  delta: float = 1.0, peakedness: str = 'max',
                  eps: float = 1e-6, mix: float = None) -> torch.Tensor:
    """
    Unified score front-end. Returns (B, C).

      'egpg'    : s_c = E_c * (1 - P_c)     full score
      'energy'  : s_c = E_c                 energy only
      'peak'    : s_c = (1 - P_c)           peakedness only   [ablation]
      'random'  : s_c = 1                   uniform            [ablation control]

    mix (egpg only) replaces the product with a weighted sum of the two factors'
    channel ranks:

        s_c = (1 - mix) * rank(E_c) + mix * rank(1 - P_c)

    The product weights whichever factor happens to vary more. Measured on a
    trained CIFAR-100-LT model at L4, energy carries 91% of the log-variance of
    s_c under the 'max' peakedness, and the Spearman correlation between the
    EGPG score and energy alone is 0.998 -- the peakedness half barely moves the
    ordering, which is why SDrop and SDropEnergy score alike. Switching to
    'entropy' peakedness rebalances this to 46/54, but by accident of the two
    distributions rather than by design, and the balance shifts with layer,
    dataset and training stage.

    Ranking each factor first makes both uniform on [0, 1], so mix sets the
    balance explicitly: 0 recovers energy-only, 1 recovers peakedness-only, and
    the intermediate values are a continuum the original product cannot express.
    """
    if score == 'random':
        return torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
    if score == 'energy':
        return channel_energy(x, gamma, delta, eps)
    if score == 'peak':
        return 1.0 - _peakedness(x, peakedness, eps)
    if score == 'egpg':
        E = channel_energy(x, gamma, delta, eps)
        D = 1.0 - _peakedness(x, peakedness, eps)
        if mix is None:
            return E * D
        return (1.0 - mix) * _unit_rank(E) + mix * _unit_rank(D)
    raise ValueError(f"Unknown score: '{score}'. "
                     "Choose from: egpg, energy, peak, random")


def drop_mask_from_scores(scores: torch.Tensor, drop_rate: float,
                          norm: str = 'max', eps: float = 1e-6,
                          beta: float = 1.0) -> torch.Tensor:
    """
    norm='max'  : p_drop,c = drop_rate * s_c / max_c(s_c)          (original)
    norm='mean' : p_drop,c = min(1, drop_rate * C * s_c / sum_c s_c)
    norm='rank' : p_drop,c = drop_rate * (beta+1) * r_c^beta,
                  r_c = rank(s_c)/(C-1) in [0, 1]

    With 'max' normalisation only the top-scoring channel is dropped at exactly
    drop_rate, so the expected dropped fraction is
        p_bar = drop_rate * mean_c(s_c) / max_c(s_c) <= drop_rate.
    Measured on CIFAR-100-LT this is 0.017 against a nominal 0.1, and it drifts
    during training as the score distribution sharpens, which makes comparisons
    at equal nominal rate meaningless. 'mean' fixes the expectation but pushes
    the top channel to drop_rate * max(s)/mean(s) -- 0.59 at the same setting,
    inside the destructive regime.

    'rank' fixes both. Ranks are uniform on [0, 1], so the expectation is
    drop_rate * (beta+1) * INT_0^1 r^beta dr = drop_rate for every beta, and the
    top channel is bounded by (beta+1) * drop_rate. beta = 0 gives a uniform
    drop_rate for all channels, i.e. plain channel dropout, which makes the
    random control a special case of the method rather than a separate baseline.
    Only the ordering of s_c is used, so the rule is invariant to any monotone
    rescaling of the score.

    mask[b, c] = Bernoulli(1 - p_drop,c)   (1 = keep, 0 = drop)

    Args:
        scores: (B, C)
        beta:   selection strength, 'rank' only. 0 = uniform, higher = more
                concentrated on the top-scoring channels.
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
    elif norm == 'rank':
        C = scores.shape[1]
        if C < 2:
            drop_probs = torch.full_like(scores, drop_rate)
        else:
            r = scores.argsort(dim=1).argsort(dim=1).to(scores.dtype) / (C - 1)
            drop_probs = (drop_rate * (beta + 1.0) * r.pow(beta)).clamp(max=1.0)
    else:
        raise ValueError(f"Unknown norm: '{norm}'. Choose 'max', 'mean' or 'rank'.")
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
                 peakedness: str = 'max', norm: str = 'max', beta: float = 1.0,
                 mix: float = None,
                 grad_mode: str = 'off', grad_ema: float = 0.9,
                 eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.score = score
        self.gamma = gamma
        self.delta = delta
        self.peakedness = peakedness
        self.norm = norm
        self.beta = beta
        self.mix = mix
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
                               self.peakedness, self.eps, self.mix)   # (B, C)
        if self.grad_mode != 'off':
            scores = self._apply_grad_guidance(scores)
            if x.requires_grad:
                x.register_hook(self._capture_grad)
        mask = drop_mask_from_scores(scores, self.drop_rate, self.norm, self.eps,
                                    self.beta)
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
                 peakedness: str = 'max', norm: str = 'max', beta: float = 1.0,
                 mix: float = None, eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.grid_size = grid_size
        self.score = score
        self.gamma = gamma
        self.delta = delta
        self.peakedness = peakedness
        self.norm = norm
        self.beta = beta
        self.mix = mix
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
                                       self.peakedness, self.eps, self.mix)
                cell_mask = drop_mask_from_scores(scores, self.drop_rate,
                                                  self.norm, self.eps, self.beta)
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
                gamma: float = 1.0, delta: float = 1.0,
                peakedness: str = 'max', norm: str = 'max', beta: float = 1.0,
                mix: float = None, grad_mode: str = 'off') -> nn.Module:
    """
    method: 'none' | 'dropout' | 'sdrop' | 'sdrop_energy'
          | 'sdrop_peak' | 'sdrop_random' | 'sgridlc'
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
    if method in _SCORE_OF_METHOD:
        return SDrop(drop_rate, score=_SCORE_OF_METHOD[method], gamma=gamma,
                     delta=delta, peakedness=peakedness, norm=norm, beta=beta, mix=mix,
                     grad_mode=grad_mode)
    if method == 'sgridlc':
        return SGridLC(drop_rate, grid_size, score='egpg', gamma=gamma,
                       delta=delta, peakedness=peakedness, norm=norm, beta=beta, mix=mix)
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
