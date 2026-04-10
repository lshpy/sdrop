"""
Suppressive Dropout (SDrop)
----------------------------
Three variants implemented from the paper:

  SDrop        : EGPG score  s_c = E_c * (1 - P_c)
  SDropEnergy  : Energy only  s_c = E_c
  SGridLC      : EGPG computed per G×G spatial grid cell (spatially-aware)

All variants:
  - During training : stochastically drop channels proportional to their score
  - During eval     : pass-through (no dropout, no rescaling needed)

Reference:
  Lee, S. "Suppressive Dropout: An Explainable Channel-Selective Regularization
  Method for Preserving Rare Features." Neurocomputing, 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Score primitives
# ---------------------------------------------------------------------------

def channel_energy(x: torch.Tensor, gamma: float = 1.0, delta: float = 1.0) -> torch.Tensor:
    """
    E_c = 1 - (1 + gamma * (1/HW) * sum_hw x_{c,h,w}^2)^{-delta}

    Args:
        x: (B, C, H, W)
    Returns:
        energy: (B, C)  in (0, 1)
    """
    mean_sq = x.pow(2).mean(dim=(2, 3))          # (B, C)
    return 1.0 - (1.0 + gamma * mean_sq).pow(-delta)


def spatial_peakedness(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    P_c = max_{h,w} |x_{c,h,w}| / (sum_{h,w} |x_{c,h,w}| + eps)

    High value  → activation is spatially localized  (informative, spare)
    Low value   → activation is diffuse              (over-inhibitory)

    Args:
        x: (B, C, H, W)
    Returns:
        peakedness: (B, C)  in (0, 1]
    """
    abs_x = x.abs()
    peak = abs_x.amax(dim=(2, 3))                # (B, C)
    total = abs_x.sum(dim=(2, 3))                # (B, C)
    return peak / (total + eps)


def egpg_score(x: torch.Tensor, gamma: float = 1.0, delta: float = 1.0,
               eps: float = 1e-6) -> torch.Tensor:
    """
    s_c = E_c * (1 - P_c)

    High score  → high energy AND diffuse spread → over-inhibitory channel
    """
    E = channel_energy(x, gamma, delta)
    P = spatial_peakedness(x, eps)
    return E * (1.0 - P)


def drop_mask_from_scores(scores: torch.Tensor, drop_rate: float,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    p_drop,c = drop_rate * s_c / max_c(s_c)
    mask[b, c] = Bernoulli(1 - p_drop,c)   (1 = keep, 0 = drop)

    Args:
        scores: (B, C)
    Returns:
        mask: (B, C, 1, 1)  float
    """
    max_s = scores.amax(dim=1, keepdim=True).clamp(min=eps)
    drop_probs = drop_rate * scores / max_s          # (B, C)
    keep = (torch.rand_like(drop_probs) > drop_probs).float()
    return keep.unsqueeze(-1).unsqueeze(-1)           # (B, C, 1, 1)


# ---------------------------------------------------------------------------
# SDrop  (EGPG variant)
# ---------------------------------------------------------------------------

class SDrop(nn.Module):
    """
    Suppressive Dropout using the full EGPG score s_c = E_c * (1 - P_c).

    Args:
        drop_rate : base drop probability p_base
        gamma     : LRN energy scale parameter
        delta     : LRN energy exponent parameter
        eps       : numerical stability constant
    """
    def __init__(self, drop_rate: float = 0.1, gamma: float = 1.0,
                 delta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.gamma = gamma
        self.delta = delta
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        scores = egpg_score(x, self.gamma, self.delta, self.eps)   # (B, C)
        mask = drop_mask_from_scores(scores, self.drop_rate, self.eps)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, gamma={self.gamma}, delta={self.delta}"


# ---------------------------------------------------------------------------
# SDropEnergy  (energy-only variant — best on CIFAR-100)
# ---------------------------------------------------------------------------

class SDropEnergy(nn.Module):
    """
    Suppressive Dropout using only Channel Energy  s_c = E_c.

    Omits the peakedness factor; simpler and empirically the strongest
    single-dataset variant (+0.77%p on CIFAR-100 vs baseline).

    Args:
        drop_rate : base drop probability p_base
        gamma     : LRN energy scale parameter
        delta     : LRN energy exponent parameter
        eps       : numerical stability constant
    """
    def __init__(self, drop_rate: float = 0.1, gamma: float = 1.0,
                 delta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.gamma = gamma
        self.delta = delta
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        scores = channel_energy(x, self.gamma, self.delta)        # (B, C)
        mask = drop_mask_from_scores(scores, self.drop_rate, self.eps)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, gamma={self.gamma}, delta={self.delta}"


# ---------------------------------------------------------------------------
# SGridLC  (spatially-aware — best on TinyImageNet)
# ---------------------------------------------------------------------------

class SGridLC(nn.Module):
    """
    Spatially-Aware Local Feature Density Control.

    Partitions the feature map X ∈ R^{C×H×W} into a G×G grid.
    For each cell (i,j): local E_c^{(i,j)} and P_c^{(i,j)} are computed;
    an independent drop mask is applied per cell, separating object and
    background treatment.

    Args:
        drop_rate  : base drop probability p_base
        grid_size  : G  (number of cells per side; total G² cells)
        gamma      : LRN energy scale parameter
        delta      : LRN energy exponent parameter
        eps        : numerical stability constant

    Note:
        Low drop_rate with large G (e.g. rate=0.1, G=4) can cause training
        instability on early epochs — start with rate=0.3 or G=2 when
        fine-tuning on a new dataset.
    """
    def __init__(self, drop_rate: float = 0.1, grid_size: int = 2,
                 gamma: float = 1.0, delta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.drop_rate = drop_rate
        self.grid_size = grid_size
        self.gamma = gamma
        self.delta = delta
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
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

                scores = egpg_score(cell, self.gamma, self.delta, self.eps)
                cell_mask = drop_mask_from_scores(scores, self.drop_rate, self.eps)
                mask[:, :, h0:h1, w0:w1] = cell_mask

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            mask = mask[:, :, :H, :W]

        return x * mask

    def extra_repr(self) -> str:
        return (f"drop_rate={self.drop_rate}, grid_size={self.grid_size}, "
                f"gamma={self.gamma}, delta={self.delta}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_sdrop(method: str, drop_rate: float = 0.1, grid_size: int = 2,
                gamma: float = 1.0, delta: float = 1.0) -> nn.Module:
    """
    method: 'none' | 'dropout' | 'sdrop' | 'sdrop_energy' | 'sgridlc'
    """
    if method == 'none':
        return nn.Identity()
    if method == 'dropout':
        return nn.Dropout2d(p=drop_rate)
    if method == 'sdrop':
        return SDrop(drop_rate, gamma, delta)
    if method == 'sdrop_energy':
        return SDropEnergy(drop_rate, gamma, delta)
    if method == 'sgridlc':
        return SGridLC(drop_rate, grid_size, gamma, delta)
    raise ValueError(f"Unknown SDrop method: '{method}'. "
                     "Choose from: none, dropout, sdrop, sdrop_energy, sgridlc")
