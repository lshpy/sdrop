"""
Comparison baselines for SDrop.

Three modules implementing well-known regularization / channel-attention
methods, designed to be drop-in replacements for the SDrop module
(applied after a residual stage's output, before the next stage).

  - DropBlock2D  (Ghiasi et al., NeurIPS 2018)
  - SEBlock      (Hu et al., CVPR 2018)
  - CBAM         (Woo et al., ECCV 2018)

All three are deliberately written to take a (B, C, H, W) tensor and
return a (B, C, H, W) tensor, so they can be plugged into the same
"L3 / L4 hook point" used by SDrop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DropBlock2D — Ghiasi, Lin, Le, NeurIPS 2018
# ---------------------------------------------------------------------------

class DropBlock2D(nn.Module):
    """
    Drop contiguous square regions of size `block_size` x `block_size`
    from each feature map. At eval time it is a no-op.

    Args:
        drop_prob   : probability of dropping a block (paper notation: gamma is
                      derived from this and the spatial size).
        block_size  : side length of the dropped region.
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 3):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        B, C, H, W = x.shape
        # gamma normalisation as in the DropBlock paper:
        #   gamma = drop_prob / block_size^2 * H*W / ((H - block_size + 1)*(W - block_size + 1))
        denom = max(1, (H - self.block_size + 1) * (W - self.block_size + 1))
        gamma = self.drop_prob * H * W / (self.block_size ** 2 * denom)
        # sample bernoulli mask at "centre" positions, then expand via maxpool
        mask = (torch.rand(B, C, H, W, device=x.device) < gamma).float()
        # zero out borders so blocks don't extend past the feature map
        if self.block_size > 1:
            pad = self.block_size // 2
            mask[:, :, :pad, :] = 0
            mask[:, :, -pad:, :] = 0
            mask[:, :, :, :pad] = 0
            mask[:, :, :, -pad:] = 0
        # expand each centre into a block_size x block_size square via maxpool
        block_mask = F.max_pool2d(mask, kernel_size=self.block_size,
                                  stride=1, padding=self.block_size // 2)
        block_mask = 1.0 - block_mask
        # rescale to keep expectation
        keep = block_mask.numel() / (block_mask.sum() + 1e-8)
        return x * block_mask * keep


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation — Hu, Shen, Sun, CVPR 2018
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Channel attention: global average pool → 2-layer MLP → sigmoid → multiply.

    Args:
        channels  : number of input channels (must be passed at runtime).
        reduction : bottleneck ratio (paper default 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        s = x.mean(dim=(2, 3))                  # (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(B, C, 1, 1)


class _LazySEWrapper(nn.Module):
    """SE block whose channel size is inferred on the first forward pass.
    Lets the build_model factory stay channel-agnostic."""

    def __init__(self, reduction: int = 16):
        super().__init__()
        self.reduction = reduction
        self.block: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block is None:
            self.block = SEBlock(channels=x.shape[1],
                                 reduction=self.reduction).to(x.device)
        return self.block(x)


# ---------------------------------------------------------------------------
# CBAM — Woo, Park, Lee, Kweon, ECCV 2018
# ---------------------------------------------------------------------------

class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        a = self.fc2(F.relu(self.fc1(avg), inplace=True))
        m = self.fc2(F.relu(self.fc1(mx),  inplace=True))
        att = torch.sigmoid(a + m).view(B, C, 1, 1)
        return x * att


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention. (Woo et al., 2018)"""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel = _ChannelAttention(channels, reduction)
        self.spatial = _SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class _LazyCBAMWrapper(nn.Module):
    """CBAM whose channel count is inferred on the first forward pass."""

    def __init__(self, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.reduction = reduction
        self.spatial_kernel = spatial_kernel
        self.block: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block is None:
            self.block = CBAM(channels=x.shape[1],
                              reduction=self.reduction,
                              spatial_kernel=self.spatial_kernel).to(x.device)
        return self.block(x)


# ---------------------------------------------------------------------------
# Factory used by build_model
# ---------------------------------------------------------------------------

class GuidedDropout(nn.Module):
    """
    Guided Dropout (Keshari, Singh & Vatsa, AAAI 2019; arXiv:1812.03965).

    Each channel (feature map) is multiplied by a **learned strength t_c**; during training
    channels with large t are dropped so that channels with small t can grow.
        y_c = t_c * x_c
        'topk' : deterministically drop the k=round(C*p) channels with the largest t (GD-top-k)
        'dr'   : randomly drop k channels inside the active region (top active_frac of t) (GD-DR)
    t is a global learned parameter: it does not vary per sample and ignores spatial structure.
    These are exactly the two points where it differs from SDrop.

    Original schedule (200 epochs): 40 epochs of strength learning (no drop), then alternating drop / no drop.
    The 'gdtopk' profile of sdrop.DropRateMonitor reproduces that alternation as fractions.
    Default schedule='gdtopk'. With 'const' it is applied continuously at the same budget as SDrop.

    SDrop with `--select topk --schedule gdtopk` is not this reproduction, since its score is EGPG.
    """
    def __init__(self, channels: int, drop_rate: float = 0.1, mode: str = 'topk',
                 schedule: str = 'gdtopk', active_frac: float = 0.5,
                 warmup_frac: float = 0.0, participation: float = 1.0):
        super().__init__()
        assert mode in ('topk', 'dr'), mode
        self.drop_rate = float(drop_rate)
        self.mode = mode
        self.active_frac = float(active_frac)
        # Initialised to 1 so that attaching it to a pretrained backbone does not change the initial forward pass.
        self.t = nn.Parameter(torch.ones(channels))
        from sdrop import DropRateMonitor
        self._mon = DropRateMonitor()
        self._mon._init_monitor()
        self._mon._init_schedule(schedule, participation, warmup_frac)
        self._mon.drop_rate = self.drop_rate
        self.schedule = schedule
        self.last_drop_frac = float('nan')

    def set_progress(self, progress: float):
        self._mon.set_progress(progress)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        y = x * self.t.view(1, C, 1, 1)
        if not self.training or self.drop_rate <= 0.0:
            return y
        rate = self._mon._current_rate()
        k = int(round(C * rate))
        if rate <= 0.0 or k <= 0:
            self.last_drop_frac = 0.0
            return y
        t = self.t.detach()
        if self.mode == 'topk':
            idx = t.topk(k).indices                                   # (k,)
            mask = torch.ones(C, device=x.device, dtype=y.dtype)
            mask[idx] = 0.0
            mask = mask.view(1, C, 1, 1).expand(x.shape[0], C, 1, 1)
        else:
            n_act = max(k, int(round(C * self.active_frac)))
            act = t.topk(n_act).indices                                # active region
            # per sample, choose k at random inside the active region
            r = torch.rand(x.shape[0], n_act, device=x.device)
            pick = r.topk(k, dim=1).indices                            # (B, k)
            chosen = act[pick]                                         # (B, k)
            mask = torch.ones(x.shape[0], C, device=x.device, dtype=y.dtype)
            mask.scatter_(1, chosen, 0.0)
            mask = mask.view(x.shape[0], C, 1, 1)
        self.last_drop_frac = float(1.0 - mask.mean())
        return y * mask

    def extra_repr(self) -> str:
        return (f"C={self.t.numel()}, drop_rate={self.drop_rate}, mode={self.mode}, "
                f"schedule={self.schedule}, active_frac={self.active_frac}")


def build_baseline(method: str, drop_rate: float = 0.1,
                   block_size: int = 3, reduction: int = 16,
                   channels: int = None, participation: float = 0.1,
                   schedule: str = 'const', warmup_frac: float = 0.0) -> nn.Module:
    """
    Build one of {'dropblock', 'senet', 'cbam', 'wcd', 'focuseddrop',
    'guided_topk', 'guided_dr'} as a single drop-in module that takes /
    returns a (B, C, H, W) tensor.
    """
    if method in ('guided_topk', 'guided_dr'):
        if channels is None:
            raise ValueError("guided dropout requires the channels argument (strength t).")
        return GuidedDropout(channels, drop_rate=drop_rate,
                             mode='topk' if method == 'guided_topk' else 'dr',
                             schedule=schedule, warmup_frac=warmup_frac)
    if method == 'dropblock':
        return DropBlock2D(drop_prob=drop_rate, block_size=block_size)
    # Lazy wrappers create their parameters on the first forward, but the optimizer is
    # built before that -> SE/CBAM weights would stay at their random init throughout training.
    # Create them immediately when the channel count is known; otherwise fall back to lazy with a warning.
    if method == 'senet':
        if channels is None:
            raise ValueError("senet requires the channels argument: with lazy creation "
                             "the optimizer does not see the parameters and they are never trained.")
        return SEBlock(channels, reduction=reduction)
    if method == 'cbam':
        if channels is None:
            raise ValueError("cbam requires the channels argument: with lazy creation "
                             "the optimizer does not see the parameters and they are never trained.")
        return CBAM(channels, reduction=reduction)
    if method == 'wcd':
        return WeightedChannelDropout(drop_rate=drop_rate)
    if method == 'focuseddrop':
        return FocusedDropout(drop_rate=drop_rate, participation=participation)
    raise ValueError(f"unknown baseline method '{method}'")



# ---------------------------------------------------------------------------
# Weighted Channel Dropout  (Hou & Wang, AAAI 2019)
# ---------------------------------------------------------------------------

class WeightedChannelDropout(nn.Module):
    """
    The closest prior work to SDrop. Channels are ranked by their mean activation (GAP)
    and *kept* with probability proportional to it (channels with larger activation survive).

        score_c = GAP(x)_c
        p_keep,c = score_c / max_c(score_c)

    Differences from SDrop:
      1) Opposite direction. WCD *keeps* strong channels, SDrop *drops* them.
      2) WCD's signal is the mean activation itself, while SDrop applies the saturating nonlinearity
         E_c = 1-(1+gamma*Sigma_c)^-delta and multiplies by the spatial concentration term (1-P_c).

    Args:
        drop_rate: overall strength. p_keep is scaled down by (1-drop_rate) so that the
                   effective drop rate matches the nominal one.
    """

    def __init__(self, drop_rate: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.drop_rate = float(drop_rate)
        self.eps = eps
        self.last_drop_frac = float('nan')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x
        score = x.mean(dim=(2, 3))                                   # (B, C)  GAP
        score = score - score.amin(dim=1, keepdim=True)              # correct for negative activations
        rng = score.amax(dim=1, keepdim=True)
        # If all channels have the same GAP (dead layer, constant input), score is all 0,
        # giving p_keep=0 -> every channel dropped (effective rate 1.0). That silently triples
        # the strength, so fall back to uniform Bernoulli in that case.
        degenerate = (rng <= self.eps)
        denom = rng.clamp_min(self.eps)
        p_keep = torch.where(degenerate,
                             torch.full_like(score, 1.0 - self.drop_rate),
                             score / denom)                          # (B, C) in [0, 1]
        # Scale so that the mean keep rate is (1 - drop_rate).
        # Simply multiplying and clamping to (0,1) cuts off mass above 1 and makes the mean lower
        # than the target (nominal 0.1 -> effective 0.20). Solve for the factor by bisection so that
        # the mean after clamping hits the target.
        target = 1.0 - self.drop_rate
        lo = torch.zeros(score.shape[0], 1, device=x.device, dtype=score.dtype)
        hi = torch.full_like(lo, 1.0)
        cur = p_keep.mean(dim=1, keepdim=True).clamp_min(self.eps)
        hi = torch.maximum(hi, target / cur * 2.0)
        for _ in range(30):                       # 30 iterations converge to double precision
            mid = 0.5 * (lo + hi)
            m = (p_keep * mid).clamp(0.0, 1.0).mean(dim=1, keepdim=True)
            too_small = (m < target)
            lo = torch.where(too_small, mid, lo)
            hi = torch.where(too_small, hi, mid)
        scaled = (p_keep * (0.5 * (lo + hi))).clamp(0.0, 1.0)
        # Samples that cannot reach the target keep rate (most channels dead) fall back to uniform.
        reached = scaled.mean(dim=1, keepdim=True) >= target - 1e-3
        p_keep = torch.where(reached | degenerate, scaled,
                             torch.full_like(scaled, target))
        with torch.no_grad():           # do not hold on to the score computation graph
            mask = torch.bernoulli(p_keep.detach()).unsqueeze(-1).unsqueeze(-1)
        self.last_drop_frac = float((1.0 - mask).mean())
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}"


# ---------------------------------------------------------------------------
# FocusedDropout  (Liu et al., 2021)
# ---------------------------------------------------------------------------

class FocusedDropout(nn.Module):
    """
    A control in the opposite direction to SDrop. The channel with the largest activation is the
    'reference channel'; only the *spatial positions* with high activation in it are kept, and the
    other positions are discarded in all channels ("focus on the target").

    Key device of the original paper: **apply it to only part of the batches** (participation, default 10%).
    An orthogonal axis that controls "how often" rather than "how hard".

    Args:
        drop_rate     : keep the top (1-drop_rate) fraction of positions of the reference channel
        participation : apply to a batch only with this probability (0.1 in the original paper)
    """

    def __init__(self, drop_rate: float = 0.1, participation: float = 0.1):
        super().__init__()
        self.drop_rate = float(drop_rate)
        self.participation = float(participation)
        self.last_drop_frac = float('nan')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x
        if float(torch.rand(())) >= self.participation:
            self.last_drop_frac = 0.0
            return x
        B, C, H, W = x.shape
        # reference channel = the channel with the largest per-channel max activation
        ref_idx = x.amax(dim=(2, 3)).argmax(dim=1)                   # (B,)
        ref = x[torch.arange(B, device=x.device), ref_idx]           # (B, H, W)
        flat = ref.reshape(B, -1)
        k = max(1, int(round(H * W * (1.0 - self.drop_rate))))
        # A threshold comparison (`>=`) fails on post-ReLU input: 50-80% of the values are exactly 0,
        # so thresh==0, every position passes and it becomes a complete no-op.
        # Build the mask directly from topk indices so exactly k positions are always kept.
        idx = flat.topk(k, dim=1).indices                            # (B, k)
        keep = torch.zeros_like(flat).scatter_(1, idx, 1.0)
        keep = keep.view(B, 1, H, W)
        self.last_drop_frac = float(1.0 - keep.mean())
        return x * keep

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, participation={self.participation}"


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 256, 8, 8).abs()
    for m in ['dropblock', 'senet', 'cbam', 'wcd', 'focuseddrop']:
        block = build_baseline(m, drop_rate=0.2)
        block.train()
        y = block(x)
        print(f"{m:12s} -> {tuple(y.shape)}  drop_frac="
              f"{getattr(block, 'last_drop_frac', float('nan')):.4f}")
