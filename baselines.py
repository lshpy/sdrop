"""
Comparison baselines for the Springer ML extended version.

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
        # gamma normalisation as in the paper:
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

def build_baseline(method: str, drop_rate: float = 0.1,
                   block_size: int = 3, reduction: int = 16) -> nn.Module:
    """
    Build one of {'dropblock', 'senet', 'cbam'} as a single drop-in module
    that takes / returns a (B, C, H, W) tensor.
    """
    if method == 'dropblock':
        return DropBlock2D(drop_prob=drop_rate, block_size=block_size)
    if method == 'senet':
        return _LazySEWrapper(reduction=reduction)
    if method == 'cbam':
        return _LazyCBAMWrapper(reduction=reduction)
    raise ValueError(f"unknown baseline method '{method}'")


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 256, 8, 8)
    for m in ['dropblock', 'senet', 'cbam']:
        block = build_baseline(m, drop_rate=0.2)
        block.train()
        y = block(x)
        print(f"{m:10s} -> output shape {tuple(y.shape)}, "
              f"diff norm {(y - x).norm().item():.3f}")
