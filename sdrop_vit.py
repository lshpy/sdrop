"""
Suppressive Dropout for Vision Transformers (SDrop-ViT)
--------------------------------------------------------
Head-level adaptation of the SDrop principle to Multi-Head Self-Attention.

For each attention block, we treat each head's attention map A^(h) in R^{N x N}
as the analogue of a CNN feature channel and compute a head-level EGPG score:

    E_h^ViT = 1 - (1 + gamma * mean(A^(h)^2))^{-delta}
    P_h^ViT = max(A^(h)) / (sum(A^(h)) + eps)
    s_h^ViT = E_h^ViT * (1 - P_h^ViT)

High-score heads are stochastically dropped during training (drop = mask the
head's contribution to the attention output by zeroing its output projection
slice). The selection mirrors the channel-level Equation (5) of the paper.

Reference (Appendix C of the Springer ML extended version):
  Lee, S. & Longo, L. "Suppressive Dropout: A Bio-Inspired and Explainable
  Channel-Selective Regularization Framework..." (in preparation, 2026)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Head-level EGPG score
# ---------------------------------------------------------------------------

def head_energy(A: torch.Tensor, gamma: float = 1.0, delta: float = 1.0) -> torch.Tensor:
    """
    E_h = 1 - (1 + gamma * mean_{i,j} A_{h,i,j}^2)^{-delta}

    Args:
        A: (B, H, N, N) attention maps (after softmax)
    Returns:
        (B, H) in (0, 1)
    """
    mean_sq = A.pow(2).mean(dim=(2, 3))           # (B, H)
    return 1.0 - (1.0 + gamma * mean_sq).pow(-delta)


def head_peakedness(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    P_h = max_{i,j} A_{h,i,j} / (sum_{i,j} A_{h,i,j} + eps)
    """
    peak = A.amax(dim=(2, 3))                     # (B, H)
    total = A.sum(dim=(2, 3))                     # (B, H)
    return peak / (total + eps)


def head_egpg(A: torch.Tensor, gamma: float = 1.0, delta: float = 1.0,
              eps: float = 1e-6) -> torch.Tensor:
    """
    s_h = E_h * (1 - P_h)
    """
    E = head_energy(A, gamma, delta)
    P = head_peakedness(A, eps)
    return E * (1.0 - P)


def head_drop_mask(scores: torch.Tensor, drop_rate: float,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    p_drop,h = drop_rate * s_h / max_h s_h
    mask[b, h] = Bernoulli(1 - p_drop,h)   (1 = keep, 0 = drop)
    """
    s_max = scores.amax(dim=1, keepdim=True).clamp_min(eps)
    p = drop_rate * (scores / s_max)
    keep_prob = (1.0 - p).clamp(0.0, 1.0)
    return torch.bernoulli(keep_prob)              # (B, H)


# ---------------------------------------------------------------------------
# SDrop multi-head self-attention block
# ---------------------------------------------------------------------------

class SDropMultiheadSelfAttention(nn.Module):
    """
    Drop-in replacement for a standard MHSA block with head-level SDrop.

    During training, computes attention maps, scores them with the EGPG criterion,
    then masks per-head outputs before the final projection.
    During eval, behaves as a standard MHSA block (no dropping).
    """

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 sdrop_rate: float = 0.0, gamma: float = 1.0, delta: float = 1.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # SDrop hyperparameters
        self.sdrop_rate = float(sdrop_rate)
        self.gamma = float(gamma)
        self.delta = float(delta)

        # Diagnostics: store the latest score / mask for visualization
        self.last_scores: Optional[torch.Tensor] = None
        self.last_mask: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ---- SDrop head selection (training only) ----
        if self.training and self.sdrop_rate > 0.0:
            with torch.no_grad():
                scores = head_egpg(attn, self.gamma, self.delta)   # (B, H)
                mask = head_drop_mask(scores, self.sdrop_rate)      # (B, H)
                self.last_scores = scores.detach()
                self.last_mask = mask.detach()
            # apply per-head mask to value-attention output
            out = attn @ v                              # (B, H, N, head_dim)
            out = out * mask.unsqueeze(-1).unsqueeze(-1)
            # rescale to keep expectation comparable (inverted dropout)
            keep_frac = mask.mean(dim=1, keepdim=True).clamp_min(1e-6)
            out = out / keep_frac.unsqueeze(-1).unsqueeze(-1)
        else:
            out = attn @ v
            self.last_scores = None
            self.last_mask = None

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# Minimal Vision Transformer with SDrop on selected layers
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image -> sequence of patch embeddings via Conv2d."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                            # (B, D, H/p, W/p)
        return x.flatten(2).transpose(1, 2)         # (B, N, D)


class MLP(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 sdrop_rate: float = 0.0, gamma: float = 1.0, delta: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDropMultiheadSelfAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            sdrop_rate=sdrop_rate, gamma=gamma, delta=delta,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SDropViT(nn.Module):
    """
    A small Vision Transformer with SDrop applied to the final two blocks
    (mirroring the L3/L4 placement used for ResNets in the CNN experiments).

    Defaults: ViT-Tiny-ish (depth=12, heads=3, dim=192) for CIFAR-100.
    Patch size 4 → 8x8=64 tokens for 32x32 inputs.
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 100, embed_dim: int = 192, depth: int = 12,
                 num_heads: int = 3, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 sdrop_rate: float = 0.0, sdrop_layers=("L3", "L4"),
                 gamma: float = 1.0, delta: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Decide which block indices receive SDrop.
        # We treat the final block as "L4" and the penultimate as "L3", matching
        # the paper's convention.
        sdrop_indices = set()
        if "L4" in sdrop_layers:
            sdrop_indices.add(depth - 1)
        if "L3" in sdrop_layers:
            sdrop_indices.add(depth - 2)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                sdrop_rate=sdrop_rate if i in sdrop_indices else 0.0,
                gamma=gamma, delta=delta,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                    # CLS token


# ---------------------------------------------------------------------------
# Convenience builder for use inside train.py
# ---------------------------------------------------------------------------

def build_sdrop_vit(num_classes: int = 100, img_size: int = 32,
                    method: str = "sdrop_vit", drop_rate: float = 0.1,
                    layers=("L3", "L4"), gamma: float = 1.0, delta: float = 1.0):
    """
    Build a ViT-Tiny with SDrop applied to selected attention blocks.

    method:
      - "vit"            : plain ViT-Tiny (no SDrop)
      - "sdrop_vit"      : SDrop on attention heads at the chosen layers
      - "sdrop_vit_full" : SDrop on every layer (ablation)
    """
    if method == "vit":
        sdrop_rate = 0.0
        sdrop_layers = ()
    elif method == "sdrop_vit":
        sdrop_rate = drop_rate
        sdrop_layers = tuple(layers) if layers else ("L3", "L4")
    elif method == "sdrop_vit_full":
        sdrop_rate = drop_rate
        sdrop_layers = ("L1", "L2", "L3", "L4")  # interpreted by index trick below
    else:
        raise ValueError(f"unknown ViT method '{method}'")

    return SDropViT(
        img_size=img_size, patch_size=4, in_chans=3, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
        sdrop_rate=sdrop_rate, sdrop_layers=sdrop_layers,
        gamma=gamma, delta=delta,
    )


if __name__ == "__main__":
    # Sanity check: build, forward, and inspect head scores.
    torch.manual_seed(0)
    model = build_sdrop_vit(num_classes=100, method="sdrop_vit", drop_rate=0.1)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    last_block = model.blocks[-1]
    print("Last-block SDrop scores:", last_block.attn.last_scores)
    print("Last-block SDrop mask  :", last_block.attn.last_mask)
