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

    NOTE: For ViT we feed the *pre-softmax* attention logits into A here,
    not the post-softmax probabilities. Post-softmax values are bounded
    in [0,1] and produce vanishingly small mean-squared activations
    (E_h ~ 0 for every head), which destroys the EGPG ranking signal.
    Pre-softmax logits preserve the relative magnitude differences
    between heads that the EGPG score is designed to exploit.

    Args:
        A: (B, H, N, N) attention LOGITS (pre-softmax), or feature tensor
    Returns:
        (B, H) in (0, 1)
    """
    mean_sq = A.pow(2).mean(dim=(2, 3))           # (B, H)
    return 1.0 - (1.0 + gamma * mean_sq).pow(-delta)


def head_peakedness(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    P_h = max_{i,j} |A_{h,i,j}| / (sum_{i,j} |A_{h,i,j}| + eps)

    Using absolute values so the function is well-defined on pre-softmax
    logits (which can be negative).
    """
    abs_A = A.abs()
    peak = abs_A.amax(dim=(2, 3))                 # (B, H)
    total = abs_A.sum(dim=(2, 3))                 # (B, H)
    return peak / (total + eps)


def head_egpg(A: torch.Tensor, gamma: float = 1.0, delta: float = 1.0,
              eps: float = 1e-6) -> torch.Tensor:
    """
    s_h = E_h * (1 - P_h)

    A is expected to be the pre-softmax attention logits in our ViT setup.
    """
    E = head_energy(A, gamma, delta)
    P = head_peakedness(A, eps)
    return E * (1.0 - P)


def head_drop_mask(scores: torch.Tensor, drop_rate: float,
                   norm: str = 'max', eps: float = 1e-6) -> torch.Tensor:
    """
    norm='max'  : p_drop,h = drop_rate * s_h / max_h s_h            (기존)
    norm='mean' : p_drop,h = min(1, drop_rate * H * s_h / sum_h s_h)

    'max' 에서는 최고점 헤드만 정확히 drop_rate 로 떨어지므로 실효 비율이
    drop_rate * mean(s)/max(s) <= drop_rate 이다. CNN 쪽 drop_mask_from_scores()
    와 같은 규약을 써야 nominal p 기준의 baseline 과 비교가 성립한다.

    mask[b, h] = Bernoulli(1 - p_drop,h)   (1 = keep, 0 = drop)
    """
    if norm == 'max':
        s_max = scores.amax(dim=1, keepdim=True).clamp_min(eps)
        p = drop_rate * (scores / s_max)
    elif norm == 'mean':
        H = scores.shape[1]
        s_sum = scores.sum(dim=1, keepdim=True).clamp_min(eps)
        p = (drop_rate * H * scores / s_sum).clamp(max=1.0)
    else:
        raise ValueError(f"Unknown norm: '{norm}'. Choose 'max' or 'mean'.")
    keep_prob = (1.0 - p).clamp(0.0, 1.0)
    return torch.bernoulli(keep_prob)              # (B, H)




# ---------------------------------------------------------------------------
# Token / Key level scoring  (헤드 3개는 해상도가 너무 낮다)
# ---------------------------------------------------------------------------

def key_egpg(attn_logits: torch.Tensor, gamma: float = 100.0,
             delta: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    키(=토큰) 단위 EGPG 점수.  attn_logits: (B, H, N, N)  ->  (B, N)

    헤드 단위 드롭은 후보가 num_heads(=3)개뿐이라 마스크의 경우의 수가 2^3 = 8
    밖에 안 된다. 점수 함수가 무작위 선택과 구별될 여지가 사실상 없다.
    키 단위로 내리면 후보가 N(=65)개가 되어 해상도가 20배 이상 올라간다.
    DropKey(CVPR'23)가 softmax 이전에 key 를 드롭하는 것과 같은 지점이다.

      E_j = 1 - (1 + gamma * Sigma_j)^-delta,  Sigma_j = mean_{b,h,i} logits[.,.,i,j]^2
      P_j = max_i |a_ij| / sum_i |a_ij|        (질의 축 위의 집중도)
      s_j = E_j * (1 - P_j)
    """
    a = attn_logits.mean(dim=1)                      # (B, N, N)  헤드 평균
    mean_sq = a.pow(2).mean(dim=1)                   # (B, N)     질의 축 평균
    E = 1.0 - (1.0 + gamma * mean_sq).pow(-delta)
    absa = a.abs()
    P = absa.amax(dim=1) / (absa.sum(dim=1) + eps)   # (B, N)
    return E * (1.0 - P)


def token_keep_prob(scores: torch.Tensor, drop_rate: float,
                    norm: str = 'mean', eps: float = 1e-6) -> torch.Tensor:
    """token_drop_mask 와 같은 규약의 채널별 keep 확률 (재스케일용)."""
    if norm == 'max':
        p = drop_rate * scores / scores.amax(dim=1, keepdim=True).clamp_min(eps)
    else:
        N = scores.shape[1]
        p = (drop_rate * N * scores /
             scores.sum(dim=1, keepdim=True).clamp_min(eps)).clamp(max=1.0)
    return (1.0 - p).clamp_min(eps)


def token_drop_mask(scores: torch.Tensor, drop_rate: float,
                    norm: str = 'mean', keep_cls: bool = True,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    (B, N) 점수 -> (B, N) keep 마스크. cls 토큰(index 0)은 절대 드롭하지 않는다
    — 분류 헤드가 그 토큰만 읽으므로 드롭하면 그 샘플의 손실이 무의미해진다.
    """
    if norm == 'max':
        denom = scores.amax(dim=1, keepdim=True).clamp_min(eps)
        p = drop_rate * scores / denom
    elif norm == 'mean':
        N = scores.shape[1]
        denom = scores.sum(dim=1, keepdim=True).clamp_min(eps)
        p = (drop_rate * N * scores / denom).clamp(max=1.0)
    else:
        raise ValueError(f"Unknown norm: '{norm}'. Choose 'max' or 'mean'.")
    keep = torch.bernoulli((1.0 - p).clamp(0.0, 1.0))
    if keep_cls:
        keep[:, 0] = 1.0
    return keep


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
                 sdrop_rate: float = 0.0, gamma: float = 1.0, delta: float = 1.0,
                 score_input: str = "pre", norm: str = "max",
                 unit: str = "head"):
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
        assert score_input in ("pre", "post"), score_input
        self.score_input = score_input
        assert norm in ("max", "mean"), norm
        self.norm_mode = norm
        assert unit in ("head", "key", "token"), unit
        self.unit = unit

        # Diagnostics: store the latest score / mask for visualization
        self.last_scores: Optional[torch.Tensor] = None
        self.last_mask: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each (B, H, N, head_dim)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N) pre-softmax
        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ---- SDrop selection (training only) ----
        if self.training and self.sdrop_rate > 0.0 and self.unit == "key":
            # DropKey 계열: softmax 이전에 키를 마스킹한다. 드롭된 키는 어떤
            # 질의로부터도 어텐션을 못 받으므로 개입이 attention 분포 자체를
            # 바꾼다 — 출력에 마스크를 곱하는 것보다 훨씬 강하게 작동한다.
            with torch.no_grad():
                _src = attn_logits if self.score_input == "pre" else attn
                scores = key_egpg(_src, self.gamma, self.delta)          # (B, N)
                mask = token_drop_mask(scores, self.sdrop_rate, self.norm_mode)
                self.last_scores = scores.detach()
                self.last_mask = mask.detach()
            neg = torch.finfo(attn_logits.dtype).min
            masked_logits = attn_logits.masked_fill(
                (mask == 0).unsqueeze(1).unsqueeze(1), neg)
            attn = masked_logits.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v
        elif self.training and self.sdrop_rate > 0.0 and self.unit == "token":
            # 토큰(=출력 위치) 단위 드롭. 어텐션 출력의 토큰 행을 통째로 0으로.
            with torch.no_grad():
                _src = attn_logits if self.score_input == "pre" else attn
                scores = key_egpg(_src, self.gamma, self.delta)          # (B, N)
                mask = token_drop_mask(scores, self.sdrop_rate, self.norm_mode)
                self.last_scores = scores.detach()
                self.last_mask = mask.detach()
            out = attn @ v                              # (B, H, N, head_dim)
            out = out * mask.unsqueeze(1).unsqueeze(-1)
            # cls(index 0) 는 절대 드롭하지 않으므로 재스케일도 하면 안 된다.
            # 배치 실현 비율이 아니라 토큰별 기대 keep 확률로 나눈다.
            keep_p = token_keep_prob(scores, self.sdrop_rate, self.norm_mode)
            keep_p[:, 0] = 1.0
            out = out / keep_p.unsqueeze(1).unsqueeze(-1)
        elif self.training and self.sdrop_rate > 0.0:
            with torch.no_grad():
                # Score from pre-softmax logits — preserves head-magnitude
                # differences that vanish after softmax.
                _src = attn_logits if self.score_input == "pre" else attn
                scores = head_egpg(_src, self.gamma, self.delta)   # (B, H)
                mask = head_drop_mask(scores, self.sdrop_rate, self.norm_mode)  # (B, H)
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
                 sdrop_rate: float = 0.0, gamma: float = 1.0, delta: float = 1.0,
                 score_input: str = "pre", norm: str = "max",
                 unit: str = "head"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDropMultiheadSelfAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            sdrop_rate=sdrop_rate, gamma=gamma, delta=delta,
            score_input=score_input, norm=norm, unit=unit,
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
                 score_input: str = "pre", norm: str = "max",
                 unit: str = "head",
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
        # "all" 은 전 블록. 이전에는 sdrop_vit_full 이 ("L1","L2","L3","L4") 를
        # 넘겼는데 여기서 L3/L4 만 해석해 마지막 두 블록에만 걸렸다 — 즉 full
        # 어블레이션이 sdrop_vit 와 동일하게 동작했다.
        sdrop_indices = set()
        if "all" in sdrop_layers:
            sdrop_indices = set(range(depth))
        else:
            unknown = [l for l in sdrop_layers if l not in ("L3", "L4")]
            if unknown:
                # 조용히 무시하면 SDrop 이 통째로 꺼진 채 sdrop_vit 이름으로 저장된다.
                raise ValueError(
                    f"ViT 는 L3(=끝에서 2번째 블록)/L4(=마지막 블록) 만 지원한다 "
                    f"({unknown} 요청됨).")
            if "L4" in sdrop_layers:
                sdrop_indices.add(depth - 1)
            if "L3" in sdrop_layers:
                sdrop_indices.add(depth - 2)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                sdrop_rate=sdrop_rate if i in sdrop_indices else 0.0,
                gamma=gamma, delta=delta, score_input=score_input, norm=norm,
                unit=unit,
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
                    layers=("L3", "L4"), gamma: float = 100.0, delta: float = 1.0,
                    score_input: str = "pre", norm: str = "max",
                 unit: str = "head"):
    """gamma defaulted to 100 to amplify per-head differences in pre-softmax
    attention logits, which are typically two orders of magnitude smaller
    than CNN feature activations."""
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
        if layers:
            raise ValueError("sdrop_vit_full 은 전 블록에 적용하므로 --layers 를 "
                             "받지 않는다 (run_id 가 태그를 붙여 유령 어블레이션이 된다).")
        sdrop_layers = ("all",)   # 전 블록 (이전 구현은 마지막 2개에만 걸렸다)
    else:
        raise ValueError(f"unknown ViT method '{method}'")

    return SDropViT(
        img_size=img_size, patch_size=4, in_chans=3, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
        sdrop_rate=sdrop_rate, sdrop_layers=sdrop_layers,
        gamma=gamma, delta=delta, score_input=score_input, norm=norm,
        unit=unit,
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
