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

class GuidedDropout(nn.Module):
    """
    Guided Dropout (Keshari, Singh & Vatsa, CVPR 2019; arXiv:1812.03965).

    각 채널(feature map)에 **학습되는 strength t_c** 를 곱하고, 학습 중에는
    t 가 큰 채널을 떨어뜨려 t 가 작은 채널이 자라게 한다.
        y_c = t_c * x_c
        'topk' : t 상위 k=round(C*p) 채널을 결정적으로 드롭 (GD-top-k)
        'dr'   : active region(t 상위 active_frac) 안에서 k 개를 무작위 드롭 (GD-DR)
    t 는 전역·학습 파라미터라 샘플에 따라 변하지 않고, 공간 구조도 보지 않는다.
    SDrop 과의 차이가 정확히 이 두 지점이다.

    원논문 스케줄(200ep): 40ep strength 학습(무드롭) → 드롭/무드롭 교대.
    sdrop.DropRateMonitor 의 'gdtopk' 프로파일이 그 교대를 비율로 재현한다.
    기본값 schedule='gdtopk'. 'const' 를 주면 SDrop 과 같은 예산으로 계속 건다.

    `--select topk --schedule gdtopk` 의 SDrop 은 점수가 EGPG 라 이 재현이 아니다.
    """
    def __init__(self, channels: int, drop_rate: float = 0.1, mode: str = 'topk',
                 schedule: str = 'gdtopk', active_frac: float = 0.5,
                 warmup_frac: float = 0.0, participation: float = 1.0):
        super().__init__()
        assert mode in ('topk', 'dr'), mode
        self.drop_rate = float(drop_rate)
        self.mode = mode
        self.active_frac = float(active_frac)
        # 사전학습 백본에 붙일 때 초기 forward 를 바꾸지 않도록 1 로 초기화.
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
            # 샘플별로 active region 안에서 k 개 무작위 선택
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
    'guided_topk', 'guided_dr', 'targeted'} as a single drop-in module that takes /
    returns a (B, C, H, W) tensor.
    """
    if method == 'targeted':
        return TargetedDropout(drop_rate=drop_rate)
    if method in ('maxdrop', 'maxdrop_ch'):
        return MaxDrop(drop_rate=drop_rate, mode='feature' if method == 'maxdrop' else 'channel')
    if method in ('guided_topk', 'guided_dr'):
        if channels is None:
            raise ValueError("guided dropout 은 channels 인자가 필요하다 (strength t).")
        return GuidedDropout(channels, drop_rate=drop_rate,
                             mode='topk' if method == 'guided_topk' else 'dr',
                             schedule=schedule, warmup_frac=warmup_frac)
    if method == 'dropblock':
        return DropBlock2D(drop_prob=drop_rate, block_size=block_size)
    # Lazy 래퍼는 첫 forward 에서 파라미터를 만드는데 optimizer 는 그 전에
    # 생성된다 -> SE/CBAM 가중치가 랜덤 초기값에 고정된 채 학습을 통과한다.
    # 채널 수를 아는 경우 즉시 생성한다. 모르면 lazy 로 떨어지되 경고한다.
    if method == 'senet':
        if channels is None:
            raise ValueError("senet 은 channels 인자가 필요하다 — lazy 생성 시 "
                             "optimizer 가 파라미터를 잡지 못해 학습되지 않는다.")
        return SEBlock(channels, reduction=reduction)
    if method == 'cbam':
        if channels is None:
            raise ValueError("cbam 은 channels 인자가 필요하다 — lazy 생성 시 "
                             "optimizer 가 파라미터를 잡지 못해 학습되지 않는다.")
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
    SDrop 의 가장 가까운 선행연구. 채널의 평균 활성값(GAP)으로 채널을 순위화하고
    그 값에 비례한 확률로 채널을 *유지*한다 (활성이 큰 채널일수록 살아남는다).

        score_c = GAP(x)_c
        p_keep,c = score_c / max_c(score_c)

    SDrop 과의 차이 — 논문에서 반드시 설명해야 할 지점:
      1) 방향이 반대다. WCD 는 강한 채널을 *남기고*, SDrop 은 강한 채널을 *떨어뜨린다*.
      2) WCD 의 신호는 평균 활성 그 자체지만, SDrop 은 포화 비선형
         E_c = 1-(1+gamma*Sigma_c)^-delta 를 씌우고 공간 집중도 (1-P_c) 를 곱한다.

    Args:
        drop_rate: 전체 강도. p_keep 을 (1-drop_rate) 만큼 아래로 눌러 실효
                   드롭 비율이 nominal 과 맞도록 스케일한다.
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
        score = score - score.amin(dim=1, keepdim=True)              # 음수 활성 보정
        rng = score.amax(dim=1, keepdim=True)
        # 전 채널 GAP 이 같으면(죽은 층, 상수 입력) score 가 전부 0 이 되어
        # p_keep=0 -> 전 채널 드롭(실효율 1.0)이 된다. 경고 없이 강도가 3배가
        # 되므로, 그럴 때는 균등 Bernoulli 로 폴백한다.
        degenerate = (rng <= self.eps)
        denom = rng.clamp_min(self.eps)
        p_keep = torch.where(degenerate,
                             torch.full_like(score, 1.0 - self.drop_rate),
                             score / denom)                          # (B, C) in [0, 1]
        # 평균 유지율이 (1 - drop_rate) 가 되도록 스케일.
        # 단순히 곱하고 clamp(0,1) 하면 1 을 넘는 질량이 잘려나가 평균이 목표보다
        # 낮아진다(nominal 0.1 -> 실효 0.20). clamp 이후의 평균이 목표가 되도록
        # 배율을 이분법으로 푼다.
        target = 1.0 - self.drop_rate
        lo = torch.zeros(score.shape[0], 1, device=x.device, dtype=score.dtype)
        hi = torch.full_like(lo, 1.0)
        cur = p_keep.mean(dim=1, keepdim=True).clamp_min(self.eps)
        hi = torch.maximum(hi, target / cur * 2.0)
        for _ in range(30):                       # 30회면 배정밀도 한계까지 수렴
            mid = 0.5 * (lo + hi)
            m = (p_keep * mid).clamp(0.0, 1.0).mean(dim=1, keepdim=True)
            too_small = (m < target)
            lo = torch.where(too_small, mid, lo)
            hi = torch.where(too_small, hi, mid)
        scaled = (p_keep * (0.5 * (lo + hi))).clamp(0.0, 1.0)
        # 목표 유지율에 도달 불가능한 표본(대부분 채널이 죽은 경우)은 균등 폴백.
        reached = scaled.mean(dim=1, keepdim=True) >= target - 1e-3
        p_keep = torch.where(reached | degenerate, scaled,
                             torch.full_like(scaled, target))
        with torch.no_grad():           # 점수 계산 그래프를 물고 있지 않도록
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
    SDrop 과 방향이 반대인 대조군. 활성이 가장 큰 채널을 '기준 채널'로 잡고,
    그 채널에서 활성이 높은 *공간 위치*만 남기고 나머지 위치를 전 채널에서 버린다.
    "표적에 집중시킨다"는 논리.

    원논문의 핵심 장치: **배치의 일부에만 적용한다** (participation, 기본 10%).
    강도를 "얼마나 세게"가 아니라 "얼마나 자주"로 조절하는 직교 축이다.

    Args:
        drop_rate     : 기준 채널에서 상위 (1-drop_rate) 비율의 위치를 남긴다
        participation : 이 확률로만 해당 배치에 적용 (원논문 0.1)
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
        # 기준 채널 = 채널별 최대 활성이 가장 큰 채널
        ref_idx = x.amax(dim=(2, 3)).argmax(dim=1)                   # (B,)
        ref = x[torch.arange(B, device=x.device), ref_idx]           # (B, H, W)
        flat = ref.reshape(B, -1)
        k = max(1, int(round(H * W * (1.0 - self.drop_rate))))
        # 임계값 비교(`>=`)를 쓰면 post-ReLU 입력에서 값의 50~80%가 정확히 0 이라
        # thresh==0 이 되고 전 위치가 통과해 완전한 no-op 이 된다.
        # topk 인덱스로 마스크를 직접 만들어 항상 정확히 k 개만 남긴다.
        idx = flat.topk(k, dim=1).indices                            # (B, k)
        keep = torch.zeros_like(flat).scatter_(1, idx, 1.0)
        keep = keep.view(B, 1, H, W)
        self.last_drop_frac = float(1.0 - keep.mean())
        return x * keep

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, participation={self.participation}"


class MaxDrop(nn.Module):
    """
    Max-Drop (Park & Kwak, ACCV 2016), the closest published rule that removes the strongest activations.

    mode='feature' : the published feature-wise rule. At every spatial position the channel holding the
                     maximum activation is zeroed with probability `drop_rate` (Bernoulli per sample and
                     position). The fraction of elements removed is drop_rate / C, far below SDrop's budget,
                     so this arm reproduces the published operation rather than matching the budget.
    mode='channel' : budget-matched channel analogue. For each sample the round(drop_rate * C) channels with
                     the largest maximum activation are zeroed (deterministic top-k, like SDrop --select topk
                     but ranked by the per-channel maximum instead of EGPG energy). Expected drop fraction = drop_rate.
    No rescaling (as published); no-op at eval.
    """
    def __init__(self, drop_rate: float = 0.1, mode: str = 'feature'):
        super().__init__()
        assert mode in ('feature', 'channel'), mode
        self.drop_rate = float(drop_rate); self.mode = mode
        self.last_drop_frac = float('nan')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x
        B, C, H, W = x.shape
        if self.mode == 'feature':
            is_max = x == x.amax(dim=1, keepdim=True)                                    # (B,C,H,W) one-hot over C
            coin = (torch.rand(B, 1, H, W, device=x.device) < self.drop_rate)
            drop = is_max & coin
        else:
            k = max(1, int(round(C * self.drop_rate)))
            top = x.detach().flatten(2).amax(2).topk(k, dim=1).indices                  # (B,k)
            drop = torch.zeros(B, C, dtype=torch.bool, device=x.device).scatter_(1, top, True).view(B, C, 1, 1).expand(B, C, H, W)
        self.last_drop_frac = float(drop.float().mean())
        return x.masked_fill(drop, 0.0)

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, mode={self.mode}"


class TargetedDropout(nn.Module):
    """
    Targeted Dropout (Gomez et al., ICLR-W 2019; arXiv:1905.13678), channel ("unit") form.

    Rank channels by magnitude, mark the *lowest* `target_frac` as the targeted set and
    drop each targeted channel with probability alpha = drop_rate / target_frac, so the
    expected overall drop fraction equals `drop_rate` (same budget as SDrop / random).
    Magnitude = per-sample mean-square activation of the channel (the activation analogue
    of the weight-norm criterion in the paper; no weights are owned by this module).
    It is SDrop's mirror image: SDrop drops the *strongest* channels, Targeted the weakest.
    Inverted-dropout rescaling as in nn.Dropout2d; no-op at eval.
    """
    def __init__(self, drop_rate: float = 0.1, target_frac: float = 0.5):
        super().__init__()
        self.drop_rate = float(drop_rate)
        self.target_frac = float(target_frac)
        self.alpha = min(1.0, self.drop_rate / self.target_frac)
        self.last_drop_frac = float('nan')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x
        B, C = x.shape[:2]
        mag = x.detach().float().pow(2).flatten(2).mean(2)                   # (B, C)
        k = max(1, int(round(C * self.target_frac)))
        targeted = torch.zeros(B, C, device=x.device, dtype=torch.bool)
        targeted.scatter_(1, mag.topk(k, dim=1, largest=False).indices, True)
        drop = targeted & (torch.rand(B, C, device=x.device) < self.alpha)
        keep = (~drop).to(x.dtype).view(B, C, 1, 1)
        self.last_drop_frac = float(drop.float().mean())
        return x * keep / (1.0 - self.alpha * self.target_frac)

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, target_frac={self.target_frac}, alpha={self.alpha:.3f}"


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 256, 8, 8).abs()
    for m in ['dropblock', 'senet', 'cbam', 'wcd', 'focuseddrop', 'targeted', 'maxdrop', 'maxdrop_ch']:
        block = build_baseline(m, drop_rate=0.2, channels=256)
        block.train()
        y = block(x)
        print(f"{m:12s} -> {tuple(y.shape)}  drop_frac="
              f"{getattr(block, 'last_drop_frac', float('nan')):.4f}")
