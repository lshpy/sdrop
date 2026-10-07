"""ImageNet 사전학습 ViT-B/16 + 마지막 블록 토큰 출력에 SDrop.

CNN 에서 (채널, 공간) 이던 것을 ViT 에서는 (임베딩 차원 D=768, 토큰 N=197) 으로 본다.
토큰 텐서 (B, N, D) → (B, D, N, 1) 로 바꿔 기존 SDrop 모듈(compute_score/_drop_probs)을 그대로 적용한다.
   methods: vitb_none | vitb_sdrop_energy | vitb_sdrop_gap | vitb_sdrop_random | vitb_sdrop_shuffle | vitb_sdrop
            vitbm_*  — 같은 점수를 **끝에서 두 번째 블록의 MLP 은닉 뉴런(GELU 뒤, 3072)** 에 적용. CNN 의 post-ReLU 채널에 해당하는 자리.
                       (뉴런=채널, 토큰 전부=공간. 마지막 블록에 걸면 패치 토큰이 CLS 로 전달될 어텐션이 없어 효과가 0 — 확인됨.
                        그래서 L-2 블록에 걸어 다음 블록의 어텐션이 마스킹 효과를 CLS 로 옮기게 한다.)
"""
import torch, torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sdrop import build_sdrop


class _NeuronSDrop(nn.Module):
    """MLP 은닉 활성 (B, N+1, H) 에 SDrop: (B, H, N+1, 1) 로 보고 뉴런=채널, 토큰(CLS 포함)=공간."""
    def __init__(self, sdrop): super().__init__(); self.sdrop = sdrop
    def forward(self, h):
        t = h.transpose(1, 2).unsqueeze(-1)
        return self.sdrop(t).squeeze(-1).transpose(1, 2)


class _TokenSDrop(nn.Module):
    """Residual stream (B, N+1, D) entering a block: tokens = channels, embedding dims = spatial; CLS excluded.
    The transformer analogue of a dominant channel is a dominant (high-norm) patch token (cf. Darcet et al. 2023)."""
    def __init__(self, sdrop): super().__init__(); self.sdrop = sdrop
    def forward(self, x):
        cls, p = x[:, :1], x[:, 1:]
        p = self.sdrop(p.unsqueeze(-1)).squeeze(-1)               # (B, N, D, 1) -> SDrop over N
        return torch.cat([cls, p], dim=1)


class _HeadSDropAttention(nn.Module):
    """torchvision MultiheadAttention with SDrop over heads: the per-head outputs (B, H, N+1, d) are
    treated as (channels=heads, spatial=tokens x head-dim) before out_proj. Identical to the wrapped
    module when the SDrop module is inactive (eval, or rate 0)."""
    def __init__(self, mha, sdrop):
        super().__init__(); self.mha = mha; self.sdrop = sdrop
    def forward(self, query, key, value, need_weights=False, **kw):
        m = self.mha; B, N, D = query.shape; H = m.num_heads; d = D // H
        q, k, v = nn.functional.linear(query, m.in_proj_weight, m.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, H, d).transpose(1, 2) for t in (q, k, v))
        o = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=m.dropout if self.training else 0.0)
        o = self.sdrop(o)                                          # (B, H, N, d): heads = channels
        return m.out_proj(o.transpose(1, 2).reshape(B, N, D)), None


class SDropViTB(nn.Module):
    def __init__(self, num_classes: int, method: str = 'vitb_none', drop_rate: float = 0.1,
                 pretrained: bool = True, **sdrop_kw):
        super().__init__()
        import re
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_classes)
        # 2026-09-29: sites picked by the extended diagnostic (analyze_units_vit_all.py)
        #   vitbt{k}_ : patch tokens of block k's output (= input of block k+1)
        #   vitbh{k}_ : attention heads inside block k
        site = re.match(r'vitb([th])(\d+)_', method)
        self.on_mlp = method.startswith('vitbm_'); self.on_tok = method.startswith('vitbt_')
        inner = re.sub(r'^vitb(m|t|[th]\d+)?_', '', method)
        mod = nn.Identity() if inner == 'none' else build_sdrop(inner, drop_rate, **sdrop_kw)
        if site:
            kind, k = site.group(1), int(site.group(2))
            if kind == 't':
                nxt = self.vit.encoder.layers[k + 1]
                self.vit.encoder.layers[k + 1] = nn.Sequential(_TokenSDrop(mod), nxt)
            else:
                blk = self.vit.encoder.layers[k]
                blk.self_attention = _HeadSDropAttention(blk.self_attention, mod)
            self.on_tok = True                                     # forward: no SDrop at the output
            self.sdrop = mod
        elif self.on_mlp:
            last = self.vit.encoder.layers[-2]              # 끝에서 두 번째 EncoderBlock; mlp = [Linear, GELU, Dropout, Linear, Dropout]
            last.mlp[2] = nn.Sequential(last.mlp[2], _NeuronSDrop(mod))   # GELU 뒤에 삽입
            self.sdrop = mod
        elif self.on_tok:
            last = self.vit.encoder.layers[-1]                # 마지막 블록 입력의 패치 토큰에 SDrop (토큰=채널, 768차원=공간)
            self.vit.encoder.layers[-1] = nn.Sequential(_TokenSDrop(mod), last)
            self.sdrop = mod
        else:
            self.sdrop = mod

    def forward(self, x):
        v = self.vit
        x = v._process_input(x)                                   # (B, N, D)
        x = torch.cat([v.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = v.encoder.dropout(x + v.encoder.pos_embedding)
        x = v.encoder.layers(x)                                   # 마지막 블록 출력 (B, N+1, D)
        if not (self.on_mlp or self.on_tok):
            t = x.transpose(1, 2).unsqueeze(-1)                   # (B, D, N+1, 1): D=채널, 토큰=공간
            t = self.sdrop(t)
            x = t.squeeze(-1).transpose(1, 2)
        x = v.encoder.ln(x)
        return v.heads(x[:, 0])


def build_vitb(num_classes, method, drop_rate, **kw):
    return SDropViTB(num_classes, method, drop_rate, **kw)
