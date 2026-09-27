"""ImageNet-pretrained ViT-B/16 with SDrop on last-block token outputs.

What is (channel, space) in a CNN is taken as (embedding dim D=768, tokens N=197) in the ViT.
The token tensor (B, N, D) is reshaped to (B, D, N, 1) so the existing SDrop module (compute_score/_drop_probs) applies unchanged.
   methods: vitb_none | vitb_sdrop_energy | vitb_sdrop_gap | vitb_sdrop_random | vitb_sdrop_shuffle | vitb_sdrop
            vitbm_*  - the same scores applied to the **MLP hidden neurons of the second-to-last block (after GELU, 3072)**, the analogue of post-ReLU channels in a CNN.
                       (neuron = channel, all tokens = space. Applied in the last block it has no effect, since no attention is left to carry patch tokens to CLS.
                        It is therefore applied in block L-2 so the next block's attention carries the masking effect to CLS.)
"""
import torch, torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sdrop import build_sdrop


class _NeuronSDrop(nn.Module):
    """SDrop on MLP hidden activations (B, N+1, H), viewed as (B, H, N+1, 1): neuron = channel, tokens (incl. CLS) = space."""
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


class SDropViTB(nn.Module):
    def __init__(self, num_classes: int, method: str = 'vitb_none', drop_rate: float = 0.1,
                 pretrained: bool = True, **sdrop_kw):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_classes)
        self.on_mlp = method.startswith('vitbm_'); self.on_tok = method.startswith('vitbt_')
        inner = method.replace('vitbm_', '', 1).replace('vitbt_', '', 1).replace('vitb_', '', 1)
        mod = nn.Identity() if inner == 'none' else build_sdrop(inner, drop_rate, **sdrop_kw)
        if self.on_mlp:
            last = self.vit.encoder.layers[-2]              # second-to-last EncoderBlock; mlp = [Linear, GELU, Dropout, Linear, Dropout]
            last.mlp[2] = nn.Sequential(last.mlp[2], _NeuronSDrop(mod))   # inserted after GELU
            self.sdrop = mod
        elif self.on_tok:
            last = self.vit.encoder.layers[-1]                # SDrop on patch tokens at the last block's input (token = channel, 768 dims = space)
            self.vit.encoder.layers[-1] = nn.Sequential(_TokenSDrop(mod), last)
            self.sdrop = mod
        else:
            self.sdrop = mod

    def forward(self, x):
        v = self.vit
        x = v._process_input(x)                                   # (B, N, D)
        x = torch.cat([v.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = v.encoder.dropout(x + v.encoder.pos_embedding)
        x = v.encoder.layers(x)                                   # last block output (B, N+1, D)
        if not (self.on_mlp or self.on_tok):
            t = x.transpose(1, 2).unsqueeze(-1)                   # (B, D, N+1, 1): D = channel, tokens = space
            t = self.sdrop(t)
            x = t.squeeze(-1).transpose(1, 2)
        x = v.encoder.ln(x)
        return v.heads(x[:, 0])


def build_vitb(num_classes, method, drop_rate, **kw):
    return SDropViTB(num_classes, method, drop_rate, **kw)
