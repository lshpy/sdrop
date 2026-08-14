#!/usr/bin/env python3
"""
정성적 비교 그림: 같은 이미지에서 Baseline 과 SDrop 학습 모델의 Grad-CAM.

    python make_qualitative.py

이미지 선정 기준이 핵심이다 — Few 그룹(학습 20장 미만) 클래스 중에서
**SDrop 모델은 맞히고 Baseline 은 틀린** 검증 이미지를 고른다. 체리피킹
반론을 피하기 위해 그 조건을 만족하는 것 중 앞에서부터 순서대로 쓴다
(캡션에 선정 기준을 명시할 것).

Grad-CAM 은 layer3(8x8)에서 계산한다. L4(4x4)는 너무 거칠다.
"""
import io
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_dataset
from model import build_model
from eval_channel_usage import class_groups

OUT_DIRS = ['../sdrop-paper/00_ACTIVE_SpringerML/manuscript', '../overleaf_upload']
N_ROWS = 4

BASE_CKPT = 'checkpoints/cifar100_lt_none_rate0.1_L4_imb100_seed0_best.pth'
SDRP_CKPT = 'checkpoints/cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_seed0_best.pth'

MEAN = np.array([0.5071, 0.4865, 0.4409])
STD = np.array([0.2673, 0.2564, 0.2762])


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    m = build_model(arch='resnet18', num_classes=100, method=a['method'],
                    drop_rate=a['drop_rate'], layers=a['layers'],
                    grid_size=a.get('grid_size', 2),
                    peakedness=a.get('peakedness', 'max'), norm=a.get('norm', 'max'),
                    gamma=1.0, grad_mode=a.get('grad_mode', 'off'), pretrained=False)
    m.load_state_dict(ck['state_dict'])
    return m.to(device).eval()


def gradcam(model, x, target, device):
    """layer3 기준 Grad-CAM. x: (1,3,32,32) 정규화된 텐서."""
    acts, grads = [], []
    h1 = model.layer3.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = model.layer3.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    model.zero_grad()
    out = model(x.to(device))
    out[0, target].backward()
    h1.remove(); h2.remove()
    A, G = acts[0][0], grads[0][0]                       # (C,8,8)
    w = G.mean(dim=(1, 2), keepdim=True)                 # GAP of gradients
    cam = F.relu((w * A).sum(0))
    cam = cam / cam.max().clamp(min=1e-8)
    return cam.detach().cpu().numpy()


def denorm(x):
    img = x.permute(1, 2, 0).numpy() * STD + MEAN
    return np.clip(img, 0, 1)


def upsample(a, size=160):
    """nearest 로 픽셀감 유지(원본), CAM 은 bilinear 로 부드럽게."""
    t = torch.tensor(a)[None, None].float()
    return F.interpolate(t, size=(size, size), mode='bilinear',
                         align_corners=False)[0, 0].numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = load_model(BASE_CKPT, device)
    sdrp = load_model(SDRP_CKPT, device)
    _, vl, n_cls, _, _ = get_dataset('cifar100', './data', 256, 0)
    grp = class_groups(n_cls)          # (100,) 'Many'|'Medium'|'Few'

    with open('data/cifar-100-python/meta', 'rb') as f:
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in pickle.load(f)['fine_label_names']]

    # Few 클래스에서 SDrop 만 맞힌 이미지 수집 (등장 순서대로)
    picks = []
    with torch.no_grad():
        for x, y in vl:
            pb = base(x.to(device)).argmax(1).cpu()
            ps = sdrp(x.to(device)).argmax(1).cpu()
            for i in range(len(y)):
                if grp[y[i]] == 'Few' and ps[i] == y[i] and pb[i] != y[i]:
                    picks.append((x[i], int(y[i]), int(pb[i])))
            if len(picks) >= N_ROWS:
                break
    picks = picks[:N_ROWS]
    print(f'선정: Few 클래스, SDrop 정답 & Baseline 오답 — {len(picks)}장')

    fig, axes = plt.subplots(len(picks), 3, figsize=(6.2, 2.05 * len(picks)))
    cols = ['Input (Few class)', 'Baseline Grad-CAM', 'SDrop Grad-CAM']
    for r, (x, y, pb) in enumerate(picks):
        img = denorm(x)
        img_up = np.stack([upsample(img[:, :, c]) for c in range(3)], -1)
        cams = [gradcam(m, x[None], y, device) for m in (base, sdrp)]
        axes[r, 0].imshow(np.clip(img_up, 0, 1))
        for c, cam in enumerate(cams, start=1):
            axes[r, c].imshow(np.clip(img_up, 0, 1))
            axes[r, c].imshow(upsample(cam), cmap='turbo', alpha=0.45,
                              vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f'{names[y]}', fontsize=9)
        axes[r, 1].set_xlabel(f'pred: {names[pb]}', fontsize=8, color='#a33')
        axes[r, 2].set_xlabel(f'pred: {names[y]}', fontsize=8, color='#161')
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)

    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, 'fig_gradcam_few.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(d, 'fig_gradcam_few.png'), dpi=200,
                    bbox_inches='tight')
    print('저장: fig_gradcam_few.pdf /', ', '.join(OUT_DIRS))


if __name__ == '__main__':
    main()
