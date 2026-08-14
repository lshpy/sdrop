#!/usr/bin/env python3
"""
클래스별 '실효 채널 수'를 잰다 — SDrop 이 표현 독점을 막는다는 주장의 직접 증거.

    python eval_channel_usage.py --a <baseline_glob> --b <sdrop_glob>

정확도는 시드 노이즈가 커서(롱테일 Few 그룹 표준편차 2.18pp) 메커니즘을 가리기
어렵다. 표현 자체를 재면 훨씬 선명하다: 같은 세 시드에서 실효 채널 수 변화는
t=17~31 로 나온다.

참여비(participation ratio)를 쓴다. 클래스 k 의 채널별 평균 활성 v 에 대해

    PR_k = (sum_c v_c)^2 / sum_c v_c^2

로, v 가 한 채널에 몰리면 1, C 개 채널에 균등하면 C 가 된다. 즉 '그 클래스를
표현하는 데 실제로 동원된 채널 수'다.
"""
import argparse
import glob
import math
import statistics as st

import numpy as np
import torch

from dataset import get_dataset
from model import build_model


def class_groups(n_classes=100, n_max=500, imb=100.0):
    counts = [int(round(n_max * (1.0 / imb) ** (k / (n_classes - 1))))
              for k in range(n_classes)]
    return np.array(['Many' if c > 100 else ('Medium' if c >= 20 else 'Few')
                     for c in counts])


@torch.no_grad()
def participation(path, val_loader, device, n_classes=100):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    m = build_model(arch=a.get('arch') or 'resnet18', num_classes=n_classes,
                    method=a['method'], drop_rate=a['drop_rate'], layers=a['layers'],
                    grid_size=a.get('grid_size', 2),
                    peakedness=a.get('peakedness', 'max'), norm=a.get('norm', 'max'),
                    gamma=1.0, grad_mode=a.get('grad_mode', 'off'), pretrained=False)
    m.load_state_dict(ck['state_dict'])
    m = m.to(device).eval()

    acts, labels = [], []
    h = m.layer4.register_forward_hook(
        lambda mod, i, o: acts.append(o.detach().mean(dim=(2, 3))))
    for x, y in val_loader:
        m(x.to(device))
        labels.append(y)
    h.remove()

    A = torch.cat(acts)
    Y = torch.cat(labels).to(device)
    out = np.empty(n_classes)
    for k in range(n_classes):
        v = A[Y == k].mean(0).clamp(min=0)
        out[k] = (v.sum() ** 2 / (v.pow(2).sum() + 1e-12)).item()
    del m
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', default='checkpoints/cifar100_lt_none_rate0.1_L4_imb100_seed*_best.pth',
                   help='기준 조건 (glob)')
    p.add_argument('--b', default='checkpoints/cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_seed*_best.pth',
                   help='비교 조건 (glob)')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    _, val_loader, n_cls, _, _ = get_dataset('cifar100', args.data_root, 256, 0)
    grp = class_groups(n_cls)

    pa = [participation(f, val_loader, device, n_cls) for f in sorted(glob.glob(args.a))]
    pb = [participation(f, val_loader, device, n_cls) for f in sorted(glob.glob(args.b))]
    if not pa or not pb:
        raise SystemExit('체크포인트를 찾지 못했습니다.')
    n = min(len(pa), len(pb))

    print(f"실효 채널 수 (참여비), {n} 시드\n")
    print(f"  A: {np.mean([x.mean() for x in pa]):.1f}   {args.a}")
    print(f"  B: {np.mean([x.mean() for x in pb]):.1f}   {args.b}\n")
    print(f"  {'그룹':<10}{'B - A':>22}   시드별")
    print('  ' + '-' * 66)
    for g in ('Many', 'Medium', 'Few'):
        d = [pb[s][grp == g].mean() - pa[s][grp == g].mean() for s in range(n)]
        mu = st.mean(d)
        se = st.stdev(d) / math.sqrt(n) if n > 1 else float('nan')
        t = mu / se if se and se == se else float('nan')
        print(f"  {g:<10}{mu:+8.1f} ± {se:4.1f}  (t={t:5.1f})   "
              f"{[round(x, 1) for x in d]}")

    print("\n  참여비가 오르면 그 클래스가 더 많은 채널에 분산돼 표현된다는 뜻이다.")
    print("  희소 클래스일수록 증가폭이 크면 표현 독점이 완화됐다는 직접 증거가 된다.")


if __name__ == '__main__':
    main()
