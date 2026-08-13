#!/usr/bin/env python3
"""
저장된 체크포인트에서 Many/Medium/Few 그룹별 정확도를 뽑는다.

집계 정확도는 메커니즘을 희석한다. CIFAR-100-LT(rho=100)에서 Few 그룹은 30개
클래스뿐이라, 거기서 얻은 +X pp 는 전체 정확도로는 +0.30X pp 로만 보인다.
그룹을 나눠 보면 같은 현상이 3.3배 크게 보인다.

    python eval_tailgroups.py                          # checkpoints/ 전체
    python eval_tailgroups.py --pattern "*lt*"         # 롱테일 런만
    python eval_tailgroups.py --csv tailgroups.csv     # CSV 로도 저장

그룹 경계는 롱테일 학습 분포 기준이며 롱테일 문헌의 관례를 따른다
(Liu et al. 2019, Kang et al. 2020):
    Many   > 100장,  Medium 20-100장,  Few < 20장
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

from dataset import get_dataset, _exp_profile
from model import build_model


GROUP_NAMES = ['Many', 'Medium', 'Few']


def group_of(n_train: int) -> str:
    """롱테일 문헌의 관례: Many > 100장, Medium 20-100장, Few < 20장."""
    if n_train > 100:
        return 'Many'
    if n_train >= 20:
        return 'Medium'
    return 'Few'


def class_groups(imb_ratio: float, n_classes: int = 100, n_max: int = 500):
    """클래스 인덱스 -> 그룹 이름. 학습 시의 롱테일 프로파일을 그대로 재현한다."""
    counts = _exp_profile(n_classes, n_max, imb_ratio)
    return counts, [group_of(c) for c in counts]


@torch.no_grad()
def per_class_correct(model, loader, device, n_classes):
    """클래스별 (맞은 개수, 전체 개수)."""
    hit = np.zeros(n_classes, dtype=np.int64)
    tot = np.zeros(n_classes, dtype=np.int64)
    model.eval()
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        y = y.numpy()
        for t, p in zip(y, pred):
            tot[t] += 1
            hit[t] += int(t == p)
    return hit, tot


def rebuild(ckpt, device, num_classes, default_arch, default_pretrained):
    a = ckpt['args']
    method = a.get('method', 'none')
    if method.startswith('vit') or method.startswith('sdrop_vit'):
        from sdrop_vit import build_sdrop_vit
        img = {'cifar100': 32, 'cifar100_lt': 32,
               'tinyimagenet': 64, 'cub200': 224}.get(a.get('dataset'), 32)
        m = build_sdrop_vit(num_classes=num_classes, img_size=img, method=method,
                            drop_rate=a.get('drop_rate', 0.1), layers=a.get('layers', []))
    else:
        m = build_model(
            arch=a.get('arch') or default_arch,
            num_classes=num_classes,
            method=method,
            drop_rate=a.get('drop_rate', 0.1),
            layers=a.get('layers', []),
            grid_size=a.get('grid_size', 2),
            peakedness=a.get('peakedness', 'max'),
            norm=a.get('norm', 'max'),
            gamma=(None if a.get('self_gamma') else 1.0),
            grad_mode=a.get('grad_mode', 'off'),
            pretrained=(a.get('pretrained') if a.get('pretrained') is not None
                        else default_pretrained),
        )
    m.load_state_dict(ckpt['state_dict'])
    return m.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', default='./checkpoints')
    p.add_argument('--pattern', default='*_best.pth')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--csv', default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, args.pattern)))
    if not paths:
        sys.exit(f'체크포인트를 찾지 못했습니다: {args.ckpt_dir}/{args.pattern}')

    # 검증셋은 어느 런이든 균형 CIFAR-100 이므로 한 번만 만들어 재사용한다.
    _, val_loader, n_cls, d_arch, d_pre = get_dataset(
        'cifar100', args.data_root, args.batch_size, 0)

    rows = []
    print(f"{'run':<52}{'Many':>7}{'Medium':>8}{'Few':>7}{'All':>7}")
    print('-' * 81)

    skipped = []
    for path in paths:
        # history.csv 는 학습이 끝나야 쓰인다. 없으면 진행 중인 런이고, 그 시점의
        # _best.pth 를 평가하면 완료된 런과 나란히 놓기에 부적절하다.
        if not os.path.isfile(path.replace('_best.pth', '_history.csv')):
            skipped.append(os.path.basename(path).replace('_best.pth', ''))
            continue

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        a = ckpt['args']
        if a.get('dataset') not in ('cifar100', 'cifar100_lt'):
            continue

        imb = a.get('imb_ratio', 1.0)
        if imb <= 1.0:
            # 균형 학습이라도 롱테일 기준으로 나눠 보면 같은 클래스 집합을 비교할 수 있다
            imb = 100.0
        counts, groups = class_groups(imb, n_cls)

        model = rebuild(ckpt, device, n_cls, d_arch, d_pre)
        hit, tot = per_class_correct(model, val_loader, device, n_cls)

        acc = {}
        for name in GROUP_NAMES:
            sel = [i for i in range(n_cls) if groups[i] == name]
            acc[name] = 100.0 * hit[sel].sum() / max(1, tot[sel].sum())
        acc['All'] = 100.0 * hit.sum() / max(1, tot.sum())

        name = os.path.basename(path).replace('_best.pth', '')
        print(f"{name:<52}{acc['Many']:7.2f}{acc['Medium']:8.2f}"
              f"{acc['Few']:7.2f}{acc['All']:7.2f}")
        rows.append({'run': name, 'imb_ratio': imb,
                     'many': f"{acc['Many']:.2f}", 'medium': f"{acc['Medium']:.2f}",
                     'few': f"{acc['Few']:.2f}", 'all': f"{acc['All']:.2f}",
                     'best_acc_logged': f"{ckpt.get('best_acc', float('nan')):.2f}"})
        del model
        torch.cuda.empty_cache()

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f'\n저장: {args.csv}')

    if skipped:
        print('\n진행 중이라 제외 (history.csv 없음): ' + ', '.join(skipped))

    if rows:
        n_few = sum(1 for g in class_groups(100.0, n_cls)[1] if g == 'Few')
        print(f'\nFew 그룹은 {n_few}/{n_cls} 클래스입니다. 여기서의 +X pp 는 '
              f'전체 정확도로는 +{n_few/n_cls:.2f}X pp 로만 나타납니다.')


if __name__ == '__main__':
    main()
