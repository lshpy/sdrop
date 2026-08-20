#!/usr/bin/env python3
"""원고에 바로 넣을 형태로 결과를 집계한다.

    python summarize_for_paper.py [checkpoints_dir ...]

각 표마다 mean ± sd (n seeds) 와, 짝지을 수 있으면 baseline 대비 차이를 낸다.
"""
import sys, os, glob, csv, statistics as st
from collections import OrderedDict

def load(dirs):
    best = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', '*_history.csv'), recursive=True):
            n = os.path.basename(f)[:-len('_history.csv')]
            try:
                rows = list(csv.DictReader(open(f)))
                if not rows: continue
                keys = [c for c in rows[0] if 'val' in c.lower() and 'acc' in c.lower()] \
                       or [c for c in rows[0] if 'acc' in c.lower()]
                if not keys: continue
                v = max(float(r[keys[0]]) for r in rows if r[keys[0]])
                if n not in best or v > best[n]: best[n] = v
            except Exception:
                pass
    return best

def agg(best, prefix, seeds=(0,1,2)):
    vals = [best[f'{prefix}_seed{s}'] for s in seeds if f'{prefix}_seed{s}' in best]
    return vals

def fmt(vals):
    if not vals: return '—', None
    if len(vals) == 1: return f'{vals[0]:.2f}', vals[0]
    return f'{vals[0] if False else st.mean(vals):.2f} ± {st.stdev(vals):.2f}', st.mean(vals)

def table(title, rows, best, baseline_key=None, seeds=(0,1,2)):
    print(f'\n## {title}\n')
    base_mean = None
    if baseline_key:
        _, base_mean = fmt(agg(best, baseline_key, seeds))
    print(f'| {"Method":38s} | Acc (%)        | n | Δ vs base |')
    print(f'|{"-"*40}|----------------|---|-----------|')
    for label, key in rows:
        vals = agg(best, key, seeds)
        s, m = fmt(vals)
        d = f'{m-base_mean:+.2f}' if (m is not None and base_mean is not None) else '—'
        print(f'| {label:38s} | {s:14s} | {len(vals)} | {d:9s} |')

def main():
    dirs = sys.argv[1:] or ['checkpoints']
    best = load(dirs)
    print(f'# SDrop 결과 요약  (history 파일 {len(best)}개)')

    table('Table 4 — CIFAR-100 아블레이션 (ResNet-18, 200ep)', [
        ('Baseline (no dropout)',       'cifar100_none_rate0.1_none'),
        ('Standard dropout',            'cifar100_dropout_rate0.1_L4'),
        ('Unstructured dropout',        'cifar100_dropout_std_rate0.1_L4'),
        ('Random channel removal',      'cifar100_sdrop_random_rate0.1_L4'),
        ('Peakedness-only',             'cifar100_sdrop_peak_rate0.1_L4'),
        ('SDrop-Energy',                'cifar100_sdrop_energy_rate0.1_L4'),
        ('SDrop (full EGPG)',           'cifar100_sdrop_rate0.1_L4'),
    ], best, baseline_key='cifar100_none_rate0.1_none')

    table('ViT-Tiny (CIFAR-100, 200ep) — 시드 5개', [
        ('ViT-Tiny baseline',           'cifar100_vit_rate0.1_none'),
        ('+ SDrop heads (p=0.3, L3+L4)','cifar100_sdrop_vit_rate0.3_L3+L4'),
    ], best, baseline_key='cifar100_vit_rate0.1_none', seeds=(0,1,2,3,4))

    table('Sec. 3.6 — head 점수의 입력 (seed 0)', [
        ('pre-softmax logits (제안)',    'cifar100_sdrop_vit_rate0.3_L3+L4'),
        ('post-softmax attention',      'cifar100_sdrop_vit_rate0.3_L3+L4_scorepost'),
    ], best, seeds=(0,))

    table('드롭률 민감도 (SDrop-Energy, L4, seed 0)', [
        (f'p = {p}', f'cifar100_sdrop_energy_rate{p}_L4') for p in
        ('0.05','0.1','0.2','0.3','0.5')
    ], best, seeds=(0,))

    table('삽입 위치 (seed 0)', [
        ('L3 only',    'cifar100_sdrop_energy_rate0.1_L3'),
        ('L4 only',    'cifar100_sdrop_energy_rate0.1_L4'),
        ('L3 + L4',    'cifar100_sdrop_energy_rate0.1_L3+L4'),
        ('SGridLC (L3, grid 4)', 'cifar100_sgridlc_rate0.3_L3_G4'),
    ], best, seeds=(0,))

    table('점수 함수 옵션 (seed 0)', [
        ('기본 (max norm, max peak, γ=1)', 'cifar100_sdrop_rate0.1_L4'),
        ('norm = mean',                    'cifar100_sdrop_rate0.1_L4_nmmean'),
        ('peakedness = entropy',           'cifar100_sdrop_rate0.1_L4_pkentropy'),
        ('self-gamma',                     'cifar100_sdrop_rate0.1_L4_sg'),
    ], best, seeds=(0,))

    table('ViT 삽입 위치 (seed 0)', [
        ('L3 + L4 (기본)', 'cifar100_sdrop_vit_rate0.3_L3+L4'),
        ('L3 only',        'cifar100_sdrop_vit_rate0.3_L3'),
        ('모든 층',         'cifar100_sdrop_vit_full_rate0.3_none'),
    ], best, seeds=(0,))

    missing = [n for n in [
        'cifar100_vit_rate0.1_none_seed3','cifar100_vit_rate0.1_none_seed4',
        'cifar100_sdrop_vit_rate0.3_L3+L4_seed3','cifar100_sdrop_vit_rate0.3_L3+L4_seed4',
        'cifar100_sdrop_vit_rate0.3_L3+L4_scorepost_seed0',
        'cifar100_sdrop_energy_rate0.05_L4_seed0','cifar100_sdrop_energy_rate0.2_L4_seed0',
        'cifar100_sdrop_energy_rate0.3_L4_seed0','cifar100_sdrop_energy_rate0.5_L4_seed0',
        'cifar100_sdrop_energy_rate0.1_L3_seed0','cifar100_sdrop_energy_rate0.1_L3+L4_seed0',
        'cifar100_sdrop_rate0.1_L4_nmmean_seed0','cifar100_sdrop_rate0.1_L4_pkentropy_seed0',
        'cifar100_sdrop_rate0.1_L4_sg_seed0',
        'cifar100_sdrop_vit_rate0.3_L3_seed0','cifar100_sdrop_vit_full_rate0.3_none_seed0',
    ] if n not in best]
    print(f'\n## 아직 없는 런: {len(missing)}개')
    for m in missing: print('  -', m)

if __name__ == '__main__':
    main()
