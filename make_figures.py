#!/usr/bin/env python3
"""
논문용 그림 생성 — CIFAR-100-LT 확정 결과 3종.

    python make_figures.py                 # 전부
    python make_figures.py --no-gpu       # fig1(참여비) 제외, CSV만으로

출력: sdrop-paper/00_ACTIVE_SpringerML/manuscript/fig_*.pdf (+ .png 미리보기)
      및 Desktop/SDROP/overleaf_upload/ 사본.

색은 개체에 고정한다 (모든 그림에서 동일):
    Baseline           #8a8880 (중립 회색)
    SpatialDropout     #eb6834 (주황)
    SDrop max-norm     #1baf7a (아쿠아)
    SDrop mean-norm    #2a78d6 (파랑, 헤드라인)
검증된 categorical 팔레트의 1~3번 슬롯 + 중립색. 시드 오차막대는 ±sd.
"""
import argparse
import csv
import glob
import math
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C_BASE  = '#8a8880'
C_DROP  = '#eb6834'
C_MAX   = '#1baf7a'
C_MEAN  = '#2a78d6'
INK     = '#0b0b0b'
INK2    = '#52514e'
GRID    = '#e5e4e0'

OUT_DIRS = [
    '../sdrop-paper/00_ACTIVE_SpringerML/manuscript',
    '../overleaf_upload',
]

plt.rcParams.update({
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,          # 저널 요구: 임베드 가능 폰트
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True,
})


def save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(d, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}.pdf 저장')


def history(run_id):
    path = f'checkpoints/{run_id}_history.csv'
    if not os.path.isfile(path):
        return None
    return list(csv.DictReader(open(path)))


def seeds_of(pattern):
    """pattern 에 {s} 를 넣어 존재하는 시드들의 history 를 모은다 (200에폭 완주만)."""
    out = {}
    for s in range(10):
        h = history(pattern.format(s=s))
        if h and len(h) >= 200:
            out[s] = h
    return out


# ── fig_effective_rate: 실효 드롭률이 명목값에서 얼마나 벗어나는가 ──────────

def fig_effective_rate():
    max_runs = seeds_of('cifar100_lt_sdrop_energy_rate0.1_L4_imb100_seed{s}')
    mean_runs = seeds_of('cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_seed{s}')

    def peff_matrix(runs):
        cols = []
        for h in runs.values():
            v = [float(r['p_eff']) for r in h if r['p_eff'] not in ('', 'nan')]
            if len(v) >= 200:
                cols.append(v[:200])
        return np.array(cols)

    M, N = peff_matrix(max_runs), peff_matrix(mean_runs)
    ep = np.arange(1, 201)

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axhline(0.1, color=INK2, lw=1.0, ls='--', zorder=1)
    ax.annotate('nominal $p_{\\mathrm{base}}=0.1$', xy=(8, 0.1),
                xytext=(0, 5), textcoords='offset points',
                ha='left', color=INK2, fontsize=8)

    for A, color, label in ((N, C_MEAN, f'mean norm. ({len(N)} seeds)'),
                            (M, C_MAX, f'max norm. ({len(M)} seeds)')):
        if len(A) == 0:
            continue
        mu = A.mean(0)
        ax.fill_between(ep, A.min(0), A.max(0), color=color, alpha=0.18, lw=0)
        ax.plot(ep, mu, color=color, lw=2.0, label=label)

    # 직접 라벨 (선 끝)
    if len(N):
        ax.annotate('mean', xy=(200, N.mean(0)[-1]), xytext=(4, 0),
                    textcoords='offset points', color=C_MEAN, fontsize=8, va='center')
    if len(M):
        ax.annotate('max', xy=(200, M.mean(0)[-1]), xytext=(4, 0),
                    textcoords='offset points', color=C_MAX, fontsize=8, va='center')

    ax.set_xlabel('epoch')
    ax.set_ylabel('effective drop rate $\\bar{p}$')
    ax.set_xlim(1, 218)
    ax.set_ylim(0, 0.115)
    ax.legend(frameon=False, loc='center right')
    save(fig, 'fig_effective_rate')


# ── fig_rate_matched: 실효율 표기와 함께 보는 방법별 정확도 ────────────────

def fig_rate_matched():
    conds = [
        ('Baseline',            'cifar100_lt_none_rate0.1_L4_imb100_seed{s}',                C_BASE, '—'),
        ('SpatialDropout',      'cifar100_lt_dropout_rate0.1_L4_imb100_seed{s}',             C_DROP, '0.100'),
        ('SDrop\n(max norm.)',  'cifar100_lt_sdrop_energy_rate0.1_L4_imb100_seed{s}',        C_MAX,  '0.017'),
        ('SDrop\n(mean norm.)', 'cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_seed{s}', C_MEAN, '0.100'),
    ]
    stats = []
    for label, pat, color, peff in conds:
        runs = seeds_of(pat)
        best = [max(float(r['acc']) for r in h) for h in runs.values()]
        stats.append((label, st.mean(best), st.stdev(best) if len(best) > 1 else 0.0,
                      len(best), color, peff))

    # 점 추정 + 오차막대. 잘린 축에서 막대는 길이 왜곡을 만들므로 쓰지 않는다.
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(stats))
    for i, (label, mu, sd, n, color, peff) in enumerate(stats):
        ax.errorbar(i, mu, yerr=sd, color=color, lw=1.4, capsize=4,
                    marker='o', ms=8, mfc=color, mec='white', mew=1.2, zorder=4)
        ax.annotate(f'{mu:.2f}', xy=(i, mu), xytext=(10, 0),
                    textcoords='offset points', ha='left', va='center',
                    color=INK, fontsize=8)
    labels = [f'{s[0]}\n$\\bar{{p}}$={s[5]}' + f'\n({s[3]} seeds)' for s in stats]
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, len(stats) - 0.3)
    ax.set_ylabel('best val. accuracy (%)')
    save(fig, 'fig_rate_matched')


# ── fig_participation: 클래스 그룹별 실효 채널 수 (GPU 필요) ───────────────

def fig_participation():
    import torch
    from dataset import get_dataset
    from eval_channel_usage import participation, class_groups

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, n_cls, _, _ = get_dataset('cifar100', './data', 256, 0)
    grp = class_groups(n_cls)

    def collect(glob_pat):
        out = []
        for f in sorted(glob.glob(glob_pat)):
            if os.path.isfile(f.replace('_best.pth', '_history.csv')):
                out.append(participation(f, val_loader, device, n_cls))
        return out

    base = collect('checkpoints/cifar100_lt_none_rate0.1_L4_imb100_seed*_best.pth')
    sdrp = collect('checkpoints/cifar100_lt_sdrop_energy_rate0.1_L4_imb100_nmmean_seed*_best.pth')
    n = min(len(base), len(sdrp))

    # 점 추정 + 오차막대 (막대 대신 — 축이 0에서 시작하지 않으므로)
    groups = ['Many', 'Medium', 'Few']
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    w = 0.18
    x = np.arange(3)
    for off, runs, color, label, dx in ((-w, base, C_BASE, f'Baseline ({n} seeds)', -10),
                                        (w, sdrp, C_MEAN, f'SDrop ({n} seeds)', 10)):
        mus, sds = [], []
        for g in groups:
            vals = [r[grp == g].mean() for r in runs[:n]]
            mus.append(st.mean(vals))
            sds.append(st.stdev(vals) if n > 1 else 0.0)
        ax.errorbar(x + off, mus, yerr=sds, color=color, lw=1.4, capsize=4,
                    marker='o', ms=8, mfc=color, mec='white', mew=1.2,
                    ls='none', label=label, zorder=4)
        # 값 라벨은 오차막대를 피해 좌/우 옆에 단다
        for xi, mu in zip(x + off, mus):
            ax.annotate(f'{mu:.0f}', xy=(xi, mu), xytext=(dx, 0),
                        textcoords='offset points', va='center', color=INK,
                        ha=('right' if dx < 0 else 'left'), fontsize=8)

    counts = {'Many': '>100', 'Medium': '20–100', 'Few': '<20'}
    ax.set_xticks(x, [f'{g}\n({counts[g]} imgs/class)' for g in groups])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('effective channels per class\n(participation ratio, L4)')
    ax.legend(frameon=False, loc='upper left')
    save(fig, 'fig_participation')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--no-gpu', action='store_true',
                   help='참여비 그림(fig_participation) 생략')
    args = p.parse_args()

    print('그림 생성:')
    fig_effective_rate()
    fig_rate_matched()
    if not args.no_gpu:
        fig_participation()
    print('완료. 출력 위치:', ', '.join(OUT_DIRS))


if __name__ == '__main__':
    main()
