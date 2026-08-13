#!/usr/bin/env python3
"""
지금까지 나온 롱테일 결과를 한 화면에 정리한다.

    python report.py                # 진행 상황 + Few 그룹 비교 + 필요 시드 수

Few 그룹(30/100 클래스)이 주지표다. 전체 정확도는 tail 이득을 0.30배로 희석한다.
시드 짝지은 차이를 쓰는 이유는 시드가 유일한 무작위원이기 때문이다 — 같은 시드끼리
빼면 초기화·데이터순서 효과가 상쇄되어 표준오차가 크게 줄어든다.
"""
import csv
import glob
import math
import os
import re
import statistics as st
import subprocess
import sys

CKPT = './checkpoints'
BASE = 'none'
ORDER = ['none', 'dropout', 'sdrop_energy', 'sdrop', 'sdrop_energy_nmmean']
LT_RE = re.compile(r'^cifar100_lt_(.+?)_rate([\d.]+)_L4_imb100(_nmmean)?_seed(\d+)$')


def parse(name):
    m = LT_RE.match(name)
    if not m:
        return None
    method, rate, nm, seed = m.groups()
    return (method + ('_nmmean' if nm else ''), float(rate), int(seed))


def load_best():
    """run_id -> best val acc (완료된 런만)."""
    out = {}
    for f in glob.glob(os.path.join(CKPT, 'cifar100_lt_*_history.csv')):
        rows = list(csv.DictReader(open(f)))
        if len(rows) < 200:                     # 200에폭 완주한 것만
            continue
        name = os.path.basename(f).replace('_history.csv', '')
        out[name] = max(float(r['acc']) for r in rows)
    return out


def load_few():
    """run_id -> Few 그룹 정확도. eval_tailgroups.py 가 만든 CSV 를 읽는다."""
    path = 'tailgroups_lt.csv'
    if not os.path.isfile(path):
        return {}
    return {r['run']: float(r['few']) for r in csv.DictReader(open(path))}


def fmt_group(vals):
    if not vals:
        return '—'
    if len(vals) == 1:
        return f'{vals[0]:.2f} (n=1)'
    return f'{st.mean(vals):.2f} ± {st.stdev(vals):.2f} (n={len(vals)})'


def main():
    best, few = load_best(), load_few()

    by_method = {}
    for name, acc in best.items():
        p = parse(name)
        if not p or p[1] != 0.1:
            continue
        by_method.setdefault(p[0], {})[p[2]] = (acc, few.get(name))

    print(f'롱테일 완료: {len(best)}/15 런\n')
    print(f"{'method':<24}{'Few 그룹':>22}{'전체 정확도':>24}")
    print('-' * 70)
    for m in ORDER:
        if m not in by_method:
            continue
        seeds = sorted(by_method[m])
        f = [by_method[m][s][1] for s in seeds if by_method[m][s][1] is not None]
        a = [by_method[m][s][0] for s in seeds]
        print(f'{m:<24}{fmt_group(f):>22}{fmt_group(a):>24}')

    # ---- 시드 짝지은 차이 (baseline 대비) ----
    if BASE in by_method:
        print(f'\n시드 짝지은 차이 (baseline 대비, Few 그룹)')
        print('-' * 70)
        for m in ORDER:
            if m == BASE or m not in by_method:
                continue
            d = []
            for s in sorted(by_method[m]):
                if s in by_method[BASE]:
                    fm, fb = by_method[m][s][1], by_method[BASE][s][1]
                    if fm is not None and fb is not None:
                        d.append(fm - fb)
            if not d:
                continue
            per = '  '.join(f's{i}={x:+.2f}' for i, x in enumerate(d))
            if len(d) > 1:
                sd = st.stdev(d)
                se = sd / math.sqrt(len(d))
                print(f'{m:<24}{st.mean(d):+6.2f} pp   (sd {sd:.2f}, se {se:.2f})   {per}')
                # 이 분산에서 se=0.3pp 를 얻으려면 몇 시드가 필요한가
                need = math.ceil((sd / 0.3) ** 2)
                print(f'{"":24}   -> se 0.30pp 까지 줄이려면 시드 {need}개')
            else:
                print(f'{m:<24}{d[0]:+6.2f} pp   (n=1, 판정 불가)   {per}')

    print('\n' + '-' * 70)
    print('Few 그룹 = 학습 이미지 20장 미만인 30개 클래스. 전체 정확도는 여기서의')
    print('이득을 0.30배로 희석하므로 Few 열로 판정한다.')


if __name__ == '__main__':
    main()
