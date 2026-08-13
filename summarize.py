#!/usr/bin/env python3
"""Collect *_history.csv from downloaded Kaggle outputs and summarise them."""
import sys, csv, glob, os, re, collections, statistics as st

root = sys.argv[1] if len(sys.argv) > 1 else './checkpoints'
rows = {}

for f in glob.glob(os.path.join(root, '**', '*_history.csv'), recursive=True):
    name = os.path.basename(f).replace('_history.csv', '')
    try:
        with open(f) as fh:
            r = list(csv.DictReader(fh))
        if not r:
            continue
        acc = [float(x['acc']) for x in r]
        peff = [float(x['p_eff']) for x in r
                if x.get('p_eff') not in (None, '', 'nan')]
        rows[name] = (max(acc), acc[-1],
                      sum(peff) / len(peff) if peff else float('nan'), len(r))
    except Exception as e:
        print('  파싱 실패', name, '->', e)

if not rows:
    print('  history.csv를 찾지 못했습니다. 학습이 실제로 돌았는지 로그를 확인하세요.')
    sys.exit(0)

print(f"\n{'run':<58}{'best':>8}{'final':>8}{'p_eff':>8}{'ep':>5}")
print('-' * 87)
for k in sorted(rows):
    b, f, p, n = rows[k]
    print(f"{k:<58}{b:8.2f}{f:8.2f}{p:8.4f}{n:5d}")

grp = collections.defaultdict(list)
for k, (b, *_ ) in rows.items():
    grp[re.sub(r'_seed\d+$', '', k)].append(b)

print(f"\n방법별 집계 (best val acc)\n{'-'*66}")
for k in sorted(grp):
    v = grp[k]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {k:<48}{sum(v)/len(v):7.2f} ± {sd:4.2f}  (n={len(v)})")
