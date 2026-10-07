#!/usr/bin/env python3
"""Neurocomputing revision RUN items (CHECKLIST_point_by_point_2026-10-06.md): CIFAR-100 10% data, paired by seed.
Reads paper_results/raw_pod/*_history.csv, writes 01_NEUCOM_revision/results_RUN_<date>.md.
    python3 scripts/neucom_run_analysis.py
Metrics: final = last-epoch test acc (primary, no test-set selection); best = max over epochs (what the submitted paper reported).
Each arm is compared with random channel dropout at the same layer/rate and with no-dropout, paired t over shared seeds, BH over the table."""
import glob, re, datetime, pathlib, numpy as np, pandas as pd
from scipy import stats
R = pathlib.Path(__file__).resolve().parent.parent / 'paper_results' / 'raw_pod'
OUT = pathlib.Path(__file__).resolve().parents[2] / '01_NEUCOM_revision' / f'results_RUN_{datetime.date.today()}.md'

def runs(stem):
    d = {}
    for f in glob.glob(str(R / f'cifar100_{stem}_seed*_history.csv')):
        s = int(re.search(r'_seed(\d+)_history', f).group(1)); h = pd.read_csv(f)
        if len(h) < 200: continue
        d[s] = (h['acc'].iloc[-1], h['acc'].max(), h['ece'].iloc[-1], h['p_eff'].mean() if 'p_eff' in h else np.nan)
    return d

NONE = 'none_rate0.1_none_sub0.1'
def rand(L='L4', p='0.1'): return f'sdrop_random_rate{p}_{L}_sub0.1_nmmean'
ARMS = [  # (section, label, stem, random-control stem)
 ('Layer sweep (R2-2)', 'SDrop L1', 'sdrop_rate0.1_L1_sub0.1_nmmean', rand('L1')),
 ('Layer sweep (R2-2)', 'SDrop L2', 'sdrop_rate0.1_L2_sub0.1_nmmean', rand('L2')),
 ('Layer sweep (R2-2)', 'SDrop L3', 'sdrop_rate0.1_L3_sub0.1_nmmean', rand('L3')),
 ('Layer sweep (R2-2)', 'SDrop L4 (headline)', 'sdrop_rate0.1_L4_sub0.1_nmmean', rand('L4')),
 ('Mechanism (R3-1)', 'SDrop-Shuffle', 'sdrop_shuffle_rate0.1_L4_sub0.1_nmmean', rand()),
 ('Baselines at L4 (R2-4)', 'SpatialDropout', 'dropout_rate0.1_L4_sub0.1', rand()),
 ('Baselines at L4 (R2-4)', 'DropBlock', 'dropblock_rate0.1_L4_sub0.1', rand()),
 ('Baselines at L4 (R2-4)', 'Targeted Dropout', 'targeted_rate0.1_L4_sub0.1', rand()),
 ('Baselines at L4 (R2-4)', 'Max-Drop (published, p=0.5/position)', 'maxdrop_rate0.5_L4_sub0.1', rand()),
 ('Baselines at L4 (R2-4)', 'Max-Drop channel top-k (budget-matched)', 'maxdrop_ch_rate0.1_L4_sub0.1', rand()),
 ('Baselines at L4 (R2-4)', 'SDrop-Energy, 'sdrop_energy_rate0.1_L4_sub0.1_nmmean', rand()),
 ('Baselines at L4 (R2-4)', 'SDrop-GAP', 'sdrop_gap_rate0.1_L4_sub0.1_nmmean', rand()),
 ('Baselines at L4 (R2-4)', 'SDrop-Peakedness', 'sdrop_peak_rate0.1_L4_sub0.1_nmmean', rand()),
 ('gamma / delta (R2-6)', 'gamma=0.5', 'sdrop_rate0.1_L4_sub0.1_nmmean_g0.5', rand()),
 ('gamma / delta (R2-6)', 'gamma=2', 'sdrop_rate0.1_L4_sub0.1_nmmean_g2', rand()),
 ('gamma / delta (R2-6)', 'delta=0.5', 'sdrop_rate0.1_L4_sub0.1_nmmean_d0.5', rand()),
 ('gamma / delta (R2-6)', 'delta=2', 'sdrop_rate0.1_L4_sub0.1_nmmean_d2', rand()),
] + [('p_base (R2-6)', f'SDrop p={p}', f'sdrop_rate{p}_L4_sub0.1_nmmean', rand(p=p)) for p in ['0.05', '0.1', '0.15', '0.2', '0.3', '0.5']]

def paired(a, b, i):
    s = sorted(set(a) & set(b))
    if len(s) < 2: return len(s), np.nan, np.nan, np.nan, np.nan
    d = np.array([a[k][i] - b[k][i] for k in s]); t, p = stats.ttest_rel([a[k][i] for k in s], [b[k][i] for k in s])
    ci = stats.t.ppf(0.975, len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    return len(s), d.mean(), ci, t, p

none = runs(NONE); rows = []
for sec, lab, stem, rs in ARMS:
    a = runs(stem); r = runs(rs)
    if not a: rows.append(dict(section=sec, arm=lab, n=0)); continue
    row = dict(section=sec, arm=lab, n=len(a), final=np.mean([v[0] for v in a.values()]), best=np.mean([v[1] for v in a.values()]),
               ece=np.mean([v[2] for v in a.values()]), p_eff=np.nanmean([v[3] for v in a.values()]),
               n_rand=len(r), rand_final=np.mean([v[0] for v in r.values()]) if r else np.nan)
    for tag, ctrl in (('rand', r), ('none', none)):
        for mi, m in ((0, 'final'), (1, 'best')):
            n, d, ci, t, p = paired(a, ctrl, mi); row.update({f'{tag}_{m}_n': n, f'{tag}_{m}_d': d, f'{tag}_{m}_ci': ci, f'{tag}_{m}_t': t, f'{tag}_{m}_p': p})
    rows.append(row)
df = pd.DataFrame(rows)
# BH per comparator family over DISTINCT arms (the L4 headline appears in two sections of the table; count it once).
df['stem'] = [a[2] for a in ARMS]
for c in [c for c in df.columns if c.endswith('_p')]:
    m = df[c].notna() & ~df.duplicated('stem'); q = np.full(len(df), np.nan)
    if m.any(): q[m.values] = stats.false_discovery_control(df.loc[m, c].values)
    df[c.replace('_p', '_q')] = q
    for stem in df.loc[df.duplicated('stem'), 'stem']:   # copy the headline's q to its duplicate row
        df.loc[df.stem == stem, c.replace('_p', '_q')] = df.loc[(df.stem == stem) & ~df.duplicated('stem'), c.replace('_p', '_q')].values[0]
FAMILY_M = int((df['rand_final_p'].notna() & ~df.duplicated('stem')).sum())

L = [f'# RUN results, CIFAR-100 10% data, ResNet-18, 200 ep ({datetime.date.today()})', '',
     f'No-dropout reference: n={len(none)}, final {np.mean([v[0] for v in none.values()]):.2f}, best {np.mean([v[1] for v in none.values()]):.2f}.',
     f'Delta = arm minus control, paired by seed; 95% CI; q = BH per comparator family over the {FAMILY_M} distinct arms in this file (the L4 headline is counted once). final = last epoch (primary), best = max over epochs (as submitted).', '']
f = lambda x, k='.2f': '' if pd.isna(x) else format(x, k)
for sec in dict.fromkeys(df.section):
    L += [f'## {sec}', '', '| arm | n | final | best | ECE | p_eff | vs random (final) | p / q | vs random (best) | vs none (final) | p / q |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in df[df.section == sec].iterrows():
        if r.n == 0: L.append(f'| {r.arm} | 0 | pending |  |  |  |  |  |  |  |  |'); continue
        L.append(f"| {r.arm} | {r.n} | {f(r.final)} | {f(r.best)} | {f(r.ece,'.3f')} | {f(r.p_eff,'.3f')} | "
                 f"{f(r.rand_final_d,'+.2f')} ± {f(r.rand_final_ci)} (n={r.rand_final_n}) | {f(r.rand_final_p,'.3f')} / {f(r.rand_final_q,'.3f')} | "
                 f"{f(r.rand_best_d,'+.2f')} ± {f(r.rand_best_ci)} | {f(r.none_final_d,'+.2f')} ± {f(r.none_final_ci)} (n={r.none_final_n}) | {f(r.none_final_p,'.3f')} / {f(r.none_final_q,'.3f')} |")
    L.append('')
OUT.write_text('\n'.join(L)); print('\n'.join(L))
