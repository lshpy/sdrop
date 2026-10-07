"""Benjamini-Hochberg FDR correction over every hypothesis test reported in main_article.tex.

For each test: recompute the p-value from the per-seed run files with the test the paper states
(seed-paired two-sided t-test unless noted), keep the reported p where the data are not on disk,
then apply BH (q=0.05, 0.10) across the family and Holm across the headline claims.

Family F1 = every comparison for which the manuscript prints a p-value, a t statistic or a CI.
Family F2 = F1 + comparisons the text calls significant / not significant or backs with a win
count only ("6/6 seeds") and that can be recomputed; reported as a sensitivity check.

v2 (2026-09-29, after the seed top-up, sdrop_pod_code/TOPUP_PLAN.md; v1 kept as scripts/bh_fdr_v1.py):
  * every test uses all seeds on disk. Each pattern 'checkpoints_X/...' is read from the volume copy
    paper_results/raw_pod/checkpoints_X/ first, then the older local mirror paper_results/checkpoints_X/
    ('raw_pod/...' = volume ./checkpoints: raw_pod/checkpoints/ then flat raw_pod/); one file per run id.
    Top-up runs whose history was not downloaded (NICO++ checkpoints_screen) fall back to the best-val
    [test] line of their queue log (raw_pod/logs/run_topup_*_queue.log), flagged in the note.
  * families per prereg/PREREG_addendum_confirmatory_family_2026-09-29.md: 'confirmatory' (WB headline 14 seeds,
    ViT-B T6, CCT20, ISIC primary; screening datasets via SCREEN below) and 'exploratory' (everything else,
    incl. the prereg secondary tests from scripts/score_prereg.py and scripts/field_posthoc.py); BH within each.
  * Spawrious and CUB top-ups are still running: those tests are pinned to seeds 0-2 (marked pending).

Run: python3 scripts/bh_fdr.py                  -> paper_results/bh_fdr_tests_v2.csv
     python3 scripts/bh_fdr.py --before-topup   -> same tests without the top-up runs (bh_fdr_tests_v2_before_topup.csv)
     python3 scripts/topup_compare.py           -> paper_results/topup_comparison.md
"""
import contextlib, fnmatch, glob, io, math, re, runpy, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PR = ROOT / 'paper_results'
BEFORE = '--before-topup' in sys.argv
OUT = PR / ('bh_fdr_tests_v2_before_topup.csv' if BEFORE else 'bh_fdr_tests_v2.csv')


# ---------------------------------------------------------------- loaders
def _seed(f): return int(re.search(r'_seed(\d+)_', Path(f).name).group(1))
def _rid(name): return re.sub(r'_(history\.csv|dump\.npz|gradclean\.csv)$', '', name)

# top-up runs: (volume folder, run id) -> best-val test metrics from the queue log
TOPUP = {}
for _f in glob.glob(str(PR / 'raw_pod/logs/run_topup_*_queue.log')):
    for m in re.finditer(r'\[test\] best-val epoch \d+: test_acc=([\d.]+)%.*?test_worst_group=([\d.]+).*?\n(?:.*\n){0,3}?'
                         r'Training history saved: (\S+)_history\.csv', open(_f, errors='ignore').read()):
        d, n = m[3].rsplit('/', 1); TOPUP[d, n] = {'acc': float(m[1]), 'wg': float(m[2])}
FROM_LOG = set()   # run ids whose value came from a queue log

def _vol(d): return 'checkpoints' if d == 'raw_pod' else d

def files(pattern):
    """Files matching 'dir/glob', volume copy first, one per file name. --before-topup drops top-up runs."""
    d, b = pattern.rsplit('/', 1)
    dirs = ['raw_pod/checkpoints', 'raw_pod'] if d == 'raw_pod' else [f'raw_pod/{d}', d] if d.startswith('checkpoints') else [d]
    out = {}
    for dd in dirs:
        for f in sorted(glob.glob(str(PR / dd / b))):
            n = Path(f).name
            if not (BEFORE and (_vol(d), _rid(n)) in TOPUP): out.setdefault(n, f)
    return sorted(out.values())

def runs(pattern, metric='wg', seeds=None):
    """seed -> value from *_history.csv files. The last row repeats the best-validation checkpoint's
    test metrics (test_worst_group / test_acc). metric: wg | acc (best-val test acc) |
    final (last-epoch 'acc', CUB) | besttest (max 'acc' over epochs, CIFAR exploratory protocol) |
    cmconf (Colored-MNIST bias-conflicting accuracy). seeds: keep only these."""
    out = {}
    for f in files(pattern):
        if not f.endswith('_history.csv'): continue
        h = pd.read_csv(f); last = h.iloc[-1]
        v = {'wg': lambda: last['test_worst_group'], 'acc': lambda: last['test_acc'],
             'final': lambda: last['acc'], 'besttest': lambda: h['acc'].max(), 'cmconf': lambda: cm_conflict(last)}[metric]()
        out[_seed(f)] = float(v)
    d, b = pattern.rsplit('/', 1)
    if not BEFORE and metric in ('wg', 'acc'):   # top-up runs finished on the pod but history not downloaded
        for (vd, rid), m in TOPUP.items():
            if vd == _vol(d) and fnmatch.fnmatch(rid + '_history.csv', b) and _seed(rid + '_') not in out:
                out[_seed(rid + '_')] = m[metric]; FROM_LOG.add(rid)
    if seeds is not None: out = {k: v for k, v in out.items() if k in seeds}
    if not out: raise FileNotFoundError(pattern)
    return out

def merge(*ds):
    """Union over seeds; a seed present twice must carry the same value (rev2a/rev2d hold identical copies)."""
    out = {}
    for d in ds:
        for k, v in d.items():
            assert k not in out or abs(out[k] - v) < 1e-9, f'seed {k} differs between folders'
            out[k] = v
    return out

# Colored-MNIST bias-conflicting accuracy (colour != class): per-cell test accuracies in the history weighted by the
# cell sizes of the test set, which is identical in every run (checked on all 27 local dumps). Equals the accuracy
# computed from the dumps' test logits (checked on seeds 0-2).
_cmd = np.load(PR / 'checkpoints_cb2/cmnist_0.95_none_rate0.1_none_ep20_seed0_dump.npz')
CM_N = {(i, j): int(((_cmd['test_y'] == i) & (_cmd['test_places'] == j)).sum()) for i in range(10) for j in range(10) if i != j}
def cm_conflict(last):
    return np.average([last[f'test_acc_g{i}{j}'] for i, j in CM_N], weights=list(CM_N.values()))

def csvcol(path, arm_col, arm, val_col, where=None):
    d = pd.read_csv(PR / path)
    if where is not None: d = d[where(d)]
    d = d[d[arm_col] == arm]
    return {int(s): float(v) for s, v in zip(d.seed, d[val_col])}


# ---------------------------------------------------------------- tests
def paired(a, b):
    """a - b over shared seeds: (delta, wins, n, p two-sided paired t, p exact sign test)."""
    s = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in s])
    return one_sample(d)

def one_sample(d):
    d = np.asarray(d, float); n = len(d)
    p = stats.ttest_1samp(d, 0).pvalue if n > 1 and d.std() > 0 else float('nan')
    nz = d[d != 0]
    ps = stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue if len(nz) else float('nan')
    return d.mean(), int((d > 0).sum()), n, float(p), float(ps)


# LA frontier (same code as sdrop_pod_code/analyze_frontier.py, rule C = interpolated frontier)
def _wg(logits, y, p, tau, lp):
    pred = (logits - tau * lp).argmax(1); g = y * 2 + p
    return min(100 * (pred[g == k] == y[g == k]).mean() for k in range(4)), 100 * (pred == y).mean()

def _load(f):
    d = np.load(f); pi1 = d['val_y'].mean()
    return d['test_logits'], d['test_y'], d['test_places'], np.log(np.array([1 - pi1, pi1]))

TAUS = np.round(np.arange(0, 3.0 + 1e-9, 0.05), 2)

def la_frontier(arm_glob, erm_glob='checkpoints_rev/waterbirds_none_rate0.1_none_ep30_seed*_dump.npz'):
    """per-seed (rule C delta, best-point delta) of arm vs ERM+LA of the same seed."""
    erm = {_seed(f.replace('_dump.npz', '_x.npz')): _load(f) for f in files(erm_glob)}
    arm = {_seed(f.replace('_dump.npz', '_x.npz')): _load(f) for f in files(arm_glob)}
    dc, db = [], []
    for s in sorted(set(erm) & set(arm)):
        L, y, p, lp = erm[s]; fr = np.array([_wg(L, y, p, t, lp) for t in TAUS])
        La, ya, pa, lpa = arm[s]; w0, a0 = _wg(La, ya, pa, 0.0, lpa)
        o = np.argsort(fr[:, 1]); dc.append(w0 - np.interp(a0, fr[o, 1], fr[o, 0]))
        fa = np.array([_wg(La, ya, pa, t, lpa) for t in TAUS]); db.append(fa[:, 0].max() - fr[:, 0].max())
    return np.array(dc), np.array(db)

def vit_la():
    """scripts/analyze_vit_frontier.py: ViT MLP-GAP (raw_pod histories) vs ERM+LA (vitt dumps),
    frontier point = best wg among tau with |avg - arm avg| <= 0.5, else nearest."""
    erm = {_seed(f.replace('_dump.npz', '_x.npz')): np.array([_wg(*_load(f)[:3], t, _load(f)[3]) for t in TAUS])
           for f in files('checkpoints_vitt/waterbirds_vitb_none_*_dump.npz')}
    d = []
    for f in files('raw_pod/waterbirds_vitbm_sdrop_gap_*_history.csv'):
        s = _seed(f); h = pd.read_csv(f); r = h[h.epoch == int(h.best_epoch.iloc[-1])].iloc[0]
        if s not in erm: continue
        a, w = erm[s][:, 1], erm[s][:, 0]; near = np.abs(a - r.test_acc) <= 0.5
        d.append(r.test_worst_group - (w[near].max() if near.any() else w[np.argmin(np.abs(a - r.test_acc))]))
    return np.array(d)


# ---------------------------------------------------------------- data
W = lambda pat, m='wg': runs(pat, m)
wb = {k: W(f'raw_pod/waterbirds_{v}_seed*') for k, v in {
    'erm': 'none_rate0.1_none_ep30', 'random': 'sdrop_random_rate0.1_L4_ep30_nmmean',
    'shuffle': 'sdrop_shuffle_rate0.1_L4_ep30_nmmean', 'spatial': 'spatialdropout_rate0.1_L4_ep30',
    'dropblock': 'dropblock_rate0.1_L4_ep30', 'wcd': 'wcd_rate0.1_L4_ep30',
    'gd_topk_alt': 'guided_topk_rate0.1_L4_ep30_schgdtopk', 'gd_dr_alt': 'guided_dr_rate0.1_L4_ep30_schgdtopk',
    'gd_topk_const': 'guided_topk_rate0.1_L4_ep30', 'gd_dr_const': 'guided_dr_rate0.1_L4_ep30',
    'gap': 'sdrop_gap_rate0.1_L4_ep30_nmmean', 'energy': 'sdrop_energy_rate0.1_L4_ep30_nmmean',
    'egpg': 'sdrop_rate0.1_L4_ep30_nmmean'}.items()}
wb['jtt'] = W('checkpoints_jtt/waterbirds_none_rate0.1_none_ep30_jttT2L50_seed*')
wb_acc = {k: W(f'raw_pod/waterbirds_{v}_seed*', 'acc') for k, v in
          {'erm': 'none_rate0.1_none_ep30', 'energy': 'sdrop_energy_rate0.1_L4_ep30_nmmean'}.items()}
mech = {k: W(f'checkpoints_{v}_seed*') for k, v in {
    'reverse': 'mech/waterbirds_sdrop_reverse_rate0.1_L4_ep30_nmmean',
    'early': 'mech/waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_schearly',
    'late': 'mech/waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_wu0.5',
    'p05': 'mech/waterbirds_sdrop_energy_rate0.05_L4_ep30_nmmean',
    'p2': 'mech/waterbirds_sdrop_energy_rate0.2_L4_ep30_nmmean',
    'p3': 'final/waterbirds_sdrop_energy_rate0.3_L4_ep30_nmmean',
    'rand_p3': 'mech/waterbirds_sdrop_random_rate0.3_L4_ep30_nmmean',
    'shuf_p3': 'final/waterbirds_sdrop_shuffle_rate0.3_L4_ep30_nmmean',
    'topk': 'final/waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk',
    'shuf_topk': 'final/waterbirds_sdrop_shuffle_rate0.1_L4_ep30_nmmean_seltopk',
    'cross': 'mech2/waterbirds_sdrop_cross_rate0.1_L4_ep30_nmmean'}.items()}
ext = lambda arm: merge(W(f'raw_pod/waterbirds_{arm}_seed*'), W(f'checkpoints_rev2b/waterbirds_{arm}_seed*'),
                        W(f'checkpoints_rev2c/waterbirds_{arm}_seed*'))
wb13 = {'erm': ext('none_rate0.1_none_ep30'), 'energy': ext('sdrop_energy_rate0.1_L4_ep30_nmmean'),
        'random': ext('sdrop_random_rate0.1_L4_ep30_nmmean')}
rwy = {k: merge(W(f'checkpoints_rev2a/waterbirds_{v}_rwy_seed*'), W(f'checkpoints_rev2d/waterbirds_{v}_rwy_seed*'))
       for k, v in {'erm': 'none_rate0.1_none_ep30', 'random': 'sdrop_random_rate0.1_L4_ep30_nmmean',
                    'energy': 'sdrop_energy_rate0.1_L4_ep30_nmmean'}.items()}
corr = {(c, a): W(f'checkpoints_sweep/waterbirds_{v}_corr{c}_seed*') for c in ('0.5', '0.75', '0.9')
        for a, v in {'erm': 'none_rate0.1_none_ep30', 'energy': 'sdrop_energy_rate0.1_L4_ep30_nmmean',
                     'random': 'sdrop_random_rate0.1_L4_ep30_nmmean'}.items()}
cb = {'erm': W('checkpoints_sweep/waterbirds_none_rate0.1_none_ep30_cb_seed*'),
      'energy': W('checkpoints_sweep/waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_cb_seed*')}
cel = {'erm': W('checkpoints_rev/celeba_none_rate0.1_none_ep8_seed*'),
       'gap': W('checkpoints_rev/celeba_sdrop_gap_rate0.1_L4_ep8_nmmean_seed*'),
       'energy': merge(W('checkpoints_extra/celeba_sdrop_energy_rate0.1_L4_ep8_nmmean_seed*'),
                       W('checkpoints_rev/celeba_sdrop_energy_rate0.1_L4_ep8_nmmean_seed*')),
       'random': merge(W('raw_pod/celeba_sdrop_random_rate0.1_L4_ep8_nmmean_seed*'),
                       W('checkpoints_more/celeba_sdrop_random_rate0.1_L4_ep8_nmmean_seed*')),
       'shuffle': merge(W('raw_pod/celeba_sdrop_shuffle_rate0.1_L4_ep8_nmmean_seed*'),
                        W('checkpoints_more/celeba_sdrop_shuffle_rate0.1_L4_ep8_nmmean_seed*'))}
vb = lambda v: W(f'raw_pod/waterbirds_{v}_L4_vit_b_16_ep30_nmmean_sg_seed*')
vit = {'erm': W('raw_pod/waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_sg_seed*'),
       'erm2': W('checkpoints_vitt/waterbirds_vitb_none_rate0.1_L4_vit_b_16_ep30_sg_seed*'),
       **{f'res_{a}': vb(f'vitb_sdrop_{a}_rate0.1') for a in ('random', 'shuffle', 'energy', 'gap')},
       **{f'mlp_{a}': vb(f'vitbm_sdrop_{a}_rate0.1') for a in ('random', 'shuffle', 'energy', 'gap')},
       **{f'tok_{a}': W(f'checkpoints_vitt/waterbirds_vitbt_sdrop_{a}_rate0.1_L4_vit_b_16_ep30_nmmean_sg_seed*')
          for a in ('random', 'energy', 'gap')},
       **{f'mlp_{a}_p{r}': W(f'checkpoints_vitd/waterbirds_vitbm_sdrop_{a}_rate0.{r}_L4_vit_b_16_ep30_nmmean_sg_seed*')
          for a, r in (('gap', 2), ('gap', 3), ('energy', 3), ('random', 3))}}
PENDING = {0, 1, 2}   # Spawrious and CUB top-ups still running on 2026-09-29: pinned to the pre-top-up seeds
spw = {(s, a): runs(f'checkpoints_spw/spawrious_o2o_{s}_{v}_ep10*seed*', seeds=PENDING) for s in ('easy', 'hard')
       for a, v in {'erm': 'none_rate0.1_none', 'energy': 'sdrop_energy_rate0.1_L4', 'random': 'sdrop_random_rate0.1_L4'}.items()}
cub = {k: runs(f'raw_pod/cub200_{v}_rate0.1_L3_ep60_nmmean_seed*', 'final', seeds=PENDING) for k, v in
       {'egpg': 'sdrop', 'shuffle': 'sdrop_shuffle'}.items()}
rel = {a: csvcol('analysis/reliance_2026-09-24.csv', 'arm', a, 'conflict_follow_bg') for a in
       ('ERM', 'Energy', 'Random', 'Energy top-k', 'Shuffled top-k', 'Energy p0.3', 'Shuffled p0.3')}
if not BEFORE:
    # TOPUP_PLAN A1: seeds 0-5 of the headline arms from the pod (checkpoints_mech2 weights), relabelled as in the
    # hand-edited 9/24 CSV, with conflict_follow_bg = 2*swap_follow_bg - swap_follow_bg_minority. Seeds 0-2 must
    # reproduce the 9/24 file (same weights, seeded stimuli).
    r6 = pd.read_csv(PR / 'raw_pod/analysis/reliance_headline6_topup.csv')
    r6['arm'] = r6.arm.map({'none_rate0.1_none_ep30': 'ERM', 'sdrop_energy_rate0.1_L4_ep30_nmmean': 'Energy',
                            'sdrop_random_rate0.1_L4_ep30_nmmean': 'Random'})
    r6['conflict_follow_bg'] = 2 * r6.swap_follow_bg - r6.swap_follow_bg_minority
    for a in ('ERM', 'Energy', 'Random'):
        new = {int(s): float(v) for s, v in zip(r6[r6.arm == a].seed, r6[r6.arm == a].conflict_follow_bg)}
        assert all(abs(new[k] - v) < 1e-6 for k, v in rel[a].items()), f'reliance {a}: seeds 0-2 not reproduced'
        rel[a] = new
    old = pd.read_csv(PR / 'analysis/reliance_2026-09-24.csv')
    pd.concat([old[~old.arm.isin(['ERM', 'Energy', 'Random'])], r6]).to_csv(PR / 'analysis/reliance_2026-09-29_topup.csv', index=False)
afr = {(a, c): csvcol('checkpoints_rev/waterbirds_afr.csv', 'arm', a, c)
       for a in ('none', 'sd', 'sdrop_energy', 'sdrop_gap') for c in ('afr_wg', 'afro_wg')}
dfr = {a: csvcol('analysis/wb_dfr_2026-09-24.csv', 'arm', a, 'dfr_wg') for a in ('none', 'sdrop_energy')}
def grad_share(f):
    """sdrop_pod_code/analyze_gradclean.py: share of squared grad norm on the top-10% channels by epoch-1 energy,
    at the last epoch, mask-free passes. (checkpoints_mech3/gradclean_summary.csv is stale: one seed only.)"""
    d = pd.read_csv(f); e1 = d[d.epoch == 1]; dom = set(e1.sort_values('energy_mean', ascending=False).channel.values[: len(e1) // 10])
    g = d[d.epoch == d.epoch.max()]; return g[g.channel.isin(dom)].grad_sq_sum.sum() / g.grad_sq_sum.sum()
grad = {}
for f in files('checkpoints_mech3/waterbirds_*_gradclean.csv'):
    grad.setdefault(re.match(r'waterbirds_(.+?)_rate', Path(f).name).group(1), {})[_seed(f)] = grad_share(f)
C = lambda v, m='besttest', d='raw_pod': W(f'{d}/cifar100_{v}_seed*', m)
cif = {'none100': C('none_rate0.1_none', d='raw'), 'energy100': C('sdrop_energy_rate0.1_L4', d='raw'),
       'rand100': C('sdrop_random_rate0.1_L4_nmmean'), 'sdrop100': C('sdrop_rate0.1_L4_nmmean'),
       **{f'{a}{f}': C(f'{v}_L4_sub{f}_nmmean') for f in ('0.1', '0.05', '0.025')
          for a, v in (('rand', 'sdrop_random_rate0.1'), ('sdrop', 'sdrop_rate0.1'))},
       **{f'val_{a}': C(v, 'acc') for a, v in {
           'none': 'none_rate0.1_L4_sub0.1_val0.1', 'rand': 'sdrop_random_rate0.1_L4_sub0.1_val0.1_nmmean',
           'gap': 'sdrop_gap_rate0.1_L4_sub0.1_val0.1_nmmean', 'sdrop': 'sdrop_rate0.1_L4_sub0.1_val0.1_nmmean'}.items()}}
other = {k: W(f'checkpoints_{v}_seed*') for k, v in {
    'ms_erm': 'metashift/metashift_none_rate0.1_none_ep30', 'ms_energy': 'metashift/metashift_sdrop_energy_rate0.1_L4_ep30_nmmean',
    'ms_random': 'metashift/metashift_sdrop_random_rate0.1_L4_ep30_nmmean',
    'lion_erm': 'screen/nicopp_lion-kangaroo_grass-water_none_rate0.1_none_ep30',
    'lion_energy': 'screen/nicopp_lion-kangaroo_grass-water_sdrop_energy_rate0.1_L4_ep30_nmmean',
    'lion_random': 'screen/nicopp_lion-kangaroo_grass-water_sdrop_random_rate0.1_L4_ep30_nmmean',
    'sheep_erm': 'screen/nicopp_sheep-kangaroo_grass-water_none_rate0.1_none_ep30',
    'sheep_energy': 'screen/nicopp_sheep-kangaroo_grass-water_sdrop_energy_rate0.1_L4_ep30_nmmean',
    'sheep_random': 'screen/nicopp_sheep-kangaroo_grass-water_sdrop_random_rate0.1_L4_ep30_nmmean',
    'ele_erm': 'screen/nicopp_none_rate0.1_none_ep30', 'ele_energy': 'screen/nicopp_sdrop_energy_rate0.1_L4_ep30_nmmean',
    'ele_random': 'screen/nicopp_sdrop_random_rate0.1_L4_ep30_nmmean',
    'bar_erm': 'cb2/bar_none_rate0.1_none_ep30', 'bar_energy': 'cb2/bar_sdrop_energy_rate0.1_L4_ep30_nmmean',
    'bar_random': 'cb2/bar_sdrop_random_rate0.1_L4_ep30_nmmean'}.items()}
other.update({k: W(f'checkpoints_{v}_seed*') for k, v in {
    'ms_shuffle': 'metashift/metashift_sdrop_shuffle_rate0.1_L4_ep30_nmmean',
    'ele_shuffle': 'screen/nicopp_sdrop_shuffle_rate0.1_L4_ep30_nmmean',
    'bar_shuffle': 'cb2/bar_sdrop_shuffle_rate0.1_L4_ep30_nmmean'}.items()})
# cells that were not in v1 but were topped up (TOPUP_PLAN #3, #4, #7, #8, #17)
l3 = {k: W(f'raw_pod/waterbirds_{v}_rate0.1_L3_ep30_nmmean{sg}_seed*') for k, v, sg in [
    ('egpg', 'sdrop', ''), ('gap', 'sdrop_gap', ''), ('random', 'sdrop_random', ''), ('energy', 'sdrop_energy', ''),
    ('shuffle', 'sdrop_shuffle', ''), ('energy_sg', 'sdrop_energy', '_sg'), ('egpg_sg', 'sdrop', '_sg')]}
W0 = lambda pat: W(pat) if any(f.endswith('_history.csv') for f in files(pat)) else {}   # folder may hold no run (yet)
wd = {(w, a): merge(W0(f'checkpoints_rev2a/waterbirds_{v}_wd{w}_seed*'), W0(f'checkpoints_rev2d/waterbirds_{v}_wd{w}_seed*'))
      for w in ('1e-3', '1e-2', '1e-1') for a, v in {'erm': 'none_rate0.1_none_ep30', 'energy': 'sdrop_energy_rate0.1_L4_ep30_nmmean'}.items()}
scr = {k: W(f'checkpoints_scratch/waterbirds_{v}_seed*') for k, v in {
    'erm': 'none_rate0.1_none_ep30_scratch', 'energy': 'sdrop_energy_rate0.1_L4_ep30_scratch_nmmean',
    'random': 'sdrop_random_rate0.1_L4_ep30_scratch_nmmean'}.items()}
cel['erm_cb'] = W('checkpoints_extra/celeba_none_rate0.1_none_ep8_cb_seed*')
cm = {(b, a): W(f'checkpoints_cb2/cmnist_{b}_{v}_ep20*_seed*', 'cmconf') for b in ('0.95', '0.98', '0.995')
      for a, v in {'erm': 'none_rate0.1_none', 'energy': 'sdrop_energy_rate0.1_L4', 'random': 'sdrop_random_rate0.1_L4'}.items()}
la = {k: la_frontier(g) for k, g in {
    'gap': 'checkpoints_rev/waterbirds_sdrop_gap_rate0.1_L4_ep30_nmmean_seed*_dump.npz',
    'energy': 'checkpoints_rev/waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seed*_dump.npz',
    'topk': 'checkpoints_final/waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed*_dump.npz',
    'p3': 'checkpoints_final/waterbirds_sdrop_energy_rate0.3_L4_ep30_nmmean_seed*_dump.npz'}.items()}
# CelebA GAP vs ERM+LA: per-seed deltas as logged in analysis/celeba_la_frontier_2026-09-15.txt (no rerun here)
cel_la_gap = np.array([+2.20, +12.09, +3.30, -11.54, -2.20, +0.55])


# ---------------------------------------------------------------- the family
T = []
def add(id, loc, comp, rep_p, rep_d=None, rep_w=None, res=None, test='paired t (two-sided)', headline='no',
        family='F1', p_rec=None, note='', conf=False):
    """rep_p: reported p as printed ('0.002', '<0.001', '<=0.002', 't=3.8', 'CI', '' if none)."""
    d = w = n = ps = None
    if res is not None: d, w, n, p_rec, ps = res
    T.append(dict(id=id, location=loc, comparison=comp, test=test, n=n, delta_reported=rep_d, delta_recomputed=d,
                  wins_reported=rep_w, wins_recomputed=None if w is None else f'{w}/{n}', p_reported=rep_p,
                  p_recomputed=p_rec, p_signtest=ps, headline=headline, v1_family=family,
                  family='confirmatory' if conf else 'exploratory', note=note))

P = paired
# --- Waterbirds main table (tab:waterbirds L784-799; text L821-842)
add('WB-01', 'tab:waterbirds L785; L191', 'WB Random vs ERM, wg, 6 seeds', '0.853', -0.21, '3/6', P(wb['random'], wb['erm']), headline='null (abstract "does nothing")')
add('WB-02', 'tab:waterbirds L786; L842', 'WB Shuffled vs ERM, wg, 6 seeds', '0.359', 0.49, '5/6', P(wb['shuffle'], wb['erm']), headline='null (abstract "does nothing")')
add('WB-03', 'tab:waterbirds L788', 'WB SpatialDropout vs ERM, wg', '0.908', 0.08, '3/6', P(wb['spatial'], wb['erm']))
add('WB-04', 'tab:waterbirds L789', 'WB DropBlock vs ERM, wg', '0.910', -0.10, '3/6', P(wb['dropblock'], wb['erm']))
add('WB-05', 'tab:waterbirds L790', 'WB WCD vs ERM, wg', '0.692', 0.29, '3/6', P(wb['wcd'], wb['erm']))
add('WB-06', 'tab:waterbirds L791; L832 (smallest p=0.42)', 'WB Guided top-k alternating vs ERM, wg', '0.423', 0.57, '3/6', P(wb['gd_topk_alt'], wb['erm']))
add('WB-07', 'tab:waterbirds L792', 'WB Guided DR alternating vs ERM, wg', '0.931', -0.08, '4/6', P(wb['gd_dr_alt'], wb['erm']))
add('WB-08', 'tab:waterbirds L793; L836', 'WB Guided top-k constant vs ERM, wg, 4 seeds', '0.021', -10.40, '0/4', P(wb['gd_topk_const'], wb['erm']))
add('WB-09', 'tab:waterbirds L794', 'WB Guided DR constant vs ERM, wg, 3 seeds', '0.652', -0.78, '1/3', P(wb['gd_dr_const'], wb['erm']))
add('WB-10', 'tab:waterbirds L795', 'WB JTT (T=2, lambda=50) vs ERM, wg', '0.973', -0.05, '1/6', P(wb['jtt'], wb['erm']))
add('WB-11', 'tab:waterbirds L797; L824', 'WB GAP vs ERM, wg', '0.003', 6.23, '6/6', P(wb['gap'], wb['erm']))
add('WB-12', 'tab:waterbirds L798; L823; tab:mech L1248; tab:corr L1099; abstract L80; intro L188; concl L1908', 'WB Energy vs ERM, wg, 6 seeds (headline)', '0.001', 5.50, '6/6', P(wb['energy'], wb['erm']), headline='yes')
add('WB-13', 'tab:waterbirds L799; L825', 'WB Full EGPG vs ERM, wg', '0.002', 5.66, '6/6', P(wb['egpg'], wb['erm']))
add('WB-14', 'L826', 'WB Energy vs ERM, average accuracy', '0.02', -0.38, None, P(wb_acc['energy'], wb_acc['erm']))
for i, (k, lab) in enumerate([('random', 'Random'), ('spatial', 'SpatialDropout'), ('dropblock', 'DropBlock'), ('wcd', 'WCD'),
                              ('gd_topk_alt', 'Guided top-k alt.'), ('gd_dr_alt', 'Guided DR alt.')]):
    add(f'WB-{15+i:02d}', 'L832-833 ("Energy beats each of them by 4.9-5.7 pp, p<=0.002")' + ('; tab:corr L1099 (+5.71, p<0.001); tab:predict L1524' if k == 'random' else ''),
        f'WB Energy vs {lab}, wg', '<0.001' if k == 'random' else '<=0.002', 5.71 if k == 'random' else None, '6/6' if k == 'random' else None, P(wb['energy'], wb[k]))
# --- LA control (tab:la L929-935)
for k, lab, rp, rd, rw, hl, loc in [
        ('gap', 'GAP', '0.82', -0.3, '3/6', 'no', 'tab:la L929; L873'),
        ('energy', 'Energy', '0.03', 3.3, '5/6', 'yes (intro L195, reported as below bar)', 'tab:la L930; L195; L877; L1856; tab:predict L1524'),
        ('topk', 'Energy top-k', '0.002', 8.0, '6/6', 'yes', 'tab:la L934; abstract L84-85; intro L192-194; L722; L1168; L1339; concl L1913'),
        ('p3', 'Energy p_base=0.3', '0.02 / 0.023', 4.4, '6/6', 'no', 'tab:la L935; L1172')]:
    add(f'LA-{k}', loc, f'WB {lab} vs ERM+LA at matched average (rule C interpolation), wg', rp, rd, rw,
        one_sample(la[k][0]), test='one-sample t on seed-paired deltas', headline=hl)
add('LA-vit', 'L1671; intro L224', 'ViT-B MLP GAP vs ERM+LA at matched average, wg', '0.040', 4.1, '6/6', one_sample(vit_la()),
    test='one-sample t on seed-paired deltas', headline='yes (intro L224)')
# --- representation / reliance
add('DFR-01', 'L948; L2186', 'WB DFR head: Energy vs ERM features, DFR wg', '0.63', 0.2, '3/6', P(dfr['sdrop_energy'], dfr['none']), headline='null (abstract "same object information")')
add('AFR-01', 'tab:afr L2209; L956; L2181; intro L203; concl L1913', 'WB AFR label-only: Energy vs ERM features', '0.010', 1.35, '6/6', P(afr['sdrop_energy', 'afr_wg'], afr['none', 'afr_wg']), headline='yes')
add('AFR-02', 'tab:afr L2208', 'WB AFR label-only: GAP vs ERM features', '0.18', 0.9, '4/6', P(afr['sdrop_gap', 'afr_wg'], afr['none', 'afr_wg']))
add('AFR-03', 'tab:afr L2207; L2191', 'WB AFR label-only: Spectral Decoupling vs ERM features', '0.023', -3.8, '1/6', P(afr['sd', 'afr_wg'], afr['none', 'afr_wg']))
add('AFR-04', 'tab:afr L2208', 'WB AFR oracle: GAP vs ERM features', '0.026', 6.5, '5/6', P(afr['sdrop_gap', 'afro_wg'], afr['none', 'afro_wg']))
add('AFR-05', 'tab:afr L2209; L2182', 'WB AFR oracle: Energy vs ERM features', '0.091', 4.2, '4/6', P(afr['sdrop_energy', 'afro_wg'], afr['none', 'afro_wg']))
add('REL-01', 'tab:reliance L997; L972-973', 'WB bg-swap follows-background: Energy vs Random, 3 seeds', '0.010', -1.7, '3/3', P(rel['Energy'], rel['Random']),
    note='wins = seeds where Energy follows the background LESS; recomputed wins count delta<0')
add('REL-02', 'tab:reliance L1000; abstract L89-91; intro L200-203; concl L1917', 'WB bg-swap follows-background: top-k vs Shuffled top-k', '<0.001', -3.6, '6/6', P(rel['Energy top-k'], rel['Shuffled top-k']), headline='yes')
add('REL-03', 'tab:reliance L1004; abstract L89-91; intro L200-203', 'WB bg-swap follows-background: p0.3 vs Shuffled p0.3', '0.002', -4.4, '6/6', P(rel['Energy p0.3'], rel['Shuffled p0.3']), headline='yes')
# --- correlation sweep (tab:corr L1096-1098; 95% row = WB-12 / WB-15)
for c, rd1, rp1, rd2, rp2, rw2 in [('0.5', 2.75, '0.016', 2.88, '0.022', '6/6'), ('0.75', 2.62, '0.005', 2.57, '0.074', '5/6'),
                                   ('0.9', 2.91, '0.007', 3.84, '<0.001', '6/6')]:
    add(f'CORR-{c}-erm', f'tab:corr L{ {"0.5": 1096, "0.75": 1097, "0.9": 1098}[c]}; L1020', f'WB corr {float(c):.0%}: Energy vs ERM, wg', rp1, rd1, '6/6', P(corr[c, 'energy'], corr[c, 'erm']))
    add(f'CORR-{c}-rand', f'tab:corr L{ {"0.5": 1096, "0.75": 1097, "0.9": 1098}[c]}', f'WB corr {float(c):.0%}: Energy vs Random, wg', rp2, rd2, rw2, P(corr[c, 'energy'], corr[c, 'random']))
add('SUBY-01', 'L1033', 'WB class-balanced subsample (SUBY): ERM-cb vs ERM, wg', '0.35', -1.9, None, P(cb['erm'], wb['erm']))
add('RWY-01', 'tab:rwy L1064; L1045', 'WB RWY: Random vs ERM+RWY, wg', '0.08', 0.5, '4/6', P(rwy['random'], rwy['erm']))
add('RWY-02', 'tab:rwy L1065; L1043; abstract L86-87; intro L205', 'WB RWY: Energy vs ERM+RWY, wg (also 95% CI [+2.5,+6.2])', '0.002', 4.3, '6/6', P(rwy['energy'], rwy['erm']), headline='yes')
add('RWY-03', 'L1045-1046', 'WB RWY: Energy vs Random, wg', '0.002', 3.8, '6/6', P(rwy['energy'], rwy['random']))
# --- mechanism table (tab:mech L1246-1258); Random and Energy rows = WB-01 / WB-12
for k, lab, rp, rd, rw, loc, hl in [
        ('reverse', 'Weakest channels (reverse)', '0.048', -2.4, '1/6', 'tab:mech L1247; L1135', 'no'),
        ('early', 'Energy first half only', '0.059', 3.6, '5/6', 'tab:mech L1249; L1138; L1176 ("timing contrast p=0.06")', 'no'),
        ('late', 'Energy second half only', '0.82', 0.2, '2/6', 'tab:mech L1250', 'no'),
        ('p05', 'Energy p_base=0.05', '0.072', 1.9, '5/6', 'tab:mech L1251', 'no'),
        ('p2', 'Energy p_base=0.2', '0.001', 9.6, '6/6', 'tab:mech L1252; intro L210 (dose)', 'no'),
        ('p3', 'Energy p_base=0.3', '0.002', 10.7, '6/6', 'tab:mech L1253', 'no'),
        ('rand_p3', 'Random p_base=0.3', '0.37', 1.7, '5/6', 'tab:mech L1254; L1145', 'no'),
        ('shuf_p3', 'Shuffled p_base=0.3', '0.66', 0.8, '2/6', 'tab:mech L1255; L1165', 'no'),
        ('topk', 'Energy deterministic top-k', '<0.001', 10.2, '6/6', 'tab:mech L1256; L1157; abstract L81; intro L190; concl L1909', 'yes'),
        ('shuf_topk', 'Shuffled top-k', '0.68', 0.3, '3/6', 'tab:mech L1257; L1164', 'no'),
        ('cross', 'Cross-sample', '0.24', 2.5, '4/6', 'tab:mech L1258; L1212', 'no')]:
    add(f'MECH-{k}', loc, f'WB {lab} vs ERM, wg', rp, rd, rw, P(mech[k], wb['erm']), headline=hl)
WB14 = set(range(10)) | set(range(16, 20))   # canonical 14 paired seeds (TOPUP_PLAN; rev2c ERM seed 20 is unpaired)
add('WB13-01', 'L1159; intro L206-207; abstract L87-88', 'CONFIRMATORY 1: WB Energy vs ERM, wg, 14 paired seeds (0-9, 16-19)', 'CI [+3.7,+5.8]', 4.7, '13/13',
    P({k: v for k, v in wb13['energy'].items() if k in WB14}, {k: v for k, v in wb13['erm'].items() if k in WB14}),
    headline='yes', conf=True, note='paper: 13/13, +4.7 [3.7,5.8]; on disk 14 seeds pair (0-9,16-19) -> +4.85 [3.8,5.9]; with checkpoints_mech2 re-runs for seeds 0-2: +4.63 [3.7,5.5] 14/14; exact 13-seed set not identifiable')
add('WB13-02', 'L1160', 'WB Energy vs Random over extended seeds (14 paired)', 'CI [+4.4,+6.3]', 5.4, '14/14', P(wb13['energy'], wb13['random']), note='with checkpoints_mech2 re-runs for seeds 0-2: +5.41 [4.4,6.4], matching the paper; canonical raw_pod seeds 0-2 give +5.00')
add('GRAD-01', 'L1195; intro L207-209; concl L1919-1920', 'WB dominant-10% gradient share at epoch 30: Energy vs ERM, 3 seeds', '0.005', -0.021, '3/3', P(grad['sdrop_energy'], grad['none']), headline='yes',
    note='negative delta = lower share; wins count delta>0')
add('GRAD-02', 'L1196; intro L207-209', 'WB dominant-10% gradient share at epoch 30: Energy vs Random, 3 seeds', '0.003', -0.022, '3/3', P(grad['sdrop_energy'], grad['sdrop_random']), headline='yes')
# --- CelebA (tab:celeba L1297-1301; L1309-1315)
for k, lab, rp, rd, rw in [('random', 'Random', '0.130', -1.01, '1/6'), ('shuffle', 'Shuffled', '0.625', 1.10, '2/6'),
                           ('gap', 'GAP', '0.003', 11.08, '6/6'), ('energy', 'Energy', '0.018', 7.14, '5/6')]:
    add(f'CEL-{k}', f'tab:celeba L{ {"random": 1297, "shuffle": 1298, "gap": 1300, "energy": 1301}[k]}; L1309-1312', f'CelebA {lab} vs ERM, wg, 6 seeds', rp, rd, rw, P(cel[k], cel['erm']))
add('CEL-e-r', 'L1314; tab:predict L1532', 'CelebA Energy vs Random, wg', '0.018', 8.2, '6/6', P(cel['energy'], cel['random']))
add('CEL-g-r', 'L1315', 'CelebA GAP vs Random, wg', '0.002', 12.1, '6/6', P(cel['gap'], cel['random']))
# --- other benchmarks
add('SPW-easy', 'tab:other L1492; L1376', 'Spawrious O2O-Easy: Energy vs ERM, worst cell, 3 seeds', '0.04 / 0.041', 1.8, '3/3', P(spw['easy', 'energy'], spw['easy', 'erm']))
add('SPW-hard', 'tab:other L1493; L1378-1379', 'Spawrious O2O-Hard: Energy vs ERM, worst cell, 3 seeds', '0.12', 1.5, '3/3', P(spw['hard', 'energy'], spw['hard', 'erm']))
add('CUB-01', 'L1589-1590', 'CUB-200 L3: Full EGPG vs Shuffled, final-epoch acc, 3 seeds', '0.026', -0.51, '0/3', P(cub['egpg'], cub['shuffle']))
# --- ViT-B (tab:vitb L1634-1652; L1661-1689)
for k, ref, lab, rp, rd, rw in [
        ('res_random', 'erm', 'residual Random', '0.229', 2.23, '2/3'), ('res_shuffle', 'erm', 'residual Shuffled', '0.890', -0.26, '1/3'),
        ('res_energy', 'erm', 'residual Energy', '0.311', 0.57, '2/3'), ('res_gap', 'erm', 'residual GAP', '0.200', 1.77, '3/3'),
        ('mlp_random', 'erm', 'MLP Random', '0.360', -0.75, '1/6'), ('mlp_shuffle', 'erm', 'MLP Shuffled', '0.891', -0.13, '4/6'),
        ('mlp_energy', 'erm', 'MLP Energy', '0.947', 0.10, '2/3'), ('mlp_gap', 'erm', 'MLP GAP', '0.042', 4.05, '5/6'),
        ('mlp_gap_p2', 'erm2', 'MLP GAP p_base=0.2', '0.20', 5.7, '3/3'), ('mlp_gap_p3', 'erm2', 'MLP GAP p_base=0.3', '0.18', 5.5, '3/3'),
        ('mlp_energy_p3', 'erm2', 'MLP Energy p_base=0.3', '0.48', 1.9, '2/3'), ('mlp_random_p3', 'erm2', 'MLP Random p_base=0.3', '0.88', -0.2, '1/3'),
        ('tok_random', 'erm2', 'patch-token Random', '0.20', -0.7, '1/6'), ('tok_energy', 'erm2', 'patch-token Energy', '0.59', -0.8, '2/6'),
        ('tok_gap', 'erm2', 'patch-token GAP', '0.61', -0.8, '1/6')]:
    add(f'VIT-{k}', 'tab:vitb' + ('; L1662' if k == 'mlp_gap' else ''), f'ViT-B {lab} vs ERM, wg', rp, rd, rw, P(vit[k], vit[ref]))
add('VIT-g-r', 'L1662-1663; tab:sites L1563', 'ViT-B MLP GAP vs MLP Random, wg', '0.036', 4.8, '6/6', P(vit['mlp_gap'], vit['mlp_random']))
add('VIT-g-s', 'L1663-1664', 'ViT-B MLP GAP vs MLP Shuffled, wg', '0.010', 4.2, '6/6', P(vit['mlp_gap'], vit['mlp_shuffle']))
add('VIT-g3-r3', 'L1689', 'ViT-B MLP GAP p0.3 vs Random p0.3, wg, 3 seeds', '0.09', 5.7, '3/3', P(vit['mlp_gap_p3'], vit['mlp_random_p3']))
# --- CIFAR-100 (appendix, exploratory protocol)
w = stats.ttest_ind(list(cif['energy100'].values()), list(cif['none100'].values()), equal_var=False, alternative='greater')
add('CIF-welch', 'L2340-2341', 'CIFAR-100 full: Energy vs no dropout, best-test acc, 3 seeds', '0.15 (one-sided)', 0.13, None,
    test='Welch t (one-sided, unpaired)', p_rec=float(w.pvalue), note=f'recomputed t={w.statistic:.2f}')
for f, n, rt, rd, rw in [('100', 3, 't=-0.4', -0.09, '1/3'), ('0.1', 18, 't=3.8', 1.37, '15/18'), ('0.05', 6, 't=3.2', 0.64, '6/6'), ('0.025', 6, 't=-0.8', -0.09, '2/6')]:
    a, b = (cif['sdrop100'], cif['rand100']) if f == '100' else (cif[f'sdrop{f}'], cif[f'rand{f}'])
    r = P(a, b); tt = r[0] / (np.std([a[k] - b[k] for k in set(a) & set(b)], ddof=1) / math.sqrt(r[2]))
    add(f'CIF-scar-{f}', f'tab:scarcity L{ {"100": 2400, "0.1": 2401, "0.05": 2402, "0.025": 2403}[f]}', f'CIFAR-100 {100 if f == "100" else float(f)*100:g}% data: SDrop (full EGPG) vs Random, best-test acc',
        f'{rt} (p={2*stats.t.sf(abs(float(rt[2:])), n-1):.3g} two-sided, derived)', rd, rw, r, note=f'recomputed t={tt:.2f}')
for k, ref, lab, rp, rd, rw in [('rand', 'none', 'Random vs no dropout', '0.012', 1.39, '6/6'), ('gap', 'none', 'GAP vs no dropout', '0.005', 2.29, '6/6'),
                                ('sdrop', 'none', 'Full EGPG vs no dropout', '0.034', 2.40, '4/6'), ('gap', 'rand', 'GAP vs Random', '0.086', 0.90, '5/6'),
                                ('sdrop', 'rand', 'Full EGPG vs Random', '0.15', 1.02, '4/6')]:
    add(f'CIF-val-{k}-{ref}', 'tab:scarcity_val L2458-2460; L2437-2439', f'CIFAR-100 10% clean protocol: {lab}, 6 seeds', rp, rd, rw, P(cif[f'val_{k}'], cif[f'val_{ref}']))

# --- F2: implied tests (text says significant / n.s. / win count only) that can be recomputed
F2 = dict(family='F2')
add('X-topk-shuf', 'L1165-1166', 'WB top-k vs Shuffled top-k, wg', '', 9.97, '6/6', P(mech['topk'], mech['shuf_topk']), **F2)
add('X-p3-shuf', 'L1165-1166', 'WB Energy p0.3 vs Shuffled p0.3, wg', '', 9.91, '6/6', P(mech['p3'], mech['shuf_p3']), **F2)
add('X-early-late', 'L1176 ("timing contrast p=0.06")', 'WB Energy first half vs second half, wg', '', None, None, P(mech['early'], mech['late']), **F2,
    note='text quotes p=0.06 for the timing contrast; tab:mech gives 0.059 for first-half vs ERM (MECH-early)')
add('X-13-rand', 'L1161', 'WB Random vs ERM over extended seeds', '', -0.7, '4/13', P(wb13['random'], wb13['erm']), **F2)
add('X-LA-best-e', 'L892; tab:la fn a', 'WB best point Energy+LA vs ERM+LA', '', 0.8, '6/6', one_sample(la['energy'][1]), test='one-sample t on seed-paired deltas', **F2)
add('X-LA-best-k', 'L897; L1170-1171', 'WB best point top-k+LA vs ERM+LA', '', 3.2, '6/6', one_sample(la['topk'][1]), test='one-sample t on seed-paired deltas', **F2)
add('X-LA-best-3', 'L1172-1173', 'WB best point p0.3+LA vs ERM+LA', '', 1.0, '5/6', one_sample(la['p3'][1]), test='one-sample t on seed-paired deltas', **F2)
add('X-cb-energy', 'L1035', 'WB SUBY: Energy-cb vs ERM-cb, wg', '', -1.5, '1/5', P(cb['energy'], cb['erm']), **F2)
add('X-cel-la', 'L1326-1327', 'CelebA GAP vs ERM+LA at matched average', '', 0.7, '4/6', one_sample(cel_la_gap),
    test='one-sample t on logged seed deltas', **F2, note='deltas from analysis/celeba_la_frontier_2026-09-15.txt')
add('X-ms-erm', 'tab:other L1488; L1353', 'MetaShift Energy vs ERM', 'n.s. (text)', -0.9, '3/6', P(other['ms_energy'], other['ms_erm']), **F2)
add('X-ms-rand', 'L1353', 'MetaShift Energy vs Random', 'n.s. (text)', -0.6, '2/5', P(other['ms_energy'], other['ms_random']), **F2)
for k, lab, d1, w1, d2 in [('lion', 'lion/kangaroo', -8.0, '1/3', 2.8), ('sheep', 'sheep/kangaroo', -3.3, '0/2', 4.0), ('ele', 'elephant/sailboat', -1.8, '0/3', -1.2)]:
    add(f'X-{k}-erm', 'tab:other; L1365', f'NICO++ {lab}: Energy vs ERM (all paired seeds)', 'n.s. (text)', d1, w1, P(other[f'{k}_energy'], other[f'{k}_erm']), **F2)
    add(f'X-{k}-rand', 'L1365-1366; tab:predict', f'NICO++ {lab}: Energy vs Random (all dropout seeds)', 'n.s. (text)', d2, None, P(other[f'{k}_energy'], other[f'{k}_random']), **F2)
add('X-spw-e-r', 'L1377-1378; tab:predict L1530', 'Spawrious O2O-Easy Energy vs Random', 'n.s. (text)', 0.7, None, P(spw['easy', 'energy'], spw['easy', 'random']), **F2)
add('X-spwh-e-r', 'tab:predict L1531', 'Spawrious O2O-Hard Energy vs Random', 'n.s. (tab:predict)', 2.0, None, P(spw['hard', 'energy'], spw['hard', 'random']), **F2)
add('X-bar-erm', 'tab:other L1497; L1391-1392', 'BAR Energy vs ERM, worst class', 'no effect (text)', -0.9, '2/5', P(other['bar_energy'], other['bar_erm']), **F2)
add('X-bar-rand', 'tab:other L1497', 'BAR Random vs ERM, worst class', 'no effect (text)', 0.7, '2/5', P(other['bar_random'], other['bar_erm']), **F2)

# --- v2: tests on topped-up cells that v1 did not cover (reported values from main_article.tex where printed)
N = dict(family='new')
for k, lab, d2, w2 in [('lion', 'lion/kangaroo', -8.7, '1/3'), ('sheep', 'sheep/kangaroo', -7.3, '0/2'), ('ele', 'elephant/sailboat', 0.0, '1/3')]:
    add(f'X-{k}-rerm', 'tab:other', f'NICO++ {lab}: Random vs ERM', '', d2, w2, P(other[f'{k}_random'], other[f'{k}_erm']), **N)
add('X-ms-rerm', 'tab:other L1488', 'MetaShift Random vs ERM', '', 1.4, '3/5', P(other['ms_random'], other['ms_erm']), **N)
add('X-ms-shuf', 'tab:other caption L1474', 'MetaShift Shuffled vs ERM', '', 1.4, '4/5', P(other['ms_shuffle'], other['ms_erm']), **N)
add('X-bar-shuf', 'tab:other (BAR arms)', 'BAR Shuffled vs ERM, worst class', '', None, None, P(other['bar_shuffle'], other['bar_erm']), **N)
add('X-bar-e-r', 'tab:other L1497', 'BAR Energy vs Random, worst class', '', None, None, P(other['bar_energy'], other['bar_random']), **N)
for b, (de, we, dr, wr) in {'0.95': (0.2, '2/3', 0.2, '1/3'), '0.98': (0.7, '2/3', 0.9, '3/3'), '0.995': (2.7, '2/3', 2.7, '2/3')}.items():
    lab = f'{float(b) * 100:g}%'
    add(f'CM-{b}-e-erm', 'tab:other L1494-1496', f'Colored-MNIST {lab}: Energy vs ERM, bias-conflicting acc', '', de, we, P(cm[b, 'energy'], cm[b, 'erm']), **N)
    add(f'CM-{b}-r-erm', 'tab:other L1494-1496', f'Colored-MNIST {lab}: Random vs ERM, bias-conflicting acc', '', dr, wr, P(cm[b, 'random'], cm[b, 'erm']), **N)
    add(f'CM-{b}-e-r', 'tab:predict L1533', f'Colored-MNIST {lab}: Energy vs Random, bias-conflicting acc', '',
        {'0.95': 0.0, '0.98': -0.2}.get(b), None, P(cm[b, 'energy'], cm[b, 'random']), **N)
for k, lab in [('egpg', 'Full EGPG'), ('gap', 'GAP'), ('random', 'Random'), ('energy', 'Energy'), ('shuffle', 'Shuffled'),
               ('energy_sg', 'Energy, self-gamma'), ('egpg_sg', 'Full EGPG, self-gamma')]:
    add(f'L3-{k}', 'App. D L3 (~L2225-2235)', f'WB L3 {lab} vs ERM, wg', '', None, None, P(l3[k], wb['erm']), **N)
add('L3-e-r', 'tab:sites L1557 (selective gain +1.2 pp)', 'WB L3 Energy vs L3 Random, wg', '', 1.2, None, P(l3['energy'], l3['random']), **N)
add('L3-e-s', 'App. D L3', 'WB L3 Energy vs L3 Shuffled, wg', '', None, None, P(l3['energy'], l3['shuffle']), **N)
add('L3-g-e', 'App. D L2233 ("spatial factor buys nothing over Energy")', 'WB L3 Full EGPG vs L3 Energy, wg', '', None, None, P(l3['egpg'], l3['energy']), **N)
for w in ('1e-3', '1e-2', '1e-1'):
    add(f'WD-{w}', 'not yet in text (outline 4.2, tab:mech rows)', f'WB weight decay {w}: Energy vs ERM, wg', '', None, None, P(wd[w, 'energy'], wd[w, 'erm']), **N)
add('SCR-e-erm', 'L1818-1820; tab:predict L1525', 'WB from scratch: Energy vs ERM, wg', '', None, None, P(scr['energy'], scr['erm']), **N)
add('SCR-r-erm', 'L1818-1820', 'WB from scratch: Random vs ERM, wg', '', None, None, P(scr['random'], scr['erm']), **N)
add('SCR-e-r', 'tab:predict L1525', 'WB from scratch: Energy vs Random, wg', '', None, None, P(scr['energy'], scr['random']), **N)
add('CEL-suby', 'tab:celeba L1303; L1334', 'CelebA SUBY: ERM-cb vs ERM, wg', '', 37.7, '3/3', P(cel['erm_cb'], cel['erm']), **N,
    note='TOPUP_PLAN #8: pair with ERM 0-5 (checkpoints_rev)')
add('REL-e-erm', 'tab:reliance L994-996', 'WB bg-swap follows-background: Energy vs ERM (headline arms)', '', -1.2, None, P(rel['Energy'], rel['ERM']), **N)
add('REL-r-erm', 'L974 ("controls stay at the ERM rate")', 'WB bg-swap follows-background: Random vs ERM', '', 0.5, None, P(rel['Random'], rel['ERM']), **N)
for r in T:
    if r['id'] in ('SPW-easy', 'SPW-hard', 'X-spw-e-r', 'X-spwh-e-r', 'CUB-01'):
        r['note'] = (r['note'] + '; ' if r['note'] else '') + 'top-up PENDING (still running): seeds 0-2 only'

# --- pre-registered tests (2026-09-29): scripts/score_prereg.py (primary + secondary) and scripts/field_posthoc.py
with contextlib.redirect_stdout(io.StringIO()):
    prereg = runpy.run_path(str(ROOT / 'scripts/score_prereg.py'))['rows']
    import warnings; warnings.filterwarnings('ignore')
    sys.path.insert(0, str(ROOT / 'scripts')); import field_posthoc as fp
    fp.isic(); fp.cct20(fp.D / 'cct20_train_counts.csv')
for r in prereg:
    conf = r['family'] == 'confirmatory'
    degenerate = r['dataset'] == 'cct20' and r['mean'] == 0 and r['ci'] == 0
    lab = {'cct20': 'CCT20', 'isic_artifact': 'ISIC-artifact', 'waterbirds_vitbt6': 'ViT-B T6', 'waterbirds_vitbh9': 'ViT-B H9'}[r['dataset']]
    id_ = ('C-' if conf else 'PR-') + f"{r['dataset']}-{r['test'].replace('PRIMARY ', '').replace(' ', '')}"
    add(id_, 'prereg 2026-09-29', ('CONFIRMATORY: ' if conf else 'prereg secondary: ') + f"{lab} {r['test'].replace('PRIMARY ', '')}",
        '', None, None, None, test='paired t (two-sided), scripts/score_prereg.py', family='prereg', conf=conf,
        p_rec=1.0 if degenerate else r['p'],
        note='DEGENERATE endpoint: worst-group 0.0 in every arm and seed; p set to 1 (not estimable)' if degenerate else '')
    T[-1].update(delta_recomputed=r['mean'], n=r['n'], wins_recomputed=f"{r['wins']}/{r['n']}")
for ds, test, d in fp.RESULTS:
    add(f'PR-{ds}-{test}'.replace(' ', ''), 'prereg field secondary (scripts/field_posthoc.py)', f'{ds} {test}', '', None, None,
        one_sample(d), test='one-sample t on seed-paired deltas', family='prereg')
# Screening datasets (addendum B): once scored, append one confirmatory row per admitted dataset here, e.g.
# SCREEN = [('dataset', energy_by_seed, random_by_seed)]; add(f'C-{ds}', ..., P(e, r), family='prereg', conf=True)
SCREEN = []


# ---------------------------------------------------------------- corrections
def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1); return out

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    a = np.maximum.accumulate(p[o] * (m - np.arange(m))); out = np.empty(m); out[o] = np.minimum(a, 1); return out

def reported_numeric(s):
    """Numeric value of the reported p for use when not recomputable ('<0.001' -> 0.001, conservative)."""
    m = re.search(r'p=([0-9.eE-]+)', s) or re.search(r'([0-9]*\.[0-9]+)', s)
    return float(m.group(1)) if m else float('nan')

def discrepancy(r):
    rep, rec = r['p_reported'], r['p_recomputed']
    if not rep or rep.startswith(('n.s', 'no effect')) or rec is None or np.isnan(rec): return ''
    if rep.startswith('CI'): return ''
    if rep.startswith('t='):
        return ''  # t-stats compared via note
    first = rep.split('/')[0].strip().split(' ')[0]
    if first.startswith('<='): return '' if rec <= float(first[2:]) + 5e-4 else f'recomputed {rec:.4f} > reported bound {first}'
    if first.startswith('<'): return '' if rec < float(first[1:]) else f'recomputed {rec:.4f} >= reported bound {first}'
    val = float(first); dec = len(first.split('.')[1]) if '.' in first else 0
    return '' if abs(rec - val) <= 0.5 * 10 ** -dec + 1e-9 else f'reported {first}, recomputed {rec:.4f}'

df = pd.DataFrame(T)
df['p_used'] = [r if r is not None and not np.isnan(r) else reported_numeric(s) for r, s in zip(df.p_recomputed, df.p_reported)]
df['p_source'] = ['recomputed' if r is not None and not np.isnan(r) else 'as reported' for r in df.p_recomputed]
df['discrepancy'] = df.apply(discrepancy, axis=1)
dd = df.delta_reported.astype(float); dr = df.delta_recomputed.astype(float)
dmis = (dd.notna() & dr.notna() & ((dd - dr).abs() > np.maximum(0.06, 0.03 * dd.abs())))
df.loc[dmis, 'discrepancy'] = df.loc[dmis, 'discrepancy'].where(df.loc[dmis, 'discrepancy'] == '', df.loc[dmis, 'discrepancy'] + '; ') + \
    [f'delta reported {a:+.2f}, recomputed {b:+.2f}' for a, b in zip(dd[dmis], dr[dmis])]
if FROM_LOG:   # values read from queue logs (history not downloaded): mark every test that used one
    for i, r in df.iterrows():
        if r.id.startswith(('X-lion', 'X-sheep', 'X-ele')):
            df.at[i, 'note'] = (r.note + '; ' if r.note else '') + 'ERM seeds 3-5 (and sheep E/R 2-5) from queue-log [test] line (2 dp); histories not downloaded'
df['bh_q'] = np.nan
for fam in ('confirmatory', 'exploratory'):   # BH within each family (addendum A)
    m = df.family == fam; df.loc[m, 'bh_q'] = bh(df.loc[m, 'p_used'])
df['survives_q05'] = df.bh_q <= 0.05; df['survives_q10'] = df.bh_q <= 0.10
f1 = df.v1_family == 'F1'   # sensitivity: BH over the v1 F1 set only (printed p / t / CI in the text), as in v1
df['bh_q_F1'] = np.nan; df.loc[f1, 'bh_q_F1'] = bh(df.loc[f1, 'p_used'])
hl = df.headline.str.startswith('yes')  # headline claims (abstract / intro / conclusion)
df['holm_headline'] = np.nan; df.loc[hl, 'holm_headline'] = holm(df.loc[hl, 'p_used'])

cols = ['id', 'family', 'location', 'comparison', 'test', 'n', 'p_reported', 'p_recomputed', 'p_used', 'bh_q', 'survives_q05',
        'survives_q10', 'headline', 'v1_family', 'p_source', 'delta_reported', 'delta_recomputed', 'wins_reported', 'wins_recomputed',
        'p_signtest', 'holm_headline', 'bh_q_F1', 'discrepancy', 'note']
df[cols].to_csv(OUT, index=False, float_format='%.6g')

if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_colwidth', 60)
    print(df[['id', 'family', 'n', 'p_reported', 'p_recomputed', 'bh_q', 'survives_q05', 'headline', 'discrepancy']].to_string())
    for fam in ('confirmatory', 'exploratory'):
        F = df[df.family == fam]
        print(f'\n{fam}: {len(F)} tests, recomputed {(F.p_source == "recomputed").sum()}, BH q<=.05: {F.survives_q05.sum()}, q<=.10: {F.survives_q10.sum()}')
    print(f'v1-F1 subset BH q<=.05: {(df.bh_q_F1 <= 0.05).sum()}/{f1.sum()}; headline Holm: {(df.holm_headline <= 0.05).sum()}/{hl.sum()} at alpha=.05')
    print(f'values from queue logs: {len(FROM_LOG)} runs')
    # self-check: BH / Holm against textbook example
    assert np.allclose(bh([0.01, 0.04, 0.03, 0.005]), [0.02, 0.04, 0.04, 0.02])
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    print(f'wrote {OUT}')
