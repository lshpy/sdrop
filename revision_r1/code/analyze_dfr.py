"""DFR (Kirichenko et al. 2023) on frozen penultimate features from *_dump.npz: is the representation better?
Logistic regression on group-balanced subsamples of the validation split (group labels used HERE only),
C tuned on one val half by worst-group, then refit on group-balanced subsamples of all val (weights averaged
over n_ens subsamples), test worst-group reported. Seed-paired against ERM.
Usage: python analyze_dfr.py <glob of dumps> [...more globs]"""
import sys, glob, re, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
def wg(pred, y, p):
    g = y * 2 + p; return min(100 * (pred[g == k] == y[g == k]).mean() for k in range(4) if (g == k).any()), 100 * (pred == y).mean()
def balanced_idx(y, p, idx, rng):
    g = y[idx] * 2 + p[idx]; n = min((g == k).sum() for k in range(4))
    return np.concatenate([rng.choice(idx[g == k], n, replace=False) for k in range(4)])
def fit_ens(X, y, p, idx, C, rng, n_ens=10):
    W = []; B = []
    for _ in range(n_ens):
        b = balanced_idx(y, p, idx, rng); clf = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=2000).fit(X[b], y[b])
        W.append(clf.coef_); B.append(clf.intercept_)
    return np.mean(W, 0), np.mean(B, 0)
def dfr(d, seed):
    rng = np.random.RandomState(seed)
    X = d['val_feats'].astype(np.float32); y = d['val_y']; p = d['val_places']
    sc = StandardScaler().fit(X); Xs = sc.transform(X); Xt = sc.transform(d['test_feats'].astype(np.float32))
    idx = rng.permutation(len(y)); A, Bh = idx[: len(idx) // 2], idx[len(idx) // 2:]
    best = (-1, None)
    for C in (1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01):
        w, b = fit_ens(Xs, y, p, A, C, rng, n_ens=5)
        s = wg(((Xs[Bh] @ w.T + b)[:, 0] > 0).astype(int), y[Bh], p[Bh])[0]
        if s > best[0]: best = (s, C)
    w, b = fit_ens(Xs, y, p, np.arange(len(y)), best[1], rng, n_ens=10)
    W, A_ = wg(((Xt @ w.T + b)[:, 0] > 0).astype(int), d['test_y'], d['test_places'])
    return W, A_, best[1]
rows = []
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        rid = f.split('/')[-1]; m = re.match(r'waterbirds_(.+?)_rate([\d.]+).*?seed(\d+)', rid)
        arm = m.group(1) + ('@' + m.group(2) if m.group(2) != '0.1' else '') + ('_topk' if 'seltopk' in rid else '') + ('_sd' if '_sd0' in rid else '')
        d = np.load(f); W, A_, C = dfr(d, int(m.group(3)))
        w0, a0 = wg(d['test_logits'].argmax(1), d['test_y'], d['test_places'])
        rows.append(dict(arm=arm, seed=int(m.group(3)), wg=w0, avg=a0, dfr_wg=W, dfr_avg=A_, C=C))
df = pd.DataFrame(rows).drop_duplicates(['arm', 'seed']); pd.set_option('display.width', 200)
print(df.groupby('arm')[['wg', 'dfr_wg', 'dfr_avg']].agg(['mean', 'std', 'count']).round(2).to_string())
base = df[df.arm == 'none'].set_index('seed')
for arm in sorted(set(df.arm) - {'none'}):
    s = df[df.arm == arm].set_index('seed'); c = s.index.intersection(base.index)
    if len(c) > 1:
        dl = s.loc[c, 'dfr_wg'] - base.loc[c, 'dfr_wg']; t = stats.ttest_rel(s.loc[c, 'dfr_wg'], base.loc[c, 'dfr_wg'])
        print(f'DFR {arm:22s} - DFR ERM: {dl.mean():+.2f} ({(dl > 0).sum()}/{len(dl)}, p={t.pvalue:.3f})')
df.to_csv('../paper_results/analysis/wb_dfr_2026-09-24.csv', index=False)
