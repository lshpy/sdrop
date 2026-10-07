"""Revision analyses from *_dump.npz (val/test logits, penultimate feats, labels, places).

1. Logit adjustment (Menon et al. 2021, post-hoc): logits - tau*log(prior_train), tau=1 (label-free) and
   tau selected on val worst-group (oracle, uses group labels) -> worst-group / avg test accuracy.
2. DFR (Kirichenko et al. 2023): logistic regression on frozen penultimate feats, fit on a group-balanced
   subsample of val (half), C selected on the other half by worst-group, then refit on all balanced val.
   Applied to ERM feats (=DFR) and to SDrop feats (=SDrop+DFR).
Usage: python analyze_revision.py <dump_dir> <dataset> [prior_waterbird]
"""
import sys, glob, re, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

def groups(y, p): return y * 2 + p
def wg(pred, y, p):
    g = groups(y, p); accs = [100 * (pred[g == k] == y[g == k]).mean() for k in range(4) if (g == k).any()]
    return min(accs), 100 * (pred == y).mean(), accs

def logit_adjust(d, log_prior):
    out = {}
    for name, tau in (('la_tau1', 1.0), ('la_oracle', None)):
        if tau is None:  # select on val worst-group
            best = max(np.linspace(0, 3, 31), key=lambda t: wg((d['val_logits'] - t * log_prior).argmax(1), d['val_y'], d['val_places'])[0])
            tau = float(best)
        w, a, _ = wg((d['test_logits'] - tau * log_prior).argmax(1), d['test_y'], d['test_places'])
        out[name + '_wg'] = w; out[name + '_avg'] = a; out[name + '_tau'] = tau
    return out

def dfr(d, seed=0):
    rng = np.random.RandomState(seed)
    Xv, yv, pv = d['val_feats'].astype(np.float32), d['val_y'], d['val_places']
    g = groups(yv, pv); n = min((g == k).sum() for k in range(4))
    idx = np.concatenate([rng.choice(np.where(g == k)[0], n, replace=False) for k in range(4)]); rng.shuffle(idx)
    half = len(idx) // 2; A, Bi = idx[:half], idx[half:]
    sc = StandardScaler().fit(Xv[idx])
    best_C, best_w = None, -1
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xv[A]), yv[A])
        w = wg(clf.predict(sc.transform(Xv[Bi])), yv[Bi], pv[Bi])[0]
        if w > best_w: best_w, best_C = w, C
    clf = LogisticRegression(C=best_C, max_iter=2000).fit(sc.transform(Xv[idx]), yv[idx])
    w, a, _ = wg(clf.predict(sc.transform(d['test_feats'].astype(np.float32))), d['test_y'], d['test_places'])
    return {'dfr_wg': w, 'dfr_avg': a, 'dfr_C': best_C}

def main(dump_dir, dataset, prior1=None):
    rows = []
    for f in sorted(glob.glob(f'{dump_dir}/{dataset}_*_dump.npz')):
        rid = re.sub(r'_dump\.npz$', '', f.split('/')[-1]); d = np.load(f)
        m = re.match(rf'{dataset}_(.+?)_rate.*seed(\d+)', rid); arm, seed = m.group(1), int(m.group(2))
        if '_sd0' in rid: arm = 'sd'
        # train prior: from val labels if prior not given (val is drawn with the train correlation on Waterbirds)
        pi1 = prior1 if prior1 is not None else d['val_y'].mean()
        log_prior = np.log(np.array([1 - pi1, pi1]))
        w, a, accs = wg(d['test_logits'].argmax(1), d['test_y'], d['test_places'])
        r = {'arm': arm, 'seed': seed, 'wg': w, 'avg': a, **{f'g{k}': accs[k] for k in range(4)}}
        r.update(logit_adjust(d, log_prior)); r.update(dfr(d, seed)); rows.append(r)
    df = pd.DataFrame(rows).sort_values(['arm', 'seed']); df.to_csv(f'{dump_dir}/{dataset}_revision.csv', index=False)
    pd.set_option('display.width', 200); print(df.round(2).to_string(index=False))
    print('\n== mean ± std by arm =='); print(df.groupby('arm').agg(['mean', 'std']).round(2)[['wg', 'la_tau1_wg', 'la_oracle_wg', 'dfr_wg']].to_string())
    if 'none' in set(df.arm):
        base = df[df.arm == 'none'].set_index('seed')
        for arm in sorted(set(df.arm) - {'none'}):
            s = df[df.arm == arm].set_index('seed'); common = s.index.intersection(base.index)
            for col, ref in (('wg', 'wg'), ('wg', 'la_tau1_wg'), ('wg', 'la_oracle_wg'), ('dfr_wg', 'dfr_wg')):
                dlt = s.loc[common, col] - base.loc[common, ref]
                t = stats.ttest_rel(s.loc[common, col], base.loc[common, ref]) if len(common) > 1 else None
                print(f'{arm}.{col} - none.{ref}: {dlt.mean():+.2f} ({(dlt>0).sum()}/{len(dlt)}, p={t.pvalue:.3f})' if t else f'{arm}.{col}: n=1')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
