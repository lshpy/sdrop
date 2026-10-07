"""Logit-adjustment frontier: SDrop arm vs ERM+LA at matched average accuracy (seed-paired).

Usage: python analyze_frontier.py <erm_dump_dir> <arm_dump_dir> <arm_regex> [tau_max]
  erm dumps  : waterbirds_none_rate0.1_none_ep30_seed{s}_dump.npz
  arm dumps  : any waterbirds_*_seed{s}_dump.npz whose run id matches <arm_regex>
Per seed: ERM+LA(tau) frontier on a tau grid; SDrop at tau=0 paired with ERM+LA at the tau whose
average accuracy is closest (tau>=0 only, so a SDrop model above ERM's average is paired at tau=0);
best point of ERM+LA vs best of SDrop+LA.
"""
import sys, glob, re, numpy as np
from scipy import stats

def wg(logits, y, p, tau, log_prior):
    pred = (logits - tau * log_prior).argmax(1); g = y * 2 + p
    return min(100 * (pred[g == k] == y[g == k]).mean() for k in range(4)), 100 * (pred == y).mean()

def load(f):
    d = np.load(f); pi1 = d['val_y'].mean(); lp = np.log(np.array([1 - pi1, pi1]))
    return d['test_logits'], d['test_y'], d['test_places'], lp

def main(erm_dir, arm_dir, arm_re, tau_max=3.0):
    taus = np.round(np.arange(0, tau_max + 1e-9, 0.05), 2)
    erm = {int(re.search(r'seed(\d+)', f).group(1)): load(f) for f in glob.glob(f'{erm_dir}/waterbirds_none_rate0.1_none_ep30_seed*_dump.npz')}
    arms = {}
    for f in sorted(glob.glob(f'{arm_dir}/waterbirds_*_seed*_dump.npz')):
        rid = f.split('/')[-1]
        if re.search(arm_re, rid): arms[int(re.search(r'seed(\d+)', rid).group(1))] = load(f)
    print(f'ERM seeds {sorted(erm)}  arm seeds {sorted(arms)}  (arm_re={arm_re})')
    d_matched, d_excl, best_erm, best_arm, arm0, d_floor, d_interp = [], [], [], [], [], [], []
    for s in sorted(set(erm) & set(arms)):
        L, y, p, lp = erm[s]; fr = np.array([wg(L, y, p, t, lp) for t in taus])          # (T, 2) wg, avg
        La, ya, pa, lpa = arms[s]; w0, a0 = wg(La, ya, pa, 0.0, lpa); arm0.append((w0, a0))
        j = int(np.argmin(np.abs(fr[:, 1] - a0))); dw = w0 - fr[j, 0]            # rule A: nearest average
        ok = np.where(fr[:, 1] >= a0)[0]; jf = int(ok[-1]) if len(ok) else 0       # rule B: largest tau with ERM avg >= arm avg
        d_floor.append(w0 - fr[jf, 0])
        # rule C: linear interpolation of the ERM+LA frontier (wg as a function of avg) at the arm's average
        order = np.argsort(fr[:, 1]); w_interp = np.interp(a0, fr[order, 1], fr[order, 0])
        d_interp.append(w0 - w_interp)
        d_matched.append(dw); (d_excl.append(dw) if taus[j] > 0 else None)
        fa = np.array([wg(La, ya, pa, t, lpa) for t in taus])
        best_erm.append(fr[:, 0].max()); best_arm.append(fa[:, 0].max())
        print(f'  seed{s}: arm tau=0 avg={a0:.2f} wg={w0:.2f} | ERM+LA tau={taus[j]:.2f} avg={fr[j,1]:.2f} wg={fr[j,0]:.2f}  dwg={dw:+.2f} | best ERM+LA {fr[:,0].max():.1f} vs best arm+LA {fa[:,0].max():.1f}')
    d = np.array(d_matched); t = stats.ttest_1samp(d, 0) if len(d) > 1 else None
    print(f'arm tau=0: wg {np.mean([a[0] for a in arm0]):.2f} avg {np.mean([a[1] for a in arm0]):.2f}')
    print(f'vs ERM+LA at matched average: {d.mean():+.2f} ({(d>0).sum()}/{len(d)}' + (f', p={t.pvalue:.3f})' if t else ')'))
    if d_excl and len(d_excl) < len(d): e = np.array(d_excl); print(f'  excluding seeds matched at tau=0: {e.mean():+.2f} ({(e>0).sum()}/{len(e)})')
    fl = np.array(d_floor); tf = stats.ttest_1samp(fl, 0) if len(fl) > 1 else None
    print(f'rule B (largest tau with ERM avg >= arm avg): {fl.mean():+.2f} ({(fl>0).sum()}/{len(fl)}' + (f', p={tf.pvalue:.3f})' if tf else ')'))
    di = np.array(d_interp); ti = stats.ttest_1samp(di, 0) if len(di) > 1 else None
    print(f'rule C (interpolated ERM+LA frontier at arm avg): {di.mean():+.2f} ({(di>0).sum()}/{len(di)}' + (f', p={ti.pvalue:.3f})' if ti else ')'))
    print(f'best point: ERM+LA {np.mean(best_erm):.2f}  arm+LA {np.mean(best_arm):.2f}  ({np.mean(best_arm)-np.mean(best_erm):+.2f}, {(np.array(best_arm)>np.array(best_erm)).sum()}/{len(best_arm)})')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 3.0)
