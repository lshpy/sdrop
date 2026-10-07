"""Label-free diagnostic from *_spatial.npz: are the backbone's most active channels spatially diffuse?
For each image rank channels by energy; top 10% vs rest; compare mean peakedness P (max/sum over the 7x7 map).
Ratio (1-P_top)/(1-P_rest) > 1 means the dominant channels are more diffuse (background-like). No labels used.
Also reports the labelled selectivity ratio on the same images for comparison. Usage: python analyze_spatial.py <dir>"""
import sys, glob, numpy as np
from sklearn.metrics import roc_auc_score
for f in sorted(glob.glob(f'{sys.argv[1]}/*_spatial.npz')):
    d = np.load(f); E, P, y, p = d['energy'], d['peak'], d['y'], d['places']; N, C = E.shape
    dom = np.argsort(-E.mean(0))[: C // 10]; rest = np.setdiff1d(np.arange(C), dom)
    diff_top = (1 - P[:, dom]).mean(); diff_rest = (1 - P[:, rest]).mean()
    # per-image version: rank within each image
    per = []
    for i in range(N):
        t = np.argsort(-E[i])[: C // 10]; r = np.setdiff1d(np.arange(C), t); per.append((1 - P[i, t]).mean() / (1 - P[i, r]).mean())
    out = f"{f.split('/')[-1][:45]:45s} N={N} diffuseness ratio (dataset-rank) {diff_top/diff_rest:.3f}  (per-image) {np.mean(per):.3f}"
    if len(np.unique(p)) > 1 and len(np.unique(y)) > 1:
        def sel(lab):
            s = np.zeros(C)
            for v in np.unique(lab):
                t = (lab == v).astype(int)
                if 0 < t.sum() < len(t): s = np.maximum(s, np.abs(np.array([roc_auc_score(t, E[:, c]) if E[:, c].std() > 0 else 0.5 for c in range(C)]) - 0.5))
            return s
        sb = sel(p); out += f" | labelled bg-selectivity ratio {sb[dom].mean()/sb[rest].mean():.2f}"
    print(out)
