"""Channel selectivity of the most dominant channels, from ERM *_dump.npz (pooled penultimate features).
For each channel: dominance = mean activation over test images; background selectivity = max over
one-vs-rest |AUC-0.5| for predicting the place/attribute; class selectivity likewise for the label.
Reports mean selectivity of the top-10% most dominant channels vs the rest, and the gap.
Usage: python analysis/analyze_selectivity.py <glob of ERM dumps>"""
import sys, glob, re, numpy as np
from sklearn.metrics import roc_auc_score
def sel(F, lab):
    out = np.zeros(F.shape[1])
    for v in np.unique(lab):
        t = (lab == v).astype(int)
        if 0 < t.sum() < len(t):
            a = np.array([roc_auc_score(t, F[:, c]) if F[:, c].std() > 0 else 0.5 for c in range(F.shape[1])])
            out = np.maximum(out, np.abs(a - 0.5))
    return out
rows = []
for f in sorted(glob.glob(sys.argv[1])):
    d = np.load(f); F = d['test_feats'].astype(np.float32); y = d['test_y']; p = d['test_places']
    if len(np.unique(p)) < 2: continue
    dom = F.mean(0); top = np.argsort(-dom)[: max(1, F.shape[1] // 10)]; rest = np.setdiff1d(np.arange(F.shape[1]), top)
    sb, sc = sel(F, p), sel(F, y)
    r = dict(run=f.split('/')[-1][:45], bg_top=sb[top].mean(), bg_rest=sb[rest].mean(), cls_top=sc[top].mean(), cls_rest=sc[rest].mean())
    rows.append(r); print(f"{r['run']:45s} bg top {r['bg_top']:.3f} rest {r['bg_rest']:.3f} | class top {r['cls_top']:.3f} rest {r['cls_rest']:.3f} | bg-cls(top) {r['bg_top']-r['cls_top']:+.3f}")
if rows:
    m = {k: np.mean([r[k] for r in rows]) for k in rows[0] if k != 'run'}
    print(f"MEAN over {len(rows)}: bg top {m['bg_top']:.3f} rest {m['bg_rest']:.3f} | class top {m['cls_top']:.3f} rest {m['cls_rest']:.3f} | bg-cls(top) {m['bg_top']-m['cls_top']:+.3f} | bg top/rest {m['bg_top']/m['bg_rest']:.2f}")
