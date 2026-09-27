"""Gradient reaching dominant vs non-dominant L4 channels over training, from *_gradchan.csv.
Dominant = top 10% channels by energy_mean at epoch 1 of the SAME run (the channels the pretrained backbone
activates most on this data). Reports per epoch the share of total squared gradient norm that reaches the
non-dominant 90%, and the mean per-channel gradient of non-dominant channels relative to epoch 1.
Usage: python analysis/analyze_gradchan.py <dir>"""
import sys, glob, re, numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob(f'{sys.argv[1]}/waterbirds_*_gradchan.csv')):
    rid = f.split('/')[-1].replace('_gradchan.csv', ''); arm = re.match(r'waterbirds_(.+?)_rate', rid).group(1); seed = int(re.search(r'seed(\d+)', rid).group(1))
    d = pd.read_csv(f); e1 = d[d.epoch == 1].sort_values('channel'); C = len(e1)
    dom = set(e1.sort_values('energy_mean', ascending=False).channel.values[: C // 10])
    for ep, g in d.groupby('epoch'):
        g = g.set_index('channel'); tot = g.grad_sq_sum.sum(); nd = g.loc[[c for c in g.index if c not in dom]].grad_sq_sum
        rows.append(dict(arm=arm, seed=seed, epoch=ep, share_nondom=nd.sum() / tot, mean_nondom=nd.mean() / g.n.iloc[0], mean_dom=g.loc[list(dom)].grad_sq_sum.mean() / g.n.iloc[0]))
df = pd.DataFrame(rows); df.to_csv(f'{sys.argv[1]}/gradchan_summary.csv', index=False)
piv = df.groupby(['arm', 'epoch'])[['share_nondom', 'mean_nondom', 'mean_dom']].mean().unstack(0)
pd.set_option('display.width', 200); print(piv.round(4).to_string())
for arm in sorted(set(df.arm)):
    a = df[df.arm == arm]; early = a[a.epoch <= 5].groupby('seed').share_nondom.mean(); print(f'{arm}: share of gradient to non-dominant 90% (epochs 1-5) = {early.mean():.3f} ± {early.std():.3f} over {len(early)} seeds')
