"""Spawrious results: worst-group over (class, location) test cells, paired deltas, and a label-free
logit-adjustment control (class prior from val) from *_dump.npz.  Usage: python analyze_spawrious.py <dir>"""
import sys, glob, re, numpy as np, pandas as pd
from scipy import stats
def wg(pred, y, p):
    keys = sorted(set(zip(y.tolist(), p.tolist()))); accs = [100*(pred[(y==a)&(p==b)]==a).mean() for a,b in keys]
    return min(accs), 100*(pred==y).mean()
rows=[]
for f in sorted(glob.glob(f'{sys.argv[1]}/spawrious_*_history.csv')):
    rid=f.split('/')[-1].replace('_history.csv',''); m=re.match(r'spawrious_(o2o_\w+?)_(none|sdrop_energy|sdrop_random)_.*seed(\d)$',rid)
    if not m: continue
    ch,arm,seed=m.group(1),m.group(2),int(m.group(3)); h=pd.read_csv(f); r=h.dropna(subset=['test_worst_group']).iloc[-1]
    row=dict(ch=ch,arm=arm,seed=seed,wg=r['test_worst_group'],avg=r['test_acc'])
    d=f.replace('_history.csv','_dump.npz')
    try:
        z=np.load(d); pi=np.bincount(z['val_y'],minlength=4)/len(z['val_y']); lp=np.log(pi+1e-9)
        for t in (0.5,1.0):
            w,a=wg((z['test_logits']-t*lp).argmax(1),z['test_y'],z['test_places']); row[f'la{t}_wg']=w; row[f'la{t}_avg']=a
    except Exception: pass
    rows.append(row)
df=pd.DataFrame(rows); pd.set_option('display.width',200); print(df.round(2).to_string(index=False))
print(df.groupby(['ch','arm']).agg(['mean','std']).round(2).to_string())
for ch,g in df.groupby('ch'):
    d=g.pivot(index='seed',columns='arm',values='wg')
    for a,b in (('sdrop_energy','none'),('sdrop_energy','sdrop_random'),('sdrop_random','none')):
        if a in d and b in d:
            dd=d[[a,b]].dropna(); dl=dd[a]-dd[b]; t=stats.ttest_rel(dd[a],dd[b]) if len(dd)>1 else None
            print(f"{ch}: {a}-{b} = {dl.mean():+.2f} ({(dl>0).sum()}/{len(dl)}" + (f", p={t.pvalue:.3f})" if t else ")"))
df.to_csv(f'{sys.argv[1]}/spawrious_summary.csv',index=False)
