"""AFR (Qiu et al. 2023, Automatic Feature Reweighting) from *_dump.npz, group-label-free.
Last layer (logistic regression) retrained on the held-out val split with per-sample weights
w_i ∝ exp(-gamma * p_i(y_i)) from the ERM/SDrop model's own val predictions; gamma and C
chosen on a val half by *worst-class* accuracy (label-only), then refit on all val.
Reports test worst-group / avg. Also 'sdrop+afr' by applying to SDrop feats.
Usage: python analysis/analyze_afr.py <dump_dir> <dataset>"""
import sys, glob, re, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
def wg(pred,y,p):
    g=y*2+p; a=[100*(pred[g==k]==y[g==k]).mean() for k in range(4) if (g==k).any()]; return min(a),100*(pred==y).mean()
def worst_class(pred,y): return min(100*(pred[y==c]==y[y==c]).mean() for c in (0,1))
def afr(d,seed=0,oracle=False):
    rng=np.random.RandomState(seed); X=d['val_feats'].astype(np.float32); y=d['val_y']; p=d['val_places']
    prob=np.exp(d['val_logits']-d['val_logits'].max(1,keepdims=True)); prob/=prob.sum(1,keepdims=True); py=prob[np.arange(len(y)),y]
    idx=rng.permutation(len(y)); A,Bi=idx[:len(idx)//2],idx[len(idx)//2:]
    sc=StandardScaler().fit(X); Xs=sc.transform(X); Xt=sc.transform(d['test_feats'].astype(np.float32))
    best=(-1,None)
    for gamma in (0,2,5,10,20,50):
        w=np.exp(-gamma*py); w/=w.mean()
        for C in (0.01,0.1,1.0):
            clf=LogisticRegression(C=C,max_iter=3000).fit(Xs[A],y[A],sample_weight=w[A])
            s=wg(clf.predict(Xs[Bi]),y[Bi],p[Bi])[0] if oracle else worst_class(clf.predict(Xs[Bi]),y[Bi])
            if s>best[0]: best=(s,(gamma,C))
    gamma,C=best[1]; w=np.exp(-gamma*py); w/=w.mean()
    clf=LogisticRegression(C=C,max_iter=3000).fit(Xs,y,sample_weight=w)
    W,A_=wg(clf.predict(Xt),d['test_y'],d['test_places']); return {'afr_wg':W,'afr_avg':A_,'afr_gamma':gamma,'afr_C':C}
def main(dd,ds):
    rows=[]
    for f in sorted(glob.glob(f'{dd}/{ds}_*_dump.npz')):
        rid=f.split('/')[-1]; d=np.load(f); m=re.match(rf'{ds}_(.+?)_rate.*seed(\d+)',rid); arm,seed=m.group(1),int(m.group(2))
        if '_sd0' in rid: arm='sd'
        if '_cb_' in rid: arm+='_cb'
        W,A_=wg(d['test_logits'].argmax(1),d['test_y'],d['test_places']); r={'arm':arm,'seed':seed,'wg':W,'avg':A_}; r.update(afr(d,seed)); r.update({k.replace('afr','afro'):v for k,v in afr(d,seed,oracle=True).items()}); rows.append(r)
    df=pd.DataFrame(rows).sort_values(['arm','seed']); df.to_csv(f'{dd}/{ds}_afr.csv',index=False)
    pd.set_option('display.width',200); print(df.round(2).to_string(index=False))
    print('\n== mean by arm =='); print(df.groupby('arm')[['wg','avg','afr_wg','afr_avg','afro_wg','afro_avg']].agg(['mean','std']).round(2).to_string())
    base=df[df.arm=='none'].set_index('seed')
    for arm in sorted(set(df.arm)-{'none'}):
        s=df[df.arm==arm].set_index('seed'); c=s.index.intersection(base.index)
        if len(c)>1:
            dl=s.loc[c,'afr_wg']-base.loc[c,'afr_wg']; t=stats.ttest_rel(s.loc[c,'afr_wg'],base.loc[c,'afr_wg'])
            print(f'{arm}+AFR - ERM+AFR: {dl.mean():+.2f} ({(dl>0).sum()}/{len(dl)}, p={t.pvalue:.3f})')
            dl=s.loc[c,'afro_wg']-base.loc[c,'afro_wg']; t=stats.ttest_rel(s.loc[c,'afro_wg'],base.loc[c,'afro_wg'])
            print(f'{arm}+AFR(oracle γ) - ERM+AFR(oracle γ): {dl.mean():+.2f} ({(dl>0).sum()}/{len(dl)}, p={t.pvalue:.3f})')
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
