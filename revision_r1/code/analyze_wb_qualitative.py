"""Waterbirds 정성 그림: 각 이미지에서 SDrop 이 가장 자주 드롭하는 채널들(상위 10%)이 어디에서 반응하는지.
   ERM 체크포인트 L4 특징 → Energy 점수 드롭확률 상위 10% 채널의 평균 활성 맵 vs 하위 50% 채널 → 224 로 업샘플 → 오버레이.
   python analyze_wb_qualitative.py --ckpt none_s0.pth --parquet test.parquet --out fig_waterbirds_channels.pdf
"""
import argparse, io, torch, numpy as np, pyarrow.parquet as pq, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from _wb_common import build_from_ckpt
from sdrop import compute_score, _drop_probs_from_scores
ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--parquet',required=True); ap.add_argument('--out',required=True); ap.add_argument('--per_group',type=int,default=2); ap.add_argument('--groups',default='minority3')
a=ap.parse_args(); dev='cpu'
t=pq.read_table(a.parquet,columns=['image','label','place']); lab=np.array(t.column('label').to_pylist()); pl=np.array(t.column('place').to_pylist()); imgs=t.column('image').to_pylist()
rng=np.random.RandomState(0); picks=[]
plan=[((1,0),2),((0,1),2),((0,0),1),((1,1),1)]
for (l,p_),k in plan:
    idx=np.where((lab==l)&(pl==p_))[0]; picks+= [(int(i),l,p_) for i in rng.choice(idx,k,replace=False)]
ck=torch.load(a.ckpt,map_location=dev,weights_only=False); model=build_from_ckpt(ck,dev,method='none',layers=[])
feats={}; model.layer4.register_forward_hook(lambda m,i,o: feats.__setitem__('x',o.detach()))
norm=transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)); tf=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224)])
names={(1,0):'waterbird on land\n(minority)',(0,1):'landbird on water\n(minority)',(0,0):'landbird on land',(1,1):'waterbird on water'}
n=len(picks); fig,axes=plt.subplots(3,n,figsize=(1.75*n,5.4)); BLUE='#2a78d6'
for j,(i,l,p_) in enumerate(picks):
    im=tf(Image.open(io.BytesIO(imgs[i]['bytes'])).convert('RGB')); x=norm(transforms.functional.to_tensor(im)).unsqueeze(0)
    with torch.no_grad(): model(x)
    f=feats['x'][0].float()                                   # (C,7,7)
    s=compute_score(f.unsqueeze(0),'energy',1.0,1.0,'max',1e-6); p=_drop_probs_from_scores(s,0.1,'mean',1e-6)[0]
    top=p>=p.quantile(0.9); low=p<=p.quantile(0.5)
    mt=f[top].mean(0); ml=f[low].mean(0)
    up=lambda m: torch.nn.functional.interpolate(m[None,None],size=224,mode='bilinear',align_corners=False)[0,0].numpy()
    mt,ml=up(mt),up(ml); mt=(mt-mt.min())/(np.ptp(mt)+1e-9); ml=(ml-ml.min())/(np.ptp(ml)+1e-9)
    gray=np.asarray(im.convert('L'),dtype=float)/255.0
    def overlay(ax,m,cmap):
        ax.imshow(gray,cmap='gray',vmin=0,vmax=1)
        rgba=matplotlib.colormaps[cmap](np.clip((m-0.35)/0.65,0,1)); rgba[...,3]=np.clip((m-0.35)/0.65,0,1)*0.85
        ax.imshow(rgba)
    axes[0,j].imshow(im); overlay(axes[1,j],mt,'Blues'); overlay(axes[2,j],ml,'Oranges')
    axes[0,j].set_title(names[(l,p_)],fontsize=6.5,color='#52514e')
    for r in range(3): axes[r,j].set_xticks([]); axes[r,j].set_yticks([]); [sp.set_visible(False) for sp in axes[r,j].spines.values()]
axes[0,0].set_ylabel('input',fontsize=7,color='#0b0b0b'); axes[1,0].set_ylabel('channels SDrop\ndrops most (top 10%)',fontsize=7,color='#0b0b0b'); axes[2,0].set_ylabel('channels it\nleaves alone (bottom 50%)',fontsize=7,color='#0b0b0b')
fig.subplots_adjust(wspace=0.04,hspace=0.12,left=0.09,right=0.995,top=0.93,bottom=0.01); fig.savefig(a.out,dpi=200); print('saved',a.out)
