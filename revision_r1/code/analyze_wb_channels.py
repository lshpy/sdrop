"""H-H: SDrop 점수가 높은(=자주 드롭되는) 채널은 배경(place)에 선택적인가, 새(label)에 선택적인가?

각 Waterbirds L4 체크포인트에 대해 테스트셋에서
  - 채널별 활성 GAP(|x|) 행렬 (N, C)
  - 세 점수(energy / gap / egpg)의 드롭 확률 평균 (C,)
  - 채널별 AUC(활성 → place), AUC(활성 → label)  → 선택성 = |AUC − 0.5|
를 계산하고, 드롭 확률과 두 선택성의 Spearman 상관, 상위 10% 드롭 채널의 선택성 평균을 낸다.
ERM(none) 체크포인트도 포함 — 학습 전 특징에서 점수가 무엇을 가리키는지가 가장 깨끗한 검증.

    python analyze_wb_channels.py [--n N] [--out results/wb_channels.csv]
"""
import argparse, csv, glob, re, torch, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sdrop import compute_score, _drop_probs_from_scores
from _wb_common import build_from_ckpt, test_loader, labels_places

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=None); ap.add_argument('--ckpt_dir', default='./checkpoints')
ap.add_argument('--out', default='./results/wb_channels.csv'); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--dump', default=None, help='채널별 배열 npz 저장'); ap.add_argument('--only', default=None, help='run_id 부분 문자열 필터'); ap.add_argument('--root', default='./data')
args = ap.parse_args()
loader = test_loader(root=args.root, n=args.n, workers=args.workers); labels, places = labels_places(loader)
pat = re.compile(r'waterbirds_(none|sdrop_energy|sdrop_gap|sdrop)_rate0.1_(none|L4)_ep30(?:_nmmean)?_seed([0-2])_best\.pth$')
rows = []
for ck_path in sorted(Path(args.ckpt_dir).glob('waterbirds_*_best.pth')):
    m = pat.search(ck_path.name)
    if not m: continue
    method, seed = m.group(1), int(m.group(3))
    if args.only and args.only not in ck_path.name: continue
    ck = torch.load(ck_path, map_location=args.device, weights_only=False)
    model = build_from_ckpt(ck, args.device, method='none', layers=[])   # 특징만 필요, 드롭 모듈 없이
    feats = {}
    h = model.layer4.register_forward_hook(lambda mod, i, o: feats.__setitem__('x', o.detach()))
    acts, probs = [], {'energy': [], 'gap': [], 'egpg': []}
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(args.device)); f = feats['x'].float()
            acts.append(f.abs().mean(dim=(2, 3)).cpu())
            for sc in probs:
                s = compute_score(f, sc, 1.0, 1.0, 'max', 1e-6)
                probs[sc].append(_drop_probs_from_scores(s, 0.1, 'mean', 1e-6).cpu())
    h.remove()
    A = torch.cat(acts).numpy(); P = {k: torch.cat(v).mean(0).numpy() for k, v in probs.items()}
    y, g = labels.numpy(), places.numpy()
    auc_l = np.array([roc_auc_score(y, A[:, c]) if A[:, c].std() > 0 else 0.5 for c in range(A.shape[1])])
    auc_p = np.array([roc_auc_score(g, A[:, c]) if A[:, c].std() > 0 else 0.5 for c in range(A.shape[1])])
    sel_l, sel_p = np.abs(auc_l - .5), np.abs(auc_p - .5)
    if args.dump: np.savez(args.dump.replace('.npz','')+'_'+ck_path.name[:-9]+'.npz', p_energy=P['energy'], p_gap=P['gap'], p_egpg=P['egpg'], sel_place=sel_p, sel_label=sel_l, auc_place=auc_p, auc_label=auc_l, meanact=A.mean(0))
    for sc, p in P.items():
        top = p >= np.quantile(p, 0.9); rest = ~top
        r = dict(run=ck_path.name[:-9], method=method, seed=seed, score=sc,
                 rho_place=spearmanr(p, sel_p).correlation, rho_label=spearmanr(p, sel_l).correlation,
                 rho_bg_minus_bird=spearmanr(p, sel_p - sel_l).correlation,
                 top10_sel_place=sel_p[top].mean(), rest_sel_place=sel_p[rest].mean(),
                 top10_sel_label=sel_l[top].mean(), rest_sel_label=sel_l[rest].mean(),
                 top10_meanact=A[:, top].mean(), n_channels=A.shape[1], n_images=A.shape[0])
        rows.append(r)
        print(f"{r['run']:52} {sc:6} rho(place)={r['rho_place']:+.3f} rho(label)={r['rho_label']:+.3f} rho(bg-bird)={r['rho_bg_minus_bird']:+.3f}  top10%: sel_place {r['top10_sel_place']:.3f} vs rest {r['rest_sel_place']:.3f} | sel_label {r['top10_sel_label']:.3f} vs rest {r['rest_sel_label']:.3f}")
    del model; torch.cuda.empty_cache()
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[H-H] {len(rows)} rows → {out}")
