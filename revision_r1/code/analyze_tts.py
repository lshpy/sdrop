"""H-J: 테스트 시간 억제 — ERM(none) 체크포인트에, 학습 없이, 추론 시 지배 채널을 드롭한 K개 마스크의 예측을 평균.

    python analyze_tts.py [--K 16] [--n N] [--out results/wb_tts.csv]
"""
import argparse, csv, re, torch
from pathlib import Path
from _wb_common import build_from_ckpt, test_loader, labels_places, group_metrics

ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=16); ap.add_argument('--n', type=int, default=None)
ap.add_argument('--ckpt_dir', default='./checkpoints'); ap.add_argument('--out', default='./results/wb_tts.csv')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--workers', type=int, default=8)
args = ap.parse_args()
loader = test_loader(n=args.n, workers=args.workers); labels, places = labels_places(loader)
rows = []
for ck_path in sorted(Path(args.ckpt_dir).glob('waterbirds_none_rate0.1_none_ep30_seed*_best.pth')):
    seed = int(re.search(r'seed(\d)', ck_path.name).group(1))
    if seed > 2: continue
    ck = torch.load(ck_path, map_location=args.device, weights_only=False)
    configs = [('plain', 'none', 0.0)] + [(m, m, p) for m in ('sdrop_energy', 'sdrop_gap', 'sdrop', 'sdrop_random') for p in (0.1, 0.2, 0.3)]
    for name, method, p in configs:
        model = build_from_ckpt(ck, args.device, method=method, drop_rate=p, layers=([] if method == 'none' else ['L4']), norm='mean')
        model.eval()
        if method != 'none': model.sdrop_l4.train()      # BN 은 eval, 마스킹만 켠다
        K = 1 if method == 'none' else args.K
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(args.device); prob = 0
                for _ in range(K): prob = prob + torch.softmax(model(x), 1)
                preds.append((prob / K).argmax(1).cpu())
        preds = torch.cat(preds); gm = group_metrics(preds, labels, places)
        rows.append(dict(seed=seed, config=name, p=p, K=K, **gm))
        print(f"seed{seed} {name:13} p={p:.1f} K={K:2d}  worst={gm['worst_group']:.2f} avg={gm['avg']:.2f}  g01={gm['g01']:.1f} g10={gm['g10']:.1f}")
        del model; torch.cuda.empty_cache()
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[H-J] {len(rows)} rows → {out}")
