"""CUB-200 test 이미지에서 Gradient×Input 귀속 질량이 바운딩박스 안에 얼마나 있는지.

원고의 '중앙 16×16 vs 테두리' 대용 지표를 실제 객체 마스크 기준으로 바꾼다 (예측 P4).
checkpoints/cub200_*_best.pth 전부에 대해 계산, results/cub_bbox_attribution.csv 로.
재실행 가능: 이미 계산된 run_id 는 건너뛴다.

    python analyze_cub_bbox.py [--n 500] [--device cuda]
"""
import argparse, csv, inspect, io, sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='./data/CUB_200_2011')
ap.add_argument('--ckpt_dir', default='./checkpoints')
ap.add_argument('--out', default='./results/cub_bbox_attribution.csv')
ap.add_argument('--n', type=int, default=500)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
root = Path(args.root)

# ---- 메타데이터: id -> path, label, bbox, split ----
id2path = dict(l.split() for l in open(root / 'images.txt').read().strip().splitlines())
id2lab = {a: int(b) - 1 for a, b in (l.split() for l in open(root / 'image_class_labels.txt').read().strip().splitlines())}
id2box = {l.split()[0]: tuple(map(float, l.split()[1:])) for l in open(root / 'bounding_boxes.txt').read().strip().splitlines()}
test_ids = [a for a, b in (l.split() for l in open(root / 'train_test_split.txt').read().strip().splitlines()) if b == '0']
test_ids = test_ids[:args.n]

norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def load(img_id):
    im = Image.open(root / 'images' / id2path[img_id]).convert('RGB')
    W, H = im.size
    x, y, w, h = id2box[img_id]
    im = im.resize((256, 256))                     # dataset.py 의 Resize(256) 은 짧은 변 기준이지만
    sx, sy = 256.0 / W, 256.0 / H                  # 마스크 계산은 정사각 리사이즈로 통일한다(귀속도 같은 입력).
    x0, y0, x1, y1 = x * sx - 16, y * sy - 16, (x + w) * sx - 16, (y + h) * sy - 16   # CenterCrop(224) 오프셋
    im = im.crop((16, 16, 240, 240))
    mask = torch.zeros(224, 224)
    xa, ya, xb, yb = [int(max(0, min(224, round(v)))) for v in (x0, y0, x1, y1)]
    if xb > xa and yb > ya: mask[ya:yb, xa:xb] = 1.0
    return norm(transforms.functional.to_tensor(im)), id2lab[img_id], mask

data = [load(i) for i in test_ids]
X = torch.stack([d[0] for d in data]); Y = torch.tensor([d[1] for d in data]); M = torch.stack([d[2] for d in data])
area_frac = M.mean(dim=(1, 2))                     # 박스가 차지하는 면적 비율 (무작위 귀속의 기대값)

def grad_x_input(model, x, y):
    x = x.clone().requires_grad_(True)
    logit = model(x).gather(1, y.view(-1, 1)).sum()
    g, = torch.autograd.grad(logit, x)
    return (g * x).abs().sum(1)                    # (B, 224, 224)

def build_from_ckpt(ck):
    a = ck['args']
    sig = inspect.signature(build_model).parameters
    kw = {k: a[k] for k in ('peakedness', 'norm', 'gamma', 'schedule', 'grid_size',
                            'rescale', 'rescale_mode', 'select', 'warmup_frac', 'participation', 'width_mult')
          if k in sig and k in a}
    if 'grad_mode' in sig and 'grad_mode' in a: kw['grad_mode'] = a['grad_mode']
    m = build_model(a.get('arch') or 'resnet50', 200, a['method'], a['drop_rate'], a['layers'], **kw)
    m.load_state_dict(ck['state_dict']); return m.to(args.device).eval()

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    done = {r['run_id'] for r in csv.DictReader(open(out))}
rows = []
for ck_path in sorted(Path(args.ckpt_dir).glob('cub200_*_best.pth')):
    rid = ck_path.name[:-len('_best.pth')]
    if rid in done: continue
    ck = torch.load(ck_path, map_location=args.device, weights_only=False)
    try: model = build_from_ckpt(ck)
    except Exception as e: print(f"[skip] {rid}: {e}"); continue
    fr = []
    for i in range(0, len(X), args.bs):
        xb, yb, mb = X[i:i+args.bs].to(args.device), Y[i:i+args.bs].to(args.device), M[i:i+args.bs].to(args.device)
        S = grad_x_input(model, xb, yb)
        S = S / (S.sum(dim=(1, 2), keepdim=True) + 1e-9)
        fr.append((S * mb).sum(dim=(1, 2)).detach().cpu())
    fr = torch.cat(fr)
    row = dict(run_id=rid, n=len(fr), mass_in_bbox=fr.mean().item(), mass_std=fr.std().item(),
               bbox_area_frac=area_frac.mean().item(), lift=(fr.mean() / area_frac.mean()).item(),
               best_epoch=ck.get('epoch'), best_acc=ck.get('best_acc'))
    rows.append(row); print(f"{rid:60} bbox mass={row['mass_in_bbox']:.4f}  (area {row['bbox_area_frac']:.3f}, lift {row['lift']:.3f})")
    del model; torch.cuda.empty_cache()
if rows:
    write_header = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        w.writerows(rows)
print(f"[bbox] {len(rows)} 런 추가 → {out}")
