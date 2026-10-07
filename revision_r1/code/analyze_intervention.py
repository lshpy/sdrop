"""Input interventions on Waterbirds test images with the ERM model: does the dominant tenth of L4 channels
respond to the background or to the bird?  Stimuli per image: full; background-only (2x2 mosaic of the four
corner patches, bird removed); object-only (centre crop, background mostly removed); background-swapped (centre
crop of this image on the corner-mosaic of an image with the OTHER background type).
Reports, for the 10% most active channels (defined on full images) and the rest, the mean energy under each
stimulus relative to the full image, and the ERM prediction under the swap.  Usage: python analyze_intervention.py <ckpt_dir> <parquet> [n_per_group]"""
import sys, io, glob, re, numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, '.')
from model import build_model
from dataset import waterbirds_transforms
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
_, val_tf = waterbirds_transforms()
d = pd.read_parquet(sys.argv[2]); n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rng = np.random.RandomState(0)
idx = {}
for y in (0, 1):
    for p in (0, 1):
        ii = np.where((d.label == y) & (d.place == p))[0]; idx[(y, p)] = rng.choice(ii, min(n_per, len(ii)), replace=False)
def load(i): return Image.open(io.BytesIO(d.iloc[i]['image']['bytes'])).convert('RGB').resize((256, 256)).crop((16, 16, 240, 240))
def mosaic(img):  # four 74px corners -> 2x2 of 112px
    W = 224; c = 74; q = [img.crop((0, 0, c, c)), img.crop((W - c, 0, W, c)), img.crop((0, W - c, c, W)), img.crop((W - c, W - c, W, W))]
    m = Image.new('RGB', (W, W)); [m.paste(qi.resize((112, 112)), ((k % 2) * 112, (k // 2) * 112)) for k, qi in enumerate(q)]; return m
def centre(img): return img.crop((56, 56, 168, 168)).resize((224, 224))
def swap(obj_img, bg_img):  # centre crop of obj pasted on the mosaic of bg
    m = mosaic(bg_img); m.paste(obj_img.crop((56, 56, 168, 168)), (56, 56)); return m
from torchvision import transforms
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
feats = {}
def hook(m, inp): feats['x'] = inp[0].detach()
for ck in sorted(glob.glob(f'{sys.argv[1]}/waterbirds_none_*_best.pth')):
    model = build_model('resnet50', 2, 'none', 0.1, [], pretrained=False); sd = torch.load(ck, map_location='cpu')
    model.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd); model.eval().to(dev); model.sdrop_l4.register_forward_pre_hook(hook)
    def run(imgs):
        E, P = [], []
        for k in range(0, len(imgs), 32):
            x = torch.stack([norm(im) for im in imgs[k:k + 32]]).to(dev)
            with torch.no_grad(): logits = model(x)
            E.append(feats['x'].float().pow(2).mean((2, 3)).cpu()); P.append(logits.argmax(1).cpu())
        return torch.cat(E).numpy(), torch.cat(P).numpy()
    allidx = np.concatenate(list(idx.values())); imgs = [load(i) for i in allidx]; y = d.label.values[allidx]; p = d.place.values[allidx]
    E_full, pred_full = run(imgs); dom = np.argsort(-E_full.mean(0))[: E_full.shape[1] // 10]; rest = np.setdiff1d(np.arange(E_full.shape[1]), dom)
    E_bg, pred_bg = run([mosaic(im) for im in imgs]); E_obj, pred_obj = run([centre(im) for im in imgs])
    # background swap: pair each image with a random image of the other background
    other = {0: np.where(p == 1)[0], 1: np.where(p == 0)[0]}
    partners = np.array([rng.choice(other[pp]) for pp in p]); E_sw, pred_sw = run([swap(imgs[i], imgs[partners[i]]) for i in range(len(imgs))])
    def rel(E): return E[:, dom].mean() / E_full[:, dom].mean(), E[:, rest].mean() / E_full[:, rest].mean()
    print(f"== {ck.split('/')[-1][:45]}  (n={len(imgs)}, {dev})")
    for name, E in [('background-only', E_bg), ('object-only', E_obj), ('background-swapped', E_sw)]:
        a, b = rel(E); print(f"  energy relative to full image: dominant 10% {a:.2f}   rest {b:.2f}   [{name}]")
    acc = lambda pr: 100 * (pr == y).mean(); bgpred = lambda pr, pl: 100 * (pr == pl).mean()
    print(f"  ERM prediction: accuracy full {acc(pred_full):.1f} | background-only predicts the background class {bgpred(pred_bg, p):.1f}% | object-only accuracy {acc(pred_obj):.1f} | after background swap: follows the NEW background {bgpred(pred_sw, 1 - p):.1f}%, still correct {acc(pred_sw):.1f}%")
