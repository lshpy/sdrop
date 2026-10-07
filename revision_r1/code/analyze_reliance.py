"""Does SDrop reduce the model's reliance on the background?  Inference only, on Waterbirds test images.

Stimuli per image (as in analyze_intervention.py):
  swap     : the centre 112x112 crop (the bird) pasted on a 2x2 mosaic of the four corner patches of a random test
             image with the OTHER background type -> "follows background" = prediction equals the pasted place.
  bg_only  : the corner mosaic of the image itself, no bird -> "follows background" = prediction equals its place.
  obj_only : the centre crop upsampled (background mostly removed) -> accuracy.
Lower "follows background" and higher object-only accuracy mean less reliance on the background.

Usage: python analyze_reliance.py <test_parquet> <out_csv> <ckpt> [<ckpt> ...]
"""
import sys, io, re, os, numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0, '.')
from model import build_model

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
d = pd.read_parquet(sys.argv[1]); out_csv = sys.argv[2]; ckpts = sys.argv[3:]
rng = np.random.RandomState(0)
N_PER = int(os.environ.get('N_PER', 300))
sel = np.concatenate([rng.choice(np.where((d.label == y) & (d.place == p))[0], N_PER, replace=False)
                      for y in (0, 1) for p in (0, 1)])
label = d.label.values[sel]; place = d.place.values[sel]


def load(i):
    return Image.open(io.BytesIO(d.iloc[i]['image']['bytes'])).convert('RGB').resize((256, 256)).crop((16, 16, 240, 240))


def mosaic(img):
    W, c = 224, 74
    q = [img.crop((0, 0, c, c)), img.crop((W - c, 0, W, c)), img.crop((0, W - c, c, W)), img.crop((W - c, W - c, W, W))]
    m = Image.new('RGB', (W, W))
    for k, qi in enumerate(q):
        m.paste(qi.resize((112, 112)), ((k % 2) * 112, (k // 2) * 112))
    return m


imgs = [load(i) for i in sel]
# partner for the swap: a random selected image with the other background type
partner = np.array([rng.choice(np.where(place != place[j])[0]) for j in range(len(sel))])
stim = {
    'swap': [(lambda m, o: (m.paste(o.crop((56, 56, 168, 168)), (56, 56)), m)[1])(mosaic(imgs[partner[j]]), imgs[j]) for j in range(len(sel))],
    'bg_only': [mosaic(im) for im in imgs],
    'obj_only': [im.crop((56, 56, 168, 168)).resize((224, 224)) for im in imgs],
    'full': imgs,
}
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
X = {k: torch.stack([norm(im) for im in v]) for k, v in stim.items()}

rows = []
for ck in ckpts:
    rid = ck.split('/')[-1].replace('_best.pth', '')
    arm = re.sub(r'_seed\d+$', '', rid).replace('waterbirds_', ''); seed = int(re.search(r'seed(\d+)', rid).group(1))
    model = build_model('resnet50', 2, 'none', 0.0, [], pretrained=False)
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    res = model.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd, strict=False)
    bad = [k for k in res.missing_keys if 'sdrop' not in k] + [k for k in res.unexpected_keys if 'sdrop' not in k]
    assert not bad, f'{rid}: weights do not match the model: {bad[:5]}'
    model.eval().to(dev)
    pred = {}
    with torch.no_grad():
        for k, x in X.items():
            pred[k] = torch.cat([model(x[i:i + 64].to(dev)).argmax(1).cpu() for i in range(0, len(x), 64)]).numpy()
    minority = label != place
    r = dict(arm=arm, seed=seed,
             swap_follow_bg=100 * (pred['swap'] == place[partner]).mean(),
             swap_follow_bg_minority=100 * (pred['swap'][minority] == place[partner][minority]).mean(),
             bg_only_follow_bg=100 * (pred['bg_only'] == place).mean(),
             obj_only_acc=100 * (pred['obj_only'] == label).mean(),
             full_acc=100 * (pred['full'] == label).mean())
    rows.append(r)
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
pd.set_option('display.width', 200)
print(df.groupby('arm')[['swap_follow_bg', 'swap_follow_bg_minority', 'bg_only_follow_bg', 'obj_only_acc', 'full_acc']].mean().round(1).to_string())
