"""Rebuild Waterbirds (Sagawa et al. 2020) with the bird scaled by a factor s -> data/waterbirds_s<s>/.

Reverse test of the object-size hypothesis (2026-09-29): SDrop's gain should SHRINK as the bird grows.
Same birds, same backgrounds, same pairings, same splits as the released Waterbirds
(waterbird_complete95_forest2water2): every row of waterbirds_metadata.csv (the release's metadata.csv:
img_filename, y, split, place, place_filename) is re-composed with the original recipe from
kohpangwei/group_DRO dataset_scripts/{generate_waterbirds.py,dataset_utils.py}:
    seg  = CUB segmentation PNG / 255 (soft, 3 channels)            bird image keeps its native size
    bg   = Places365 image crop_and_resize'd to the bird image size (centre crop to its aspect, LANCZOS)
    out  = round(bg * (1 - seg)) + round(img * seg)                  saved as JPEG (PIL default quality 75)
Only change for s != 1: the bird (img and seg over the bbox of seg > 0) is resized by s (LANCZOS for the
image, bilinear for the mask) and pasted with its bbox centre at the ORIGINAL bbox centre, shifted the
minimum amount needed to stay inside the frame; if the scaled bbox is larger than the frame it is centre-
cropped (the bird gets clipped; the fraction clipped is printed). The background is untouched, so s is the
only variable. s = 1 takes the identical code path with no resize = the original image.

Outputs (Waterbirds parquet format, loads with dataset.get_waterbirds(name='waterbirds_s2.0')):
    <out>/{train,validation,test}-00000-of-00001.parquet   image{bytes,path}, label, place, bird
    <out>/test_masks.npz   'mask' (N_test, 224, 224) float16, the pasted bird mask resized to 224x224, row
                           order = test parquet (for analyze_bgshare.py, which resizes it 224->256->crop 224)

Sources (no login): CUB images = data/CUB_200_2011/images (already on the volume); segmentations.tgz
(38 MB) from data.caltech.edu; Places365 backgrounds = 4 categories cut out of the MIT
train_large_places365standard.tar (112.6 GB) with HTTP range requests (~1.3 GB, see fetch-places).

Usage:
    python prep_waterbirds_scaled.py fetch-places --out data/places_wb          # once, ~1.3 GB
    python prep_waterbirds_scaled.py build --scale 2.0 --out data/waterbirds_s2.0 \
        [--cub data/CUB_200_2011] [--seg data/segmentations.tgz] [--places data/places_wb] [--limit N]
    python prep_waterbirds_scaled.py compare --out data/waterbirds_s1.0 --ref data/waterbirds   # s=1 vs HF copy
"""
import argparse, io, sys, tarfile, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
META = HERE / 'waterbirds_metadata.csv'
SEG_URL = 'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz'
PLACES_TAR = 'https://data.csail.mit.edu/places/places365/train_large_places365standard.tar'
CATS = ['b/bamboo_forest', 'f/forest/broadleaf', 'o/ocean', 'l/lake/natural']   # land, land, water, water
SPLITS = {0: 'train', 1: 'validation', 2: 'test'}


# ---------------------------------------------------------------- Places365: 4 categories via range requests
def _get(url, a, b):
    req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b - 1}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _header_at(url, off, span=1 << 20):
    """First valid tar header at or after byte `off` -> (position, name). Headers are 512-aligned."""
    a = off - off % 512
    buf = _get(url, a, a + span)
    for i in range(0, len(buf) - 511, 512):
        h = buf[i:i + 512]
        if h[257:262] != b'ustar':
            continue
        try:
            ok = int(h[148:156].split(b'\0')[0].strip() or b'0', 8) == sum(h[:148]) + 256 + sum(h[156:])
        except ValueError:
            continue
        if ok:
            return a + i, h[:100].split(b'\0')[0].decode()
    raise RuntimeError(f'no tar header in {span} bytes at {off}')


def fetch_places(out, url=PLACES_TAR):
    """The MIT tar is sorted by path, so binary-search each category's first header, then stream from there
    with tarfile and keep members until the prefix ends. Downloads only the 4 categories (~1.3 GB)."""
    size = int(urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60).headers['Content-Length'])
    for cat in CATS:
        pre = f'data_large/{cat}/'
        lo, hi = 0, size
        while hi - lo > (8 << 20):
            mid = (lo + hi) // 2
            pos, name = _header_at(url, mid)
            if name < pre: lo = pos
            else: hi = mid
        start, _ = _header_at(url, lo)
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-'})
        n = 0
        with urllib.request.urlopen(req, timeout=120) as r, tarfile.open(fileobj=r, mode='r|') as t:
            for m in t:
                if m.name.startswith(pre):
                    if m.isfile():
                        p = Path(out) / m.name; p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(t.extractfile(m).read()); n += 1
                elif m.name > pre:
                    break
        print(f'[places] {cat}: {n} images', flush=True)


# ---------------------------------------------------------------- the original recipe (group_DRO dataset_utils.py)
def crop_and_resize(src, tw, th):
    sw, sh = src.size
    if sw < tw or sh < th:
        wr = (tw, int((tw / sw) * sh))
        if wr[0] >= tw and wr[1] >= th:
            src = src.resize(wr, Image.LANCZOS)
        else:
            src = src.resize((int((th / sh) * sw), th), Image.LANCZOS)
        return crop_and_resize(src, tw, th)
    if sw / sh > tw / th:
        nw = int(tw / th * sh); o = (sw - nw) // 2; box = (o, 0, sw - o, sh)
    else:
        nh = int(sw / (tw / th)); o = (sh - nh) // 2; box = (0, o, sw, sh - o)
    return src.crop(box).resize((tw, th), Image.LANCZOS)


def scale_bird(img, seg, s):
    """img uint8 (H,W,3), seg float (H,W,3) in [0,1] -> the bird resized by s, pasted at its bbox centre."""
    if s == 1.0:
        return img, seg, 0.0
    H, W = seg.shape[:2]
    ys, xs = np.nonzero(seg[..., 0] > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = max(1, round((y1 - y0) * s)), max(1, round((x1 - x0) * s))
    bi = np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((w, h), Image.LANCZOS))
    bm = np.asarray(Image.fromarray(np.ascontiguousarray(seg[y0:y1, x0:x1, 0]).astype(np.float32)).resize((w, h), Image.BILINEAR))
    bm = np.clip(bm, 0, 1)
    full = bm.sum()
    # target top-left: centred on the original bbox centre, clamped into the frame; if bigger than the frame, crop
    def place(c, n, N):                                    # -> (dst0, dst1, src0, src1)
        t = int(round(c - n / 2))
        if n <= N:
            t = min(max(t, 0), N - n); return t, t + n, 0, n
        k = (n - N) // 2; return 0, N, k, k + N
    ty0, ty1, sy0, sy1 = place((y0 + y1) / 2, h, H)
    tx0, tx1, sx0, sx1 = place((x0 + x1) / 2, w, W)
    oi = np.zeros_like(img); om = np.zeros((H, W), np.float32)
    oi[ty0:ty1, tx0:tx1] = bi[sy0:sy1, sx0:sx1]; om[ty0:ty1, tx0:tx1] = bm[sy0:sy1, sx0:sx1]
    return oi, np.repeat(om[..., None], 3, axis=2), 1 - om.sum() / max(full, 1e-6)


def compose(img_b, seg_b, place_b, s):
    img = np.asarray(Image.open(io.BytesIO(img_b)).convert('RGB'))
    seg = np.asarray(Image.open(io.BytesIO(seg_b)).convert('RGB')) / 255
    img, seg, clipped = scale_bird(img, seg, s)
    black = np.around(img * seg).astype(np.uint8)
    bg = np.asarray(crop_and_resize(Image.open(io.BytesIO(place_b)).convert('RGB'), img.shape[1], img.shape[0]))
    out = np.around(bg * (1 - seg)).astype(np.uint8) + black
    buf = io.BytesIO(); Image.fromarray(out).save(buf, 'JPEG')               # PIL default quality = release
    return buf.getvalue(), seg[..., 0], clipped


# ---------------------------------------------------------------- build
def _seg_reader(path):
    p = Path(path)
    if p.is_dir():
        return lambda f: (p / f).read_bytes()
    if not p.exists():
        print(f'[seg] downloading {SEG_URL}', flush=True); urllib.request.urlretrieve(SEG_URL, p)
    with tarfile.open(p) as t:
        d = {m.name.split('segmentations/', 1)[1]: t.extractfile(m).read() for m in t if m.isfile() and m.name.endswith('.png')}
    return d.__getitem__


def build(o):
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    meta = pd.read_csv(o.meta)
    if o.only:                                        # local smoke test: rows whose files exist
        meta = meta[[(Path(o.cub) / 'images' / f).exists() and (Path(o.places) / 'data_large' / pf[1:]).exists()
                     for f, pf in zip(meta.img_filename, meta.place_filename)]]
    seg = _seg_reader(o.seg)
    out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    for k, name in SPLITS.items():
        d = meta[meta.split == k]
        if o.limit: d = d.head(o.limit)
        imgs, masks, area, clip = [], [], [], []
        for f, pf in zip(d.img_filename, d.place_filename):
            b, m, c = compose((Path(o.cub) / 'images' / f).read_bytes(), seg(f.replace('.jpg', '.png')),
                              (Path(o.places) / 'data_large' / pf[1:]).read_bytes(), o.scale)
            imgs.append({'bytes': b, 'path': Path(f).name}); area.append(float(m.mean())); clip.append(c)
            if name == 'test':
                masks.append(np.asarray(Image.fromarray(m.astype(np.float32)).resize((224, 224), Image.BILINEAR), np.float16))
        tab = pa.table({'image': imgs, 'label': d.y.to_numpy(), 'place': d.place.to_numpy(),
                        'bird': [int(f[:3]) - 1 for f in d.img_filename]})
        pq.write_table(tab, out / f'{name}-00000-of-00001.parquet')
        if name == 'test':
            np.savez_compressed(out / 'test_masks.npz', mask=np.stack(masks) if masks else np.zeros((0, 224, 224), np.float16))
        print(f'[wb s={o.scale}] {name}: {len(d)} images, bird area {np.mean(area) if area else 0:.3f}, '
              f'clipped {np.mean(clip) if clip else 0:.3f} of bird mass (mean), >5% clipped in '
              f'{np.mean(np.array(clip) > 0.05) if clip else 0:.3f} of images', flush=True)


def compare(o):
    """Mean |diff| (0-255) between the rebuilt images and a reference Waterbirds parquet, matched by filename."""
    import pyarrow.parquet as pq
    for name in SPLITS.values():
        p, r = Path(o.out) / f'{name}-00000-of-00001.parquet', Path(o.ref) / f'{name}-00000-of-00001.parquet'
        if not (p.exists() and r.exists()): continue
        ref = {x['path']: x['bytes'] for x in pq.read_table(r, columns=['image']).column('image').to_pylist()}
        t = pq.read_table(p).to_pylist(); diffs = []
        for x in t:
            rb = ref.get(x['image']['path'])
            if rb is None: continue
            a = np.asarray(Image.open(io.BytesIO(x['image']['bytes'])).convert('RGB'), np.float32)
            b = np.asarray(Image.open(io.BytesIO(rb)).convert('RGB'), np.float32)
            diffs.append(np.abs(a - b).mean() if a.shape == b.shape else np.inf)
        if diffs:
            print(f'[compare] {name}: {len(diffs)} matched, mean|diff| {np.mean(diffs):.2f}/255, max {np.max(diffs):.2f}, '
                  f'bytes-identical {sum(x["image"]["bytes"] == ref.get(x["image"]["path"]) for x in t)}', flush=True)


def _selftest():
    img = np.full((40, 60, 3), 200, np.uint8); seg = np.zeros((40, 60, 3)); seg[15:25, 25:35] = 1
    i2, s2, c = scale_bird(img, seg, 2.0)
    assert c < 1e-6 and abs(s2[..., 0].sum() - 400) < 1 and s2[10:30, 20:40, 0].min() > 0.99, 'centred x2 paste'
    i3, s3, c3 = scale_bird(img, seg, 5.0)                 # 50x50 bird in a 40x60 frame -> clipped vertically
    assert s3.shape == seg.shape and 0.1 < c3 < 0.3, c3
    assert scale_bird(img, seg, 1.0)[1] is seg
    print('selftest ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    f = sp.add_parser('fetch-places'); f.add_argument('--out', default='data/places_wb')
    b = sp.add_parser('build')
    b.add_argument('--scale', type=float, required=True); b.add_argument('--out', required=True)
    b.add_argument('--cub', default='data/CUB_200_2011'); b.add_argument('--seg', default='data/segmentations.tgz')
    b.add_argument('--places', default='data/places_wb'); b.add_argument('--meta', default=str(META))
    b.add_argument('--limit', type=int, default=0, help='smoke test: first N rows per split')
    b.add_argument('--only', action='store_true', help='smoke test: keep only rows whose image+place files exist')
    c = sp.add_parser('compare'); c.add_argument('--out', required=True); c.add_argument('--ref', default='data/waterbirds')
    sp.add_parser('selftest')
    o = ap.parse_args()
    {'fetch-places': lambda: fetch_places(o.out), 'build': lambda: build(o), 'compare': lambda: compare(o),
     'selftest': _selftest}[o.cmd]()
