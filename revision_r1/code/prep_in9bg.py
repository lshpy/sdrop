"""Build data/in9bg/{train,validation,test}-00000-of-00001.parquet (Waterbirds format) from ImageNet-9.

Source: Xiao, Engstrom, Ilyas & Madry, "Noise or Signal: The Role of Image Backgrounds in Object
Recognition" (ICLR 2021), MadryLab/backgrounds_challenge release (280 MB): 9 super-classes x 450 images,
each with an object mask (fg_mask/*.npy) and a background with the object region tiled over (only_bg_t).
Only the evaluation images are released (the IN-9 train set needs ImageNet), so we build a Waterbirds-style
benchmark from them. Pre-registered construction (PREREG_SCREEN_DATASETS.md):
  per class, a seeded permutation of the 450 object ids -> 300 train / 50 validation / 100 test, and the
  same for the 450 background ids (only_bg_t is a partly different image set: 1,516 of its 4,050 ids
  are also objects), so no background image is shared between splits
  composite = mask * original_i + (1 - mask) * only_bg_t_j   (j from the same split's background pool,
              never the object's own photo)
  label = IN-9 class (0..8);  place = IN-9 class of the background image j (the confound)
  train : place = label with probability RHO, else uniform over the other 8 classes
  valid : place uniform over the 9 classes (independent of label)
  test  : every test object pasted on one random background of EACH class -> 9 x 9 cells x 100 images
Stored at 256x256 (bilinear up from 224), JPEG q=90, so waterbirds_transforms applies unchanged.

Usage: python prep_in9bg.py --tar backgrounds_challenge_data.tar.gz --out data/in9bg [--limit N per class]
"""
import argparse, io, tarfile
from pathlib import Path
import numpy as np

RHO = 0.9          # 9-way confound: 0.9 leaves ~3.8 train images per off-diagonal cell (0.95 -> ~1.9)
SPLIT = {'train': 300, 'validation': 50, 'test': 100}
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', required=True); ap.add_argument('--out', default='data/in9bg')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: images per class per split')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='object scale (2026-09-29 object-size experiment): the masked object is cropped to its '
                         'bounding box, resized by this factor and pasted at a seeded random position; 1.0 = original')
    ap.add_argument('--classes', type=str, default='',
                    help='2026-10-01 "shared-axis" experiment (prereg/PREREG_in9wb_2026-10-01.md): comma list of IN-9 '
                         'object classes to keep (labels remapped to 0..K-1). Default: all 9.')
    ap.add_argument('--bgs', type=str, default='',
                    help='comma list of IN-9 classes whose only_bg_t images form the background pools (place = index in '
                         'this list, paired with --classes position-wise for the train correlation). Default: same as classes.')
    ap.add_argument('--rho', type=float, default=RHO, help='P(train place = paired background)')
    ap.add_argument('--save-masks', action='store_true',
                    help='also write <out>/test_masks.npz: the pasted object mask of every test image at 224x224 '
                         '(same RNG stream, so identical to the parquet built without this flag)')
    o = ap.parse_args()
    from PIL import Image
    import pyarrow as pa, pyarrow.parquet as pq
    orig, bg, mask = {}, {}, {}
    with tarfile.open(o.tar, 'r|*') as t:
        for m in t:
            p = Path(m.name)
            if not m.isfile() or len(p.parts) < 5: continue
            kind, cls, k = p.parts[1], int(p.parts[3][:2]), (int(p.parts[3][:2]), p.stem)
            if kind == 'original': orig[k] = t.extractfile(m).read()
            elif kind == 'only_bg_t': bg[k] = t.extractfile(m).read()
            elif kind == 'fg_mask': mask[k] = np.load(io.BytesIO(t.extractfile(m).read()))
    assert set(orig) == set(mask) and len(orig) == len(bg) == 4050, 'incomplete archive'
    rng = np.random.default_rng(SEED)
    CLS = [int(c) for c in o.classes.split(',')] if o.classes else list(range(9))
    BGS = [int(c) for c in o.bgs.split(',')] if o.bgs else list(CLS)
    assert len(BGS) == len(CLS), '--bgs must pair position-wise with --classes'

    def cut(keys):                                              # {split: {class: [keys]}}
        d = {s: {} for s in SPLIT}
        for c in range(9):
            ks = sorted(k for k in keys if k[0] == c); ks = [ks[i] for i in rng.permutation(len(ks))]
            a = 0
            for s, n in SPLIT.items():
                d[s][c] = ks[a:a + n][:o.limit or n]; a += n
        return d
    split, bsplit = cut(orig), cut(bg)
    rgb = lambda b: np.asarray(Image.open(io.BytesIO(b)).convert('RGB'), dtype=np.float32)

    prng = np.random.default_rng(SEED + 1)                      # placement RNG, separate from the split RNG

    def compose(k, j):
        m = mask[k][..., None].astype(np.float32)
        if o.scale == 1.0:
            im = Image.fromarray((m * rgb(orig[k]) + (1 - m) * rgb(bg[j])).round().astype(np.uint8))
            last_mask[0] = m[..., 0]
        else:
            fg, B = rgb(orig[k]), rgb(bg[j]); H, W = B.shape[:2]
            ys, xs = np.nonzero(m[..., 0] > 0.5)
            if len(ys) == 0:
                y0, y1, x0, x1 = 0, fg.shape[0], 0, fg.shape[1]
            else:
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            h, w = max(1, round((y1 - y0) * o.scale)), max(1, round((x1 - x0) * o.scale))
            crop = Image.fromarray(fg[y0:y1, x0:x1].astype(np.uint8)).resize((w, h), Image.BILINEAR)
            mc = Image.fromarray((m[y0:y1, x0:x1, 0] * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            ty, tx = int(prng.integers(0, H - h + 1)), int(prng.integers(0, W - w + 1))
            mm = np.zeros((H, W, 1), np.float32); ff = np.zeros_like(B)
            mm[ty:ty + h, tx:tx + w, 0] = np.asarray(mc, np.float32) / 255; ff[ty:ty + h, tx:tx + w] = np.asarray(crop, np.float32)
            im = Image.fromarray((mm * ff + (1 - mm) * B).round().astype(np.uint8))
            area.append(float(mm.mean())); last_mask[0] = mm[..., 0]
        out = io.BytesIO(); im.resize((256, 256), Image.BILINEAR).save(out, 'JPEG', quality=90)
        return out.getvalue()

    area = []; last_mask = [None]; test_masks = []
    out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    for s in SPLIT:
        pool, bpool = split[s], bsplit[s]
        rows = []                                                   # (fg key, bg key)
        for ci, c in enumerate(CLS):
            for k in pool[c]:
                if s == 'train':
                    bcs = [BGS[ci]] if rng.random() < o.rho else [rng.choice([d for d in BGS if d != BGS[ci]])]
                elif s == 'validation':
                    bcs = [BGS[int(rng.integers(len(BGS)))]]
                else:
                    bcs = list(BGS)
                for b in bcs:
                    cand = [j for j in bpool[b] if j[1] != k[1]]
                    rows.append((k, cand[rng.integers(len(cand))]))
        imgs = []
        for k, j in rows:
            imgs.append({'bytes': compose(k, j), 'path': f'{k[1]}__bg_{j[1]}.jpg'})
            if s == 'test' and o.save_masks: test_masks.append(np.asarray(last_mask[0], np.float16))
        tab = pa.table({'image': imgs,
                        'label': [CLS.index(k[0]) for k, _ in rows], 'place': [BGS.index(j[0]) for _, j in rows],
                        'fg_id': [k[1] for k, _ in rows], 'bg_id': [j[1] for _, j in rows]})
        pq.write_table(tab, out / f'{s}-00000-of-00001.parquet')
        if s == 'test' and o.save_masks: np.savez_compressed(out / 'test_masks.npz', mask=np.stack(test_masks))
        lab, pl = np.array(tab.column('label')), np.array(tab.column('place'))
        print(f"[in9bg] {s}: {len(rows)} images, P(place = label) = {(lab == pl).mean():.3f}"
              + (f", object area {np.mean(area):.3f}" if area else ''), flush=True); area.clear()


if __name__ == '__main__':
    main()
