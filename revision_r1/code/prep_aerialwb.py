"""Build data/aerialwb/{train,validation,test}-00000-of-00001.parquet: an aerial-image Waterbirds (composite).

New domain for the small-object condition (AERIALWB.md). Waterbirds pastes CUB birds (mask) on Places land/water
photos; here we paste iSAID instance masks of aerial PLANES and SHIPS on MLRSNet natural LAND / WATER scenes:
  label 0 = plane (the "land" object), 1 = ship (the "water" object)            (Waterbirds: landbird / waterbird)
  place 0 = land scene, 1 = water scene     -> aligned group  <=>  label == place (same convention as Waterbirds,
                                               so train.py --train_corr / evaluate_groups work unchanged)
Sources (all no-login HF mirrors, revisions pinned in AERIALWB.md):
  objects      iSAID annotations (isaaccorley/isaid, isaid_annotations_{train,val}.tar.gz, COCO polygons)
               on DOTA-v1.0 images (torchgeo/dota, dotav1.0_images_{train,val}.tar.gz)
  backgrounds  MLRSNet (jonathan-roberts1/MLRSNet, 3 parquet shards, 256x256, multi-label)
Construction (seed 0, fixed before any run; see AERIALWB.md for the numbers it produced):
  objects   category plane/ship, iscrowd 0, bbox long side >= MIN_SIDE px, bbox >= 2 px inside the image,
            colour source image (grey GF-2/JL-1 panchromatic images dropped), at most CAP instances per class
            per source image.  iSAID-train images -> our train/val (split by SOURCE IMAGE, VAL_IMG_FRAC),
            iSAID-val images -> our test, so no object scene is shared between train and test.
  backgrounds  water = MLRSNet tags contain 'water' and ('sea' or 'lake') and nothing but
            {sea, lake, water, island} + natural-land tags;  land = only natural-land tags (NAT), no water.
            No ships/planes/airport/harbour/buildings/roads in any background. Each pool is split once
            (BG_FRAC) and every composite takes a different background image while the split's pool lasts.
  composite object rotated k*90 deg + random flip, scaled so its MASK covers `a` of the 256x256 frame,
            a ~ U(area*(1-spread), area*(1+spread)) (default 0.13 +- 30%, Waterbirds' bird covers 0.133),
            capped so the bbox fits in the 224 centre crop the network sees and the upscale is <= MAX_UP;
            pasted at a uniform position inside that crop.  out = m*obj + (1-m)*bg, JPEG q90 at 256x256.
  train     place = label with probability RHO = 0.95 (Waterbirds), val: place ~ Bernoulli(0.5) independent
            of label, test: every test object on one land AND one water background (4 cells, equal n).
Columns: image{bytes,path}, label, place, obj_id, bg_id, area (realised mask fraction of the 256 frame).
--save-masks writes <out>/test_masks.npz ('mask' (N_test,224,224) float16 = the pasted mask resized 256->224,
row order = test parquet) for analyze_bgshare.py.

Usage:
  python prep_aerialwb.py --isaid-train isaid_annotations_train.tar.gz --isaid-val isaid_annotations_val.tar.gz \
      --dota dotav1.0_images_train.tar.gz dotav1.0_images_val.tar.gz --mlrsnet 'mlrsnet/*.parquet' \
      --out data/aerialwb --save-masks [--area 0.13] [--limit N]
  --limit N  smoke test: at most N objects per class per split (a truncated DOTA archive is then tolerated).
"""
import argparse, glob, io, json, tarfile
from pathlib import Path
import numpy as np

RHO = 0.95
N = {'train': 1200, 'validation': 200, 'test': 500}     # objects per class per split
MIN_SIDE, CAP, MAX_UP = 64, 30, 4.0
VAL_IMG_FRAC = 0.15                                      # share of iSAID-train source images that go to our val
BG_FRAC = {'train': 0.5, 'validation': 0.1, 'test': 0.4}
FRAME, CROP = 256, 224                                   # stored size / centre crop seen by the network
SEED = 0
CLS = {'plane': 0, 'ship': 1}
NAT = {'bare soil', 'chaparral', 'desert', 'sand', 'field', 'forest', 'trees', 'grass', 'terrace', 'gully', 'trail'}
MLRS = ('airplane,airport,bare soil,baseball diamond,basketball court,beach,bridge,buildings,cars,chaparral,cloud,'
        'containers,crosswalk,dense residential area,desert,dock,factory,field,football field,forest,freeway,golf course,'
        'grass,greenhouse,gully,habor,intersection,island,lake,mobile home,mountain,overpass,park,parking lot,parkway,'
        'pavement,railway,railway station,river,road,roundabout,runway,sand,sea,ships,snow,snowberg,sparse residential '
        'area,stadium,swimming pool,tanks,tennis court,terrace,track,trail,transmission tower,trees,water,wetland,'
        'wind turbine').split(',')


def bg_place(tags):
    """1 water / 0 land / None (unused) for one MLRSNet tag set."""
    s = {MLRS[i] for i in tags}
    if 'water' in s and s & {'sea', 'lake'} and s <= NAT | {'sea', 'lake', 'water', 'island'}: return 1
    if s and s <= NAT: return 0
    return None


def load_ann(path):
    """COCO json of iSAID, from the json itself or from the HF tar.gz that contains it."""
    if path.endswith('.json'): return json.load(open(path))
    with tarfile.open(path, 'r|*') as t:
        for m in t:
            if m.name.endswith('.json'): return json.load(t.extractfile(m))
    raise FileNotFoundError(f'no json in {path}')


def candidates(d, rng):
    """{file_name: [ann, ...]} of eligible plane/ship instances (size + per-image cap; border/colour checked later)."""
    fn = {i['id']: i['file_name'] for i in d['images']}
    per = {}
    for a in d['annotations']:                       # category ids differ between the train and val json: use names
        if a['category_name'] in CLS and not a['iscrowd'] and max(a['bbox'][2:]) >= MIN_SIDE:
            per.setdefault((fn[a['image_id']], a['category_name']), []).append(a)
    out = {}
    for (f, c), v in sorted(per.items()):
        v = sorted(v, key=lambda a: a['id'])
        out.setdefault(f, []).extend(v[i] for i in sorted(rng.permutation(len(v))[:CAP]))
    return out


def cut_objects(want, archives, tolerate_trunc):
    """Stream the DOTA archives once; return {ann id: (rgb crop uint8, mask crop bool, class, file)}."""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    objs, grey, border, seen = {}, 0, 0, 0
    for arc in archives:
        try:
            with tarfile.open(arc, 'r|*') as t:
                for m in t:
                    f = Path(m.name).name
                    if not m.isfile() or f not in want: continue
                    seen += 1
                    im = Image.open(io.BytesIO(t.extractfile(m).read()))
                    a = np.asarray(im.convert('RGB'))
                    sm = a[::8, ::8].astype(np.int16)
                    if im.mode in ('L', 'I', 'I;16') or np.abs(sm[..., 0] - sm[..., 1]).mean() + np.abs(sm[..., 1] - sm[..., 2]).mean() < 2:
                        grey += len(want[f]); continue
                    H, W = a.shape[:2]
                    for ann in want[f]:
                        x, y, w, h = ann['bbox']
                        x0, y0, x1, y1 = int(np.floor(x)), int(np.floor(y)), int(np.ceil(x + w)), int(np.ceil(y + h))
                        if x0 < 2 or y0 < 2 or x1 > W - 2 or y1 > H - 2: border += 1; continue
                        mk = Image.new('L', (x1 - x0, y1 - y0), 0); dr = ImageDraw.Draw(mk)
                        for poly in ann['segmentation']:
                            dr.polygon([(px - x0, py - y0) for px, py in zip(poly[0::2], poly[1::2])], fill=255)
                        mk = np.asarray(mk) > 127
                        if mk.sum() < 64: continue
                        objs[ann['id']] = (a[y0:y1, x0:x1].copy(), mk, CLS[ann['category_name']], f)
        except (tarfile.ReadError, EOFError, OSError) as e:
            if not tolerate_trunc: raise
            print(f'[aerialwb] {arc}: archive ended early ({type(e).__name__}), smoke test continues', flush=True)
    print(f'[aerialwb] source images read {seen}/{len(want)}, objects kept {len(objs)}, '
          f'dropped grey-image {grey}, touching border {border}', flush=True)
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--isaid-train', required=True); ap.add_argument('--isaid-val', required=True)
    ap.add_argument('--dota', nargs='+', required=True, help='DOTA-v1.0 image archives (train and val tar.gz)')
    ap.add_argument('--mlrsnet', required=True, help="glob of the MLRSNet parquet shards")
    ap.add_argument('--out', default='data/aerialwb')
    ap.add_argument('--area', type=float, default=0.13, help='mean object-mask fraction of the 256x256 frame')
    ap.add_argument('--spread', type=float, default=0.3, help='area ~ U(area*(1-spread), area*(1+spread))')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: objects per class per split')
    ap.add_argument('--save-masks', action='store_true')
    o = ap.parse_args()
    from PIL import Image
    import pyarrow as pa, pyarrow.parquet as pq
    rng = np.random.default_rng(SEED)

    # ---- objects: candidates, image-level split, crops -----------------------------------------------------
    ctr, cte = candidates(load_ann(o.isaid_train), rng), candidates(load_ann(o.isaid_val), rng)
    objs = cut_objects({**ctr, **cte}, o.dota, tolerate_trunc=bool(o.limit))
    imgs = sorted({v[3] for v in objs.values()} & set(ctr))  # iSAID-train images that kept >= 1 object
    imgs = [imgs[i] for i in rng.permutation(len(imgs))]
    n_val_img = round(VAL_IMG_FRAC * len(imgs))
    src = {'validation': set(imgs[:n_val_img]), 'train': set(imgs[n_val_img:]), 'test': set(cte)}
    pool = {s: {c: sorted(k for k, v in objs.items() if v[3] in src[s] and v[2] == c) for c in (0, 1)} for s in N}
    for s in N:
        for c in (0, 1):
            k = pool[s][c]; n = o.limit or N[s]
            print(f'[aerialwb] {s} class {c}: {len(k)} eligible objects from '
                  f'{len({objs[i][3] for i in k})} source images, using {min(n, len(k))}', flush=True)
            assert o.limit or len(k) >= n, f'{s}/{c}: only {len(k)} objects for {n}'
            pool[s][c] = [k[i] for i in sorted(rng.permutation(len(k))[:n])]

    # ---- backgrounds ------------------------------------------------------------------------------------------
    bgs = {0: [], 1: []}
    for f in sorted(glob.glob(o.mlrsnet)):
        t = pq.read_table(f, columns=['label'])
        for i, tags in enumerate(t.column('label').to_pylist()):
            p = bg_place(tags)
            if p is not None: bgs[p].append((Path(f).name, i))
    bsplit = {}
    for p, v in bgs.items():
        v = [v[i] for i in rng.permutation(len(v))]; a = 0
        for s, fr in BG_FRAC.items():
            n = round(fr * len(v)); bsplit[s, p] = v[a:a + n]; a += n
        print(f'[aerialwb] backgrounds place {p}: {len(v)} ' +
              ' '.join(f'{s}={len(bsplit[s, p])}' for s in BG_FRAC), flush=True)
    need = {(s, p) for s in N for p in (0, 1)}
    bg_bytes = {}
    for f in sorted(glob.glob(o.mlrsnet)):                     # read only the images we may use
        rows = {i for (s, p) in need for (ff, i) in bsplit[s, p] if ff == Path(f).name}
        col = pq.read_table(f, columns=['image']).column('image')
        for i in rows: bg_bytes[Path(f).name, i] = col[i].as_py()['bytes']

    prng = np.random.default_rng(SEED + 1)                     # composition RNG, separate from the split RNG
    lo, hi = (FRAME - CROP) // 2, (FRAME + CROP) // 2

    def compose(oid, bkey):
        rgb, mk = objs[oid][0], objs[oid][1]
        k, fl = int(prng.integers(4)), bool(prng.integers(2))
        rgb, mk = np.rot90(rgb, k), np.rot90(mk, k)
        if fl: rgb, mk = rgb[:, ::-1], mk[:, ::-1]
        a = prng.uniform(o.area * (1 - o.spread), o.area * (1 + o.spread))
        s = min(np.sqrt(a * FRAME * FRAME / mk.sum()), CROP / max(mk.shape), MAX_UP)
        h, w = max(1, round(mk.shape[0] * s)), max(1, round(mk.shape[1] * s))
        oi = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((w, h), Image.BILINEAR), np.float32)
        om = np.asarray(Image.fromarray(np.ascontiguousarray(mk).astype(np.uint8) * 255).resize((w, h), Image.BILINEAR), np.float32)[..., None] / 255
        B = np.asarray(Image.open(io.BytesIO(bg_bytes[bkey])).convert('RGB').resize((FRAME, FRAME), Image.BILINEAR), np.float32)
        ty, tx = int(prng.integers(lo, hi - h + 1)), int(prng.integers(lo, hi - w + 1))
        M = np.zeros((FRAME, FRAME, 1), np.float32); M[ty:ty + h, tx:tx + w] = om
        F = np.zeros_like(B); F[ty:ty + h, tx:tx + w] = oi
        out = io.BytesIO(); Image.fromarray((M * F + (1 - M) * B).round().astype(np.uint8)).save(out, 'JPEG', quality=90)
        return out.getvalue(), M[..., 0]

    out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    for s in N:
        rows = []                                              # (obj id, label, place)
        for c in (0, 1):
            for oid in pool[s][c]:
                if s == 'train': ps = [c if rng.random() < RHO else 1 - c]
                elif s == 'validation': ps = [int(rng.integers(2))]
                else: ps = [0, 1]
                rows += [(oid, c, p) for p in ps]
        nxt = {0: 0, 1: 0}; recs, masks = [], []
        for oid, c, p in rows:
            b = bsplit[s, p][nxt[p] % len(bsplit[s, p])]; nxt[p] += 1   # distinct backgrounds while the pool lasts
            jpg, M = compose(oid, b)
            recs.append(dict(image={'bytes': jpg, 'path': f'{oid}__{b[0][:14]}_{b[1]}.jpg'}, label=c, place=p,
                             obj_id=int(oid), bg_id=f'{b[0]}:{b[1]}', area=float(M.mean())))
            if s == 'test' and o.save_masks:
                masks.append(np.asarray(Image.fromarray(M).resize((CROP, CROP), Image.BILINEAR), np.float16))
        pq.write_table(pa.Table.from_pylist(recs), out / f'{s}-00000-of-00001.parquet')
        if masks: np.savez_compressed(out / 'test_masks.npz', mask=np.stack(masks))
        lab, pl, ar = (np.array([r[k] for r in recs]) for k in ('label', 'place', 'area'))
        cells = {(y, q): int(((lab == y) & (pl == q)).sum()) for y in (0, 1) for q in (0, 1)}
        print(f'[aerialwb] {s}: {len(recs)} images, cells (label,place) {cells}, P(place = label) = '
              f'{(lab == pl).mean():.3f}, background reuse {max(0, max(nxt[q] - len(bsplit[s, q]) for q in (0, 1)))}, '
              f'area mean {ar.mean():.3f} [plane {ar[lab == 0].mean():.3f}, ship {ar[lab == 1].mean():.3f}] '
              f'p5/p50/p95 {np.percentile(ar, 5):.3f}/{np.median(ar):.3f}/{np.percentile(ar, 95):.3f}', flush=True)


if __name__ == '__main__':
    main()
