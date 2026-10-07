"""Build data/oi_waterbirds/{train,validation,test}-00000-of-00001.parquet (Waterbirds format) + test_masks.npz:
a REAL-PHOTO Waterbirds from Open Images V7 (see OI_WATERBIRDS.md, pre-registration draft).

Source: Open Images V7 (Kuznetsova et al. 2020), train+validation+test combined. Annotations CC BY 4.0 (Google),
images CC BY 2.0 (Flickr; OriginalURL/Author kept per row). All selection uses human annotations only:
  label = 1 waterbird if every bird box in the image is of a WATERBIRD class, 0 landbird if every one is of a
          LANDBIRD class (Waterbirds convention: 1 = waterbird, 1 = water). Several classes of the same side are
          allowed: OI annotators often box one duck as Duck + Goose (1,719 train images), so they are the same bird.
          Dropped: any box of an OTHER_BIRD class (generic 'Bird', raptors, poultry) or of the other side, any bird
          box with IsGroupOf / IsDepiction / IsInside, Rotation != 0 (boxes and pixels would disagree).
  area  = union of all bird boxes / image (normalised coords, 200x200 raster); keep AREA_LO..AREA_HI
  place = 1 water if >= 1 human-verified POSITIVE (Confidence 1) label in WATER_PLACE and none in LAND_PLACE,
          0 land  if >= 1 in LAND_PLACE and none in WATER_PLACE, else dropped
Label == place is the majority group (landbird/land g00, waterbird/water g11), as Waterbirds, so
evaluate_groups.py / resample_groups / --train_corr work unchanged.
Splits: same rule as prep_coco_smallobj.split (imported): per (label, place) cell a seeded permutation of the
  sorted ImageIDs -> first TEST_PER_CELL test, next VAL_PER_CELL validation, rest train subsampled per class to
  P(place = label | class) = RHO.  OI's own train/val/test split is ignored (ids are disjoint, so no overlap).
Images squashed to 256x256 (bilinear), JPEG q=90, so waterbirds_transforms applies unchanged.
--save-masks: test_masks.npz ('mask' N x 224 x 224 float16, 'from_seg' N bool) = union of the target class's
  bird instance masks (OI V5 segmentation, pulled from the per-prefix zips by HTTP range) where they exist,
  else the union of its boxes; in the squashed full-image frame analyze_bgshare.py expects.

Usage:
  python prep_oi_waterbirds.py --fetch --ann data/oi_ann          # streams ~6.8 GB of CSVs (~5 min at 25 MB/s), keeps ~68 MB
  python prep_oi_waterbirds.py --ann data/oi_ann --counts-only    # exact cell / split counts, no images
  python prep_oi_waterbirds.py --ann data/oi_ann --out data/oi_waterbirds --save-masks
  (--limit 12: smoke test, <= 12 images per cell, test 3 / val 1 per cell; landbird/water has only 6)
"""
import argparse, csv, io, sys, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import prep_coco_smallobj as coco
from prep_coco_smallobj import _get, _HTTPRange

AREA_LO, AREA_HI = 0.01, 0.15
coco.RHO, coco.TEST_PER_CELL, coco.VAL_PER_CELL, coco.SEED = 0.95, 100, 25, 0   # same as COCO / Waterbirds
# bird classes = the Bird subtree of the 600 boxable classes, split by natural habitat
WATERBIRD = {'/m/09ddx': 'Duck', '/m/0dbvp': 'Goose', '/m/0dftk': 'Swan', '/m/05z6w': 'Penguin'}
LANDBIRD = {'/m/012074': 'Magpie', '/m/01dy8n': 'Woodpecker', '/m/01f8m5': 'Blue jay', '/m/06j2d': 'Raven',
            '/m/09d5_': 'Owl', '/m/0ccs93': 'Canary', '/m/0gv1x': 'Parrot', '/m/0h23m': 'Sparrow',
            '/m/05n4y': 'Ostrich'}
# excluding: generic Bird (species unknown); raptors (Eagle includes sea/fish eagles, and Eagle + Falcon are boxed on
# the same bird in 733 train images); poultry (farm / food scenes, not a natural habitat)
OTHER_BIRD = {'/m/015p6': 'Bird', '/m/09csl': 'Eagle', '/m/0f6wt': 'Falcon', '/m/09b5t': 'Chicken', '/m/0jly1': 'Turkey'}
BIRDS = {**WATERBIRD, **LANDBIRD, **OTHER_BIRD}
# image-level place labels (human-verified positives only)
WATER_PLACE = {'/m/0838f': 'Water', '/m/04h4w': 'Lake', '/m/06npx': 'Sea', '/m/05kq4': 'Ocean', '/m/06cnp': 'River',
               '/m/0184rb': 'Pond', '/m/017m8l': 'Waterway', '/m/0pvlz': 'Wetland', '/m/016nsr': 'Marsh',
               '/m/01lxd': 'Coast', '/m/02fm9k': 'Shore', '/m/0b3yr': 'Beach', '/m/0wf0kbf': 'Wave'}
LAND_PLACE = {'/m/07j7r': 'Tree', '/m/02zr8': 'Forest', '/m/02hnx9': 'Woody plant', '/m/0b5gs': 'Branch',
              '/m/016nqt': 'Twig', '/m/09t49': 'Leaf', '/m/08t9c_': 'Grass', '/m/01v327': 'Lawn',
              '/m/025st_8': 'Meadow', '/m/01c7cq': 'Grassland'}
GS = 'https://storage.googleapis.com/openimages/'
SPLITS = {'train': dict(bbox='v6/oidv6-train-annotations-bbox.csv', labels='v7/oidv7-train-annotations-human-imagelabels.csv',
                        seg='v5/train-annotations-object-segmentation.csv', images='2018_04/train/train-images-boxable-with-rotation.csv'),
          'validation': dict(bbox='v5/validation-annotations-bbox.csv', labels='v7/oidv7-val-annotations-human-imagelabels.csv',
                             seg='v5/validation-annotations-object-segmentation.csv', images='2018_04/validation/validation-images-with-rotation.csv'),
          'test': dict(bbox='v5/test-annotations-bbox.csv', labels='v7/oidv7-test-annotations-human-imagelabels.csv',
                       seg='v5/test-annotations-object-segmentation.csv', images='2018_04/test/test-images-with-rotation.csv')}


def _lines(url, tries=20):
    """stream a remote text file line by line; on a dropped connection resume with an HTTP Range from the last byte."""
    pos, buf, t = 0, b'', 0
    while True:
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={pos}-'} if pos else {}), timeout=60)
            while (c := r.read(8 << 20)):
                pos += len(c); *ls, buf = (buf + c).split(b'\n')
                yield from (l.decode() for l in ls)
            if buf: yield buf.decode()
            return
        except Exception as e:
            t += 1
            if t >= tries: raise
            print(f'  resume {url} at {pos}: {e}', file=sys.stderr); time.sleep(3 * t)


def _filter(url, dst, keep):
    """write header + the lines for which keep(line) is true; skip if dst exists."""
    if dst.exists(): return
    t0, n, k = time.time(), 0, 0
    with open(str(dst) + '.part', 'w') as g:
        it = _lines(url); g.write(next(it) + '\n')
        for l in it:
            n += 1
            if keep(l): g.write(l + '\n'); k += 1
    Path(str(dst) + '.part').rename(dst)
    print(f'[fetch] {dst.name}: kept {k} of {n} lines ({time.time() - t0:.0f}s)', flush=True)


def fetch(ann):
    ann.mkdir(parents=True, exist_ok=True)
    for s, f in SPLITS.items():               # bbox first: it defines the bird image set the other files are cut to
        _filter(GS + f['bbox'], ann / f'bbox_{s}.csv', lambda l: l.split(',', 3)[2] in BIRDS)
        ids = {r['ImageID'] for r in csv.DictReader(open(ann / f'bbox_{s}.csv'))}
        _filter(GS + f['labels'], ann / f'labels_{s}.csv', lambda l: l.split(',', 1)[0] in ids)
        _filter(GS + f['images'], ann / f'images_{s}.csv', lambda l: l.split(',', 1)[0] in ids)
        _filter(GS + f['seg'], ann / f'seg_{s}.csv', lambda l: l.split(',', 3)[2] in BIRDS)


def _union(boxes, n=200):
    g = np.zeros((n, n), bool)
    for x0, x1, y0, y1 in boxes: g[int(y0 * n):int(np.ceil(y1 * n)), int(x0 * n):int(np.ceil(x1 * n))] = True
    return g


def candidates(ann, s, stats):
    box = {}
    for r in csv.DictReader(open(ann / f'bbox_{s}.csv')): box.setdefault(r['ImageID'], []).append(r)
    pos = {}
    for r in csv.DictReader(open(ann / f'labels_{s}.csv')):
        if r['Confidence'] == '1': pos.setdefault(r['ImageID'], set()).add(r['LabelName'])
    meta = {r['ImageID']: r for r in csv.DictReader(open(ann / f'images_{s}.csv'))}
    rows = []
    for i, bs in sorted(box.items()):
        cls = {b['LabelName'] for b in bs}
        def drop(why): stats[why] = stats.get(why, 0) + 1
        if cls & OTHER_BIRD.keys(): drop('other bird class (Bird/raptor/poultry)'); continue
        if not (cls <= WATERBIRD.keys() or cls <= LANDBIRD.keys()): drop('water- and landbird classes both'); continue
        if any(b[f] == '1' for b in bs for f in ('IsGroupOf', 'IsDepiction', 'IsInside')): drop('group/depiction/inside box'); continue
        m = meta.get(i, {})
        if m.get('Rotation', '0') not in ('', '0', '0.0'): drop('rotation != 0'); continue
        area = _union([tuple(float(b[k]) for k in ('XMin', 'XMax', 'YMin', 'YMax')) for b in bs]).mean()
        if not AREA_LO <= area <= AREA_HI: drop(f'area outside [{AREA_LO}, {AREA_HI}]'); continue
        p = pos.get(i, set()); w, l = bool(p & WATER_PLACE.keys()), bool(p & LAND_PLACE.keys())
        if w == l: drop('place both' if w else 'place neither'); continue
        rows.append(dict(image_id=i, src=s, label=int(cls <= WATERBIRD.keys()), place=int(w), bird='+'.join(sorted(BIRDS[c] for c in cls)), area=float(area),
                         n_boxes=len(bs), url=m.get('OriginalURL', ''), author=m.get('Author', ''),
                         license=m.get('License', ''), boxes=[tuple(float(b[k]) for k in ('XMin', 'XMax', 'YMin', 'YMax')) for b in bs]))
    return rows


def masks(ann, rows):
    """union of the target class's OI instance masks (else its boxes), squashed to 224x224, in row order."""
    from PIL import Image
    seg = {}
    for s in {r['src'] for r in rows}:
        for r in csv.DictReader(open(ann / f'seg_{s}.csv')): seg.setdefault((s, r['ImageID']), []).append(r['MaskPath'])
    zips = {}

    def one(r):
        paths = seg.get((r['src'], r['image_id']), [])   # seg_*.csv holds bird classes only; every bird here is the target
        if not paths:
            return np.asarray(Image.fromarray(_union(r['boxes'], 224).astype(np.uint8) * 255), np.float32) / 255, False
        m = np.zeros((224, 224), np.float32)
        for p in paths:
            key = (r['src'], p[0])
            if key not in zips:
                zips[key] = zipfile.ZipFile(io.BufferedReader(_HTTPRange(f"{GS}v5/{r['src']}-masks/{r['src']}-masks-{p[0]}.zip"), 1 << 16))
            m = np.maximum(m, np.asarray(Image.open(io.BytesIO(zips[key].read(p))).convert('L').resize((224, 224), Image.BILINEAR), np.float32) / 255)
        return m, True
    out = [one(r) for r in rows]      # ponytail: sequential range reads (~1 s/img); pool per zip if it gets slow
    return np.stack([m for m, _ in out]).astype(np.float16), np.array([f for _, f in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='dir with the filtered CSVs written by --fetch')
    ap.add_argument('--out', default='data/oi_waterbirds')
    ap.add_argument('--fetch', action='store_true', help='stream + filter the annotation CSVs into --ann and exit')
    ap.add_argument('--img-cache', default=None, help='keep downloaded originals here (default: <out>/_orig)')
    ap.add_argument('--save-masks', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: cap each cell at N, test N//4, val N//8 per cell')
    ap.add_argument('--counts-only', action='store_true', help='print drop reasons, cell counts and split sizes; fetch nothing')
    o = ap.parse_args()
    ann = Path(o.ann)
    if o.fetch: fetch(ann); return
    stats = {}
    rows = [r for s in SPLITS for r in candidates(ann, s, stats)]
    for r in rows: r['coco_id'] = r['image_id']          # coco.split sorts / permutes on this key
    print('[oi] dropped:', dict(sorted(stats.items(), key=lambda kv: -kv[1])))
    test_n, val_n = coco.TEST_PER_CELL, coco.VAL_PER_CELL
    by = {}
    for r in sorted(rows, key=lambda r: r['image_id']): by.setdefault((r['label'], r['place']), []).append(r)
    for k, v in sorted(by.items()):      # printed before the split so an infeasible cell still shows its count
        birds = {}
        for r in v: birds[r['bird']] = birds.get(r['bird'], 0) + 1
        print(f'[oi] cell {("landbird", "waterbird")[k[0]]}/{("land", "water")[k[1]]}: {len(v)} images, '
              f'median area {np.median([r["area"] for r in v]):.3f}, OI splits {sorted({r["src"] for r in v})}, '
              f'birds {dict(sorted(birds.items(), key=lambda kv: -kv[1]))}', flush=True)
    if o.limit:
        rows = [r for v in by.values() for r in v[:o.limit]]; test_n, val_n = max(1, o.limit // 4), max(1, o.limit // 8)
    cells, sp = coco.split(rows, test_n, val_n)
    for s, v in sp.items():
        g = {}
        for r in v: g[(r['label'], r['place'])] = g.get((r['label'], r['place']), 0) + 1
        print(f'[oi] {s}: {len(v)} images, groups {dict(sorted(g.items()))}', flush=True)
    if o.counts_only: return
    from PIL import Image
    import pyarrow as pa, pyarrow.parquet as pq
    out = Path(o.out); cache = Path(o.img_cache or out / '_orig'); cache.mkdir(parents=True, exist_ok=True)

    def load(r):
        p = cache / f"{r['image_id']}.jpg"
        if not p.exists():
            p.write_bytes(_get(f"https://s3.amazonaws.com/open-images-dataset/{r['src']}/{r['image_id']}.jpg"))
        b = io.BytesIO(); Image.open(p).convert('RGB').resize((256, 256), Image.BILINEAR).save(b, 'JPEG', quality=90)
        return b.getvalue()

    for s, v in sp.items():
        with ThreadPoolExecutor(16) as ex: imgs = list(ex.map(load, v))
        tab = pa.table({'image': [{'bytes': b, 'path': r['image_id'] + '.jpg'} for b, r in zip(imgs, v)],
                        **{k: [r[k] for r in v] for k in ('label', 'place', 'image_id', 'src', 'bird', 'area', 'n_boxes',
                                                           'url', 'author', 'license')}})
        pq.write_table(tab, out / f'{s}-00000-of-00001.parquet')
        lab, pl = np.array(tab.column('label')), np.array(tab.column('place'))
        print(f'[oi] wrote {s}: {len(v)} images, P(place = label) = {(lab == pl).mean():.3f}, '
              f'mean area {np.mean([r["area"] for r in v]):.3f}', flush=True)
        if s == 'test' and o.save_masks:
            M, F = masks(ann, v)
            np.savez_compressed(out / 'test_masks.npz', mask=M, from_seg=F)
            print(f'[oi] test_masks.npz {M.shape}, {F.sum()} from OI segmentation, {(~F).sum()} from boxes, '
                  f'mean mask area {M.astype(np.float32).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
