"""Build data/<name>/{train,validation,test}-00000-of-00001.parquet (Waterbirds format) + test_masks.npz
from NATURAL COCO photos in which the labelled object is SMALL (see NATURAL_SMALLOBJ.md, pre-registration).

Source: COCO 2017 train+val (Lin et al. 2014; annotations CC BY 4.0, images Flickr under their own CC licences,
licence id kept per row). Selection uses the panoptic annotations (things + stuff, non-overlapping segments):
  label  = class index in --classes (default cat=0, dog=1); the image must contain that class and NOT the other
  area   = (sum of the target class's panoptic segment areas) / (H*W); keep AREA_LO <= area <= AREA_HI;
           images with a crowd segment of the target class are dropped
  place  = 0 indoor / 1 outdoor from stuff shares: indoor if s_in >= TAU and s_out < EPS, outdoor if
           s_out >= TAU and s_in < EPS, else dropped (INDOOR / OUTDOOR lists below)
Label == place is the majority group (class 0 <-> indoor, class 1 <-> outdoor), as Waterbirds, so
resample_groups / --train_corr work unchanged.
Splits, per (label, place) cell, seeded permutation of the sorted COCO ids:
  test = first TEST_PER_CELL (group balanced), validation = next VAL_PER_CELL (group balanced),
  train = the rest, subsampled per class to P(place = label | class) = RHO:
          n_maj = min(avail_maj, floor(avail_min * RHO / (1 - RHO))), n_min = round(n_maj * (1 - RHO) / RHO)
  images left over after that are not used.
Images squashed to 256x256 (bilinear), JPEG q=90, so waterbirds_transforms applies unchanged.
--save-masks: test_masks.npz ('mask', N x 224 x 224 float16) = union of the target class's instance polygons
  (instances_*.json) in the squashed full-image frame, the frame analyze_bgshare.py expects.

Usage:
  python prep_coco_smallobj.py --fetch --ann data/coco_ann                        # annotations only (~200 MB via HTTP range)
  python prep_coco_smallobj.py --ann data/coco_ann --out data/coco_catdog --save-masks
  python prep_coco_smallobj.py --ann data/coco_ann --out data/coco_birddog --classes dog,bird --save-masks
  (--src val2017 --limit 40: smoke test on val2017 only)
"""
import argparse, io, json, sys, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

AREA_LO, AREA_HI = 0.01, 0.15      # object share of the image; Waterbirds' bird is ~0.13
TAU, EPS = 0.20, 0.05              # place: own stuff share >= TAU, the other side's < EPS
RHO = 0.95                         # Waterbirds' train correlation
TEST_PER_CELL, VAL_PER_CELL = 100, 25
SEED = 0
INDOOR = ['wall-other-merged', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'floor-other-merged', 'floor-wood',
          'ceiling-merged', 'cabinet-merged', 'rug-merged', 'table-merged', 'shelf', 'counter', 'window-blind', 'curtain']
OUTDOOR = ['sea', 'river', 'water-other', 'grass-merged', 'dirt-merged', 'tree-merged', 'mountain-merged', 'playingfield',
           'gravel', 'rock-merged', 'sand', 'snow', 'sky-other-merged', 'road', 'pavement-merged', 'building-other-merged',
           'house', 'railroad', 'roof', 'bridge']
ZIPS = {'panoptic': 'http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip',
        'instances': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'}


class _HTTPRange(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range, so zipfile pulls single members
    (the panoptic zip is 860 MB but its two json members are 32 MB compressed)."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60).headers['Content-Length'])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else self.pos + o if w == 1 else self.size + o; return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        d = _get(self.url, {'Range': f'bytes={self.pos}-{self.pos + n - 1}'})
        b[:len(d)] = d; self.pos += len(d); return len(d)


def _get(url, headers=None, tries=6):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60).read()
        except Exception as e:
            if t == tries - 1: raise
            print(f'  retry {t + 1} {url}: {e}', file=sys.stderr); time.sleep(2 * (t + 1))


def fetch(ann, srcs):
    ann.mkdir(parents=True, exist_ok=True)
    for kind, url in ZIPS.items():
        z = zipfile.ZipFile(io.BufferedReader(_HTTPRange(url), buffer_size=8 << 20))
        for s in srcs:
            dst = ann / f'{kind}_{s}.json'
            if dst.exists(): continue
            with z.open(f'annotations/{kind}_{s}.json') as f, open(str(dst) + '.part', 'wb') as g:
                while (c := f.read(8 << 20)): g.write(c)
            Path(str(dst) + '.part').rename(dst); print(f'[fetch] {dst}', flush=True)


def candidates(ann, src, classes):
    d = json.load(open(ann / f'panoptic_{src}.json'))
    cat = {c['id']: c['name'] for c in d['categories']}
    ims = {i['id']: i for i in d['images']}
    rows = []
    for a in d['annotations']:
        I = ims[a['image_id']]; A = I['height'] * I['width']
        area, crowd, s_in, s_out = {c: 0.0 for c in classes}, set(), 0.0, 0.0
        for s in a['segments_info']:
            n = cat[s['category_id']]
            if n in area:
                area[n] += s['area'] / A
                if s['iscrowd']: crowd.add(n)
            s_in += s['area'] / A if n in INDOOR else 0.0
            s_out += s['area'] / A if n in OUTDOOR else 0.0
        present = [c for c in classes if area[c] > 0]
        if len(present) != 1 or present[0] in crowd or not AREA_LO <= area[present[0]] <= AREA_HI: continue
        place = 0 if (s_in >= TAU and s_out < EPS) else 1 if (s_out >= TAU and s_in < EPS) else None
        if place is None: continue
        rows.append(dict(coco_id=I['id'], src=src, file_name=I['file_name'], license=I.get('license', -1),
                         label=classes.index(present[0]), place=place, area=area[present[0]]))
    return rows


def split(rows, test_n, val_n):
    rng = np.random.default_rng(SEED)
    cells = {}
    for r in sorted(rows, key=lambda r: r['coco_id']): cells.setdefault((r['label'], r['place']), []).append(r)
    cells = {k: [v[i] for i in rng.permutation(len(v))] for k, v in sorted(cells.items())}
    out = {'test': [], 'validation': [], 'train': []}
    pool = {}
    for k, v in cells.items():
        assert len(v) > test_n + val_n, f'cell {k} has {len(v)} images, needs > {test_n + val_n}'
        out['test'] += v[:test_n]; out['validation'] += v[test_n:test_n + val_n]; pool[k] = v[test_n + val_n:]
    for y in (0, 1):
        maj, mino = pool[(y, y)], pool[(y, 1 - y)]
        n_maj = min(len(maj), int(len(mino) * RHO / (1 - RHO)))
        n_min = min(len(mino), round(n_maj * (1 - RHO) / RHO))
        out['train'] += maj[:n_maj] + mino[:n_min]
    return cells, out


def masks(ann, rows, classes):
    """union of the target class's instance polygons, squashed to 224x224 (float16), in row order."""
    from PIL import Image, ImageDraw
    M = []
    for src in sorted({r['src'] for r in rows}):
        d = json.load(open(ann / f'instances_{src}.json'))
        cid = {c['name']: c['id'] for c in d['categories']}
        by = {}
        for a in d['annotations']: by.setdefault(a['image_id'], []).append(a)
        dims = {i['id']: (i['width'], i['height']) for i in d['images']}
        for r in rows:
            if r['src'] != src: continue
            W, H = dims[r['coco_id']]; im = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(im)
            for a in by.get(r['coco_id'], []):
                if a['category_id'] == cid[classes[r['label']]] and isinstance(a['segmentation'], list):
                    for poly in a['segmentation']:
                        if len(poly) >= 6: dr.polygon(poly, fill=255)
            M.append((r['coco_id'], np.asarray(im.resize((224, 224), Image.BILINEAR), np.float32) / 255))
    order = {cid_: m for cid_, m in M}
    return np.stack([order[r['coco_id']] for r in rows]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='dir with panoptic_<src>.json and instances_<src>.json')
    ap.add_argument('--out', default='data/coco_catdog')
    ap.add_argument('--classes', default='cat,dog', help='class0,class1 (COCO names); class0 <-> indoor, class1 <-> outdoor')
    ap.add_argument('--src', default='train2017,val2017')
    ap.add_argument('--fetch', action='store_true', help='download the annotation jsons into --ann and exit')
    ap.add_argument('--img-cache', default=None, help='keep downloaded originals here (default: <out>/_orig)')
    ap.add_argument('--save-masks', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: cap each cell at N, test N//4, val N//8 per cell')
    ap.add_argument('--counts-only', action='store_true', help='print cell counts and split sizes, fetch nothing')
    o = ap.parse_args()
    ann, srcs, classes = Path(o.ann), o.src.split(','), o.classes.split(',')
    if o.fetch: fetch(ann, srcs); return
    rows = [r for s in srcs for r in candidates(ann, s, classes)]
    test_n, val_n = TEST_PER_CELL, VAL_PER_CELL
    if o.limit:
        by = {}
        for r in sorted(rows, key=lambda r: r['coco_id']): by.setdefault((r['label'], r['place']), []).append(r)
        rows = [r for v in by.values() for r in v[:o.limit]]; test_n, val_n = max(1, o.limit // 4), max(1, o.limit // 8)
    cells, sp = split(rows, test_n, val_n)
    for k, v in cells.items():
        print(f'[coco] cell {classes[k[0]]}/{("indoor", "outdoor")[k[1]]}: {len(v)} images, '
              f'median area {np.median([r["area"] for r in v]):.3f}')
    for s, v in sp.items():
        g = {}
        for r in v: g[(r['label'], r['place'])] = g.get((r['label'], r['place']), 0) + 1
        print(f'[coco] {s}: {len(v)} images, groups {dict(sorted(g.items()))}', flush=True)
    if o.counts_only: return
    from PIL import Image
    import pyarrow as pa, pyarrow.parquet as pq
    out = Path(o.out); cache = Path(o.img_cache or out / '_orig'); cache.mkdir(parents=True, exist_ok=True)

    def load(r):
        p = cache / r['file_name']
        if not p.exists():
            p.write_bytes(_get(f"http://images.cocodataset.org/{r['src']}/{r['file_name']}"))
        b = io.BytesIO(); Image.open(p).convert('RGB').resize((256, 256), Image.BILINEAR).save(b, 'JPEG', quality=90)
        return b.getvalue()

    for s, v in sp.items():
        with ThreadPoolExecutor(16) as ex: imgs = list(ex.map(load, v))
        tab = pa.table({'image': [{'bytes': b, 'path': r['file_name']} for b, r in zip(imgs, v)],
                        **{k: [r[k] for r in v] for k in ('label', 'place', 'coco_id', 'src', 'area', 'license')}})
        pq.write_table(tab, out / f'{s}-00000-of-00001.parquet')
        lab, pl = np.array(tab.column('label')), np.array(tab.column('place'))
        print(f'[coco] wrote {s}: {len(v)} images, P(place = label) = {(lab == pl).mean():.3f}, '
              f'mean area {np.mean([r["area"] for r in v]):.3f}', flush=True)
        if s == 'test' and o.save_masks:
            M = masks(ann, v, classes)
            np.savez_compressed(out / 'test_masks.npz', mask=M)
            print(f'[coco] test_masks.npz {M.shape}, mean mask area {M.astype(np.float32).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
