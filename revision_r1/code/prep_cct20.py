"""Build data/cct20/{train,validation,test}-00000-of-00001.parquet (Waterbirds format) from LILA CCT20.

CCT20 = the "Recognition in Terra Incognita" benchmark (Beery, Van Horn & Perona, ECCV 2018), a subset of
Caltech Camera Traps. Pre-registered construction (PREREG_DATASETS.md, 2026-09-29):
  classes  : the 12 categories with >= 100 single-label train images (CLASSES below; empty/badger/fox/deer out)
  images   : only images whose annotations carry exactly one category
  label    : index into CLASSES;  place = camera location id (int, the confound)
  train    : official 'train' (10 locations);  validation = official 'cis_val' (same 10 locations)
  test     : official 'cis_test' (seen locations) + 'trans_test' (9 unseen locations), keeping only
             (class, location) cells with >= MIN_CELL images, counted on the annotations
             (so the cell set does not depend on the download or on any model)
  column 'split' = 'cis' / 'trans' marks seen vs unseen locations; 'trans_val' (1 location) is unused.
Images: 1024-px 'sm' release, resized to 256x256 (what waterbirds_transforms does first), JPEG q=90.

Usage: python prep_cct20.py --tar eccv_18_all_images_sm.tar.gz --ann eccv_18_annotations.tar.gz --out data/cct20
       (--allow_missing only for a partial tarball in a local smoke test)
"""
import argparse, collections, io, json, tarfile
from pathlib import Path

CLASSES = ['bird', 'bobcat', 'car', 'cat', 'coyote', 'dog', 'opossum', 'rabbit', 'raccoon', 'rodent', 'skunk', 'squirrel']
MIN_CELL = 20
SPLITS = {'train': ['train'], 'validation': ['cis_val'], 'test': ['cis_test', 'trans_test']}


def to_jpeg256(b):
    from PIL import Image
    im = Image.open(io.BytesIO(b)); im.draft('RGB', (256, 256))           # fast DCT downscale
    out = io.BytesIO(); im.convert('RGB').resize((256, 256), Image.BILINEAR).save(out, 'JPEG', quality=90)
    return out.getvalue()


def load_annotations(ann_tgz):
    """-> {image_id: (parquet_split, source_split, class_idx, location)} after the class/cell rules."""
    rows = {}
    with tarfile.open(ann_tgz) as t:
        js = {Path(m.name).name.replace('_annotations.json', ''): json.load(t.extractfile(m))
              for m in t if m.name.endswith('_annotations.json')}
    for split, srcs in SPLITS.items():
        for src in srcs:
            d = js[src]; cats = {c['id']: c['name'] for c in d['categories']}
            lab = collections.defaultdict(set)
            for a in d['annotations']:
                lab[a['image_id']].add(cats[a['category_id']])
            for im in d['images']:
                c = lab[im['id']]
                if len(c) == 1 and next(iter(c)) in CLASSES:
                    rows[im['id']] = (split, src, CLASSES.index(next(iter(c))), int(im['location']))
    cell = collections.Counter((r[2], r[3]) for r in rows.values() if r[0] == 'test')
    rows = {k: r for k, r in rows.items() if r[0] != 'test' or cell[(r[2], r[3])] >= MIN_CELL}
    n = collections.Counter(r[0] for r in rows.values())
    print(f"[cct20] annotations: {dict(n)}; test cells kept {sum(v >= MIN_CELL for v in cell.values())}/{len(cell)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', required=True); ap.add_argument('--ann', required=True)
    ap.add_argument('--out', default='data/cct20'); ap.add_argument('--allow_missing', action='store_true')
    o = ap.parse_args()
    rows = load_annotations(o.ann)
    img = {}
    with tarfile.open(o.tar, 'r|*') as t:                                   # streaming: no extraction to disk
        try:
            for m in t:
                k = Path(m.name).stem
                if m.isfile() and k in rows:
                    img[k] = to_jpeg256(t.extractfile(m).read())
                    if len(img) % 5000 == 0: print(f"[cct20] {len(img)}/{len(rows)}", flush=True)
        except (tarfile.ReadError, EOFError) as e:                          # truncated archive
            if not o.allow_missing: raise
            print(f"[cct20] archive ended early ({e}) -- allowed by --allow_missing")
    miss = len(rows) - len(img)
    if miss and not o.allow_missing:
        raise SystemExit(f"[cct20] {miss} annotated images missing from {o.tar}")
    import pyarrow as pa, pyarrow.parquet as pq
    out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        ks = sorted(k for k, r in rows.items() if r[0] == split and k in img)
        tab = pa.table({'image': [{'bytes': img[k], 'path': f'{k}.jpg'} for k in ks],
                        'label': [rows[k][2] for k in ks], 'place': [rows[k][3] for k in ks],
                        'split': ['trans' if rows[k][1] == 'trans_test' else 'cis' for k in ks],
                        'image_id': ks})
        pq.write_table(tab, out / f'{split}-00000-of-00001.parquet')
        print(f"[cct20] {split}: {len(ks)} images, {len(set(tab.column('place').to_pylist()))} locations")
    print(f"[cct20] done, missing {miss}")


if __name__ == '__main__':
    main()
