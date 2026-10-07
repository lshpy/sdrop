"""Build data/isic_artifact/{train,validation,test}-00000-of-00001.parquet (Waterbirds format).

Source: Bissoto, Valle & Avila, "Debiasing Skin Lesion Datasets and Models? Not So Fast" (ISIC Skin Image
Analysis Workshop @ CVPR 2020): 7 artifact labels (dark corner, hair, gel border, gel bubble, ruler, ink,
patches) for the 2,594 ISIC 2018 Task 1/2 images, and "trap" splits in which the artifacts correlate with
the label in training and anti-correlate in test. Pre-registered construction (PREREG_DATASETS.md):
  label = 1 malignant (melanoma) / 0 benign;  place = ruler present (the artifact most correlated with
          the label: phi +0.41 in trap-train, -0.67 in trap-test)
  split = trap split 1:  train = train1 minus val1 (val1 is a subset of train1),  validation = val1,
          test = test1 (identical in all five trap splits)
  all 8 artifact columns are kept for later analysis.
Images: full-resolution JPEGs from the public ISIC archive bucket (no login), resized to 256x256, JPEG q=90.
CSV files are pinned to repository commit COMMIT.

Usage: python prep_isic_artifact.py --out data/isic_artifact [--limit N (local smoke test: first N per split)]
"""
import argparse, io, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMIT = 'f93c8654f50c581544f32ad6b1450cb925a176ff'
CSV = f'https://raw.githubusercontent.com/alceubissoto/debiasing-skin/{COMMIT}/trap-sets/isic_annotated_{{}}1.csv'
IMG = 'https://isic-archive.s3.amazonaws.com/images/{}.jpg'
ARTIFACTS = ['dark_corner', 'hair', 'gel_border', 'gel_bubble', 'ruler', 'ink', 'patches', 'vasc']


def fetch(url, tries=5):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            if k == tries - 1: raise
            time.sleep(2 ** k)


def to_jpeg256(b):
    from PIL import Image
    im = Image.open(io.BytesIO(b)); im.draft('RGB', (256, 256))
    out = io.BytesIO(); im.convert('RGB').resize((256, 256), Image.BILINEAR).save(out, 'JPEG', quality=90)
    return out.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/isic_artifact'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=32)
    o = ap.parse_args()
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    d = {s: pd.read_csv(io.BytesIO(fetch(CSV.format(s)))) for s in ('train', 'val', 'test')}
    d['train'] = d['train'][~d['train'].image.isin(d['val'].image)]
    assert not set(d['train'].image) & set(d['test'].image)
    out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    for s, name in (('train', 'train'), ('val', 'validation'), ('test', 'test')):
        x = d[s].sort_values('image').reset_index(drop=True)
        if o.limit: x = x.head(o.limit)
        ids = [i.rsplit('.', 1)[0] for i in x.image]
        with ThreadPoolExecutor(o.workers) as ex:
            imgs = list(ex.map(lambda i: to_jpeg256(fetch(IMG.format(i))), ids))
        cols = {'image': [{'bytes': b, 'path': f'{i}.jpg'} for i, b in zip(ids, imgs)],
                'label': x.label.astype(int).tolist(), 'place': x.ruler.astype(int).tolist(), 'image_id': ids}
        cols.update({a: x[a].astype(int).tolist() for a in ARTIFACTS})
        pq.write_table(pa.table(cols), out / f'{name}-00000-of-00001.parquet')
        g = x.groupby(['label', 'ruler']).size().to_dict()
        print(f"[isic_artifact] {name}: {len(x)} images, (label, ruler) cells {g}", flush=True)


if __name__ == '__main__':
    main()
