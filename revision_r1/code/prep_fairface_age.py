"""Build data/fairface_age/{train,validation,test}-00000-of-00001.parquet (Waterbirds format).

A BFFHQ-style age/gender benchmark (Lee et al., "Learning Debiased Representation via Disentangled Feature
Augmentation", NeurIPS 2021: target age, bias gender, young women and old men aligned) rebuilt on FairFace
(Karkkainen & Joo, WACV 2021; CC BY 4.0), because BFFHQ itself is only on Google Drive. Pre-registered
construction (PREREG_SCREEN_DATASETS.md):
  label = 0 young (FairFace age 10-29) / 1 old (40-59); other ages dropped (BFFHQ's age bands)
  place = gender (0 male, 1 female)
  train : from FairFace 'train' (padding 0.25): N_MAJ images in each aligned cell (young/female, old/male)
          and round(N_MAJ * (1 - RHO) / RHO) in each conflicting cell (young/male, old/female) -> P(aligned) = RHO
  valid / test : FairFace 'validation', seeded 1/3 : 2/3 split, then every (label, gender) cell cut to the
          same size (N_VAL / N_TEST), so both are group-balanced like the Waterbirds test set
Images: FairFace 224x224 crops, resized to 256x256 (bilinear), JPEG q=90. HF files pinned to COMMIT.

Usage: python prep_fairface_age.py --src <dir with the FairFace 0.25 parquet files> --out data/fairface_age
       [--n_maj 9000 --n_val 200 --n_test 500] (smaller values for a local smoke test)
"""
import argparse, glob
from pathlib import Path
import numpy as np

COMMIT = '54d573cdb8b5af490ba8da9da2799628f6e5c496'
FILES = ['0.25/train-00000-of-00002-d405faba4f4b9b85.parquet', '0.25/train-00001-of-00002-dd3cb68164727418.parquet',
         '0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet']     # https://huggingface.co/datasets/HuggingFaceM4/FairFace/resolve/<COMMIT>/<file>
RHO = 0.95                       # Waterbirds' train correlation
AGE = {2: 0, 3: 0, 5: 1, 6: 1}   # FairFace age class -> label: 10-19, 20-29 young; 40-49, 50-59 old


def load(files):
    import pyarrow.parquet as pq, pandas as pd
    ds = []
    for f in files:
        d = pq.read_table(f, columns=['age', 'gender']).to_pandas()
        d['file'], d['row'] = f, range(len(d)); ds.append(d)
    d = pd.concat(ds, ignore_index=True)
    d = d[d.age.isin(list(AGE))].copy()
    d['label'], d['place'] = d.age.map(AGE), d.gender
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', default='data/fairface_age')
    ap.add_argument('--n_maj', type=int, default=9000); ap.add_argument('--n_val', type=int, default=200)
    ap.add_argument('--n_test', type=int, default=500)
    o = ap.parse_args()
    import pyarrow as pa, pyarrow.parquet as pq
    from prep_isic_artifact import to_jpeg256
    rng = np.random.default_rng(0)
    tr = load(sorted(glob.glob(f'{o.src}/**/train-*.parquet', recursive=True)))
    va = load(sorted(glob.glob(f'{o.src}/**/validation-*.parquet', recursive=True)))
    n_min = round(o.n_maj * (1 - RHO) / RHO)

    def take(d, n):
        if len(d) < n: raise SystemExit(f'[fairface_age] cell has {len(d)} < {n} images')
        return d.iloc[rng.permutation(len(d))[:n]]
    parts = {'train': [take(tr[(tr.label == y) & (tr.place == g)], o.n_maj if y != g else n_min)
                       for y in (0, 1) for g in (0, 1)]}       # aligned: (0 young, 1 female), (1 old, 0 male)
    perm = np.random.default_rng(1).permutation(len(va)); cut = len(va) // 3     # own seed: independent of --n_maj
    vv, tt = va.iloc[perm[:cut]], va.iloc[perm[cut:]]
    parts['validation'] = [take(vv[(vv.label == y) & (vv.place == g)], o.n_val) for y in (0, 1) for g in (0, 1)]
    parts['test'] = [take(tt[(tt.label == y) & (tt.place == g)], o.n_test) for y in (0, 1) for g in (0, 1)]
    out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    for s, ps in parts.items():
        d = pd.concat(ps).sort_values(['file', 'row'])
        img = []
        for f, g in d.groupby('file', sort=True):                   # one image column read per source file
            col = pq.read_table(f, columns=['image']).column('image')
            img += [to_jpeg256(col[int(r)].as_py()['bytes']) for r in g.row]
        tab = pa.table({'image': [{'bytes': b, 'path': f'{Path(f).stem}_{r}.jpg'} for b, f, r in zip(img, d.file, d.row)],
                        'label': d.label.astype(int).tolist(), 'place': d.place.astype(int).tolist(),
                        'age': d.age.astype(int).tolist()})
        pq.write_table(tab, out / f'{s}-00000-of-00001.parquet')
        print(f"[fairface_age] {s}: {len(d)} images, (label, gender) cells {d.groupby(['label', 'place']).size().to_dict()}", flush=True)


if __name__ == '__main__':
    main()
