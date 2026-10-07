"""Build data/counteranimal/{train,validation,test}-00000-of-00001.parquet (Waterbirds format).

Source: Wang et al., "A Sober Look at the Robustness of CLIPs to Spurious Features" (NeurIPS 2024),
CounterAnimal: 13,334 iNaturalist photos of 45 ImageNet animal classes, each tagged with a background
('common-snow', 'counter-grass', ...). Released as a test set only (HF TMLR-Group-HF/counteranimal, 4 parquet
files, 1.8 GB, pinned to COMMIT), so we cut our own splits. Pre-registered construction
(PREREG_SCREEN_DATASETS.md):
  label = class index 0..44 in ImageNet-index order (column 'imagenet_idx' keeps the ImageNet id)
  place = 0 common background / 1 counter background (the 'type' prefix; full string kept in 'type')
  per class, seeded permutations of its common and its counter images:
    common  -> 60 % train, 10 % validation, 30 % test
    counter -> round(n_train_common * (1 - RHO) / RHO) to train (P(common | train) = RHO),
               the rest 1 : 3 to validation : test
Images resized to 256x256 (bilinear), JPEG q=90.

Usage: python prep_counteranimal.py --src <dir with test-0000?-of-00004.parquet> --out data/counteranimal
"""
import argparse, glob
from pathlib import Path
import numpy as np

COMMIT = 'e7811153bef8cf5aa274e22b9db4bf002156e376'   # https://huggingface.co/datasets/TMLR-Group-HF/counteranimal/resolve/<COMMIT>/data/test-0000{0..3}-of-00004.parquet
RHO = 0.95                                            # Waterbirds' train correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', default='data/counteranimal')
    o = ap.parse_args()
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    from prep_isic_artifact import to_jpeg256
    files = sorted(glob.glob(f'{o.src}/**/test-*.parquet', recursive=True))
    d = []
    for f in files:
        x = pq.read_table(f, columns=['type', 'label']).to_pandas(); x['file'], x['row'] = f, range(len(x)); d.append(x)
    d = pd.concat(d, ignore_index=True)
    d['imagenet_idx'] = d.label.str.split().str[0].astype(int)
    d['place'] = (~d.type.str.startswith('common')).astype(int)
    ids = sorted(d.imagenet_idx.unique())
    d['y'] = d.imagenet_idx.map({v: i for i, v in enumerate(ids)})
    rng = np.random.default_rng(0)
    d['split'] = ''
    for c in range(len(ids)):
        com = d.index[(d.y == c) & (d.place == 0)].to_numpy(); com = com[rng.permutation(len(com))]
        cnt = d.index[(d.y == c) & (d.place == 1)].to_numpy(); cnt = cnt[rng.permutation(len(cnt))]
        a, b = round(.6 * len(com)), round(.7 * len(com))
        d.loc[com[:a], 'split'], d.loc[com[a:b], 'split'], d.loc[com[b:], 'split'] = 'train', 'validation', 'test'
        k = round(a * (1 - RHO) / RHO); v = k + round((len(cnt) - k) / 4)
        d.loc[cnt[:k], 'split'], d.loc[cnt[k:v], 'split'], d.loc[cnt[v:], 'split'] = 'train', 'validation', 'test'
    assert (d.split != '').all()
    cells = d.groupby(['split', 'y', 'place']).size()                   # train.py's val/test AUC needs all 45 classes
    if len(cells) != 3 * 2 * len(ids) or len(ids) != 45:
        raise SystemExit(f'[counteranimal] {len(ids)} classes, {len(cells)}/{6 * len(ids)} (split, class, place) cells non-empty')
    img = {}
    for f in files:                                                   # one image-column read per file
        col = pq.read_table(f, columns=['image']).column('image')
        for i, r in zip(d.index[d.file == f], d.row[d.file == f]):
            img[i] = to_jpeg256(col[int(r)].as_py()['bytes'])
    out = Path(o.out); out.mkdir(parents=True, exist_ok=True)
    for s in ('train', 'validation', 'test'):
        x = d[d.split == s]
        tab = pa.table({'image': [{'bytes': img[i], 'path': f'{i}.jpg'} for i in x.index],
                        'label': x.y.astype(int).tolist(), 'place': x.place.astype(int).tolist(),
                        'imagenet_idx': x.imagenet_idx.astype(int).tolist(), 'type': x.type.tolist()})
        pq.write_table(tab, out / f'{s}-00000-of-00001.parquet')
        print(f"[counteranimal] {s}: {len(x)} images, {x.y.nunique()} classes, counter {x.place.mean():.3f}, "
              f"smallest (class, place) cell {x.groupby(['y', 'place']).size().min()}", flush=True)


if __name__ == '__main__':
    main()
