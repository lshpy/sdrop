"""
Dataset loaders for SDrop experiments.

Supported datasets
------------------
  CIFAR-100      : torchvision built-in, auto-download
  TinyImageNet   : requires manual download → see setup instructions below
  CUB-200-2011   : requires manual download → see setup instructions below

Directory structure expected
----------------------------
  data/
  ├── cifar100/                     ← auto-created by torchvision
  ├── tiny-imagenet-200/
  │   ├── train/
  │   │   ├── n01443537/
  │   │   │   ├── images/
  │   │   │   │   └── *.JPEG
  │   │   └── ...
  │   └── val/
  │       ├── images/               ← flat folder; run reformat_tinyimagenet()
  │       └── val_annotations.txt
  └── CUB_200_2011/
      ├── images/
      │   ├── 001.Black_footed_Albatross/
      │   └── ...
      ├── train_test_split.txt
      ├── images.txt
      └── classes.txt

Download links (fill in later)
-------------------------------
  TinyImageNet : http://cs231n.stanford.edu/tiny-imagenet-200.zip
  CUB-200-2011 : https://www.vision.caltech.edu/datasets/cub_200_2011/
                 (CUB_200_2011.tgz)
"""

import os
import io
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


def _pin_memory() -> bool:
    """pin_memory is only beneficial (and supported) on CUDA, not MPS or CPU."""
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def cifar100_transforms(strong: bool = False):
    """
    Return train/val transforms for CIFAR-100.

    strong=True   : adds RandAugment + Random Erasing — recommended for
                    ViT training-from-scratch (mitigates the heavy
                    overfitting otherwise observed in ViT-Tiny on CIFAR).
    strong=False  : the lightweight crop-and-flip pipeline used in the
                    original ResNet experiments (kept identical for
                    backward-compatible reproductions).
    """
    norm = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
    if strong:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.25),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, val_tf


def tinyimagenet_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2770, 0.2691, 0.2821)),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2770, 0.2691, 0.2821)),
    ])
    return train_tf, val_tf


def cub200_transforms():
    """Standard 224×224 transforms for CUB fine-tuning on ImageNet-pretrained ResNet."""
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


# 추가적인 데이터셋 사용 가능 

# ---------------------------------------------------------------------------
# CIFAR-100
# ---------------------------------------------------------------------------

def _ensure_cifar100(data_root: str):
    """Robust CIFAR-100 fetch.

    Strategy:
      1. Download the HuggingFace `uoft-cs/cifar100` parquet shards and
         convert them into torchvision's pickled format. This host is
         CDN-backed and consistently fast.
      2. Fall back to the canonical cs.toronto.edu tar.gz, which is often
         throttled below 100 KB/s and can take half an hour for 169 MB.
    """
    import os, urllib.request, tarfile
    extracted = os.path.join(data_root, 'cifar-100-python')
    if os.path.isdir(extracted) and os.path.isfile(os.path.join(extracted, 'meta')):
        return
    os.makedirs(data_root, exist_ok=True)

    # The canonical cs.toronto.edu host is frequently throttled to well under
    # 100 KB/s, which turns a 169 MB fetch into a half-hour stall, so it is the
    # *fallback* here rather than the first choice.
    target = os.path.join(data_root, 'cifar-100-python.tar.gz')

    # ---- attempt 1: HuggingFace parquet -> repickle into torchvision format ----
    try:
        print('  CIFAR-100: falling back to HuggingFace parquet mirror')
        import io, pickle, numpy as np
        from PIL import Image
        try:
            import pyarrow.parquet as pq
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyarrow'])
            import pyarrow.parquet as pq

        os.makedirs(extracted, exist_ok=True)
        meta = {
            'fine_label_names': [s.encode() for s in [
                'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
                'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
                'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
                'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
                'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
                'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
                'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
                'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
                'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
                'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']],
            'coarse_label_names': [s.encode() for s in [
                'aquatic_mammals','fish','flowers','food_containers','fruit_and_vegetables','household_electrical_devices',
                'household_furniture','insects','large_carnivores','large_man-made_outdoor_things',
                'large_natural_outdoor_scenes','large_omnivores_and_herbivores','medium_mammals','non-insect_invertebrates',
                'people','reptiles','small_mammals','trees','vehicles_1','vehicles_2']],
        }
        with open(os.path.join(extracted, 'meta'), 'wb') as f:
            pickle.dump(meta, f)

        for split, url, out_name in [
            ('train', 'https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet', 'train'),
            ('test',  'https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet',  'test'),
        ]:
            print(f'  fetching HF {split} parquet')
            tmp = os.path.join(data_root, f'{split}.parquet')
            urllib.request.urlretrieve(url, tmp)
            tbl = pq.read_table(tmp).to_pandas()
            data = np.empty((len(tbl), 3 * 32 * 32), dtype=np.uint8)
            fine = []
            coarse = []
            filenames = []
            for i, row in tbl.iterrows():
                img_bytes = row['img']['bytes']
                img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))   # (32,32,3) HWC
                data[i] = img.transpose(2, 0, 1).reshape(-1)                       # CHW flat as torchvision expects
                fine.append(int(row['fine_label']))
                coarse.append(int(row['coarse_label']))
                filenames.append(f'{split}_{i:05d}'.encode())
            obj = {
                'data': data,
                'fine_labels': fine,
                'coarse_labels': coarse,
                'filenames': filenames,
                'batch_label': split.encode(),
            }
            with open(os.path.join(extracted, out_name), 'wb') as f:
                pickle.dump(obj, f)
            os.remove(tmp)
        print('  CIFAR-100 (from HF) ready at', extracted)
        return
    except Exception as e:
        print('  HuggingFace mirror failed:', e)

    # ---- attempt 3: canonical cs.toronto.edu tar.gz (often heavily throttled) ----
    try:
        print('  CIFAR-100: falling back to cs.toronto.edu tar.gz (may be slow)')
        urllib.request.urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', target)
        with tarfile.open(target) as t:
            t.extractall(data_root)
        print('  CIFAR-100 ready at', extracted)
        return
    except Exception as e:
        raise RuntimeError(f'all CIFAR-100 mirrors failed; last error: {e}')


class _CIFAR100NoIntegrityCheck(datasets.CIFAR100):
    """torchvision CIFAR100 with integrity checks disabled — needed when the
    files come from a non-canonical mirror (HF parquet) whose MD5 differs."""
    def _check_integrity(self) -> bool:
        import os
        # accept the dataset as long as the pickled split files exist
        for name, _ in (self.train_list + self.test_list):
            if not os.path.isfile(os.path.join(self.root, self.base_folder, name)):
                return False
        return True

    def _load_meta(self) -> None:
        import os, pickle
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        self.classes = data[self.meta["key"]]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}


def get_cifar100(data_root: str = './data', batch_size: int = 128,
                 num_workers: int = 4, strong_aug: bool = False):
    train_tf, val_tf = cifar100_transforms(strong=strong_aug)
    _ensure_cifar100(data_root)
    train_ds = _CIFAR100NoIntegrityCheck(data_root, train=True,  download=False, transform=train_tf)
    val_ds   = _CIFAR100NoIntegrityCheck(data_root, train=False, download=False, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=_pin_memory())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_pin_memory())
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CIFAR-100-LT  (long-tailed) and label-noise / subset variants
# ---------------------------------------------------------------------------

def _exp_profile(n_classes: int, n_max: int, imb_ratio: float):
    """
    Exponential imbalance profile used by the long-tailed literature
    (Cui et al. 2019, Cao et al. 2019):

        n_k = n_max * (1/imb_ratio)^{k/(K-1)},   k = 0 .. K-1

    so class 0 keeps all n_max samples and class K-1 keeps n_max/imb_ratio.
    """
    return [int(round(n_max * (1.0 / imb_ratio) ** (k / (n_classes - 1))))
            for k in range(n_classes)]


def make_longtail_indices(targets, n_classes: int, imb_ratio: float, seed: int = 0):
    """Return indices of a class-imbalanced subset with an exponential profile."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    n_max = int(np.bincount(targets, minlength=n_classes).max())
    counts = _exp_profile(n_classes, n_max, imb_ratio)
    idx = []
    for k, n_k in enumerate(counts):
        cls_idx = np.where(targets == k)[0]
        rng.shuffle(cls_idx)
        idx.extend(cls_idx[:n_k].tolist())
    rng.shuffle(idx)
    return idx, counts


def corrupt_labels(targets, n_classes: int, noise_rate: float, seed: int = 0):
    """Symmetric label noise: with prob. noise_rate replace the label uniformly."""
    rng = np.random.RandomState(seed)
    targets = np.array(targets, copy=True)
    n_flip = int(round(noise_rate * len(targets)))
    flip_idx = rng.choice(len(targets), size=n_flip, replace=False)
    targets[flip_idx] = rng.randint(0, n_classes, size=n_flip)
    return targets.tolist()


def get_cifar100_lt(data_root: str = './data', batch_size: int = 128,
                    num_workers: int = 4, strong_aug: bool = False,
                    imb_ratio: float = 100.0, subset_frac: float = 1.0,
                    label_noise: float = 0.0, seed: int = 0):
    """
    CIFAR-100 with optional long-tail imbalance, random subsetting and
    symmetric label noise. The *validation* set is always left untouched and
    balanced, which is the standard protocol for long-tailed benchmarks.

    imb_ratio    : n_max / n_min  (100 -> head 500 imgs, tail 5 imgs).
                   1.0 disables imbalance.
    subset_frac  : keep this fraction of the (already imbalanced) train set.
    label_noise  : symmetric label-noise rate applied to the train split.
    """
    train_tf, val_tf = cifar100_transforms(strong=strong_aug)
    _ensure_cifar100(data_root)
    train_ds = _CIFAR100NoIntegrityCheck(data_root, train=True,  download=False, transform=train_tf)
    val_ds   = _CIFAR100NoIntegrityCheck(data_root, train=False, download=False, transform=val_tf)

    if label_noise > 0.0:
        train_ds.targets = corrupt_labels(train_ds.targets, 100, label_noise, seed)

    if imb_ratio and imb_ratio > 1.0:
        idx, counts = make_longtail_indices(train_ds.targets, 100, imb_ratio, seed)
        print(f"[CIFAR-100-LT] imb_ratio={imb_ratio:g}  "
              f"head={counts[0]}  tail={counts[-1]}  total={len(idx)}")
    else:
        idx = list(range(len(train_ds)))

    if subset_frac < 1.0:
        rng = np.random.RandomState(seed + 1)
        keep = int(round(len(idx) * subset_frac))
        idx = rng.permutation(idx)[:keep].tolist()
        print(f"[subset] keeping {keep} training images ({subset_frac:.0%})")

    if label_noise > 0.0:
        print(f"[label noise] symmetric rate = {label_noise:.0%}")

    train_subset = Subset(train_ds, idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=_pin_memory(),
                              drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_pin_memory())
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# TinyImageNet
# ---------------------------------------------------------------------------

def reformat_tinyimagenet_val(data_root: str = './data'):
    """
    TinyImageNet validation set ships as a flat folder of images +
    val_annotations.txt.  This function reorganises it into ImageFolder
    format:  val/<class_id>/image.JPEG

    Run once before training.
    """
    val_dir   = Path(data_root) / 'tiny-imagenet-200' / 'val'
    img_dir   = val_dir / 'images'
    annot_file = val_dir / 'val_annotations.txt'

    if not annot_file.exists():
        print(f"[WARNING] {annot_file} not found — skipping reformat.")
        return

    # parse annotations
    img_to_class = {}
    with open(annot_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            img_to_class[parts[0]] = parts[1]

    # move images into class sub-directories
    for img_name, class_id in img_to_class.items():
        src = img_dir / img_name
        dst_dir = val_dir / class_id
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / img_name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))

    # 빈 images/ 가 남으면 ImageFolder 가 201번째 클래스로 잡아 val 라벨이
    # 전부 +1 밀린다 (에러 없이 정확도만 0 이 된다).
    if img_dir.exists() and not any(img_dir.iterdir()):
        img_dir.rmdir()
    elif img_dir.exists():
        shutil.rmtree(img_dir)
    print(f"[INFO] TinyImageNet val reformatted: {val_dir}")


def get_tinyimagenet(data_root: str = './data', batch_size: int = 128,
                     num_workers: int = 4):
    root = Path(data_root) / 'tiny-imagenet-200'
    # val/ 이 평평한 구조면 ImageFolder 가 1-클래스 셋을 만든다. 자동 재구성한다.
    _val = root / 'val'
    _ncls = len([d for d in _val.iterdir() if d.is_dir()]) if _val.exists() else 0
    if (_val / 'images').exists() or _ncls != 200:
        reformat_tinyimagenet_val(data_root)
        _ncls = len([d for d in _val.iterdir() if d.is_dir()])
        if _ncls != 200:
            raise RuntimeError(
                f"TinyImageNet val 재구성 실패: 클래스 디렉터리 {_ncls}개 (200 이어야 함)")
    train_tf, val_tf = tinyimagenet_transforms()
    train_ds = datasets.ImageFolder(root / 'train', transform=train_tf)
    val_ds   = datasets.ImageFolder(root / 'val',   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=_pin_memory())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_pin_memory())
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CUB-200-2011
# ---------------------------------------------------------------------------

class CUB200Dataset(Dataset):
    """
    CUB-200-2011 dataset.

    Reads the official train/test split from train_test_split.txt and
    images.txt; no extra preprocessing needed beyond the zip extraction.

    Args:
        root      : path to 'CUB_200_2011/' directory
        train     : True for train split, False for test split
        transform : torchvision transforms
    """
    def __init__(self, root: str, train: bool = True,
                 transform=None):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self._load_metadata()

    def _load_metadata(self):
        # image_id → file_path
        id2path = {}
        with open(self.root / 'images.txt') as f:
            for line in f:
                img_id, path = line.strip().split()
                id2path[int(img_id)] = path

        # image_id → label (1-indexed → 0-indexed)
        id2label = {}
        with open(self.root / 'image_class_labels.txt') as f:
            for line in f:
                img_id, label = line.strip().split()
                id2label[int(img_id)] = int(label) - 1

        # image_id → is_training_image
        with open(self.root / 'train_test_split.txt') as f:
            split = {int(a): int(b) for a, b in
                     (line.strip().split() for line in f)}

        self.samples = []
        for img_id, is_train in split.items():
            if bool(is_train) == self.train:
                path = self.root / 'images' / id2path[img_id]
                label = id2label[img_id]
                self.samples.append((str(path), label))

    @property
    def targets(self):
        return [lab for _, lab in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cub200(data_root: str = './data', batch_size: int = 64,
               num_workers: int = 4):
    """
    Args:
        data_root : parent directory containing 'CUB_200_2011/'
    """
    root = Path(data_root) / 'CUB_200_2011'
    if not root.exists():
        raise FileNotFoundError(
            f"CUB-200-2011 not found at {root}.\n"
            "Download from: https://www.vision.caltech.edu/datasets/cub_200_2011/\n"
            "Extract so that: data/CUB_200_2011/images/ exists."
        )
    train_tf, val_tf = cub200_transforms()
    train_ds = CUB200Dataset(root, train=True,  transform=train_tf)
    val_ds   = CUB200Dataset(root, train=False, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=_pin_memory())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_pin_memory())
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Waterbirds (Sagawa et al., 2020) — subpopulation shift 벤치마크
# ---------------------------------------------------------------------------
# CUB 새를 Places 배경(land/water)에 합성. 학습에서는 label(landbird/waterbird)과
# place 가 95% 상관, 테스트에서는 독립. 소수 그룹(waterbird+land, landbird+water)
# 의 worst-group accuracy 가 표준 지표. HF `grodino/waterbirds` parquet 사용.
#   data/waterbirds/{train,validation,test}-00000-of-00001.parquet
# 공식 validation split 이 있으므로 checkpoint 선택은 validation, 최종 평가는 test.

class WaterbirdsDataset(Dataset):
    """parquet 한 장을 통째로 메모리에 올린다(≈200MB). 이미지는 JPEG bytes 로
    저장돼 있어 __getitem__ 에서 디코딩한다. `places` 로 그룹 평가가 가능하다."""
    def __init__(self, parquet_path, transform=None):
        import pyarrow.parquet as pq
        t = pq.read_table(parquet_path, columns=['image', 'label', 'place'])
        self._bytes = [d['bytes'] for d in t.column('image').to_pylist()]
        self.targets = t.column('label').to_pylist()
        self.places = t.column('place').to_pylist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self._bytes[i])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


def waterbirds_transforms():
    # Group-DRO/JTT/DFR 관례: 증강 없이 Resize(256)→CenterCrop(224).
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return tf, tf


def get_waterbirds(data_root: str = './data', batch_size: int = 64,
                   num_workers: int = 4, name: str = 'waterbirds'):
    """Returns (train_loader, val_loader, test_loader).
    name: sub-directory under data_root. Any parquet triple with columns image{bytes}/label/place in
    the same layout loads the same way -- used by the pre-registered 'cct20' and 'isic_artifact'
    (built by prep_cct20.py / prep_isic_artifact.py, see PREREG_DATASETS.md)."""
    root = Path(data_root) / name
    files = {s: root / f'{s}-00000-of-00001.parquet' for s in ('train', 'validation', 'test')}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Waterbirds parquet 없음: " + ", ".join(missing) + "\n"
            "받기: huggingface_hub.snapshot_download('grodino/waterbirds', repo_type='dataset', "
            "allow_patterns='data/*', local_dir=...) 후 data/ 안의 파일을 data/waterbirds/ 로.")
    train_tf, val_tf = waterbirds_transforms()
    tr = WaterbirdsDataset(files['train'], train_tf)
    va = WaterbirdsDataset(files['validation'], val_tf)
    te = WaterbirdsDataset(files['test'], val_tf)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=_pin_memory())
    return mk(tr, True), mk(va, False), mk(te, False)


# ---------------------------------------------------------------------------
# CelebA (Blond_Hair × Male) — subpopulation shift 표준 2번째 벤치마크
# ---------------------------------------------------------------------------
# label = Blond_Hair, spurious = Male. 소수 그룹 = 금발 남성. HF `tpremoli/CelebA-attrs`
# parquet (공식 train/validation/test 분할, 40개 속성 열). data/celeba/*.parquet
# 관례(Sagawa'20): CenterCrop(178) → Resize(224), 증강 없음.

class CelebADataset(Dataset):
    def __init__(self, parquet_paths, transform=None, label='Blond_Hair', spurious='Male'):
        import pyarrow.parquet as pq
        tabs=[pq.read_table(p, columns=['image', label, spurious]) for p in parquet_paths]
        self._bytes=[d['bytes'] for t in tabs for d in t.column('image').to_pylist()]
        self.targets=[int(v > 0) for t in tabs for v in t.column(label).to_pylist()]
        self.places =[int(v > 0) for t in tabs for v in t.column(spurious).to_pylist()]
        self.transform=transform
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        img=Image.open(io.BytesIO(self._bytes[i])).convert('RGB')
        if self.transform is not None: img=self.transform(img)
        return img, self.targets[i]


def celeba_transforms():
    tf=transforms.Compose([transforms.CenterCrop(178), transforms.Resize(224), transforms.ToTensor(),
                           transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    return tf, tf


def get_celeba(data_root: str = './data', batch_size: int = 128, num_workers: int = 4,
               label: str = 'Blond_Hair', spurious: str = 'Male'):
    """Returns (train_loader, val_loader, test_loader). label/spurious: CelebA attribute columns
    (defaults = the standard Blond_Hair x Male; other pairs are screening datasets, PREREG_SCREEN_DATASETS.md)."""
    root=Path(data_root)/'celeba'
    tr=sorted(root.glob('train-*.parquet')); va=sorted(root.glob('validation-*.parquet')); te=sorted(root.glob('test-*.parquet'))
    if not (tr and va and te):
        raise FileNotFoundError(f"CelebA parquet 없음 in {root}: snapshot_download('tpremoli/CelebA-attrs', repo_type='dataset', allow_patterns='data/*') 후 data/celeba/ 로")
    tf,_=celeba_transforms()
    mk=lambda ds,sh: DataLoader(ds,batch_size=batch_size,shuffle=sh,num_workers=num_workers,pin_memory=_pin_memory())
    ds=lambda p: CelebADataset(p,tf,label=label,spurious=spurious)
    return mk(ds(tr),True), mk(ds(va),False), mk(ds(te),False)


# ---------------------------------------------------------------------------
# Unified getter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CIFAR-10 / SVHN — Guided Dropout(AAAI'19) 계열이 표준으로 쓰는 데이터셋 축
#
# Guided Dropout 은 MNIST/CIFAR10/CIFAR100/SVHN/TinyImageNet 5종을 보고한다.
# 부호 일관성 전략에서는 데이터셋 축을 늘리는 것이 시드를 쌓는 것보다
# 훨씬 싸고 강하다. 두 데이터셋 모두 32x32 라 CIFAR-100 과 같은 백본·같은
# 학습 예산을 그대로 쓸 수 있어 셀 추가 비용이 가장 낮다.
# ---------------------------------------------------------------------------

def cifar10_transforms(strong: bool = False):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
    base = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if strong:
        base.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_tf = transforms.Compose(base + [transforms.ToTensor(), norm]
                                  + ([transforms.RandomErasing(p=0.25)] if strong else []))
    val_tf = transforms.Compose([transforms.ToTensor(), norm])
    return train_tf, val_tf


def svhn_transforms(strong: bool = False):
    # SVHN 은 숫자라 좌우 반전을 쓰면 안 된다 (2 <-> 5 등이 뒤집힌다).
    norm = transforms.Normalize((0.4377, 0.4438, 0.4728),
                                (0.1980, 0.2010, 0.1970))
    base = [transforms.RandomCrop(32, padding=4)]
    if strong:
        base.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_tf = transforms.Compose(base + [transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.ToTensor(), norm])
    return train_tf, val_tf


def _ensure_cifar10(data_root: str):
    """CIFAR-10 을 HuggingFace parquet 미러에서 받아 torchvision 형식으로 변환한다.

    canonical cs.toronto.edu 는 실측 34 KB/s 로 스로틀돼 170 MB 에 1시간 이상
    걸린다. HF 미러는 같은 시점 48 MB/s 였다 (_ensure_cifar100 과 같은 이유).
    """
    import os, io, pickle, urllib.request
    import numpy as np
    from PIL import Image
    extracted = os.path.join(data_root, 'cifar-10-batches-py')
    if os.path.isdir(extracted) and os.path.isfile(os.path.join(extracted, 'batches.meta')):
        return
    os.makedirs(extracted, exist_ok=True)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyarrow'])
        import pyarrow.parquet as pq

    names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']
    with open(os.path.join(extracted, 'batches.meta'), 'wb') as f:
        pickle.dump({'label_names': names}, f)

    base = 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text'
    for split, out_name in [('train', 'data_batch_1'), ('test', 'test_batch')]:
        print(f'  CIFAR-10: fetching HF {split} parquet')
        tmp = os.path.join(data_root, f'cifar10_{split}.parquet')
        urllib.request.urlretrieve(f'{base}/{split}-00000-of-00001.parquet', tmp)
        tbl = pq.read_table(tmp).to_pandas()
        img_col = 'img' if 'img' in tbl.columns else 'image'
        lab_col = 'label' if 'label' in tbl.columns else 'fine_label'
        data = np.empty((len(tbl), 3 * 32 * 32), dtype=np.uint8)
        labels = []
        for i, (im, lb) in enumerate(zip(tbl[img_col], tbl[lab_col])):
            raw = im['bytes'] if isinstance(im, dict) else im
            arr = np.array(Image.open(io.BytesIO(raw)).convert('RGB'), dtype=np.uint8)
            data[i] = arr.transpose(2, 0, 1).reshape(-1)      # CHW flatten
            labels.append(int(lb))
        # torchvision 은 encoding='latin1' 로 읽는데, Python3 pickle 의 bytes 키는
        # 그대로 bytes 로 남는다. 문자열 키를 써야 entry["data"] 가 잡힌다.
        with open(os.path.join(extracted, out_name), 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        os.remove(tmp)
        print(f'  CIFAR-10: {split} -> {out_name} ({len(labels)} images)')


class _CIFAR10NoIntegrityCheck(datasets.CIFAR10):
    """변환본은 원본 체크섬과 다르므로 무결성 검사를 건너뛴다."""
    train_list = [['data_batch_1', None]]
    test_list = [['test_batch', None]]
    meta = {'filename': 'batches.meta', 'key': 'label_names', 'md5': None}

    def _check_integrity(self):
        import os
        root = os.path.join(self.root, self.base_folder)
        return all(os.path.isfile(os.path.join(root, f))
                   for f in ('batches.meta', 'data_batch_1', 'test_batch'))


def get_cifar10(data_root: str = './data', batch_size: int = 128,
                num_workers: int = 4, strong_aug: bool = False):
    train_tf, val_tf = cifar10_transforms(strong_aug)
    _ensure_cifar10(data_root)
    tr = _CIFAR10NoIntegrityCheck(data_root, train=True, download=False, transform=train_tf)
    va = _CIFAR10NoIntegrityCheck(data_root, train=False, download=False, transform=val_tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=_pin_memory()),
            DataLoader(va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=_pin_memory()))


def get_svhn(data_root: str = './data', batch_size: int = 128,
             num_workers: int = 4, strong_aug: bool = False):
    train_tf, val_tf = svhn_transforms(strong_aug)
    tr = datasets.SVHN(data_root, split='train', download=True, transform=train_tf)
    va = datasets.SVHN(data_root, split='test', download=True, transform=val_tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=_pin_memory()),
            DataLoader(va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=_pin_memory()))


def mnist_transforms(strong: bool = False):
    """MNIST. Guided Dropout(AAAI'19) 이 벤치마크로 쓰는 5종 중 하나.

    28x28 회색조를 32x32 3채널로 맞춘다 — CIFAR 용 백본(3ch, 32x32)을
    그대로 쓸 수 있어야 아키텍처 축이 데이터셋 축과 직교한다.
    좌우 반전은 쓰지 않는다 (숫자가 뒤집힌다).
    """
    norm = transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)
    base = [transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3)]
    aug = [transforms.RandomCrop(32, padding=4)]
    if strong:
        aug.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_tf = transforms.Compose(base + aug + [transforms.ToTensor(), norm])
    val_tf = transforms.Compose(base + [transforms.ToTensor(), norm])
    return train_tf, val_tf


def get_mnist(data_root: str = './data', batch_size: int = 128,
              num_workers: int = 4, strong_aug: bool = False):
    train_tf, val_tf = mnist_transforms(strong_aug)
    tr = datasets.MNIST(data_root, train=True, download=True, transform=train_tf)
    va = datasets.MNIST(data_root, train=False, download=True, transform=val_tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=_pin_memory()),
            DataLoader(va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=_pin_memory()))


DATASET_CONFIG = {
    'cifar100': {
        'num_classes': 100,
        'arch':        'resnet18',
        'pretrained':  False,
        'getter':      get_cifar100,
    },
    'cifar100_lt': {
        'num_classes': 100,
        'arch':        'resnet18',
        'pretrained':  False,
        'getter':      get_cifar100_lt,
    },
    'mnist': {
        'num_classes': 10,
        'arch':        'resnet18',
        'pretrained':  False,
        'getter':      get_mnist,
    },
    'cifar10': {
        'num_classes': 10,
        'arch':        'resnet18',
        'pretrained':  False,
        'getter':      get_cifar10,
    },
    'svhn': {
        'num_classes': 10,
        'arch':        'resnet18',
        'pretrained':  False,
        'getter':      get_svhn,
    },
    'tinyimagenet': {
        'num_classes': 200,
        'arch':        'resnet50',
        'pretrained':  False,
        'getter':      get_tinyimagenet,
    },
    'cub200': {
        'num_classes': 200,
        'arch':        'resnet50',
        'pretrained':  True,          # ImageNet pretrain strongly recommended
        'getter':      get_cub200,
    },
    'waterbirds': {
        'num_classes': 2,
        'arch':        'resnet50',
        'pretrained':  True,
        'getter':      get_waterbirds,   # (train, val, test) 3-튜플을 낸다
        'has_test':    True,
    },
    'celeba': {
        'num_classes': 2,
        'arch':        'resnet50',
        'pretrained':  True,
        'getter':      get_celeba,
        'has_test':    True,
    },
    **{f'spawrious_{c}': {'num_classes': 4, 'arch': 'resnet50', 'pretrained': True, 'has_test': True,
                          'getter': (lambda c: (lambda *a, **k: get_spawrious(c, *a, **k)))(c)}
       for c in ('o2o_easy', 'o2o_medium', 'o2o_hard')},
    **{f'cmnist_{b}': {'num_classes': 10, 'arch': 'resnet18', 'pretrained': False, 'has_test': True,
                       'getter': (lambda b: (lambda *a, **k: get_cmnist(b, *a, **k)))(float(b))}
       for b in ('0.95', '0.98', '0.995')},
    'bar': {'num_classes': 6, 'arch': 'resnet50', 'pretrained': True, 'has_test': True, 'getter': (lambda *a, **k: get_bar(*a, **k))},
    'metashift': {'num_classes': 2, 'arch': 'resnet50', 'pretrained': True, 'has_test': True, 'getter': (lambda *a, **k: get_metashift(*a, **k))},
    'nicopp': {'num_classes': 2, 'arch': 'resnet50', 'pretrained': True, 'has_test': True, 'getter': (lambda *a, **k: get_nicopp(*a, **k))},
    # pre-registered field datasets (2026-09-29, PREREG_DATASETS.md): Waterbirds-format parquet, place = confound
    'cct20': {'num_classes': 12, 'arch': 'resnet50', 'pretrained': True, 'has_test': True,
              'getter': (lambda *a, **k: get_waterbirds(*a, name='cct20', **k))},          # place = camera location id
    'isic_artifact': {'num_classes': 2, 'arch': 'resnet50', 'pretrained': True, 'has_test': True,
                      'getter': (lambda *a, **k: get_waterbirds(*a, name='isic_artifact', **k))},  # place = ruler present
    # screening set (2026-09-29, PREREG_SCREEN_DATASETS.md): Waterbirds-format parquet, place = confound
    **{n: {'num_classes': c, 'arch': 'resnet50', 'pretrained': True, 'has_test': True,
           'getter': (lambda n: (lambda *a, **k: get_waterbirds(*a, name=n, **k)))(n)}
       for n, c in (('in9bg', 9),              # place = IN-9 class of the pasted background
                    ('counteranimal', 45),     # place = counter (1) vs common (0) background
                    ('fairface_age', 2),       # label young/old, place = gender
                    ('in9bg_s05', 9),          # object-size experiment (prereg/PREREG_objsize_2026-09-29.md): object x0.5
                    ('in9bg_s03', 9),          # object x0.3
                    ('in9wb', 2),              # shared-axis test (prereg/PREREG_in9wb_2026-10-01.md): dog/bird x fish-bg/vehicle-bg, rho 0.95
                    ('wbscale_s05', 2), ('wbscale_s10', 2), ('wbscale_s15', 2), ('wbscale_s20', 2),  # Waterbirds, bird x0.5/1.5/2.0 (prep_waterbirds_scaled.py)
                    ('aerialwb', 2),           # aerial Waterbirds (composite): label 1 ship / 0 plane, place 1 water / 0 land scene (prep_aerialwb.py, AERIALWB.md)
                    ('coco_catdog', 2),        # natural small-object COCO cat/dog, place 0 indoor / 1 outdoor (prep_coco_smallobj.py, NATURAL_SMALLOBJ.md)
                    ('oi_waterbirds', 2))},     # real-photo Waterbirds from Open Images V7, label 1 waterbird, place 1 water (prep_oi_waterbirds.py, OI_WATERBIRDS.md)
    # screening CelebA pairs: same images as 'celeba', other target attribute, confound Male
    **{n: {'num_classes': 2, 'arch': 'resnet50', 'pretrained': True, 'has_test': True,
           'getter': (lambda t: (lambda *a, **k: get_celeba(*a, label=t, spurious='Male', **k)))(t)}
       for n, t in (('celeba_wavy', 'Wavy_Hair'), ('celeba_glasses', 'Eyeglasses'))},
}


def get_dataset(name: str, data_root: str = './data', batch_size: int = 128,
                num_workers: int = 4, strong_aug: bool = False,
                imb_ratio: float = 1.0, subset_frac: float = 1.0,
                label_noise: float = 0.0, seed: int = 0,
                val_frac: float = 0.0, split_seed: int = None):
    """
    Returns (train_loader, val_loader, num_classes, default_arch, pretrained, test_loader).

    val_frac > 0 이면 (감사 ⑤ 프로토콜) 훈련 집합 안에서 클래스 층화로 val_frac 만큼을
    떼어 validation 으로 쓰고, 원래 val_loader(=공식 test)는 test_loader 로 넘긴다.
    checkpoint 선택은 validation, 최종 보고는 test. 분할 인덱스는
    train_loader.split_indices 에 담겨 train.py 가 json 으로 저장한다.
    split_seed 는 모델 seed 와 분리(기본값은 seed 와 같음).
    test_loader 는 공식 val/test 가 분리된 데이터셋(waterbirds)에서만 있고 그 외 None.
    None 이면 train.py 는 기존 관례대로 val_loader(=test set)로 checkpoint 를 고른다.

    imb_ratio / label_noise 는 'cifar100' / 'cifar100_lt' 에만 적용된다.
    subset_frac 은 **모든 데이터셋**에 적용된다 (아래 _apply_subset).

    [주의] 예전에는 subset_frac 이 cifar100 이외에서 조용히 무시됐다.
    train.py 의 run_id 는 _sub 태그를 붙이므로, 무시되면 전량 런이
    저데이터 런의 이름으로 저장된다. 지원하지 않는 조합은 무시가 아니라
    오류를 내야 한다.
    """
    if name.startswith('nicopp:'):                  # 'nicopp:<classA>,<classB>:<ctx0>,<ctx1>'
        cls, ctx = (x.split(',') for x in name[len('nicopp:'):].split(':'))
        assert len(cls) == 2 and len(ctx) == 2, name
        cfg = dict(DATASET_CONFIG['nicopp'], getter=lambda *a, **k: get_nicopp(*a, classes=cls, contexts=ctx, **k))
    elif name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from: {list(DATASET_CONFIG.keys())} or nicopp:<A>,<B>:<ctx0>,<ctx1>")
    else:
        cfg = DATASET_CONFIG[name]
    getter = cfg['getter']
    test_loader = None
    if name == 'cifar100_lt' or (name == 'cifar100' and
                                 (imb_ratio > 1.0 or subset_frac < 1.0 or label_noise > 0.0)):
        train_loader, val_loader = get_cifar100_lt(
            data_root, batch_size, num_workers, strong_aug=strong_aug,
            imb_ratio=imb_ratio, subset_frac=subset_frac,
            label_noise=label_noise, seed=seed)
    elif name in ('cifar100', 'cifar10', 'svhn', 'mnist'):
        train_loader, val_loader = getter(data_root, batch_size, num_workers,
                                          strong_aug=strong_aug)
        if subset_frac < 1.0:
            train_loader = _apply_subset(train_loader, subset_frac, seed,
                                         batch_size, num_workers)
    else:
        if imb_ratio > 1.0 or label_noise > 0.0:
            raise ValueError(
                f"'{name}' 은 imb_ratio / label_noise 를 지원하지 않는다 "
                f"(요청: imb_ratio={imb_ratio}, label_noise={label_noise}). "
                "조용히 무시하면 run_id 태그와 실제 설정이 어긋난다.")
        out = getter(data_root, batch_size, num_workers)
        if cfg.get('has_test'):
            train_loader, val_loader, test_loader = out
        else:
            train_loader, val_loader = out
        if subset_frac < 1.0:
            train_loader = _apply_subset(train_loader, subset_frac, seed,
                                         batch_size, num_workers)
    if val_frac > 0.0:
        if test_loader is not None:
            raise ValueError(f"'{name}' 은 공식 validation split 이 있어 val_frac 을 쓰지 않는다.")
        test_loader = val_loader
        train_loader._val_transform = getattr(val_loader.dataset, 'transform', None)
        train_loader, val_loader = _split_train_val(
            train_loader, val_frac, seed if split_seed is None else split_seed,
            batch_size, num_workers)
    return (train_loader, val_loader,
            cfg['num_classes'], cfg['arch'], cfg['pretrained'], test_loader)


def _split_train_val(train_loader, val_frac: float, split_seed: int,
                     batch_size: int, num_workers: int):
    """훈련 로더의 데이터셋(Subset 일 수 있음)을 클래스 층화로 train/val 로 나눈다.
    val 쪽은 증강 없는 test 변환을 쓴다(같은 원본 데이터셋을 얕게 복사해 transform 만 교체)."""
    import copy
    ds = train_loader.dataset
    if isinstance(ds, Subset):
        base, idx = ds.dataset, list(ds.indices)
    else:
        base, idx = ds, list(range(len(ds)))
    targets = base.targets
    by_cls = {}
    for i in idx:
        by_cls.setdefault(int(targets[i]), []).append(i)
    rng = np.random.RandomState(split_seed + 2)
    tr, va = [], []
    for c, ids in sorted(by_cls.items()):
        ids = list(rng.permutation(ids))
        k = int(round(len(ids) * val_frac)) if len(ids) >= 2 else 0
        va += ids[:k]; tr += ids[k:]
    tr, va = sorted(map(int, tr)), sorted(map(int, va))
    # val 은 증강 없이: 원본 데이터셋을 얕게 복사하고 transform 을 test 변환으로
    base_eval = copy.copy(base)
    vt = getattr(train_loader, '_val_transform', None)
    if vt is not None:
        base_eval.transform = vt
    print(f"[val split] train {len(tr)} / val {len(va)}  (val_frac={val_frac:g}, split_seed={split_seed})")
    tl = DataLoader(Subset(base, tr), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=_pin_memory())
    vl = DataLoader(Subset(base_eval, va), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=_pin_memory())
    tl.split_indices = {'train': tr, 'val': va, 'split_seed': split_seed, 'val_frac': val_frac}
    return tl, vl


# ---------------------------------------------------------------------------
# Spawrious (Lynch et al. 2023) O2O challenges — 4 dog breeds × backgrounds.
# ---------------------------------------------------------------------------
# Train: each breed on its own background (folders 0/1: 3072 + 2756 images) plus a
# filler background shared by all breeds (96 + 412 per breed) → own-background ≈ 92%.
# Test: each breed on a *different* breed's training background (both folders).
# Layout: data/spawrious/spawrious224/{0,1}/{location}/{breed}/*.png  (official tarballs).
# Group = (breed, location); worst-group over the 4 test (breed, location) cells.

class SpawriousDataset(Dataset):
    CLASSES = ["bulldog", "corgi", "dachshund", "labrador"]
    LOCS = ["desert", "jungle", "dirt", "mountain", "snow", "beach"]
    COMBOS = {  # (train group per class, test group per class, filler)
        'o2o_easy':   (["desert", "jungle", "dirt", "snow"],     ["dirt", "snow", "desert", "jungle"],  "beach"),
        'o2o_medium': (["mountain", "beach", "dirt", "jungle"],  ["jungle", "dirt", "beach", "snow"],   "desert"),
        'o2o_hard':   (["jungle", "mountain", "snow", "desert"], ["mountain", "snow", "desert", "jungle"], "beach"),
    }
    ORDER = ["bulldog", "dachshund", "labrador", "corgi"]   # order used by the official combination table

    def __init__(self, root, challenge, split, transform=None):
        import os
        group, test, filler = self.COMBOS[challenge]
        total = 3168; counts = [int(0.97 * total), int(0.87 * total)]
        items = []
        def folder(ind, loc, cls, limit=None):
            d = Path(root) / 'spawrious224' / str(ind) / loc / cls
            fs = sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            if limit: fs = fs[:limit]
            return [(str(d / f), self.CLASSES.index(cls), self.LOCS.index(loc)) for f in fs]
        if split == 'train':
            for k, cls in enumerate(self.ORDER):
                for ind in (0, 1):
                    items += folder(ind, group[k], cls, counts[ind])
            for ind in (0, 1):
                for cls in self.ORDER:
                    items += folder(ind, filler, cls, total - counts[ind])
        else:
            for k, cls in enumerate(self.ORDER):
                for ind in (0, 1):
                    items += folder(ind, test[k], cls)
        self.paths = [i[0] for i in items]; self.targets = [i[1] for i in items]; self.places = [i[2] for i in items]
        self.transform = transform

    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, self.targets[i]


def get_spawrious(challenge, data_root='./data', batch_size=64, num_workers=4, val_frac=0.1):
    """(train, val, test). val = class-stratified 10% of the official train set (fixed RandomState(0))."""
    root = Path(data_root) / 'spawrious'
    if not (root / 'spawrious224').exists():
        raise FileNotFoundError(f"Spawrious 없음: {root}/spawrious224 — 공식 tar를 받아 풀어라 (pod_entry_spawrious.sh)")
    tf, _ = waterbirds_transforms()
    tr = SpawriousDataset(root, challenge, 'train', tf); te = SpawriousDataset(root, challenge, 'test', tf)
    rng = np.random.RandomState(0); tr_idx, va_idx = [], []
    y = np.asarray(tr.targets)
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0]); k = int(round(len(ids) * val_frac))
        va_idx += ids[:k].tolist(); tr_idx += ids[k:].tolist()
    keep = lambda ds, idx: (setattr(ds, 'paths', [ds.paths[i] for i in idx]), setattr(ds, 'targets', [ds.targets[i] for i in idx]),
                            setattr(ds, 'places', [ds.places[i] for i in idx]), ds)[-1]
    import copy
    va = keep(copy.copy(tr), sorted(va_idx)); tr = keep(tr, sorted(tr_idx))
    print(f"[spawrious {challenge}] train {len(tr)} / val {len(va)} / test {len(te)}")
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers, pin_memory=_pin_memory())
    return mk(tr, True), mk(va, False), mk(te, False)


# ---------------------------------------------------------------------------
# Colored-MNIST (LfF protocol, Nam et al. 2020): digit colour correlated with the label.
# ---------------------------------------------------------------------------
# Train: with probability `bias` the digit gets its class colour, else a random other colour.
# Test: colour uniformly random (unbiased). Group = (label, colour): 100 cells; the
# bias-conflicting accuracy (colour != class colour) is computed from the dump.

_CMNIST_COLORS = np.array([[230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],
                           [145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,190]], dtype=np.float32)

class ColoredMNIST(Dataset):
    def __init__(self, data_root, train, bias, seed=0, transform=None):
        base = datasets.MNIST(data_root, train=train, download=True)
        x = base.data.numpy(); y = base.targets.numpy(); rng = np.random.RandomState(seed + (0 if train else 12345))
        if train:
            aligned = rng.rand(len(y)) < bias
            col = np.where(aligned, y, (y + rng.randint(1, 10, len(y))) % 10)
        else:
            col = rng.randint(0, 10, len(y))
        mask = (x.astype(np.float32) / 255.0)[:, None]                       # N,1,28,28
        colours = _CMNIST_COLORS[col] + rng.randn(len(y), 3).astype(np.float32) * 10.0   # per-image jitter
        img = (mask * colours[:, :, None, None]).clip(0, 255).astype(np.uint8)  # N,3,28,28
        self.imgs = img; self.targets = y.tolist(); self.places = col.tolist(); self.transform = transform
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        img = Image.fromarray(np.transpose(self.imgs[i], (1, 2, 0)))
        if self.transform is not None: img = self.transform(img)
        return img, self.targets[i]

def get_cmnist(bias, data_root='./data', batch_size=128, num_workers=4, val_frac=0.1):
    norm = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    tf = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), norm])
    tr = ColoredMNIST(data_root, True, bias, transform=tf); te = ColoredMNIST(data_root, False, bias, transform=tf)
    rng = np.random.RandomState(0); y = np.asarray(tr.targets); tr_idx, va_idx = [], []
    for c in range(10):
        ids = rng.permutation(np.where(y == c)[0]); k = int(round(len(ids) * val_frac)); va_idx += ids[:k].tolist(); tr_idx += ids[k:].tolist()
    import copy
    def keep(ds, idx):
        ds.imgs = ds.imgs[idx]; ds.targets = [ds.targets[i] for i in idx]; ds.places = [ds.places[i] for i in idx]; return ds
    va = keep(copy.copy(tr), sorted(va_idx)); tr = keep(tr, sorted(tr_idx))
    yy = np.asarray(tr.targets); pp = np.asarray(tr.places)
    print(f"[cmnist bias={bias}] train {len(tr)} (aligned {100*(yy==pp).mean():.1f}%) / val {len(va)} / test {len(te)}")
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers, pin_memory=_pin_memory())
    return mk(tr, True), mk(va, False), mk(te, False)


# ---------------------------------------------------------------------------
# BAR — Biased Action Recognition (Nam et al. 2020). data/BAR/{train,test}/<class>/*.jpg
# Train images are place-aligned, test images are place-conflicting; no explicit place label,
# so `places` = class index and the group table reduces to per-class test accuracy.
# ---------------------------------------------------------------------------
class BARDataset(Dataset):
    CLASSES = ['climbing', 'diving', 'fishing', 'racing', 'throwing', 'pole vaulting']
    def __init__(self, root, split, transform=None):
        import os
        d = Path(root) / split; self.classes = self.CLASSES; self.paths = []; self.targets = []
        for f in sorted(os.listdir(d)):                       # flat layout: <class>_<n>.jpg
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            c = f.rsplit('_', 1)[0].lower()                   # 'pole vaulting_12.jpg' -> 'pole vaulting'
            if c in self.CLASSES: self.paths.append(str(d / f)); self.targets.append(self.CLASSES.index(c))
        self.places = list(self.targets); self.transform = transform
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, self.targets[i]

def get_bar(data_root='./data', batch_size=64, num_workers=4, val_frac=0.1):
    root = Path(data_root) / 'BAR'
    if not (root / 'train').exists(): raise FileNotFoundError(f"BAR 없음: {root} — git clone https://github.com/alinlab/BAR")
    tf, _ = waterbirds_transforms()
    tr = BARDataset(root, 'train', tf); te = BARDataset(root, 'test', tf)
    rng = np.random.RandomState(0); y = np.asarray(tr.targets); tr_idx, va_idx = [], []
    for c in range(len(tr.classes)):
        ids = rng.permutation(np.where(y == c)[0]); k = int(round(len(ids) * val_frac)); va_idx += ids[:k].tolist(); tr_idx += ids[k:].tolist()
    import copy
    def keep(ds, idx):
        ds.paths = [ds.paths[i] for i in idx]; ds.targets = [ds.targets[i] for i in idx]; ds.places = [ds.places[i] for i in idx]; return ds
    va = keep(copy.copy(tr), sorted(va_idx)); tr = keep(tr, sorted(tr_idx))
    print(f"[BAR] classes {tr.classes} train {len(tr)} / val {len(va)} / test {len(te)}")
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers, pin_memory=_pin_memory())
    return mk(tr, True), mk(va, False), mk(te, False)


# ---------------------------------------------------------------------------
# MetaShift cat-vs-dog (Liang & Zou 2022), DISC subset (Wu et al. 2023): background shortcut.
# ---------------------------------------------------------------------------
# Train: cat(bed, sofa) = indoor vs dog(bench, bike) = outdoor -> background fully
# correlated with the label (380+231 / 145+367). Test: both classes on 'shelf'
# (235 / 306), a context unseen in training. place: 0 indoor, 1 outdoor, 2 shelf;
# worst-group over the 2 test cells (cat/shelf, dog/shelf) -- dog/shelf is the
# label-conflicting one (indoor-looking background, dog label).
# Layout: data/metashifts/MetaDatasetCatDog/{train,test}/{cat,dog}/<cls>(<ctx>)/*.jpg
# Download: the DISC Google-Drive zip streams non-interactively (pod_entry_metashift.sh).

class MetaShiftDataset(BARDataset):
    CLASSES = ['cat', 'dog']
    PLACES = {'bed': 0, 'sofa': 0, 'bench': 1, 'bike': 1, 'shelf': 2}
    def __init__(self, root, split, transform=None):
        import os
        self.classes = self.CLASSES; self.paths = []; self.targets = []; self.places = []
        for cls in self.CLASSES:
            for ctx in sorted(os.listdir(Path(root) / split / cls)):          # 'cat(bed)'
                d = Path(root) / split / cls / ctx
                if not d.is_dir(): continue
                pl = self.PLACES[ctx[ctx.index('(') + 1:ctx.index(')')]]
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.paths.append(str(d / f)); self.targets.append(self.CLASSES.index(cls)); self.places.append(pl)
        self.transform = transform

def get_metashift(data_root='./data', batch_size=64, num_workers=4, val_frac=0.1):
    """(train, val, test). val = class-stratified 10% of the official train set (fixed RandomState(0), as get_bar)."""
    root = Path(data_root) / 'metashifts' / 'MetaDatasetCatDog'
    if not (root / 'train').exists(): raise FileNotFoundError(f"MetaShift 없음: {root} — pod_entry_metashift.sh 의 zip 을 받아 풀어라")
    tf, _ = waterbirds_transforms()
    tr = MetaShiftDataset(root, 'train', tf); te = MetaShiftDataset(root, 'test', tf)
    rng = np.random.RandomState(0); y = np.asarray(tr.targets); tr_idx, va_idx = [], []
    for c in range(len(tr.classes)):
        ids = rng.permutation(np.where(y == c)[0]); k = int(round(len(ids) * val_frac)); va_idx += ids[:k].tolist(); tr_idx += ids[k:].tolist()
    import copy
    def keep(ds, idx):
        ds.paths = [ds.paths[i] for i in idx]; ds.targets = [ds.targets[i] for i in idx]; ds.places = [ds.places[i] for i in idx]; return ds
    va = keep(copy.copy(tr), sorted(va_idx)); tr = keep(tr, sorted(tr_idx))
    print(f"[metashift] train {len(tr)} / val {len(va)} / test {len(te)}")
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers, pin_memory=_pin_memory())
    return mk(tr, True), mk(va, False), mk(te, False)


# ---------------------------------------------------------------------------
# NICO++ (Zhang et al. CVPR 2023) elephant-vs-sailboat, grass-vs-water: Waterbirds-style build.
# ---------------------------------------------------------------------------
# Source: HF `2018LZY/NICOplusplus` (unofficial 20-class x 6-domain DG subset, NICOpp.zip 1.37 GB).
# Cells used (all images in the zip): elephant/grass 455, elephant/water 402,
# sailboat/grass 332, sailboat/water 434. Construction (RandomState(0), sorted filenames):
#   test  = 150 per cell (600, balanced over the 4 (class, context) cells);
#   train = every remaining majority image (elephant/grass 305, sailboat/water 284) plus
#           round(maj * 5/95) minority images (elephant/water 16, sailboat/grass 15)
#           -> P(context == class-context | class) = 95% as in Waterbirds (620 images);
#   val   = class-stratified 10% of train (62; drawn like train, as get_metashift).
# place: 0 grass, 1 water; label==place is the majority group, so resample_groups works.
# Layout: data/nicopp/NICOpp/{grass,water}/{elephant,sailboat}/*.jpg (zip as-is).
# Generic form: dataset name 'nicopp:<classA>,<classB>:<ctx0>,<ctx1>' (parsed in get_dataset) builds
# the same split for any pair; class A's majority context is ctx0, class B's is ctx1.

class NICOppDataset(BARDataset):
    CLASSES = ['elephant', 'sailboat']       # class c's majority context = CONTEXTS[c]
    CONTEXTS = ['grass', 'water']
    N_TEST, CORR = 150, 0.95
    def __init__(self, root, split, transform=None, classes=None, contexts=None):
        import os
        self.classes = list(classes or self.CLASSES); self.contexts = list(contexts or self.CONTEXTS)
        self.paths = []; self.targets = []; self.places = []
        rng = np.random.RandomState(0)
        for c, cls in enumerate(self.classes):
            pools = {}
            for pl, ctx in enumerate(self.contexts):
                d = Path(root) / ctx / cls
                fs = [str(d / f) for f in sorted(os.listdir(d)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                fs = [fs[i] for i in rng.permutation(len(fs))]
                pools[pl] = (fs[:self.N_TEST], fs[self.N_TEST:])            # (test, rest)
            maj = pools[c][1]; mino = pools[1 - c][1][:int(round(len(maj) * (1 - self.CORR) / self.CORR))]
            sel = {c: pools[c][0], 1 - c: pools[1 - c][0]} if split == 'test' else {c: maj, 1 - c: mino}
            for pl, fs in sel.items():
                self.paths += fs; self.targets += [c] * len(fs); self.places += [pl] * len(fs)
        self.transform = transform

def get_nicopp(data_root='./data', batch_size=64, num_workers=4, val_frac=0.1, classes=None, contexts=None):
    """(train, val, test). val = class-stratified 10% of the constructed train set (fixed RandomState(0), as get_metashift)."""
    root = Path(data_root) / 'nicopp' / 'NICOpp'
    if not (root / 'grass').exists(): raise FileNotFoundError(f"NICO++ 없음: {root} — pod_entry_screen.sh 의 zip 을 받아 풀어라")
    tf, _ = waterbirds_transforms()
    tr = NICOppDataset(root, 'train', tf, classes, contexts); te = NICOppDataset(root, 'test', tf, classes, contexts)
    rng = np.random.RandomState(0); y = np.asarray(tr.targets); tr_idx, va_idx = [], []
    for c in range(len(tr.classes)):
        ids = rng.permutation(np.where(y == c)[0]); k = int(round(len(ids) * val_frac)); va_idx += ids[:k].tolist(); tr_idx += ids[k:].tolist()
    import copy
    def keep(ds, idx):
        ds.paths = [ds.paths[i] for i in idx]; ds.targets = [ds.targets[i] for i in idx]; ds.places = [ds.places[i] for i in idx]; return ds
    va = keep(copy.copy(tr), sorted(va_idx)); tr = keep(tr, sorted(tr_idx))
    g = lambda ds: {(a, b): int(((np.asarray(ds.targets) == a) & (np.asarray(ds.places) == b)).sum()) for a in (0, 1) for b in (0, 1)}
    print(f"[nicopp {tr.classes} x {tr.contexts}] train {len(tr)} {g(tr)} / val {len(va)} {g(va)} / test {len(te)} {g(te)}")
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers, pin_memory=_pin_memory())
    return mk(tr, True), mk(va, False), mk(te, False)


def resample_groups(ds, corr=None, class_balance=False, seed=0):
    """In-place subsample of a dataset with `targets`/`places` lists (Waterbirds, CelebA).

    corr:  keep every minority image (label != place) and subsample the majority group of
           each class to m_c * corr/(1-corr), so P(place==label | class) == corr.
    class_balance: subsample the larger class to the smaller class size (SUBY), uniform
           within class, so the group ratio inside each class is unchanged.
    Same seed -> same subset across methods (RandomState(seed+1), as _apply_subset).
    """
    assert hasattr(ds, 'targets') and hasattr(ds, 'places'), 'dataset needs targets/places'
    rng = np.random.RandomState(seed + 1)
    y = np.asarray(ds.targets); pl = np.asarray(ds.places); keep = np.arange(len(y))
    if corr is not None:
        assert 0.0 < corr < 1.0
        sel = []
        for c in np.unique(y):
            mino = np.where((y == c) & (pl != c))[0]; majo = np.where((y == c) & (pl == c))[0]
            k = min(len(majo), int(round(len(mino) * corr / (1 - corr))))
            sel += [mino, rng.choice(majo, k, replace=False)]
        keep = np.sort(np.concatenate(sel))
    if class_balance:
        yk = y[keep]; n = min((yk == c).sum() for c in np.unique(yk))
        sel = [rng.choice(keep[yk == c], n, replace=False) for c in np.unique(yk)]
        keep = np.sort(np.concatenate(sel))
    keep = keep.tolist()
    ds._bytes = [ds._bytes[i] for i in keep]
    ds.targets = [ds.targets[i] for i in keep]; ds.places = [ds.places[i] for i in keep]
    y = np.asarray(ds.targets); pl = np.asarray(ds.places)
    print(f"[resample] corr={corr} class_balance={class_balance}: {len(keep)} images, "
          f"groups(y,p)=" + str({(int(a), int(b)): int(((y == a) & (pl == b)).sum()) for a in (0, 1) for b in (0, 1)}))


def _apply_subset(loader, subset_frac: float, seed: int,
                  batch_size: int, num_workers: int):
    """훈련 로더를 subset_frac 비율로 줄인다 (CIFAR 이외 데이터셋용).

    같은 seed 는 반드시 같은 부분집합을 내야 한다 — 그래야 방법 간 비교가
    블록 설계가 된다. cifar100 쪽과 같은 RandomState(seed+1) 규약을 쓴다.
    """
    ds = loader.dataset
    n = len(ds)
    k = max(1, int(round(n * subset_frac)))
    idx = np.random.RandomState(seed + 1).permutation(n)[:k]
    sub = torch.utils.data.Subset(ds, np.sort(idx).tolist())
    print(f"[subset] keeping {k} training images ({subset_frac:.0%})")
    return DataLoader(sub, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=_pin_memory())
