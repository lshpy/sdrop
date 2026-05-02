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
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Dataset
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
      1. If cs.toronto.edu (the canonical host) is up, use it.
      2. Otherwise download the HuggingFace `uoft-cs/cifar100` parquet
         shards and convert them into torchvision's pickled format.
    """
    import os, urllib.request, tarfile
    extracted = os.path.join(data_root, 'cifar-100-python')
    if os.path.isdir(extracted) and os.path.isfile(os.path.join(extracted, 'meta')):
        return
    os.makedirs(data_root, exist_ok=True)

    # ---- attempt 1: original tar.gz mirror ----
    target = os.path.join(data_root, 'cifar-100-python.tar.gz')
    try:
        print('  CIFAR-100: trying cs.toronto.edu tar.gz')
        urllib.request.urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', target)
        with tarfile.open(target) as t:
            t.extractall(data_root)
        print('  CIFAR-100 ready at', extracted)
        return
    except Exception as e:
        print('  toronto mirror failed:', e)

    # ---- attempt 2: HuggingFace parquet → repickle into torchvision format ----
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

    print(f"[INFO] TinyImageNet val reformatted: {val_dir}")


def get_tinyimagenet(data_root: str = './data', batch_size: int = 128,
                     num_workers: int = 4):
    root = Path(data_root) / 'tiny-imagenet-200'
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
# Unified getter
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    'cifar100': {
        'num_classes': 100,
        'arch':        'resnet18',
        'pretrained':  False,
        'getter':      get_cifar100,
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
}


def get_dataset(name: str, data_root: str = './data', batch_size: int = 128,
                num_workers: int = 4, strong_aug: bool = False):
    """
    Returns (train_loader, val_loader, num_classes, default_arch, pretrained).
    """
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from: {list(DATASET_CONFIG.keys())}")
    cfg = DATASET_CONFIG[name]
    getter = cfg['getter']
    if name == 'cifar100':
        train_loader, val_loader = getter(data_root, batch_size, num_workers,
                                          strong_aug=strong_aug)
    else:
        train_loader, val_loader = getter(data_root, batch_size, num_workers)
    return (train_loader, val_loader,
            cfg['num_classes'], cfg['arch'], cfg['pretrained'])
