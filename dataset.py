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

def cifar100_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
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

def get_cifar100(data_root: str = './data', batch_size: int = 128,
                 num_workers: int = 4):
    train_tf, val_tf = cifar100_transforms()
    train_ds = datasets.CIFAR100(data_root, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR100(data_root, train=False, download=True, transform=val_tf)
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
                num_workers: int = 4):
    """
    Returns (train_loader, val_loader, num_classes, default_arch, pretrained).
    """
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from: {list(DATASET_CONFIG.keys())}")
    cfg = DATASET_CONFIG[name]
    train_loader, val_loader = cfg['getter'](data_root, batch_size, num_workers)
    return (train_loader, val_loader,
            cfg['num_classes'], cfg['arch'], cfg['pretrained'])
