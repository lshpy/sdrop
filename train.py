"""
Training script for SDrop experiments.

Usage examples
--------------
# Replicate best CIFAR-100 result (SDrop_Energy, Rate=0.1, L4)
python train.py --dataset cifar100 --method sdrop_energy --drop_rate 0.1 --layers L4

# Replicate best TinyImageNet result (SGridLC, Rate=0.3, L3, G=4)
python train.py --dataset tinyimagenet --method sgridlc --drop_rate 0.3 --layers L3 --grid_size 4

# Baseline (no dropout)
python train.py --dataset cifar100 --method none

# CUB-200-2011 with pretrained backbone
python train.py --dataset cub200 --method sdrop_energy --drop_rate 0.1 --layers L4 --epochs 60 --lr 0.01

# Run 3 seeds and get mean±std (as in paper)
for seed in 0 1 2; do
    python train.py --dataset cifar100 --method sdrop_energy --drop_rate 0.1 --layers L4 --seed $seed
done
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataset
from model import build_model
from evaluate import evaluate


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='SDrop Training')

    # Dataset
    p.add_argument('--dataset',    type=str, default='cifar100',
                   choices=['cifar100', 'tinyimagenet', 'cub200'])
    p.add_argument('--data_root',  type=str, default='./data')
    p.add_argument('--num_workers',type=int, default=4)

    # Model
    p.add_argument('--arch',       type=str, default=None,
                   help='Override default arch (resnet18/resnet50)')
    p.add_argument('--pretrained', action='store_true', default=None,
                   help='Override pretrained flag')

    # SDrop configuration
    p.add_argument('--method',     type=str, default='none',
                   choices=['none', 'dropout', 'sdrop', 'sdrop_energy', 'sgridlc'])
    p.add_argument('--drop_rate',  type=float, default=0.1)
    p.add_argument('--layers',     type=str,   nargs='+', default=[],
                   choices=['L3', 'L4'],
                   help='Layers to insert SDrop: e.g. --layers L3 L4')
    p.add_argument('--grid_size',  type=int,   default=2,
                   help='G for SGridLC (grid cells per side)')

    # Training hyperparameters
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--lr',         type=float, default=0.1)
    p.add_argument('--momentum',   type=float, default=0.9)
    p.add_argument('--weight_decay',type=float,default=5e-4)

    # Misc
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--device',     type=str,   default='cuda')
    p.add_argument('--save_dir',   type=str,   default='./checkpoints')
    p.add_argument('--log_interval',type=int,  default=50)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_id(args) -> str:
    """Human-readable run identifier used for checkpoint naming."""
    layers_str = '+'.join(args.layers) if args.layers else 'none'
    grid_str   = f'_G{args.grid_size}' if args.method == 'sgridlc' else ''
    return (f"{args.dataset}_{args.method}_rate{args.drop_rate}"
            f"_{layers_str}{grid_str}_seed{args.seed}")


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, log_interval):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={100.*correct/total:.2f}%")

    return total_loss / total, 100. * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Run: {run_id(args)}")

    # ---- dataset ----
    train_loader, val_loader, num_classes, default_arch, default_pretrained = \
        get_dataset(args.dataset, args.data_root, args.batch_size, args.num_workers)

    arch       = args.arch       if args.arch       is not None else default_arch
    pretrained = args.pretrained if args.pretrained is not None else default_pretrained

    # ---- model ----
    if args.method.startswith("vit") or args.method.startswith("sdrop_vit"):
        # ViT branch (head-level SDrop, see sdrop_vit.py)
        from sdrop_vit import build_sdrop_vit
        # CIFAR-100/TinyImageNet/CUB are all 3-channel; pick image size by dataset
        img_size = {"cifar100": 32, "tinyimagenet": 64,
                    "cub200": 224, "imagenet100": 224}.get(args.dataset, 32)
        model = build_sdrop_vit(
            num_classes=num_classes, img_size=img_size,
            method=args.method, drop_rate=args.drop_rate,
            layers=args.layers,
        ).to(device)
        arch = f"vit-tiny-p4-d12-h3"
    else:
        model = build_model(
            arch=arch,
            num_classes=num_classes,
            method=args.method,
            drop_rate=args.drop_rate,
            layers=args.layers,
            grid_size=args.grid_size,
            pretrained=pretrained,
        ).to(device)

    print(f"Model: {arch}  |  SDrop: {args.method}  |  "
          f"Rate: {args.drop_rate}  |  Layers: {args.layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- optimizer + scheduler ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- checkpoint dir ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.log_interval)
        scheduler.step()

        # evaluate every epoch
        metrics = evaluate(model, val_loader, device=str(device))
        val_acc = metrics['acc']

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_acc={val_acc:.2f}%  "
              f"f1_mac={metrics['f1_macro']:.4f}  "
              f"auc={metrics['auc']:.4f}  "
              f"ece={metrics['ece']:.4f}  "
              f"({elapsed:.1f}s)")

        history.append({'epoch': epoch, **metrics,
                        'train_loss': train_loss, 'train_acc': train_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = save_dir / f"{run_id(args)}_best.pth"
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'best_acc': best_acc, 'args': vars(args)},
                       ckpt_path)
            print(f"  --> New best: {best_acc:.2f}%  saved to {ckpt_path}")

    print(f"\nFinal best val accuracy: {best_acc:.2f}%")

    # save history as CSV
    import csv
    csv_path = save_dir / f"{run_id(args)}_history.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved: {csv_path}")


if __name__ == '__main__':
    main()
