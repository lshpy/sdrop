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
from sdrop import collect_drop_rates
from evaluate import evaluate


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='SDrop Training')

    # Dataset
    p.add_argument('--dataset',    type=str, default='cifar100',
                   choices=['cifar100', 'cifar100_lt', 'tinyimagenet', 'cub200'])
    p.add_argument('--imb_ratio',  type=float, default=1.0,
                   help='long-tail imbalance n_max/n_min (100 -> head 500, tail 5); '
                        '1.0 disables imbalance')
    p.add_argument('--subset_frac', type=float, default=1.0,
                   help='keep this fraction of the training set (low-data regime)')
    p.add_argument('--label_noise', type=float, default=0.0,
                   help='symmetric label-noise rate applied to the training split')
    p.add_argument('--data_root',  type=str, default='./data')
    p.add_argument('--num_workers',type=int, default=4)

    # Model
    p.add_argument('--arch',       type=str, default=None,
                   help='Override default arch (resnet18/resnet50)')
    p.add_argument('--pretrained', action='store_true', default=None,
                   help='Override pretrained flag')

    # SDrop configuration
    p.add_argument('--method',     type=str, default='none',
                   choices=['none', 'dropout', 'dropout_std', 'spatialdropout',
                            'sdrop', 'sdrop_energy', 'sdrop_peak', 'sdrop_random',
                            'sgridlc', 'dropblock', 'senet', 'cbam',
                            'vit', 'sdrop_vit', 'sdrop_vit_full'])
    p.add_argument('--drop_rate',  type=float, default=0.1)
    p.add_argument('--layers',     type=str,   nargs='+', default=[],
                   choices=['L3', 'L4'],
                   help='Layers to insert SDrop: e.g. --layers L3 L4')
    p.add_argument('--grid_size',  type=int,   default=2,
                   help='G for SGridLC (grid cells per side)')
    p.add_argument('--peakedness', type=str, default='max', choices=['max', 'entropy'],
                   help="P_c definition: 'max' (paper) or 'entropy' (resolution-invariant)")
    p.add_argument('--norm',       type=str, default='max',
                   choices=['max', 'mean', 'rank'],
                   help="drop-probability normalisation: 'max' (paper), 'mean', "
                        "or 'rank' (effective rate equals nominal for any beta)")
    p.add_argument('--mix',        type=float, default=None,
                   help="combine the EGPG factors by rank instead of multiplying: "
                        "s_c = (1-mix)*rank(E_c) + mix*rank(1-P_c). 0 is energy-only, "
                        "1 peakedness-only. Unset keeps the product")
    p.add_argument('--beta',       type=float, default=1.0,
                   help="selection strength for --norm rank. 0 reproduces uniform "
                        "channel dropout; higher concentrates drops on top-scoring "
                        "channels. Max drop probability is (beta+1)*drop_rate")
    p.add_argument('--self_gamma', action='store_true',
                   help='use self-normalising gamma = 1/median(Sigma_c)')
    p.add_argument('--amp', action='store_true',
                   help='mixed-precision training (bf16/fp16 autocast). Roughly '
                        '1.5-2x faster on tensor-core GPUs (sm_70+); ignored on CPU')
    p.add_argument('--grad_mode',  type=str, default='off',
                   choices=['off', 'suppress', 'amplify'],
                   help="gradient-guided scoring: 'suppress' drops loud-but-"
                        "task-irrelevant units, 'amplify' drops loss-dominant ones")

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
    # ViT-friendly training recipe extras
    p.add_argument('--strong_aug', action='store_true',
                   help='Use RandAugment + RandomErasing (recommended for ViT).')
    p.add_argument('--mixup_alpha', type=float, default=0.0,
                   help='Mixup alpha; 0 disables mixup.')
    p.add_argument('--warmup_epochs', type=int, default=0,
                   help='Linear LR warmup over this many epochs (ViT recommended).')
    p.add_argument('--label_smoothing', type=float, default=0.0,
                   help='Label smoothing (0.1 standard for ViT).')

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
    extra = ''
    if args.imb_ratio   > 1.0:  extra += f'_imb{args.imb_ratio:g}'
    if args.subset_frac < 1.0:  extra += f'_sub{args.subset_frac:g}'
    if args.label_noise > 0.0:  extra += f'_noise{args.label_noise:g}'
    if args.grad_mode  != 'off': extra += f'_grad{args.grad_mode}'
    if args.peakedness != 'max': extra += f'_pk{args.peakedness}'
    if args.norm       != 'max': extra += f'_nm{args.norm}'
    if args.norm == 'rank':      extra += f'_b{args.beta:g}'
    if args.mix is not None:     extra += f'_mix{args.mix:g}'
    if args.self_gamma:          extra += '_sg'
    return (f"{args.dataset}_{args.method}_rate{args.drop_rate}"
            f"_{layers_str}{grid_str}{extra}_seed{args.seed}")


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def _mixup_batch(x, y, alpha: float, num_classes: int):
    """Apply mixup with mixing coefficient lam ~ Beta(alpha, alpha)."""
    if alpha <= 0:
        return x, y, None
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    return x_mixed, y, (y[perm], lam)


def train_epoch(model, loader, criterion, optimizer, device, log_interval,
                mixup_alpha: float = 0.0, num_classes: int = 100,
                scaler=None, amp_dtype=None):
    """One training epoch.

    scaler/amp_dtype are set when --amp is on. Autocast keeps the forward pass
    and loss in reduced precision while master weights stay fp32; on tensor-core
    GPUs (sm_70+) this is typically 1.5-2x faster at unchanged accuracy. The
    SDrop scores are computed inside autocast too, which is harmless because the
    mask depends only on the *ranking* of channel scores, not their magnitude.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None or amp_dtype is not None

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            if mixup_alpha > 0:
                data, target, mix_extra = _mixup_batch(data, target, mixup_alpha, num_classes)
                output = model(data)
                if mix_extra is None:
                    loss = criterion(output, target)
                else:
                    target_b, lam = mix_extra
                    loss = lam * criterion(output, target) + (1 - lam) * criterion(output, target_b)
            else:
                output = model(data)
                loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
        get_dataset(args.dataset, args.data_root, args.batch_size, args.num_workers,
                    strong_aug=args.strong_aug, imb_ratio=args.imb_ratio,
                    subset_frac=args.subset_frac, label_noise=args.label_noise,
                    seed=args.seed)

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
            peakedness=args.peakedness,
            norm=args.norm,
            beta=args.beta,
            mix=args.mix,
            gamma=(None if args.self_gamma else 1.0),
            grad_mode=args.grad_mode,
            pretrained=pretrained,
        ).to(device)

    print(f"Model: {arch}  |  SDrop: {args.method}  |  "
          f"Rate: {args.drop_rate}  |  Layers: {args.layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- optimizer + scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.method.startswith("vit") or args.method.startswith("sdrop_vit"):
        # ViT works much better with AdamW + warmup
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.05, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0,
            total_iters=args.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- mixed precision ----
    scaler, amp_dtype = None, None
    if args.amp:
        if device.type == 'cuda':
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:                       # Ampere and newer: bf16, no scaler needed
                amp_dtype = torch.bfloat16
                print(f"AMP: bfloat16 autocast (sm_{cap[0]}{cap[1]})")
            elif cap[0] == 7:                     # Volta/Turing: fp16 with loss scaling
                amp_dtype = torch.float16
                scaler = torch.amp.GradScaler('cuda')
                print(f"AMP: float16 autocast + GradScaler (sm_{cap[0]}{cap[1]})")
            else:
                print(f"AMP requested but sm_{cap[0]}{cap[1]} has no tensor cores; "
                      "running in fp32")
        else:
            print('AMP requested but device is not CUDA; running in fp32')

    # ---- checkpoint dir ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.log_interval,
            mixup_alpha=args.mixup_alpha, num_classes=num_classes,
            scaler=scaler, amp_dtype=amp_dtype)
        scheduler.step()

        # evaluate every epoch
        metrics = evaluate(model, val_loader, device=str(device))
        val_acc = metrics['acc']

        # realised (effective) drop fraction — p_base is only an upper bound
        # under 'max' normalisation, so we log what actually happened.
        drop_rates = collect_drop_rates(model)
        eff_drop = (sum(drop_rates.values()) / len(drop_rates)) if drop_rates else float('nan')

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_acc={val_acc:.2f}%  "
              f"f1_mac={metrics['f1_macro']:.4f}  "
              f"auc={metrics['auc']:.4f}  "
              f"ece={metrics['ece']:.4f}  "
              + (f"p_eff={eff_drop:.4f}  " if eff_drop == eff_drop else "")
              + f"({elapsed:.1f}s)")

        history.append({'epoch': epoch, **metrics,
                        'train_loss': train_loss, 'train_acc': train_acc,
                        'p_eff': eff_drop})

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
