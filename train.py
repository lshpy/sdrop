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

class _DatasetChoices(list):
    """argparse choices that also accept the generic NICO++ form 'nicopp:<A>,<B>:<ctx0>,<ctx1>'."""
    def __contains__(self, x): return list.__contains__(self, x) or str(x).startswith('nicopp:')


def parse_args():
    p = argparse.ArgumentParser(description='SDrop Training')

    # Dataset
    p.add_argument('--dataset',    type=str, default='cifar100',
                   choices=_DatasetChoices(['cifar100', 'cifar100_lt', 'cifar10', 'svhn',
                            'mnist', 'tinyimagenet', 'cub200', 'waterbirds', 'celeba',
                            'spawrious_o2o_easy', 'spawrious_o2o_medium', 'spawrious_o2o_hard',
                            'cmnist_0.95', 'cmnist_0.98', 'cmnist_0.995', 'bar', 'metashift', 'nicopp']))
    p.add_argument('--imb_ratio',  type=float, default=1.0,
                   help='long-tail imbalance n_max/n_min (100 -> head 500, tail 5); '
                        '1.0 disables imbalance')
    p.add_argument('--val_frac',    type=float, default=0.0,
                   help='fraction of the training set held out as a stratified validation split (0 = legacy convention: select on test)')
    p.add_argument('--split_seed',  type=int,   default=None,
                   help='seed of the val split (default: same as --seed); keeps the split independent of the model seed')
    p.add_argument('--subset_frac', type=float, default=1.0,
                   help='keep this fraction of the training set (low-data regime)')
    p.add_argument('--label_noise', type=float, default=0.0,
                   help='symmetric label-noise rate applied to the training split')
    p.add_argument('--data_root',  type=str, default='./data')
    p.add_argument('--num_workers',type=int, default=4)

    # Model
    p.add_argument('--width_mult', type=float, default=1.0,
                   help='Scale ResNet-18 stage widths (64/128/256/512 x this). '
                        'Narrowing makes channels scarce, which is the condition '
                        'under which the monopolisation hypothesis predicts a '
                        'larger SDrop effect.')
    p.add_argument('--arch',       type=str, default=None,
                   help='resnet18 | resnet50 | resnet56 | resnet110 | wrn28_10 '
                        '(56/110/wrn are CIFAR-only 3-stage nets, insertion points L2-L4)')
    p.add_argument('--pretrained', action='store_true', default=None,
                   help='Override pretrained flag')
    p.add_argument('--no_pretrained', action='store_true',
                   help='Train the backbone from scratch (random init); run id gets a _scratch tag')

    # SDrop configuration
    p.add_argument('--method',     type=str, default='none',
                   choices=['none', 'dropout', 'dropout_std', 'spatialdropout',
                            'sdrop', 'sdrop_energy', 'sdrop_peak', 'sdrop_random',
                            'sdrop_class',
                            'sgridlc', 'dropblock', 'senet', 'cbam',
                            'wcd', 'focuseddrop', 'sdrop_gap',
                            'sdrop_shuffle', 'sdrop_reverse', 'sdrop_cross',
                            'guided_topk', 'guided_dr',
                            'vit', 'sdrop_vit', 'sdrop_vit_full',
                            'vitb_none', 'vitb_sdrop', 'vitb_sdrop_energy', 'vitb_sdrop_gap',
                            'vitb_sdrop_random', 'vitb_sdrop_shuffle',
                            'vitbm_sdrop', 'vitbm_sdrop_energy', 'vitbm_sdrop_gap', 'vitbm_sdrop_random', 'vitbm_sdrop_shuffle',
                            'vitbt_sdrop_energy', 'vitbt_sdrop_gap', 'vitbt_sdrop_random', 'vitbt_sdrop_shuffle'])
    p.add_argument('--drop_rate',  type=float, default=0.1)
    p.add_argument('--layers',     type=str,   nargs='+', default=[],
                   choices=['L1', 'L2', 'L3', 'L4'],
                   help='Layers to insert SDrop: e.g. --layers L3 L4')
    p.add_argument('--grid_size',  type=int,   default=2,
                   help='G for SGridLC (grid cells per side)')
    p.add_argument('--peakedness', type=str, default='max', choices=['max', 'entropy'],
                   help="P_c definition: 'max' (paper) or 'entropy' (resolution-invariant)")
    p.add_argument('--norm',       type=str, default='mean', choices=['max', 'mean'],
                   help="drop-probability normalisation. 'mean' (default) "
                        "realises the nominal drop rate exactly. 'max' "
                        "applies the intervention at only mean(s)/max(s) of the "
                        "nominal strength.")
    p.add_argument('--rescale',    action='store_true',
                   help='inverted-dropout rescaling (divide by the per-channel keep probability). '
                        'Default off = original behaviour. All baselines rescale, so this is '
                        'an ablation axis for a fair comparison.')
    p.add_argument('--select',     type=str, default='bernoulli',
                   choices=['bernoulli', 'topk'],
                   help="channel selection. bernoulli = tilt probabilities by score (default), "
                        "topk = deterministically drop the k highest-scoring channels "
                        "(top-k of Guided Dropout, AAAI'19)")
    p.add_argument('--warmup_frac', type=float, default=0.0,
                   help='no dropping during this initial fraction of training '
                        "(Guided Dropout uses 40/200 = 0.2)")
    p.add_argument('--rescale_mode', type=str, default='prob',
                   choices=['prob', 'count'],
                   help="rescaling mode. prob = divide by per-channel keep probability (exact for p<=0.3), "
                        "count = correct by the number of channels actually kept (WCD style, no clamp needed)")
    p.add_argument('--schedule',   type=str, default='const',
                   choices=['const', 'linear', 'early', 'gd', 'gdtopk'],
                   help="drop-rate schedule. const = fixed, "
                        "linear = ramp from 0 to the target (DropBlock, NeurIPS'18), "
                        "gd = no drop for the first 20%%, then p -> 0.75p -> 0.5p "
                        "(Guided Dropout, AAAI'19; with p_base=0.2: 0.2/0.15/0.1)")
    p.add_argument('--participation', type=float, default=1.0,
                   help='apply the drop only to this fraction of batches (FocusedDropout 2021 uses 0.1)')
    p.add_argument('--vit_unit',   type=str, default='head',
                   choices=['head', 'key', 'token'],
                   help='ViT drop unit. head = attention head (3 candidates), '
                        'key = key masking before softmax (65 candidates, DropKey-style), '
                        'token = rows of the attention output')
    p.add_argument('--vit_score',  type=str, default='pre', choices=['pre', 'post'],
                   help="ViT head scoring source: pre-softmax logits (default) or "
                        "post-softmax attention (ablation for Sec. 3.6).")
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
    p.add_argument('--sd_lambda', type=float, default=0.0,
                   help='Spectral Decoupling (Pezeshki et al. 2021): add sd_lambda/2 * ||logits||^2 to the loss (label-free baseline)')
    p.add_argument('--train_corr', type=float, default=None,
                   help='waterbirds: subsample majority (label==place) groups so that P(place==label)=train_corr; keeps all minority images')
    p.add_argument('--run_tag', type=str, default='',
                   help='free-form suffix for the run id (e.g. wd1e-3), so hyperparameter sweeps do not overwrite each other')
    p.add_argument('--class_weight', action='store_true',
                   help='RWY: cross-entropy weighted by inverse class frequency of the training set (label-only, no group labels, keeps every image)')
    p.add_argument('--class_balance', action='store_true',
                   help='subsample the larger class to the smaller class size (Idrissi et al. 2022 SUBY; label-only, no group labels)')
    p.add_argument('--jtt_stage1_epochs', type=int, default=0,
                   help='JTT (Liu et al. 2021): train --method none for T epochs, collect the training '
                        'error set, re-initialise and retrain --epochs with the error set upweighted. 0 = off')
    p.add_argument('--jtt_lambda', type=float, default=50,
                   help='JTT upweight: each error-set example appears this many times per stage-2 epoch')
    p.add_argument('--dump_feats', action='store_true',
                   help='after training, save val/test logits, penultimate features and group ids to <run_id>_dump.npz')
    p.add_argument('--label_smoothing', type=float, default=0.0,
                   help='Label smoothing (0.1 standard for ViT).')
    p.add_argument('--log_grad_channels', action='store_true',
                   help='per-channel per-epoch sum of grad^2 and mean energy of the tensor entering sdrop_l4 '
                        '(= layer4 output) -> <run_id>_gradchan.csv (epoch,channel,grad_sq_sum,energy_mean,n)')
    p.add_argument('--log_grad_clean', type=int, default=0,
                   help='K>0: after every epoch, run K fixed training batches (drawn once with the run seed) through '
                        'the model in eval() mode (SDrop = identity, BN in eval) with grad enabled and no optimizer '
                        'step, and log the same per-channel stats as --log_grad_channels -> <run_id>_gradclean.csv. '
                        'Removes the mask confound: under SDrop a dropped channel gets zero gradient on that batch.')
    p.add_argument('--dump_spatial', type=int, default=0,
                   help='with test split: per-image per-L4-channel energy and peakedness of the first N test images '
                        '-> <run_id>_spatial.npz (energy [N,C], peak [N,C], y, places); run id gets a _sp tag')

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


# Methods whose drop probability passes through drop_mask_from_scores() and
# therefore depend on --norm. Everything else (none, dropout, dropout_std,
# plain vit) ignores it, so tagging those would only add noise to filenames.
_SCORE_METHODS = {'sdrop', 'sdrop_energy', 'sdrop_peak', 'sdrop_random',
                  'sdrop_class', 'sdrop_gap', 'sdrop_shuffle', 'sdrop_reverse', 'sdrop_cross',
                  'sgridlc', 'sdrop_vit', 'sdrop_vit_full',
                  'vitb_sdrop', 'vitb_sdrop_energy', 'vitb_sdrop_gap', 'vitb_sdrop_random', 'vitb_sdrop_shuffle',
                  'vitbm_sdrop', 'vitbm_sdrop_energy', 'vitbm_sdrop_gap', 'vitbm_sdrop_random', 'vitbm_sdrop_shuffle',
                  'vitbt_sdrop_energy', 'vitbt_sdrop_gap', 'vitbt_sdrop_random', 'vitbt_sdrop_shuffle'}


def _set_sdrop_targets(model, target):
    """Hand this batch's labels to every class-aware SDrop module."""
    for m in model.modules():
        if hasattr(m, 'set_targets'):
            m.set_targets(target)


# ---------------------------------------------------------------------------
# Overwrite protection
#
# run_id does not encode the full training budget or all optimiser hyperparameters.
# A short diagnostic run once overwrote and destroyed a full-length baseline with the same name.
# If existing outputs have a different configuration, refuse to run.
# ---------------------------------------------------------------------------

# Arguments that are not encoded in the name but change the result
_SEMANTIC_ARGS = ('epochs', 'lr', 'batch_size', 'momentum', 'weight_decay',
                  'strong_aug', 'mixup_alpha', 'warmup_epochs', 'label_smoothing',
                  'arch', 'pretrained', 'grid_size', 'amp')


def _config_fingerprint(args) -> dict:
    return {k: getattr(args, k, None) for k in _SEMANTIC_ARGS}


def guard_overwrite(args, rid: str, save_dir: str):
    """Abort if outputs for the same rid already exist with a different configuration."""
    import json
    hist = os.path.join(save_dir, f'{rid}_history.csv')
    meta = os.path.join(save_dir, f'{rid}_config.json')
    if not os.path.exists(hist):
        return
    cur = _config_fingerprint(args)
    if os.path.exists(meta):
        try:
            old = json.load(open(meta))
        except Exception:
            old = None
        if old is not None:
            # The config of older outputs may contain only some keys (back-filled).
            # Missing keys are "unknown" and not compared; otherwise every legitimate
            # re-run would be blocked.
            diff = {k: (old.get(k), cur.get(k)) for k in cur
                    if k in old and old.get(k) != cur.get(k)}
            if not diff:
                return                      # identical settings -> re-run allowed
            raise SystemExit(
                f"[abort] outputs for '{rid}' already exist with a different configuration.\n"
                + "\n".join(f"    {k}: existing={a!r}  requested={b!r}" for k, (a, b) in diff.items())
                + f"\n  Overwriting would destroy the existing results. Add a distinguishing tag to run_id or "
                  f"move the existing files first.\n  (existing: {hist})")
    else:
        # Older outputs have no config; at least compare the number of epochs.
        try:
            n = sum(1 for _ in open(hist)) - 1
        except Exception:
            return
        if n != args.epochs:
            raise SystemExit(
                f"[abort] '{rid}' already has results for {n} epochs, but you are running "
                f"{args.epochs} epochs. Overwriting would destroy the existing results.\n  ({hist})")


def save_config(args, rid: str, save_dir: str):
    import json
    with open(os.path.join(save_dir, f'{rid}_config.json'), 'w') as f:
        json.dump(_config_fingerprint(args), f, indent=2, sort_keys=True, default=str)


def run_id(args) -> str:
    """Human-readable run identifier used for checkpoint naming."""
    layers_str = '+'.join(args.layers) if args.layers else 'none'
    grid_str   = f'_G{args.grid_size}' if args.method == 'sgridlc' else ''
    extra = ''
    # Without an arch tag, resnet56 results would overwrite resnet18 results.
    # All older untagged files are resnet18, so the tag is omitted only for the default.
    # args.arch is an override and may be None. The arch actually used is
    # resolved together with the dataset default (the arch variable in train.py), so the
    # effective arch must be computed with the same rule here. Otherwise
    #   --dataset cub200            (actually resnet50) -> no tag
    #   --dataset cub200 --arch resnet18                -> no tag   <- overwrite
    # i.e. two different networks would get the same file name.
    _default_arch = {'cifar100': 'resnet18', 'cifar100_lt': 'resnet18',
                     'cifar10': 'resnet18', 'svhn': 'resnet18',
                     'mnist': 'resnet18',
                     'tinyimagenet': 'resnet50', 'cub200': 'resnet50',
                     'waterbirds': 'resnet50', 'celeba': 'resnet50',
                     'spawrious_o2o_easy': 'resnet50', 'spawrious_o2o_medium': 'resnet50', 'spawrious_o2o_hard': 'resnet50',
                     'cmnist_0.95': 'resnet18', 'cmnist_0.98': 'resnet18', 'cmnist_0.995': 'resnet18', 'bar': 'resnet50', 'metashift': 'resnet50', 'nicopp': 'resnet50',
                     'imagenet100': 'resnet50'}.get(args.dataset.split(':')[0], 'resnet18')
    _arch = getattr(args, 'arch', None) or _default_arch
    if _arch != _default_arch or _arch != 'resnet18':
        if _arch != _default_arch:
            extra += f'_{_arch}'
    # If the epoch count is not in the file name, a short diagnostic run overwrites a
    # full-length run with the same settings (this happened once). Older files are all
    # 200/100 epochs, so the tag is omitted only for 200
    # to stay compatible with the older names.
    if args.epochs != 200:      extra += f'_ep{args.epochs}'
    if getattr(args, 'no_pretrained', False): extra += '_scratch'
    if args.imb_ratio   > 1.0:  extra += f'_imb{args.imb_ratio:g}'
    if args.subset_frac < 1.0:  extra += f'_sub{args.subset_frac:g}'
    if getattr(args, 'val_frac', 0.0) > 0.0: extra += f'_val{args.val_frac:g}'
    if getattr(args, 'split_seed', None) is not None and args.split_seed != args.seed:
        extra += f'_ss{args.split_seed}'
    if args.label_noise > 0.0:  extra += f'_noise{args.label_noise:g}'
    if args.grad_mode  != 'off': extra += f'_grad{args.grad_mode}'
    if args.peakedness != 'max': extra += f'_pk{args.peakedness}'
    # Always tag score-based methods: the default flipped to 'mean', so an
    # untagged legacy filename must keep meaning the old 'max' run.
    if args.method in _SCORE_METHODS: extra += f'_nm{args.norm}'
    if args.self_gamma:          extra += '_sg'
    if getattr(args,'sd_lambda',0.0) > 0: extra += f'_sd{args.sd_lambda:g}'
    if getattr(args,'train_corr',None) is not None: extra += f'_corr{args.train_corr:g}'
    if getattr(args,'class_balance',False): extra += '_cb'
    if getattr(args,'class_weight',False): extra += '_rwy'
    if getattr(args,'run_tag',''): extra += f'_{args.run_tag}'
    if getattr(args,'dump_spatial',0) > 0: extra += '_sp'
    if getattr(args,'jtt_stage1_epochs',0) > 0: extra += f'_jttT{args.jtt_stage1_epochs}L{args.jtt_lambda:g}'
    if args.schedule != 'const': extra += f'_sch{args.schedule}'
    if args.rescale:             extra += '_rs' + ('c' if args.rescale_mode == 'count' else '')
    if args.select != 'bernoulli': extra += f'_sel{args.select}'
    if args.warmup_frac > 0.0:   extra += f'_wu{args.warmup_frac:g}'
    if args.participation < 1.0: extra += f'_pt{args.participation:g}'
    if args.width_mult != 1.0:   extra += f'_w{args.width_mult:g}'
    if args.vit_score == 'post': extra += '_scorepost'
    if args.vit_unit  != 'head': extra += f'_u{args.vit_unit}'
    # 'nicopp:lion,kangaroo:grass,water' -> 'nicopp_lion-kangaroo_grass-water' (filename-safe, pair-unique)
    ds = args.dataset.replace(',', '-').replace(':', '_')
    return (f"{ds}_{args.method}_rate{args.drop_rate}"
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
        # Class-aware SDrop needs the labels to update its class-channel usage
        # table. Pass the *pre-mixup* targets: mixed labels would smear the
        # per-class statistics the score depends on.
        _set_sdrop_targets(model, target)
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


@torch.no_grad()
def jtt_error_set(model, ds, batch_size, num_workers, device):
    """JTT stage 1: indices (into ds) that the model misclassifies in eval mode."""
    from torch.utils.data import DataLoader
    model.eval()
    wrong, ys = [], []
    for data, target in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        wrong.append(model(data.to(device)).argmax(1).cpu().ne(target)); ys.append(target)
    wrong, ys = torch.cat(wrong), torch.cat(ys)
    err = wrong.nonzero().flatten().tolist()
    print(f"[jtt] error set: {len(err)}/{len(ds)} ({100.*len(err)/len(ds):.1f}%)")
    try:
        from evaluate_groups import _places
        pl = torch.tensor(_places(ds))
        print("[jtt] error-set groups (y,p): " + str({(a, b): int((wrong & (ys == a) & (pl == b)).sum())
                                                      for a in ys.unique().tolist() for b in pl.unique().tolist()}))
    except AttributeError:
        pass
    return err


class L4GradLogger:
    """--log_grad_channels: forward pre-hook on model.sdrop_l4 (its input = layer4 output, the SDrop input).
    Accumulates on GPU per epoch: sum_{b,h,w} grad^2 per channel, sum_b mean_{hw} x^2 per channel, n samples.
    clean=True (--log_grad_clean): records only while self.armed, in eval mode (see clean_grad_pass)."""
    def __init__(self, model, scaler=None, clean=False):
        self.scaler = scaler; self.clean = clean; self.armed = False; self.reset()
        self.h = model.sdrop_l4.register_forward_pre_hook(self._pre)

    def reset(self):
        self.grad_sq = self.energy = None; self.n = 0

    def _pre(self, m, inp):
        x = inp[0]
        if not (self.armed if self.clean else m.training):
            return
        e = (x.detach().float() ** 2).mean((2, 3)).sum(0)
        self.energy = e if self.energy is None else self.energy + e
        self.n += x.shape[0]
        if x.requires_grad:
            x.register_hook(self._grad)

    def _grad(self, g):
        g2 = (g.detach().float() ** 2).sum((0, 2, 3))
        if self.scaler is not None:               # fp16 GradScaler: undo the loss scale (bf16 has none)
            g2 = g2 / self.scaler.get_scale() ** 2
        self.grad_sq = g2 if self.grad_sq is None else self.grad_sq + g2

    def write(self, path, epoch):
        import csv
        gs = self.grad_sq.cpu().tolist(); em = (self.energy / max(self.n, 1)).cpu().tolist()
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if epoch == 1: w.writerow(['epoch', 'channel', 'grad_sq_sum', 'energy_mean', 'n'])
            w.writerows((epoch, c, gs[c], em[c], self.n) for c in range(len(gs)))
        self.reset()


def clean_grad_batches(loader, k, seed):
    """First k batches of the (shuffled) training loader, drawn once with a fixed seed, kept on CPU.
    The global torch RNG is restored so the training trajectory is identical with or without the flag."""
    import itertools
    st = torch.get_rng_state(); torch.manual_seed(seed)
    out = [(x.clone(), y.clone()) for x, y in itertools.islice(loader, k)]
    torch.set_rng_state(st)
    return out


def clean_grad_pass(model, batches, criterion, device, amp_dtype, logger):
    """Forward + backward on the fixed batches in eval() mode: every SDrop is the identity and BN uses running
    stats, so the per-channel gradient reflects the weights, not the mask. No optimizer step, no GradScaler
    (fp16 autocast is replaced by fp32 here since unscaled fp16 grads can underflow)."""
    model.eval(); logger.armed = True
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is torch.bfloat16):
            loss = criterion(model(x), y)
        loss.backward()
    model.zero_grad(set_to_none=True); logger.armed = False; model.train()


@torch.no_grad()
def dump_spatial(model, loader, n, device):
    """--dump_spatial: first n images of loader -> per-image per-channel energy (mean x^2 over HW) and
    peakedness max|x|/(sum|x|+eps) of the sdrop_l4 input. Full maps are not kept."""
    from evaluate_groups import _places
    energy, peak, ys = [], [], []
    def _pre(m, inp):
        x = inp[0].detach().float(); a = x.abs()
        energy.append((x ** 2).mean((2, 3)).cpu()); peak.append((a.amax((2, 3)) / (a.sum((2, 3)) + 1e-6)).cpu())
    h = model.sdrop_l4.register_forward_pre_hook(_pre)
    model.eval()
    for x, y in loader:
        model(x.to(device)); ys.append(y)
        if sum(len(t) for t in ys) >= n: break
    h.remove()
    return {'energy': torch.cat(energy)[:n].numpy(), 'peak': torch.cat(peak)[:n].numpy(),
            'y': torch.cat(ys)[:n].numpy(), 'places': np.array(_places(loader.dataset))[:n]}


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
    if args.method.startswith(('vitb_','vitbm_','vitbt_')) and args.arch is None:
        args.arch = 'vit_b_16'          # run_id gets a _vit_b_16 tag
    _rid = run_id(args)
    print(f"Run: {_rid}")
    # Abort immediately if this would overwrite existing outputs with different settings.
    os.makedirs(args.save_dir, exist_ok=True)
    guard_overwrite(args, _rid, args.save_dir)
    save_config(args, _rid, args.save_dir)

    # ---- dataset ----
    train_loader, val_loader, num_classes, default_arch, default_pretrained, test_loader = \
        get_dataset(args.dataset, args.data_root, args.batch_size, args.num_workers,
                    strong_aug=args.strong_aug, imb_ratio=args.imb_ratio,
                    subset_frac=args.subset_frac, label_noise=args.label_noise,
                    seed=args.seed, val_frac=args.val_frac, split_seed=args.split_seed)
    if args.train_corr is not None or args.class_balance:
        from dataset import resample_groups
        resample_groups(train_loader.dataset, corr=args.train_corr,
                        class_balance=args.class_balance, seed=args.seed)
    if getattr(train_loader, 'split_indices', None):
        import json as _json
        with open(Path(args.save_dir) / f"{_rid}_split.json", 'w') as f:
            _json.dump(train_loader.split_indices, f)

    arch       = args.arch       if args.arch       is not None else default_arch
    pretrained = args.pretrained if args.pretrained is not None else default_pretrained
    if getattr(args, 'no_pretrained', False): pretrained = False

    # ---- model ----
    if args.method.startswith(("vitb_","vitbm_","vitbt_")):
        # ImageNet-pretrained ViT-B/16 with SDrop on the last-block tokens (embedding dim = channel). See vit_pretrained.py
        from vit_pretrained import build_vitb
        model = build_vitb(num_classes, args.method, args.drop_rate, pretrained=pretrained,
                           norm=args.norm, gamma=(None if args.self_gamma else 1.0),
                           peakedness=args.peakedness, schedule=args.schedule).to(device)
        arch = 'vit_b_16'
    elif args.method.startswith("vit") or args.method.startswith("sdrop_vit"):
        # ViT branch (head-level SDrop, see sdrop_vit.py)
        from sdrop_vit import build_sdrop_vit
        # CIFAR-100/TinyImageNet/CUB are all 3-channel; pick image size by dataset
        img_size = {"cifar100": 32, "tinyimagenet": 64,
                    "cub200": 224, "imagenet100": 224}.get(args.dataset, 32)
        _unsupported = [n for n, v in [
            ('--peakedness', args.peakedness != 'max'),
            ('--self_gamma', bool(args.self_gamma)),
            ('--grad_mode', args.grad_mode != 'off'),
            ('--schedule', args.schedule != 'const'),
            ('--participation', args.participation < 1.0),
            ('--rescale', bool(args.rescale)),
            ('--rescale_mode', args.rescale_mode != 'prob'),
            ('--select', args.select != 'bernoulli'),
            ('--warmup_frac', args.warmup_frac > 0.0),
            ('--width_mult', args.width_mult != 1.0),
        ] if v]
        if _unsupported:
            # run_id appends tags for these flags, so silently ignoring them would save
            # runs with identical settings under different file names (phantom ablation).
            raise SystemExit(f"The ViT branch does not support {_unsupported}.")
        model = build_sdrop_vit(
            num_classes=num_classes, img_size=img_size,
            method=args.method, drop_rate=args.drop_rate,
            layers=args.layers, score_input=args.vit_score, norm=args.norm,
            unit=args.vit_unit,
        ).to(device)
        arch = f"vit-tiny-p4-d12-h3"
    else:
        model = build_model(
            arch=arch,
            num_classes=num_classes,
            method=args.method,
            drop_rate=args.drop_rate,
            schedule=args.schedule,
            participation=args.participation,
            rescale=args.rescale,
            rescale_mode=args.rescale_mode,
            select=args.select,
            warmup_frac=args.warmup_frac,
            layers=args.layers,
            grid_size=args.grid_size,
            peakedness=args.peakedness,
            norm=args.norm,
            gamma=(None if args.self_gamma else 1.0),
            grad_mode=args.grad_mode,
            pretrained=pretrained,
            width_mult=args.width_mult,
        ).to(device)

    print(f"Model: {arch}  |  SDrop: {args.method}  |  "
          f"Rate: {args.drop_rate}  |  Layers: {args.layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- optimizer + scheduler ----
    _cw = None
    if args.class_weight:
        _t = torch.as_tensor(list(train_loader.dataset.targets))
        _n = torch.bincount(_t, minlength=num_classes).float()
        _cw = (_n.sum() / (num_classes * _n.clamp(min=1))).to(device)
        print(f"RWY class weights: {[round(float(w), 3) for w in _cw]}")
    criterion = nn.CrossEntropyLoss(weight=_cw, label_smoothing=args.label_smoothing)
    if args.sd_lambda > 0:
        _ce = criterion
        class _SDLoss(nn.Module):
            def forward(self, out, tgt):
                return _ce(out, tgt) + 0.5 * args.sd_lambda * (out.float() ** 2).sum(1).mean()
        criterion = _SDLoss()
        print(f"Spectral Decoupling: lambda={args.sd_lambda}")
    def _make_optim():      # wrapped in a function because JTT stage 2 recreates it
        if (args.method.startswith("vit") or args.method.startswith("sdrop_vit")) and not args.method.startswith(("vitb_","vitbm_","vitbt_")):
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
        return optimizer, scheduler
    optimizer, scheduler = _make_optim()

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

    # ---- JTT (Liu et al. 2021) ----
    # stage 1: ERM for T epochs -> error set on the training set -> re-initialise
    # (same weights, seed, optimizer, schedule) -> stage 2 = the normal loop below on the
    # training set with each error-set example repeated lambda times.
    if args.jtt_stage1_epochs > 0:
        if args.method != 'none':
            raise SystemExit('--jtt_stage1_epochs is only for --method none (it would not match the run_id tag).')
        import copy
        from torch.utils.data import DataLoader, Subset
        from dataset import _pin_memory
        init_state = copy.deepcopy(model.state_dict())
        for epoch in range(1, args.jtt_stage1_epochs + 1):
            tl, ta = train_epoch(model, train_loader, criterion, optimizer, device, args.log_interval,
                                 mixup_alpha=args.mixup_alpha, num_classes=num_classes,
                                 scaler=scaler, amp_dtype=amp_dtype)
            scheduler.step()
            print(f"[jtt] stage1 epoch {epoch}/{args.jtt_stage1_epochs}  train_loss={tl:.4f}  train_acc={ta:.2f}%")
        ds = train_loader.dataset
        err = jtt_error_set(model, ds, args.batch_size, args.num_workers, device)
        set_seed(args.seed)
        model.load_state_dict(init_state)
        optimizer, scheduler = _make_optim()
        idx = list(range(len(ds))) + err * (int(args.jtt_lambda) - 1)
        train_loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=_pin_memory())
        print(f"[jtt] stage2: {len(ds)} + {len(err)}x{int(args.jtt_lambda)-1} = {len(idx)} samples/epoch, model re-initialised")

    # ---- training loop ----
    best_acc = 0.0
    history = []
    gradlog = L4GradLogger(model, scaler) if args.log_grad_channels else None
    if gradlog is not None:
        open(save_dir / f"{_rid}_gradchan.csv", 'w').close()     # fresh file; write() appends per epoch
    cleanlog = None
    if args.log_grad_clean > 0:
        clean_batches = clean_grad_batches(train_loader, args.log_grad_clean, args.seed)
        cleanlog = L4GradLogger(model, clean=True)
        open(save_dir / f"{_rid}_gradclean.csv", 'w').close()
        print(f"[gradclean] {len(clean_batches)} fixed batches ({sum(len(y) for _, y in clean_batches)} samples)")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Tell the drop-rate schedule the training progress (ignored for schedule='const')
        for _m in model.modules():
            if hasattr(_m, 'set_progress'):
                _m.set_progress(epoch / args.epochs)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.log_interval,
            mixup_alpha=args.mixup_alpha, num_classes=num_classes,
            scaler=scaler, amp_dtype=amp_dtype)
        scheduler.step()
        if gradlog is not None:
            gradlog.write(save_dir / f"{_rid}_gradchan.csv", epoch)
        if cleanlog is not None:
            clean_grad_pass(model, clean_batches, criterion, device, amp_dtype, cleanlog)
            cleanlog.write(save_dir / f"{_rid}_gradclean.csv", epoch)

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

    # If there is a separate official test split (waterbirds), evaluate test once with the best checkpoint.
    # In history, 'acc' is validation; the 'test_*' values here are the final reported numbers.
    if test_loader is not None:
        ck = torch.load(save_dir / f"{run_id(args)}_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(ck['state_dict'])
        tm = evaluate(model, test_loader, device=str(device))
        extra = {}
        try:
            from evaluate_groups import group_accuracies
            extra = group_accuracies(model, test_loader, device=str(device))
        except Exception as e:
            print(f"[test] group evaluation skipped: {e}")
        print(f"[test] best-val epoch {ck['epoch']}: test_acc={tm['acc']:.2f}%  "
              + "  ".join(f"{k}={v:.2f}" for k, v in extra.items()))
        for h in history:
            h['test_acc'] = tm['acc']; h['best_epoch'] = ck['epoch']
            for k, v in extra.items(): h[k] = v
        if args.dump_feats:
            from evaluate_groups import dump_split
            out = {}
            for nm, ld in (('val', val_loader), ('test', test_loader)):
                out.update(dump_split(model, ld, nm, device=str(device)))
            np.savez_compressed(save_dir / f"{run_id(args)}_dump.npz", **out)
            print(f"[dump] saved {run_id(args)}_dump.npz")
        if args.dump_spatial > 0:
            sp = dump_spatial(model, test_loader, args.dump_spatial, device)
            np.savez_compressed(save_dir / f"{_rid}_spatial.npz", **sp)
            print(f"[dump] saved {_rid}_spatial.npz  energy{sp['energy'].shape}")

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
