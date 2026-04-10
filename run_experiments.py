"""
run_experiments.py
------------------
Reproduces every row in the paper's experiment tables by calling train.py
programmatically.  Results are aggregated into a single CSV summary.

Usage
-----
# Paper datasets only (CIFAR-100 + TinyImageNet)
python run_experiments.py --dataset cifar100
python run_experiments.py --dataset tinyimagenet

# New dataset (CUB-200-2011) — requires data download first
python run_experiments.py --dataset cub200

# Dry run: print commands without executing
python run_experiments.py --dataset cifar100 --dry_run

# Single config (for debugging)
python run_experiments.py --dataset cifar100 --method sdrop_energy --drop_rate 0.1 --layers L4
"""

import argparse
import subprocess
import sys
import csv
import re
import json
from pathlib import Path
from itertools import product


# ---------------------------------------------------------------------------
# Experiment grids  (matches paper Tables 1 & 2, plus CUB-200-2011 plan)
# ---------------------------------------------------------------------------

# Each entry: (method, drop_rate, layers, grid_size)
CIFAR100_GRID = [
    # Baseline
    ('none',          0.0, [],         None),
    # Standard Dropout
    ('dropout',       0.1, ['L3'],     None),
    ('dropout',       0.1, ['L4'],     None),
    ('dropout',       0.3, ['L3'],     None),
    ('dropout',       0.3, ['L4'],     None),
    # SDrop (EGPG)
    ('sdrop',         0.1, ['L3'],     None),
    ('sdrop',         0.1, ['L4'],     None),
    ('sdrop',         0.3, ['L3'],     None),
    ('sdrop',         0.3, ['L4'],     None),
    # SDrop_Energy  ← best result
    ('sdrop_energy',  0.1, ['L3'],     None),
    ('sdrop_energy',  0.1, ['L4'],     None),   # <-- 77.42%
    ('sdrop_energy',  0.3, ['L3'],     None),
    ('sdrop_energy',  0.3, ['L4'],     None),
    # SGridLC
    ('sgridlc',       0.1, ['L3'],     2),
    ('sgridlc',       0.1, ['L3'],     4),
    ('sgridlc',       0.1, ['L4'],     2),
    ('sgridlc',       0.1, ['L4'],     4),
    ('sgridlc',       0.3, ['L3'],     2),
    ('sgridlc',       0.3, ['L3'],     4),
    ('sgridlc',       0.3, ['L4'],     2),
    ('sgridlc',       0.3, ['L4'],     4),
]

TINYIMAGENET_GRID = [
    # Baseline
    ('none',          0.0, [],         None),
    # Standard Dropout
    ('dropout',       0.3, ['L3'],     None),
    ('dropout',       0.3, ['L4'],     None),
    # SDrop (EGPG)
    ('sdrop',         0.1, ['L3'],     None),
    ('sdrop',         0.1, ['L4'],     None),
    ('sdrop',         0.3, ['L3'],     None),
    ('sdrop',         0.3, ['L4'],     None),
    # SDrop_Energy
    ('sdrop_energy',  0.1, ['L3'],     None),
    ('sdrop_energy',  0.1, ['L4'],     None),
    ('sdrop_energy',  0.3, ['L3'],     None),
    ('sdrop_energy',  0.3, ['L4'],     None),
    # SGridLC  ← best: rate=0.3, L3, G=4
    ('sgridlc',       0.1, ['L3'],     2),
    ('sgridlc',       0.1, ['L3'],     4),   # ⚠️ known instability
    ('sgridlc',       0.1, ['L4'],     2),
    ('sgridlc',       0.3, ['L3'],     2),
    ('sgridlc',       0.3, ['L3'],     4),   # <-- 48.66%
    ('sgridlc',       0.3, ['L4'],     2),
]

# CUB-200-2011: planned experiment grid
CUB200_GRID = [
    # Baseline
    ('none',          0.0, [],         None),
    # Standard Dropout
    ('dropout',       0.1, ['L4'],     None),
    ('dropout',       0.3, ['L4'],     None),
    # SDrop (EGPG)
    ('sdrop',         0.1, ['L4'],     None),
    ('sdrop',         0.3, ['L4'],     None),
    # SDrop_Energy
    ('sdrop_energy',  0.1, ['L4'],     None),
    ('sdrop_energy',  0.3, ['L4'],     None),
    # SGridLC
    ('sgridlc',       0.1, ['L3'],     2),
    ('sgridlc',       0.3, ['L3'],     2),
    ('sgridlc',       0.3, ['L3'],     4),
]

DATASET_GRIDS = {
    'cifar100':     CIFAR100_GRID,
    'tinyimagenet': TINYIMAGENET_GRID,
    'cub200':       CUB200_GRID,
}

# Default training hyperparameters per dataset
DATASET_DEFAULTS = {
    'cifar100':     {'epochs': 200, 'lr': 0.1,  'batch_size': 128},
    'tinyimagenet': {'epochs': 200, 'lr': 0.1,  'batch_size': 128},
    'cub200':       {'epochs': 60,  'lr': 0.01, 'batch_size': 64},
}


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(dataset: str, method: str, drop_rate: float,
                  layers: list, grid_size: int | None,
                  seed: int, data_root: str, save_dir: str,
                  num_workers: int) -> list[str]:
    defaults = DATASET_DEFAULTS[dataset]
    cmd = [
        sys.executable, 'train.py',
        '--dataset',     dataset,
        '--method',      method,
        '--drop_rate',   str(drop_rate),
        '--seed',        str(seed),
        '--epochs',      str(defaults['epochs']),
        '--lr',          str(defaults['lr']),
        '--batch_size',  str(defaults['batch_size']),
        '--data_root',   data_root,
        '--save_dir',    save_dir,
        '--num_workers', str(num_workers),
    ]
    if layers:
        cmd += ['--layers'] + layers
    if grid_size is not None:
        cmd += ['--grid_size', str(grid_size)]
    return cmd


def config_label(method: str, drop_rate: float,
                 layers: list, grid_size: int | None) -> str:
    """Human-readable label for a config, matching paper table rows."""
    if method == 'none':
        return 'Baseline'
    layer_str = '+'.join(layers) if layers else '—'
    grid_str  = f' G={grid_size}' if grid_size is not None else ''
    return f'{method}  rate={drop_rate}  {layer_str}{grid_str}'


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def parse_best_acc_from_log(log: str) -> float | None:
    """Extract best val accuracy from train.py stdout."""
    matches = re.findall(r'New best:\s*([\d.]+)%', log)
    if matches:
        return float(matches[-1])
    # fallback: last val_acc line
    matches = re.findall(r'val_acc=([\d.]+)%', log)
    return float(matches[-1]) if matches else None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',     type=str, required=True,
                   choices=['cifar100', 'tinyimagenet', 'cub200'])
    p.add_argument('--seeds',       type=int, nargs='+', default=[0, 1, 2],
                   help='Random seeds to run (default: 0 1 2 → mean±std over 3 runs)')
    p.add_argument('--data_root',   type=str, default='./data')
    p.add_argument('--save_dir',    type=str, default='./checkpoints')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dry_run',     action='store_true',
                   help='Print commands without executing')
    # single-config override (for debugging one row)
    p.add_argument('--method',      type=str, default=None)
    p.add_argument('--drop_rate',   type=float, default=None)
    p.add_argument('--layers',      type=str, nargs='+', default=None)
    p.add_argument('--grid_size',   type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    grid = DATASET_GRIDS[args.dataset]

    # single-config override
    if args.method is not None:
        grid = [(args.method,
                 args.drop_rate if args.drop_rate is not None else 0.1,
                 args.layers    if args.layers    is not None else [],
                 args.grid_size)]

    results_dir = Path(args.save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f'summary_{args.dataset}.csv'

    summary_rows = []

    total = len(grid) * len(args.seeds)
    done  = 0

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset}")
    print(f"Configs : {len(grid)}")
    print(f"Seeds   : {args.seeds}  ({len(args.seeds)} runs per config)")
    print(f"Total   : {total} runs")
    print(f"{'='*60}\n")

    for method, drop_rate, layers, grid_size in grid:
        label = config_label(method, drop_rate, layers, grid_size)
        accs  = []

        for seed in args.seeds:
            cmd = build_command(
                args.dataset, method, drop_rate, layers, grid_size,
                seed, args.data_root, args.save_dir, args.num_workers)

            done += 1
            print(f"[{done}/{total}]  {label}  seed={seed}")
            print(f"  CMD: {' '.join(cmd)}\n")

            if args.dry_run:
                accs.append(None)
                continue

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  [ERROR] Run failed:\n{result.stderr[-500:]}")
                accs.append(None)
            else:
                acc = parse_best_acc_from_log(result.stdout)
                accs.append(acc)
                print(f"  Best acc: {acc:.2f}%" if acc else "  [WARN] Could not parse accuracy")

        # aggregate over seeds
        valid_accs = [a for a in accs if a is not None]
        if valid_accs:
            import statistics
            mean_acc = statistics.mean(valid_accs)
            std_acc  = statistics.stdev(valid_accs) if len(valid_accs) > 1 else 0.0
            print(f"  --> {label}: {mean_acc:.2f} ± {std_acc:.2f}%\n")
        else:
            mean_acc = std_acc = None

        summary_rows.append({
            'dataset':    args.dataset,
            'method':     method,
            'drop_rate':  drop_rate,
            'layers':     '+'.join(layers) if layers else '—',
            'grid_size':  grid_size if grid_size else '—',
            'seeds':      str(args.seeds),
            'accs':       str(valid_accs),
            'mean_acc':   f'{mean_acc:.4f}' if mean_acc else '',
            'std_acc':    f'{std_acc:.4f}'  if std_acc  else '',
            'label':      label,
        })

        # write CSV after every config so partial results are saved
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nSummary saved to: {summary_path}")

    # pretty-print final table
    if not args.dry_run and summary_rows:
        print(f"\n{'='*60}")
        print(f"{'Method':<45} {'Mean Acc':>10} {'±Std':>8}")
        print(f"{'-'*65}")
        for row in summary_rows:
            mean = row['mean_acc'] if row['mean_acc'] else 'TBD'
            std  = row['std_acc']  if row['std_acc']  else ''
            print(f"  {row['label']:<43} {mean:>10}  ±{std}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
