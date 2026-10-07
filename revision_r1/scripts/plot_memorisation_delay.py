#!/usr/bin/env python3
"""Memorisation-delay figure for the Neurocomputing R1 Part I (CIFAR-100 10% data, 18 paired seeds, L4, p=0.1).
Train accuracy and test accuracy per epoch, mean over seeds, for none / random channel dropout / SDrop; crossover epoch
where SDrop's train accuracy first exceeds random's; final train-accuracy difference. Writes fig + numbers.
    python3 scripts/plot_memorisation_delay.py"""
import glob, re, pathlib, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parent.parent / 'paper_results' / 'raw_pod'
OUT = pathlib.Path(__file__).resolve().parents[2] / '01_NEUCOM_revision' / 'manuscript'
ARMS = {'No dropout': 'none_rate0.1_none_sub0.1', 'Random channel dropout': 'sdrop_random_rate0.1_L4_sub0.1_nmmean', 'SDrop': 'sdrop_rate0.1_L4_sub0.1_nmmean'}
H = {}
for lab, stem in ARMS.items():
    d = {}
    for f in glob.glob(str(R / f'cifar100_{stem}_seed*_history.csv')):
        s = int(re.search(r'_seed(\d+)_', f).group(1)); h = pd.read_csv(f)
        if len(h) >= 200: d[s] = h[['train_acc', 'acc']].values[:200]
    H[lab] = d
seeds = sorted(set.intersection(*[set(d) for d in H.values()]))
A = {lab: np.stack([H[lab][s] for s in seeds]) for lab in ARMS}        # (n, 200, 2)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2), sharex=True)
ep = np.arange(1, 201)
for lab, c in zip(ARMS, ['0.45', 'tab:blue', 'tab:red']):
    for j, (axx, name) in enumerate(zip(ax, ['Training accuracy (%)', 'Test accuracy (%)'])):
        m = A[lab][:, :, j].mean(0); se = A[lab][:, :, j].std(0, ddof=1) / np.sqrt(len(seeds))
        axx.plot(ep, m, color=c, lw=1.4, label=lab); axx.fill_between(ep, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.18, lw=0)
        axx.set_ylabel(name); axx.set_xlabel('Epoch')
d_tr = A['SDrop'][:, :, 0] - A['Random channel dropout'][:, :, 0]          # (n, 200)
m_tr = d_tr.mean(0); cross = int(np.argmax(m_tr > 0)) + 1 if (m_tr > 0).any() else None
d_te = A['SDrop'][:, :, 1] - A['Random channel dropout'][:, :, 1]; m_te = d_te.mean(0)
cross_te = int(np.argmax(np.cumsum((m_te <= 0)[::-1])[::-1] == 0)) + 1
ax[0].axvline(cross, color='k', ls=':', lw=0.9); ax[0].text(cross + 3, 15, f'train-acc parity\nepoch {cross}', fontsize=8)
ax[1].axvline(cross_te, color='k', ls=':', lw=0.9); ax[1].text(cross_te + 3, 8, f'SDrop test acc ahead\nfrom epoch {cross_te}', fontsize=8)
ax[0].legend(frameon=False, fontsize=8, loc='lower right'); fig.tight_layout()
fig.savefig(OUT / 'fig_memdelay.pdf'); fig.savefig(OUT / 'fig_memdelay.png', dpi=200)
def pr(d): t, p = stats.ttest_rel(d, np.zeros_like(d)); ci = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return f'{d.mean():+.2f} ± {ci:.2f} (n={len(d)}, p={p:.3g})'
lines = [f'seeds paired: {len(seeds)}', f'test-acc crossover (SDrop mean test acc > random from this epoch to the end): {cross_te}', f'crossover epoch (SDrop train acc first exceeds random, mean curve): {cross}',
         f'SDrop - random train acc, epoch 10: {pr(d_tr[:, 9])}', f'epoch 20: {pr(d_tr[:, 19])}', f'epoch 40: {pr(d_tr[:, 39])}',
         f'epoch 100: {pr(d_tr[:, 99])}', f'final (ep 200): {pr(d_tr[:, 199])}',
         f'final train acc: none {A["No dropout"][:, 199, 0].mean():.2f}, random {A["Random channel dropout"][:, 199, 0].mean():.2f}, SDrop {A["SDrop"][:, 199, 0].mean():.2f}',
         f'SDrop - random test acc, final: {pr(A["SDrop"][:, 199, 1] - A["Random channel dropout"][:, 199, 1])}',
         f'generalisation gap (train - test) final: random {(A["Random channel dropout"][:, 199, 0] - A["Random channel dropout"][:, 199, 1]).mean():.2f}, SDrop {(A["SDrop"][:, 199, 0] - A["SDrop"][:, 199, 1]).mean():.2f}']
(OUT / 'memdelay_numbers.txt').write_text('\n'.join(lines)); print('\n'.join(lines))
