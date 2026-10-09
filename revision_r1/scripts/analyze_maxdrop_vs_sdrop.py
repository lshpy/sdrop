"""Offline comparison of the SDrop top-k rule (per-sample spatial mean-square energy) and the
budget-matched Max-Drop channel rule (per-sample spatial maximum) on the Waterbirds test set.
No training; two checkpoints (ERM, SDrop top-k) are run in eval mode and layer4 (B,2048,7,7) is hooked.
Usage: python analyze_maxdrop_vs_sdrop.py   (CPU, ~minutes)"""
import json, math, os, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sdrop_pod_code'))
from model import build_model                       # noqa: E402
from dataset import WaterbirdsDataset, waterbirds_transforms  # noqa: E402

CK_DIR = HERE.parent / 'paper_results' / 'ckpt_for_fig'
CKPTS = {'erm': 'waterbirds_none_rate0.1_none_ep30_seed0_best.pth',
         'sdrop_topk': 'waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed0_best.pth'}
OUT = HERE.parent.parent / '01_NEUCOM_revision' / 'analysis_maxdrop_vs_sdrop_2026-10-08'
RATE, C, HW = 0.1, 2048, 49
K = int(round(RATE * C))                          # 205, same rule as sdrop.py / baselines.MaxDrop(channel)
GROUPS = ['g0_landbird_land', 'g1_landbird_water', 'g2_waterbird_land', 'g3_waterbird_water']
MINORITY = [1, 2]
CATS = ['max_only', 'energy_only', 'both']
FEATS = ['peakedness', 'mean_sq', 'max', 'centre3x3_frac', 'border_ring_frac']


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = build_model('resnet50', 2, 'none', 0.0, [], norm='mean', pretrained=False)
    missing, unexpected = m.load_state_dict(ck['state_dict'], strict=False)
    print(f'  loaded {path.name}: epoch={ck["epoch"]} best_acc={ck["best_acc"]:.2f} '
          f'missing={missing} unexpected={unexpected}')
    assert all('sdrop' in k for k in missing + unexpected), (missing, unexpected)
    return m.eval()


def entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def run(name, model, loader, labels, places, gen):
    feat = {}
    model.layer4.register_forward_hook(lambda m, i, o: feat.__setitem__('x', o.detach()))
    g_all = labels * 2 + places
    centre = torch.zeros(7, 7, dtype=torch.bool); centre[2:5, 2:5] = True
    border = torch.ones(7, 7, dtype=torch.bool); border[1:6, 1:6] = False
    # accumulators: per-sample scalars (N,) and per-category feature sums
    N = len(g_all)
    ps = {k: torch.zeros(N) for k in ['jaccard', 'inter_frac', 'mass_energy', 'mass_max', 'mass_random',
                                      'n_max_only', 'n_energy_only', 'n_both']}
    preds = torch.zeros(N, dtype=torch.long)
    freq = {'energy': torch.zeros(4, C), 'max': torch.zeros(4, C)}           # per group x channel
    cat_sum = {c: torch.zeros(4, len(FEATS)) for c in CATS}; cat_n = {c: torch.zeros(4) for c in CATS}
    t0, done = time.time(), 0
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs); x = feat['x']; B = x.shape[0]
            sl = slice(done, done + B); g = g_all[sl]
            preds[sl] = out.argmax(1)
            ms = x.pow(2).mean((2, 3))                      # mean-square energy (B,C); monotone with 1-(1+g*ms)^-d
            mx = x.flatten(2).amax(2)                       # spatial maximum (B,C)
            ax = x.abs(); s = ax.flatten(2).sum(2)
            peak = mx.abs() / (s + 1e-6)                    # layer4 is post-ReLU so |x|=x, max/sum in [1/49,1]
            cfrac = (ax * centre).flatten(2).sum(2) / (s + 1e-6)
            bfrac = (ax * border).flatten(2).sum(2) / (s + 1e-6)
            m_e = torch.zeros(B, C, dtype=torch.bool).scatter_(1, ms.topk(K, 1).indices, True)
            m_m = torch.zeros(B, C, dtype=torch.bool).scatter_(1, mx.topk(K, 1).indices, True)
            m_r = torch.zeros(B, C, dtype=torch.bool).scatter_(1, torch.rand(B, C, generator=gen).topk(K, 1).indices, True)
            inter = (m_e & m_m).sum(1).float()
            ps['jaccard'][sl] = inter / (2 * K - inter); ps['inter_frac'][sl] = inter / K
            tot = ms.sum(1)
            for key, m in (('mass_energy', m_e), ('mass_max', m_m), ('mass_random', m_r)):
                ps[key][sl] = (ms * m).sum(1) / tot
            cats = {'max_only': m_m & ~m_e, 'energy_only': m_e & ~m_m, 'both': m_e & m_m}
            ps['n_max_only'][sl] = cats['max_only'].sum(1).float()
            ps['n_energy_only'][sl] = cats['energy_only'].sum(1).float()
            ps['n_both'][sl] = inter
            F = torch.stack([peak, ms, mx, cfrac, bfrac], 2)       # (B,C,5)
            for gi in range(4):
                gm = g == gi
                if not gm.any(): continue
                freq['energy'][gi] += m_e[gm].sum(0).float(); freq['max'][gi] += m_m[gm].sum(0).float()
                for c, m in cats.items():
                    w = (m[gm]).float().unsqueeze(2)
                    cat_sum[c][gi] += (F[gm] * w).sum((0, 1)); cat_n[c][gi] += w.sum()
            done += B
            if (done // B) % 10 == 0 or done == N:
                print(f'  [{name}] {done}/{N}  {time.time() - t0:.0f}s', flush=True)
    assert done == N
    # ---- reduce ----
    def mean_by(v, mask): return float(v[mask].mean())
    res = {'k': K, 'C': C, 'n': N, 'accuracy': {}, 'overall': {}, 'per_group': {}, 'minority': {},
           'channel_stats_by_category': {}, 'selection_coverage': {}}
    for gi, gn in enumerate(GROUPS):
        gm = g_all == gi
        res['accuracy'][gn] = 100 * float((preds[gm] == labels[gm]).float().mean())
    res['accuracy']['avg'] = 100 * float((preds == labels).float().mean())
    res['accuracy']['worst_group'] = min(res['accuracy'][gn] for gn in GROUPS)
    keys = list(ps)
    res['overall'] = {k: float(ps[k].mean()) for k in keys}
    res['overall']['mass_random_expected'] = RATE
    for gi, gn in enumerate(GROUPS):
        res['per_group'][gn] = {k: mean_by(ps[k], g_all == gi) for k in keys}
    mino = (g_all == 1) | (g_all == 2)
    res['minority'] = {k: mean_by(ps[k], mino) for k in keys}

    def cat_block(gsel):
        out = {}
        for c in CATS:
            n = cat_n[c][gsel].sum(); sm = cat_sum[c][gsel].sum(0)
            out[c] = {'n_channel_sample_pairs': int(n), 'mean_per_sample': float(n / int((torch.isin(g_all, torch.tensor(gsel))).sum()))}
            out[c].update({f: (float(sm[i] / n) if n > 0 else None) for i, f in enumerate(FEATS)})
        return out
    res['channel_stats_by_category'] = {'overall': cat_block([0, 1, 2, 3]), 'minority': cat_block(MINORITY)}
    for gi, gn in enumerate(GROUPS):
        res['channel_stats_by_category'][gn] = cat_block([gi])
    res['channel_stats_note'] = ('centre3x3_frac / border_ring_frac are the fraction of a channel\'s |x| mass in the '
                                 'centre 3x3 cells / the outer ring of the 7x7 map. They are a crude spatial proxy, '
                                 'NOT a measurement of object vs background content.')
    def cov_block(gsel):
        out = {}
        for rule in ('energy', 'max'):
            f = freq[rule][gsel].sum(0); nsamp = float(f.sum() / K)
            out[rule] = {'n_channels_ever_selected': int((f > 0).sum()),
                         'entropy_nats': entropy(f), 'max_entropy_nats': math.log(C),
                         'effective_n_channels_exp_entropy': math.exp(entropy(f)),
                         'top20_channels_share_of_selections': float(f.topk(20).values.sum() / f.sum()),
                         'n_channels_selected_in_over_half_of_samples': int((f > 0.5 * nsamp).sum())}
        fe, fm = freq['energy'][gsel].sum(0), freq['max'][gsel].sum(0)
        out['pearson_corr_of_selection_frequency_between_rules'] = float(torch.corrcoef(torch.stack([fe, fm]))[0, 1])
        return out
    res['selection_coverage'] = {'overall': cov_block([0, 1, 2, 3]), 'minority': cov_block(MINORITY)}
    return res


def fmt(v, d=3):
    return '-' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))


def to_md(all_res):
    L = ['# Energy top-k (SDrop) vs max top-k (Max-Drop channel) at layer4 -- Waterbirds test, offline',
         f'Date 2026-10-08. k = {K} of {C} channels (rate {RATE}); ranking by mean-square energy is identical to ranking by the bounded transform 1-(1+gamma*Sigma)^-delta (monotone). Both checkpoints run in eval mode with no dropout.', '']
    for name, r in all_res.items():
        a = r['accuracy']
        L += [f'## Checkpoint: {name}', f'Accuracy avg {a["avg"]:.2f}, worst-group {a["worst_group"]:.2f} '
              + ', '.join(f'{g} {a[g]:.1f}' for g in GROUPS), '',
              '### Overlap and removed mass (per-sample means)',
              '| subset | Jaccard | inter/k | mass energy-rule | mass max-rule | mass random-k |', '|---|---|---|---|---|---|']
        rows = [('overall', r['overall'])] + [(g, r['per_group'][g]) for g in GROUPS] + [('minority (g1+g2)', r['minority'])]
        for n, d in rows:
            L.append(f'| {n} | {d["jaccard"]:.3f} | {d["inter_frac"]:.3f} | {d["mass_energy"]:.3f} | {d["mass_max"]:.3f} | {d["mass_random"]:.3f} |')
        L += ['', '### Channels by category (means over channel-sample pairs)',
              '| subset | category | pairs | per sample | peakedness max/sum | mean-square | max | centre 3x3 frac | border ring frac |',
              '|---|---|---|---|---|---|---|---|---|']
        for sub in ['overall', 'minority']:
            for c in CATS:
                d = r['channel_stats_by_category'][sub][c]
                L.append(f'| {sub} | {c} | {d["n_channel_sample_pairs"]} | {d["mean_per_sample"]:.1f} | {fmt(d["peakedness"])} | '
                         f'{fmt(d["mean_sq"])} | {fmt(d["max"])} | {fmt(d["centre3x3_frac"])} | {fmt(d["border_ring_frac"])} |')
        L += ['', '### Selection coverage over the test set',
              '| subset | rule | channels ever selected | entropy (nats, max 7.625) | effective #channels | top-20 share | selected in >50% of samples |',
              '|---|---|---|---|---|---|---|']
        for sub in ['overall', 'minority']:
            for rule in ('energy', 'max'):
                d = r['selection_coverage'][sub][rule]
                L.append(f'| {sub} | {rule} | {d["n_channels_ever_selected"]} | {d["entropy_nats"]:.3f} | {d["effective_n_channels_exp_entropy"]:.0f} | '
                         f'{d["top20_channels_share_of_selections"]:.3f} | {d["n_channels_selected_in_over_half_of_samples"]} |')
            L.append(f'Correlation of per-channel selection frequency between rules ({sub}): '
                     f'{r["selection_coverage"][sub]["pearson_corr_of_selection_frequency_between_rules"]:.3f}')
        L.append('')
    L += ['Centre/border fractions are a crude spatial proxy on the 7x7 map, not a measurement of object or background content.', '', '## Reading', '']
    e, s = all_res['erm'], all_res['sdrop_topk']
    def cs(r, c, f): return r['channel_stats_by_category']['overall'][c][f]
    L.append(
        f'The two rules pick largely the same channels: Jaccard {e["overall"]["jaccard"]:.2f} (ERM) and {s["overall"]["jaccard"]:.2f} (SDrop), '
        f'i.e. {100*e["overall"]["inter_frac"]:.0f}% and {100*s["overall"]["inter_frac"]:.0f}% of the k={K} channels coincide; the minority groups give '
        f'{e["minority"]["jaccard"]:.2f} and {s["minority"]["jaccard"]:.2f}. The energy rule removes {100*e["overall"]["mass_energy"]:.0f}% / {100*s["overall"]["mass_energy"]:.0f}% '
        f'of the squared-activation mass (ERM / SDrop), the max rule {100*e["overall"]["mass_max"]:.0f}% / {100*s["overall"]["mass_max"]:.0f}%, against {100*e["overall"]["mass_random"]:.0f}% for a random k-subset. '
        f'Channels chosen only by the max rule are spatially peaked (max/sum {cs(e,"max_only","peakedness"):.2f} / {cs(s,"max_only","peakedness"):.2f}) with low mean-square energy '
        f'({cs(e,"max_only","mean_sq"):.3f} / {cs(s,"max_only","mean_sq"):.3f}), whereas channels chosen only by the energy rule are diffuse (max/sum {cs(e,"energy_only","peakedness"):.2f} / {cs(s,"energy_only","peakedness"):.2f}) '
        f'with higher mean-square energy ({cs(e,"energy_only","mean_sq"):.3f} / {cs(s,"energy_only","mean_sq"):.3f}) but a lower maximum ({cs(e,"energy_only","max"):.2f} / {cs(s,"energy_only","max"):.2f} vs {cs(e,"max_only","max"):.2f} / {cs(s,"max_only","max"):.2f}); '
        f'their centre-3x3 mass fractions are {cs(e,"energy_only","centre3x3_frac"):.2f} / {cs(s,"energy_only","centre3x3_frac"):.2f} (energy-only) against {cs(e,"max_only","centre3x3_frac"):.2f} / {cs(s,"max_only","centre3x3_frac"):.2f} (max-only), a spatial proxy only. '
        f'Over the test set the energy rule ever selects {e["selection_coverage"]["overall"]["energy"]["n_channels_ever_selected"]} / {s["selection_coverage"]["overall"]["energy"]["n_channels_ever_selected"]} channels (entropy {e["selection_coverage"]["overall"]["energy"]["entropy_nats"]:.2f} / {s["selection_coverage"]["overall"]["energy"]["entropy_nats"]:.2f} nats) '
        f'and the max rule {e["selection_coverage"]["overall"]["max"]["n_channels_ever_selected"]} / {s["selection_coverage"]["overall"]["max"]["n_channels_ever_selected"]} (entropy {e["selection_coverage"]["overall"]["max"]["entropy_nats"]:.2f} / {s["selection_coverage"]["overall"]["max"]["entropy_nats"]:.2f}). '
        f'These are descriptive statistics of which channels each rule removes at test time; they do not establish why the two training arms differ.')
    return '\n'.join(L) + '\n'


def main():
    torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
    _, tf = waterbirds_transforms()
    ds = WaterbirdsDataset(CK_DIR / 'test-00000-of-00001.parquet', tf)
    labels, places = torch.tensor(ds.targets), torch.tensor(ds.places)
    g = labels * 2 + places
    print('groups', torch.bincount(g, minlength=4).tolist(), 'k =', K, 'threads', torch.get_num_threads())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    all_res = {}
    for name, fn in CKPTS.items():
        print(f'== {name}')
        all_res[name] = run(name, load_model(CK_DIR / fn), loader, labels, places, torch.Generator().manual_seed(0))
    OUT.with_suffix('.json').write_text(json.dumps(all_res, indent=1))
    OUT.with_suffix('.md').write_text(to_md(all_res))
    print('wrote', OUT.with_suffix('.json'), OUT.with_suffix('.md'))


if __name__ == '__main__':
    main()
