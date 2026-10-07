"""The backbone diagnostic of analyze_units.py, extended to every ViT-B/16 block and site (reviewer item 3).

For each of the 12 encoder blocks and each site, take the ERM model's activations on the Waterbirds test set,
reduce each unit to one number per image, call the 10% of units with the largest mean value "dominant", and report

    background ratio = mean bg-selectivity of dominant units / that of the rest
    class ratio      = mean class-selectivity of dominant units / that of the rest

with selectivity = max over one-vs-rest |AUC - 0.5| (same statistic as analyze_units.py; AUC by average ranks,
identical to sklearn's roc_auc_score, vectorised so all 12 blocks fit in minutes).

Sites per block b:
  residual_b   block output, per embedding dim, mean over patch tokens         (768 units)
  mlp_b        MLP hidden neurons after GELU, mean over patch tokens            (3072 units)  <- the vitbm_ site is b=10
  mlpabs_b     same neurons, mean |activation| (the magnitude/GAP-style score)  (3072 units)
  tokens_b     patch-token norm of the block output                            (196 units)
  heads_b      per-head attention output norm, mean over patch tokens           (12 units)

Usage: python analyze_units_vit_all.py <out.csv> <ckpt.pth> [<ckpt.pth> ...]
"""
import sys, pathlib, csv
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.stats import rankdata


def selectivity(F, lab):
    """F: (N, U), lab: (N,) -> (U,) max over one-vs-rest |AUC - 0.5|; constant units give 0."""
    R = rankdata(F, axis=0)                                   # average ranks, ties as in sklearn
    const = F.std(0) == 0
    out = np.zeros(F.shape[1])
    for v in np.unique(lab):
        t = lab == v
        n1, n0 = t.sum(), (~t).sum()
        if n1 == 0 or n0 == 0:
            continue
        auc = (R[t].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        auc[const] = 0.5
        out = np.maximum(out, np.abs(auc - 0.5))
    return out


def ratios(F, y, p):
    k = max(1, F.shape[1] // 10)
    top = np.argsort(-F.mean(0))[:k]
    rest = np.setdiff1d(np.arange(F.shape[1]), top)
    sb, sc = selectivity(F, p), selectivity(F, y)
    return dict(units=F.shape[1], bg_top=sb[top].mean(), bg_rest=sb[rest].mean(),
                cls_top=sc[top].mean(), cls_rest=sc[rest].mean(),
                bg_ratio=sb[top].mean() / max(sb[rest].mean(), 1e-9),
                cls_ratio=sc[top].mean() / max(sc[rest].mean(), 1e-9))


def head_norms(mha, x):
    """Per-head attention output of torchvision's MultiheadAttention on input x (B, N+1, D) -> (B, H) mean norm over patches."""
    B, N, D = x.shape
    H = mha.num_heads; d = D // H
    q, k, v = Fn.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
    q, k, v = (t.view(B, N, H, d).transpose(1, 2) for t in (q, k, v))          # (B, H, N, d)
    o = torch.softmax(q @ k.transpose(-2, -1) / d ** 0.5, dim=-1) @ v            # (B, H, N, d)
    return o[:, :, 1:].norm(dim=-1).mean(dim=-1)                                 # patches only


def collect(model, loader, dev):
    layers = model.vit.encoder.layers
    buf = {}
    def put(name, t):
        buf.setdefault(name, []).append(t.detach().float().cpu())
    # hooks must return None: a non-None return value would replace the module's output
    def on_block(b):
        def h(m, i, o):
            put(f'residual_{b}', o[:, 1:].mean(1)); put(f'tokens_{b}', o[:, 1:].norm(dim=2))
        return h
    def on_mlp(b):
        def h(m, i, o):
            put(f'mlp_{b}', o[:, 1:].mean(1)); put(f'mlpabs_{b}', o[:, 1:].abs().mean(1))
        return h
    def on_ln1(b, attn):
        def h(m, i, o):
            put(f'heads_{b}', head_norms(attn, o))
        return h
    hooks = []
    for b, blk in enumerate(layers):
        hooks.append(blk.register_forward_hook(on_block(b)))
        hooks.append(blk.mlp[1].register_forward_hook(on_mlp(b)))
        hooks.append(blk.ln_1.register_forward_hook(on_ln1(b, blk.self_attention)))
    ys = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            model(x.to(dev)); ys.append(y)
    for h in hooks:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in buf.items()}, torch.cat(ys).numpy()


def main(out_csv, ckpts):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    from dataset import get_dataset
    from evaluate_groups import _places
    from vit_pretrained import build_vitb
    new = not pathlib.Path(out_csv).exists()
    for path in ckpts:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        a = ck['args']
        *_, test_loader = get_dataset('waterbirds', a.get('data_root', './data'), 64, 8)
        places = np.array(_places(test_loader.dataset))
        model = build_vitb(2, 'vitb_none', 0.0, pretrained=False).to(dev)
        model.load_state_dict(ck['state_dict'])
        feats, y = collect(model, test_loader, dev)
        rid = pathlib.Path(path).name.replace('_best.pth', '')
        rows = []
        for name, F in feats.items():
            site, b = name.rsplit('_', 1)
            r = ratios(F, y, places); r.update(run=rid, site=site, block=int(b))
            rows.append(r)
            print(f"{rid[-6:]} {name:12s} U={r['units']:5d}  bg {r['bg_ratio']:.2f}  class {r['cls_ratio']:.2f}", flush=True)
        with open(out_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader(); new = False
            w.writerows(rows)
    print('wrote', out_csv)


def _selfcheck():
    """AUC-by-ranks must equal sklearn's roc_auc_score, ties included."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(0)
    F = np.round(rng.randn(200, 5), 1); F[:, 4] = 1.0; lab = rng.randint(0, 3, 200)
    ref = np.zeros(5)
    for v in np.unique(lab):
        t = (lab == v).astype(int)
        ref = np.maximum(ref, np.abs(np.array([roc_auc_score(t, F[:, u]) if F[:, u].std() > 0 else 0.5 for u in range(5)]) - 0.5))
    assert np.allclose(selectivity(F, lab), ref), (selectivity(F, lab), ref)


if __name__ == '__main__':
    if sys.argv[1:] == ['--selfcheck']:
        _selfcheck(); print('selfcheck ok')
    else:
        main(sys.argv[1], sys.argv[2:])
