"""Where does the backbone put the shortcut? Two selectivity ratios per candidate suppression site.

For each site (a place where SDrop could act) we take the ERM model's activations on the Waterbirds
test set, reduce each unit to one number per image, call the 10% with the largest mean activation
"dominant", and report

    background ratio = mean bg-selectivity of dominant units / that of the rest
    class ratio      = mean class-selectivity of dominant units / that of the rest

with selectivity = max over one-vs-rest |AUC - 0.5|, as in analyze_selectivity.py. The claim under test
is that suppression helps exactly where the background ratio is above one and the class ratio is not.

Sites: ResNet-50 layer1..layer4 channels; ViT-B/16 residual-stream dimensions, MLP hidden neurons of
each of the last three blocks, patch tokens, and attention heads.

Usage: python analysis/analyze_units.py <ckpt.pth> [out.csv]   (arch is read from the checkpoint's args)
"""
import os, sys, re, json, pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for sdrop/model/dataset
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def selectivity(F, lab):
    """F: (N, U) activations, lab: (N,) -> (U,) max over one-vs-rest |AUC-0.5|."""
    out = np.zeros(F.shape[1])
    for v in np.unique(lab):
        t = (lab == v).astype(int)
        if 0 < t.sum() < len(t):
            a = np.array([roc_auc_score(t, F[:, u]) if F[:, u].std() > 0 else 0.5
                          for u in range(F.shape[1])])
            out = np.maximum(out, np.abs(a - 0.5))
    return out


def ratios(F, y, p):
    dom = F.mean(0)
    k = max(1, F.shape[1] // 10)
    top = np.argsort(-dom)[:k]
    rest = np.setdiff1d(np.arange(F.shape[1]), top)
    sb, sc = selectivity(F, p), selectivity(F, y)
    return dict(units=F.shape[1],
                bg_top=sb[top].mean(), bg_rest=sb[rest].mean(),
                cls_top=sc[top].mean(), cls_rest=sc[rest].mean(),
                bg_ratio=sb[top].mean() / max(sb[rest].mean(), 1e-9),
                cls_ratio=sc[top].mean() / max(sc[rest].mean(), 1e-9))


def collect(model, loader, sites, device):
    """sites: {name: (module, reduce_fn)}; reduce_fn(output_tensor) -> (B, U)."""
    buf = {k: [] for k in sites}
    hooks = []

    def mk(name, fn):
        def hook(m, i, o):
            buf[name].append(fn(o).detach().float().cpu())
        return hook

    for name, (mod, fn) in sites.items():
        hooks.append(mod.register_forward_hook(mk(name, fn)))
    ys = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            model(x.to(device))
            ys.append(y)
    for h in hooks:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in buf.items()}, torch.cat(ys).numpy()


def main(ckpt_path, out_csv=None):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck['args']
    from dataset import get_dataset
    from evaluate_groups import _places
    *_, test_loader = get_dataset('waterbirds', a.get('data_root', './data'), 64, 8)
    places = np.array(_places(test_loader.dataset))

    is_vit = str(a.get('method', '')).startswith(('vitb_', 'vitbm_', 'vitbt_'))
    if is_vit:
        from vit_pretrained import build_vitb
        model = build_vitb(2, 'vitb_none', 0.0, pretrained=False).to(dev)
    else:
        from model import build_model
        model = build_model('resnet50', 2, 'none', 0.0, [], pretrained=False).to(dev)
    model.load_state_dict(ck['state_dict'])

    sp = lambda o: o.mean(dim=(2, 3))                      # conv map  -> per-channel
    tok = lambda o: o[:, 1:].mean(dim=1)                   # (B,N+1,D) -> per-dimension, patches only
    tokdim = lambda o: o[:, 1:].norm(dim=2)                # (B,N+1,D) -> per-patch-token norm
    if is_vit:
        enc = model.vit.encoder.layers
        sites = {'vit_residual_last': (enc[-1], tok),
                 'vit_tokens_last': (enc[-1], tokdim)}
        for off in (1, 2, 3):
            # mlp = [Linear, GELU, Dropout, Linear, Dropout]; hook the GELU -> hidden neurons
            sites[f'vit_mlp_neurons_L-{off}'] = (enc[-off].mlp[1], tok)
    else:
        sites = {f'resnet_layer{i}': (getattr(model, f'layer{i}'), sp) for i in (1, 2, 3, 4)}

    feats, y = collect(model, test_loader, sites, dev)
    rid = pathlib.Path(ckpt_path).name.replace('_best.pth', '')
    rows = []
    for name, F in feats.items():
        r = ratios(F, y, places)
        r.update(run=rid, site=name)
        rows.append(r)
        print(f"{name:24s} U={r['units']:5d}  bg {r['bg_top']:.3f}/{r['bg_rest']:.3f} = {r['bg_ratio']:.2f}"
              f"   class {r['cls_top']:.3f}/{r['cls_rest']:.3f} = {r['cls_ratio']:.2f}", flush=True)
    if out_csv:
        import csv
        new = not pathlib.Path(out_csv).exists()
        with open(out_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print('wrote', out_csv)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
