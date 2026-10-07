"""Waterbirds 그룹별 정확도 — worst-group accuracy 가 subpopulation-shift 표준 지표.

그룹 = (label, place): (0,0) landbird/land, (0,1) landbird/water [소수],
                       (1,0) waterbird/land [소수], (1,1) waterbird/water.
loader.dataset 이 `places` 속성을 가져야 한다 (WaterbirdsDataset).
Subset 으로 감싼 경우 indices 를 따라간다.
"""
import torch


def _places(ds):
    if hasattr(ds, 'places'):
        return list(ds.places)
    if hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
        base = _places(ds.dataset)
        return [base[i] for i in ds.indices]
    raise AttributeError("dataset 에 places 없음")


@torch.no_grad()
def group_accuracies(model, loader, device='cuda'):
    model.eval()
    places = torch.tensor(_places(loader.dataset))
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu()); labels.append(y)
    preds = torch.cat(preds); labels = torch.cat(labels)
    assert len(places) == len(labels), (len(places), len(labels))
    res = {}
    accs = []
    pairs = sorted({(int(a), int(b)) for a, b in zip(labels.tolist(), places.tolist())})
    if all(a in (0, 1) and b in (0, 1) for a, b in pairs):      # binary benchmarks: keep the 4 fixed keys
        pairs = [(a, b) for a in (0, 1) for b in (0, 1)]
    # multi-digit ids (cct20: 12 classes x location ids up to 130) would make g{lab}{pl} ambiguous
    # ('g338' = (3,38) or (33,8)); add '_' only then, so every existing benchmark keeps its keys.
    sep = '_' if any(a >= 10 or b >= 10 for a, b in pairs) else ''
    for lab, pl in pairs:
        m = (labels == lab) & (places == pl)
        acc = 100.0 * (preds[m] == labels[m]).float().mean().item() if m.any() else float('nan')
        res[f'test_acc_g{lab}{sep}{pl}'] = acc
        accs.append(acc)
    res['test_worst_group'] = min(accs)
    res['test_avg_group'] = sum(accs) / len(accs)
    return res


@torch.no_grad()
def dump_split(model, loader, name, device='cuda'):
    """logits, penultimate features (fp16), labels, places for one split -> dict with <name>_ prefix."""
    model.eval()
    feats = []
    head = getattr(model, 'fc', None) or model.vit.heads          # ResNet .fc / SDropViTB .vit.heads
    h = head.register_forward_hook(lambda m, i, o: feats.append(i[0].detach().half().cpu()))
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu()); labels.append(y)
    h.remove()
    import numpy as np
    return {f'{name}_logits': torch.cat(logits).numpy(), f'{name}_feats': torch.cat(feats).numpy(),
            f'{name}_y': torch.cat(labels).numpy(), f'{name}_places': np.array(_places(loader.dataset))}
