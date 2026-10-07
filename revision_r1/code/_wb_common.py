"""Waterbirds 사후 분석 공통: 체크포인트 → 모델, 테스트 로더, 그룹 정확도."""
import inspect, torch
from pathlib import Path
from model import build_model
from dataset import WaterbirdsDataset, waterbirds_transforms
from torch.utils.data import DataLoader, Subset

def build_from_ckpt(ck, device, method=None, drop_rate=None, layers=None, **override):
    a = dict(ck['args']); a.update(override)
    sig = inspect.signature(build_model).parameters
    kw = {k: a[k] for k in ('peakedness', 'norm', 'schedule', 'grid_size', 'rescale', 'rescale_mode',
                            'select', 'warmup_frac', 'participation', 'width_mult', 'grad_mode') if k in sig and k in a}
    if a.get('self_gamma'): kw['gamma'] = None
    m = build_model(a.get('arch') or 'resnet50', 2, method or a['method'],
                    drop_rate if drop_rate is not None else a['drop_rate'],
                    layers if layers is not None else a['layers'], **kw)
    missing, unexpected = m.load_state_dict(ck['state_dict'], strict=False)
    assert not unexpected and all('.t' in k for k in missing), (missing, unexpected)
    return m.to(device).eval()

def test_loader(root='./data', bs=64, n=None, workers=8):
    _, tf = waterbirds_transforms()
    ds = WaterbirdsDataset(Path(root) / 'waterbirds' / 'test-00000-of-00001.parquet', tf)
    if n: ds = Subset(ds, list(range(n)))
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)

def labels_places(loader):
    ds = loader.dataset
    if isinstance(ds, Subset):
        base = ds.dataset; idx = ds.indices
        return torch.tensor([base.targets[i] for i in idx]), torch.tensor([base.places[i] for i in idx])
    return torch.tensor(ds.targets), torch.tensor(ds.places)

def group_metrics(preds, labels, places):
    out = {}
    accs = []
    for lab in (0, 1):
        for pl in (0, 1):
            m = (labels == lab) & (places == pl)
            acc = 100.0 * (preds[m] == labels[m]).float().mean().item()
            out[f'g{lab}{pl}'] = acc; accs.append(acc)
    out['worst_group'] = min(accs); out['avg'] = 100.0 * (preds == labels).float().mean().item()
    return out
