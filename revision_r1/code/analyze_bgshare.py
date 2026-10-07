"""Explanation-level diagnostic: how much of the layer-4 activation energy sits on the BACKGROUND.

For each test image, x = layer4 output (C, 7, 7) of the model, E = x^2, and m = object mask pooled to 7x7.
  share_all  = sum_{c,hw} E (1-m) / sum_{c,hw} E                         energy on background, all channels
  share_dom  = the same over the image's top-10% channels by energy        (the channels SDrop suppresses)
  area_bg    = mean(1-m)                                                   background share of the image itself
share_dom > share_all means the dominant channels are background channels; share_dom close to 1 means the
background owns them. No class or confound labels are used, only the object mask, which an attribution map
approximates when no mask exists.

Usage: python analyze_bgshare.py <ckpt.pth> <test.parquet> <test_masks.npz> [out.csv] [--limit N]
"""
import sys, io, pathlib, csv
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def main(ckpt, parquet, masks, out=None, limit=0):
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    from model import build_model
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    nc = ck['state_dict']['fc.weight'].shape[0] if 'fc.weight' in ck['state_dict'] else None
    model = build_model('resnet50', nc, 'none', 0.0, [], pretrained=False)
    res = model.load_state_dict(ck['state_dict'], strict=False)
    bad = [k for k in res.missing_keys + res.unexpected_keys if 'sdrop' not in k]
    assert not bad, f'weights do not match: {bad[:5]}'
    model.eval().to(dev)
    feats = []
    model.layer4.register_forward_hook(lambda m, i, o: feats.append(o.detach().float().cpu()))
    d = pd.read_parquet(parquet, columns=['image'])
    M = np.load(masks)['mask'].astype(np.float32)
    assert len(M) == len(d), (len(M), len(d))
    n = limit or len(d)
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),   # = waterbirds_transforms()
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    rows = []
    with torch.no_grad():
        for s in range(0, n, 64):
            idx = range(s, min(n, s + 64))
            x = torch.stack([tf(Image.open(io.BytesIO(d.image[i]['bytes'])).convert('RGB')) for i in idx]).to(dev)
            feats.clear(); model(x); E = feats[0] ** 2                                        # (B, C, 7, 7)
            # the mask is in the 224 composite frame; the image went 224 -> 256 (prep) -> CenterCrop 224 (eval),
            # so apply the same resize + centre crop to the mask before pooling to the 7x7 grid
            mt = F.interpolate(torch.from_numpy(M[list(idx)])[:, None], size=(256, 256), mode='bilinear', align_corners=False)
            mt = mt[:, :, 16:240, 16:240]
            m = F.adaptive_avg_pool2d(mt, E.shape[-2:])                                        # (B, 1, 7, 7)
            bg = 1 - m
            tot = E.sum(dim=(1, 2, 3)); share_all = (E * bg).sum(dim=(1, 2, 3)) / tot
            ce = E.sum(dim=(2, 3)); k = max(1, E.shape[1] // 10)
            top = ce.topk(k, dim=1).indices
            Et = torch.gather(E, 1, top[:, :, None, None].expand(-1, -1, *E.shape[-2:]))
            share_dom = (Et * bg).sum(dim=(1, 2, 3)) / Et.sum(dim=(1, 2, 3))
            for a, b, c in zip(share_all.tolist(), share_dom.tolist(), bg.mean(dim=(1, 2, 3)).tolist()):
                rows.append((a, b, c))
    r = np.array(rows)
    summ = dict(run=pathlib.Path(ckpt).name.replace('_best.pth', ''), n=len(r), area_bg=r[:, 2].mean(),
                share_all=r[:, 0].mean(), share_dom=r[:, 1].mean(), dom_over_all=(r[:, 1] / r[:, 0]).mean())
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}, flush=True)
    if out:
        new = not pathlib.Path(out).exists()
        with open(out, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summ)); new and w.writeheader(); w.writerow(summ)
    return summ


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    lim = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else 0
    if lim: a = [x for x in a if x != str(lim)]
    main(a[0], a[1], a[2], a[3] if len(a) > 3 else None, lim)
