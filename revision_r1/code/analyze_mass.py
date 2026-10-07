"""How much activation mass does each arm remove? (reviewer: budget-matched controls match channel count, not mass)

On an ERM Waterbirds ResNet-50 checkpoint, take the L4 output on training images and apply each dropout
module in training mode. Report, per arm, the fraction of channels zeroed and the fraction of L1 mass
(sum |x|) and L2 mass (sum x^2) zeroed, averaged over images and over 20 mask draws.

Usage: python analyze_mass.py <erm_ckpt.pth> [n_batches]
"""
import sys
import numpy as np
import torch
from sdrop import build_sdrop


def main(ckpt_path, n_batches=8):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    from dataset import get_dataset
    from model import build_model
    train_loader, *_ = get_dataset('waterbirds', ck['args'].get('data_root', './data'), 64, 8)
    model = build_model('resnet50', 2, 'none', 0.0, [], pretrained=False).to(dev).eval()
    model.load_state_dict(ck['state_dict'])
    feats = []
    h = model.layer4.register_forward_hook(lambda m, i, o: feats.append(o.detach()))
    with torch.no_grad():
        for b, (x, _) in enumerate(train_loader):
            if b >= n_batches:
                break
            model(x.to(dev))
    h.remove()
    X = torch.cat(feats)                                              # (N, C, H, W), post-ReLU
    arms = {}
    for name, rate in [('sdrop_energy', 0.1), ('sdrop_gap', 0.1), ('sdrop_random', 0.1),
                       ('sdrop_random', 0.2), ('sdrop_random', 0.3), ('sdrop_energy', 0.3)]:
        arms[f'{name}@{rate}'] = build_sdrop(name, rate, norm='mean').to(dev).train()
    arms['sdrop_energy@0.1 top-k'] = build_sdrop('sdrop_energy', 0.1, norm='mean', select='topk').to(dev).train()
    l1 = X.abs().sum(dim=(2, 3)); l2 = (X ** 2).sum(dim=(2, 3))    # (N, C)
    print(f"{len(X)} training images, {X.shape[1]} channels")
    for name, mod in arms.items():
        ch, m1, m2 = [], [], []
        with torch.no_grad():
            for _ in range(20):
                live = l1 > 0                                      # channels that carry any mass on this image
                drop = ((mod(X).abs().sum(dim=(2, 3)) == 0) & live).float()
                ch.append(drop.mean().item())
                m1.append(((drop * l1).sum(1) / l1.sum(1)).mean().item())
                m2.append(((drop * l2).sum(1) / l2.sum(1)).mean().item())
        print(f"{name:26s} channels dropped {np.mean(ch):.3f}   L1 mass removed {np.mean(m1):.3f}   L2 mass removed {np.mean(m2):.3f}")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
