"""
ResNet backbone with SDrop inserted at Layer 3 and/or Layer 4.

Supported architectures:
  resnet18  → CIFAR-100  (modified: 3×3 conv1, no maxpool)
  resnet50  → TinyImageNet, CUB-200-2011  (standard)

SDrop is applied *after* the residual block output, before passing
to the next stage — insertion points L3 and L4 (L1/L2 also available).
"""

import torch
import torch.nn as nn
from torchvision import models
from sdrop import build_sdrop
from baselines import build_baseline


def _resnet18_cifar(num_classes: int, width_mult: float = 1.0,
                    pretrained: bool = False):
    """CIFAR-adapted ResNet-18 whose stage widths scale by `width_mult`.

    Narrowing the network is how we manipulate *channel scarcity*. SDrop's
    hypothesis is that a few channels monopolise the representation, so its
    effect should grow as channels become scarce relative to the number of
    classes. width_mult=1.0 reproduces the standard 64/128/256/512 stages.
    """
    from torchvision.models.resnet import ResNet, BasicBlock

    if pretrained and width_mult != 1.0:
        raise ValueError('pretrained weights only exist for width_mult=1.0')

    m = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    planes = [max(8, int(round(c * width_mult))) for c in (64, 128, 256, 512)]

    if width_mult != 1.0:
        m.inplanes = planes[0]
        m.bn1 = nn.BatchNorm2d(planes[0])
        m.layer1 = m._make_layer(BasicBlock, planes[0], 2)
        m.layer2 = m._make_layer(BasicBlock, planes[1], 2, stride=2)
        m.layer3 = m._make_layer(BasicBlock, planes[2], 2, stride=2)
        m.layer4 = m._make_layer(BasicBlock, planes[3], 2, stride=2)
        m.fc = nn.Linear(planes[3] * BasicBlock.expansion, num_classes)
    else:
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    # CIFAR adaptation: 3x3 stride-1 stem, no maxpool
    m.conv1 = nn.Conv2d(3, planes[0], kernel_size=3, stride=1,
                        padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


# ---------------------------------------------------------------------------
# CIFAR-specific ResNet (He et al. 2016, 3-stage) and WideResNet
#
# the standard CIFAR backbones used by FocusedDropout, DropBlock, etc. torchvision's
# ResNet-18/50 are ImageNet models, oversized for 32x32 and not directly comparable
# with the papers in this area. ResNet-56/110 are thin, deep nets with 16/32/64 channels,
# so their "channel sparsity" differs strongly from ResNet-18 (64-512 channels), which
# makes them a useful additional test condition for SDrop.
# ---------------------------------------------------------------------------

class _CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inp != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class _WideBasicBlock(nn.Module):
    """Pre-activation block of WideResNet."""
    expansion = 1

    def __init__(self, inp, planes, stride=1, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.equal = (inp == planes and stride == 1)
        self.shortcut = None if self.equal else nn.Conv2d(inp, planes, 1, stride, bias=False)

    def forward(self, x):
        o = self.relu(self.bn1(x))
        sc = x if self.equal else self.shortcut(o)
        o = self.conv1(o)
        o = self.conv2(self.drop(self.relu(self.bn2(o))))
        return o + sc


def _make_cifar_stages(arch: str, num_classes: int):
    """Return (stem, [stage...], head_dim); each stage is an nn.Sequential."""
    if arch.startswith('resnet') and arch not in ('resnet18', 'resnet50'):
        depth = int(arch.replace('resnet', ''))
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR ResNet depth must be 6n+2: '{arch}'")
        n = (depth - 2) // 6
        widths = [16, 32, 64]
        stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                             nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        stages, inp = [], 16
        for i, w in enumerate(widths):
            blocks = []
            for b in range(n):
                blocks.append(_CifarBasicBlock(inp, w, stride=(2 if i > 0 and b == 0 else 1)))
                inp = w
            stages.append(nn.Sequential(*blocks))
        return stem, stages, widths[-1]

    if arch.startswith('wrn'):                      # e.g. wrn28_10
        body = arch.replace('wrn', '')
        depth, widen = (int(v) for v in body.split('_'))
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth must be 6n+4: '{arch}'")
        n = (depth - 4) // 6
        widths = [16 * widen, 32 * widen, 64 * widen]
        stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False))
        stages, inp = [], 16
        for i, w in enumerate(widths):
            blocks = []
            for b in range(n):
                blocks.append(_WideBasicBlock(inp, w, stride=(2 if i > 0 and b == 0 else 1)))
                inp = w
            stages.append(nn.Sequential(*blocks))
        stages[-1] = nn.Sequential(stages[-1], nn.BatchNorm2d(widths[-1]),
                                   nn.ReLU(inplace=True))
        return stem, stages, widths[-1]

    raise ValueError(f"Unsupported CIFAR arch: '{arch}'")


class SDroCifarNet(nn.Module):
    """3-stage CIFAR backbone + SDrop.

    Insertion points are mapped **from the end**: L4 = last stage, L3 = the one
    before it, L2 = the one before that. With 3 stages, L1 does not exist (error if requested).
    This way L4 of ResNet-18 (4 stages) and L4 of ResNet-56 (3 stages) both
    mean "the deepest stage".
    """

    def __init__(self, arch, num_classes, sdrop_modules: dict):
        super().__init__()
        stem, stages, dim = _make_cifar_stages(arch, num_classes)
        self.stem = stem
        self.stages = nn.ModuleList(stages)
        n = len(stages)
        names = [f'L{4 - (n - 1 - i)}' for i in range(n)]      # L4 counted from the end
        self.stage_names = names
        for want in sdrop_modules:
            if sdrop_modules[want] is not None and want not in names:
                raise ValueError(
                    f"'{arch}' has {n} stages, so the only insertion points are {names} "
                    f"('{want}' requested).")
        # WRN is pre-activation, so the outputs of all stages except the last are
        # raw signed pre-activations (measured negative fraction 0.52). The score
        # E_c = 1-(1+gamma*Sigma_c)^-delta and the spatial concentration P_c assume post-ReLU input
        # and would measure something else, so the same --layers flag would mean a different intervention per architecture.
        # The last stage gets BN+ReLU appended above, so it is post-ReLU (negative fraction 0.00).
        if arch.startswith('wrn'):
            bad = [w for w in sdrop_modules
                   if sdrop_modules[w] is not None and w != names[-1]]
            if bad:
                raise ValueError(
                    f"'{arch}' is pre-activation, so the input at {bad} is a raw "
                    f"pre-activation (52% negative). It does not match the post-ReLU assumption of the SDrop "
                    f"scores and is not comparable with other architectures. "
                    f"Use only '{names[-1]}'.")
        self.drops = nn.ModuleList(
            [sdrop_modules.get(nm) or nn.Identity() for nm in names])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        """He initialisation. The PyTorch default (kaiming_uniform_, a=sqrt(5)) gives a conv std
        0.41x that of He fan_out. At 110 layers this difference breaks training itself
        and would be confounded with any SDrop effect.

        The inserted modules (self.drops) are left alone: SE/CBAM have their own initialisation,
        and their fc has bias=False, so an unconditional zeros_ would crash."""
        skip = {id(mm) for d in self.drops for mm in d.modules()}
        for m in self.modules():
            if id(m) in skip:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage, drop in zip(self.stages, self.drops):
            x = drop(stage(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


class SDroResNet(nn.Module):
    """
    ResNet with optional SDrop at L3 and/or L4.

    Args:
        arch        : 'resnet18' or 'resnet50'
        num_classes : number of output classes
        sdrop_l3    : nn.Module (SDrop variant) or None
        sdrop_l4    : nn.Module (SDrop variant) or None
        pretrained  : load ImageNet weights (useful for fine-tuning on CUB)
    """
    def __init__(self, arch: str = 'resnet18', num_classes: int = 100,
                 sdrop_l3: nn.Module = None, sdrop_l4: nn.Module = None,
                 sdrop_l1: nn.Module = None, sdrop_l2: nn.Module = None,
                 pretrained: bool = False, width_mult: float = 1.0):
        super().__init__()

        if arch == 'resnet18':
            backbone = _resnet18_cifar(num_classes, width_mult, pretrained)
        elif arch == 'resnet50':
            if width_mult != 1.0:
                raise ValueError('width_mult is only implemented for resnet18')
            backbone = models.resnet50(
                weights='IMAGENET1K_V1' if pretrained else None)
            backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported arch: '{arch}'. Use 'resnet18' or 'resnet50'.")

        # store sub-modules individually so state_dict / parameters work correctly
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc      = backbone.fc

        # L1/L2 are also opened as insertion points. If the drop acts only at L4 (right before GAP),
        # average pooling absorbs the mask effect and the training loss barely changes.
        self.sdrop_l1 = sdrop_l1 if sdrop_l1 is not None else nn.Identity()
        self.sdrop_l2 = sdrop_l2 if sdrop_l2 is not None else nn.Identity()
        self.sdrop_l3 = sdrop_l3 if sdrop_l3 is not None else nn.Identity()
        self.sdrop_l4 = sdrop_l4 if sdrop_l4 is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.sdrop_l1(x)       # SDrop at L1 (if configured)

        x = self.layer2(x)
        x = self.sdrop_l2(x)       # SDrop at L2 (if configured)

        x = self.layer3(x)
        x = self.sdrop_l3(x)       # SDrop at L3 (if configured)

        x = self.layer4(x)
        x = self.sdrop_l4(x)       # SDrop at L4 (if configured)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(arch: str, num_classes: int, method: str,
                drop_rate: float, layers: list, grid_size: int = 2,
                peakedness: str = 'max', norm: str = 'mean',
                gamma: float = 1.0, grad_mode: str = 'off',
                schedule: str = 'const', participation: float = 1.0,
                rescale: bool = False, rescale_mode: str = 'prob',
                select: str = 'bernoulli', warmup_frac: float = 0.0,
                pretrained: bool = False,
                width_mult: float = 1.0) -> SDroResNet:
    """
    Convenience builder.

    Args:
        arch        : 'resnet18' | 'resnet50'
        num_classes : number of classes
        method      : 'none' | 'dropout' | 'dropout_std' | 'sdrop' | 'sdrop_energy'
                      | 'sdrop_peak' | 'sdrop_random' | 'sgridlc'
        peakedness  : 'max' (paper) | 'entropy' (resolution-invariant)
        norm        : 'max' (paper) | 'mean' (mean-preserving)
        gamma       : energy scale; None selects self-normalising gamma
        drop_rate   : drop probability
        layers      : list of insertion points, e.g. ['L3'], ['L4'], ['L3','L4']
        grid_size   : G for SGridLC (ignored for other methods)
        pretrained  : load ImageNet weights

    Example:
        # Replicate best CIFAR-100 result: SDrop_Energy, Rate=0.1, L4
        model = build_model('resnet18', 100, 'sdrop_energy', 0.1, ['L4'])

        # Replicate best TinyImageNet result: SGridLC, Rate=0.3, L3, G=4
        model = build_model('resnet50', 200, 'sgridlc', 0.3, ['L3'], grid_size=4)

        # Baseline (no dropout)
        model = build_model('resnet18', 100, 'none', 0.0, [])
    """
    # SE/CBAM need the channel count to create their parameters before the optimizer is built.
    if arch in ('resnet18',):
        _ch = {'L1': 64, 'L2': 128, 'L3': 256, 'L4': 512}
    elif arch in ('resnet50',):
        _ch = {'L1': 256, 'L2': 512, 'L3': 1024, 'L4': 2048}
    elif arch.startswith('wrn'):
        _w = int(arch.replace('wrn', '').split('_')[1])
        _ch = {'L2': 16 * _w, 'L3': 32 * _w, 'L4': 64 * _w}
    else:
        _ch = {'L2': 16, 'L3': 32, 'L4': 64}
    if width_mult != 1.0:
        _ch = {k: max(8, int(round(v * width_mult))) for k, v in _ch.items()}

    def _make(where):
        if where not in layers:
            return None
        if method in ('guided_topk', 'guided_dr'):
            # Guided Dropout has its own schedule (gdtopk alternation). Passing 'const' gives,
            # unlike the original paper, a fair comparison at the same budget as SDrop throughout.
            _bad = [n for n, v in (
                ('--rescale', bool(rescale)), ('--select', select != 'bernoulli'),
                ('--peakedness', peakedness != 'max'), ('--grad_mode', grad_mode != 'off'),
                ('--participation', participation < 1.0)) if v]
            if _bad:
                raise ValueError(f"'{method}' does not support {_bad}.")
            return build_baseline(method, drop_rate, channels=_ch.get(where),
                                  schedule=schedule, warmup_frac=warmup_frac)
        if method in ('dropblock', 'senet', 'cbam', 'wcd', 'focuseddrop'):
            # comparison baselines (baselines.py)
            # These modules do not accept SDrop-specific flags. Silently ignoring them would
            # let run_id append tags, so runs with identical settings would be saved under
            # different file names (phantom ablations). Reject explicitly.
            _ignored = [n for n, v in (
                ('--schedule', schedule != 'const'),
                ('--rescale', bool(rescale)),
                ('--rescale_mode', rescale_mode != 'prob'),
                ('--select', select != 'bernoulli'),
                ('--warmup_frac', warmup_frac > 0.0),
                ('--norm', False),
                ('--peakedness', peakedness != 'max'),
                ('--grad_mode', grad_mode != 'off'),
            ) if v]
            # participation is used only by focuseddrop.
            if participation < 1.0 and method != 'focuseddrop':
                _ignored.append('--participation')
            if _ignored:
                raise ValueError(
                    f"baseline '{method}' does not support {_ignored} "
                    f"(run_id would append tags, creating a phantom ablation).")
            return build_baseline(method, drop_rate, channels=_ch.get(where),
                                  participation=participation)
        return build_sdrop(method, drop_rate, grid_size,
                           num_classes=num_classes, gamma=gamma,
                           peakedness=peakedness, norm=norm,
                           grad_mode=grad_mode,
                           schedule=schedule, participation=participation,
                           rescale=rescale, rescale_mode=rescale_mode,
                           select=select, warmup_frac=warmup_frac)

    if arch not in ('resnet18', 'resnet50'):
        # CIFAR-specific 3-stage backbones (resnet56/110, wrn28_10 ...)
        # Unsupported arguments are not silently ignored: run_id appends tags, so ignoring
        # them would save runs with identical settings under different file names (phantom ablation).
        if width_mult != 1.0:
            raise ValueError(f"'{arch}' does not support width_mult "
                             f"(requested {width_mult}). Use resnet18.")
        if pretrained:
            raise ValueError(f"'{arch}' has no pretrained weights.")
        return SDroCifarNet(arch, num_classes,
                            {'L2': _make('L2'), 'L3': _make('L3'),
                             'L4': _make('L4'),
                             'L1': _make('L1')})

    return SDroResNet(
        arch=arch,
        num_classes=num_classes,
        sdrop_l1=_make('L1'),
        sdrop_l2=_make('L2'),
        sdrop_l3=_make('L3'),
        sdrop_l4=_make('L4'),
        pretrained=pretrained,
        width_mult=width_mult,
    )
