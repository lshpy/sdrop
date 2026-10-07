"""
ResNet backbone with SDrop inserted at Layer 3 and/or Layer 4.

Supported architectures:
  resnet18  → CIFAR-100  (modified: 3×3 conv1, no maxpool)
  resnet50  → TinyImageNet, CUB-200-2011  (standard)

SDrop is applied *after* the residual block output, before passing
to the next stage — matching the paper's insertion points L3 and L4.
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
# CIFAR 전용 ResNet (He et al. 2016, 3-stage) 과 WideResNet
#
# FocusedDropout, DropBlock 등이 쓰는 표준 CIFAR 백본이다. torchvision 의
# ResNet-18/50 은 ImageNet 용이라 32x32 에는 과하고, 이 분야 논문들과
# 직접 비교가 안 된다. ResNet-56/110 은 16/32/64 채널의 얇고 깊은 구조로
# "채널 희소성" 조건이 ResNet-18(64~512채널)과 크게 다르다 — SDrop 의 독점
# 가설을 검증하기에 오히려 더 적합한 조건이다.
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
    """WideResNet 의 pre-activation 블록."""
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
    """(stem, [stage...], head_dim) 를 돌려준다. stage 는 nn.Sequential."""
    if arch.startswith('resnet') and arch not in ('resnet18', 'resnet50'):
        depth = int(arch.replace('resnet', ''))
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR ResNet depth 는 6n+2 여야 한다: '{arch}'")
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

    if arch.startswith('wrn'):                      # 예: wrn28_10
        body = arch.replace('wrn', '')
        depth, widen = (int(v) for v in body.split('_'))
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth 는 6n+4 여야 한다: '{arch}'")
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
    """3-stage CIFAR 백본 + SDrop.

    삽입 지점은 **뒤에서부터** 매핑한다: L4 = 마지막 스테이지, L3 = 그 앞,
    L2 = 그 앞. 3-stage 구조라 L1 은 존재하지 않는다 (요청하면 오류).
    이렇게 해야 ResNet-18(4-stage)의 L4 와 ResNet-56(3-stage)의 L4 가
    둘 다 "가장 깊은 스테이지"라는 같은 뜻을 갖는다.
    """

    def __init__(self, arch, num_classes, sdrop_modules: dict):
        super().__init__()
        stem, stages, dim = _make_cifar_stages(arch, num_classes)
        self.stem = stem
        self.stages = nn.ModuleList(stages)
        n = len(stages)
        names = [f'L{4 - (n - 1 - i)}' for i in range(n)]      # 뒤에서부터 L4
        self.stage_names = names
        for want in sdrop_modules:
            if sdrop_modules[want] is not None and want not in names:
                raise ValueError(
                    f"'{arch}' 는 {n}-stage 라 삽입 지점이 {names} 뿐이다 "
                    f"('{want}' 요청됨).")
        # WRN 은 pre-activation 이라 마지막 스테이지를 제외한 스테이지 출력이
        # raw signed pre-activation 이다(실측 음수 비율 0.52). post-ReLU 를 가정한
        # E_c = 1-(1+gamma*Sigma_c)^-delta 와 공간 집중도 P_c 가 다른 것을 재게 되어,
        # 같은 --layers 플래그가 아키텍처마다 다른 개입이 된다(부호검정 무효).
        # 마지막 스테이지는 위에서 BN+ReLU 를 덧붙였으므로 post-ReLU 다(음수 0.00).
        if arch.startswith('wrn'):
            bad = [w for w in sdrop_modules
                   if sdrop_modules[w] is not None and w != names[-1]]
            if bad:
                raise ValueError(
                    f"'{arch}' 는 pre-activation 이라 {bad} 지점의 입력이 raw "
                    f"pre-activation 이다(음수 52%). post-ReLU 를 가정하는 SDrop "
                    f"점수와 맞지 않아 다른 아키텍처와 비교할 수 없다. "
                    f"'{names[-1]}' 만 사용하라.")
        self.drops = nn.ModuleList(
            [sdrop_modules.get(nm) or nn.Identity() for nm in names])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        """He 초기화. PyTorch 기본값(kaiming_uniform_, a=sqrt(5))은 conv 표준편차를
        He fan_out 기준의 0.41배로 만든다. 110층에서는 이 차이가 학습 자체를
        망가뜨려 SDrop 효과와 구분되지 않는다.

        삽입 모듈(self.drops)은 건드리지 않는다 — SE/CBAM 은 고유 초기화가 있고,
        그 fc 는 bias=False 라 무조건 zeros_ 하면 크래시한다."""
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
        elif arch in ('resnet50', 'resnet101', 'resnet152', 'wide_resnet101_2'):
            if width_mult != 1.0:
                raise ValueError('width_mult is only implemented for resnet18')
            # 101/152 는 50 과 같은 bottleneck 구조(L4 = 2048ch, post-ReLU)라
            # SDrop 삽입 지점·점수가 그대로 성립한다. ConvNeXt 등은 스테이지
            # 출력이 post-ReLU 가 아니라서 여기 넣지 않는다(pre-activation 거부와 같은 이유).
            backbone = getattr(models, arch)(
                weights='IMAGENET1K_V1' if pretrained else None)
            backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported arch: '{arch}'. Use 'resnet18' or 'resnet50/101/152'.")

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

        # L1/L2 도 삽입 지점으로 연다. 드롭이 L4(=GAP 직전)에만 걸리면
        # 평균 풀링이 마스크 효과를 흡수해 훈련 손실이 거의 변하지 않는다.
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

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate (GAP) features, for RSC (train.py --rsc_f): forward without the classifier."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.sdrop_l1(self.layer1(x)); x = self.sdrop_l2(self.layer2(x))
        x = self.sdrop_l3(self.layer3(x)); x = self.sdrop_l4(self.layer4(x))
        return torch.flatten(self.avgpool(x), 1)


def build_model(arch: str, num_classes: int, method: str,
                drop_rate: float, layers: list, grid_size: int = 2,
                peakedness: str = 'max', norm: str = 'mean',
                gamma: float = 1.0, delta: float = 1.0, grad_mode: str = 'off',
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
    # SE/CBAM 은 채널 수를 알아야 optimizer 생성 전에 파라미터를 만들 수 있다.
    if arch in ('resnet18',):
        _ch = {'L1': 64, 'L2': 128, 'L3': 256, 'L4': 512}
    elif arch in ('resnet50', 'resnet101', 'resnet152', 'wide_resnet101_2'):
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
            # Guided Dropout 은 자체 스케줄(gdtopk 교대)을 갖는다. 'const' 를 넘기면
            # 원논문과 달리 SDrop 과 같은 예산으로 계속 거는 공정 비교가 된다.
            _bad = [n for n, v in (
                ('--rescale', bool(rescale)), ('--select', select != 'bernoulli'),
                ('--peakedness', peakedness != 'max'), ('--grad_mode', grad_mode != 'off'),
                ('--participation', participation < 1.0)) if v]
            if _bad:
                raise ValueError(f"'{method}' 는 {_bad} 를 지원하지 않는다.")
            return build_baseline(method, drop_rate, channels=_ch.get(where),
                                  schedule=schedule, warmup_frac=warmup_frac)
        if method in ('dropblock', 'senet', 'cbam', 'wcd', 'focuseddrop', 'targeted', 'maxdrop', 'maxdrop_ch'):
            # comparison baselines from Springer ML extended version
            # 이 모듈들은 SDrop 전용 플래그를 받지 않는다. 조용히 무시하면
            # run_id 가 태그를 붙여, 설정이 동일한 런이 서로 다른 파일명으로
            # 저장된다(유령 어블레이션). 명시적으로 거부한다.
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
            # participation 은 focuseddrop 만 쓴다.
            if participation < 1.0 and method != 'focuseddrop':
                _ignored.append('--participation')
            if _ignored:
                raise ValueError(
                    f"baseline '{method}' 는 {_ignored} 를 지원하지 않는다 "
                    f"(run_id 는 태그를 붙이므로 유령 어블레이션이 된다).")
            return build_baseline(method, drop_rate, channels=_ch.get(where),
                                  participation=participation)
        return build_sdrop(method, drop_rate, grid_size,
                           num_classes=num_classes, gamma=gamma, delta=delta,
                           peakedness=peakedness, norm=norm,
                           grad_mode=grad_mode,
                           schedule=schedule, participation=participation,
                           rescale=rescale, rescale_mode=rescale_mode,
                           select=select, warmup_frac=warmup_frac)

    if arch not in ('resnet18', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet101_2'):
        # CIFAR 전용 3-stage 백본 (resnet56/110, wrn28_10 ...)
        # 지원하지 않는 인자는 조용히 무시하지 않는다 — run_id 는 태그를 붙이므로
        # 무시하면 설정이 동일한 런이 서로 다른 파일명으로 저장된다(유령 어블레이션).
        if width_mult != 1.0:
            raise ValueError(f"'{arch}' 는 width_mult 를 지원하지 않는다 "
                             f"(요청값 {width_mult}). resnet18 을 쓰라.")
        if pretrained:
            raise ValueError(f"'{arch}' 에는 pretrained 가중치가 없다.")
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
