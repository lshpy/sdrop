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
                 pretrained: bool = False):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        if arch == 'resnet18':
            backbone = models.resnet18(weights=weights)
            # CIFAR-100 adaptation: replace 7×7 stride-2 conv with 3×3 stride-1
            # and remove maxpool so spatial resolution is preserved longer
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                       padding=1, bias=False)
            backbone.maxpool = nn.Identity()
        elif arch == 'resnet50':
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported arch: '{arch}'. Use 'resnet18' or 'resnet50'.")

        # replace classifier head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

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

        self.sdrop_l3 = sdrop_l3 if sdrop_l3 is not None else nn.Identity()
        self.sdrop_l4 = sdrop_l4 if sdrop_l4 is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

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
                pretrained: bool = False) -> SDroResNet:
    """
    Convenience builder.

    Args:
        arch        : 'resnet18' | 'resnet50'
        num_classes : number of classes
        method      : 'none' | 'dropout' | 'sdrop' | 'sdrop_energy' | 'sgridlc'
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
    sdrop_l3 = build_sdrop(method, drop_rate, grid_size) if 'L3' in layers else None
    sdrop_l4 = build_sdrop(method, drop_rate, grid_size) if 'L4' in layers else None

    return SDroResNet(
        arch=arch,
        num_classes=num_classes,
        sdrop_l3=sdrop_l3,
        sdrop_l4=sdrop_l4,
        pretrained=pretrained,
    )
