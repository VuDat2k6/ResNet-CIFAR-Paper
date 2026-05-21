"""ResNet-20/32 CIFAR-variant implementation in PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class _ResidualBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: Literal["relu", "silu"] = "relu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        act_fn = nn.SiLU() if activation == "silu" else nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = act_fn

        shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            shortcut = nn.Identity()
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.act(out)


class _ResNetCIFAR(nn.Module):
    """ResNet for CIFAR-10/SVHN (32x32 RGB input)."""

    def __init__(
        self,
        num_blocks: list[int],
        num_classes: int = 10,
        activation: Literal["relu", "silu"] = "relu",
    ) -> None:
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        act_fn = "silu" if activation == "silu" else "relu"
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1, activation=act_fn)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2, activation=act_fn)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2, activation=act_fn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        activation: str,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                _ResidualBlock(
                    self.in_channels, out_channels, s,
                    activation=activation,  # type: ignore[arg-type]
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet20(activation: Literal["relu", "silu"] = "relu") -> nn.Module:
    """ResNet-20 for CIFAR (3 stages of 3 blocks each)."""
    return _ResNetCIFAR([3, 3, 3], num_classes=10, activation=activation)


def resnet32(activation: Literal["relu", "silu"] = "relu") -> nn.Module:
    """ResNet-32 for CIFAR (3 stages of 5 blocks each)."""
    return _ResNetCIFAR([5, 5, 5], num_classes=10, activation=activation)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        loss = -log_probs.sum(dim=1).mean()
        nll = F.nll_loss(log_probs, target)
        return self.epsilon * (loss / n_classes) + (1 - self.epsilon) * nll
