"""ResNet-RS: SE-ResNet with Stochastic Depth for CIFAR.

Combines techniques from:
- SE-Net (Hu et al., 2017): Squeeze-and-Excitation channel attention
- Stochastic Depth (Huang et al., 2016): Random layer dropout during training
- ResNet-RS (Bello et al., 2021): Improved training regime
- ConvNeXt (Liu et al., 2022): LayerScale for training stability
- Bag of Tricks (2018): GELU, CutMix, MixUp augmentation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    From: "Squeeze-and-Excitation Networks" (Hu et al., CVPR 2018).
    Adaptively recalibrates channel-wise feature responses by learning
    the importance of each channel.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.excitation.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# Pre-activation Residual Block with SE, Stochastic Depth, and LayerScale
# ---------------------------------------------------------------------------

class SEResBlock(nn.Module):
    """Pre-activation residual block with SE attention, Stochastic Depth, and LayerScale.

    From: ConvNeXt (Liu et al., 2022) — LayerScale improves training stability.
    GELU from: Gaussian Error Linear Units (Hendrycks & Gimpel, 2017).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se_reduction: int = 16,
        activation: Literal["relu", "silu", "gelu"] = "relu",
        drop_rate: float = 0.0,
        survival_prob: float = 1.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.survival_prob = survival_prob
        self.drop_rate = drop_rate

        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = act_fn
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.se = SEBlock(out_channels, reduction=se_reduction)

        # LayerScale: learnable per-channel scalar (from ConvNeXt)
        self.layer_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * layer_scale_init)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.se(out)
        out = out * self.layer_scale  # LayerScale

        if self.training and self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=True)

        # Stochastic depth
        if self.training and self.survival_prob < 1.0:
            if torch.rand(1).item() >= self.survival_prob:
                return identity * self.survival_prob

        out = out + identity
        return out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class _SEResNetCIFAR(nn.Module):
    """SE-ResNet for CIFAR (32x32 RGB input) with SE, SD, and LayerScale."""

    def __init__(
        self,
        num_blocks: list[int],
        num_classes: int = 10,
        width_multiplier: int = 1,
        se_reduction: int = 16,
        activation: Literal["relu", "silu", "gelu"] = "relu",
        drop_rate: float = 0.0,
        stochastic_depth_prob: float = 1.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        assert len(num_blocks) == 3

        base_channels = 64
        self.in_channels = base_channels * width_multiplier

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(
            base_channels * width_multiplier, num_blocks[0],
            stride=1, se_reduction=se_reduction, activation=activation,
            drop_rate=drop_rate,
            stochastic_depth_prob=stochastic_depth_prob,
            block_idx_start=0, num_total=sum(num_blocks),
            layer_scale_init=layer_scale_init,
        )

        self.layer2 = self._make_layer(
            base_channels * width_multiplier * 2, num_blocks[1],
            stride=2, se_reduction=se_reduction, activation=activation,
            drop_rate=drop_rate,
            stochastic_depth_prob=stochastic_depth_prob,
            block_idx_start=num_blocks[0], num_total=sum(num_blocks),
            layer_scale_init=layer_scale_init,
        )

        self.layer3 = self._make_layer(
            base_channels * width_multiplier * 4, num_blocks[2],
            stride=2, se_reduction=se_reduction, activation=activation,
            drop_rate=drop_rate,
            stochastic_depth_prob=stochastic_depth_prob,
            block_idx_start=num_blocks[0] + num_blocks[1], num_total=sum(num_blocks),
            layer_scale_init=layer_scale_init,
        )

        self.bn_final = nn.BatchNorm2d(base_channels * width_multiplier * 4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * width_multiplier * 4, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        se_reduction: int,
        activation: str,
        drop_rate: float,
        stochastic_depth_prob: float,
        block_idx_start: int,
        num_total: int,
        layer_scale_init: float,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, s in enumerate(strides):
            block_idx = block_idx_start + i
            linear_survival = 1.0 - (block_idx / num_total) * (1.0 - stochastic_depth_prob)

            layers.append(
                SEResBlock(
                    self.in_channels, out_channels, s,
                    se_reduction=se_reduction,
                    activation=activation,  # type: ignore[arg-type]
                    drop_rate=drop_rate,
                    survival_prob=linear_survival,
                    layer_scale_init=layer_scale_init,
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn_final(out)
        out = F.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# Model Factories
# ---------------------------------------------------------------------------

def seresnet20(
    width_multiplier: int = 1,
    activation: Literal["relu", "silu"] = "relu",
    drop_rate: float = 0.0,
    stochastic_depth_prob: float = 1.0,
) -> nn.Module:
    """SE-ResNet-20 for CIFAR with configurable width and regularization."""
    return _SEResNetCIFAR(
        num_blocks=[3, 3, 3],
        num_classes=10,
        width_multiplier=width_multiplier,
        se_reduction=16,
        activation=activation,
        drop_rate=drop_rate,
        stochastic_depth_prob=stochastic_depth_prob,
        layer_scale_init=1e-5,
    )


def seresnet32(
    width_multiplier: int = 1,
    activation: Literal["relu", "silu"] = "relu",
    drop_rate: float = 0.0,
    stochastic_depth_prob: float = 1.0,
) -> nn.Module:
    """SE-ResNet-32 for CIFAR."""
    return _SEResNetCIFAR(
        num_blocks=[5, 5, 5],
        num_classes=10,
        width_multiplier=width_multiplier,
        se_reduction=16,
        activation=activation,
        drop_rate=drop_rate,
        stochastic_depth_prob=stochastic_depth_prob,
        layer_scale_init=1e-5,
    )


def seresnet20_wide(
    width_multiplier: int = 2,
    activation: Literal["relu", "silu"] = "relu",
    drop_rate: float = 0.3,
    stochastic_depth_prob: float = 0.5,
) -> nn.Module:
    """Wide SE-ResNet-20 (WRN-style) for CIFAR."""
    return _SEResNetCIFAR(
        num_blocks=[3, 3, 3],
        num_classes=10,
        width_multiplier=width_multiplier,
        se_reduction=16,
        activation=activation,
        drop_rate=drop_rate,
        stochastic_depth_prob=stochastic_depth_prob,
        layer_scale_init=1e-5,
    )


def seresnet20_v2(
    width_multiplier: int = 1,
    activation: Literal["relu", "silu", "gelu"] = "gelu",
    drop_rate: float = 0.0,
    stochastic_depth_prob: float = 0.8,
    layer_scale_init: float = 1e-5,
) -> nn.Module:
    """SE-ResNet-20 v2: Bag of Tricks + ConvNeXt enhancements.

    Adds:
    - GELU activation (from BERT, ConvNeXt)
    - LayerScale (from ConvNeXt) with configurable init value
    - Stochastic Depth (survival_prob=0.8 default)
    - SE channel attention

    Args:
        width_multiplier: Multiply base channels.
        activation: "relu", "silu", or "gelu".
        drop_rate: Dropout rate inside SE blocks.
        stochastic_depth_prob: Survival probability (1.0=disabled).
        layer_scale_init: Initial value for LayerScale params (1e-5 from ConvNeXt-tiny).
    """
    return _SEResNetCIFAR(
        num_blocks=[3, 3, 3],
        num_classes=10,
        width_multiplier=width_multiplier,
        se_reduction=16,
        activation=activation,
        drop_rate=drop_rate,
        stochastic_depth_prob=stochastic_depth_prob,
        layer_scale_init=layer_scale_init,
    )


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing (epsilon).

    From: "Rethinking the Inception Architecture" (Szegedy et al., 2016).
    Prevents overconfident predictions, improves generalization.
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        loss = -log_probs.sum(dim=1).mean()
        nll = F.nll_loss(log_probs, target)
        return self.epsilon * (loss / n_classes) + (1 - self.epsilon) * nll


class CutMixCriterion(nn.Module):
    """Wrapper loss for CutMix that handles mixed labels."""

    def __init__(self, base_criterion: nn.Module) -> None:
        super().__init__()
        self.base_criterion = base_criterion

    def forward(
        self,
        pred: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        return lam * self.base_criterion(pred, targets_a) + \
               (1 - lam) * self.base_criterion(pred, targets_b)


class MixUpCutMixCriterion(nn.Module):
    """Combined loss for MixUp + CutMix that handles both mixed labels."""

    def __init__(self, base_criterion: nn.Module) -> None:
        super().__init__()
        self.base_criterion = base_criterion

    def forward(
        self,
        pred: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        return lam * self.base_criterion(pred, targets_a) + \
               (1 - lam) * self.base_criterion(pred, targets_b)
