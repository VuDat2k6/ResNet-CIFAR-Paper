"""Unit tests for ResNet implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.resnet import resnet20, resnet32, LabelSmoothingCrossEntropy, _ResidualBlock


class TestResidualBlockShape:
    """TEST 1: Residual Block output shape matches input (identity path preserved)."""

    def test_identity_shape_preserved_stride1(self) -> None:
        block = _ResidualBlock(in_channels=16, out_channels=16, stride=1)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == x.shape, (
            f"Shape mismatch: input {x.shape} vs output {out.shape}"
        )

    def test_identity_shape_preserved_stride2(self) -> None:
        block = _ResidualBlock(in_channels=16, out_channels=32, stride=2)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        expected = (2, 32, 16, 16)
        assert out.shape == expected, (
            f"Expected shape {expected}, got {out.shape}"
        )

    def test_projection_shortcut_channels_change(self) -> None:
        block = _ResidualBlock(in_channels=3, out_channels=64, stride=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        expected = (2, 64, 32, 32)
        assert out.shape == expected


class TestForwardPass:
    """TEST 2: Forward pass smoke test — correct output shape, no NaN."""

    def test_resnet20_output_shape(self) -> None:
        model = resnet20(activation="relu")
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_resnet32_output_shape(self) -> None:
        model = resnet32(activation="relu")
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_resnet20_no_nan(self) -> None:
        model = resnet20(activation="relu")
        x = torch.randn(8, 3, 32, 32)
        out = model(x)
        assert not torch.isnan(out).any(), "NaN detected in output tensor"

    def test_resnet20_silu_no_nan(self) -> None:
        model = resnet20(activation="silu")
        x = torch.randn(8, 3, 32, 32)
        out = model(x)
        assert not torch.isnan(out).any(), "NaN detected in SiLU model output"


class TestLossDecrease:
    """TEST 3: Sanity check — loss decreases over 3 iterations."""

    def test_loss_decreases_original(self) -> None:
        model = resnet20(activation="relu")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4,
        )

        losses = []
        x = torch.randn(16, 3, 32, 32)
        targets = torch.randint(0, 10, (16,))
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[2] < losses[0], (
            f"Loss did not decrease: iter0={losses[0]:.4f}, iter2={losses[2]:.4f}"
        )

    def test_loss_decreases_optimized(self) -> None:
        model = resnet20(activation="silu")
        criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        losses = []
        x = torch.randn(16, 3, 32, 32)
        targets = torch.randint(0, 10, (16,))
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: iter0={losses[0]:.4f}, iter4={losses[-1]:.4f}"
        )


class TestShortcutIntegrity:
    """TEST 4: Shortcut connection integrity — weights exist and require grad."""

    def test_shortcut_requires_grad(self) -> None:
        block = _ResidualBlock(in_channels=16, out_channels=32, stride=2)
        for name, param in block.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"

    def test_shortcut_params_not_none(self) -> None:
        block = _ResidualBlock(in_channels=16, out_channels=32, stride=2)
        shortcut_params = {n: p for n, p in block.named_parameters() if "shortcut" in n}
        assert len(shortcut_params) > 0, "No shortcut parameters found"
        for name, param in shortcut_params.items():
            assert param is not None, f"Shortcut parameter {name} is None"


class TestLabelSmoothing:
    """TEST: Label Smoothing loss produces valid outputs."""

    def test_label_smoothing_output(self) -> None:
        criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = criterion(logits, targets)
        assert not torch.isnan(loss), "Label smoothing loss is NaN"
        assert loss.item() > 0, "Loss should be positive"
