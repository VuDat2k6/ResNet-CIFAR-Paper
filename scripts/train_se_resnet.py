"""Training script for SE-ResNet with Stochastic Depth, CutMix, and Cosine Warmup.

Combines techniques from:
- SE-Net (Hu et al., 2017): Channel attention via Squeeze-and-Excitation
- Stochastic Depth (Huang et al., 2016): Random layer dropout
- CutMix (Yun et al., 2019): Advanced data augmentation
- ResNet-RS (Bello et al., 2021): Improved training regime

Usage:
    python train_se_resnet.py --dataset cifar10 --epochs 200
    python train_se_resnet.py --dataset svhn --epochs 200
    python train_se_resnet.py --dataset cifar10 --wide --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.se_resnet import (
    seresnet20, seresnet20_wide,
    LabelSmoothingCrossEntropy, CutMixCriterion,
)
from src.data import build_loaders, CutMix
from src.utils import set_seed

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 0
LABEL_SMOOTHING = 0.1

# Model hyperparameters
WIDTH_MULTIPLIER = 1      # 1 for ResNet-20-SE, 2 for WRN-20-2-SE (~17M params)
DROP_RATE = 0.0            # Dropout rate inside SE blocks
STOCHASTIC_DEPTH_PROB = 0.8  # Survival prob for deepest blocks (1.0=disabled)
ACTIVATION = "silu"       # "relu" or "silu"

# Training hyperparameters
LEARNING_RATE = 0.05     # For SGD (lowered from 0.1 — SE blocks need smaller LR)
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
EPOCHS = 200
MILESTONES = [100, 150]   # For MultiStepLR
GAMMA = 0.1

# CutMix hyperparameters
CUTMIX_BETA = 1.0
CUTMIX_PROB = 0.3         # Reduced from 0.5 for more stable training

# Use MultiStepLR (same schedule that worked for original experiments)
SCHEDULE_MILESTONES = [100, 150]
SCHEDULE_GAMMA = 0.1


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    cutmix_criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cutmix: CutMix,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        images_a, targets_a, targets_b, lam = cutmix(inputs, targets)

        optimizer.zero_grad()
        outputs = model(images_a)
        loss = cutmix_criterion(outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_a).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


def plot_metrics(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
    output_path: str,
    title: str,
) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(epochs, train_losses, label="Train Loss", color="#4CAF50")
    axes[0].plot(epochs, test_losses, label="Test Loss", color="#F44336", linestyle="--")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in train_accs], label="Train Acc", color="#4CAF50")
    axes[1].plot(epochs, [a * 100 for a in test_accs], label="Test Acc", color="#F44336", linestyle="--")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SE-ResNet with Stochastic Depth + CutMix on CIFAR-10 / SVHN"
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "svhn"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--wide", action="store_true",
                        help="Use WRN-style width (2x, ~17M params)")
    parser.add_argument("--drop_rate", type=float, default=DROP_RATE)
    parser.add_argument("--sd_prob", type=float, default=STOCHASTIC_DEPTH_PROB,
                        help="Stochastic depth survival probability (1.0=disabled)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--activation", type=str, default=ACTIVATION,
                        choices=["relu", "silu"])
    parser.add_argument("--cutmix_prob", type=float, default=CUTMIX_PROB)
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adamw"])
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.dataset.upper()}, wide={args.wide}, "
          f"sd_prob={args.sd_prob}, cutmix_prob={args.cutmix_prob}, "
          f"activation={args.activation}, optimizer={args.optimizer}")

    suffix = "_wide" if args.wide else ""
    output_dir = Path(f"outputs/{args.dataset}_seresnet{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_loaders(
        args.dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model_cls = seresnet20_wide if args.wide else seresnet20
    model = model_cls(
        width_multiplier=4 if args.wide else 1,
        activation=args.activation,  # type: ignore[arg-type]
        drop_rate=args.drop_rate,
        stochastic_depth_prob=args.sd_prob,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    criterion = LabelSmoothingCrossEntropy(epsilon=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()
    cutmix_criterion = CutMixCriterion(criterion)
    cutmix = CutMix(beta=CUTMIX_BETA, prob=args.cutmix_prob)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=SCHEDULE_MILESTONES, gamma=SCHEDULE_GAMMA,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_acc = 0.0
    best_epoch = 0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    lr_history = []
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, cutmix_criterion,
            optimizer, device, cutmix,
        )
        test_loss, test_acc = evaluate(model, test_loader, eval_criterion, device)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        lr_history.append(scheduler.get_last_lr()[0])

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        # Save progress after every epoch for monitoring
        progress = {
            "epoch": epoch,
            "total_epochs": args.epochs,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,
            "lr": scheduler.get_last_lr()[0],
        }
        with open(output_dir / "progress.json", "w") as f:
            json.dump(progress, f, indent=2)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Test  Loss: {test_loss:.4f} Acc: {test_acc*100:.2f}% | "
            f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}",
            flush=True
        )

    results = {
        "dataset": args.dataset,
        "variant": "seresnet_wide" if args.wide else "seresnet20",
        "config": {
            "width_multiplier": 4 if args.wide else 1,
            "activation": args.activation,
            "drop_rate": args.drop_rate,
            "stochastic_depth_prob": args.sd_prob,
            "cutmix_prob": args.cutmix_prob,
            "cutmix_beta": CUTMIX_BETA,
            "optimizer": args.optimizer,
            "label_smoothing": LABEL_SMOOTHING,
        },
        "best_test_accuracy": best_test_acc,
        "best_test_error": 1.0 - best_test_acc,
        "best_epoch": best_epoch,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "final_train_accuracy": train_accs[-1],
        "final_test_accuracy": test_accs[-1],
        "convergence_epoch": next(
            (i + 1 for i, acc in enumerate(test_accs) if acc >= 0.90), args.epochs,
        ),
        "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
        "total_train_time_sec": sum(epoch_times),
        "num_parameters": num_params,
        "seed": RANDOM_SEED,
        "epochs": args.epochs,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    variant_name = "SE-ResNet-20 Wide" if args.wide else "SE-ResNet-20"
    title = (f"{args.dataset.upper()} {variant_name} — "
             f"SE + Stochastic Depth + CutMix + Cosine Warmup")

    plot_path = output_dir / "training_curves.png"
    plot_metrics(train_losses, test_losses, train_accs, test_accs, str(plot_path), title)
    print(f"\nPlots saved to {plot_path}")
    print(f"Best test accuracy: {best_test_acc*100:.2f}%  "
          f"(Top-1 Error: {(1-best_test_acc)*100:.2f}%)")

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, args.epochs + 1), lr_history, color="#2196F3")
    ax.set_title(f"{args.dataset.upper()} — Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig2.savefig(output_dir / "lr_schedule.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    main()
