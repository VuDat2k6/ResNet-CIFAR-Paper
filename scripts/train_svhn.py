"""Training script for ResNet-20 on SVHN (cross-dataset evaluation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.resnet import resnet20
from src.data import build_loaders
from src.utils import set_seed

RANDOM_SEED = 42
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 200
MILESTONES = [100, 150]
GAMMA = 0.1
NUM_WORKERS = 0


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

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
) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SVHN ResNet-20 Original — Training Curves", fontsize=14)

    axes[0].plot(epochs, train_losses, label="Train Loss", color="#4CAF50")
    axes[0].plot(epochs, test_losses, label="Test Loss", color="#FF9800", linestyle="--")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in train_accs], label="Train Acc", color="#4CAF50")
    axes[1].plot(epochs, [a * 100 for a in test_accs], label="Test Acc", color="#FF9800", linestyle="--")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-20 on SVHN")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/svhn_original")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_loaders(
        "svhn", batch_size=args.batch_size, num_workers=NUM_WORKERS,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = resnet20(activation="relu").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=GAMMA,
    )

    import time
    best_test_acc = 0.0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                f"Test  Loss: {test_loss:.4f} Acc: {test_acc*100:.2f}% | "
                f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    results = {
        "dataset": "svhn",
        "variant": "original",
        "best_test_accuracy": best_test_acc,
        "best_test_error": 1.0 - best_test_acc,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "final_train_accuracy": train_accs[-1],
        "final_test_accuracy": test_accs[-1],
        "convergence_epoch": next(
            (i + 1 for i, acc in enumerate(test_accs) if acc >= 0.90), args.epochs,
        ),
        "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
        "total_train_time_sec": sum(epoch_times),
        "seed": RANDOM_SEED,
        "epochs": args.epochs,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_path = output_dir / "training_curves.png"
    plot_metrics(train_losses, test_losses, train_accs, test_accs, str(plot_path))
    print(f"\nPlots saved to {plot_path}")
    print(f"Best test accuracy: {best_test_acc*100:.2f}%  (Top-1 Error: {(1-best_test_acc)*100:.2f}%)")


if __name__ == "__main__":
    main()
