"""Training script for Knowledge Distillation (KD) on CIFAR-10.

Distills a pre-trained high-performance seresnet20 teacher (~93.11% accuracy) 
into a lightweight resnet20 student (~91.93% baseline).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.resnet import resnet20
from src.models.se_resnet import seresnet20
from src.data import build_loaders
from src.utils import set_seed

RANDOM_SEED = 42
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
EPOCHS = 200
NUM_WORKERS = 0

# KD Hyperparameters
TEMPERATURE = 4.0
ALPHA = 0.6


class KDLoss(nn.Module):
    """Hinton's Knowledge Distillation Loss."""

    def __init__(self, alpha: float = 0.6, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Standard Cross-Entropy loss on hard targets
        hard_loss = self.ce_loss(student_logits, targets)
        
        # KL-Divergence loss on soft targets (logits scaled by Temperature)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        return (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss


def train_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    criterion: KDLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    student.train()
    teacher.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Get soft labels from the teacher under no_grad to save memory
        with torch.no_grad():
            teacher_logits = teacher(inputs)
            
        student_logits = student(inputs)
        loss = criterion(student_logits, teacher_logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
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
    fig.suptitle("CIFAR-10 ResNet-20 Student (KD-Distilled) — Training Curves", fontsize=14)

    axes[0].plot(epochs, train_losses, label="Train KD Loss", color="#9C27B0")
    axes[0].plot(epochs, test_losses, label="Test CE Loss", color="#F44336", linestyle="--")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in train_accs], label="Train Student Acc", color="#9C27B0")
    axes[1].plot(epochs, [a * 100 for a in test_accs], label="Test Student Acc", color="#F44336", linestyle="--")
    axes[1].axhline(y=91.93, color="#2196F3", linestyle=":", label="Baseline ResNet-20 (91.93%)")
    axes[1].axhline(y=93.11, color="#4CAF50", linestyle="-.", label="Teacher seresnet20 (93.11%)")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Distillation on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="KD alpha weight")
    parser.add_argument("--temp", type=float, default=TEMPERATURE, help="KD soft label temperature")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/cifar10_kd")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_loader, test_loader = build_loaders(
        "cifar10", batch_size=args.batch_size, num_workers=NUM_WORKERS,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 2. Build Models
    # Initialize high-capacity teacher seresnet20
    teacher = seresnet20(
        width_multiplier=1,
        activation="silu",
        drop_rate=0.0,
        stochastic_depth_prob=0.8,
    ).to(device)
    
    # Load teacher weights
    teacher_path = Path("outputs/cifar10_seresnet/best_model.pth")
    if teacher_path.exists():
        print(f"Loading teacher weights from {teacher_path}...")
        # Load with strict=False because layer_scale might be missing from the original checkpoint
        missing_keys, unexpected_keys = teacher.load_state_dict(
            torch.load(teacher_path, map_location=device), strict=False
        )
        if missing_keys:
            print(f"Teacher loaded with missing keys: {missing_keys}")
            # If layer_scale was missing, initialize it to 1.0 (original teacher was trained without LayerScale, meaning scale was 1.0)
            for name, param in teacher.named_parameters():
                if "layer_scale" in name:
                    print(f"Initializing missing key {name} to 1.0 (identity) for teacher compatibility.")
                    nn.init.constant_(param, 1.0)
        if unexpected_keys:
            print(f"Teacher loaded with unexpected keys: {unexpected_keys}")
    else:
        print(f"WARNING: Teacher model checkpoint not found at {teacher_path}!")
        print("We will proceed using an untrained teacher model (random initialization).")
        
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Initialize compact student resnet20
    student = resnet20(activation="relu").to(device)

    num_student_params = sum(p.numel() for p in student.parameters())
    num_teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Student parameters: {num_student_params:,}")
    print(f"Teacher parameters: {num_teacher_params:,} (compression ratio: {num_teacher_params/num_student_params:.1f}x)")

    # 3. Criterion, Optimizer, Scheduler
    kd_criterion = KDLoss(alpha=args.alpha, temperature=args.temp)
    ce_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Linear Warmup scheduler + Cosine Annealing scheduler
    warmup_epochs = 5
    
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        import math
        progress = float(epoch - warmup_epochs) / float(args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 4. Training loop
    best_test_acc = 0.0
    best_epoch = 0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    epoch_times = []

    print("\nStarting Knowledge Distillation Training...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            student, teacher, train_loader, kd_criterion, optimizer, device,
        )
        test_loss, test_acc = evaluate(student, test_loader, ce_criterion, device)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(student.state_dict(), output_dir / "best_model.pth")

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

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                f"Test  Loss: {test_loss:.4f} Acc: {test_acc*100:.2f}% | "
                f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}",
                flush=True
            )

    results = {
        "dataset": "cifar10",
        "variant": "resnet20_kd",
        "config": {
            "temperature": args.temp,
            "alpha": args.alpha,
            "optimizer": "sgd",
            "base_lr": args.lr,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": warmup_epochs,
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
