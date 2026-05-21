"""Script to analyze and plot the class distribution of datasets."""

import sys
from pathlib import Path
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import build_loaders

def get_class_counts(loader):
    """Function to count label occurrences quickly by accessing the label arrays directly."""
    dataset = loader.dataset
    
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        labels = [target for _, target in dataset]
    
    counter = collections.Counter(labels)
    counts = [counter[i] for i in range(10)]
    return counts

def main():
    print("Counting CIFAR-10 images...")
    cifar_train, _ = build_loaders("cifar10", batch_size=128, num_workers=0)
    cifar_counts = get_class_counts(cifar_train)
    
    print("Counting SVHN images...")
    svhn_train, _ = build_loaders("svhn", batch_size=128, num_workers=0)
    svhn_counts = get_class_counts(svhn_train)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
   
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Data Distribution: CIFAR-10 vs SVHN (Training Set)", fontsize=16, fontweight="bold", y=1.02)

    axes[0].bar(classes, cifar_counts, color="#2196F3", edgecolor="black", linewidth=0.5)
    axes[0].set_title(f"CIFAR-10 (Total: {sum(cifar_counts):,} images)", fontsize=14)
    axes[0].set_xlabel("Classes", fontsize=12)
    axes[0].set_ylabel("Number of Images", fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].set_ylim(0, max(cifar_counts) + 2000)
    for i, v in enumerate(cifar_counts):
        axes[0].text(i, v + 300, str(v), ha='center', fontsize=10)

    axes[1].bar(classes, svhn_counts, color="#4CAF50", edgecolor="black", linewidth=0.5)
    axes[1].set_title(f"SVHN (Total: {sum(svhn_counts):,} images)", fontsize=14)
    axes[1].set_xlabel("Classes (Digits)", fontsize=12)
    axes[1].set_ylabel("Number of Images", fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].set_ylim(0, max(svhn_counts) + 2000)

    for i, v in enumerate(svhn_counts):
        axes[1].text(i, v + 300, str(v), ha='center', fontsize=10)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / "class_distribution.png"
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n-> SUCCESS! Saved class distribution plot at: {out_path}")

if __name__ == "__main__":
    main()