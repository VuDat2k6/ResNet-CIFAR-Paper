"""Script to visualize CIFAR-10 and SVHN datasets."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import build_loaders

def imshow(img, title, ax):
    """Helper function to convert an image tensor to displayable format."""
    img = img - img.min()
    img = img / img.max()
    
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")

def main():
    print("Loading CIFAR-10 data...")
    cifar_train, _ = build_loaders("cifar10", batch_size=16, num_workers=0)
    
    print("Loading SVHN data...")
    svhn_train, _ = build_loaders("svhn", batch_size=16, num_workers=0)

    cifar_images, _ = next(iter(cifar_train))
    svhn_images, _ = next(iter(svhn_train))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Data Visualization: CIFAR-10 vs SVHN", fontsize=18, fontweight="bold", y=1.05)

    cifar_grid = torchvision.utils.make_grid(cifar_images, nrow=4, padding=2)
    svhn_grid = torchvision.utils.make_grid(svhn_images, nrow=4, padding=2)

    imshow(cifar_grid, "CIFAR-10 (10 Classes of Objects)", axes[0])
    imshow(svhn_grid, "SVHN (Street View House Numbers)", axes[1])

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / "dataset_samples.png"
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"-> SUCCESS! Saved dataset samples visualization at: {out_path}")

if __name__ == "__main__":
    main()