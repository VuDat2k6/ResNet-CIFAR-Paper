"""Data loaders for CIFAR-10 and SVHN with CutMix augmentation."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(dataset: str) -> transforms.Compose:
    """Return train and test transforms for the given dataset."""
    normalize_mean = [0.4914, 0.4822, 0.4465]
    normalize_std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    return train_transform, test_transform


def get_svhn_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return train and test transforms for SVHN."""
    normalize_mean = [0.4377, 0.4438, 0.4728]
    normalize_std = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    return train_transform, test_transform


def build_loaders(
    dataset: str,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test dataloaders for the specified dataset."""
    if dataset == "cifar10":
        train_tf, test_tf = get_transforms("cifar10")
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_tf,
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_tf,
        )
    elif dataset == "svhn":
        train_tf, test_tf = get_svhn_transforms()
        train_dataset = datasets.SVHN(
            root=data_root, split="train", download=True, transform=train_tf,
        )
        test_dataset = datasets.SVHN(
            root=data_root, split="test", download=True, transform=test_tf,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# CutMix Augmentation
# ---------------------------------------------------------------------------

class CutMix:
    """CutMix data augmentation.

    From: "CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features" (Yun et al., ICCV 2019).

    Cuts rectangular patches from training images and pastes them into
    other images. Labels are mixed proportionally to the patch area.
    This forces the network to learn from less discriminative regions,
    improving generalization and localization.
    """

    def __init__(self, beta: float = 1.0, prob: float = 0.5) -> None:
        self.beta = beta
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix to a batch of images.

        Returns:
            mixed_images: CutMix-applied images
            targets_a: original targets
            targets_b: shuffled targets
            lam: actual lambda (adjusted by patch area)
        """
        if random.random() > self.prob:
            return images, targets, targets, 1.0

        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)
        targets_b = targets[indices]

        lam = random.betavariate(self.beta, self.beta)
        lam = max(lam, 1.0 - lam)

        _, _, h, w = images.size()
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = random.randint(0, w)
        cy = random.randint(0, h)

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(w, cx + cut_w // 2)
        bby2 = min(h, cy + cut_h // 2)

        images_mixed = images.clone()
        images_mixed[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]

        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1)) / (w * h)

        return images_mixed, targets, targets_b, lam


class MixUp:
    """MixUp data augmentation.

    From: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., ICLR 2018).

    Linearly interpolates between pairs of images and their labels.
    Improves calibration and reduces overfitting.
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5) -> None:
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp to a batch of images.

        Returns:
            mixed_images: MixUp-applied images
            targets_a: original targets
            targets_b: shuffled targets
            lam: mixing coefficient
        """
        if random.random() > self.prob or self.alpha <= 0:
            return images, targets, targets, 1.0

        lam = random.betavariate(self.alpha, self.alpha)
        lam = max(lam, 1.0 - lam)

        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)

        images_mixed = lam * images + (1.0 - lam) * images[indices]
        targets_a = targets
        targets_b = targets[indices]

        return images_mixed, targets_a, targets_b, lam
