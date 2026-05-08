"""Data loaders for CIFAR-10 and SVHN."""

from __future__ import annotations

import torch
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
