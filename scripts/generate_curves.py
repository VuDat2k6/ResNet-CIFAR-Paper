"""Generate training curve plots from progress.json files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_training_curves() -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    outputs_dir = Path("outputs")

    experiments = {
        "cifar10_original": "CIFAR-10 Original (ReLU+SGD)",
        "cifar10_optimized": "CIFAR-10 Optimized (SiLU+AdamW)",
        "cifar10_seresnet": "CIFAR-10 SE-ResNet-20 (SiLU+SD+CutMix)",
        "svhn_original": "SVHN Original (ReLU+SGD)",
        "svhn_optimized": "SVHN Optimized (SiLU+AdamW)",
        "svhn_seresnet": "SVHN SE-ResNet-20 (SiLU+SD+CutMix)",
    }

    datasets = ["cifar10", "svhn"]

    for dataset in datasets:
        curves = {}
        for key, label in experiments.items():
            if not key.startswith(dataset):
                continue
            progress_path = outputs_dir / key / "progress.json"
            if progress_path.exists():
                with open(progress_path) as f:
                    data = json.load(f)
                    epochs = data.get("epochs", data.get("total_epochs", 0))
                    if epochs == 0:
                        continue
                    # Reconstruct from individual progress.json entries
                    # Each progress.json only has the LAST epoch, not the full history.
                    # We need to load the full training log. Let's check results.json instead.
                    pass

        # Load from results.json + reconstruct from available data
        for key, label in experiments.items():
            if not key.startswith(dataset):
                continue
            progress_path = outputs_dir / key / "progress.json"
            results_path = outputs_dir / key / "results.json"

            if not results_path.exists():
                print(f"Skipping {key} (no results.json)")
                continue

            with open(results_path) as f:
                r = json.load(f)

            epochs = r.get("epochs", 0)
            variant = r.get("variant", key)

            # Load all epoch data from progress.json if it has full history
            # The progress.json only has the last epoch, so we need training logs.
            # Check if training_curves.png already exists
            out_dir = outputs_dir / key
            existing_curve = out_dir / "training_curves.png"
            if existing_curve.exists():
                print(f"  {key}: training_curves.png already exists, skipping")
                continue

            # We don't have full epoch-by-epoch data from progress.json alone.
            # But we can extract what we know and use placeholders for the curve shape.
            # Actually, let's just report what we have.
            print(f"  {key}: epochs={epochs}, best_acc={r.get('best_test_accuracy', 'N/A')}")

    print("\nTraining curve PNGs need to be generated from full training logs.")
    print("Checking for existing curves...")

    for exp_key, label in experiments.items():
        out_dir = outputs_dir / exp_key
        curve_path = out_dir / "training_curves.png"
        lr_path = out_dir / "lr_schedule.png"
        print(f"  {exp_key}: curves={curve_path.exists()}, lr={lr_path.exists()}")


if __name__ == "__main__":
    generate_training_curves()
