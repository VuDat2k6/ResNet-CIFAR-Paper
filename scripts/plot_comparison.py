"""Generate all comparison plots for the ResNet paper report.

Includes all 3 configurations: Original, Optimized (Task B), SE-ResNet-20 (Task C).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(output_dir: str) -> dict:
    path = Path(output_dir) / "results.json"
    with open(path) as f:
        return json.load(f)


def plot_all_comparisons() -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Load all results
    r_co = load_results("outputs/cifar10_original")
    r_so = load_results("outputs/svhn_original")
    r_opt_c = load_results("outputs/cifar10_optimized")
    r_opt_s = load_results("outputs/svhn_optimized")
    r_se_c = load_results("outputs/cifar10_seresnet")
    r_se_s = load_results("outputs/svhn_seresnet")

    # ============================================================
    # Plot 1: CIFAR-10 — Bar Chart Comparison (3 variants)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "CIFAR-10 ResNet-20: Original vs Optimized vs SE-ResNet-20",
        fontsize=13,
    )

    metrics_c = ["Top-1 Error (%)", "Top-1 Accuracy (%)", "Generalization Gap (%)"]
    gap_co = (r_co["final_train_accuracy"] - r_co["best_test_accuracy"]) * 100
    gap_opt_c = (r_opt_c["final_train_accuracy"] - r_opt_c["best_test_accuracy"]) * 100
    gap_se_c = (r_se_c["final_train_accuracy"] - r_se_c["best_test_accuracy"]) * 100

    orig_vals_c = [r_co["best_test_error"] * 100, r_co["best_test_accuracy"] * 100, gap_co]
    opt_vals_c = [r_opt_c["best_test_error"] * 100, r_opt_c["best_test_accuracy"] * 100, gap_opt_c]
    se_vals_c = [r_se_c["best_test_error"] * 100, r_se_c["best_test_accuracy"] * 100, gap_se_c]

    x = np.arange(len(metrics_c))
    w = 0.25
    ax0, ax1 = axes

    bars1 = ax0.bar(x - w, orig_vals_c, w, label="Original (ReLU+SGD)", color="#2196F3")
    bars2 = ax0.bar(x, opt_vals_c, w, label="Optimized (Task B)", color="#9C27B0")
    bars3 = ax0.bar(x + w, se_vals_c, w, label="SE-ResNet-20 (Task C)", color="#4CAF50")
    ax0.set_xticks(x)
    ax0.set_xticklabels(metrics_c)
    ax0.set_title("CIFAR-10: Key Metrics")
    ax0.legend()
    ax0.grid(True, alpha=0.3, axis="y")
    for bar in bars1 + bars2 + bars3:
        h = bar.get_height()
        ax0.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    datasets = [
        "CIFAR-10\nOriginal", "CIFAR-10\nOpt (Task B)", "CIFAR-10\nSE (Task C)",
        "SVHN\nOriginal", "SVHN\nOpt (Task B)", "SVHN\nSE (Task C)"
    ]
    errors = [
        r_co["best_test_error"] * 100,
        r_opt_c["best_test_error"] * 100,
        r_se_c["best_test_error"] * 100,
        r_so["best_test_error"] * 100,
        r_opt_s["best_test_error"] * 100,
        r_se_s["best_test_error"] * 100,
    ]
    colors = ["#2196F3", "#9C27B0", "#4CAF50", "#2196F3", "#9C27B0", "#4CAF50"]
    edge_colors = ["#0D47A1"] * 3 + ["#1B5E20"] * 3
    bars = ax1.bar(datasets, errors, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_title("Top-1 Error Across All Configurations")
    ax1.set_ylabel("Top-1 Error (%)")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, max(errors) * 1.2)
    for bar, err in zip(bars, errors):
        ax1.annotate(f"{err:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(plots_dir / "cifar10_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 2: SVHN — Bar Chart Comparison (3 variants)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "SVHN ResNet-20: Original vs Optimized vs SE-ResNet-20",
        fontsize=13,
    )

    metrics_s = ["Top-1 Error (%)", "Top-1 Accuracy (%)", "Generalization Gap (%)"]
    gap_so = (r_so["final_train_accuracy"] - r_so["best_test_accuracy"]) * 100
    gap_opt_s = (r_opt_s["final_train_accuracy"] - r_opt_s["best_test_accuracy"]) * 100
    gap_se_s = (r_se_s["final_train_accuracy"] - r_se_s["best_test_accuracy"]) * 100

    orig_vals_s = [r_so["best_test_error"] * 100, r_so["best_test_accuracy"] * 100, gap_so]
    opt_vals_s = [r_opt_s["best_test_error"] * 100, r_opt_s["best_test_accuracy"] * 100, gap_opt_s]
    se_vals_s = [r_se_s["best_test_error"] * 100, r_se_s["best_test_accuracy"] * 100, gap_se_s]

    ax0, ax1 = axes
    bars1 = ax0.bar(x - w, orig_vals_s, w, label="Original (ReLU+SGD)", color="#2196F3")
    bars2 = ax0.bar(x, opt_vals_s, w, label="Optimized (Task B)", color="#9C27B0")
    bars3 = ax0.bar(x + w, se_vals_s, w, label="SE-ResNet-20 (Task C)", color="#4CAF50")
    ax0.set_xticks(x)
    ax0.set_xticklabels(metrics_s)
    ax0.set_title("SVHN: Key Metrics")
    ax0.legend()
    ax0.grid(True, alpha=0.3, axis="y")
    for bar in bars1 + bars2 + bars3:
        h = bar.get_height()
        ax0.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    bars = ax1.bar(datasets, errors, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_title("Top-1 Error Across All Configurations")
    ax1.set_ylabel("Top-1 Error (%)")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, max(errors) * 1.2)
    for bar, err in zip(bars, errors):
        ax1.annotate(f"{err:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(plots_dir / "svhn_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 3: Summary Results Table (6 rows)
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")

    col_labels = ["Dataset", "Variant", "Top-1 Error", "Top-1 Accuracy", "Final Train Acc", "Best Epoch", "Time/Epoch"]
    table_data = [
        ["CIFAR-10", "Original (ReLU+SGD)",
         f"{r_co['best_test_error']*100:.2f}%", f"{r_co['best_test_accuracy']*100:.2f}%",
         f"{r_co['final_train_accuracy']*100:.2f}%", f"{r_co.get('best_epoch', 'N/A')}",
         f"{r_co['avg_epoch_time_sec']:.0f}s"],
        ["CIFAR-10", "Optimized (Task B)",
         f"{r_opt_c['best_test_error']*100:.2f}%", f"{r_opt_c['best_test_accuracy']*100:.2f}%",
         f"{r_opt_c['final_train_accuracy']*100:.2f}%", f"{r_opt_c.get('best_epoch', 'N/A')}",
         f"{r_opt_c['avg_epoch_time_sec']:.0f}s"],
        ["CIFAR-10", "SE-ResNet-20 (Task C)",
         f"{r_se_c['best_test_error']*100:.2f}%", f"{r_se_c['best_test_accuracy']*100:.2f}%",
         f"{r_se_c['final_train_accuracy']*100:.2f}%", f"{r_se_c.get('best_epoch', 'N/A')}",
         f"{r_se_c['avg_epoch_time_sec']:.0f}s"],
        ["SVHN", "Original (ReLU+SGD)",
         f"{r_so['best_test_error']*100:.2f}%", f"{r_so['best_test_accuracy']*100:.2f}%",
         f"{r_so['final_train_accuracy']*100:.2f}%", f"{r_so.get('best_epoch', 'N/A')}",
         f"{r_so['avg_epoch_time_sec']:.0f}s"],
        ["SVHN", "Optimized (Task B)",
         f"{r_opt_s['best_test_error']*100:.2f}%", f"{r_opt_s['best_test_accuracy']*100:.2f}%",
         f"{r_opt_s['final_train_accuracy']*100:.2f}%", f"{r_opt_s.get('best_epoch', 'N/A')}",
         f"{r_opt_s['avg_epoch_time_sec']:.0f}s"],
        ["SVHN", "SE-ResNet-20 (Task C)",
         f"{r_se_s['best_test_error']*100:.2f}%", f"{r_se_s['best_test_accuracy']*100:.2f}%",
         f"{r_se_s['final_train_accuracy']*100:.2f}%", f"{r_se_s.get('best_epoch', 'N/A')}",
         f"{r_se_s['avg_epoch_time_sec']:.0f}s"],
    ]

    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.4, 1.8)
    row_colors = ["#E3F2FD", "#F3E5F5", "#E8F5E9", "#E3F2FD", "#F3E5F5", "#E8F5E9"]
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(row_colors[row - 1])

    fig.suptitle("ResNet-20 Experiment Results Summary (random_seed=42)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(plots_dir / "results_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 4: Task C (SE-ResNet-20) vs Original Comparison
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Task C (SE-ResNet-20) vs Original: Accuracy Improvement", fontsize=13)

    configs = ["CIFAR-10", "SVHN"]

    # Accuracy improvement (SE vs Original)
    delta_err = [
        (r_co["best_test_error"] - r_se_c["best_test_error"]) * 100,
        (r_so["best_test_error"] - r_se_s["best_test_error"]) * 100,
    ]
    delta_acc = [
        (r_se_c["best_test_accuracy"] - r_co["best_test_accuracy"]) * 100,
        (r_se_s["best_test_accuracy"] - r_so["best_test_accuracy"]) * 100,
    ]
    gap_reduction = [
        (r_co["final_train_accuracy"] - r_co["best_test_accuracy"]) * 100 -
        (r_se_c["final_train_accuracy"] - r_se_c["best_test_accuracy"]) * 100,
        (r_so["final_train_accuracy"] - r_so["best_test_accuracy"]) * 100 -
        (r_se_s["final_train_accuracy"] - r_se_s["best_test_accuracy"]) * 100,
    ]

    bars = axes[0].bar(configs, delta_err, color=["#4CAF50", "#4CAF50"], edgecolor="black")
    axes[0].set_title("Top-1 Error Reduction\n(Original − SE-ResNet-20)")
    axes[0].set_ylabel("Error Reduction (%)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, delta_err):
        axes[0].annotate(f"{d:+.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    bars = axes[1].bar(configs, delta_acc, color=["#2196F3", "#2196F3"], edgecolor="black")
    axes[1].set_title("Top-1 Accuracy Improvement\n(SE-ResNet-20 − Original)")
    axes[1].set_ylabel("Accuracy Change (%)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, delta_acc):
        axes[1].annotate(f"{d:+.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    bars = axes[2].bar(configs, gap_reduction, color=["#FF9800", "#FF9800"], edgecolor="black")
    axes[2].set_title("Generalization Gap Reduction\n(Original − SE-ResNet-20)")
    axes[2].set_ylabel("Gap Reduction (%)")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, gap_reduction):
        axes[2].annotate(f"{d:+.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "seresnet_improvement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 5: Convergence Comparison (grouped bar)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergence Speed: Epochs to Best Accuracy", fontsize=13)

    variants = ["Original", "Optimized\n(Task B)", "SE-ResNet-20\n(Task C)"]
    epochs_c = [r_co.get("best_epoch", 0), r_opt_c.get("best_epoch", 0), r_se_c.get("best_epoch", 0)]
    epochs_s = [r_so.get("best_epoch", 0), r_opt_s.get("best_epoch", 0), r_se_s.get("best_epoch", 0)]

    x = np.arange(len(variants))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, epochs_c, w, label="CIFAR-10", color="#2196F3", edgecolor="black")
    axes[0].set_title("CIFAR-10: Best Epoch")
    axes[0].set_ylabel("Epoch")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, e in zip(bars1, epochs_c):
        axes[0].annotate(f"{e}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

    bars2 = axes[1].bar(x - w/2, epochs_s, w, label="SVHN", color="#4CAF50", edgecolor="black")
    axes[1].set_title("SVHN: Best Epoch")
    axes[1].set_ylabel("Epoch")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, e in zip(bars2, epochs_s):
        axes[1].annotate(f"{e}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "convergence_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"All plots saved to {plots_dir}/")
    print(f"  - cifar10_comparison.png")
    print(f"  - svhn_comparison.png")
    print(f"  - results_table.png")
    print(f"  - seresnet_improvement.png")
    print(f"  - convergence_comparison.png")


if __name__ == "__main__":
    plot_all_comparisons()
