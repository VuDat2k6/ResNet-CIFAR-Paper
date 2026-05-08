"""Generate all comparison plots for the ResNet paper report."""

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

    r_co = load_results("outputs/cifar10_original")
    r_so = load_results("outputs/svhn_original")
    r_opt_c = load_results("outputs/cifar10_optimized")
    r_opt_s = load_results("outputs/svhn_optimized")

    # ============================================================
    # Plot 1: CIFAR-10 — Bar Chart Comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "CIFAR-10 ResNet-20: Original (ReLU + SGD) vs Optimized (SiLU + AdamW + Label Smoothing)",
        fontsize=13,
    )

    metrics_c = ["Top-1 Error (%)", "Top-1 Accuracy (%)", "Generalization Gap (%)"]
    gap_co = (r_co["final_train_accuracy"] - r_co["best_test_accuracy"]) * 100
    gap_opt_c = (r_opt_c["final_train_accuracy"] - r_opt_c["best_test_accuracy"]) * 100
    orig_vals_c = [r_co["best_test_error"] * 100, r_co["best_test_accuracy"] * 100, gap_co]
    opt_vals_c = [r_opt_c["best_test_error"] * 100, r_opt_c["best_test_accuracy"] * 100, gap_opt_c]

    x = np.arange(len(metrics_c))
    w = 0.35
    ax0, ax1 = axes

    bars1 = ax0.bar(x - w / 2, orig_vals_c, w, label="Original (ReLU+SGD)", color="#2196F3")
    bars2 = ax0.bar(x + w / 2, opt_vals_c, w, label="Optimized (SiLU+AdamW+LS)", color="#9C27B0")
    ax0.set_xticks(x)
    ax0.set_xticklabels(metrics_c)
    ax0.set_title("CIFAR-10: Key Metrics")
    ax0.legend()
    ax0.grid(True, alpha=0.3, axis="y")
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax0.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    datasets = ["CIFAR-10\nOriginal", "CIFAR-10\nOptimized", "SVHN\nOriginal", "SVHN\nOptimized"]
    errors = [
        r_co["best_test_error"] * 100,
        r_opt_c["best_test_error"] * 100,
        r_so["best_test_error"] * 100,
        r_opt_s["best_test_error"] * 100,
    ]
    colors = ["#2196F3", "#9C27B0", "#4CAF50", "#FF9800"]
    bars = ax1.bar(datasets, errors, color=colors)
    ax1.set_title("Top-1 Error Across All Configurations")
    ax1.set_ylabel("Top-1 Error (%)")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, err in zip(bars, errors):
        ax1.annotate(f"{err:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(plots_dir / "cifar10_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 2: SVHN — Bar Chart Comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "SVHN ResNet-20: Original (ReLU + SGD) vs Optimized (SiLU + AdamW + Label Smoothing)",
        fontsize=13,
    )

    metrics_s = ["Top-1 Error (%)", "Top-1 Accuracy (%)", "Generalization Gap (%)"]
    gap_so = (r_so["final_train_accuracy"] - r_so["best_test_accuracy"]) * 100
    gap_opt_s = (r_opt_s["final_train_accuracy"] - r_opt_s["best_test_accuracy"]) * 100
    orig_vals_s = [r_so["best_test_error"] * 100, r_so["best_test_accuracy"] * 100, gap_so]
    opt_vals_s = [r_opt_s["best_test_error"] * 100, r_opt_s["best_test_accuracy"] * 100, gap_opt_s]

    ax0, ax1 = axes
    bars1 = ax0.bar(x - w / 2, orig_vals_s, w, label="Original (ReLU+SGD)", color="#4CAF50")
    bars2 = ax0.bar(x + w / 2, opt_vals_s, w, label="Optimized (SiLU+AdamW+LS)", color="#FF9800")
    ax0.set_xticks(x)
    ax0.set_xticklabels(metrics_s)
    ax0.set_title("SVHN: Key Metrics")
    ax0.legend()
    ax0.grid(True, alpha=0.3, axis="y")
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax0.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    bars = ax1.bar(datasets, errors, color=colors)
    ax1.set_title("Top-1 Error Across All Configurations")
    ax1.set_ylabel("Top-1 Error (%)")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, err in zip(bars, errors):
        ax1.annotate(f"{err:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(plots_dir / "svhn_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 3: Summary Results Table
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    col_labels = ["Dataset", "Variant", "Top-1 Error", "Top-1 Accuracy", "Final Train Acc", "Conv. Epoch", "Time/Epoch"]
    table_data = [
        ["CIFAR-10", "Original (ReLU+SGD)",
         f"{r_co['best_test_error']*100:.2f}%", f"{r_co['best_test_accuracy']*100:.2f}%",
         f"{r_co['final_train_accuracy']*100:.2f}%", "40",
         f"{r_co['avg_epoch_time_sec']:.0f}s"],
        ["CIFAR-10", "Optimized (SiLU+AdamW+LS)",
         f"{r_opt_c['best_test_error']*100:.2f}%", f"{r_opt_c['best_test_accuracy']*100:.2f}%",
         f"{r_opt_c['final_train_accuracy']*100:.2f}%", "40",
         f"{r_opt_c['avg_epoch_time_sec']:.0f}s"],
        ["SVHN", "Original (ReLU+SGD)",
         f"{r_so['best_test_error']*100:.2f}%", f"{r_so['best_test_accuracy']*100:.2f}%",
         f"{r_so['final_train_accuracy']*100:.2f}%", "4",
         f"{r_so['avg_epoch_time_sec']:.0f}s"],
        ["SVHN", "Optimized (SiLU+AdamW+LS)",
         f"{r_opt_s['best_test_error']*100:.2f}%", f"{r_opt_s['best_test_accuracy']*100:.2f}%",
         f"{r_opt_s['final_train_accuracy']*100:.2f}%", "3",
         f"{r_opt_s['avg_epoch_time_sec']:.0f}s"],
    ]

    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#E3F2FD")

    fig.suptitle("ResNet-20 Experiment Results Summary (random_seed=42)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(plots_dir / "results_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 4: Optimization Delta Comparison
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Optimization Impact: SiLU + AdamW + Label Smoothing", fontsize=13)

    configs = ["CIFAR-10", "SVHN"]
    delta_err = [
        (r_co["best_test_error"] - r_opt_c["best_test_error"]) * 100,
        (r_so["best_test_error"] - r_opt_s["best_test_error"]) * 100,
    ]
    delta_acc = [
        (r_opt_c["best_test_accuracy"] - r_co["best_test_accuracy"]) * 100,
        (r_opt_s["best_test_accuracy"] - r_so["best_test_accuracy"]) * 100,
    ]
    delta_conv = [40 - 40, 4 - 3]

    bars = axes[0].bar(configs, delta_err, color=["#2196F3", "#4CAF50"])
    axes[0].set_title("Top-1 Error Reduction")
    axes[0].set_ylabel("Error Reduction (%)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, delta_err):
        axes[0].annotate(f"{d:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    bars = axes[1].bar(configs, delta_acc, color=["#9C27B0", "#FF9800"])
    axes[1].set_title("Top-1 Accuracy Improvement")
    axes[1].set_ylabel("Accuracy Gain (%)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, delta_acc):
        axes[1].annotate(f"+{d:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    bars = axes[2].bar(configs, delta_conv, color=["#607D8B", "#795548"])
    axes[2].set_title("Convergence Epoch Change")
    axes[2].set_ylabel("Epochs (Orig − Opt)")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, delta_conv):
        axes[2].annotate(f"{d}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "optimization_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"All plots saved to {plots_dir}/")
    print(f"  - cifar10_comparison.png")
    print(f"  - svhn_comparison.png")
    print(f"  - results_table.png")
    print(f"  - optimization_delta.png")


if __name__ == "__main__":
    plot_all_comparisons()
