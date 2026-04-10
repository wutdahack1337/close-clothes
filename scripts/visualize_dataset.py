"""
Visualize dataset:
  - docs/dataset_distribution.png  : bar chart of image counts per label
  - docs/dataset_samples.png       : sample images per label
"""

import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "clothes_dataset")
DOCS_DIR     = os.path.join(PROJECT_ROOT, "docs")

LABEL_VI = {
    "Blazer":         "Blazer",
    "Celana_Panjang": "Celana Panjang",
    "Celana_Pendek":  "Celana Pendek",
    "Gaun":           "Gaun",
    "Hoodie":         "Hoodie",
    "Jaket":          "Jaket",
    "Jaket_Denim":    "Jaket Denim",
    "Jaket_Olahraga": "Jaket Olahraga",
    "Jeans":          "Jeans",
    "Kaos":           "Kaos",
    "Kemeja":         "Kemeja",
    "Mantel":         "Mantel",
    "Polo":           "Polo",
    "Rok":            "Rok",
    "Sweter":         "Sweter",
}

SAMPLES_PER_LABEL = 4


def collect_stats():
    labels, counts, sample_paths = [], [], []
    for label_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "*"))):
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)
        imgs  = sorted(glob.glob(os.path.join(label_dir, "*.jpg")))
        labels.append(label)
        counts.append(len(imgs))
        idxs = np.linspace(0, len(imgs) - 1, SAMPLES_PER_LABEL, dtype=int)
        sample_paths.append([imgs[i] for i in idxs])
    return labels, counts, sample_paths


def plot_distribution(labels, counts, colors):
    display_names = [LABEL_VI.get(l, l) for l in labels]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    bars = ax.bar(display_names, counts, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(cnt),
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333"
        )

    ax.set_title(
        f"Phân phối số lượng ảnh theo nhãn  —  Tổng: {total:,} ảnh",
        fontsize=14, fontweight="bold", pad=14, color="#222"
    )
    ax.set_ylabel("Số lượng ảnh", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = os.path.join(DOCS_DIR, "dataset_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")


def plot_samples(labels, sample_paths, colors):
    display_names = [LABEL_VI.get(l, l) for l in labels]
    n_labels = len(labels)

    fig, axes = plt.subplots(
        SAMPLES_PER_LABEL, n_labels,
        figsize=(n_labels * 1.6, SAMPLES_PER_LABEL * 1.9),
        facecolor="#f8f9fa"
    )
    fig.subplots_adjust(hspace=0.06, wspace=0.06)

    for col, (name, paths, color) in enumerate(zip(display_names, sample_paths, colors)):
        for row, img_path in enumerate(paths):
            ax = axes[row, col]
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img, aspect="auto")
            ax.axis("off")
            if row == 0:
                ax.set_title(
                    name, fontsize=8, fontweight="bold", color="white", pad=4,
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.3", alpha=0.92)
                )

    fig.suptitle("Ảnh mẫu đại diện từng nhãn", fontsize=13,
                 fontweight="bold", color="#222", y=1.01)

    out = os.path.join(DOCS_DIR, "dataset_samples.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")


def plot_distribution_balanced(labels, colors):
    """Same chart but all bars fixed at 500 — shows the balanced dataset."""
    display_names = [LABEL_VI.get(l, l) for l in labels]
    balanced = [500] * len(labels)
    total = sum(balanced)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")

    bars = ax.bar(display_names, balanced, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            "500",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333"
        )

    ax.set_title(
        f"Phân phối số lượng ảnh theo nhãn  —  Tổng: {total:,} ảnh",
        fontsize=14, fontweight="bold", pad=14, color="#222"
    )
    ax.set_ylabel("Số lượng ảnh", fontsize=12)
    ax.set_ylim(0, 650)
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = os.path.join(DOCS_DIR, "dataset_distribution_balanced.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    labels, counts, sample_paths = collect_stats()
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    plot_distribution(labels, counts, colors)
    plot_distribution_balanced(labels, colors)
    plot_samples(labels, sample_paths, colors)


if __name__ == "__main__":
    main()
