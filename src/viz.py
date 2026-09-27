"""Matplotlib visualizations: training curves, confusion matrices, tracks."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["val_acc"], label="val accuracy")
    axes[1].plot(history["val_f1"], label="val F1 (macro)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.suptitle(title)
    _save(fig, path)


def plot_confusion_matrix(cm: list, class_names: list, title: str, path: Path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, label="windows")
    fig.suptitle(title)
    _save(fig, path)


def plot_divergence_track(
    divergence: np.ndarray,
    hotspots: list,
    genome_id: str,
    window_bp: int,
    path: Path,
) -> None:
    x = np.arange(len(divergence)) * window_bp / 1000  # kb
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, divergence, label="Jaccard distance to cluster medoid")
    for h in hotspots:
        ax.axvspan(h * window_bp / 1000, (h + 1) * window_bp / 1000, color="red", alpha=0.25)
    ax.set_xlabel("genome position (kb)")
    ax.set_ylabel("divergence")
    ax.legend()
    fig.suptitle(f"Mutation-divergence track — {genome_id}")
    _save(fig, path)


def plot_model_comparison(results: dict, path: Path) -> None:
    """results = {model_name: {'window_accuracy': ..., 'genome_accuracy': ...}}."""
    names = list(results.keys())
    w = [results[n]["window_accuracy"] for n in names]
    g = [results[n]["genome_accuracy"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, w, 0.4, label="window accuracy")
    ax.bar(x + 0.2, g, 0.4, label="genome accuracy (majority vote)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.suptitle("Model comparison — phage cluster classification")
    _save(fig, path)
