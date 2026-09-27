#!/usr/bin/env python3
"""Evaluate trained checkpoints: window + genome-level metrics and comparison.

Example:
    python scripts/evaluate.py --data-dir data/demo --out outputs
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import load_dataset
from src.config import SeqConfig
from src.models import MODELS
from src.trainer import evaluate_windows, predict_genomes
from src.viz import plot_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained phage classifiers.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--models", default="cnn,bilstm,transformer")
    args = parser.parse_args()

    config = SeqConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.labels:
        config.labels_csv = args.labels
    if args.out:
        config.out_dir = args.out

    from torch.utils.data import DataLoader
    from src.datasets import WindowDataset, collate_windows

    by_split, clusters, splits = load_dataset(config.data_dir, config.labels_csv, config)
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    test_windows = by_split["test"]
    test_genome_ids = sorted({w[0] for w in test_windows})
    loader = DataLoader(
        WindowDataset(test_windows, pad_id=config.pad_id),
        batch_size=config.batch_size,
        collate_fn=collate_windows,
    )
    true_genome_labels = {}
    with open(config.labels_csv) as fh:
        import csv

        for row in csv.DictReader(fh):
            true_genome_labels[row["genome_id"]] = cluster_to_idx[row["cluster"]]

    results = {}
    for model_name in args.models.split(","):
        model_name = model_name.strip()
        ckpt = torch.load(config.out_dir / f"{model_name}.pt", map_location="cpu", weights_only=False)
        model = MODELS[model_name](config, num_classes=len(clusters))
        model.load_state_dict(ckpt["state_dict"])

        window_metrics = evaluate_windows(model, loader)
        genome_preds = predict_genomes(model, test_windows, test_genome_ids, config)
        y_true = [true_genome_labels[g] for g in genome_preds]
        y_pred = [genome_preds[g] for g in genome_preds]
        genome_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

        results[model_name] = {
            "window_accuracy": window_metrics["accuracy"],
            "window_f1_macro": window_metrics["f1_macro"],
            "genome_accuracy": genome_acc,
            "n_test_genomes": len(y_true),
        }
        print(
            f"{model_name}: window_acc={window_metrics['accuracy']:.4f} "
            f"genome_acc={genome_acc:.4f} ({len(y_true)} genomes)"
        )

    plot_model_comparison(results, config.out_dir / "model_comparison.png")
    with open(config.out_dir / "eval.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"comparison plot -> {config.out_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
