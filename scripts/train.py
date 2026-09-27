#!/usr/bin/env python3
"""Train CNN / BiLSTM / Transformer classifiers on genome windows.

Example:
    python scripts/train.py --data-dir data/demo --models cnn,bilstm,transformer --epochs 10
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.config import SeqConfig
from src.datasets import (
    WindowDataset,
    build_windows,
    collate_windows,
    stratified_genome_split,
)
from src.models import MODELS
from src.seqio import read_fasta
from src.trainer import evaluate_windows, train_model
from src.viz import plot_confusion_matrix, plot_history


def load_dataset(data_dir: Path, labels_csv: Path, config: SeqConfig):
    labels = {}
    with open(labels_csv) as fh:
        for row in csv.DictReader(fh):
            labels[row["genome_id"]] = row["cluster"]
    records = []
    for fasta in sorted(data_dir.glob("*.fasta")):
        records.extend(read_fasta(fasta))
    records = [(gid, seq) for gid, seq in records if gid in labels]
    clusters = sorted(set(labels.values()))
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    splits = stratified_genome_split([gid for gid, _ in records], labels, seed=config.seed)
    windows = build_windows(records, labels, cluster_to_idx, config)
    by_split = {
        name: [w for w in windows if w[0] in set(splits[name])] for name in splits
    }
    return by_split, clusters, splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phage cluster classifiers.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--models", default="cnn,bilstm,transformer")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = SeqConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.labels:
        config.labels_csv = args.labels
    if args.epochs:
        config.epochs = args.epochs
    if args.out:
        config.out_dir = args.out

    by_split, clusters, splits = load_dataset(config.data_dir, config.labels_csv, config)
    print(f"clusters: {clusters}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {len({w[0] for w in by_split[name]})} genomes, {len(by_split[name])} windows")

    loaders = {
        name: DataLoader(
            WindowDataset(by_split[name], pad_id=config.pad_id),
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            collate_fn=collate_windows,
        )
        for name in ("train", "val", "test")
    }

    summary = {"clusters": clusters, "models": {}}
    for model_name in args.models.split(","):
        model_name = model_name.strip()
        print(f"\n=== training {model_name} ===")
        model = MODELS[model_name](config, num_classes=len(clusters))
        ckpt = config.out_dir / f"{model_name}.pt"
        history = train_model(model, loaders["train"], loaders["val"], config, ckpt)
        test_metrics = evaluate_windows(model, loaders["test"])
        print(f"test: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1_macro']:.4f}")
        plot_history(history, f"{model_name} training", config.out_dir / f"{model_name}_history.png")
        plot_confusion_matrix(
            test_metrics["confusion_matrix"], clusters,
            f"{model_name} — test confusion matrix",
            config.out_dir / f"{model_name}_confusion.png",
        )
        summary["models"][model_name] = {
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "best_val_f1": history["best_val_f1"],
        }

    with open(config.out_dir / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\ndone — metrics written to {config.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
