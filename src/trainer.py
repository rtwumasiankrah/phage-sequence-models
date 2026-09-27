"""Training loop with early stopping, plus window- and genome-level metrics."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import SeqConfig
from .datasets import collate_windows


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SeqConfig,
    out_path: Path,
    device: str = "cpu",
) -> dict:
    """Train with early stopping on val F1; save best checkpoint. Returns history."""
    set_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # Linear warmup over the first 10% of steps (stabilizes the Transformer).
    total_steps = config.epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 10)
    step = 0

    def _lr_scale(s: int) -> float:
        return min(1.0, s / warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale)

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state, stale = -1.0, None, 0

    for epoch in range(config.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for ids, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{config.epochs}", leave=False):
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            total_loss += loss.item() * len(ids)
            n += len(ids)
        history["train_loss"].append(total_loss / n)

        metrics = evaluate_windows(model, val_loader, device)
        history["val_acc"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["f1_macro"])
        print(
            f"epoch {epoch+1}: loss={history['train_loss'][-1]:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_f1:
            best_f1, best_state, stale = metrics["f1_macro"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "history": history}, out_path)
    history["best_val_f1"] = best_f1
    return history


@torch.no_grad()
def evaluate_windows(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    """Per-window classification metrics."""
    model.eval()
    preds, targets = [], []
    for ids, labels in loader:
        logits = model(ids.to(device))
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        "n": len(targets),
    }


@torch.no_grad()
def predict_genomes(
    model: nn.Module,
    windows: list,
    genome_ids: list,
    config: SeqConfig,
    device: str = "cpu",
) -> dict:
    """Aggregate window predictions to genome level by majority vote."""
    model.eval()
    by_genome: dict = {}
    loader = DataLoader(
        [(torch.tensor(w, dtype=torch.long), 0) for _, w, _ in windows],
        batch_size=config.batch_size,
    )
    all_preds = []
    for (ids, _) in loader:
        all_preds.extend(model(ids.to(device)).argmax(dim=1).cpu().tolist())
    for (gid, _, _), p in zip(windows, all_preds):
        by_genome.setdefault(gid, []).append(p)
    voted = {
        gid: int(np.bincount(ps).argmax()) for gid, ps in by_genome.items()
    }
    return {gid: voted[gid] for gid in genome_ids if gid in voted}
