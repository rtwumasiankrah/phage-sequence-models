"""Windowing and PyTorch datasets.

Genomes are split into fixed-size token windows; each window inherits the
genome's cluster label. Splits are stratified **by genome** (never by window)
so no genome contributes windows to both train and test — otherwise the
model could memorize genome-specific quirks instead of cluster signal.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SeqConfig
from .tokenize import encode_sequence


def window_tokens(token_ids: list, window_tokens: int) -> list:
    """Split token ids into non-overlapping full windows (drop remainder)."""
    return [
        token_ids[i : i + window_tokens]
        for i in range(0, len(token_ids) - window_tokens + 1, window_tokens)
    ]


def build_windows(records: list, labels: dict, cluster_to_idx: dict, config: SeqConfig) -> list:
    """records=[(genome_id, seq)] -> [(genome_id, window_ids, label_idx)]."""
    windows = []
    for genome_id, seq in records:
        if genome_id not in labels:
            continue
        for w in window_tokens(encode_sequence(seq, config), config.window_tokens):
            windows.append((genome_id, w, cluster_to_idx[labels[genome_id]]))
    return windows


def stratified_genome_split(
    genome_ids: list, labels: dict, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> dict:
    """Stratified split of genome ids -> {'train','val','test'} (by genome)."""
    rng = np.random.RandomState(seed)
    by_cluster: dict = {}
    for gid in genome_ids:
        by_cluster.setdefault(labels[gid], []).append(gid)
    splits = {"train": [], "val": [], "test": []}
    for cluster, gids in by_cluster.items():
        gids = list(gids)
        rng.shuffle(gids)
        n = len(gids)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        splits["test"].extend(gids[:n_test])
        splits["val"].extend(gids[n_test : n_test + n_val])
        splits["train"].extend(gids[n_test + n_val :])
    return splits


class WindowDataset(Dataset):
    def __init__(self, windows: list, pad_id: int = 0):
        self.windows = windows
        self.pad_id = pad_id

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        _, ids, label = self.windows[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_windows(batch: list):
    ids, labels = zip(*batch)
    return torch.stack(ids), torch.stack(labels)
