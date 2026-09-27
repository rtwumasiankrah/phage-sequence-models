"""Central configuration for the phage sequence-modeling pipeline."""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class SeqConfig:
    """Tunable parameters for tokenization, windowing, models, and training."""

    # --- tokenization (fixed k-mer vocab: 4**k + PAD + UNK) ---
    k: int = field(default_factory=lambda: int(_env("KMER_K", "6")))
    stride: int = field(default_factory=lambda: int(_env("KMER_STRIDE", "6")))
    pad_id: int = 0
    unk_id: int = 1

    # --- windowing (genomes are split into fixed token windows; windows
    # inherit the genome's label, and genome predictions use majority vote) ---
    window_tokens: int = field(default_factory=lambda: int(_env("WINDOW_TOKENS", "512")))

    # --- model ---
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2

    # --- training ---
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-3
    patience: int = 8
    seed: int = 42

    # --- paths ---
    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data" / "demo")))
    )
    labels_csv: Path = field(
        default_factory=lambda: Path(_env("LABELS_CSV", str(PROJECT_ROOT / "data" / "demo" / "labels.csv")))
    )
    out_dir: Path = field(
        default_factory=lambda: Path(_env("OUT_DIR", str(PROJECT_ROOT / "outputs")))
    )

    @property
    def vocab_size(self) -> int:
        return 4**self.k + 2  # PAD + UNK
