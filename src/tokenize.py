"""Fixed-vocabulary k-mer tokenization.

With k=6 and stride=6, every 6-mer maps deterministically to an integer in
[0, 4095] (base-4 encoding), so no vocabulary fitting is needed and the same
tokenizer works identically on synthetic and real genomes. PAD=0, UNK=1
(reserved for ambiguous bases), real tokens are offset by +2.
"""
from .config import SeqConfig

_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_to_id(kmer: str) -> int:
    """Base-4 encode a k-mer; return -1 if it contains ambiguous bases."""
    value = 0
    for base in kmer:
        b = _BASE_TO_INT.get(base)
        if b is None:
            return -1
        value = value * 4 + b
    return value


def encode_sequence(seq: str, config: SeqConfig) -> list:
    """Tokenize a nucleotide string into k-mer ids (non-overlapping stride)."""
    ids = []
    for i in range(0, len(seq) - config.k + 1, config.stride):
        kmer = seq[i : i + config.k]
        token = kmer_to_id(kmer)
        ids.append(token + 2 if token >= 0 else config.unk_id)
    return ids


def decode_id(token_id: int, k: int) -> str:
    """Inverse of kmer_to_id for interpretability (k-mer lookup)."""
    if token_id < 2:
        return "<PAD>" if token_id == 0 else "<UNK>"
    value = token_id - 2
    chars = []
    for _ in range(k):
        chars.append("ACGT"[value % 4])
        value //= 4
    return "".join(reversed(chars))


def kmers_of_sequence(seq: str, k: int) -> set:
    """Set of all k-mers (stride 1) — used for alignment-free comparisons."""
    return {seq[i : i + k] for i in range(len(seq) - k + 1) if set(seq[i : i + k]) <= set("ACGT")}
