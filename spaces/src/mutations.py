"""Alignment-free mutation-trend analysis.

Without a multiple-sequence alignment, per-position comparison across genomes
is invalid. Instead this module works with k-mer sets per window:

- **Divergence track**: for each genome, the Jaccard distance of every window's
  k-mer set to the cluster medoid's corresponding window. Sustained high
  divergence marks candidate hypervariable regions ("hotspots").
- **Hotspot k-mer enrichment**: k-mers overrepresented inside hotspots vs. the
  rest of the genome (odds ratio) — candidate hypervariable motifs.
- **Diversity summary**: mean pairwise k-mer Jaccard distance per cluster.

This is exploratory: it finds *where* variability concentrates, not the
evolutionary mechanism behind it.
"""
import numpy as np

from .tokenize import kmers_of_sequence


def window_kmer_sets(seq: str, k: int, window_bp: int) -> list:
    """Non-overlapping windows -> k-mer set per window."""
    return [
        kmers_of_sequence(seq[i : i + window_bp], k)
        for i in range(0, len(seq) - window_bp + 1, window_bp)
    ]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _medoid_index(kmer_windows: list) -> int:
    """Index of the genome minimizing total Jaccard distance to the others."""
    n = len(kmer_windows)
    totals = np.zeros(n)
    n_windows = min(len(w) for w in kmer_windows)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.mean(
                [jaccard(kmer_windows[i][t], kmer_windows[j][t]) for t in range(n_windows)]
            )
            totals[i] += d
            totals[j] += d
    return int(np.argmin(totals))


def divergence_track(
    genome_id: str,
    seq: str,
    medoid_windows: list,
    k: int = 6,
    window_bp: int = 1000,
) -> np.ndarray:
    """Per-window Jaccard distance of *genome* to the medoid genome's windows."""
    own = window_kmer_sets(seq, k, window_bp)
    n = min(len(own), len(medoid_windows))
    return np.array([jaccard(own[t], medoid_windows[t]) for t in range(n)])


def find_hotspots(divergence: np.ndarray, sigma: float = 3.0) -> list:
    """Window indices with divergence above median + sigma * scaled MAD.

    Median/MAD is used instead of mean/std so a few strong hotspots don't
    inflate the threshold and hide themselves.
    """
    if len(divergence) == 0:
        return []
    median = float(np.median(divergence))
    mad = float(np.median(np.abs(divergence - median)))
    threshold = median + sigma * 1.4826 * mad + 1e-9
    return [int(i) for i, d in enumerate(divergence) if d > threshold]


def analyze_cluster(
    records: list,
    k: int = 6,
    window_bp: int = 1000,
    sigma: float = 3.0,
) -> dict:
    """Divergence tracks + hotspots for every genome in a cluster.

    records = [(genome_id, seq), ...]. Returns per-genome tracks, hotspot
    windows, and a cluster diversity summary.
    """
    kmer_windows = [window_kmer_sets(seq, k, window_bp) for _, seq in records]
    medoid = _medoid_index(kmer_windows)
    medoid_windows = kmer_windows[medoid]

    genomes = {}
    for (gid, seq), _ in zip(records, kmer_windows):
        track = divergence_track(gid, seq, medoid_windows, k, window_bp)
        genomes[gid] = {
            "divergence": track,
            "hotspots": find_hotspots(track, sigma),
            "mean_divergence": float(track.mean()) if len(track) else 0.0,
        }

    # Mean pairwise distance across the cluster (diversity summary).
    n = len(kmer_windows)
    pair_ds = []
    n_w = min(len(w) for w in kmer_windows)
    for i in range(n):
        for j in range(i + 1, n):
            pair_ds.append(
                np.mean([jaccard(kmer_windows[i][t], kmer_windows[j][t]) for t in range(n_w)])
            )
    return {
        "medoid": records[medoid][0],
        "genomes": genomes,
        "mean_pairwise_divergence": float(np.mean(pair_ds)) if pair_ds else 0.0,
        "window_bp": window_bp,
    }


def hotspot_kmer_enrichment(
    seq: str, hotspot_windows: list, k: int = 6, window_bp: int = 1000, top: int = 20
) -> list:
    """k-mers enriched inside hotspot windows vs. the rest (odds ratio)."""
    hot_kmers: dict = {}
    bg_kmers: dict = {}
    for t in range(len(seq) // window_bp):
        window = seq[t * window_bp : (t + 1) * window_bp]
        target = hot_kmers if t in set(hotspot_windows) else bg_kmers
        for kmer in kmers_of_sequence(window, k):
            target[kmer] = target.get(kmer, 0) + 1
    scored = []
    for kmer, h in hot_kmers.items():
        b = bg_kmers.get(kmer, 0)
        odds = (h + 0.5) / (b + 0.5)
        scored.append((kmer, odds, h, b))
    scored.sort(key=lambda x: -x[1])
    return scored[:top]
