"""FASTA I/O and synthetic genome generation.

The synthetic generator plants cluster-specific motifs and hypervariable
hotspots so the full pipeline (classification + mutation-trend analysis)
can be validated end-to-end without downloading real genomes.
"""
import random
from pathlib import Path

BASES = "ACGT"
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def read_fasta(path: Path) -> list:
    """Parse a FASTA file -> [(record_id, sequence)]."""
    records, rec_id, chunks = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if rec_id is not None:
                    records.append((rec_id, "".join(chunks).upper()))
                rec_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if rec_id is not None:
        records.append((rec_id, "".join(chunks).upper()))
    return records


def write_fasta(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec_id, seq in records:
            fh.write(f">{rec_id}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def _random_seq(length: int, rng: random.Random, gc: float = 0.5) -> list:
    weights = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G T
    return rng.choices(BASES, weights=weights, k=length)


def _plant_motif(seq: list, motif: str, rng: random.Random, copies: int) -> None:
    for _ in range(copies):
        pos = rng.randrange(0, len(seq) - len(motif))
        seq[pos : pos + len(motif)] = list(motif)


def _mutate_region(seq: list, start: int, end: int, rate: float, rng: random.Random) -> None:
    for i in range(start, min(end, len(seq))):
        if rng.random() < rate:
            seq[i] = rng.choice([b for b in BASES if b != seq[i]])


def synthesize_genomes(
    num_clusters: int = 3,
    genomes_per_cluster: int = 24,
    length: int = 6144,
    motif_copies: int = 8,
    hotspot_rate: float = 0.08,
    background_rate: float = 0.01,
    seed: int = 42,
) -> tuple:
    """Generate synthetic phage-like genomes with known structure.

    Each cluster descends from a single ancestor carrying distinctive planted
    motifs (the classification signal). Descendant genomes accumulate low
    background mutations plus two hypervariable hotspots each (the
    mutation-trend signal). Returns (records, labels, motifs) where
    records = [(genome_id, sequence)] and labels = {genome_id: cluster}.
    """
    rng = random.Random(seed)
    cluster_names = [f"cluster_{chr(ord('A') + i)}" for i in range(num_clusters)]

    # Distinctive motifs per cluster: random 12-mers, unique across clusters.
    motifs, used = {}, set()
    for name in cluster_names:
        ms = []
        while len(ms) < 3:
            m = "".join(rng.choices(BASES, k=12))
            if m not in used:
                used.add(m)
                ms.append(m)
        motifs[name] = ms

    records, labels = [], {}
    for ci, name in enumerate(cluster_names):
        # One ancestor per cluster (planted motifs + GC bias); each genome is
        # a diverged descendant — mirroring how real phage clusters share most
        # of their sequence. This makes intra-cluster divergence tracks
        # meaningful: flat low background with spikes at hypervariable
        # hotspots.
        ancestor = _random_seq(length, rng, gc=0.45 + 0.05 * ci)
        _plant_motif(ancestor, motifs[name][0], rng, motif_copies)
        _plant_motif(ancestor, motifs[name][1], rng, motif_copies)
        _plant_motif(ancestor, motifs[name][2], rng, motif_copies // 2)
        # Extra copies of the primary motif: with stride-k tokenization a
        # motif lands in 6 possible phases, so dense planting keeps every
        # phase well represented in each window.
        _plant_motif(ancestor, motifs[name][0], rng, motif_copies)
        for gi in range(genomes_per_cluster):
            genome_id = f"{name}_genome_{gi:02d}"
            seq = list(ancestor)
            # Background point mutations (low — genomes stay closely related).
            _mutate_region(seq, 0, length, background_rate, rng)
            # Two hypervariable hotspots (300 bp each) per genome.
            for _ in range(2):
                start = rng.randrange(0, length - 300)
                _mutate_region(seq, start, start + 300, hotspot_rate, rng)
            records.append((genome_id, "".join(seq)))
            labels[genome_id] = name
    return records, labels, motifs
