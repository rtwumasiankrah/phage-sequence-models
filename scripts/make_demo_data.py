#!/usr/bin/env python3
"""Generate a synthetic demo dataset with known structure.

Each cluster gets distinctive planted motifs (classification signal) and
every genome gets hypervariable hotspots on a low-mutation background
(mutation-trend signal). Lets the full pipeline run without network access.

Example:
    python scripts/make_demo_data.py --out data/demo --genomes-per-cluster 24
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.seqio import synthesize_genomes, write_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic phage genomes.")
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--genomes-per-cluster", type=int, default=24)
    parser.add_argument("--length", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records, labels, motifs = synthesize_genomes(
        num_clusters=args.clusters,
        genomes_per_cluster=args.genomes_per_cluster,
        length=args.length,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for genome_id, seq in records:
        write_fasta(args.out / f"{genome_id}.fasta", [(genome_id, seq)])
    with open(args.out / "labels.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["genome_id", "cluster"])
        writer.writerows(sorted(labels.items()))
    print(f"wrote {len(records)} genomes to {args.out}")
    for cluster, ms in motifs.items():
        print(f"  {cluster} motifs: {ms}")


if __name__ == "__main__":
    main()
