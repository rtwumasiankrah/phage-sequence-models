#!/usr/bin/env python3
"""Alignment-free mutation-trend analysis for a set of related genomes.

Computes per-genome divergence tracks against the cluster medoid, flags
hypervariable hotspots, and reports enriched k-mers inside hotspots.

Example:
    python scripts/analyze_mutations.py --data-dir data/demo --out outputs/mutations
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mutations import analyze_cluster, hotspot_kmer_enrichment
from src.seqio import read_fasta
from src.viz import plot_divergence_track


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutation-trend hotspot analysis.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/mutations"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--window-bp", type=int, default=1000)
    parser.add_argument("--max-genomes-per-cluster", type=int, default=8,
                        help="cap genomes per cluster for the plot grid")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = {}
    labels_csv = args.labels or (args.data_dir / "labels.csv")
    with open(labels_csv) as fh:
        for row in csv.DictReader(fh):
            labels[row["genome_id"]] = row["cluster"]

    records = []
    for fasta in sorted(args.data_dir.glob("*.fasta")):
        records.extend(read_fasta(fasta))
    records = [(gid, seq) for gid, seq in records if gid in labels]

    by_cluster: dict = {}
    for gid, seq in records:
        by_cluster.setdefault(labels[gid], []).append((gid, seq))

    report = {}
    for cluster, cluster_records in sorted(by_cluster.items()):
        print(f"\n=== {cluster} ({len(cluster_records)} genomes) ===")
        analysis = analyze_cluster(cluster_records, k=args.k, window_bp=args.window_bp)
        print(f"  medoid: {analysis['medoid']}")
        print(f"  mean pairwise divergence: {analysis['mean_pairwise_divergence']:.4f}")
        cluster_out = args.out / cluster
        cluster_report = {
            "medoid": analysis["medoid"],
            "mean_pairwise_divergence": analysis["mean_pairwise_divergence"],
            "genomes": {},
        }
        for gid, seq in cluster_records[: args.max_genomes_per_cluster]:
            g = analysis["genomes"][gid]
            hotspots = g["hotspots"]
            print(f"  {gid}: mean_div={g['mean_divergence']:.4f} hotspots={hotspots}")
            plot_divergence_track(
                g["divergence"], hotspots, gid, args.window_bp, cluster_out / f"{gid}.png"
            )
            enriched = []
            if hotspots:
                enriched = [
                    {"kmer": kmer, "odds_ratio": round(odds, 2), "hotspot_hits": h, "bg_hits": b}
                    for kmer, odds, h, b in hotspot_kmer_enrichment(
                        seq, hotspots, k=args.k, window_bp=args.window_bp
                    )
                ]
            cluster_report["genomes"][gid] = {
                "mean_divergence": g["mean_divergence"],
                "hotspot_windows": hotspots,
                "enriched_kmers": enriched[:10],
            }
        report[cluster] = cluster_report

    with open(args.out / "mutation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport -> {args.out / 'mutation_report.json'}")


if __name__ == "__main__":
    main()
