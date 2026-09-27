#!/usr/bin/env python3
"""Download phage genomes from NCBI (Entrez) given an accession -> cluster CSV.

CSV format: accession,cluster  (header row required)

Example:
    python scripts/download_genomes.py --csv data/accessions_example.csv --out data/genomes

Cluster labels are not in GenBank records — get them from PhagesDB
(https://phagesdb.org): search Mycobacterium phages, note each phage's
cluster, and build the CSV. GingkoMaracino (MK359341.1) is included in the
example CSV.
"""
import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.seqio import write_fasta


def efetch_fasta(accession: str) -> tuple:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nucleotide&id={accession}&rettype=fasta&retmode=text"
    )
    data = urllib.request.urlopen(url, timeout=120).read().decode()
    lines = data.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise RuntimeError(f"Unexpected response for {accession}")
    return lines[0][1:].split()[0], "".join(lines[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download phage genomes from NCBI.")
    parser.add_argument("--csv", type=Path, required=True, help="accession,cluster CSV")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = {}
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            acc, cluster = row["accession"].strip(), row["cluster"].strip()
            print(f"fetching {acc} ({cluster}) ...")
            rec_id, seq = efetch_fasta(acc)
            write_fasta(args.out / f"{acc}.fasta", [(rec_id, seq)])
            labels[rec_id] = cluster
            print(f"  {len(seq)} bp")
            time.sleep(args.sleep)

    with open(args.out / "labels.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["genome_id", "cluster"])
        writer.writerows(sorted(labels.items()))
    print(f"wrote {len(labels)} genomes + labels.csv to {args.out}")


if __name__ == "__main__":
    main()
