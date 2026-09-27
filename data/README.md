# Data

## Included example

`GingkoMaracino_MK359341.fasta` — the complete 50,013 bp genome of
*Mycobacterium* phage GingkoMaracino (GenBank MK359341.1), co-discovered
during the author's SEA-PHAGES research. Publicly available via NCBI;
included here as a ready-made example input.

## Real genomes (not included)

To train on real phage genomes:

1. Browse [PhagesDB](https://phagesdb.org) for *Mycobacterium* phages and note
   each phage's cluster assignment.
2. Build a CSV with header `accession,cluster` (see `accessions_example.csv`).
3. Download:
   ```bash
   python scripts/download_genomes.py --csv my_accessions.csv --out data/genomes
   ```
4. Train:
   ```bash
   python scripts/train.py --data-dir data/genomes --epochs 20
   ```

## Synthetic demo data

No network? Generate a labeled synthetic dataset with planted motifs and
hypervariable hotspots:

```bash
python scripts/make_demo_data.py --out data/demo --genomes-per-cluster 24
python scripts/train.py --data-dir data/demo
```
