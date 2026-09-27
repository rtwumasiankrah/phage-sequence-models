# 🧬 Phage Sequence Models

[![CI](https://github.com/rtwumasiankrah/phage-sequence-models/actions/workflows/ci.yml/badge.svg)](https://github.com/rtwumasiankrah/phage-sequence-models/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rtwumasiankrah/phage-sequence-models/blob/main/notebooks/phage_sequence_models.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep sequence modeling for bacteriophage genomics. In virology, a genome is
essentially a long string of text — so the same architectures built for
language (CNNs, LSTMs, Transformers) can learn directly from DNA. This
pipeline:

1. **Classifies** phage genomes into clusters from raw sequence with CNN,
   BiLSTM, and Transformer models (trained and compared head-to-head).
2. **Analyzes mutation trends** with an alignment-free divergence method that
   flags hypervariable hotspots and the k-mers enriched inside them.

The repo ships with the complete 50 kb genome of *Mycobacterium* phage
**GingkoMaracino** (GenBank MK359341.1) — co-discovered by the author during
SEA-PHAGES research — plus a synthetic data generator so everything runs
without network access.

## Live demo

The `spaces/` folder is a push-ready **Hugging Face Spaces** demo: a Gradio app
serving the trained CNN cluster classifier. Paste DNA (or upload a FASTA file)
and it predicts the phage cluster by majority vote over 512-token 6-mer
windows, with a per-window vote chart. It reuses the repo's real pipeline code
(vendored under `spaces/src/`) and a ~2 MB trained CNN (`spaces/cnn_demo.pt`,
1.00 accuracy on the synthetic test set).

Go-live is three steps — see `spaces/README.md`:

1. Create a new Space on huggingface.co/spaces with the **Gradio** SDK.
2. Upload the contents of `spaces/` (including `src/` and `examples/`) to the Space root.
3. The Space builds from `requirements.txt` and launches `app.py` — no extra setup.

<!-- Uncomment once the Space exists:
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rtwumasiankrah/phage-cluster-classifier)
-->

## How it works

**Classification pipeline:**

- **Input** — FASTA genomes with cluster labels.
- **Tokenization** — non-overlapping 6-mers map deterministically to a fixed
  4,098-token vocabulary (base-4 encoding) — no fitting step, identical
  behavior on synthetic and real genomes.
- **Windowing** — genomes are split into 512-token windows, each inheriting
  the genome's cluster label; genome-level predictions use majority vote.
  Splits are stratified **by genome**, so no genome leaks windows across
  train/test.
- **Models** — a multi-scale 1D CNN (motif detector), a BiLSTM
  (order-sensitive context), and a Transformer encoder with CLS token
  (long-range dependencies), all trained through one shared loop with early
  stopping on validation F1.
- **Output** — cluster prediction per window and per genome.

**Mutation-trend pipeline:**

- **Input** — related genomes within a cluster.
- **Divergence tracks** — per-window k-mer Jaccard distance to the cluster
  medoid genome.
- **Hotspot detection** — windows above median + scaled MAD are flagged as
  hotspots.
- **Output** — hotspot windows plus the k-mers enriched inside them, as
  candidate hypervariable motifs. Alignment-free, and honest about its limits
  (see below).

## Quickstart

```bash
git clone https://github.com/rtwumasiankrah/phage-sequence-models.git
cd phage-sequence-models
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU build

# 1. synthetic demo data (planted motifs + hypervariable hotspots)
python scripts/make_demo_data.py --out data/demo --genomes-per-cluster 24

# 2. train all three models (~5 min on CPU)
python scripts/train.py --data-dir data/demo

# 3. evaluate + compare (window-level and genome-level accuracy)
python scripts/evaluate.py --data-dir data/demo

# 4. mutation-trend hotspot analysis
python scripts/analyze_mutations.py --data-dir data/demo --out outputs/mutations
```

To use real genomes instead, see `data/README.md` (NCBI download via accession list).

## Results (synthetic demo)

On 72 synthetic genomes (3 clusters × 24, cluster-descended with planted
motifs and hypervariable hotspots):

| model | window accuracy | genome accuracy (majority vote) |
|---|---|---|
| CNN | 1.00 | 1.00 |
| BiLSTM | 1.00 | 1.00 |
| Transformer | 1.00 | 1.00 |

The demo is designed to be learnable — it validates the pipeline, not the
models' limits. The interesting difference is *how* they learn: the CNN
converges in a handful of epochs, while the Transformer needs LR warmup and
roughly twice the epochs to get there (visible in the `*_history.png`
plots). On real, noisier phage clusters the accuracy ranking is where the
comparison harness earns its keep.

Mutation-trend analysis recovers the planted hypervariable hotspots as
divergence spikes against the cluster medoid (see
`outputs/mutations/`) and surfaces the k-mers enriched inside them.

## What's in this repo

| Path | What it is |
|---|---|
| `src/tokenize.py` | Fixed-vocab k-mer tokenizer |
| `src/datasets.py` | Windowing, genome-stratified splits, PyTorch datasets |
| `src/models.py` | CNN / BiLSTM / Transformer classifiers, one interface |
| `src/trainer.py` | Training loop, early stopping, window + genome metrics |
| `src/mutations.py` | Alignment-free divergence tracks, hotspot detection |
| `src/viz.py` | Training curves, confusion matrices, divergence plots |
| `scripts/` | `download_genomes.py`, `make_demo_data.py`, `train.py`, `evaluate.py`, `analyze_mutations.py` |
| `notebooks/` | Thin Colab walkthrough — shells out to `scripts/`, runs end-to-end in the browser |
| `data/GingkoMaracino_MK359341.fasta` | Real 50 kb phage genome co-discovered by the author |
| `tests/` | Unit tests (tokenizer, splits, hotspot recovery, model shapes) |

## Limitations & honesty notes

- The demo is synthetic by design — real mycobacteriophage clusters are far
  noisier, and cluster labels must come from PhagesDB (GenBank records don't
  carry them).
- Hotspot analysis is alignment-free: it finds *where* variability
  concentrates, not the evolutionary mechanism. A production follow-up would
  add proper multiple-sequence alignment.
- This pipeline classifies and analyzes — it does not design phages. The
  roadmap below is where generative work belongs.

## Roadmap

- Autoregressive genome language model for zero-shot mutation-effect scoring.
- Proper MSA-based (MAFFT) per-position conservation analysis on real clusters.
- Codon-aware tokenization and gene-annotation-conditioned models.
- The long-term direction: generative design of synthetic phages against
  antibiotic-resistant bacteria — an active research frontier this pipeline is
  a first step toward.

## Author

**Richard Twumasi-Ankrah** — M.S. Computer Science, building AI that makes
medical intelligence widely accessible. [GitHub](https://github.com/rtwumasiankrah)
