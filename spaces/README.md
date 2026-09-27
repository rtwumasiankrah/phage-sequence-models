---
title: Phage Cluster Classifier
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🧬 Phage Cluster Classifier — Hugging Face Spaces demo

Interactive demo of the CNN phage-cluster classifier from
[phage-sequence-models](https://github.com/rtwumasiankrah/phage-sequence-models).
Paste a DNA sequence (or upload a FASTA file); the app tokenizes it into 6-mers,
splits it into 512-token windows, classifies each window with the trained 1D-CNN,
and predicts the cluster by majority vote.

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | Gradio app (the Space entrypoint) |
| `requirements.txt` | `gradio`, CPU-only `torch`, `matplotlib`, `numpy` |
| `cnn_demo.pt` | Trained CNN weights (state_dict, ~2 MB) |
| `labels.json` | Cluster index mapping + the inference config used to rebuild the model |
| `src/` | Vendored copy of the repo's real pipeline code (`tokenize.py`, `datasets.py`, `models.py`, `seqio.py`, `config.py`). Regenerate with: `rm -rf spaces/src && cp -r src spaces/src && find spaces/src -name __pycache__ -type d -exec rm -rf {} +` |
| `examples/` | Two held-out synthetic demo genomes (one per cluster) for the "load example" buttons |

## Claim discipline

The bundled model was trained on **synthetic cluster-descended genomes**.
Its 1.00 test accuracy validates pipeline behavior only — it does **not**
establish real-world biological performance. The UI states this caveat next to
every prediction. The GingkoMaracino genome is unclustered and is deliberately
*not* offered as an example.

## Deploy to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Pick the **Gradio** SDK (leave the hardware on the free CPU tier).
3. Upload **every file in this folder** (including `src/` and `examples/`) to the Space root, so `app.py` sits at the top level.
4. The Space builds from `requirements.txt` and launches `app.py` automatically. No secrets or extra setup needed.

To update the demo later, retrain with `scripts/train.py`, re-export the
state_dict to `spaces/cnn_demo.pt`, and re-upload.
