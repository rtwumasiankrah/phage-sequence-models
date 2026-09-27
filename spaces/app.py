"""Phage cluster classifier — Hugging Face Spaces demo.

Reuses this repo's real pipeline code (vendored under ``spaces/src/``):
base-4 6-mer tokenization (``src/tokenize.py``), fixed 512-token windowing
(``src/datasets.py``), and the trained multi-scale 1D CNN
(``src/models.py``). Each window is classified independently and the genome
prediction is a majority vote over its windows.

HARD CLAIM DISCIPLINE: the bundled CNN was trained on *synthetic*
cluster-descended genomes. Its 1.00 test accuracy validates pipeline
behavior only — it says nothing about real-world biological performance.
Do not present predictions on real genomes (e.g. GingkoMaracino, which is
unclustered) as scientifically meaningful.
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import gradio as gr

SPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPACE_DIR))  # vendored src/ lives next to app.py

from src.config import SeqConfig
from src.datasets import window_tokens
from src.models import CNNClassifier
from src.seqio import read_fasta
from src.tokenize import encode_sequence

with open(SPACE_DIR / "labels.json") as fh:
    META = json.load(fh)

CLUSTERS = META["clusters"]
IDX_TO_CLUSTER = {i: c for c, i in META["cluster_to_idx"].items()}

_config = SeqConfig()
for key in ("k", "stride", "window_tokens", "embed_dim", "hidden_dim"):
    setattr(_config, key, META[key])

_model = CNNClassifier(_config, num_classes=len(CLUSTERS), num_filters=META["num_filters"])
_model.load_state_dict(
    torch.load(SPACE_DIR / "cnn_demo.pt", map_location="cpu", weights_only=True)
)
_model.eval()

MIN_BP = META["window_tokens"] * META["k"]  # one full window
VALID_BASES = set("ACGT")

CAVEAT = (
    "Demo model trained on **synthetic cluster-descended genomes** — test accuracy "
    "1.00 on that synthetic data validates the pipeline, not real-world biological "
    "performance. Predictions on real phage genomes are illustrative only."
)


def _clean(raw: str) -> str:
    """Strip FASTA headers/whitespace, uppercase."""
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith(">")]
    return "".join(lines).upper().replace(" ", "").replace("\t", "")


@torch.no_grad()
def predict_sequence(seq: str) -> dict:
    """Classify a DNA string. Returns prediction details or {'error': msg}."""
    seq = _clean(seq)
    if not seq:
        return {"error": "No sequence provided — paste DNA or upload a FASTA file."}
    if len(seq) < MIN_BP:
        return {
            "error": (
                f"Sequence too short ({len(seq):,} bp). Need at least {MIN_BP:,} bp "
                f"for one {META['window_tokens']}-token window."
            )
        }
    bad = sorted(set(seq) - VALID_BASES)
    n_bad = sum(1 for b in seq if b not in VALID_BASES)
    if n_bad / len(seq) > 0.5:
        return {
            "error": (
                f"Only {100 * (1 - n_bad / len(seq)):.0f}% of characters are A/C/G/T — "
                "this doesn't look like a DNA sequence."
            )
        }

    tokens = encode_sequence(seq, _config)
    windows = window_tokens(tokens, _config.window_tokens)
    batch = torch.tensor(windows, dtype=torch.long)
    preds = _model(batch).argmax(dim=1).tolist()

    votes = [0] * len(CLUSTERS)
    for p in preds:
        votes[p] += 1
    winner = max(range(len(CLUSTERS)), key=lambda i: votes[i])
    fractions = {c: votes[i] / len(preds) for i, c in enumerate(CLUSTERS)}

    warning = ""
    if bad:
        warning = (
            f"Note: {n_bad:,} ambiguous characters "
            f"({', '.join(bad)}) were treated as unknown tokens."
        )
    return {
        "predicted_cluster": IDX_TO_CLUSTER[winner],
        "vote_fractions": fractions,
        "vote_counts": {c: votes[i] for i, c in enumerate(CLUSTERS)},
        "n_windows": len(windows),
        "seq_len": len(seq),
        "warning": warning,
    }


def _vote_chart(result: dict):
    counts = result["vote_counts"]
    labels = list(counts)
    values = [counts[c] for c in labels]
    colors = [
        "#2ca02c" if c == result["predicted_cluster"] else "#9e9e9e" for c in labels
    ]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("windows")
    ax.set_title(f"Window votes → {result['predicted_cluster']} ({result['n_windows']} windows)")
    for c, v in zip(labels, values):
        ax.text(c, v + 0.05, str(v), ha="center", fontsize=11)
    fig.tight_layout()
    return fig


def _run(seq_text: str, fasta_file):
    if fasta_file is not None:
        try:
            records = read_fasta(Path(fasta_file.name))
        except Exception as exc:  # corrupt upload — fail gracefully
            return None, None, f"Could not parse the uploaded file: {exc}", CAVEAT
        if not records:
            return None, None, "The uploaded FASTA file contains no sequences.", CAVEAT
        seq_text = records[0][1]
    result = predict_sequence(seq_text or "")
    if "error" in result:
        return None, None, result["error"], CAVEAT
    details = (
        f"Sequence length: {result['seq_len']:,} bp → {result['n_windows']} windows of "
        f"{META['window_tokens']} tokens.\n"
        f"Votes: " + ", ".join(f"{c}: {result['vote_counts'][c]}" for c in CLUSTERS)
    )
    if result["warning"]:
        details += f"\n{result['warning']}"
    return result["vote_fractions"], _vote_chart(result), details, CAVEAT


def _load_example(cluster: str) -> str:
    path = SPACE_DIR / META["examples"][cluster]
    return read_fasta(path)[0][1]


with gr.Blocks(title="Phage Cluster Classifier") as demo:
    gr.Markdown(
        "# 🧬 Phage Cluster Classifier\n"
        "Paste a DNA sequence (or upload a FASTA file) and a trained 1D-CNN predicts "
        "its phage cluster by majority vote over 512-token 6-mer windows.\n\n"
        f"_{CAVEAT}_"
    )
    with gr.Row():
        seq_box = gr.Textbox(
            label="DNA sequence (A/C/G/T)",
            lines=6,
            placeholder="Paste raw DNA here — at least 3,072 bp for one window…",
        )
        fasta_box = gr.File(
            label="…or upload a FASTA file (first record is used)",
            file_types=[".fasta", ".fa", ".txt"],
            type="filepath",
        )
    with gr.Row():
        ex_a = gr.Button("Load example: cluster_A genome")
        ex_b = gr.Button("Load example: cluster_B genome")
        predict_btn = gr.Button("Predict cluster", variant="primary")
    with gr.Row():
        label_out = gr.Label(label="Predicted cluster (vote share per window)")
    with gr.Row():
        plot_out = gr.Plot(label="Per-window votes")
    details_out = gr.Textbox(label="Details", lines=3, interactive=False)
    caption_out = gr.Markdown(CAVEAT)

    ex_a.click(fn=lambda: _load_example("cluster_A"), outputs=seq_box)
    ex_b.click(fn=lambda: _load_example("cluster_B"), outputs=seq_box)
    predict_btn.click(
        fn=_run, inputs=[seq_box, fasta_box],
        outputs=[label_out, plot_out, details_out, caption_out],
        api_name="predict_cluster",
    )
    seq_box.submit(
        fn=_run, inputs=[seq_box, fasta_box],
        outputs=[label_out, plot_out, details_out, caption_out],
        api_name="predict_cluster",
    )

if __name__ == "__main__":
    demo.launch()
