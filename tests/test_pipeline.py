"""Tests for tokenization, windowing, synthetic data, and mutation stats.

Model forward-pass tests need torch; everything else is numpy/stdlib only.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SeqConfig
from src.datasets import build_windows, stratified_genome_split, window_tokens
from src.mutations import analyze_cluster, find_hotspots, jaccard
from src.seqio import read_fasta, synthesize_genomes, write_fasta
from src.tokenize import decode_id, encode_sequence, kmer_to_id


@pytest.fixture
def config():
    return SeqConfig()


def test_kmer_encoding_is_deterministic(config):
    ids = encode_sequence("ACGTACGTACGT", config)
    assert ids == encode_sequence("ACGTACGTACGT", config)
    assert len(ids) == 12 // config.k  # k=6, stride=6
    assert all(2 <= i < config.vocab_size for i in ids)


def test_kmer_vocab_size(config):
    assert config.vocab_size == 4**6 + 2
    assert kmer_to_id("AAAAAA") == 0
    assert kmer_to_id("NNNNNN") == -1
    assert encode_sequence("NNNNNN", config) == [config.unk_id]


def test_decode_roundtrip(config):
    for kmer in ("ACGTAC", "TTGGCC", "GATTACA"[:6]):
        assert decode_id(kmer_to_id(kmer) + 2, config.k) == kmer


def test_windowing_drops_remainder():
    tokens = list(range(1000))
    windows = window_tokens(tokens, 512)
    assert len(windows) == 1
    assert len(windows[0]) == 512


def test_synthetic_genomes_have_structure():
    records, labels, motifs = synthesize_genomes(
        num_clusters=2, genomes_per_cluster=4, length=2048, seed=1
    )
    assert len(records) == 8
    assert set(labels.values()) == {"cluster_A", "cluster_B"}
    # planted motifs should be present in their cluster's genomes
    for gid, seq in records:
        cluster = labels[gid]
        assert any(m in seq for m in motifs[cluster])


def test_fasta_roundtrip(tmp_path):
    p = tmp_path / "x.fasta"
    write_fasta(p, [("g1", "ACGT" * 100)])
    assert read_fasta(p) == [("g1", "ACGT" * 100)]


def test_split_is_by_genome_no_leakage():
    records, labels, _ = synthesize_genomes(num_clusters=2, genomes_per_cluster=6, length=1024, seed=2)
    splits = stratified_genome_split([g for g, _ in records], labels, seed=2)
    train = set(splits["train"])
    assert not (train & set(splits["val"])) and not (train & set(splits["test"]))
    assert set(splits["train"]) | set(splits["val"]) | set(splits["test"]) == set(labels)


def test_build_windows_labels(config):
    records, labels, _ = synthesize_genomes(num_clusters=2, genomes_per_cluster=2, length=6144, seed=3)
    cluster_to_idx = {"cluster_A": 0, "cluster_B": 1}
    windows = build_windows(records, labels, cluster_to_idx, config)
    assert windows
    assert all(len(w) == config.window_tokens for _, w, _ in windows)
    assert {lbl for _, _, lbl in windows} == {0, 1}


def test_jaccard():
    assert jaccard({1, 2}, {1, 2}) == 0.0
    assert jaccard({1}, {2}) == 1.0
    assert jaccard(set(), set()) == 0.0


def test_hotspot_detection_finds_planted_hotspots():
    # Two near-identical genomes, one with a heavily mutated region.
    base = "ACGT" * 1500  # 6000 bp
    mutated = list(base)
    for i in range(2000, 2600):
        mutated[i] = {"A": "T", "T": "A", "C": "G", "G": "C"}[mutated[i]]
    records = [("g0", base), ("g1", "".join(mutated))]
    analysis = analyze_cluster(records, k=6, window_bp=1000, sigma=1.0)
    hot = analysis["genomes"]["g1"]["hotspots"]
    assert 2 in hot  # window covering bp 2000-3000
    assert analysis["genomes"]["g0"]["hotspots"] == []


def test_model_forward_shapes(config):
    torch = pytest.importorskip("torch")
    from src.models import MODELS

    batch = torch.randint(2, config.vocab_size, (4, config.window_tokens))
    for name, cls in MODELS.items():
        model = cls(config, num_classes=3)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (4, 3), name
