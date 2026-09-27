"""Sequence classifiers: CNN, BiLSTM, and Transformer encoder.

All three share an interface — ``forward(input_ids) -> logits`` — so they
can be trained and compared with the same loop. Inputs are fixed-length
k-mer token windows; no padding mask is needed since windows are full.
"""
import math

import torch
import torch.nn as nn

from .config import SeqConfig


class _BaseClassifier(nn.Module):
    def __init__(self, config: SeqConfig, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_id
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder_out_dim(config), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes),
        )

    def encoder_out_dim(self, config: SeqConfig) -> int:  # noqa: D102
        raise NotImplementedError


class CNNClassifier(_BaseClassifier):
    """Multi-scale 1D convolutions + global pooling (strong motif detector)."""

    def __init__(self, config: SeqConfig, num_classes: int, num_filters: int = 128):
        self._num_filters = num_filters
        super().__init__(config, num_classes)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(config.embed_dim, num_filters, kernel_size=k, padding=k // 2)
                for k in (3, 5, 9)
            ]
        )

    def encoder_out_dim(self, config: SeqConfig) -> int:
        return self._num_filters * 3 * 2  # max+avg pool per scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids)).transpose(1, 2)  # (B, E, T)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            feats.append(h.max(dim=2).values)
            feats.append(h.mean(dim=2))
        return self.head(self.dropout(torch.cat(feats, dim=1)))


class BiLSTMClassifier(_BaseClassifier):
    """Bidirectional LSTM with mean pooling (order-sensitive context)."""

    def __init__(self, config: SeqConfig, num_classes: int):
        super().__init__(config, num_classes)
        self.lstm = nn.LSTM(
            config.embed_dim,
            config.hidden_dim // 2,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

    def encoder_out_dim(self, config: SeqConfig) -> int:
        return config.hidden_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        out, _ = self.lstm(x)
        return self.head(self.dropout(out.mean(dim=1)))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TransformerClassifier(_BaseClassifier):
    """Transformer encoder with a CLS token (long-range dependencies)."""

    def __init__(self, config: SeqConfig, num_classes: int):
        super().__init__(config, num_classes)
        self.cls = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.pos = _PositionalEncoding(config.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

    def encoder_out_dim(self, config: SeqConfig) -> int:
        return config.embed_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = self.pos(torch.cat([cls, x], dim=1))
        h = self.encoder(self.dropout(x))
        return self.head(self.dropout(h[:, 0]))


MODELS = {
    "cnn": CNNClassifier,
    "bilstm": BiLSTMClassifier,
    "transformer": TransformerClassifier,
}
