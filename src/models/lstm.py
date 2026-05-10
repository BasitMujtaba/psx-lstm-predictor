"""
src/models/lstm.py
====================
Enhanced LSTM with multi-head attention, residual connections,
and a deeper FC head for PSX return prediction.

input  : (batch, seq_len, n_features)   float32
output : (batch,)                       float32  scaled return

Architecture
------------
  Input
    -> Feature projection (linear embedding)
    -> Stacked Bidirectional LSTM
    -> Multi-Head Self-Attention over LSTM outputs
    -> Residual connection + LayerNorm
    -> Temporal pooling (mean + max concatenated)
    -> Deep FC head with GELU + Dropout
  Output (B,)

config.yaml section used:
  model:
    architecture:  "lstm"
    input_size:    32
    lstm_hidden:   128
    lstm_layers:   2
    lstm_dropout:  0.2
    fc_hidden:     64
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the time dimension.
    Allows the model to focus on the most relevant timesteps.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim   = hidden_size,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))   # residual


class PSXLSTMModel(nn.Module):
    """
    Enhanced LSTM model with:
      - Input feature projection
      - Bidirectional LSTM (captures past + future context in sequence)
      - Multi-head temporal attention
      - Residual connections + LayerNorm
      - Mean + max temporal pooling
      - Deep GELU FC head
    """

    def __init__(self, input_size=32, hidden_size=128,
                 num_layers=2, dropout=0.2, fc_hidden=64):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Feature projection ─────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # ── Bidirectional LSTM ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = True,
        )

        lstm_out_size = hidden_size * 2   # bidirectional

        # ── Project bidir output back to hidden_size for attention ─────
        self.lstm_proj = nn.Linear(lstm_out_size, hidden_size)

        # ── Temporal attention ─────────────────────────────────────────
        self.attention = TemporalAttention(
            hidden_size = hidden_size,
            num_heads   = 4,
            dropout     = dropout,
        )

        # ── LayerNorm after attention ──────────────────────────────────
        self.norm = nn.LayerNorm(hidden_size)

        # ── FC head (mean + max pooling concatenated) ──────────────────
        # input = hidden_size * 2 (mean-pool + max-pool)
        fc_in = hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_hidden * 2),
            nn.LayerNorm(fc_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden * 2, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()
        log.info(
            "PSXLSTMModel (enhanced)  input=%d  hidden=%d  layers=%d  "
            "dropout=%.2f  fc_hidden=%d  bidirectional=True  attention=True",
            input_size, hidden_size, num_layers, dropout, fc_hidden,
        )

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, seq_len, input_size)

        Returns
        -------
        out : (B,)
        """
        # ── Project input features ─────────────────────────────────────
        x = self.input_proj(x)                      # (B, seq, hidden)

        # ── Bidirectional LSTM ─────────────────────────────────────────
        lstm_out, _ = self.lstm(x)                  # (B, seq, hidden*2)
        lstm_out    = self.lstm_proj(lstm_out)       # (B, seq, hidden)

        # ── Residual + attention ───────────────────────────────────────
        attn_out = self.attention(lstm_out)          # (B, seq, hidden)
        attn_out = self.norm(attn_out + lstm_out)    # residual

        # ── Temporal pooling ───────────────────────────────────────────
        mean_pool = attn_out.mean(dim=1)             # (B, hidden)
        max_pool  = attn_out.max(dim=1).values       # (B, hidden)
        pooled    = torch.cat([mean_pool, max_pool], dim=-1)  # (B, hidden*2)

        # ── FC head ────────────────────────────────────────────────────
        out = self.fc(pooled).squeeze(-1)            # (B,)
        return out


def build_model(cfg):
    """
    Instantiates PSXLSTMModel from config.yaml.

    Reads these keys from cfg["model"]:
      input_size   : len(FEATURE_COLS)   default 32
      lstm_hidden  : hidden units        default 128
      lstm_layers  : stacked layers      default 2
      lstm_dropout : dropout             default 0.2
      fc_hidden    : FC head units       default 64
    """
    m = cfg.get("model", {})

    model = PSXLSTMModel(
        input_size  = m.get("input_size",   32),
        hidden_size = m.get("lstm_hidden",  128),
        num_layers  = m.get("lstm_layers",   2),
        dropout     = m.get("lstm_dropout", 0.2),
        fc_hidden   = m.get("fc_hidden",    64),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model built: Enhanced LSTM  |  trainable params: %s",
             f"{n_params:,}")
    return model
