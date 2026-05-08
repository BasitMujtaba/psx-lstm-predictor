"""
src/models/lstm.py
====================
LSTM (and optional Transformer) architectures for PSX return prediction.

Two models are provided:
  1. PSXLSTMModel       — stacked LSTM with dropout + FC head
  2. PSXTransformerModel — Transformer encoder + FC head

Both share the same input/output contract:
  input  : (batch, seq_len, n_features)   float32
  output : (batch,)                       float32  scaled return

Why two architectures?
  LSTM  — strong on short-range temporal dependencies, fast to train,
           works well with PSX's relatively small dataset size.
  Transformer — better at capturing long-range dependencies (macro cycles,
           multi-month sentiment shifts); needs more data to shine.

config.yaml section used:
  model:
    architecture:   "lstm"          # "lstm" or "transformer"
    input_size:     26              # len(FEATURE_COLS)
    hidden_size:    128
    num_layers:     2
    dropout:        0.2
    fc_hidden:      64
    # Transformer-only
    nhead:          4
    dim_feedforward: 256
    num_encoder_layers: 2
"""

import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# =============================================================================
# 1.  LSTM Model
# =============================================================================

class PSXLSTMModel(nn.Module):
    """
    Stacked LSTM with a two-layer FC regression head.

    Architecture
    ------------
    Input  (B, seq_len, input_size)
      -> LSTM (num_layers, hidden_size, dropout between layers)
      -> last hidden state  (B, hidden_size)
      -> LayerNorm
      -> FC(hidden_size -> fc_hidden)  + ReLU + Dropout
      -> FC(fc_hidden   -> 1)
    Output (B,)

    Parameters
    ----------
    input_size  : number of features  (len(FEATURE_COLS), default 26)
    hidden_size : LSTM hidden units   (default 128)
    num_layers  : stacked LSTM layers (default 2)
    dropout     : dropout between LSTM layers and in FC head (default 0.2)
    fc_hidden   : hidden units in FC head (default 64)
    """

    def __init__(self, input_size=26, hidden_size=128,
                 num_layers=2, dropout=0.2, fc_hidden=64):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # LSTM — dropout only applied between layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        # Normalise the last hidden state before the FC head
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Two-layer FC regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()
        log.info(
            "PSXLSTMModel  input=%d  hidden=%d  layers=%d  "
            "dropout=%.2f  fc_hidden=%d",
            input_size, hidden_size, num_layers, dropout, fc_hidden,
        )

    def _init_weights(self):
        """Xavier uniform for LSTM weights, zeros for biases."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, seq_len, input_size)

        Returns
        -------
        out : (B,)
        """
        # lstm_out : (B, seq_len, hidden_size)
        # h_n      : (num_layers, B, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)

        # Take the last layer's hidden state at the final time step
        last_hidden = h_n[-1]                    # (B, hidden_size)
        last_hidden = self.layer_norm(last_hidden)

        out = self.fc(last_hidden).squeeze(-1)   # (B,)
        return out


# =============================================================================
# 2.  Transformer Model
# =============================================================================

class PSXTransformerModel(nn.Module):
    """
    Transformer encoder with a positional encoding + FC regression head.

    Architecture
    ------------
    Input  (B, seq_len, input_size)
      -> Linear projection  (input_size -> d_model)
      -> Positional Encoding
      -> TransformerEncoder (num_encoder_layers x TransformerEncoderLayer)
      -> mean-pool over seq_len   (B, d_model)
      -> LayerNorm
      -> FC(d_model -> fc_hidden) + ReLU + Dropout
      -> FC(fc_hidden -> 1)
    Output (B,)

    Parameters
    ----------
    input_size        : number of features      (default 26)
    d_model           : internal transformer dim (default 128)
                        Must be divisible by nhead.
    nhead             : attention heads          (default 4)
    dim_feedforward   : FF sublayer width        (default 256)
    num_encoder_layers: stacked encoder layers   (default 2)
    dropout           : dropout throughout       (default 0.2)
    fc_hidden         : FC head hidden units     (default 64)
    max_seq_len       : max sequence length for positional encoding
                        (default 512)
    """

    def __init__(self, input_size=26, d_model=128, nhead=4,
                 dim_feedforward=256, num_encoder_layers=2,
                 dropout=0.2, fc_hidden=64, max_seq_len=512):
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        # Project raw features to transformer dimension
        self.input_proj = nn.Linear(input_size, d_model)

        # Learnable positional encoding
        self.pos_enc = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_encoder_layers,
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()
        log.info(
            "PSXTransformerModel  input=%d  d_model=%d  nhead=%d  "
            "ff=%d  layers=%d  dropout=%.2f",
            input_size, d_model, nhead,
            dim_feedforward, num_encoder_layers, dropout,
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, seq_len, input_size)

        Returns
        -------
        out : (B,)
        """
        B, T, _ = x.shape

        # Project input to d_model
        x = self.input_proj(x)                          # (B, T, d_model)

        # Add positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_enc(positions)                 # (B, T, d_model)

        # Transformer encoder
        x = self.transformer(x)                         # (B, T, d_model)

        # Mean-pool across the time dimension
        x = x.mean(dim=1)                               # (B, d_model)
        x = self.layer_norm(x)

        out = self.fc(x).squeeze(-1)                    # (B,)
        return out


# =============================================================================
# Model factory  (called from trainer.py and pipeline.py)
# =============================================================================

def build_model(cfg):
    """
    Instantiates the correct model from config.yaml.

    config.yaml  model section:
      model:
        architecture:      "lstm"    # or "transformer"
        input_size:        26
        hidden_size:       128       # LSTM only
        num_layers:        2         # LSTM only
        dropout:           0.2
        fc_hidden:         64
        nhead:             4         # Transformer only
        dim_feedforward:   256       # Transformer only
        num_encoder_layers: 2        # Transformer only

    Returns
    -------
    nn.Module
    """
    m    = cfg.get("model", {})
    arch = m.get("architecture", "lstm").lower()

    if arch == "lstm":
        model = PSXLSTMModel(
            input_size  = m.get("input_size",  26),
            hidden_size = m.get("hidden_size", 128),
            num_layers  = m.get("num_layers",  2),
            dropout     = m.get("dropout",     0.2),
            fc_hidden   = m.get("fc_hidden",   64),
        )

    elif arch == "transformer":
        model = PSXTransformerModel(
            input_size         = m.get("input_size",         26),
            d_model            = m.get("hidden_size",        128),
            nhead              = m.get("nhead",              4),
            dim_feedforward    = m.get("dim_feedforward",    256),
            num_encoder_layers = m.get("num_encoder_layers", 2),
            dropout            = m.get("dropout",            0.2),
            fc_hidden          = m.get("fc_hidden",          64),
        )

    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose 'lstm' or 'transformer'."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model built: %s  |  trainable params: %s",
             arch.upper(), f"{n_params:,}")
    return model


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    B, T, F = 32, 30, 26
    dummy   = torch.randn(B, T, F)

    print("\n--- LSTM ---")
    lstm_cfg = {"model": {"architecture": "lstm", "input_size": F}}
    lstm     = build_model(lstm_cfg)
    out      = lstm(dummy)
    print(f"  output shape : {out.shape}   (expected: ({B},))")
    assert out.shape == (B,), "LSTM output shape mismatch"

    print("\n--- Transformer ---")
    tf_cfg = {"model": {"architecture": "transformer", "input_size": F,
                        "hidden_size": 128, "nhead": 4}}
    tf     = build_model(tf_cfg)
    out    = tf(dummy)
    print(f"  output shape : {out.shape}   (expected: ({B},))")
    assert out.shape == (B,), "Transformer output shape mismatch"

    print("\n✓ lstm.py smoke-test passed.")
