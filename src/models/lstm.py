"""
src/models/lstm.py
====================
LSTM architecture for PSX return prediction.

input  : (batch, seq_len, n_features)   float32
output : (batch,)                       float32  scaled return

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

log = logging.getLogger(__name__)


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
    """

    def __init__(self, input_size=32, hidden_size=128,
                 num_layers=2, dropout=0.2, fc_hidden=64):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

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
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        last_hidden = self.layer_norm(last_hidden)
        out = self.fc(last_hidden).squeeze(-1)
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
        num_layers  = m.get("lstm_layers",  2),
        dropout     = m.get("lstm_dropout", 0.2),
        fc_hidden   = m.get("fc_hidden",    64),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model built: LSTM  |  trainable params: %s", f"{n_params:,}")
    return model
