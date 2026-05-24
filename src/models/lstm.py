"""
src/models/lstm.py
====================
Deep hierarchical LSTM with cross-scale attention, gated feature fusion,
Fourier temporal encoding, and stacking ensemble with XGBoost / RandomForest
for PSX stock direction classification (UP=1 / DOWN=0).

Architecture
------------
  Input (B, seq_len, n_features)
    -> GatedProjection          (highway connection to d_model)
    -> FourierTemporalEncoding  (learnable + fixed sinusoidal)
    -> MultiScaleLSTMTower      (short / mid / long, parallel BiLSTMs)
    -> CrossScaleAttention      (each scale attends to all others)
    -> TemporalAttention        (intra-scale self-attention + rel-pos bias)
    -> Hierarchical gated fusion + scale gate
    -> LearnableTemporalPool    (soft + mean + max)
    -> ClassificationHead       (GELU FC -> single logit)
    -> [Optional] XGBoost + RandomForest on encode() features
    -> [Optional] Logistic Regression meta-learner (stacking)

Output
------
  forward()       -> (B,)  raw logit   use BCEWithLogitsLoss in trainer
  encode()        -> (B, D*3)  pooled representation for tree ensemble
  predict_proba() -> (B,)  sigmoid probability  (inference only)

config.yaml keys used
---------------------
  model:
    input_size:   31
    lstm_hidden:  256
    lstm_layers:  4
    lstm_dropout: 0.3
    fc_hidden:    256
    num_heads:    8
    cross_heads:  4
    use_ensemble: true
    xgb_trees:    500
    xgb_depth:    6
    xgb_lr:       0.02
    rf_trees:     300
    rf_depth:     14
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FOURIER TEMPORAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class FourierTemporalEncoding(nn.Module):
    """
    Combines learnable frequency embeddings with fixed sinusoidal encoding.
    Gives the model a richer sense of time within each sequence.
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.freq  = nn.Parameter(torch.randn(d_model // 2) * 0.02)
        self.phase = nn.Parameter(torch.zeros(d_model // 2))
        self.proj  = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        t = torch.arange(T, device=x.device).float().unsqueeze(1)
        learned = torch.cat([
            torch.sin(t * self.freq + self.phase),
            torch.cos(t * self.freq + self.phase),
        ], dim=-1).unsqueeze(0)
        encoding = self.proj(learned + self.pe[:, :T, :])
        return x + encoding


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GATED HIGHWAY FEATURE PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

class GatedProjection(nn.Module):
    """
    Projects raw features to d_model with a highway gating mechanism.
    Gate controls how much of the nonlinear vs linear signal passes through.
    """
    def __init__(self, in_features, d_model, dropout=0.1):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate     = nn.Sequential(nn.Linear(in_features, d_model), nn.Sigmoid())
        self.shortcut = nn.Linear(in_features, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        g = self.gate(x)
        return self.norm(g * self.transform(x) + (1 - g) * self.shortcut(x))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MULTI-SCALE LSTM TOWER
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleLSTMTower(nn.Module):
    """
    Three independent Bidirectional LSTMs in parallel:
      Scale 0 (short) – shallow, narrow   -> fast patterns
      Scale 1 (mid)   – medium depth      -> medium-term momentum
      Scale 2 (long)  – deep, wide        -> macro regime features
    All scales project back to d_model so they can be fused uniformly.
    """
    def __init__(self, d_model, hidden_size, num_layers, dropout):
        super().__init__()
        configs = [
            dict(hidden=hidden_size // 2, layers=max(1, num_layers - 1)),
            dict(hidden=hidden_size,      layers=num_layers),
            dict(hidden=hidden_size * 2,  layers=num_layers + 1),
        ]
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size    = d_model,
                hidden_size   = c["hidden"],
                num_layers    = c["layers"],
                dropout       = dropout if c["layers"] > 1 else 0.0,
                batch_first   = True,
                bidirectional = True,
            )
            for c in configs
        ])
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c["hidden"] * 2, d_model),
                nn.LayerNorm(d_model),
            )
            for c in configs
        ])
        self.n_scales = len(configs)

    def forward(self, x):
        outputs = []
        for lstm, proj in zip(self.lstms, self.projs):
            out, _ = lstm(x)
            outputs.append(proj(out))
        return outputs   # list of 3 tensors, each (B, T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CROSS-SCALE ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class CrossScaleAttention(nn.Module):
    """
    Each scale acts as query; all other scales (concatenated over time)
    act as key/value. Lets the short-term LSTM query macro regime context
    from the long-term LSTM.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_scale, context_scales):
        ctx = torch.stack(context_scales, dim=2)   # (B, T, N, D)
        B, T, N, D = ctx.shape
        ctx = ctx.view(B, T * N, D)
        out, _ = self.attn(query_scale, ctx, ctx)
        return self.norm(query_scale + self.dropout(out))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INTRA-SCALE TEMPORAL ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the time axis within a single scale,
    with learnable relative position bias.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1, max_len=512):
        super().__init__()
        self.attn      = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm      = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.rel_bias  = nn.Embedding(2 * max_len - 1, num_heads)
        self.num_heads = num_heads
        self.max_len   = max_len

    def _rel_bias(self, T, device):
        pos  = torch.arange(T, device=device)
        rel  = pos.unsqueeze(1) - pos.unsqueeze(0) + self.max_len - 1
        rel  = rel.clamp(0, 2 * self.max_len - 2)
        bias = self.rel_bias(rel)
        return bias.permute(2, 0, 1)

    def forward(self, x):
        B, T, D   = x.shape
        bias      = self._rel_bias(T, x.device)
        attn_mask = bias.unsqueeze(0).expand(B, -1, -1, -1)
        attn_mask = attn_mask.reshape(B * self.num_heads, T, T)
        out, _    = self.attn(x, x, x, attn_mask=attn_mask)
        return self.norm(x + self.dropout(out))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LEARNABLE TEMPORAL POOLING
# ─────────────────────────────────────────────────────────────────────────────

class LearnableTemporalPool(nn.Module):
    """
    Soft weighted sum over timesteps concatenated with mean and max pooling.
    Output size: d_model * 3
    """
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        weights  = F.softmax(self.score(x).squeeze(-1), dim=-1)  # (B, T)
        soft     = (x * weights.unsqueeze(-1)).sum(1)             # (B, D)
        mean_out = x.mean(1)
        max_out  = x.max(1).values
        return torch.cat([soft, mean_out, max_out], dim=-1)       # (B, D*3)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Two-layer GELU MLP -> single logit.
    Use BCEWithLogitsLoss in trainer (applies sigmoid internally).
    At inference use torch.sigmoid(model(x)) to get probability.
    """
    def __init__(self, in_features, fc_hidden, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, fc_hidden * 2),
            nn.LayerNorm(fc_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden * 2, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,)  raw logit


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FULL DEEP LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PSXLSTMModel(nn.Module):
    """
    Full architecture — see module docstring for details.

    Parameters
    ----------
    input_size  : number of input features (default 31)
    hidden_size : LSTM hidden dimension    (default 256)
    num_layers  : base LSTM depth          (default 4)
    dropout     : dropout rate            (default 0.3)
    fc_hidden   : FC head hidden dim      (default 256)
    num_heads   : temporal attention heads (default 8)
    cross_heads : cross-scale attn heads  (default 4)
    """

    def __init__(self, input_size=31, hidden_size=256,
                 num_layers=4, dropout=0.3, fc_hidden=256,
                 num_heads=8, cross_heads=4):
        super().__init__()
        D = hidden_size

        self.input_proj   = GatedProjection(input_size, D, dropout)
        self.temporal_enc = FourierTemporalEncoding(D)
        self.ms_lstm      = MultiScaleLSTMTower(D, hidden_size, num_layers, dropout)
        n_scales          = self.ms_lstm.n_scales

        self.cross_attn = nn.ModuleList([
            CrossScaleAttention(D, num_heads=cross_heads, dropout=dropout)
            for _ in range(n_scales)
        ])
        self.temp_attn = nn.ModuleList([
            TemporalAttention(D, num_heads=num_heads, dropout=dropout)
            for _ in range(n_scales)
        ])
        self.scale_gate = nn.Sequential(
            nn.Linear(D * n_scales, n_scales),
            nn.Softmax(dim=-1),
        )

        self.pool = LearnableTemporalPool(D)
        self.head = ClassificationHead(D * 3, fc_hidden, dropout)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            "PSXLSTMModel  input=%d  hidden=%d  layers=%d  dropout=%.2f  "
            "fc_hidden=%d  heads=%d  cross_heads=%d  scales=3  params=%s",
            input_size, hidden_size, num_layers, dropout,
            fc_hidden, num_heads, cross_heads, f"{n_params:,}",
        )

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode(self, x):
        """
        Returns pooled representation before the classification head.
        Used by TreeEnsemble to extract features for XGBoost / RF.

        Parameters
        ----------
        x : (B, seq_len, input_size)

        Returns
        -------
        pooled : (B, hidden_size * 3)
        """
        x          = self.input_proj(x)
        x          = self.temporal_enc(x)
        scale_outs = self.ms_lstm(x)

        cross_outs = [
            self.cross_attn[i](
                scale_outs[i],
                [scale_outs[j] for j in range(len(scale_outs)) if j != i]
            )
            for i in range(len(scale_outs))
        ]
        attn_outs = [self.temp_attn[i](cross_outs[i]) for i in range(len(cross_outs))]

        stacked = torch.stack(attn_outs, dim=-1)                      # (B, T, D, S)
        gates   = self.scale_gate(torch.cat(attn_outs, dim=-1))       # (B, T, S)
        fused   = (stacked * gates.unsqueeze(2)).sum(-1)              # (B, T, D)
        return self.pool(fused)                                        # (B, D*3)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, seq_len, input_size)

        Returns
        -------
        logit : (B,)  raw logit — pass to BCEWithLogitsLoss during training
                      use torch.sigmoid(logit) to get probability at inference
        """
        return self.head(self.encode(x))

    @torch.no_grad()
    def predict_proba(self, x):
        """Convenience method for inference. Returns UP probability (B,)."""
        self.eval()
        return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  TREE ENSEMBLE  (XGBoost + RandomForest + LR meta-learner)
# ─────────────────────────────────────────────────────────────────────────────

class TreeEnsemble:
    """
    Stacking ensemble on top of a trained PSXLSTMModel.

    Stage 1 — XGBoost + RandomForest trained on LSTM encode() features
               from the TRAIN set.
    Stage 2 — Logistic Regression meta-learner trained on VAL set
               probabilities [lstm_prob, xgb_prob, rf_prob] to avoid leakage.

    Usage
    -----
    ensemble = TreeEnsemble(lstm_model, cfg)
    ensemble.fit(train_loader, val_loader, device)
    probs = ensemble.predict_proba(test_loader, device)   # (N,)
    preds = ensemble.predict(test_loader, device)         # (N,) binary
    """

    def __init__(self, lstm_model, cfg):
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        m         = cfg.get("model", {})
        self.lstm = lstm_model

        self.xgb = XGBClassifier(
            n_estimators     = m.get("xgb_trees", 500),
            max_depth        = m.get("xgb_depth", 6),
            learning_rate    = m.get("xgb_lr", 0.02),
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 3,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            eval_metric      = "logloss",
            random_state     = 42,
            verbosity        = 0,
        )
        self.rf = RandomForestClassifier(
            n_estimators = m.get("rf_trees", 300),
            max_depth    = m.get("rf_depth", 14),
            max_features = 0.6,
            n_jobs       = -1,
            random_state = 42,
        )
        # Meta-learner: 3 inputs (lstm_prob, xgb_prob, rf_prob) -> direction
        self.meta = LogisticRegression(C=1.0, max_iter=1000)

    @torch.no_grad()
    def _get_features_and_probs(self, loader, device):
        """
        Returns encoded features, lstm probabilities, and true labels
        for all batches in a loader.
        """
        self.lstm.eval()
        feats, probs, targets = [], [], []
        for xb, yb in loader:
            xb = xb.to(device)
            feats.append(self.lstm.encode(xb).cpu().numpy())
            probs.append(torch.sigmoid(self.lstm(xb)).cpu().numpy())
            targets.append(yb.numpy())
        return (
            np.concatenate(feats),     # (N, D*3)
            np.concatenate(probs),     # (N,)
            np.concatenate(targets),   # (N,)
        )

    def fit(self, train_loader, val_loader, device):
        """
        Train XGB + RF on train encoded features.
        Train meta-learner on val set probabilities (avoids leakage).
        """
        log.info("TreeEnsemble.fit — extracting train features ...")
        X_train, _, y_train = self._get_features_and_probs(train_loader, device)

        log.info("Fitting XGBoost on %d samples, %d features ...", *X_train.shape)
        self.xgb.fit(X_train, y_train)

        log.info("Fitting RandomForest ...")
        self.rf.fit(X_train, y_train)

        log.info("Building meta-features on val set ...")
        X_val, lstm_val_prob, y_val = self._get_features_and_probs(val_loader, device)

        xgb_val_prob = self.xgb.predict_proba(X_val)[:, 1]
        rf_val_prob  = self.rf.predict_proba(X_val)[:, 1]
        meta_val     = np.stack([lstm_val_prob, xgb_val_prob, rf_val_prob], axis=1)

        log.info("Fitting meta-learner on %d val samples ...", len(y_val))
        self.meta.fit(meta_val, y_val)

        coefs = self.meta.coef_[0]
        log.info(
            "Meta-learner coefs — LSTM: %.3f  XGB: %.3f  RF: %.3f",
            coefs[0], coefs[1], coefs[2],
        )

    def predict_proba(self, loader, device):
        """Returns blended UP probability. Shape: (N,)"""
        X_enc, lstm_prob, _ = self._get_features_and_probs(loader, device)
        xgb_prob  = self.xgb.predict_proba(X_enc)[:, 1]
        rf_prob   = self.rf.predict_proba(X_enc)[:, 1]
        meta_X    = np.stack([lstm_prob, xgb_prob, rf_prob], axis=1)
        return self.meta.predict_proba(meta_X)[:, 1]

    def predict(self, loader, device, threshold=0.5):
        """Returns binary predictions (0=DOWN, 1=UP). Shape: (N,)"""
        return (self.predict_proba(loader, device) >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  BUILD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg):
    """
    Instantiates PSXLSTMModel from config.yaml model section.

    Parameters
    ----------
    cfg : dict  (loaded config.yaml)

    Returns
    -------
    PSXLSTMModel
    """
    m     = cfg.get("model", {})
    model = PSXLSTMModel(
        input_size  = m.get("input_size",   31),
        hidden_size = m.get("lstm_hidden",  256),
        num_layers  = m.get("lstm_layers",   4),
        dropout     = m.get("lstm_dropout", 0.3),
        fc_hidden   = m.get("fc_hidden",    256),
        num_heads   = m.get("num_heads",     8),
        cross_heads = m.get("cross_heads",   4),
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model built: PSXLSTMModel | trainable params: %s", f"{n:,}")
    return model


def build_ensemble(cfg, lstm_model, train_loader, val_loader, device):
    """
    Wraps a trained PSXLSTMModel with XGBoost + RF + LR stacking ensemble.
    Call AFTER lstm_model has been fully trained.

    Parameters
    ----------
    cfg          : dict  (loaded config.yaml)
    lstm_model   : trained PSXLSTMModel
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split (used for meta-learner)
    device       : torch.device

    Returns
    -------
    TreeEnsemble or None if use_ensemble=false in config
    """
    if not cfg.get("model", {}).get("use_ensemble", False):
        log.info("Ensemble disabled in config — skipping.")
        return None
    ensemble = TreeEnsemble(lstm_model, cfg)
    ensemble.fit(train_loader, val_loader, device)
    return ensemble
