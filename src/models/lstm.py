"""
src/models/lstm.py
====================
Deep hierarchical LSTM with cross-scale attention, gated feature fusion,
Fourier temporal encoding, uncertainty-aware output, and optional ensemble
with XGBoost / Random Forest for PSX return prediction.

input  : (batch, seq_len, n_features)   float32
output : (batch,)                       float32  scaled return

Architecture
------------
  Input
    -> Fourier Temporal Encoding (learnable + fixed sinusoidal)
    -> Gated Feature Projection (highway connection)
    -> Multi-Scale LSTM Tower  (short / mid / long horizon, parallel)
    -> Cross-Scale Attention   (each scale attends to all others)
    -> Intra-Scale Temporal Attention (per-scale self-attention)
    -> Hierarchical Residual Fusion + LayerNorm
    -> Learnable Temporal Pooling (soft attention weights)
    -> Deep GELU FC Head with uncertainty estimation
    -> [Optional] XGBoost / RandomForest ensemble on extracted features
    -> [Optional] Stacking meta-learner (Ridge) to blend all predictions

config.yaml keys used:
  model:
    architecture:    "lstm"
    input_size:      50
    lstm_hidden:     256
    lstm_layers:     4
    lstm_dropout:    0.3
    fc_hidden:       256
    num_heads:       8
    cross_heads:     4
    use_ensemble:    true
    xgb_trees:       1000
    xgb_depth:       6
    xgb_lr:          0.02
    rf_trees:        500
    rf_depth:        14
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
        pe = torch.zeros(max_len, d_model)
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
    Gate controls how much of the raw (linearly projected) signal passes
    vs. the nonlinear transformed signal.
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
      Scale 0 (short)  – shallow, narrow   -> fast patterns
      Scale 1 (mid)    – medium depth      -> medium-term momentum
      Scale 2 (long)   – deep, wide        -> macro regime features
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
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CROSS-SCALE ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class CrossScaleAttention(nn.Module):
    """
    Each scale acts as query; all other scales (concatenated) act as
    key/value. Lets the short-term LSTM ask the long-term LSTM about
    macro regime context.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_scale, context_scales):
        ctx = torch.stack(context_scales, dim=2)
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
        B, T, D = x.shape
        bias      = self._rel_bias(T, x.device)
        attn_mask = bias.unsqueeze(0).expand(B, -1, -1, -1)
        attn_mask = attn_mask.reshape(B * self.num_heads, T, T)
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return self.norm(x + self.dropout(out))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LEARNABLE TEMPORAL POOLING
# ─────────────────────────────────────────────────────────────────────────────

class LearnableTemporalPool(nn.Module):
    """
    Soft weighted sum over timesteps + mean + max concatenated.
    """
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        weights  = self.score(x).squeeze(-1)
        weights  = F.softmax(weights, dim=-1)
        soft     = (x * weights.unsqueeze(-1)).sum(1)
        mean_out = x.mean(1)
        max_out  = x.max(1).values
        return torch.cat([soft, mean_out, max_out], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  UNCERTAINTY-AWARE OUTPUT HEAD
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyHead(nn.Module):
    """
    Outputs point prediction (mu) + log-variance for Gaussian NLL training.
    """
    def __init__(self, in_features, fc_hidden, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, fc_hidden * 2),
            nn.LayerNorm(fc_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden * 2, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.mu      = nn.Linear(fc_hidden, 1)
        self.log_var = nn.Linear(fc_hidden, 1)

    def forward(self, x, return_uncertainty=False):
        h       = self.shared(x)
        mu      = self.mu(h).squeeze(-1)
        log_var = self.log_var(h).squeeze(-1)
        if return_uncertainty:
            return mu, log_var
        return mu


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FULL DEEP LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PSXLSTMModel(nn.Module):
    """
    Full architecture:
      FourierTemporalEncoding
      -> GatedProjection
      -> MultiScaleLSTMTower  (short / mid / long)
      -> CrossScaleAttention  (per scale)
      -> TemporalAttention    (per scale, with relative position bias)
      -> Hierarchical fusion  (learned gate across scales)
      -> LearnableTemporalPool (soft + mean + max)
      -> UncertaintyHead
    """

    def __init__(self, input_size=50, hidden_size=256,
                 num_layers=4, dropout=0.3, fc_hidden=256,
                 num_heads=8, cross_heads=4):
        super().__init__()
        D = hidden_size

        self.temporal_enc = FourierTemporalEncoding(D)
        self.input_proj   = GatedProjection(input_size, D, dropout)

        self.ms_lstm  = MultiScaleLSTMTower(D, hidden_size, num_layers, dropout)
        n_scales      = self.ms_lstm.n_scales

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
        self.head = UncertaintyHead(D * 3, fc_hidden, dropout)

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

    def forward(self, x, return_uncertainty=False):
        """
        Parameters
        ----------
        x                  : (B, seq_len, input_size)
        return_uncertainty : if True, also returns log-variance tensor

        Returns
        -------
        mu      : (B,)
        log_var : (B,)   only when return_uncertainty=True
        """
        x = self.input_proj(x)
        x = self.temporal_enc(x)

        scale_outs = self.ms_lstm(x)

        cross_outs = []
        for i, ca in enumerate(self.cross_attn):
            others = [scale_outs[j] for j in range(len(scale_outs)) if j != i]
            cross_outs.append(ca(scale_outs[i], others))

        attn_outs = [self.temp_attn[i](cross_outs[i])
                     for i in range(len(cross_outs))]

        stacked = torch.stack(attn_outs, dim=-1)
        B, T, D, S = stacked.shape
        cat   = torch.cat(attn_outs, dim=-1)
        gates = self.scale_gate(cat).unsqueeze(2)
        fused = (stacked * gates).sum(-1)

        pooled = self.pool(fused)
        return self.head(pooled, return_uncertainty)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  TREE-BASED ENSEMBLE WRAPPER  (XGBoost + RandomForest)
# ─────────────────────────────────────────────────────────────────────────────

class TreeEnsemble:
    """
    Trains XGBoost and RandomForest on pooled LSTM features, then blends
    all three predictions with a Ridge meta-learner (stacking).

    Usage
    -----
    ensemble = TreeEnsemble(lstm_model, cfg)
    ensemble.fit(train_loader, device)
    preds = ensemble.predict(val_loader, device)
    """

    def __init__(self, lstm_model, cfg):
        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge

        m = cfg.get("model", {})
        self.lstm = lstm_model
        self.xgb  = XGBRegressor(
            n_estimators     = m.get("xgb_trees",  1000),
            max_depth        = m.get("xgb_depth",     6),
            learning_rate    = m.get("xgb_lr",      0.02),
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 3,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            tree_method      = "hist",
            random_state     = 42,
        )
        self.rf   = RandomForestRegressor(
            n_estimators = m.get("rf_trees", 500),
            max_depth    = m.get("rf_depth",  14),
            max_features = 0.6,
            n_jobs       = -1,
            random_state = 42,
        )
        self.meta = Ridge(alpha=1.0)

    @torch.no_grad()
    def _extract_features(self, loader, device):
        self.lstm.eval()
        feats, targets = [], []
        for xb, yb in loader:
            xb = xb.to(device)
            x  = self.lstm.input_proj(xb)
            x  = self.lstm.temporal_enc(x)
            sc = self.lstm.ms_lstm(x)
            co = [self.lstm.cross_attn[i](sc[i],
                  [sc[j] for j in range(len(sc)) if j != i])
                  for i in range(len(sc))]
            ao = [self.lstm.temp_attn[i](co[i]) for i in range(len(co))]
            st = torch.stack(ao, dim=-1)
            B, T, D, S = st.shape
            ga = self.lstm.scale_gate(torch.cat(ao, dim=-1)).unsqueeze(2)
            fu = (st * ga).sum(-1)
            po = self.lstm.pool(fu)
            feats.append(po.cpu().numpy())
            targets.append(yb.cpu().numpy())
        return np.concatenate(feats), np.concatenate(targets)

    def fit(self, train_loader, device):
        log.info("Extracting LSTM features for tree ensemble ...")
        X, y = self._extract_features(train_loader, device)
        log.info("Fitting XGBoost on %d samples, %d features ...", *X.shape)
        self.xgb.fit(X, y)
        log.info("Fitting RandomForest ...")
        self.rf.fit(X, y)
        p_xgb  = self.xgb.predict(X).reshape(-1, 1)
        p_rf   = self.rf.predict(X).reshape(-1, 1)
        with torch.no_grad():
            p_lstm = []
            for xb, _ in train_loader:
                p_lstm.append(self.lstm(xb.to(device)).cpu().numpy())
        p_lstm = np.concatenate(p_lstm).reshape(-1, 1)
        meta_X = np.hstack([p_lstm, p_xgb, p_rf])
        self.meta.fit(meta_X, y)
        log.info("Meta-learner fitted. Blend coefs: %s", self.meta.coef_.round(3))

    def predict(self, loader, device):
        X, _ = self._extract_features(loader, device)
        p_xgb  = self.xgb.predict(X).reshape(-1, 1)
        p_rf   = self.rf.predict(X).reshape(-1, 1)
        with torch.no_grad():
            p_lstm = []
            for xb, _ in loader:
                p_lstm.append(self.lstm(xb.to(device)).cpu().numpy())
        p_lstm = np.concatenate(p_lstm).reshape(-1, 1)
        meta_X = np.hstack([p_lstm, p_xgb, p_rf])
        return self.meta.predict(meta_X)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  GAUSSIAN NLL LOSS
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll_loss(mu, log_var, target):
    """
    Gaussian negative log-likelihood loss.
    Penalises wrong predictions AND miscalibrated uncertainty.
    Use when return_uncertainty=True.
    """
    var = torch.exp(log_var).clamp(min=1e-6)
    return (0.5 * (log_var + (target - mu) ** 2 / var)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 11.  BUILD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg):
    m     = cfg.get("model", {})
    model = PSXLSTMModel(
        input_size  = m.get("input_size",   50),
        hidden_size = m.get("lstm_hidden",  256),
        num_layers  = m.get("lstm_layers",    4),
        dropout     = m.get("lstm_dropout", 0.3),
        fc_hidden   = m.get("fc_hidden",    256),
        num_heads   = m.get("num_heads",      8),
        cross_heads = m.get("cross_heads",    4),
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model built: Deep Hierarchical LSTM  |  trainable params: %s", f"{n:,}")
    return model


def build_ensemble(cfg, lstm_model, train_loader, device):
    """
    Wraps a trained PSXLSTMModel with XGBoost + RF ensemble.
    Call AFTER lstm_model has been fully trained.
    Returns TreeEnsemble or None if use_ensemble=false in config.
    """
    if not cfg.get("model", {}).get("use_ensemble", False):
        log.info("Ensemble disabled in config — returning None.")
        return None
    ensemble = TreeEnsemble(lstm_model, cfg)
    ensemble.fit(train_loader, device)
    return ensemble
