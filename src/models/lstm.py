"""
src/models/lstm.py
====================
Simple single-layer Bidirectional LSTM for PSX stock direction
classification (UP=1 / DOWN=0).

Architecture
------------
  Input (B, seq_len, n_features)
    -> Linear input projection + LayerNorm
    -> Bidirectional LSTM
    -> Dropout
    -> mean + last timestep pooling (concatenated)
    -> FC head (Linear -> BN -> GELU -> Dropout -> Linear)
    -> single logit

Output
------
  forward()       -> (B,)  raw logit   use BCEWithLogitsLoss in trainer
  encode()        -> (B, hidden*4)  pooled representation
  predict_proba() -> (B,)  sigmoid probability  (inference only)

config.yaml keys used
---------------------
  model:
    input_size:   31
    lstm_hidden:  64
    lstm_layers:  1
    lstm_dropout: 0.3
    fc_hidden:    64
    use_ensemble: false
"""

import logging
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ── PSX LSTM Model ────────────────────────────────────────────────────────────

class PSXLSTMModel(nn.Module):
    """
    Lightweight Bidirectional LSTM for binary stock direction prediction.

    Parameters
    ----------
    input_size  : number of input features  (default 31)
    hidden_size : LSTM hidden dimension     (default 64)
    num_layers  : LSTM depth                (default 1)
    dropout     : dropout rate             (default 0.3)
    fc_hidden   : FC head hidden dim       (default 64)
    """

    def __init__(self, input_size=31, hidden_size=64,
                 num_layers=1, dropout=0.3, fc_hidden=64,
                 **kwargs):   # absorbs unused keys like num_heads, cross_heads
        super().__init__()

        # -- input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # -- BiLSTM
        self.lstm = nn.LSTM(
            input_size    = hidden_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = True,
        )
        self.lstm_drop = nn.Dropout(dropout)

        # -- FC head
        # encode() output: mean_pool (H*2) + last_timestep (H*2) = H*4
        head_in = hidden_size * 4
        self.head = nn.Sequential(
            nn.Linear(head_in, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            "PSXLSTMModel  input=%d  hidden=%d  layers=%d  "
            "dropout=%.2f  fc_hidden=%d  params=%s",
            input_size, hidden_size, num_layers,
            dropout, fc_hidden, f"{n_params:,}",
        )

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        """
        Parameters
        ----------
        x : (B, seq_len, input_size)

        Returns
        -------
        pooled : (B, hidden_size * 4)
        """
        x   = self.input_proj(x)              # (B, T, H)
        out, _ = self.lstm(x)                 # (B, T, H*2)
        out = self.lstm_drop(out)

        mean_pool = out.mean(dim=1)           # (B, H*2)
        last_step = out[:, -1, :]             # (B, H*2)
        return torch.cat([mean_pool, last_step], dim=-1)  # (B, H*4)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, seq_len, input_size)

        Returns
        -------
        logit : (B,) raw logit
        """
        return self.head(self.encode(x)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x):
        """Convenience method for inference. Returns UP probability (B,)."""
        self.eval()
        return torch.sigmoid(self.forward(x))


# ── Tree Ensemble (XGBoost + RF + LR meta-learner) ────────────────────────────

class TreeEnsemble:
    """
    Stacking ensemble on top of a trained PSXLSTMModel.

    Stage 1 — XGBoost + RandomForest trained on LSTM encode() features
               from the TRAIN set.
    Stage 2 — Logistic Regression meta-learner trained on VAL set
               probabilities [lstm_prob, xgb_prob, rf_prob].
               Using val set for meta-learner avoids train leakage.
    """

    def __init__(self, lstm_model, cfg):
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        m         = cfg.get("model", {})
        self.lstm = lstm_model

        self.xgb = XGBClassifier(
            n_estimators     = m.get("xgb_trees",  300),
            max_depth        = m.get("xgb_depth",    6),
            learning_rate    = m.get("xgb_lr",    0.02),
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
            n_estimators = m.get("rf_trees",  200),
            max_depth    = m.get("rf_depth",   14),
            max_features = 0.6,
            n_jobs       = -1,
            random_state = 42,
        )
        self.meta = LogisticRegression(C=1.0, max_iter=1000)

    @torch.no_grad()
    def _get_features_and_probs(self, loader, device):
        self.lstm.eval()
        feats, probs, targets = [], [], []
        for xb, yb in loader:
            xb = xb.to(device)
            feats.append(self.lstm.encode(xb).cpu().numpy())
            probs.append(torch.sigmoid(self.lstm(xb)).cpu().numpy())
            targets.append(yb.numpy())
        return (
            np.concatenate(feats),
            np.concatenate(probs),
            np.concatenate(targets),
        )

    def fit(self, train_loader, val_loader, device):
        log.info("TreeEnsemble.fit — extracting train features ...")
        X_train, _, y_train = self._get_features_and_probs(train_loader, device)

        log.info("Fitting XGBoost on %d samples %d features ...", *X_train.shape)
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
        X_enc, lstm_prob, _ = self._get_features_and_probs(loader, device)
        xgb_prob = self.xgb.predict_proba(X_enc)[:, 1]
        rf_prob  = self.rf.predict_proba(X_enc)[:, 1]
        meta_X   = np.stack([lstm_prob, xgb_prob, rf_prob], axis=1)
        return self.meta.predict_proba(meta_X)[:, 1]

    def predict(self, loader, device, threshold=0.5):
        return (self.predict_proba(loader, device) >= threshold).astype(int)


# ── Build Functions ───────────────────────────────────────────────────────────

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
        input_size  = m.get("input_size",    31),
        hidden_size = m.get("lstm_hidden",   64),
        num_layers  = m.get("lstm_layers",    1),
        dropout     = m.get("lstm_dropout", 0.3),
        fc_hidden   = m.get("fc_hidden",     64),
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model built: PSXLSTMModel | trainable params: %s", f"{n:,}")
    return model


def build_ensemble(cfg, lstm_model, train_loader, val_loader, device):
    """
    Wraps a trained PSXLSTMModel with XGBoost + RF + LR stacking ensemble.
    Call AFTER lstm_model has been fully trained.
    """
    if not cfg.get("model", {}).get("use_ensemble", False):
        log.info("Ensemble disabled in config — skipping.")
        return None
    ensemble = TreeEnsemble(lstm_model, cfg)
    ensemble.fit(train_loader, val_loader, device)
    return ensemble


if __name__ == "__main__":
    import yaml, os
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = build_model(cfg).to(device)

    B, T, F = 8, cfg["model"]["seq_len"], cfg["model"]["input_size"]
    dummy   = torch.randn(B, T, F).to(device)
    logits  = model(dummy)
    encoded = model.encode(dummy)
    probs   = model.predict_proba(dummy)

    print(f"Smoke test passed:")
    print(f"  Input   : {dummy.shape}")
    print(f"  Logits  : {logits.shape}")
    print(f"  Encoded : {encoded.shape}")
    print(f"  Probs   : {probs.shape}  range [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Params  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
