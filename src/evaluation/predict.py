"""
src/evaluation/predict.py
===========================
Loads a saved checkpoint and generates predictions for a ticker.

What this module does
---------------------
  1. Loads the best checkpoint from checkpoints/<TICKER>_best.pt
  2. Loads the matching scaler artefact from data/processed/<TICKER>_scaler.pkl
  3. Runs inference on the test split (and optionally val + train)
  4. Inverse-transforms scaled predictions back to real log-returns
  5. Converts log-returns to percentage price changes for readability
  6. Saves predictions to data/predictions/<TICKER>_predictions.csv

Output CSV columns
------------------
  date          — trading date
  close         — actual closing price (unscaled, for reference)
  actual_return — actual next-day log return  (real scale)
  pred_return   — predicted next-day log return (real scale)
  actual_pct    — actual_return as % change
  pred_pct      — pred_return   as % change
  split         — "train" / "val" / "test"

config.yaml section used
------------------------
  data:
    processed_dir:    data/processed
    predictions_dir:  data/predictions
  model:
    checkpoints_dir:  src/models/checkpoints
  training:
    seq_len:     30
    batch_size:  64
    num_workers: 2
"""

import os
import logging
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


# =============================================================================
# Core inference function
# =============================================================================

def predict_split(model, loader, device):
    """
    Runs inference on one DataLoader split.

    Returns
    -------
    np.array of shape (N,)  — scaled predictions
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            out     = model(X_batch)            # (B,)
            preds.append(out.cpu().numpy())

    return np.concatenate(preds, axis=0)        # (N,)


# =============================================================================
# Build prediction DataFrame for one ticker
# =============================================================================

def build_prediction_df(ticker, scaled_dict, scaler_dict,
                        cfg, model=None):
    """
    Generates a prediction DataFrame for all three splits of one ticker.

    Parameters
    ----------
    ticker       : ticker symbol string  e.g. "OGDC.KA"
    scaled_dict  : { ticker -> (train_scaled, val_scaled, test_scaled) }
    scaler_dict  : { ticker -> (feature_scaler, target_scaler) }
    cfg          : full config dict
    model        : pre-loaded nn.Module  (optional).
                   If None, the checkpoint is loaded from disk.

    Returns
    -------
    pd.DataFrame  with columns described in module docstring
    """
    from src.models.lstm    import build_model
    from src.models.dataset import make_single_loader
    from src.models.trainer import load_checkpoint
    from src.models.scaler  import inverse_transform_target

    t_cfg       = cfg.get("training", {})
    seq_len     = t_cfg.get("seq_len",     30)
    batch_size  = t_cfg.get("batch_size",  64)
    num_workers = t_cfg.get("num_workers",  2)

    ckpt_dir    = cfg["model"]["checkpoints_dir"]
    ckpt_path   = os.path.join(ckpt_dir, f"{ticker}_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────
    if model is None:
        model = build_model(cfg)
        load_checkpoint(ckpt_path, model, device=str(device))
    model.to(device)

    _, target_scaler = scaler_dict[ticker]
    train_s, val_s, test_s = scaled_dict[ticker]

    feature_cols = [
        c for c in train_s.columns
        if c not in ("date", "close", "ret_1d_future")
    ]
    target_col = "ret_1d_future"

    frames = []

    for split_name, df_split in [
        ("train", train_s),
        ("val",   val_s),
        ("test",  test_s),
    ]:
        loader = make_single_loader(
            df_split, feature_cols, target_col,
            seq_len     = seq_len,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
        )

        # Number of valid windows = len(df_split) - seq_len
        n_windows = len(df_split) - seq_len

        # Rows that have a corresponding prediction window
        # (the first seq_len rows are "warm-up" and have no prediction)
        df_pred = df_split.iloc[seq_len : seq_len + n_windows].copy()

        scaled_preds = predict_split(model, loader, device)
        real_preds   = inverse_transform_target(scaled_preds, target_scaler)

        # Inverse-transform actual targets too
        actual_scaled = df_pred[target_col].values.reshape(-1, 1)
        real_actual   = target_scaler.inverse_transform(
            actual_scaled
        ).flatten()

        df_out = pd.DataFrame({
            "date":          df_pred["date"].values,
            "close":         df_pred["close"].values,
            "actual_return": real_actual,
            "pred_return":   real_preds,
            "actual_pct":    (np.exp(real_actual) - 1) * 100,
            "pred_pct":      (np.exp(real_preds)  - 1) * 100,
            "split":         split_name,
        })
        frames.append(df_out)
        log.info(
            "%s [%s]  windows=%d  "
            "MAE=%.6f  RMSE=%.6f",
            ticker, split_name, len(df_out),
            np.mean(np.abs(real_preds - real_actual)),
            np.sqrt(np.mean((real_preds - real_actual) ** 2)),
        )

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Metrics summary
# =============================================================================

def compute_metrics(pred_df, split="test"):
    """
    Returns a dict of regression metrics for one split.

    Metrics
    -------
    MAE   — mean absolute error  (log-return scale)
    RMSE  — root mean squared error
    MAPE  — mean absolute % error on pct changes
    DirectionalAccuracy — fraction of days where sign(pred) == sign(actual)
    """
    df = pred_df[pred_df["split"] == split].copy()
    if df.empty:
        log.warning("No rows for split=%s", split)
        return {}

    actual = df["actual_return"].values
    pred   = df["pred_return"].values

    mae   = float(np.mean(np.abs(pred - actual)))
    rmse  = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mape  = float(np.mean(np.abs(
        (df["actual_pct"].values - df["pred_pct"].values)
        / (np.abs(df["actual_pct"].values) + 1e-8)
    )) * 100)
    dir_acc = float(np.mean(np.sign(pred) == np.sign(actual)) * 100)

    metrics = {
        "split":               split,
        "n":                   len(df),
        "MAE":                 round(mae,   6),
        "RMSE":                round(rmse,  6),
        "MAPE_pct":            round(mape,  4),
        "DirectionalAcc_pct":  round(dir_acc, 2),
    }

    log.info(
        "Metrics [%s]  MAE=%.6f  RMSE=%.6f  "
        "MAPE=%.2f%%  DirAcc=%.2f%%",
        split, mae, rmse, mape, dir_acc,
    )
    return metrics


# =============================================================================
# Run for all tickers  (called from pipeline.py)
# =============================================================================

def run(scaled_dict, scaler_dict, cfg):
    """
    Generates and saves predictions for every ticker.

    Parameters
    ----------
    scaled_dict : { ticker -> (train_scaled, val_scaled, test_scaled) }
    scaler_dict : { ticker -> (feature_scaler, target_scaler) }
    cfg         : full config dict

    Returns
    -------
    pred_dict    : { ticker -> prediction DataFrame }
    metrics_dict : { ticker -> metrics dict (test split) }
    """
    out_dir = cfg["data"]["predictions_dir"]
    os.makedirs(out_dir, exist_ok=True)

    pred_dict    = {}
    metrics_dict = {}

    for ticker in scaled_dict:
        log.info("Generating predictions: %s", ticker)

        pred_df = build_prediction_df(
            ticker, scaled_dict, scaler_dict, cfg
        )

        # Save CSV
        csv_path = os.path.join(out_dir, f"{ticker}_predictions.csv")
        pred_df.to_csv(csv_path, index=False)
        log.info("Predictions saved -> %s", csv_path)

        metrics = compute_metrics(pred_df, split="test")

        pred_dict[ticker]    = pred_df
        metrics_dict[ticker] = metrics

    # ── Summary table ──────────────────────────────────────────────────
    summary = pd.DataFrame(metrics_dict).T
    summary_path = os.path.join(out_dir, "metrics_summary.csv")
    summary.to_csv(summary_path)
    log.info("Metrics summary saved -> %s", summary_path)
    print("\n=== Test Metrics Summary ===")
    print(summary.to_string())

    return pred_dict, metrics_dict


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.feature_engineering.indicators import add_all_indicators
    from src.feature_engineering.features   import (
        build_features, FEATURE_COLS, TARGET_COL,
    )
    from src.models.scaler  import scale_ticker
    from src.models.lstm    import build_model
    from src.models.dataset import make_loaders
    from src.models.trainer import train as train_model

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    raw = yf.download("OGDC.KA", start="2020-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
    raw["date"] = pd.to_datetime(raw["date"])

    df_feat = build_features(add_all_indicators(raw), sentiment_df=None)
    train_s, val_s, test_s, feat_sc, tgt_sc = scale_ticker(
        df_feat, FEATURE_COLS, TARGET_COL
    )

    cfg = {
        "model": {
            "architecture":    "lstm",
            "input_size":      len(FEATURE_COLS),
            "checkpoints_dir": "/tmp/checkpoints",
        },
        "data": {
            "predictions_dir": "/tmp/predictions",
            "processed_dir":   "/tmp/processed",
        },
        "training": {
            "max_epochs":  3,
            "patience":    3,
            "lr":          1e-3,
            "weight_decay":1e-4,
            "lr_factor":   0.5,
            "lr_patience": 2,
            "grad_clip":   1.0,
            "batch_size":  32,
            "seq_len":     30,
            "num_workers": 0,
        },
    }

    scaled_dict = {"OGDC.KA": (train_s, val_s, test_s)}
    scaler_dict = {"OGDC.KA": (feat_sc, tgt_sc)}

    train_loader, val_loader, _ = make_loaders(
        scaled_dict, FEATURE_COLS, TARGET_COL,
        seq_len=30, batch_size=32, num_workers=0, pin_memory=False,
    )

    model = build_model(cfg)
    train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        ckpt_path    = "/tmp/checkpoints/OGDC.KA_best.pt",
        feature_cols = FEATURE_COLS,
        target_col   = TARGET_COL,
    )

    pred_df = build_prediction_df(
        "OGDC.KA", scaled_dict, scaler_dict, cfg, model=model
    )

    print("\n--- Prediction sample (test split, last 5 rows) ---")
    test_rows = pred_df[pred_df["split"] == "test"].tail(5)
    print(test_rows[["date", "actual_pct", "pred_pct"]].to_string(index=False))

    metrics = compute_metrics(pred_df, split="test")
    print("\n--- Test metrics ---")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    print("\n✓ predict.py smoke-test passed.")
