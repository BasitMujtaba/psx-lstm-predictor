"""
src/evaluation/predict.py
===========================
Loads a saved checkpoint and generates directional predictions
for any PSX ticker (UP=1 / DOWN=0).

What this module does
---------------------
  1. Loads the best checkpoint from checkpoints/best_model.pt
  2. Loads the matching scaler artefact from data/scalers/<TICKER>_scaler.pkl
  3. Runs inference on train / val / test splits
  4. Returns probability of UP move and binary direction prediction
  5. Computes accuracy, AUC, F1 per split
  6. Saves predictions to data/predictions/<TICKER>_predictions.csv

Output CSV columns
------------------
  date           -- trading date
  close          -- actual closing price
  actual         -- actual direction  (0=DOWN  1=UP)
  prob_up        -- model probability of UP move  (0.0 - 1.0)
  pred           -- predicted direction (0 or 1, threshold=0.5)
  correct        -- 1 if pred == actual else 0
  split          -- train / val / test

config.yaml sections used
-------------------------
  data:
    predictions_dir: data/predictions
  model:
    checkpoints_dir: src/models/checkpoints
  training:
    seq_len:     30
    batch_size:  256
    num_workers: 2
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

log = logging.getLogger(__name__)


# -- Inference -----------------------------------------------------------------

@torch.no_grad()
def predict_split(model, loader, device):
    """
    Runs inference on one DataLoader split.

    Returns
    -------
    probs : np.array (N,)  -- probability of UP move
    """
    model.eval()
    all_probs = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(all_probs)


# -- Metrics -------------------------------------------------------------------

def compute_metrics(probs, actuals, threshold=0.5):
    """
    Parameters
    ----------
    probs   : np.array (N,)  probabilities
    actuals : np.array (N,)  true binary labels

    Returns
    -------
    dict with keys: acc, auc, f1, n
    """
    preds = (probs >= threshold).astype(int)
    return {
        "n":   len(actuals),
        "acc": round(float(accuracy_score(actuals, preds)),  4),
        "auc": round(float(roc_auc_score(actuals, probs)),   4),
        "f1":  round(float(f1_score(actuals, preds, zero_division=0)), 4),
    }


# -- Build prediction DataFrame for one ticker ---------------------------------

def build_prediction_df(ticker, scaled_dict, cfg,
                        model=None, threshold=0.5):
    """
    Generates a prediction DataFrame for all three splits of one ticker.

    Parameters
    ----------
    ticker      : ticker symbol string  e.g. OGDC
    scaled_dict : dict  { ticker -> (train_scaled, val_scaled, test_scaled) }
    cfg         : full config dict
    model       : pre-loaded PSXLSTMModel (optional)
                  if None loads best_model.pt from checkpoints_dir
    threshold   : float  decision threshold (default 0.5)

    Returns
    -------
    pd.DataFrame with columns described in module docstring
    """
    from src.models.lstm    import build_model
    from src.models.dataset import make_single_loader
    from src.models.trainer import load_best_checkpoint
    from src.feature_engineering.features import FEATURE_COLS, TARGET_COL

    t_cfg       = cfg.get("training", {})
    seq_len     = t_cfg.get("seq_len",     30)
    batch_size  = t_cfg.get("batch_size", 256)
    num_workers = t_cfg.get("num_workers",   2)

    ckpt_dir  = cfg["model"]["checkpoints_dir"]
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = build_model(cfg)
        load_best_checkpoint(best_ckpt, model, device=str(device))
    model.to(device)
    model.eval()

    train_s, val_s, test_s = scaled_dict[ticker]
    frames = []

    for split_name, df_split in [
        ("train", train_s),
        ("val",   val_s),
        ("test",  test_s),
    ]:
        loader = make_single_loader(
            df_split,
            feature_cols = FEATURE_COLS,
            target_col   = TARGET_COL,
            seq_len      = seq_len,
            batch_size   = batch_size,
            shuffle      = False,
            num_workers  = num_workers,
        )

        n_windows = len(df_split) - seq_len
        df_pred   = df_split.iloc[seq_len : seq_len + n_windows].copy()

        probs   = predict_split(model, loader, device)
        preds   = (probs >= threshold).astype(int)
        actuals = df_pred[TARGET_COL].values.astype(int)

        df_out = pd.DataFrame({
            "date":    df_pred["date"].values,
            "close":   df_pred["close"].values,
            "actual":  actuals,
            "prob_up": probs.round(4),
            "pred":    preds,
            "correct": (preds == actuals).astype(int),
            "split":   split_name,
        })
        frames.append(df_out)

        m = compute_metrics(probs, actuals, threshold)
        log.info(
            "%s [%s]  n=%d  acc=%.4f  auc=%.4f  f1=%.4f",
            ticker, split_name, m["n"], m["acc"], m["auc"], m["f1"],
        )

    return pd.concat(frames, ignore_index=True)


# -- Run for all tickers -------------------------------------------------------

def run(scaled_dict, cfg, model=None, threshold=0.5):
    """
    Generates and saves predictions for every ticker.

    Parameters
    ----------
    scaled_dict : dict  { ticker -> (train_scaled, val_scaled, test_scaled) }
    cfg         : full config dict
    model       : trained PSXLSTMModel (optional, loads from checkpoint if None)
    threshold   : decision threshold (default 0.5)

    Returns
    -------
    pred_dict    : dict  { ticker -> prediction DataFrame }
    metrics_dict : dict  { ticker -> { train, val, test metrics } }
    """
    from src.feature_engineering.features import TARGET_COL

    out_dir = cfg["data"]["predictions_dir"]
    os.makedirs(out_dir, exist_ok=True)

    pred_dict    = {}
    metrics_dict = {}

    for ticker in scaled_dict:
        log.info("Generating predictions: %s", ticker)

        pred_df = build_prediction_df(
            ticker, scaled_dict, cfg,
            model=model, threshold=threshold,
        )

        csv_path = os.path.join(out_dir, ticker + "_predictions.csv")
        pred_df.to_csv(csv_path, index=False)
        log.info("Saved -> %s", csv_path)

        ticker_metrics = {}
        for split in ["train", "val", "test"]:
            df_s = pred_df[pred_df["split"] == split]
            if df_s.empty:
                continue
            ticker_metrics[split] = compute_metrics(
                df_s["prob_up"].values,
                df_s["actual"].values,
                threshold,
            )
        metrics_dict[ticker] = ticker_metrics
        pred_dict[ticker]    = pred_df

    # -- Summary table
    rows = []
    for ticker, splits in metrics_dict.items():
        for split, m in splits.items():
            rows.append({"ticker": ticker, "split": split, **m})

    summary      = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    log.info("Metrics summary saved -> %s", summary_path)

    print("")
    print("=== Prediction Metrics Summary ===")
    print(summary.to_string(index=False))

    return pred_dict, metrics_dict


if __name__ == "__main__":
    run()
