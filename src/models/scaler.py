"""
src/models/scaler.py
======================
Per-ticker StandardScaler that is fit ONLY on training data.

Why per-ticker?
  Even though all features are scale-free (ratios, % changes,
  bounded oscillators), each ticker still has its own distribution
  of values. A per-ticker scaler normalizes each feature relative
  to that ticker's own training history — no cross-ticker leakage.

Why StandardScaler over MinMaxScaler?
  Features like news_count (0-25), turbulence (0-20), and rsi (0-100)
  have skewed distributions with outliers. MinMaxScaler compresses most
  values into a small range when outliers are present. StandardScaler
  (zero mean, unit variance) spreads values more evenly, giving the
  LSTM cleaner gradient signals.

Why fit on train only?
  Fitting on the full dataset leaks future information (test period
  statistics) into the scaling of past data. We fit strictly on the
  training window and apply the same transform to val and test.

Why no target scaler?
  Target is binary (0 = DOWN, 1 = UP). Scaling 0/1 is meaningless
  and breaks BCELoss. Target is kept as raw integers.

Saved artefacts  (one file per ticker)
  data/scalers/<TICKER>_scaler.pkl   — dict with
      {
        "feature_scaler": fitted StandardScaler on features,
        "feature_cols":   list of feature column names,
        "target_col":     target column name,
        "train_end_date": last date in the training window,
      }
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ── Train / Val / Test split ──────────────────────────────────────────────────

def split_by_date(df, train_ratio=0.7, val_ratio=0.15):
    """
    Splits a single-ticker feature DataFrame into train / val / test
    by date order — no shuffling, no leakage.

    Ratios:
      train : train_ratio          (default 70%)
      val   : val_ratio            (default 15%)
      test  : 1 - train - val      (default 15%)

    Returns
    -------
    train_df, val_df, test_df
    """
    n       = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:i_train].copy()
    val_df   = df.iloc[i_train:i_val].copy()
    test_df  = df.iloc[i_val:].copy()

    log.info(
        "Split -> train=%d  val=%d  test=%d  (train_end=%s  test_start=%s)",
        len(train_df), len(val_df), len(test_df),
        train_df["date"].iloc[-1].date() if "date" in train_df.columns else "?",
        test_df["date"].iloc[0].date()   if "date" in test_df.columns  else "?",
    )
    return train_df, val_df, test_df


# ── Fit scaler on train ───────────────────────────────────────────────────────

def fit_scaler(train_df, feature_cols):
    """
    Fits a StandardScaler on training data only.

    Returns
    -------
    feature_scaler : fitted StandardScaler
    """
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_cols].values)
    log.info(
        "StandardScaler fit on %d training rows (%d feature cols)",
        len(train_df), len(feature_cols),
    )
    return feature_scaler


# ── Transform a split ─────────────────────────────────────────────────────────

def transform_split(df, feature_scaler, feature_cols, target_col):
    """
    Applies already-fitted scaler to a DataFrame split.
    Target column is kept as raw 0/1 — not scaled.
    date and close columns are preserved unscaled.

    Returns
    -------
    New DataFrame with scaled features, raw target.
    """
    df                = df.copy()
    df[feature_cols]  = feature_scaler.transform(df[feature_cols].values)
    return df


# ── Full pipeline for one ticker ──────────────────────────────────────────────

def scale_ticker(df, feature_cols, target_col,
                 train_ratio=0.7, val_ratio=0.15):
    """
    Full pipeline for one ticker:
      1. Split into train / val / test by date
      2. Fit StandardScaler on train only
      3. Transform all three splits

    Returns
    -------
    train_scaled, val_scaled, test_scaled : DataFrames with scaled features
    feature_scaler                         : fitted StandardScaler
    """
    train_df, val_df, test_df = split_by_date(
        df, train_ratio=train_ratio, val_ratio=val_ratio,
    )

    feature_scaler = fit_scaler(train_df, feature_cols)

    train_scaled = transform_split(train_df, feature_scaler, feature_cols, target_col)
    val_scaled   = transform_split(val_df,   feature_scaler, feature_cols, target_col)
    test_scaled  = transform_split(test_df,  feature_scaler, feature_cols, target_col)

    return train_scaled, val_scaled, test_scaled, feature_scaler


# ── Save and load scaler artefacts ────────────────────────────────────────────

def save_scaler(ticker, feature_scaler, feature_cols,
                target_col, train_end_date, out_dir):
    """
    Saves scaler and metadata to a single pickle file.
    One file per ticker: <out_dir>/<TICKER>_scaler.pkl
    """
    os.makedirs(out_dir, exist_ok=True)
    artefact = {
        "feature_scaler": feature_scaler,
        "feature_cols":   feature_cols,
        "target_col":     target_col,
        "train_end_date": train_end_date,
    }
    path = os.path.join(out_dir, f"{ticker}_scaler.pkl")
    with open(path, "wb") as f:
        pickle.dump(artefact, f)
    log.info("Scaler saved -> %s", path)
    return path


def load_scaler(ticker, out_dir):
    """
    Loads scaler artefact for a ticker.

    Returns
    -------
    dict with keys: feature_scaler, feature_cols, target_col, train_end_date
    """
    path = os.path.join(out_dir, f"{ticker}_scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scaler found for {ticker} at {path}")
    with open(path, "rb") as f:
        artefact = pickle.load(f)
    log.info("Scaler loaded <- %s", path)
    return artefact


# ── Run for all tickers ───────────────────────────────────────────────────────

def run(features_dict, feature_cols, target_col,
        out_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Scales features for every ticker and saves scaler artefacts.

    Parameters
    ----------
    features_dict : dict  { ticker -> DataFrame from features.run() }
    feature_cols  : FEATURE_COLS from features.py
    target_col    : TARGET_COL   from features.py
    out_dir       : directory to save scaler pkl files
    train_ratio   : fraction of data for training   (default 0.70)
    val_ratio     : fraction of data for validation (default 0.15)

    Returns
    -------
    scaled_dict : dict  { ticker -> (train_scaled, val_scaled, test_scaled) }
    scaler_dict : dict  { ticker -> feature_scaler }
    """
    scaled_dict = {}
    scaler_dict = {}

    for ticker, df in features_dict.items():
        log.info("Scaling ticker: %s  (%d rows)", ticker, len(df))

        train_s, val_s, test_s, feat_sc = scale_ticker(
            df,
            feature_cols = feature_cols,
            target_col   = target_col,
            train_ratio  = train_ratio,
            val_ratio    = val_ratio,
        )

        train_end = (
            train_s["date"].iloc[-1]
            if "date" in train_s.columns else None
        )

        save_scaler(
            ticker         = ticker,
            feature_scaler = feat_sc,
            feature_cols   = feature_cols,
            target_col     = target_col,
            train_end_date = train_end,
            out_dir        = out_dir,
        )

        scaled_dict[ticker] = (train_s, val_s, test_s)
        scaler_dict[ticker] = feat_sc

    log.info("Scaling complete for %d tickers.", len(scaled_dict))
    return scaled_dict, scaler_dict


if __name__ == "__main__":
    run()
