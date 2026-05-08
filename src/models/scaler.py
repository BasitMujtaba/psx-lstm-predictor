"""
src/models/scaler.py
======================
Per-ticker MinMaxScaler that is fit ONLY on training data.

Why per-ticker?
  Even though all features are scale-free (ratios, % changes,
  bounded oscillators), each ticker still has its own distribution
  of values. OGDC turbulence spikes look different from LUCK turbulence
  spikes. A per-ticker scaler tightens each feature into [0, 1]
  relative to that ticker's own history — no cross-ticker leakage.

Why fit on train only?
  Fitting on the full dataset would let future information (test period
  min/max) influence the scaling of past data. We fit strictly on the
  training window and apply the same transform to val and test.

What gets scaled?
  Every column in FEATURE_COLS from features.py.
  The TARGET_COL (ret_1d_future) is scaled separately so we can
  inverse-transform predictions back to real return values.

Saved artefacts  (one file per ticker)
  data/processed/<TICKER>_scaler.pkl   — dict with
      {
        "feature_scaler": fitted MinMaxScaler on features,
        "target_scaler":  fitted MinMaxScaler on target,
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
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)


# =============================================================================
# Train / Val / Test split
# =============================================================================

def split_by_date(df, train_ratio=0.7, val_ratio=0.15):
    """
    Splits a single-ticker feature DataFrame into train / val / test
    by date order — no shuffling, no leakage.

    Ratios:
      train : train_ratio          (default 70%)
      val   : val_ratio            (default 15%)
      test  : 1 - train - val      (default 15%)

    Parameters
    ----------
    df          : output of features.build_features() for one ticker
    train_ratio : fraction of rows for training
    val_ratio   : fraction of rows for validation

    Returns
    -------
    train_df, val_df, test_df  — three non-overlapping DataFrames
    """
    n       = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:i_train].copy()
    val_df   = df.iloc[i_train:i_val].copy()
    test_df  = df.iloc[i_val:].copy()

    log.info(
        "Split -> train=%d  val=%d  test=%d  "
        "(train_end=%s  test_start=%s)",
        len(train_df), len(val_df), len(test_df),
        train_df["date"].iloc[-1].date() if "date" in train_df.columns else "?",
        test_df["date"].iloc[0].date()   if "date" in test_df.columns  else "?",
    )
    return train_df, val_df, test_df


# =============================================================================
# Fit scaler on train, transform all splits
# =============================================================================

def fit_scalers(train_df, feature_cols, target_col):
    """
    Fits two MinMaxScalers on the training data only.

    Returns
    -------
    feature_scaler : fitted on feature_cols
    target_scaler  : fitted on target_col  (needed for inverse transform)
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler  = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_df[feature_cols].values)
    target_scaler.fit(train_df[[target_col]].values)

    log.info(
        "Scalers fit on %d training rows  "
        "(%d feature cols + 1 target col)",
        len(train_df), len(feature_cols)
    )
    return feature_scaler, target_scaler


def transform_split(df, feature_scaler, target_scaler,
                    feature_cols, target_col):
    """
    Applies already-fitted scalers to a DataFrame split.
    Returns a new DataFrame with scaled values.
    Original "date" and "close" columns are preserved unscaled
    so they can be used for plotting / inverse transform later.
    """
    df = df.copy()

    scaled_features             = feature_scaler.transform(
                                      df[feature_cols].values)
    df[feature_cols]            = scaled_features

    scaled_target               = target_scaler.transform(
                                      df[[target_col]].values)
    df[target_col]              = scaled_target.flatten()

    return df


def scale_ticker(df, feature_cols, target_col,
                 train_ratio=0.7, val_ratio=0.15):
    """
    Full pipeline for one ticker:
      1. Split into train / val / test
      2. Fit scalers on train
      3. Transform all three splits

    Returns
    -------
    train_scaled, val_scaled, test_scaled : DataFrames with scaled values
    feature_scaler, target_scaler         : fitted scaler objects
    """
    train_df, val_df, test_df = split_by_date(
        df, train_ratio=train_ratio, val_ratio=val_ratio
    )

    feature_scaler, target_scaler = fit_scalers(
        train_df, feature_cols, target_col
    )

    train_scaled = transform_split(
        train_df, feature_scaler, target_scaler, feature_cols, target_col
    )
    val_scaled   = transform_split(
        val_df,   feature_scaler, target_scaler, feature_cols, target_col
    )
    test_scaled  = transform_split(
        test_df,  feature_scaler, target_scaler, feature_cols, target_col
    )

    return train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler


# =============================================================================
# Save and load scaler artefacts
# =============================================================================

def save_scalers(ticker, feature_scaler, target_scaler,
                 feature_cols, target_col, train_end_date, out_dir):
    """
    Saves both scalers and metadata to a single pickle file.
    One file per ticker:  <out_dir>/<TICKER>_scaler.pkl
    """
    os.makedirs(out_dir, exist_ok=True)
    artefact = {
        "feature_scaler": feature_scaler,
        "target_scaler":  target_scaler,
        "feature_cols":   feature_cols,
        "target_col":     target_col,
        "train_end_date": train_end_date,
    }
    path = os.path.join(out_dir, f"{ticker}_scaler.pkl")
    with open(path, "wb") as f:
        pickle.dump(artefact, f)
    log.info("Scaler saved -> %s", path)
    return path


def load_scalers(ticker, out_dir):
    """
    Loads scaler artefact for a ticker.

    Returns
    -------
    dict with keys:
        feature_scaler, target_scaler,
        feature_cols, target_col, train_end_date
    """
    path = os.path.join(out_dir, f"{ticker}_scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scaler found for {ticker} at {path}")
    with open(path, "rb") as f:
        artefact = pickle.load(f)
    log.info("Scaler loaded <- %s", path)
    return artefact


# =============================================================================
# Inverse transform  (predictions -> real return values)
# =============================================================================

def inverse_transform_target(scaled_values, target_scaler):
    """
    Converts scaled predictions back to real log-return values.

    Parameters
    ----------
    scaled_values : np.array of shape (N,) or (N, 1)
    target_scaler : fitted MinMaxScaler for the target column

    Returns
    -------
    np.array of shape (N,)  in original log-return scale
    """
    scaled_values = np.array(scaled_values).reshape(-1, 1)
    return target_scaler.inverse_transform(scaled_values).flatten()


# =============================================================================
# Run for all tickers  (called from pipeline.py)
# =============================================================================

def run(features_dict, feature_cols, target_col,
        out_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Scales features for every ticker and saves scaler artefacts.

    Parameters
    ----------
    features_dict : dict  { ticker -> DataFrame from features.build_features() }
    feature_cols  : list of feature column names  (FEATURE_COLS from features.py)
    target_col    : target column name            (TARGET_COL  from features.py)
    out_dir       : directory to save scaler pkl files
    train_ratio   : fraction of data for training
    val_ratio     : fraction of data for validation

    Returns
    -------
    scaled_dict : dict  { ticker -> (train_scaled, val_scaled, test_scaled) }
    scaler_dict : dict  { ticker -> (feature_scaler, target_scaler) }
    """
    scaled_dict = {}
    scaler_dict = {}

    for ticker, df in features_dict.items():
        log.info("Scaling ticker: %s  (%d rows)", ticker, len(df))

        train_s, val_s, test_s, feat_sc, tgt_sc = scale_ticker(
            df,
            feature_cols  = feature_cols,
            target_col    = target_col,
            train_ratio   = train_ratio,
            val_ratio     = val_ratio,
        )

        train_end = (
            train_s["date"].iloc[-1]
            if "date" in train_s.columns else None
        )

        save_scalers(
            ticker         = ticker,
            feature_scaler = feat_sc,
            target_scaler  = tgt_sc,
            feature_cols   = feature_cols,
            target_col     = target_col,
            train_end_date = train_end,
            out_dir        = out_dir,
        )

        scaled_dict[ticker] = (train_s, val_s, test_s)
        scaler_dict[ticker] = (feat_sc, tgt_sc)

    log.info("Scaling complete for %d tickers.", len(features_dict))
    return scaled_dict, scaler_dict


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    from feature_engineering.indicators import add_all_indicators
    from feature_engineering.features   import build_features, FEATURE_COLS, TARGET_COL

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    raw = yf.download("OGDC.KA", start="2020-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw[["open","high","low","close","volume"]].dropna()
    raw = raw.reset_index().rename(columns={"Date":"date","index":"date"})
    raw["date"] = pd.to_datetime(raw["date"])

    df_feat = build_features(add_all_indicators(raw), sentiment_df=None)

    train_s, val_s, test_s, feat_sc, tgt_sc = scale_ticker(
        df_feat, FEATURE_COLS, TARGET_COL
    )

    print(f"Train : {len(train_s)} rows")
    print(f"Val   : {len(val_s)}   rows")
    print(f"Test  : {len(test_s)}  rows")

    print("\nScaled feature ranges (should all be within [0, 1]):")
    for col in FEATURE_COLS[:6]:
        s = train_s[col]
        print(f"  {col:30s}  min={s.min():.4f}  max={s.max():.4f}")

    print("\nInverse-transform test:")
    dummy = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    real  = inverse_transform_target(dummy, tgt_sc)
    print(f"  scaled -> {dummy}")
    print(f"  real   -> {np.round(real, 6)}")
