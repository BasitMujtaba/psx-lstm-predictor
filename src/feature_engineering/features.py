"""
src/feature_engineering/features.py
======================================
Builds the final scale-free feature matrix for the LSTM.

Inputs
------
  - Per-ticker DataFrame processed by indicators.add_all_indicators()
  - Daily sentiment DataFrame from news_scraper.run()

Price-ratio features added here
--------------------------------
  close_to_high        how close to day high          [0, 1]
  close_to_low         how close to day low           [0, 1]
  high_low_range_pct   daily range / close            ratio
  gap_pct              open vs ldcp %                 [-0.5, +0.5]
  ret_1d               1-day log return  (close / ldcp)
  ret_5d               5-day log return
  ret_10d              10-day log return
  ret_20d              20-day log return
  vol_ratio_5_20       5-day vol / 20-day vol         regime signal

Sentiment features (9 active categories from news_scraper)
-----------------------------------------------------------
  sentiment_score
  macro_pakistan_score
  psx_market_score
  energy_oil_score
  banking_finance_score
  geopolitical_global_score
  political_stability_score
  pakistani_media_score
  international_media_pakistan_score

  NOTE: psx_official_corporate_score and sector_specific_score are present
  in the CSV but always 0 — excluded from SENTIMENT_COLS until scraper
  populates them.

Final feature set  (32 columns total)
--------------------------------------
  From indicators  : macd_pct, macd_sig_pct, macd_hist_pct,
                     rsi_norm, cci_norm,
                     di_plus_norm, di_minus_norm, adx_norm, di_diff_norm,
                     atr_pct, bb_pct_b, bb_width, obv_pct, turbulence  (14)
  From this file   : close_to_high, close_to_low, high_low_range_pct,
                     gap_pct, ret_1d, ret_5d, ret_10d, ret_20d,
                     vol_ratio_5_20  (9)
  From sentiment   : 9 active category scores                           (9)
  Target           : ret_1d_future  (next-day log return)
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Column lists ──────────────────────────────────────────────────────────────

INDICATOR_COLS = [
    "macd_pct", "macd_sig_pct", "macd_hist_pct",
    "rsi_norm", "cci_norm",
    "di_plus_norm", "di_minus_norm", "adx_norm", "di_diff_norm",
    "atr_pct", "bb_pct_b", "bb_width",
    "obv_pct", "turbulence",
]

PRICE_RATIO_COLS = [
    "close_to_high", "close_to_low", "high_low_range_pct",
    "gap_pct", "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "vol_ratio_5_20",
]

# psx_official_corporate_score and sector_specific_score excluded —
# always 0 in current scraper output (verified in testing)
SENTIMENT_COLS = [
    "sentiment_score",
    "macro_pakistan_score",
    "psx_market_score",
    "energy_oil_score",
    "banking_finance_score",
    "geopolitical_global_score",
    "political_stability_score",
    "pakistani_media_score",
    "international_media_pakistan_score",
]

# Cols present in sentiment CSV but excluded until scraper populates them
SENTIMENT_COLS_UNUSED = [
    "psx_official_corporate_score",
    "sector_specific_score",
]

# Full ordered feature list fed into the LSTM (32 total)
FEATURE_COLS = INDICATOR_COLS + PRICE_RATIO_COLS + SENTIMENT_COLS

# What the model predicts
TARGET_COL = "ret_1d_future"


# ── Price-ratio features ──────────────────────────────────────────────────────

def add_price_ratios(df):
    """
    All features derived purely from OHLCV — no absolute prices.
    Uses ldcp (last day closing price) from raw CSV for ret_1d and gap_pct
    instead of close.shift(1) — more robust across missing trading days.
    """
    df = df.copy()
    c  = df["close"]
    o  = df["open"]
    h  = df["high"]
    l  = df["low"]

    day_range = (h - l).replace(0, np.nan)
    df["close_to_high"]      = (h - c) / day_range
    df["close_to_low"]       = (c - l) / day_range
    df["high_low_range_pct"] = day_range / c

    # ldcp = last day closing price, already in raw CSV — use directly
    df["gap_pct"] = (o / df["ldcp"] - 1).clip(-0.5, 0.5)
    df["ret_1d"]  = np.log(c / df["ldcp"])

    df["ret_5d"]  = np.log(c / c.shift(5))
    df["ret_10d"] = np.log(c / c.shift(10))
    df["ret_20d"] = np.log(c / c.shift(20))

    vol_5  = df["ret_1d"].rolling(5).std()
    vol_20 = df["ret_1d"].rolling(20).std()
    df["vol_ratio_5_20"] = (vol_5 / vol_20.replace(0, np.nan)).clip(0, 5)

    return df


# ── Target column ─────────────────────────────────────────────────────────────

def add_target(df):
    """
    ret_1d_future = next trading day log return.
    Row T holds the return that occurs on day T+1.
    Last row is NaN and dropped later.
    """
    df = df.copy()
    df[TARGET_COL] = np.log(df["close"].shift(-1) / df["close"])
    return df


# ── Sentiment merge ───────────────────────────────────────────────────────────

def merge_sentiment(df, sentiment_df):
    """
    Left-joins OHLCV+indicator DataFrame with daily sentiment on date.
    Missing days forward-filled then filled with 0 (neutral).
    Drops non-score cols (article_count, sentiment_label, etc.) before merge.
    """
    df = df.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])

    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    # drop non-score metadata cols
    drop_cols = ["article_count", "positive_count", "negative_count",
                 "neutral_count", "sentiment_label"]
    sentiment_df = sentiment_df.drop(
        columns=[c for c in drop_cols if c in sentiment_df.columns]
    )

    available = [c for c in SENTIMENT_COLS if c in sentiment_df.columns]
    missing   = [c for c in SENTIMENT_COLS if c not in sentiment_df.columns]
    if missing:
        log.warning("Sentiment columns missing (will be 0): %s", missing)

    df = df.merge(sentiment_df[["date"] + available], on="date", how="left")

    for col in missing:
        df[col] = 0.0
    for col in SENTIMENT_COLS:
        df[col] = df[col].ffill().fillna(0.0)

    log.info("Sentiment merged: %d rows, %d sentiment cols",
             len(df), len(available))
    return df


# ── Master function ───────────────────────────────────────────────────────────

def build_features(df, sentiment_df=None, cfg=None):
    """
    Full pipeline: price ratios -> target -> sentiment -> final select.

    Parameters
    ----------
    df           : Output of indicators.add_all_indicators()
                   Must contain: date, ldcp, open, high, low, close, volume
                   + all INDICATOR_COLS
    sentiment_df : Output of news_scraper.run()
                   If None, all sentiment columns filled with 0.
    cfg          : Full config.yaml dict (reserved for future use)

    Returns
    -------
    DataFrame with columns:
        ["date", "ticker", "close"] + FEATURE_COLS + [TARGET_COL]
    Rows with any NaN in features or target are dropped.
    ticker and close kept for inverse-transform in predict.py.
    """
    log.info("Building price-ratio features ...")
    df = add_price_ratios(df)

    log.info("Adding target column ...")
    df = add_target(df)

    if sentiment_df is not None:
        df = merge_sentiment(df, sentiment_df)
    else:
        log.warning("No sentiment_df provided — all sentiment cols set to 0.")
        for col in SENTIMENT_COLS:
            df[col] = 0.0

    if "date" not in df.columns and df.index.dtype == "datetime64[ns]":
        df = df.reset_index().rename(columns={"index": "date"})

    required_cols = FEATURE_COLS + [TARGET_COL]
    before = len(df)
    df     = df.dropna(subset=required_cols)
    log.info("NaN rows dropped: %d -> %d (removed %d)",
             before, len(df), before - len(df))

    id_cols   = [c for c in ["date", "ticker", "close"] if c in df.columns]
    keep_cols = id_cols + FEATURE_COLS + [TARGET_COL]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df        = df[keep_cols].reset_index(drop=True)

    log.info("Final feature matrix: %d rows x %d feature cols",
             len(df), len(FEATURE_COLS))
    return df
