"""
src/feature_engineering/features.py
======================================
Builds the final scale-free feature matrix for the LSTM.

Final feature set  (47 columns total)
--------------------------------------
  From indicators  : macd_pct, macd_sig_pct, macd_hist_pct,
                     rsi_norm, cci_norm, williams_r_norm,
                     stoch_k_norm, stoch_d_norm, roc_5_norm, roc_10_norm,
                     di_plus_norm, di_minus_norm, adx_norm, di_diff_norm,
                     atr_pct, bb_pct_b, bb_width, obv_pct, turbulence  (19)
  From this file   : rolling_vol_10, rolling_vol_20, parkinson_vol,
                     volume_ma5_ratio, volume_ma20_ratio,
                     close_to_high, close_to_low, high_low_range_pct,
                     gap_pct, ret_1d, ret_5d, ret_10d, ret_20d, ret_60d,
                     vol_ratio_5_20, market_return, relative_strength,
                     sector_return                                       (18)
  From sentiment   : 9 active category scores + sentiment_ma5           (10)
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
    "williams_r_norm", "stoch_k_norm", "stoch_d_norm",
    "roc_5_norm", "roc_10_norm",
    "di_plus_norm", "di_minus_norm", "adx_norm", "di_diff_norm",
    "atr_pct", "bb_pct_b", "bb_width",
    "obv_pct", "turbulence",
]

PRICE_RATIO_COLS = [
    "rolling_vol_10", "rolling_vol_20", "parkinson_vol",
    "volume_ma5_ratio", "volume_ma20_ratio",
    "close_to_high", "close_to_low", "high_low_range_pct",
    "gap_pct", "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_ratio_5_20", "market_return", "relative_strength", "sector_return",
]

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
    "sentiment_ma5",
]

# Full ordered feature list fed into the LSTM (47 total)
FEATURE_COLS = INDICATOR_COLS + PRICE_RATIO_COLS + SENTIMENT_COLS

# What the model predicts
TARGET_COL = "ret_1d_future"


# ── Indicator extension ───────────────────────────────────────────────────────

def add_extended_indicators(df):
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]

    if "williams_r_norm" not in df.columns:
        high14 = h.rolling(14).max()
        low14  = l.rolling(14).min()
        wr     = (high14 - c) / (high14 - low14).replace(0, np.nan)
        df["williams_r_norm"] = (1 - wr).clip(0, 1) * 2 - 1

    if "stoch_k_norm" not in df.columns:
        low14  = l.rolling(14).min()
        high14 = h.rolling(14).max()
        k      = (c - low14) / (high14 - low14).replace(0, np.nan)
        df["stoch_k_norm"] = k.clip(0, 1)
        df["stoch_d_norm"] = df["stoch_k_norm"].rolling(3).mean()

    if "roc_5_norm" not in df.columns:
        df["roc_5_norm"]  = (c / c.shift(5)  - 1).clip(-0.5, 0.5)
        df["roc_10_norm"] = (c / c.shift(10) - 1).clip(-0.5, 0.5)

    return df


# ── Price-ratio features ──────────────────────────────────────────────────────

def add_price_ratios(df):
    df = df.copy()
    c  = df["close"]
    o  = df["open"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]

    day_range = (h - l).replace(0, np.nan)
    df["close_to_high"]      = (h - c) / day_range
    df["close_to_low"]       = (c - l) / day_range
    df["high_low_range_pct"] = day_range / c

    df["gap_pct"] = (o / df["ldcp"] - 1).clip(-0.5, 0.5)
    df["ret_1d"]  = np.log(c / df["ldcp"])
    df["ret_5d"]  = np.log(c / c.shift(5))
    df["ret_10d"] = np.log(c / c.shift(10))
    df["ret_20d"] = np.log(c / c.shift(20))
    df["ret_60d"] = np.log(c / c.shift(60))

    df["rolling_vol_10"] = df["ret_1d"].rolling(10).std()
    df["rolling_vol_20"] = df["ret_1d"].rolling(20).std()

    df["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) *
        (np.log(h / l.replace(0, np.nan)) ** 2).rolling(20).mean()
    )

    vol_ma5  = v.rolling(5).mean().replace(0, np.nan)
    vol_ma20 = v.rolling(20).mean().replace(0, np.nan)
    df["volume_ma5_ratio"]  = (v / vol_ma5).clip(0, 10)
    df["volume_ma20_ratio"] = (v / vol_ma20).clip(0, 10)

    vol_5  = df["ret_1d"].rolling(5).std()
    vol_20 = df["ret_1d"].rolling(20).std()
    df["vol_ratio_5_20"] = (vol_5 / vol_20.replace(0, np.nan)).clip(0, 5)

    return df


# ── Cross-sectional features ──────────────────────────────────────────────────

def add_cross_sectional(df_dict):
    # Build returns aligned on DATE index
    all_returns = {}
    for t, df in df_dict.items():
        tmp = df.set_index("date")["ret_1d"] if "date" in df.columns else df["ret_1d"]
        all_returns[t] = tmp

    all_returns = pd.DataFrame(all_returns)
    market_ret  = all_returns.mean(axis=1)  # indexed by date

    for ticker, df in df_dict.items():
        df    = df.copy()
        dates = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
        mr    = market_ret.reindex(dates).values
        df["market_return"]     = mr
        df["relative_strength"] = df["ret_1d"].values - mr
        df["sector_return"]     = mr
        df_dict[ticker] = df

    return df_dict


# ── Target column ─────────────────────────────────────────────────────────────

def add_target(df):
    df = df.copy()
    df[TARGET_COL] = np.log(df["close"].shift(-1) / df["close"])
    return df


# ── Sentiment merge ───────────────────────────────────────────────────────────

def merge_sentiment(df, sentiment_df):
    df = df.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])

    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    drop_cols = ["article_count", "positive_count", "negative_count",
                 "neutral_count", "sentiment_label",
                 "psx_official_corporate_score", "sector_specific_score"]
    sentiment_df = sentiment_df.drop(
        columns=[c for c in drop_cols if c in sentiment_df.columns]
    )

    base_sentiment = [c for c in SENTIMENT_COLS
                      if c != "sentiment_ma5" and c in sentiment_df.columns]
    missing = [c for c in SENTIMENT_COLS
               if c != "sentiment_ma5" and c not in sentiment_df.columns]

    if missing:
        log.warning("Sentiment columns missing (will be 0): %s", missing)

    df = df.merge(sentiment_df[["date"] + base_sentiment], on="date", how="left")

    for col in missing:
        df[col] = 0.0
    for col in base_sentiment:
        df[col] = df[col].ffill().fillna(0.0)

    df["sentiment_ma5"] = df["sentiment_score"].rolling(5).mean().fillna(0.0)

    log.info("Sentiment merged: %d rows, %d sentiment cols",
             len(df), len(base_sentiment) + 1)
    return df


# ── Master function (single ticker) ──────────────────────────────────────────

def build_features(df, sentiment_df=None, cfg=None):
    log.info("Adding extended indicators ...")
    df = add_extended_indicators(df)

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

    for col in ["market_return", "relative_strength", "sector_return"]:
        if col not in df.columns:
            log.warning("%s not found — filling with 0. "
                        "Call add_cross_sectional() before build_features().", col)
            df[col] = 0.0

    if "date" not in df.columns and df.index.dtype == "datetime64[ns]":
        df = df.reset_index().rename(columns={"index": "date"})

    required_cols = FEATURE_COLS + [TARGET_COL]
    before = len(df)
    df     = df.dropna(subset=required_cols)
    log.info("NaN rows dropped: %d -> %d (removed %d)", before, len(df), before - len(df))

    id_cols   = [c for c in ["date", "ticker", "close"] if c in df.columns]
    keep_cols = id_cols + FEATURE_COLS + [TARGET_COL]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df        = df[keep_cols].reset_index(drop=True)

    log.info("Final feature matrix: %d rows x %d feature cols", len(df), len(FEATURE_COLS))
    return df


# ── Master function (all tickers) ────────────────────────────────────────────

def build_all_features(df_dict, sentiment_df=None, cfg=None):
    log.info("Adding price ratios to all tickers ...")
    for ticker in df_dict:
        df_dict[ticker] = add_extended_indicators(df_dict[ticker])
        df_dict[ticker] = add_price_ratios(df_dict[ticker])

    log.info("Computing cross-sectional features ...")
    df_dict = add_cross_sectional(df_dict)

    result = {}
    for ticker, df in df_dict.items():
        log.info("Finalising features for %s ...", ticker)
        df = add_target(df)
        if sentiment_df is not None:
            df = merge_sentiment(df, sentiment_df)
        else:
            for col in SENTIMENT_COLS:
                df[col] = 0.0

        required_cols = FEATURE_COLS + [TARGET_COL]
        before = len(df)
        df     = df.dropna(subset=required_cols)
        log.info("%s: %d -> %d rows after NaN drop", ticker, before, len(df))

        id_cols   = [c for c in ["date", "ticker", "close"] if c in df.columns]
        keep_cols = id_cols + FEATURE_COLS + [TARGET_COL]
        keep_cols = [c for c in keep_cols if c in df.columns]
        result[ticker] = df[keep_cols].reset_index(drop=True)

    log.info("All tickers: feature build complete. Total features: %d", len(FEATURE_COLS))
    return result
