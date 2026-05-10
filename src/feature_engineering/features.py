"""
src/feature_engineering/features.py
======================================
Builds the final scale-free feature matrix for the LSTM.

Final feature set  (50 columns total)
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

# Full ordered feature list fed into the LSTM (50 total)
FEATURE_COLS = INDICATOR_COLS + PRICE_RATIO_COLS + SENTIMENT_COLS

# What the model predicts
TARGET_COL = "ret_1d_future"


# ── Indicator extension (new indicators not in indicators.py yet) ─────────────

def add_extended_indicators(df):
    """
    Adds indicators that may not be in indicators.py yet:
      williams_r_norm, stoch_k_norm, stoch_d_norm, roc_5_norm, roc_10_norm
    Skips any that already exist.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Williams %R  normalised to [-1, 1]
    if "williams_r_norm" not in df.columns:
        high14 = h.rolling(14).max()
        low14  = l.rolling(14).min()
        wr     = (high14 - c) / (high14 - low14).replace(0, np.nan)
        df["williams_r_norm"] = (1 - wr).clip(0, 1) * 2 - 1   # [-1,1]

    # Stochastic %K and %D  normalised to [0, 1]
    if "stoch_k_norm" not in df.columns:
        low14  = l.rolling(14).min()
        high14 = h.rolling(14).max()
        k      = (c - low14) / (high14 - low14).replace(0, np.nan)
        df["stoch_k_norm"] = k.clip(0, 1)
        df["stoch_d_norm"] = df["stoch_k_norm"].rolling(3).mean()

    # Rate of Change  clipped to [-0.5, 0.5]
    if "roc_5_norm" not in df.columns:
        df["roc_5_norm"]  = (c / c.shift(5)  - 1).clip(-0.5, 0.5)
        df["roc_10_norm"] = (c / c.shift(10) - 1).clip(-0.5, 0.5)

    return df


# ── Price-ratio features ──────────────────────────────────────────────────────

def add_price_ratios(df):
    """
    All features derived purely from OHLCV — no absolute prices.
    """
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

    # Volatility
    df["rolling_vol_10"] = df["ret_1d"].rolling(10).std()
    df["rolling_vol_20"] = df["ret_1d"].rolling(20).std()

    # Parkinson volatility estimator (high-low based)
    df["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) *
        (np.log(h / l.replace(0, np.nan)) ** 2).rolling(20).mean()
    )

    # Volume ratios
    vol_ma5  = v.rolling(5).mean().replace(0, np.nan)
    vol_ma20 = v.rolling(20).mean().replace(0, np.nan)
    df["volume_ma5_ratio"]  = (v / vol_ma5).clip(0, 10)
    df["volume_ma20_ratio"] = (v / vol_ma20).clip(0, 10)

    # Vol regime signal
    vol_5  = df["ret_1d"].rolling(5).std()
    vol_20 = df["ret_1d"].rolling(20).std()
    df["vol_ratio_5_20"] = (vol_5 / vol_20.replace(0, np.nan)).clip(0, 5)

    return df


# ── Cross-sectional features ──────────────────────────────────────────────────

def add_cross_sectional(df_dict):
    """
    Adds market_return, relative_strength, sector_return to each ticker df.

    Parameters
    ----------
    df_dict : dict { ticker -> DataFrame }  after add_price_ratios()

    Returns
    -------
    dict { ticker -> DataFrame }  with 3 new columns added
    """
    # Market return = equal-weight mean of all tickers' ret_1d
    all_returns = pd.DataFrame({
        t: df["ret_1d"] for t, df in df_dict.items()
    })
    market_ret = all_returns.mean(axis=1)   # Series indexed by date

    for ticker, df in df_dict.items():
        df = df.copy()
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        else:
            dates = df.index

        mr = market_ret.reindex(dates).values
        df["market_return"]    = mr
        df["relative_strength"] = df["ret_1d"].values - mr

        # Sector return: for now same as market_return
        # (extend with sector mapping dict when available)
        df["sector_return"] = mr

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
                 "neutral_count", "sentiment_label"]
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

    # sentiment_ma5 — 5-day rolling mean of sentiment_score
    df["sentiment_ma5"] = df["sentiment_score"].rolling(5).mean().fillna(0.0)

    log.info("Sentiment merged: %d rows, %d sentiment cols",
             len(df), len(base_sentiment) + 1)
    return df


# ── Master function (single ticker) ──────────────────────────────────────────

def build_features(df, sentiment_df=None, cfg=None):
    """
    Full pipeline for a single ticker:
      extended indicators -> price ratios -> target -> sentiment -> select

    Note: call add_cross_sectional(df_dict) BEFORE this for market_return,
    relative_strength, sector_return to be populated.

    Parameters
    ----------
    df           : Output of indicators.add_all_indicators()
    sentiment_df : Output of news_scraper.run()
    cfg          : Full config.yaml dict (reserved)

    Returns
    -------
    DataFrame with columns:
        ["date", "ticker", "close"] + FEATURE_COLS + [TARGET_COL]
    """
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

    # Ensure cross-sectional cols exist (filled with 0 if not yet computed)
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
    log.info("NaN rows dropped: %d -> %d (removed %d)",
             before, len(df), before - len(df))

    id_cols   = [c for c in ["date", "ticker", "close"] if c in df.columns]
    keep_cols = id_cols + FEATURE_COLS + [TARGET_COL]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df        = df[keep_cols].reset_index(drop=True)

    log.info("Final feature matrix: %d rows x %d feature cols",
             len(df), len(FEATURE_COLS))
    return df


# ── Master function (all tickers) ────────────────────────────────────────────

def build_all_features(df_dict, sentiment_df=None, cfg=None):
    """
    Runs the full pipeline for all tickers including cross-sectional features.

    Parameters
    ----------
    df_dict      : dict { ticker -> DataFrame from indicators.py }
    sentiment_df : combined sentiment DataFrame
    cfg          : config dict

    Returns
    -------
    dict { ticker -> featured DataFrame }
    """
    # Step 1 — add price ratios to all tickers first (needed for market_return)
    log.info("Adding price ratios to all tickers ...")
    for ticker in df_dict:
        df_dict[ticker] = add_extended_indicators(df_dict[ticker])
        df_dict[ticker] = add_price_ratios(df_dict[ticker])

    # Step 2 — cross-sectional features (needs all tickers simultaneously)
    log.info("Computing cross-sectional features ...")
    df_dict = add_cross_sectional(df_dict)

    # Step 3 — target + sentiment + final select per ticker
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
