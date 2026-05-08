"""
src/feature_engineering/features.py
======================================
Builds the final scale-free feature matrix for the LSTM.

Inputs
------
  - OHLCV DataFrame already processed by indicators.py
  - Daily sentiment DataFrame from news_scraper.py

What this module adds on top of indicators.py
---------------------------------------------
  Price-ratio features  (no absolute prices, just relationships):
    close_to_high       - how close to the day high          [0, 1]
    close_to_low        - how close to the day low           [0, 1]
    high_low_range_pct  - daily range / close                [0, +inf)
    gap_pct             - open vs prior close % change       [-1, +1]
    ret_1d              - 1-day log return
    ret_5d              - 5-day log return
    ret_10d             - 10-day log return
    ret_20d             - 20-day log return
    vol_ratio_5_20      - 5-day vol / 20-day vol             (regime)

  Sentiment features  (from news_scraper.py):
    sentiment_score          - overall FinBERT daily score   [-1, +1]
    macro_pakistan_score     - macro sentiment               [-1, +1]
    geopolitical_global_score- geo sentiment                 [-1, +1]
    psx_market_score         - market sentiment              [-1, +1]
    energy_oil_score         - sector sentiment              [-1, +1]
    banking_finance_score    - sector sentiment              [-1, +1]

Final feature set (26 columns total)
-------------------------------------
  From indicators  : macd_pct, macd_sig_pct, macd_hist_pct,
                     rsi_norm, cci_norm,
                     di_plus_norm, di_minus_norm, adx_norm, di_diff_norm,
                     atr_pct, bb_pct_b, bb_width, obv_pct, turbulence
  From this file   : close_to_high, close_to_low, high_low_range_pct,
                     gap_pct, ret_1d, ret_5d, ret_10d, ret_20d, vol_ratio_5_20
  From sentiment   : sentiment_score, macro_pakistan_score,
                     geopolitical_global_score, psx_market_score,
                     energy_oil_score, banking_finance_score

Target column
-------------
  ret_1d_future  - next trading day log return  (what LSTM predicts)
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Indicator columns produced by indicators.py
INDICATOR_COLS = [
    "macd_pct", "macd_sig_pct", "macd_hist_pct",
    "rsi_norm", "cci_norm",
    "di_plus_norm", "di_minus_norm", "adx_norm", "di_diff_norm",
    "atr_pct", "bb_pct_b", "bb_width",
    "obv_pct", "turbulence",
]

# Price-ratio columns added here
PRICE_RATIO_COLS = [
    "close_to_high", "close_to_low", "high_low_range_pct",
    "gap_pct", "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "vol_ratio_5_20",
]

# Sentiment columns merged from news_scraper output
SENTIMENT_COLS = [
    "sentiment_score",
    "macro_pakistan_score",
    "geopolitical_global_score",
    "psx_market_score",
    "energy_oil_score",
    "banking_finance_score",
]

# The complete ordered feature list fed into the LSTM
FEATURE_COLS = INDICATOR_COLS + PRICE_RATIO_COLS + SENTIMENT_COLS

# What the model is trained to predict
TARGET_COL = "ret_1d_future"


# =============================================================================
# Price-ratio features
# =============================================================================

def add_price_ratios(df):
    """
    All features derived purely from OHLCV relationships.
    No absolute prices — every value is a ratio or log return.
    """
    df = df.copy()
    c  = df["close"]
    o  = df["open"]
    h  = df["high"]
    l  = df["low"]

    # Where did close land within the day range?
    day_range = (h - l).replace(0, np.nan)
    df["close_to_high"]      = (h - c) / day_range        # 0=at high, 1=at low
    df["close_to_low"]       = (c - l) / day_range        # 0=at low,  1=at high
    df["high_low_range_pct"] = day_range / c               # daily volatility proxy

    # Gap: how did today open vs yesterday close?
    df["gap_pct"] = (o / c.shift(1) - 1).clip(-0.5, 0.5)

    # Log returns over multiple horizons
    df["ret_1d"]  = np.log(c / c.shift(1))
    df["ret_5d"]  = np.log(c / c.shift(5))
    df["ret_10d"] = np.log(c / c.shift(10))
    df["ret_20d"] = np.log(c / c.shift(20))

    # Volatility regime: short vol vs long vol
    vol_5  = df["ret_1d"].rolling(5).std()
    vol_20 = df["ret_1d"].rolling(20).std()
    df["vol_ratio_5_20"] = (vol_5 / vol_20.replace(0, np.nan)).clip(0, 5)

    return df


# =============================================================================
# Target column
# =============================================================================

def add_target(df):
    """
    ret_1d_future = next trading day log return.
    This is what the LSTM predicts.
    Shift(-1) so row T holds the return that will occur on day T+1.
    The last row will be NaN (no future yet) and is dropped later.
    """
    df = df.copy()
    df[TARGET_COL] = np.log(
        df["close"].shift(-1) / df["close"]
    )
    return df


# =============================================================================
# Sentiment merge
# =============================================================================

def merge_sentiment(df, sentiment_df):
    """
    Left-joins the OHLCV+indicator DataFrame with the daily sentiment
    DataFrame on the date column.

    Missing sentiment days are forward-filled then filled with 0
    (neutral) so every trading day has a value.

    Parameters
    ----------
    df           : DataFrame with a "date" column (or DatetimeIndex)
    sentiment_df : Output of news_scraper.aggregate_daily_sentiment()
    """
    df = df.copy()

    # Ensure both sides have a plain "date" column for merging
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])

    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    # Keep only the columns we actually use
    available_sent_cols = [
        c for c in SENTIMENT_COLS if c in sentiment_df.columns
    ]
    missing = [c for c in SENTIMENT_COLS if c not in sentiment_df.columns]
    if missing:
        log.warning(
            "Sentiment columns not found, will be filled with 0: %s", missing
        )

    merge_cols = ["date"] + available_sent_cols
    df = df.merge(sentiment_df[merge_cols], on="date", how="left")

    # Fill any columns that were missing entirely
    for col in missing:
        df[col] = 0.0

    # Forward-fill gaps then neutral-fill start
    for col in SENTIMENT_COLS:
        df[col] = df[col].ffill().fillna(0.0)

    log.info(
        "Sentiment merged: %d rows, %d sentiment cols",
        len(df), len(SENTIMENT_COLS)
    )
    return df


# =============================================================================
# Master function
# =============================================================================

def build_features(df, sentiment_df=None, cfg=None):
    """
    Full pipeline: price ratios -> target -> sentiment merge -> final select.

    Parameters
    ----------
    df           : Output of indicators.add_all_indicators()
    sentiment_df : Output of news_scraper.run()  (optional)
                   If None, sentiment columns are filled with 0.
    cfg          : Full config.yaml dict (reserved for future use)

    Returns
    -------
    DataFrame with columns  FEATURE_COLS + [TARGET_COL] + ["date", "close"]
    Rows with any NaN in features or target are dropped.
    "close" is kept (not as a feature) so predict.py can inverse-transform.
    """
    log.info("Building price-ratio features ...")
    df = add_price_ratios(df)

    log.info("Adding target column ...")
    df = add_target(df)

    # Merge sentiment
    if sentiment_df is not None:
        df = merge_sentiment(df, sentiment_df)
    else:
        log.warning(
            "No sentiment_df provided — filling all sentiment cols with 0."
        )
        for col in SENTIMENT_COLS:
            df[col] = 0.0

    # Ensure date column exists
    if "date" not in df.columns and df.index.dtype == "datetime64[ns]":
        df = df.reset_index().rename(columns={"index": "date"})

    # Drop rows with NaN in any feature or target
    required_cols = FEATURE_COLS + [TARGET_COL]
    before = len(df)
    df     = df.dropna(subset=required_cols)
    log.info(
        "NaN rows dropped: %d -> %d (removed %d)",
        before, len(df), before - len(df)
    )

    # Select and order final columns
    keep_cols = ["date", "close"] + FEATURE_COLS + [TARGET_COL]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df        = df[keep_cols].reset_index(drop=True)

    log.info(
        "Final feature matrix: %d rows x %d feature cols",
        len(df), len(FEATURE_COLS)
    )
    return df


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    from indicators import add_all_indicators

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s"
    )

    log.info("Downloading OGDC.KA for smoke test ...")
    raw = yf.download("OGDC.KA", start="2020-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
    raw["date"] = pd.to_datetime(raw["date"])

    df_ind = add_all_indicators(raw)
    df_out = build_features(df_ind, sentiment_df=None)

    print("\n--- Shape ---")
    print(df_out.shape)

    print("\n--- Last 3 rows ---")
    print(df_out[["date", "close"] + FEATURE_COLS[:6] + [TARGET_COL]].tail(3).to_string())

    print("\n--- Feature summary ---")
    for col in FEATURE_COLS:
        s = df_out[col]
        print(f"  {col:30s}  mean={s.mean():+.4f}"
              f"  std={s.std():.4f}"
              f"  nulls={s.isna().sum()}")
