"""
src/feature_engineering/features.py
=====================================
Builds the final feature matrix for the LSTM model.

Feature set (31 features)
--------------------------
  Momentum        : ret_1d, ret_5d, ret_10d, ret_20d
  Indicators      : rsi, macd, bb_pct, bb_width, turbulence
  Trend ratios    : close_to_ema9, close_to_ema21, close_to_ema50
  Price structure : close_to_high, close_to_low, gap_pct, high_low_range_pct
  Volume          : volume_ma5_ratio, volume_ma20_ratio, vol_ratio_5_20,
                    rolling_vol_10, rolling_vol_20, volume_trend
  Range           : 52w_high_ratio, 52w_low_ratio
  Circuit         : near_upper_circuit, near_lower_circuit
  Calendar        : day_of_week, month
  Sentiment       : sentiment_macro, sentiment_energy, news_count

Target
------
  Binary 1 (UP) / 0 (DOWN) based on next-day return with ±0.5% noise filter.
  Rows where |next_ret| <= 0.5% are dropped as unactionable noise.

Inputs
------
  data/processed/prices_news_joined_decay.csv
  data/processed/prices_news_joined_flags.csv

Outputs
-------
  data/processed/features_decay.csv
  data/processed/features_flags.csv
"""

import os
import subprocess
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE         = os.path.join(PROJECT_ROOT, "data")

THRESHOLD = 0.005  # ±0.5% noise filter

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "rsi", "macd", "bb_pct", "bb_width", "turbulence",
    "close_to_ema9", "close_to_ema21", "close_to_ema50",
    "close_to_high", "close_to_low", "gap_pct", "high_low_range_pct",
    "volume_ma5_ratio", "volume_ma20_ratio", "vol_ratio_5_20",
    "rolling_vol_10", "rolling_vol_20", "volume_trend",
    "52w_high_ratio", "52w_low_ratio",
    "near_upper_circuit", "near_lower_circuit",
    "day_of_week", "month",
    "sentiment_macro", "sentiment_energy", "news_count",
]

TARGET_COL = "target"


# ── GitHub push ───────────────────────────────────────────────────────────────

def _push_to_github(decay_path, flags_path):
    try:
        cmds = [
            ["git", "-C", PROJECT_ROOT, "pull", "--rebase", "origin", "main"],
            ["git", "-C", PROJECT_ROOT, "add", decay_path, flags_path],
            ["git", "-C", PROJECT_ROOT, "commit", "-m",
             "Update features_decay and features_flags"],
            ["git", "-C", PROJECT_ROOT, "push"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        log.info("Pushed feature CSVs to GitHub")
    except subprocess.CalledProcessError as e:
        log.warning("GitHub push failed: %s", e.stderr.decode())


# ── Per-ticker feature builder ────────────────────────────────────────────────

def _build_ticker_features(df):
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Momentum
    df["ret_1d"]  = np.log(df["close"] / df["ldcp"]).clip(-0.15, 0.15)
    df["ret_5d"]  = np.log(df["close"] / df["close"].shift(5)).clip(-0.40, 0.40)
    df["ret_10d"] = np.log(df["close"] / df["close"].shift(10)).clip(-0.50, 0.50)
    df["ret_20d"] = np.log(df["close"] / df["close"].shift(20)).clip(-0.60, 0.60)

    # ── Volatility
    df["rolling_vol_10"] = df["ret_1d"].rolling(10).std()
    df["rolling_vol_20"] = df["ret_1d"].rolling(20).std()
    vol_5                = df["ret_1d"].rolling(5).std()
    vol_20               = df["ret_1d"].rolling(20).std()
    df["vol_ratio_5_20"] = (vol_5 / vol_20.replace(0, np.nan)).clip(0, 5)

    # ── Indicators (normalize macd, clip turbulence)
    df["macd"]       = (df["macd"] / df["close"].replace(0, np.nan)).clip(-0.5, 0.5)
    df["bb_pct"]     = df["bb_pct"].clip(-0.5, 1.5)
    df["bb_width"]   = df["bb_width"].clip(0, 1.0)
    df["turbulence"] = df["turbulence"].clip(0, 20)

    # ── Trend ratios (scale-free)
    df["close_to_ema9"]  = ((df["close"] / df["ema_9"].replace(0,  np.nan)) - 1).clip(-0.5, 0.5)
    df["close_to_ema21"] = ((df["close"] / df["ema_21"].replace(0, np.nan)) - 1).clip(-0.5, 0.5)
    df["close_to_ema50"] = ((df["close"] / df["ema_50"].replace(0, np.nan)) - 1).clip(-0.5, 0.5)

    # ── Price structure
    day_range                = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_to_high"]      = ((df["high"] - df["close"]) / day_range).clip(0, 1)
    df["close_to_low"]       = ((df["close"] - df["low"])  / day_range).clip(0, 1)
    df["high_low_range_pct"] = (day_range / df["close"]).clip(0, 0.2)
    df["gap_pct"]            = (df["open"] / df["ldcp"] - 1).clip(-0.5, 0.5)

    # ── Volume
    vol_ma5              = df["volume"].rolling(5).mean().replace(0,  np.nan)
    vol_ma20             = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_ma5_ratio"]  = (df["volume"] / vol_ma5).clip(0, 10)
    df["volume_ma20_ratio"] = (df["volume"] / vol_ma20).clip(0, 10)
    df["volume_trend"]      = np.log(vol_ma5 / vol_ma20.replace(0, np.nan)).clip(-2, 2)

    # ── 52-week range ratios (using 126-day window)
    df["52w_high_ratio"] = ((df["close"] / df["close"].rolling(126).max()) - 1).clip(-1, 0)
    df["52w_low_ratio"]  = ((df["close"] / df["close"].rolling(126).min()) - 1).clip(0, 3)

    # ── Circuit breaker proximity
    df["near_upper_circuit"] = (df["close"] / df["ldcp"] > 1.045).astype(int)
    df["near_lower_circuit"] = (df["close"] / df["ldcp"] < 0.955).astype(int)

    # ── Calendar
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"]       = pd.to_datetime(df["date"]).dt.month

    # ── Target: next-day direction with ±0.5% noise filter
    df["next_ret"] = np.log(df["close"].shift(-1) / df["close"])
    df = df[(df["next_ret"] > THRESHOLD) | (df["next_ret"] < -THRESHOLD)].copy()
    df[TARGET_COL] = (df["next_ret"] > THRESHOLD).astype(int)

    return df


# ── Master builder ────────────────────────────────────────────────────────────

def _build(input_path, output_path, label):
    log.info("Loading %s ...", input_path)
    raw = pd.read_csv(input_path, parse_dates=["date"])
    log.info("Loaded : %s | %d tickers", raw.shape, raw["ticker"].nunique())

    frames = []
    for ticker, grp in raw.groupby("ticker"):
        try:
            frames.append(_build_ticker_features(grp))
        except Exception as e:
            log.warning("Skipping %s: %s", ticker, e)

    df = pd.concat(frames, ignore_index=True)

    # ── Drop NaNs on feature + target cols
    before = len(df)
    df     = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    log.info("%s: %d -> %d rows after dropna (dropped %d)", label, before, len(df), before - len(df))

    # ── Keep only useful columns
    id_cols   = ["date", "ticker", "close"]
    keep_cols = id_cols + FEATURE_COLS + [TARGET_COL]
    df        = df[keep_cols].sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Summary
    log.info("%s: %d rows | %d tickers | %d features", label, len(df), df["ticker"].nunique(), len(FEATURE_COLS))
    log.info("%s: Target UP %.1f%% | DOWN %.1f%%", label, df[TARGET_COL].mean()*100, (1-df[TARGET_COL].mean())*100)

    # ── Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("%s: Saved -> %s (%.1f MB)", label, output_path, os.path.getsize(output_path) / 1e6)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    processed_dir = os.path.join(BASE, "processed")

    decay_path = os.path.join(processed_dir, "features_decay.csv")
    flags_path = os.path.join(processed_dir, "features_flags.csv")

    df_decay = _build(
        input_path  = os.path.join(processed_dir, "prices_news_joined_decay.csv"),
        output_path = decay_path,
        label       = "DECAY",
    )
    df_flags = _build(
        input_path  = os.path.join(processed_dir, "prices_news_joined_flags.csv"),
        output_path = flags_path,
        label       = "FLAGS",
    )

    _push_to_github(decay_path, flags_path)

    return df_decay, df_flags


if __name__ == "__main__":
    run()
