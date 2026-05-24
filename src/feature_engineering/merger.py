"""
src/feature_engineering/merger.py
=======================================
Shifts news sentiment by 1 trading day (no look-ahead bias) and
joins it onto the price data for both decay and flags variants.

Outputs
-------
  data/processed/prices_news_joined_decay.csv
  data/processed/prices_news_joined_flags.csv
"""

import os
import subprocess
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE         = os.path.join(PROJECT_ROOT, "data")


# ── GitHub push ───────────────────────────────────────────────────────────────

def _push_to_github(decay_path, flags_path):
    try:
        cmds = [
            ["git", "-C", PROJECT_ROOT, "pull", "--rebase", "origin", "main"],
            ["git", "-C", PROJECT_ROOT, "add", decay_path, flags_path],
            ["git", "-C", PROJECT_ROOT, "commit", "-m",
             "Update prices_news_joined_decay and prices_news_joined_flags"],
            ["git", "-C", PROJECT_ROOT, "push"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        log.info("Pushed joined CSVs to GitHub")
    except subprocess.CalledProcessError as e:
        log.warning("GitHub push failed: %s", e.stderr.decode())


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # ── Load files
    prices = pd.read_csv(os.path.join(BASE, "processed", "psx_prices_processed.csv"),         parse_dates=["date"])
    decay  = pd.read_csv(os.path.join(BASE, "processed", "news_aggregated_decay_catwise.csv"), parse_dates=["date"])
    flags  = pd.read_csv(os.path.join(BASE, "processed", "news_aggregated_flags.csv"),         parse_dates=["date"])

    log.info("Prices : %s | %s -> %s", prices.shape, prices["date"].min().date(), prices["date"].max().date())
    log.info("Decay  : %s | %s -> %s", decay.shape,  decay["date"].min().date(),  decay["date"].max().date())
    log.info("Flags  : %s | %s -> %s", flags.shape,  flags["date"].min().date(),  flags["date"].max().date())

    # ── Build trading calendar from price data
    trading_dates = (prices[["date"]]
                     .drop_duplicates()
                     .sort_values("date")
                     .reset_index(drop=True))
    trading_dates["prev_trading_date"] = trading_dates["date"].shift(1)
    log.info("Unique trading days: %d", len(trading_dates))

    # ── Shift news by 1 trading day
    decay_shifted = (trading_dates
                     .merge(decay, left_on="prev_trading_date", right_on="date", how="left")
                     .rename(columns={"date_x": "date"})
                     .drop(columns=["date_y", "prev_trading_date"]))

    flags_shifted = (trading_dates
                     .merge(flags, left_on="prev_trading_date", right_on="date", how="left")
                     .rename(columns={"date_x": "date"})
                     .drop(columns=["date_y", "prev_trading_date"]))

    # ── Fill NaNs
    decay_sentiment_cols = ["sentiment_corporate", "sentiment_energy", "sentiment_forex", "sentiment_macro", "news_count"]
    flags_sentiment_cols = ["sentiment_corporate", "has_corporate", "sentiment_energy", "has_energy",
                            "sentiment_forex", "has_forex", "sentiment_macro", "has_macro", "news_count"]

    decay_shifted[decay_sentiment_cols] = decay_shifted[decay_sentiment_cols].fillna(0.0)
    flags_shifted[flags_sentiment_cols] = flags_shifted[flags_sentiment_cols].fillna(0.0)

    log.info("Decay shifted : %s | NaNs: %d", decay_shifted.shape, decay_shifted.isna().sum().sum())
    log.info("Flags shifted : %s | NaNs: %d", flags_shifted.shape, flags_shifted.isna().sum().sum())

    # ── Join onto prices
    prices_decay = prices.merge(decay_shifted, on="date", how="left")
    prices_flags = prices.merge(flags_shifted, on="date", how="left")

    log.info("Prices + Decay : %s (expected %d rows x %d cols)",
             prices_decay.shape, prices.shape[0], prices.shape[1] + len(decay_sentiment_cols))
    log.info("Prices + Flags : %s (expected %d rows x %d cols)",
             prices_flags.shape, prices.shape[0], prices.shape[1] + len(flags_sentiment_cols))

    # ── Verify broadcast — all tickers on same date share same sentiment
    log.info("Broadcast check (DECAY) — all tickers on 2012-03-16:")
    sample = prices_decay[prices_decay["date"] == "2012-03-16"][
        ["date", "ticker", "close", "sentiment_energy", "sentiment_macro", "news_count"]
    ]
    log.info("  Unique sentiment_energy values : %d (should be 1)", sample["sentiment_energy"].nunique())
    log.info("  Unique sentiment_macro  values : %d (should be 1)", sample["sentiment_macro"].nunique())

    # ── Verify Fri -> Mon
    log.info("Fri -> Mon check (DECAY, first 3 Mondays):")
    mondays = prices_decay[prices_decay["date"].dt.weekday == 0].drop_duplicates("date").head(3)
    for _, row in mondays.iterrows():
        monday   = row["date"]
        friday   = trading_dates.loc[trading_dates["date"] == monday, "prev_trading_date"].values[0]
        fri_news = decay[decay["date"] == friday][["sentiment_energy", "sentiment_macro"]]
        if not fri_news.empty:
            match_e = round(fri_news["sentiment_energy"].values[0], 4) == round(row["sentiment_energy"], 4)
            match_m = round(fri_news["sentiment_macro"].values[0],  4) == round(row["sentiment_macro"],  4)
            log.info("  Friday %s -> Monday %s | energy %s | macro %s",
                     pd.Timestamp(friday).date(), pd.Timestamp(monday).date(),
                     "OK" if match_e else "MISMATCH", "OK" if match_m else "MISMATCH")
        else:
            log.info("  Friday %s -> Monday %s | no news found for Friday",
                     pd.Timestamp(friday).date(), pd.Timestamp(monday).date())

    # ── NaN report after join
    nan_d = prices_decay[decay_sentiment_cols].isna().sum()
    log.info("NaN report (DECAY): %s", "clean" if not nan_d.any() else nan_d.to_string())
    nan_f = prices_flags[flags_sentiment_cols].isna().sum()
    log.info("NaN report (FLAGS): %s", "clean" if not nan_f.any() else nan_f.to_string())

    # ── Sort by ticker -> date (for LSTM sequence training)
    prices_decay = prices_decay.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_flags = prices_flags.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info("Final columns (DECAY): %s", prices_decay.columns.tolist())
    log.info("Final columns (FLAGS): %s", prices_flags.columns.tolist())

    # ── Save
    processed_dir = os.path.join(BASE, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    decay_path = os.path.join(processed_dir, "prices_news_joined_decay.csv")
    flags_path = os.path.join(processed_dir, "prices_news_joined_flags.csv")

    prices_decay.to_csv(decay_path, index=False)
    prices_flags.to_csv(flags_path, index=False)

    log.info("Saved -> %s  (%s, %.1f MB)", decay_path, prices_decay.shape, os.path.getsize(decay_path) / 1e6)
    log.info("Saved -> %s  (%s, %.1f MB)", flags_path, prices_flags.shape, os.path.getsize(flags_path) / 1e6)

    # ── Push to GitHub
    _push_to_github(decay_path, flags_path)

    return prices_decay, prices_flags


if __name__ == "__main__":
    run()
