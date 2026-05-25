"""
src/feature_engineering/merger.py
=======================================
Joins category-level price CSVs with their corresponding sentiment
column from news_sentiment_pivoted.csv (shifted 1 trading day).

Outputs
-------
  data/processed/prices_news_joined_banking.csv
  data/processed/prices_news_joined_corporate.csv
  data/processed/prices_news_joined_energy.csv
  data/processed/prices_news_joined_forex.csv
  data/processed/prices_news_joined_macro.csv
"""

import os
import subprocess
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE         = os.path.join(PROJECT_ROOT, "data")

CATEGORY_MAP = {
    "banking":   ("psx_prices_banking.csv",   "sentiment_banking"),
    "corporate": ("psx_prices_corporate.csv", "sentiment_corporate"),
    "energy":    ("psx_prices_energy.csv",    "sentiment_energy"),
    "forex":     ("psx_prices_forex.csv",     "sentiment_forex"),
    "macro":     ("psx_prices_macro.csv",     "sentiment_macro"),
}


# ── GitHub push ───────────────────────────────────────────────────────────────

def _push_to_github(paths):
    try:
        cmds = [
            ["git", "-C", PROJECT_ROOT, "pull", "--rebase", "origin", "main"],
            ["git", "-C", PROJECT_ROOT, "add"] + paths,
            ["git", "-C", PROJECT_ROOT, "commit", "-m",
             "Add category-joined price+sentiment CSVs"],
            ["git", "-C", PROJECT_ROOT, "push"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        log.info("Pushed joined CSVs to GitHub")
    except subprocess.CalledProcessError as e:
        log.warning("GitHub push failed: %s", e.stderr.decode())


# ── Collapse pivot: multiple rows per date -> one row per date ────────────────

def _load_pivot(pivot_path):
    """
    Pivot has one row per (date, category) with only the matching
    sentiment column filled. Groupby date + sum collapses to one row
    per date with all 5 sentiment columns correctly populated.
    """
    log.info("Loading pivot: %s", pivot_path)
    pivot = pd.read_csv(pivot_path, parse_dates=["date"])

    sentiment_cols = [c for c in pivot.columns if c.startswith("sentiment_")]
    pivot_collapsed = (pivot
                       .groupby("date")[sentiment_cols]
                       .sum()
                       .reset_index())

    log.info("Pivot collapsed -> shape: %s  |  columns: %s",
             pivot_collapsed.shape, pivot_collapsed.columns.tolist())
    log.info("Pivot date range: %s -> %s",
             pivot_collapsed["date"].min().date(),
             pivot_collapsed["date"].max().date())
    return pivot_collapsed


# ── Build shifted sentiment for a given set of trading dates ─────────────────

def _shift_sentiment(pivot_collapsed, trading_dates):
    """
    Shift sentiment by 1 trading day to avoid look-ahead bias.
    trading_dates: DataFrame with columns [date, prev_trading_date].
    """
    shifted = (trading_dates
               .merge(pivot_collapsed,
                      left_on  = "prev_trading_date",
                      right_on = "date",
                      how      = "left")
               .rename(columns={"date_x": "date"})
               .drop(columns=["date_y", "prev_trading_date"]))

    sentiment_cols = [c for c in shifted.columns if c.startswith("sentiment_")]
    shifted[sentiment_cols] = shifted[sentiment_cols].fillna(0.0)
    return shifted


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    processed_dir  = os.path.join(BASE, "processed")
    pivot_path     = os.path.join(processed_dir, "news", "news_sentiment_pivoted.csv")

    pivot_collapsed = _load_pivot(pivot_path)

    output_paths = []

    for category, (price_file, sentiment_col) in CATEGORY_MAP.items():
        price_path = os.path.join(processed_dir, price_file)

        if not os.path.exists(price_path):
            log.warning("Price file not found, skipping: %s", price_path)
            continue

        log.info("\n--- Processing category: %s ---", category.upper())
        prices = pd.read_csv(price_path, parse_dates=["date"])
        log.info("Loaded %s -> shape: %s  tickers: %s",
                 price_file, prices.shape,
                 sorted(prices["ticker"].unique().tolist()))

        # Build trading dates from this category's price data
        trading_dates = (prices[["date"]]
                         .drop_duplicates()
                         .sort_values("date")
                         .reset_index(drop=True))
        trading_dates["prev_trading_date"] = trading_dates["date"].shift(1)
        log.info("Unique trading days: %d", len(trading_dates))

        # Shift sentiment by 1 trading day
        shifted = _shift_sentiment(pivot_collapsed, trading_dates)

        # Keep only the relevant sentiment column for this category
        if sentiment_col not in shifted.columns:
            log.warning("Column %s not found in pivot, skipping %s", sentiment_col, category)
            continue

        shifted_slim = shifted[["date", sentiment_col]].copy()

        # Left join: all price rows preserved, sentiment attached by date
        joined = prices.merge(shifted_slim, on="date", how="left")
        joined[sentiment_col] = joined[sentiment_col].fillna(0.0)
        joined = joined.sort_values(["ticker", "date"]).reset_index(drop=True)

        log.info("%s: final shape: %s  |  %s mean=%.4f  zeros=%.1f%%",
                 category.upper(),
                 joined.shape,
                 sentiment_col,
                 joined[sentiment_col].mean(),
                 (joined[sentiment_col] == 0).mean() * 100)

        out_path = os.path.join(processed_dir, f"prices_news_joined_{category}.csv")
        joined.to_csv(out_path, index=False)
        output_paths.append(out_path)
        log.info("Saved -> %s  (%.2f MB)", out_path, os.path.getsize(out_path) / 1e6)

    _push_to_github(output_paths)
    log.info("\nAll done. %d files written.", len(output_paths))


if __name__ == "__main__":
    run()
