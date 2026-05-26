"""
merger.py
---------
Merges per-category PSX price CSVs with lagged sentiment scores.

For each category (macro, energy, banking, forex, corporate):
  1. Loads psx_prices_{cat}.csv
  2. Loads news_sentiment_pivoted.csv, filters rows for that category,
     extracts only the sentiment_{cat} column.
  3. Lags the sentiment by 1 trading day (shift forward by 1 row after
     sorting by date) so today's row carries yesterday's signal —
     prevents lookahead bias / overfitting.
  4. Left-joins on date so all price rows are preserved; missing
     sentiment days become NaN then forward-filled.
  5. Saves psx_merged_{cat}.csv to data/processed/.
"""

import os
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
NEWS_DIR      = os.path.join(PROCESSED_DIR, "news")

SENTIMENT_PATH = os.path.join(NEWS_DIR, "news_sentiment_pivoted.csv")
CATEGORIES     = ["macro", "energy", "banking", "forex", "corporate"]


def load_sentiment(category: str) -> pd.DataFrame:
    """
    Returns a two-column DataFrame: date | sentiment_{category}
    with a 1-day lag applied (value shifted down by 1 row).
    """
    df = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])

    # Keep only rows belonging to this category
    cat_df = df[df["category"] == category][["date", f"sentiment_{category}"]].copy()
    cat_df = cat_df.sort_values("date").reset_index(drop=True)

    # --- THE KEY STEP ---
    # Shift sentiment forward by 1 row so each date receives the
    # previous trading day's sentiment signal.
    cat_df[f"sentiment_{category}"] = cat_df[f"sentiment_{category}"].shift(1)

    cat_df = cat_df.dropna(subset=[f"sentiment_{category}"])
    return cat_df


def merge_category(category: str) -> pd.DataFrame:
    price_path = os.path.join(PROCESSED_DIR, f"psx_prices_{category}.csv")

    if not os.path.exists(price_path):
        print(f"[merger] ⚠  Price file not found, skipping: {price_path}")
        return pd.DataFrame()

    print(f"[merger] Processing category: {category}")

    price_df = pd.read_csv(price_path, parse_dates=["date"])
    print(f"[merger]   Price rows   : {len(price_df):,}")
    print(f"[merger]   Price tickers: {price_df['ticker'].nunique() if 'ticker' in price_df.columns else 'N/A'}")

    sent_df = load_sentiment(category)
    print(f"[merger]   Sentiment rows (after lag): {len(sent_df):,}")

    # Left join — every price row survives; sentiment fills where available
    merged = price_df.merge(sent_df, on="date", how="left")

    # Forward-fill any remaining NaN sentiment gaps (e.g. weekends, holidays)
    merged[f"sentiment_{category}"] = (
        merged[f"sentiment_{category}"]
        .sort_index()
        .ffill()
        .fillna(0.0)          # If no history at all (very start), default to neutral
    )

    merged = merged.sort_values(["date", "ticker"] if "ticker" in merged.columns else ["date"])
    merged = merged.reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DIR, f"psx_merged_{category}.csv")
    merged.to_csv(out_path, index=False)
    print(f"[merger]   ✅ Saved {len(merged):,} rows → {out_path}")
    return merged


def run():
    if not os.path.exists(SENTIMENT_PATH):
        raise FileNotFoundError(f"[merger] Sentiment pivoted CSV not found: {SENTIMENT_PATH}")

    print(f"[merger] Sentiment source : {SENTIMENT_PATH}")
    print(f"[merger] Output directory : {PROCESSED_DIR}")
    print()

    summaries = []
    for cat in CATEGORIES:
        merged = merge_category(cat)
        if not merged.empty:
            summaries.append({
                "category"        : cat,
                "rows"            : len(merged),
                "sentiment_col"   : f"sentiment_{cat}",
                "sentiment_nonzero": (merged[f"sentiment_{cat}"] != 0).sum(),
            })

    print()
    print("=" * 60)
    print("MERGER SUMMARY")
    print("=" * 60)
    for s in summaries:
        pct = 100 * s["sentiment_nonzero"] / s["rows"] if s["rows"] else 0
        print(f"  {s['category']:<12} {s['rows']:>7,} rows  |  "
              f"non-zero sentiment: {s['sentiment_nonzero']:>6,} ({pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run()
