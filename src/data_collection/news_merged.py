"""
================================================================================
 File   : src/data_collection/news_merged.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source
          Categories are standardized to 4 values:
            macro | corporate | energy | forex
 Output : data/processed/news_merged.csv
================================================================================
"""

import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
PROCESSED  = BASE_DIR / "data" / "processed"

NEWS_FILES = {
    "dawn"      : PROCESSED / "dawn_news_processed.csv",
    "brecorder" : PROCESSED / "brecorder_news_processed.csv",
    "mettis"    : PROCESSED / "mettis_news_processed.csv",
}

OUTPUT_PATH = PROCESSED / "news_merged.csv"

# ── Category Mapping ──────────────────────────────────────────────────────────
# Any value not listed here → NaN → row dropped
CATEGORY_MAP = {
    # macro
    "macro"             : "macro",
    "fiscal"            : "macro",
    "monetary"          : "macro",
    "market_political"  : "macro",
    "macro|monetary"    : "macro",
    "macro|fiscal"      : "macro",

    # corporate
    "corporates"        : "corporate",
    "banking"           : "corporate",
    "equities"          : "corporate",
    "energy|banking"    : "corporate",
    "equities|forex"    : "corporate",
    "equities|commodities" : "corporate",

    # energy
    "energy"            : "energy",
    "commodities"       : "energy",
    "fiscal|energy"     : "energy",

    # forex
    "forex"             : "forex",
}

# ── Standardize Category ──────────────────────────────────────────────────────
def standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    # Mettis stores category as "economy" and real category in subcategory
    # If subcategory column exists, use it as the category source
    if "subcategory" in df.columns:
        df["category"] = df["subcategory"].fillna(df["category"])

    df["category"] = (
        df["category"]
        .str.strip()
        .str.lower()
        .map(CATEGORY_MAP)          # unmapped values become NaN
    )

    before = len(df)
    df.dropna(subset=["category"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"   🗑️  Dropped {dropped:,} rows with unmapped category")

    return df


# ── Merge ─────────────────────────────────────────────────────────────────────
def merge_news() -> pd.DataFrame:
    dfs = []

    for source, path in NEWS_FILES.items():
        if not path.exists():
            print(f"⚠️  Not found, skipping: {path}")
            continue

        df = pd.read_csv(path)
        df["source"] = source

        df = standardize_category(df)

        df = df[["date", "category", "title", "source"]]
        dfs.append(df)
        print(f"✅ Loaded {len(df):>6,} rows  ← {source}")

    if not dfs:
        raise FileNotFoundError("No news files found. Run scrapers first.")

    merged = pd.concat(dfs, ignore_index=True)

    # Standardize date
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Drop rows with missing date or title
    before = len(merged)
    merged.dropna(subset=["date", "title"], inplace=True)
    dropped = before - len(merged)
    if dropped:
        print(f"🗑️  Dropped {dropped:,} rows with null date/title")

    merged.reset_index(drop=True, inplace=True)
    return merged


# ── Save ──────────────────────────────────────────────────────────────────────
def save_merged(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved → {OUTPUT_PATH}")
    print(f"   Total rows : {len(df):,}")
    print(f"   Date range : {df['date'].min()} → {df['date'].max()}")
    print("\n📊 Rows per source:")
    print(df["source"].value_counts().to_string())
    print("\n📊 Rows per category:")
    print(df["category"].value_counts().to_string())


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Merging news sources")
    print("=" * 60)
    df = merge_news()
    save_merged(df)
    print("=" * 60)
    print("  Done")
    print("=" * 60)
