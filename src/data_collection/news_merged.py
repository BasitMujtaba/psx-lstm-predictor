"""
================================================================================
 File   : src/data_collection/news_merged.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source
          Categories are standardized to 4 values:
            macro | corporate | energy | forex
          Rows are sorted by date across all sources
          Irrelevant non-Pakistan articles are filtered out
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
CATEGORY_MAP = {
    # macro
    "macro"                : "macro",
    "fiscal"               : "macro",
    "monetary"             : "macro",
    "market_political"     : "macro",
    "macro|monetary"       : "macro",
    "macro|fiscal"         : "macro",

    # corporate
    "corporates"           : "corporate",
    "banking"              : "corporate",
    "equities"             : "corporate",
    "energy|banking"       : "corporate",
    "equities|forex"       : "corporate",
    "equities|commodities" : "corporate",

    # energy
    "energy"               : "energy",
    "commodities"          : "energy",
    "fiscal|energy"        : "energy",

    # forex
    "forex"                : "forex",
}

# ── Irrelevance Filter — Substring ────────────────────────────────────────────
# Titles containing ANY of these phrases (case-insensitive) will be dropped
IRRELEVANT_KEYWORDS = [
    # Foreign currencies
    "yuan", "renminbi", "ringgit", "baht", "peso",
    "lira", "rand", "ruble", "shekel", "sterling", "pound sterling",

    # Foreign markets / indices
    "sensex", "nifty", "bse ", "nse india", "bombay stock",
    "shanghai", "hang seng", "nikkei", "ftse", "dow jones",
    "s&p 500", "nasdaq", "wall street",
    "us federal reserve", "european central bank",

    # Other irrelevant geographies
    "bangladesh", "sri lanka", "myanmar", "vietnam",
    "latin america", "brazil ", "argentina ",
    "turkey inflation", "iran sanction",

    # Sports (misclassified articles common in dawn/brecorder)
    "hat-trick", "hat trick", "wicket", "century puts",
    "innings", "thrash", "outplay", "ppfl", "krl", "wapda",
    "pia beat", "nbp beat", "hbl beat", "ztbl", "kpt score",
    "navy thrash", "paf beat", "army thrash", "ssgc beat",
    "kesc crush", "scores hat", "slams hat",
]

# ── Irrelevance Filter — Regex (word boundary) ────────────────────────────────
# Catches all forms: "India", "India's", "in India", "India," etc.
# \b ensures "indiana" or "indicate" are NOT matched
IRRELEVANT_REGEX = [
    r"\bindia\b",          # India / India's / in India / India, — all forms
    r"\bindian\b",         # Indian rupee, Indian economy etc
    r"\bmodi\b",           # Indian PM
    r"\bnew delhi\b",      # Indian capital
    r"\brbi\b",            # Reserve Bank of India
    r"\bsensex\b",
    r"\bnifty\b",
    r"\byen\b",            # Japanese yen
    r"\beuro\b",           # Euro currency
    r"\bwon\b",            # Korean won
]


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


# ── Irrelevance Filter ────────────────────────────────────────────────────────
def filter_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    title_lower = df["title"].str.lower()

    # Simple substring match
    mask = pd.Series(False, index=df.index)
    for keyword in IRRELEVANT_KEYWORDS:
        mask |= title_lower.str.contains(keyword, na=False)

    # Regex word boundary match
    for pattern in IRRELEVANT_REGEX:
        mask |= title_lower.str.contains(pattern, regex=True, na=False)

    before  = len(df)
    df      = df[~mask].copy()
    dropped = before - len(df)

    if dropped:
        print(f"   🚫 Dropped {dropped:,} irrelevant articles (foreign/sports)")

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
        df = filter_irrelevant(df)

        df = df[["date", "category", "title", "source"]]
        dfs.append(df)
        print(f"✅ Loaded {len(df):>6,} rows  ← {source}")

    if not dfs:
        raise FileNotFoundError("No news files found. Run scrapers first.")

    merged = pd.concat(dfs, ignore_index=True)

    # Convert to datetime before sorting
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    # Drop rows with missing date or title
    before = len(merged)
    merged.dropna(subset=["date", "title"], inplace=True)
    dropped = before - len(merged)
    if dropped:
        print(f"🗑️  Dropped {dropped:,} rows with null date/title")

    # Sort by date across all sources
    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Format date after sorting
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    return merged


# ── Sanity Check ──────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame) -> None:
    print("\n🔍 Sanity Check:")

    # Check no India articles leaked
    india_leak = df[df["title"].str.lower().str.contains(r"\bindia\b", regex=True, na=False)]
    print(f"   India articles remaining  : {len(india_leak):,}  ✅" if len(india_leak) == 0
          else f"   ⚠️  India articles leaked   : {len(india_leak):,}")

    # Check only 4 categories exist
    cats = df["category"].unique().tolist()
    print(f"   Unique categories         : {sorted(cats)}")

    # Check sources
    sources = df["source"].unique().tolist()
    print(f"   Unique sources            : {sorted(sources)}")


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
    sanity_check(df)
    print("=" * 60)
    print("  Done")
    print("=" * 60)
