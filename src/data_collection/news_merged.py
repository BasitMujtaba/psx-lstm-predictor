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

# ── Irrelevance Filter ────────────────────────────────────────────────────────
# Titles containing ANY of these phrases (case-insensitive) will be dropped.
# These are foreign market / sports / non-Pakistan articles that add no signal
# for Pakistani stock market prediction.
IRRELEVANT_KEYWORDS = [
    # ── India (broad catch) ───────────────────────────────────────────────────
    "india ",                  # catches: india pulls, india tries, india gdp etc
    "india's ",                # catches: india's economy, india's rupee etc
    "indian ",                 # catches: indian rupee, indian economy etc
    "modi ",                   # indian PM news
    "new delhi",
    "reserve bank of india",
    "rbi ",

    # Foreign currencies
    "yuan", "renminbi", "yen ", "won ",
    "ringgit", "baht", "peso", "lira", "rand", "ruble", "shekel",
    "euro ", "sterling", "pound sterling",

    # Foreign markets / indices
    "sensex", "nifty", "bse ", "nse india", "bombay stock",
    "shanghai", "hang seng", "nikkei", "ftse", "dow jones",
    "s&p 500", "nasdaq", "wall street", "fed reserve",
    "us federal reserve", "european central bank",

    # Sports (misclassified articles common in dawn/brecorder)
    "hat-trick", "hat trick", "wicket", "century puts",
    "innings", "thrash", "outplay",
    "football", "cricket match", "ppfl", "krl", "wapda",
    "pia beat", "nbp beat", "hbl beat", "ztbl", "kpt score",
    "navy thrash", "paf beat", "army thrash", "ssgc beat",
    "kesc crush", "scores hat", "slams hat",

    # Other irrelevant geographies
    "bangladesh", "sri lanka", "myanmar", "vietnam",
    "african ", "latin america", "brazil ", "argentina ",
    "turkey inflation", "iran sanction",
]

# ── Standardize Category ──────────────────────────────────────────────────────
def standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    if "subcategory" in df.columns:
        df["category"] = df["subcategory"].fillna(df["category"])

    df["category"] = (
        df["category"]
        .str.strip()
        .str.lower()
        .map(CATEGORY_MAP)
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

    # Build a single mask: True = irrelevant (contains any blacklisted keyword)
    mask_irrelevant = pd.Series(False, index=df.index)
    for keyword in IRRELEVANT_KEYWORDS:
        mask_irrelevant |= title_lower.str.contains(keyword, na=False)

    before  = len(df)
    df      = df[~mask_irrelevant].copy()
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
        df = filter_irrelevant(df)         # ← filter after category clean

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
