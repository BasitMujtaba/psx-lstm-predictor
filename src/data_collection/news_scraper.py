"""
================================================================================
 File   : src/data_collection/news_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source | sentiment_score | sentiment_label
          Categories are standardized to 4 values:
            macro | corporate | energy | forex
          Rows are sorted by date across all sources
          Irrelevant non-Pakistan articles are filtered out
          Sentiment scored using FinBERT (GPU if available else CPU)
 Outputs:
          data/processed/news_merged.csv                   <- per-article sentiment
          data/processed/news_aggregated_flags.csv         <- flag approach
          data/processed/news_aggregated_decay_catwise.csv <- category-wise decay approach
================================================================================
"""

import os
import base64
import requests
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
PROCESSED  = BASE_DIR / "data" / "processed"

NEWS_FILES = {
    "dawn"      : PROCESSED / "dawn_news_processed.csv",
    "brecorder" : PROCESSED / "brecorder_news_processed.csv",
    "mettis"    : PROCESSED / "mettis_news_processed.csv",
}

OUTPUT_PATH      = PROCESSED / "news_merged.csv"
FLAGS_PATH       = PROCESSED / "news_aggregated_flags.csv"
DECAY_PATH       = PROCESSED / "news_aggregated_decay_catwise.csv"

FINBERT_MODEL    = "ProsusAI/finbert"
BATCH_SIZE       = 32

SENTIMENT_COLS   = [
    "sentiment_corporate",
    "sentiment_energy",
    "sentiment_forex",
    "sentiment_macro",
]

# ── Category-wise decay factors ───────────────────────────────────────────────
# corporate : 0.7  — earnings/corporate news — fast reaction
# energy    : 0.7  — energy prices change daily — fast decay
# forex     : 0.8  — rupee moves linger 1-2 days longer
# macro     : 0.85 — SBP/policy news takes longer to digest
DECAY_FACTORS = {
    "sentiment_corporate" : 0.7,
    "sentiment_energy"    : 0.7,
    "sentiment_forex"     : 0.8,
    "sentiment_macro"     : 0.85,
}

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
IRRELEVANT_KEYWORDS = [
    "yuan", "renminbi", "ringgit", "baht", "peso",
    "lira", "rand", "ruble", "shekel", "sterling", "pound sterling",
    "sensex", "nifty", "bse ", "nse india", "bombay stock",
    "shanghai", "hang seng", "nikkei", "ftse", "dow jones",
    "s&p 500", "nasdaq", "wall street",
    "us federal reserve", "european central bank",
    "brexit", "uk budget", "uk economy", "uk inflation",
    "bank of england", "theresa may", "boris johnson",
    "cutlery export", "cutlery import",
    "surgical export", "surgical instrument",
    "sports goods export", "leather export",
    "bangladesh", "sri lanka", "myanmar", "vietnam",
    "latin america", "brazil ", "argentina ",
    "turkey inflation", "iran sanction",
    "afghanistan ", "african ",
    "hat-trick", "hat trick", "wicket", "century puts",
    "innings", "thrash", "outplay", "ppfl", "krl", "wapda",
    "pia beat", "nbp beat", "hbl beat", "ztbl", "kpt score",
    "navy thrash", "paf beat", "army thrash", "ssgc beat",
    "kesc crush", "scores hat", "slams hat",
]

# ── Irrelevance Filter — Regex ────────────────────────────────────────────────
IRRELEVANT_REGEX = [
    r"\bindia\b", r"\bindian\b", r"\bmodi\b",
    r"\bnew delhi\b", r"\brbi\b",
    r"\buk\b", r"\bbrexit\b", r"\bpound\b",
    r"\bsterling\b", r"\beuro\b", r"\beuros\b", r"\becb\b",
    r"\bfederal reserve\b", r"\bwall street\b",
    r"\bsensex\b", r"\bnifty\b", r"\byen\b",
    r"\bwon\b", r"\byuan\b", r"\bchina\b", r"\bchinese\b",
]


# ── Standardize Category ──────────────────────────────────────────────────────
def standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    if "subcategory" in df.columns:
        df["category"] = df["subcategory"].fillna(df["category"])
    df["category"] = (
        df["category"].str.strip().str.lower().map(CATEGORY_MAP)
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
    mask = pd.Series(False, index=df.index)
    for keyword in IRRELEVANT_KEYWORDS:
        mask |= title_lower.str.contains(keyword, na=False)
    for pattern in IRRELEVANT_REGEX:
        mask |= title_lower.str.contains(pattern, regex=True, na=False)
    before  = len(df)
    df      = df[~mask].copy()
    dropped = before - len(df)
    if dropped:
        print(f"   🚫 Dropped {dropped:,} irrelevant articles (foreign/sports)")
    return df


# ── FinBERT Sentiment Scoring ─────────────────────────────────────────────────
def load_finbert(device: torch.device):
    print(f"\n🤖 Loading FinBERT on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.to(device)
    model.eval()
    print(f"   ✅ FinBERT loaded")
    return tokenizer, model


def compute_sentiment(titles, tokenizer, model, device):
    scores = []
    for i in tqdm(range(0, len(titles), BATCH_SIZE), desc="   Scoring"):
        batch   = titles[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs  = softmax(logits, dim=1).cpu()
        scores.extend((probs[:, 0] - probs[:, 1]).tolist())
    return scores


def score_to_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_finbert(device)
    print(f"\n📊 Computing sentiment for {len(df):,} articles ...")
    titles = df["title"].fillna("").tolist()
    scores = compute_sentiment(titles, tokenizer, model, device)
    df["sentiment_score"] = [round(s, 4) for s in scores]
    df["sentiment_label"] = df["sentiment_score"].apply(score_to_label)
    print(f"   ✅ Sentiment scoring complete")
    print(f"   Score range : {df['sentiment_score'].min():.4f} → {df['sentiment_score'].max():.4f}")
    print(f"   Mean score  : {df['sentiment_score'].mean():.4f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
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
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    before = len(merged)
    merged.dropna(subset=["date", "title"], inplace=True)
    dropped = before - len(merged)
    if dropped:
        print(f"🗑️  Dropped {dropped:,} rows with null date/title")

    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged = add_sentiment(merged)
    return merged


# ── Aggregation Helper ────────────────────────────────────────────────────────
def _base_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared first step for both aggregation approaches.
    Returns wide-format df with one row per date and 4 raw sentiment columns.
    Missing category on a date = NaN (not yet filled).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    agg = (
        df.groupby(["date", "category"])["sentiment_score"]
        .mean()
        .unstack(level="category")
        .reset_index()
    )
    agg.columns.name = None
    agg = agg.rename(columns={
        "corporate" : "sentiment_corporate",
        "energy"    : "sentiment_energy",
        "forex"     : "sentiment_forex",
        "macro"     : "sentiment_macro",
    })

    # Ensure all 4 columns exist even if a category had zero articles ever
    for col in SENTIMENT_COLS:
        if col not in agg.columns:
            agg[col] = float("nan")

    # Article count per day
    news_count = df.groupby("date").size().reset_index(name="news_count")
    agg        = agg.merge(news_count, on="date", how="left")
    agg["news_count"] = agg["news_count"].fillna(0).astype(int)

    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


# ── Approach 1 — Flag Aggregation ────────────────────────────────────────────
def aggregate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per date with:
      sentiment_corporate | has_corporate |
      sentiment_energy    | has_energy    |
      sentiment_forex     | has_forex     |
      sentiment_macro     | has_macro     |
      news_count

    has_X = 1.0  → real news existed for that category that day
    has_X = 0.0  → no news for that category (sentiment filled with 0.0)

    Model can use has_X to know whether to trust sentiment_X.
    Missing sentiment filled with 0.0 AFTER flags are set.
    """
    print("\n📊 Building flag aggregation ...")
    agg = _base_aggregate(df)

    flag_map = {
        "sentiment_corporate" : "has_corporate",
        "sentiment_energy"    : "has_energy",
        "sentiment_forex"     : "has_forex",
        "sentiment_macro"     : "has_macro",
    }

    # Set flags BEFORE filling zeros — NaN means no news
    for sent_col, flag_col in flag_map.items():
        agg[flag_col] = agg[sent_col].notna().astype(float)

    # Fill missing sentiment with 0.0 (neutral placeholder)
    agg[SENTIMENT_COLS] = agg[SENTIMENT_COLS].fillna(0.0)

    # Reorder columns cleanly
    col_order = [
        "date",
        "sentiment_corporate", "has_corporate",
        "sentiment_energy",    "has_energy",
        "sentiment_forex",     "has_forex",
        "sentiment_macro",     "has_macro",
        "news_count",
    ]
    agg = agg[col_order]

    print(f"   ✅ Flag aggregation complete — shape: {agg.shape}")
    print(f"   📅 Date range: {agg['date'].min().date()} → {agg['date'].max().date()}")
    print("   📊 Coverage (% of days with real news):")
    for flag_col in flag_map.values():
        print(f"      {flag_col}: {agg[flag_col].mean()*100:.1f}%")
    return agg


# ── Approach 2 — Category-wise Decay Aggregation ──────────────────────────────
def aggregate_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar day (including weekends) with:
      sentiment_corporate | sentiment_energy | sentiment_forex | sentiment_macro | news_count

    Missing category days filled via exponential decay:
      corporate : 0.7  — fast reaction, fades in ~1 week
      energy    : 0.7  — daily price changes, fast decay
      forex     : 0.8  — rupee moves linger slightly longer
      macro     : 0.85 — SBP/IMF policy news lingers longest

    Real news day    → decay_value = actual FinBERT score (resets signal)
    No news day      → decay_value = previous_value × decay_factor
    Signal gone      → approaches 0.0 naturally (genuine neutral = 0.0 is valid)

    Reindexed to full calendar so weekends/holidays are included.
    """
    print("\n📊 Building category-wise decay aggregation ...")
    agg = _base_aggregate(df)

    # Reindex to full calendar — handles weekends + market holidays
    full_dates = pd.date_range(start=agg["date"].min(), end=agg["date"].max(), freq="D")
    agg        = agg.set_index("date").reindex(full_dates)
    agg.index.name = "date"

    # Carry news_count = 0 on reindexed days
    agg["news_count"] = agg["news_count"].fillna(0).astype(int)

    # Apply decay per category
    for col, factor in DECAY_FACTORS.items():
        values = agg[col].copy()
        for i in range(1, len(values)):
            if pd.isna(values.iloc[i]):
                values.iloc[i] = values.iloc[i - 1] * factor
        agg[col] = values
        print(f"   {col:<25} decay factor = {factor}")

    # Fill any remaining NaNs at very start of data (before first article)
    agg[SENTIMENT_COLS] = agg[SENTIMENT_COLS].fillna(0.0).round(4)

    agg = agg.reset_index()
    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    agg = agg[["date"] + SENTIMENT_COLS + ["news_count"]]

    real_days  = (agg["news_count"] > 0).sum()
    decay_days = (agg["news_count"] == 0).sum()
    print(f"   ✅ Decay aggregation complete — shape: {agg.shape}")
    print(f"   📅 Date range : {agg['date'].min().date()} → {agg['date'].max().date()}")
    print(f"   📊 Real news days  : {real_days:,} ({real_days/len(agg)*100:.1f}%)")
    print(f"   📊 Decay-only days : {decay_days:,} ({decay_days/len(agg)*100:.1f}%)")
    return agg


# ── Sanity Check ──────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame) -> None:
    print("\n🔍 Sanity Check (per-article):")
    checks = {
        "India"  : r"\bindia\b",  "Indian" : r"\bindian\b",
        "UK"     : r"\buk\b",     "Brexit" : r"\bbrexit\b",
        "Pound"  : r"\bpound\b",  "China"  : r"\bchina\b",
        "Euro"   : r"\beuro\b",
    }
    all_clear = True
    for label, pattern in checks.items():
        leaked = df[df["title"].str.lower().str.contains(pattern, regex=True, na=False)]
        if len(leaked):
            print(f"   ⚠️  {label} leaked: {len(leaked):,}")
            all_clear = False
        else:
            print(f"   ✅ {label:<20}: 0")

    cats = sorted(df["category"].unique().tolist())
    if cats == ["corporate", "energy", "forex", "macro"]:
        print(f"   ✅ Categories: {cats}")
    else:
        print(f"   ⚠️  Unexpected categories: {cats}")

    if all_clear:
        print("\n   ✅ All sanity checks passed")


# ── GitHub Publisher ──────────────────────────────────────────────────────────
def push_to_github(local_path: Path, repo_relative_path: str) -> None:
    """Push a single file to GitHub via REST API (create or update)."""
    token = os.environ.get("GITHUB_TOKEN")
    repo  = os.environ.get("GITHUB_REPO")   # e.g. "yourusername/psx-lstm-predictor"

    if not token or not repo:
        print(f"   ⚠️  GITHUB_TOKEN or GITHUB_REPO not set — skipping GitHub push for {local_path.name}")
        return

    url     = f"https://api.github.com/repos/{repo}/contents/{repo_relative_path}"
    headers = {
        "Authorization" : f"token {token}",
        "Accept"        : "application/vnd.github.v3+json",
    }

    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    # Fetch existing SHA if file already exists (required for updates)
    existing = requests.get(url, headers=headers)
    payload  = {
        "message" : f"Auto-publish {local_path.name}",
        "content" : content_b64,
    }
    if existing.status_code == 200:
        payload["sha"] = existing.json()["sha"]

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in (200, 201):
        action = "Updated" if existing.status_code == 200 else "Created"
        print(f"   ✅ GitHub → {action} {repo_relative_path}")
    else:
        print(f"   ❌ GitHub push failed [{response.status_code}]: {response.json().get('message')}")


# ── Save ──────────────────────────────────────────────────────────────────────
def save_merged(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved per-article → {OUTPUT_PATH}  ({len(df):,} rows)")
    print(f"   Columns    : {df.columns.tolist()}")
    print(f"   Date range : {df['date'].min()} → {df['date'].max()}")
    print("\n📊 Rows per source:")
    print(df["source"].value_counts().to_string())
    print("\n📊 Rows per category:")
    print(df["category"].value_counts().to_string())
    print("\n📊 Sentiment label distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    push_to_github(OUTPUT_PATH, f"data/processed/{OUTPUT_PATH.name}")


def save_flags(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(FLAGS_PATH, index=False)
    print(f"\n💾 Saved flags      → {FLAGS_PATH}  ({len(df):,} rows)")
    print(f"   Columns : {df.columns.tolist()}")
    print("\n📊 Nulls per column:")
    print(df.isnull().sum().to_string())
    push_to_github(FLAGS_PATH, f"data/processed/{FLAGS_PATH.name}")


def save_decay(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DECAY_PATH, index=False)
    print(f"\n💾 Saved decay      → {DECAY_PATH}  ({len(df):,} rows)")
    print(f"   Columns : {df.columns.tolist()}")
    print("\n📊 Score ranges:")
    for col in SENTIMENT_COLS:
        print(f"   {col}: [{df[col].min():.4f}, {df[col].max():.4f}]  mean={df[col].mean():.4f}")
    print("\n📊 Nulls per column:")
    print(df.isnull().sum().to_string())
    push_to_github(DECAY_PATH, f"data/processed/{DECAY_PATH.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Merging news + FinBERT Sentiment Scoring")
    print("=" * 60)

    # Step 1 — merge all sources + score sentiment
    df = merge_news()
    save_merged(df)
    sanity_check(df)

    # Step 2 — flag aggregation
    print("\n" + "=" * 60)
    print("  Approach 1 — Flag Aggregation")
    print("=" * 60)
    flags_df = aggregate_flags(df)
    save_flags(flags_df)

    # Step 3 — category-wise decay aggregation
    print("\n" + "=" * 60)
    print("  Approach 2 — Category-wise Decay Aggregation")
    print("=" * 60)
    decay_df = aggregate_decay(df)
    save_decay(decay_df)

    print("\n" + "=" * 60)
    print("  Done — 3 CSVs saved and pushed to GitHub:")
    print(f"    {OUTPUT_PATH.name}")
    print(f"    {FLAGS_PATH.name}")
    print(f"    {DECAY_PATH.name}")
    print("=" * 60)
