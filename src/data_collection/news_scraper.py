"""
================================================================================
 File   : src/data_collection/news_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source | sentiment_score | sentiment_label
          Categories are standardized to 4 values:
            macro | corporate | energy | forex
          Rows are sorted by date across all sources
          Duplicates removed across all sources and categories
          Sentiment scored using FinBERT (GPU if available else CPU)
 Outputs:
          data/processed/news_merged.csv                   <- per-article sentiment
          data/processed/news_aggregated_flags.csv         <- flag approach
          data/processed/news_aggregated_decay_catwise.csv <- category-wise decay approach

 Cache logic:
   1. Raw CSVs exist -> filter + dedupe + score + save + push merged, then aggregate + push
   2. Raw CSVs missing -> raise error, run scrapers first
================================================================================
"""

import re
import subprocess
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"
RAW_NEWS  = BASE_DIR / "data" / "raw" / "news"

RAW_NEWS_FILES = {
    "dawn"      : RAW_NEWS / "dawn_pakistan_raw.csv",
    "brecorder" : RAW_NEWS / "brecorder_pakistan_raw.csv",
    "mettis"    : RAW_NEWS / "mettis_pakistan_raw.csv",
}

OUTPUT_PATH = PROCESSED / "news_merged.csv"
FLAGS_PATH  = PROCESSED / "news_aggregated_flags.csv"
DECAY_PATH  = PROCESSED / "news_aggregated_decay_catwise.csv"

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE    = 32

SENTIMENT_COLS = [
    "sentiment_corporate",
    "sentiment_energy",
    "sentiment_forex",
    "sentiment_macro",
]

DECAY_FACTORS = {
    "sentiment_corporate" : 0.7,
    "sentiment_energy"    : 0.7,
    "sentiment_forex"     : 0.8,
    "sentiment_macro"     : 0.85,
}

# ── Category Mapping ──────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "macro"                : "macro",
    "fiscal"               : "macro",
    "monetary"             : "macro",
    "market_political"     : "macro",
    "macro|monetary"       : "macro",
    "macro|fiscal"         : "macro",
    "corporates"           : "corporate",
    "banking"              : "corporate",
    "equities"             : "corporate",
    "energy|banking"       : "corporate",
    "equities|forex"       : "corporate",
    "equities|commodities" : "corporate",
    "energy"               : "energy",
    "commodities"          : "energy",
    "fiscal|energy"        : "energy",
    "forex"                : "forex",
}


# ── Standardize Category ──────────────────────────────────────────────────────
def standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    if "subcategory" in df.columns:
        df["category"] = df["subcategory"].fillna(df["category"])
    df["category"] = df["category"].str.strip().str.lower().map(CATEGORY_MAP)
    before = len(df)
    df.dropna(subset=["category"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"   🗑️  Dropped {dropped:,} rows with unmapped category")
    return df


# ── Deduplication ─────────────────────────────────────────────────────────────
def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    SOURCE_PRIORITY = {"brecorder": 0, "dawn": 1, "mettis": 2}

    df = df.copy()
    df["_norm"] = df["title"].fillna("").apply(_normalize_title)

    # Pass 1: exact normalized title same date
    df["_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(99)
    df.sort_values(["date", "_norm", "_priority"], inplace=True)
    df = df.drop_duplicates(subset=["date", "_norm"], keep="first")

    # Pass 2: near-duplicate via 3-gram overlap same date
    def ngrams(text, n=3):
        words = text.split()
        return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set(words)

    keep_idx = []
    for date, group in df.groupby("date"):
        indices = group.index.tolist()
        norms   = group["_norm"].tolist()
        titles  = group["title"].tolist()
        dropped = set()
        for i in range(len(indices)):
            if indices[i] in dropped:
                continue
            grams_i = ngrams(norms[i])
            for j in range(i + 1, len(indices)):
                if indices[j] in dropped:
                    continue
                grams_j = ngrams(norms[j])
                union   = grams_i | grams_j
                if not union:
                    continue
                overlap = len(grams_i & grams_j) / len(union)
                if overlap >= 0.60:
                    if len(titles[i]) >= len(titles[j]):
                        dropped.add(indices[j])
                    else:
                        dropped.add(indices[i])
                        break
            if indices[i] not in dropped:
                keep_idx.append(indices[i])
    df = df.loc[keep_idx]

    # Pass 3: exact normalized title across all dates keep earliest
    df.sort_values("date", inplace=True)
    df = df.drop_duplicates(subset=["_norm"], keep="first")

    df = df.drop(columns=["_norm", "_priority"])
    df.reset_index(drop=True, inplace=True)
    dropped_total = before - len(df)
    print(f"   🔁 Removed {dropped_total:,} duplicates — {len(df):,} unique articles remain")
    return df


# ── FinBERT Sentiment ─────────────────────────────────────────────────────────
def load_finbert(device):
    print(f"\n🤖 Loading FinBERT on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.to(device)
    model.eval()
    print("   ✅ FinBERT loaded")
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
        probs = softmax(logits, dim=1).cpu()
        scores.extend((probs[:, 0] - probs[:, 1]).tolist())
    return scores


def score_to_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    return "neutral"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_finbert(device)
    print(f"\n📊 Computing sentiment for {len(df):,} articles ...")
    titles = df["title"].fillna("").tolist()
    scores = compute_sentiment(titles, tokenizer, model, device)
    df["sentiment_score"] = [round(s, 4) for s in scores]
    df["sentiment_label"] = df["sentiment_score"].apply(score_to_label)
    print("   ✅ Sentiment scoring complete")
    print(f"   Score range : {df['sentiment_score'].min():.4f} → {df['sentiment_score'].max():.4f}")
    print(f"   Mean score  : {df['sentiment_score'].mean():.4f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df


# ── GitHub Push ───────────────────────────────────────────────────────────────
def push_to_github(files: list, message: str):
    try:
        project_root = str(BASE_DIR)
        subprocess.run(["git", "-C", project_root, "pull", "--rebase", "origin", "main"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", project_root, "add"] + files,
                       check=True, capture_output=True)
        commit = subprocess.run(
            ["git", "-C", project_root, "commit", "-m", message],
            capture_output=True, text=True
        )
        if "nothing to commit" in commit.stdout or "nothing to commit" in commit.stderr:
            print("   ℹ️  Nothing new to commit")
            return
        if commit.returncode != 0:
            print(f"   ⚠️  Commit failed: {commit.stderr.strip()}")
            return
        subprocess.run(["git", "-C", project_root, "push"],
                       check=True, capture_output=True)
        print(f"   ✅ Pushed: {message}")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  GitHub push failed: {e.stderr.decode()}")


# ── Merge ─────────────────────────────────────────────────────────────────────
def merge_news() -> pd.DataFrame:
    raw_found = {k: v for k, v in RAW_NEWS_FILES.items() if v.exists()}
    if not raw_found:
        raise FileNotFoundError(
            "No raw news CSVs found. Run dawn_scraper, brecorder_scraper, mettis_scraper first."
        )

    print(f"\n📰 Loading {len(raw_found)} raw CSVs ...")
    dfs = []
    for source, path in raw_found.items():
        df = pd.read_csv(path)
        df["source"] = source
        df = standardize_category(df)
        df = df[["date", "category", "title", "source"]]
        dfs.append(df)
        print(f"   ✅ Loaded {len(df):>6,} rows  ← {source}")

    merged = pd.concat(dfs, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    before = len(merged)
    merged.dropna(subset=["date", "title"], inplace=True)
    dropped = before - len(merged)
    if dropped:
        print(f"   🗑️  Dropped {dropped:,} rows with null date/title")

    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    print("\n🔍 Deduplicating across all sources and categories ...")
    merged = deduplicate(merged)

    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged = add_sentiment(merged)
    return merged


# ── Aggregation ───────────────────────────────────────────────────────────────
def _base_aggregate(df: pd.DataFrame) -> pd.DataFrame:
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
    for col in SENTIMENT_COLS:
        if col not in agg.columns:
            agg[col] = float("nan")
    news_count        = df.groupby("date").size().reset_index(name="news_count")
    agg               = agg.merge(news_count, on="date", how="left")
    agg["news_count"] = agg["news_count"].fillna(0).astype(int)
    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


def aggregate_flags(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Building flag aggregation ...")
    agg = _base_aggregate(df)
    flag_map = {
        "sentiment_corporate" : "has_corporate",
        "sentiment_energy"    : "has_energy",
        "sentiment_forex"     : "has_forex",
        "sentiment_macro"     : "has_macro",
    }
    for sent_col, flag_col in flag_map.items():
        agg[flag_col] = agg[sent_col].notna().astype(float)
    agg[SENTIMENT_COLS] = agg[SENTIMENT_COLS].fillna(0.0)
    col_order = [
        "date",
        "sentiment_corporate", "has_corporate",
        "sentiment_energy",    "has_energy",
        "sentiment_forex",     "has_forex",
        "sentiment_macro",     "has_macro",
        "news_count",
    ]
    agg = agg[col_order]
    print(f"   ✅ Flags done — shape: {agg.shape}")
    return agg


def aggregate_decay(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Building decay aggregation ...")
    agg        = _base_aggregate(df)
    full_dates = pd.date_range(start=agg["date"].min(), end=agg["date"].max(), freq="D")
    agg        = agg.set_index("date").reindex(full_dates)
    agg.index.name    = "date"
    agg["news_count"] = agg["news_count"].fillna(0).astype(int)
    for col, factor in DECAY_FACTORS.items():
        values = agg[col].copy()
        for i in range(1, len(values)):
            if pd.isna(values.iloc[i]):
                values.iloc[i] = values.iloc[i - 1] * factor
        agg[col] = values
        print(f"   {col:<25} decay={factor}")
    agg[SENTIMENT_COLS] = agg[SENTIMENT_COLS].fillna(0.0).round(4)
    agg = agg.reset_index()
    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    agg = agg[["date"] + SENTIMENT_COLS + ["news_count"]]
    print(f"   ✅ Decay done — shape: {agg.shape}")
    return agg


# ── Sanity Check ──────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame) -> None:
    print("\n🔍 Sanity Check:")
    norm_titles = df["title"].fillna("").apply(_normalize_title)
    exact_dups  = norm_titles.duplicated().sum()
    print(f"   Exact dupes : {'⚠️  ' + str(exact_dups) if exact_dups else '✅ 0'}")
    cats = sorted(df["category"].unique().tolist())
    print(f"   Categories  : {cats}")
    print(f"   Total rows  : {len(df):,}")
    print(f"   Date range  : {df['date'].min()} → {df['date'].max()}")


# ── run() ─────────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  PSX News Scraper — Full Pipeline")
    print("=" * 60)

    print("\n📋 Raw CSV check:")
    for source, path in RAW_NEWS_FILES.items():
        print(f"   {source:<12}: {'✅ exists' if path.exists() else '❌ missing'}")

    df = merge_news()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved merged → {OUTPUT_PATH}  ({len(df):,} rows)")
    sanity_check(df)
    push_to_github(
        [str(OUTPUT_PATH.relative_to(BASE_DIR))],
        "Update news_merged.csv — deduped + FinBERT"
    )

    print("\n" + "=" * 60)
    flags_df = aggregate_flags(df)
    flags_df.to_csv(FLAGS_PATH, index=False)
    print(f"💾 Saved flags  → {FLAGS_PATH}  ({len(flags_df):,} rows)")
    push_to_github(
        [str(FLAGS_PATH.relative_to(BASE_DIR))],
        "Update news_aggregated_flags.csv"
    )

    print("\n" + "=" * 60)
    decay_df = aggregate_decay(df)
    decay_df.to_csv(DECAY_PATH, index=False)
    print(f"💾 Saved decay  → {DECAY_PATH}  ({len(decay_df):,} rows)")
    push_to_github(
        [str(DECAY_PATH.relative_to(BASE_DIR))],
        "Update news_aggregated_decay_catwise.csv"
    )

    print("\n" + "=" * 60)
    print("  ✅ All done")
    print(f"    {OUTPUT_PATH.name:<40} {len(df):>7,} rows")
    print(f"    {FLAGS_PATH.name:<40} {len(flags_df):>7,} rows")
    print(f"    {DECAY_PATH.name:<40} {len(decay_df):>7,} rows")
    print("=" * 60)
    return df, flags_df, decay_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
