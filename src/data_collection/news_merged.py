"""
================================================================================
 File   : src/data_collection/news_merged.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source | sentiment_score | sentiment_label
          Categories are standardized to 4 values:
            macro | corporate | energy | forex
          Rows are sorted by date across all sources
          Irrelevant non-Pakistan articles are filtered out
          Sentiment scored using FinBERT (GPU if available else CPU)
 Output : data/processed/news_merged.csv
================================================================================
"""

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

OUTPUT_PATH = PROCESSED / "news_merged.csv"

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE    = 32

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
    # Foreign currencies
    "yuan", "renminbi", "ringgit", "baht", "peso",
    "lira", "rand", "ruble", "shekel", "sterling", "pound sterling",

    # Foreign markets / indices
    "sensex", "nifty", "bse ", "nse india", "bombay stock",
    "shanghai", "hang seng", "nikkei", "ftse", "dow jones",
    "s&p 500", "nasdaq", "wall street",
    "us federal reserve", "european central bank",

    # UK / Brexit
    "brexit", "uk budget", "uk economy", "uk inflation",
    "bank of england", "theresa may", "boris johnson",

    # Specific exports not relevant to PSX
    "cutlery export", "cutlery import",
    "surgical export", "surgical instrument",
    "sports goods export", "leather export",

    # Other irrelevant geographies
    "bangladesh", "sri lanka", "myanmar", "vietnam",
    "latin america", "brazil ", "argentina ",
    "turkey inflation", "iran sanction",
    "afghanistan ", "african ",

    # Sports
    "hat-trick", "hat trick", "wicket", "century puts",
    "innings", "thrash", "outplay", "ppfl", "krl", "wapda",
    "pia beat", "nbp beat", "hbl beat", "ztbl", "kpt score",
    "navy thrash", "paf beat", "army thrash", "ssgc beat",
    "kesc crush", "scores hat", "slams hat",
]

# ── Irrelevance Filter — Regex (word boundary) ────────────────────────────────
IRRELEVANT_REGEX = [
    # India
    r"\bindia\b",
    r"\bindian\b",
    r"\bmodi\b",
    r"\bnew delhi\b",
    r"\brbi\b",

    # UK / Europe
    r"\buk\b",
    r"\bbrexit\b",
    r"\bpound\b",
    r"\bsterling\b",
    r"\beuro\b",
    r"\beuros\b",
    r"\becb\b",

    # US
    r"\bfederal reserve\b",
    r"\bwall street\b",

    # Asia / Other foreign
    r"\bsensex\b",
    r"\bnifty\b",
    r"\byen\b",
    r"\bwon\b",
    r"\byuan\b",
    r"\bchina\b",
    r"\bchinese\b",
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


def compute_sentiment(
    titles: list,
    tokenizer,
    model,
    device: torch.device,
) -> list:
    """
    Returns a sentiment score per title in range [-1, +1]:
        +1  = strong positive
         0  = neutral
        -1  = strong negative

    FinBERT output labels: positive=0, negative=1, neutral=2
    Score = P(positive) - P(negative)
    """
    scores = []

    for i in tqdm(range(0, len(titles), BATCH_SIZE), desc="   Scoring"):
        batch = titles[i : i + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits

        probs = softmax(logits, dim=1).cpu()

        pos = probs[:, 0]
        neg = probs[:, 1]

        batch_scores = (pos - neg).tolist()
        scores.extend(batch_scores)

    return scores


def score_to_label(score: float) -> str:
    """
    Convert numeric sentiment score to human-readable label.
        score >  0.1  → positive
        score < -0.1  → negative
        otherwise     → neutral
    """
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

    # Free GPU memory
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

    # Add FinBERT sentiment score + label
    merged = add_sentiment(merged)

    return merged


# ── Sanity Check ──────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame) -> None:
    print("\n🔍 Sanity Check:")

    checks = {
        "India articles"  : r"\bindia\b",
        "Indian articles" : r"\bindian\b",
        "UK articles"     : r"\buk\b",
        "Brexit articles" : r"\bbrexit\b",
        "Pound articles"  : r"\bpound\b",
        "China articles"  : r"\bchina\b",
        "Euro articles"   : r"\beuro\b",
    }

    all_clear = True
    for label, pattern in checks.items():
        leaked = df[df["title"].str.lower().str.contains(pattern, regex=True, na=False)]
        if len(leaked) > 0:
            print(f"   ⚠️  {label} leaked      : {len(leaked):,}")
            print(leaked["title"].head(3).to_string())
            all_clear = False
        else:
            print(f"   ✅ {label:<20} : 0")

    # Check only 4 categories exist
    cats     = sorted(df["category"].unique().tolist())
    expected = ["corporate", "energy", "forex", "macro"]
    if cats == expected:
        print(f"   ✅ Categories                : {cats}")
    else:
        print(f"   ⚠️  Unexpected categories    : {cats}")

    # Check sentiment_score
    if "sentiment_score" in df.columns:
        nulls = df["sentiment_score"].isna().sum()
        if nulls == 0:
            print(f"   ✅ sentiment_score           : no nulls, range [{df['sentiment_score'].min():.4f}, {df['sentiment_score'].max():.4f}]")
        else:
            print(f"   ⚠️  sentiment_score nulls    : {nulls:,}")
    else:
        print(f"   ⚠️  sentiment_score column missing")

    # Check sentiment_label
    if "sentiment_label" in df.columns:
        nulls  = df["sentiment_label"].isna().sum()
        labels = sorted(df["sentiment_label"].unique().tolist())
        if nulls == 0 and set(labels) <= {"positive", "negative", "neutral"}:
            print(f"   ✅ sentiment_label           : {labels}")
        else:
            print(f"   ⚠️  sentiment_label issue     : nulls={nulls}, labels={labels}")
    else:
        print(f"   ⚠️  sentiment_label column missing")

    # Check sources
    sources = sorted(df["source"].unique().tolist())
    print(f"   ✅ Sources                   : {sources}")

    if all_clear:
        print("\n   ✅ All checks passed")


# ── Save ──────────────────────────────────────────────────────────────────────
def save_merged(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved → {OUTPUT_PATH}")
    print(f"   Total rows : {len(df):,}")
    print(f"   Columns    : {df.columns.tolist()}")
    print(f"   Date range : {df['date'].min()} → {df['date'].max()}")
    print("\n📊 Rows per source:")
    print(df["source"].value_counts().to_string())
    print("\n📊 Rows per category:")
    print(df["category"].value_counts().to_string())
    print("\n📊 Sentiment label distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"\n📊 Sentiment score stats:")
    print(f"   Min    : {df['sentiment_score'].min():.4f}")
    print(f"   Max    : {df['sentiment_score'].max():.4f}")
    print(f"   Mean   : {df['sentiment_score'].mean():.4f}")
    print(f"   Median : {df['sentiment_score'].median():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Merging news sources + FinBERT Sentiment Scoring")
    print("=" * 60)
    df = merge_news()
    save_merged(df)
    sanity_check(df)
    print("=" * 60)
    print("  Done")
    print("=" * 60)
