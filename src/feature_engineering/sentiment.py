"""
sentiment.py
------------
Stage 1 – Inference
    Loads news_processed.csv, runs each title through FinBERT
    (ProsusAI/finbert) in batches and writes per-article scores to
    data/processed/news/news_sentiment.csv.

Stage 2 – Aggregation
    Pivots the per-article scores into one row per calendar date with
    one set of columns per news category (macro, energy, banking,
    forex, corporate).  Result is written to
    data/processed/news/news_sentiment_daily.csv and is ready to be
    merged with psx_prices_processed.csv on the 'date' column.

Output columns (news_sentiment_daily.csv)
    date
    sentiment_<cat>_mean        – mean signed compound score  (-1 → +1)
    sentiment_<cat>_std         – std of compound scores that day
    sentiment_<cat>_count       – number of articles that day
    sentiment_<cat>_pos_ratio   – fraction of positive articles
    sentiment_<cat>_neg_ratio   – fraction of negative articles

    where <cat> ∈ {macro, energy, banking, forex, corporate}

Usage
    python -m src.feature_engineering.sentiment          # run both stages
    python -m src.feature_engineering.sentiment --agg-only  # skip inference
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWS_DIR        = os.path.join(BASE_DIR, "data", "processed", "news")
INPUT_PATH      = os.path.join(NEWS_DIR, "news_processed.csv")
ARTICLE_OUT     = os.path.join(NEWS_DIR, "news_sentiment.csv")
DAILY_OUT       = os.path.join(NEWS_DIR, "news_sentiment_daily.csv")

# ── inference config ───────────────────────────────────────────────────────────
MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 64
MAX_LENGTH  = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# FinBERT output order: 0 = positive, 1 = negative, 2 = neutral
LABEL_MAP   = {0: "positive", 1: "negative", 2: "neutral"}
SIGN_MAP    = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# categories we expect in the data
CATEGORIES  = ["macro", "energy", "banking", "forex", "corporate"]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – PER-ARTICLE FINBERT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Download / load FinBERT tokenizer and classification head."""
    print(f"[sentiment] Loading {MODEL_NAME} on {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE).eval()
    return tokenizer, model


def predict_batch(texts: list, tokenizer, model) -> list:
    """
    Run one batch through FinBERT.

    Returns a list of dicts:
        sentiment_label    : positive | negative | neutral
        sentiment_score    : confidence of predicted label  (0–1)
        sentiment_compound : signed score (+score / -score / 0)
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(**encoded).logits, dim=-1)   # (B, 3)

    results = []
    for row in probs.cpu().tolist():
        idx      = row.index(max(row))
        label    = LABEL_MAP[idx]
        score    = round(row[idx], 6)
        compound = round(SIGN_MAP[label] * score, 6)
        results.append({
            "sentiment_label"   : label,
            "sentiment_score"   : score,
            "sentiment_compound": compound,
        })
    return results


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment_label, sentiment_score, sentiment_compound to df.
    Skips inference and reloads from disk if ARTICLE_OUT already exists.
    """
    if os.path.exists(ARTICLE_OUT):
        print(f"[sentiment] Article-level file found – loading from cache: {ARTICLE_OUT}")
        return pd.read_csv(ARTICLE_OUT, parse_dates=["date"])

    if "title" not in df.columns:
        raise ValueError("news_processed.csv must contain a 'title' column.")

    titles           = df["title"].fillna("").tolist()
    tokenizer, model = load_model()

    all_results = []
    for start in tqdm(range(0, len(titles), BATCH_SIZE), desc="FinBERT inference"):
        all_results.extend(predict_batch(titles[start: start + BATCH_SIZE], tokenizer, model))

    sentiment_cols = pd.DataFrame(all_results)
    out_df         = pd.concat([df.reset_index(drop=True), sentiment_cols], axis=1)

    os.makedirs(NEWS_DIR, exist_ok=True)
    out_df.to_csv(ARTICLE_OUT, index=False)
    print(f"[sentiment] Saved {len(out_df):,} article-level rows → {ARTICLE_OUT}")
    return out_df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – DAILY CATEGORY AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots per-article sentiment into one row per date.

    For each category in CATEGORIES the following columns are produced:
        sentiment_<cat>_mean
        sentiment_<cat>_std
        sentiment_<cat>_count
        sentiment_<cat>_pos_ratio
        sentiment_<cat>_neg_ratio

    Days with no articles for a category are filled with 0 (neutral baseline).
    """
    required = {"date", "category", "sentiment_compound", "sentiment_label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for aggregation: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # build a full date spine so every trading day appears even with no news
    date_spine = pd.DataFrame(
        {"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")}
    )

    agg_frames = [date_spine.set_index("date")]

    for cat in CATEGORIES:
        cat_df  = df[df["category"] == cat]

        if cat_df.empty:
            print(f"[sentiment]  ⚠  No articles found for category '{cat}' – columns will be 0")
            # create empty frame so concat still works
            empty = date_spine.copy()
            for suffix in ("mean", "std", "count", "pos_ratio", "neg_ratio"):
                empty[f"sentiment_{cat}_{suffix}"] = 0.0
            agg_frames.append(empty.set_index("date"))
            continue

        def pos_ratio(x):
            return (x == "positive").sum() / len(x) if len(x) else 0.0

        def neg_ratio(x):
            return (x == "negative").sum() / len(x) if len(x) else 0.0

        daily = (
            cat_df
            .groupby("date")
            .agg(
                **{
                    f"sentiment_{cat}_mean"     : ("sentiment_compound", "mean"),
                    f"sentiment_{cat}_std"      : ("sentiment_compound", "std"),
                    f"sentiment_{cat}_count"    : ("sentiment_compound", "count"),
                    f"sentiment_{cat}_pos_ratio": ("sentiment_label",    pos_ratio),
                    f"sentiment_{cat}_neg_ratio": ("sentiment_label",    neg_ratio),
                }
            )
        )
        agg_frames.append(daily)

    result = pd.concat(agg_frames, axis=1).reset_index()
    result.rename(columns={"index": "date"}, inplace=True)

    # ensure date column is clean after concat
    result["date"] = pd.to_datetime(result["date"])

    # fill NaN → 0  (days where a category had zero articles)
    sent_cols = [c for c in result.columns if c.startswith("sentiment_")]
    result[sent_cols] = result[sent_cols].fillna(0)

    # sort chronologically
    result.sort_values("date", inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


def print_summary(article_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    """Print a concise quality report after both stages complete."""
    print("\n" + "═" * 60)
    print("SENTIMENT SUMMARY")
    print("═" * 60)

    print("\n── Article-level label distribution ────────────────────")
    print(article_df["sentiment_label"].value_counts().to_string())

    print("\n── Mean compound score by source ────────────────────────")
    if "source" in article_df.columns:
        print(article_df.groupby("source")["sentiment_compound"]
                        .mean().round(4).to_string())

    print("\n── Mean compound score by category ──────────────────────")
    if "category" in article_df.columns:
        print(article_df.groupby("category")["sentiment_compound"]
                        .mean().round(4).to_string())

    print("\n── Daily aggregated table shape ─────────────────────────")
    print(f"  {daily_df.shape[0]:,} rows  ×  {daily_df.shape[1]} columns")
    print(f"  Date range : {daily_df['date'].min().date()}  →  {daily_df['date'].max().date()}")

    print("\n── Columns in news_sentiment_daily.csv ──────────────────")
    for col in daily_df.columns:
        print(f"  {col}")

    print("\n── Days with zero macro news (potential data gaps) ──────")
    zero_macro = (daily_df["sentiment_macro_count"] == 0).sum()
    print(f"  {zero_macro:,} days out of {len(daily_df):,}")

    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(agg_only: bool = False) -> None:
    print(f"[sentiment] Input  : {INPUT_PATH}")
    print(f"[sentiment] Device : {DEVICE}")

    # ── load raw news ──────────────────────────────────────────────────────────
    print(f"[sentiment] Reading news_processed.csv …")
    news_df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    print(f"[sentiment] Loaded {len(news_df):,} rows")

    # ── stage 1: inference ────────────────────────────────────────────────────
    if agg_only:
        if not os.path.exists(ARTICLE_OUT):
            raise FileNotFoundError(
                f"--agg-only requested but article-level file not found: {ARTICLE_OUT}"
            )
        print(f"[sentiment] --agg-only: skipping inference, loading {ARTICLE_OUT}")
        article_df = pd.read_csv(ARTICLE_OUT, parse_dates=["date"])
    else:
        article_df = run_inference(news_df)

    # ── stage 2: aggregation ──────────────────────────────────────────────────
    print("[sentiment] Aggregating sentiment by date × category …")
    daily_df = aggregate_by_category(article_df)

    os.makedirs(NEWS_DIR, exist_ok=True)
    daily_df.to_csv(DAILY_OUT, index=False)
    print(f"[sentiment] Saved {len(daily_df):,} daily rows → {DAILY_OUT}")

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(article_df, daily_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBERT sentiment pipeline")
    parser.add_argument(
        "--agg-only",
        action="store_true",
        help="Skip FinBERT inference and re-aggregate from existing news_sentiment.csv",
    )
    args = parser.parse_args()
    run(agg_only=args.agg_only)
