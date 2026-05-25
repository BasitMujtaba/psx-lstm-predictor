"""
sentiment.py
------------
Stage 1 - Inference
    Loads news_processed.csv, runs each title through FinBERT
    (ProsusAI/finbert) in batches and writes per-article scores to
    data/processed/news/news_sentiment.csv.

Stage 2 - Category-Pivoted CSV
    Creates a new CSV with one row per date+category combination.
    Columns: date, category, title, sentiment_macro, sentiment_energy,
    sentiment_banking, sentiment_forex, sentiment_corporate.
    If multiple articles exist for the same date+category, their
    sentiment_compound values are averaged and titles joined with | .
    Categories with no articles for a date are set to 0.
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWS_DIR    = os.path.join(BASE_DIR, "data", "processed", "news")
INPUT_PATH  = os.path.join(NEWS_DIR, "news_processed.csv")
ARTICLE_OUT = os.path.join(NEWS_DIR, "news_sentiment.csv")
PIVOTED_OUT = os.path.join(NEWS_DIR, "news_sentiment_pivoted.csv")

MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 64
MAX_LENGTH  = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP   = {0: "positive", 1: "negative", 2: "neutral"}
SIGN_MAP    = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
CATEGORIES  = ["macro", "energy", "banking", "forex", "corporate"]


def load_model():
    print(f"[sentiment] Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE).eval()
    return tokenizer, model


def predict_batch(texts, tokenizer, model):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(**encoded).logits, dim=-1)

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


def run_inference(df):
    if os.path.exists(ARTICLE_OUT):
        print(f"[sentiment] Cache found - loading: {ARTICLE_OUT}")
        return pd.read_csv(ARTICLE_OUT, parse_dates=["date"])

    if "title" not in df.columns:
        raise ValueError("news_processed.csv must contain a title column.")

    titles           = df["title"].fillna("").tolist()
    tokenizer, model = load_model()

    all_results = []
    for start in tqdm(range(0, len(titles), BATCH_SIZE), desc="FinBERT inference"):
        all_results.extend(predict_batch(titles[start: start + BATCH_SIZE], tokenizer, model))

    sentiment_cols = pd.DataFrame(all_results)
    out_df         = pd.concat([df.reset_index(drop=True), sentiment_cols], axis=1)

    os.makedirs(NEWS_DIR, exist_ok=True)
    out_df.to_csv(ARTICLE_OUT, index=False)
    print(f"[sentiment] Saved {len(out_df):,} article-level rows to {ARTICLE_OUT}")
    return out_df


def build_pivoted(df):
    required = {"date", "category", "title", "sentiment_compound"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for pivoted build: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    grouped = (
        df.groupby(["date", "category"], sort=True)
        .agg(
            title        = ("title",              lambda x: " | ".join(x.dropna().astype(str))),
            avg_compound = ("sentiment_compound", "mean"),
        )
        .reset_index()
    )

    for cat in CATEGORIES:
        grouped[f"sentiment_{cat}"] = grouped.apply(
            lambda row: round(row["avg_compound"], 6) if row["category"] == cat else 0.0,
            axis=1,
        )

    grouped.drop(columns=["avg_compound"], inplace=True)

    col_order = ["date", "category", "title"] + [f"sentiment_{c}" for c in CATEGORIES]
    grouped   = grouped[col_order]

    grouped.sort_values(["date", "category"], inplace=True)
    grouped.reset_index(drop=True, inplace=True)
    return grouped


def print_summary(article_df, pivoted_df):
    print("\n" + "=" * 60)
    print("SENTIMENT SUMMARY")
    print("=" * 60)

    print("\n-- Article-level label distribution --")
    print(article_df["sentiment_label"].value_counts().to_string())

    print("\n-- Mean compound score by category --")
    if "category" in article_df.columns:
        print(article_df.groupby("category")["sentiment_compound"].mean().round(4).to_string())

    print("\n-- Pivoted table --")
    print(f"  {pivoted_df.shape[0]:,} rows x {pivoted_df.shape[1]} columns")
    print(f"  Unique dates: {pivoted_df['date'].nunique():,}")
    print("  Rows per category:")
    print(pivoted_df["category"].value_counts().to_string())

    print("\n-- Sample rows (first 5) --")
    print(pivoted_df.head().to_string(index=False))
    print("=" * 60 + "\n")


def run(agg_only=False):
    print(f"[sentiment] Input  : {INPUT_PATH}")
    print(f"[sentiment] Device : {DEVICE}")

    print("[sentiment] Reading news_processed.csv ...")
    news_df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    print(f"[sentiment] Loaded {len(news_df):,} rows")

    if agg_only:
        if not os.path.exists(ARTICLE_OUT):
            raise FileNotFoundError(f"--agg-only requested but file not found: {ARTICLE_OUT}")
        print(f"[sentiment] --agg-only: loading {ARTICLE_OUT}")
        article_df = pd.read_csv(ARTICLE_OUT, parse_dates=["date"])
    else:
        article_df = run_inference(news_df)

    print("[sentiment] Building category-pivoted CSV ...")
    pivoted_df = build_pivoted(article_df)
    os.makedirs(NEWS_DIR, exist_ok=True)
    pivoted_df.to_csv(PIVOTED_OUT, index=False)
    print(f"[sentiment] Saved {len(pivoted_df):,} pivoted rows to {PIVOTED_OUT}")

    print_summary(article_df, pivoted_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBERT sentiment pipeline")
    parser.add_argument("--agg-only", action="store_true",
                        help="Skip inference, re-aggregate from existing news_sentiment.csv")
    args = parser.parse_args()
    run(agg_only=args.agg_only)
