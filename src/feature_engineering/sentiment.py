"""
sentiment.py
------------
Computes FinBERT-based sentiment scores for news titles in news_processed.csv
and saves the result to data/processed/news/news_sentiment.csv.

Columns added:
    sentiment_label : positive | negative | neutral
    sentiment_score : confidence score of the predicted label
    sentiment_compound : signed score  (+score if positive, -score if negative, 0 if neutral)
"""

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "news", "news_processed.csv")
OUTPUT_PATH= os.path.join(BASE_DIR, "data", "processed", "news", "news_sentiment.csv")

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 64
MAX_LENGTH  = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load FinBERT tokenizer and model."""
    print(f"[sentiment] Loading FinBERT on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE).eval()
    return tokenizer, model


def predict_batch(texts: list[str], tokenizer, model) -> list[dict]:
    """Return list of {label, score, compound} dicts for a batch of texts."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**encoded).logits          # (B, 3)
        probs  = F.softmax(logits, dim=-1)        # (B, 3)

    # FinBERT label order: positive=0, negative=1, neutral=2
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    sign_map  = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    results = []
    for prob_row in probs.cpu().tolist():
        idx      = int(prob_row.index(max(prob_row)))
        label    = label_map[idx]
        score    = round(prob_row[idx], 6)
        compound = round(sign_map[label] * score, 6)
        results.append({"sentiment_label": label,
                         "sentiment_score": score,
                         "sentiment_compound": compound})
    return results


def run():
    # ── load data ──────────────────────────────────────────────────────────────
    print(f"[sentiment] Reading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[sentiment] Loaded {len(df):,} rows")

    if "title" not in df.columns:
        raise ValueError("Expected a 'title' column in news_processed.csv")

    titles = df["title"].fillna("").tolist()

    # ── batch inference ────────────────────────────────────────────────────────
    tokenizer, model = load_model()

    all_results = []
    for start in tqdm(range(0, len(titles), BATCH_SIZE), desc="FinBERT inference"):
        batch = titles[start : start + BATCH_SIZE]
        all_results.extend(predict_batch(batch, tokenizer, model))

    # ── merge & save ───────────────────────────────────────────────────────────
    sentiment_df = pd.DataFrame(all_results)
    out_df       = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[sentiment] Saved {len(out_df):,} rows → {OUTPUT_PATH}")

    # ── quick summary ──────────────────────────────────────────────────────────
    print("\n── Sentiment distribution ──────────────────────────────")
    print(out_df["sentiment_label"].value_counts())
    print("\n── Mean compound score by source ───────────────────────")
    if "source" in out_df.columns:
        print(out_df.groupby("source")["sentiment_compound"].mean().round(4))
    print("\n── Mean compound score by category ─────────────────────")
    if "category" in out_df.columns:
        print(out_df.groupby("category")["sentiment_compound"].mean().round(4))


if __name__ == "__main__":
    run()
