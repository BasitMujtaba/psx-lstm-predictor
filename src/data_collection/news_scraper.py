"""
src/data_collection/news_scraper.py
=====================================
Fetches event-driven, date-aligned news for PSX.
Queries are categorised by macro, sector, and
geopolitical topics so sentiment on each trading
day reflects what actually moved the market.

Saves:
  data/raw/news/articles_raw.csv           <- raw scraped articles
  data/raw/news/articles_scored.csv        <- articles + FinBERT scores
  data/processed/news_sentiment_daily.csv  <- daily aggregated sentiment

Date alignment:
  - Article published on day T  -> mapped to trading day T
  - Weekend / holiday articles  -> forward-filled to next trading day
  - Pre-market news (before 9am)-> impacts same day T
  - After-market news           -> forward-filled to T+1
"""

import os, time, logging, hashlib, warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests, feedparser
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

GNEWS_RSS = "https://news.google.com/rss/search"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))


def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


# ── Categorised Query Bank ────────────────────────────────────────────────────

QUERY_CATEGORIES = {

    "macro_pakistan": [
        "IMF Pakistan loan bailout package",
        "Pakistan economy GDP growth inflation",
        "State Bank Pakistan interest rate monetary policy",
        "Pakistan current account deficit balance of payments",
        "Pakistan federal budget fiscal deficit",
        "Pakistani rupee PKR exchange rate devaluation",
        "Pakistan foreign exchange reserves",
        "Pakistan FATF grey list black list",
    ],

    "psx_market": [
        "PSX Pakistan Stock Exchange KSE100",
        "Pakistan stock market rally crash",
        "KSE100 index bullish bearish",
        "PSX listed company earnings results profit loss",
        "Pakistan stock dividend announcement",
        "SECP Pakistan securities regulation",
    ],

    "energy_oil": [
        "Pakistan oil gas OGDC PPL exploration",
        "crude oil price OPEC impact Pakistan",
        "Pakistan LNG import price energy crisis",
        "Pakistan circular debt energy sector",
        "Pakistan petroleum levy fuel price",
        "Pakistan natural gas shortage",
    ],

    "banking_finance": [
        "HBL MCB UBL NBP Pakistan bank earnings",
        "Pakistan banking sector NPL non performing loans",
        "Pakistan microfinance banking regulation",
        "Pakistan stock broker margin financing",
        "Pakistan bank interest spread profit",
    ],

    "fertilizer_agriculture": [
        "Engro Fatima FFC EFERT Pakistan fertilizer",
        "Pakistan urea DAP fertilizer price subsidy",
        "Pakistan agriculture wheat cotton crop",
        "Pakistan food inflation commodity prices",
    ],

    "cement_construction": [
        "Lucky Cement DGKC MLCF Pakistan cement",
        "Pakistan cement dispatches offtake demand",
        "Pakistan construction CPEC infrastructure",
        "Pakistan real estate housing sector",
    ],

    "geopolitical_global": [
        "Pakistan India tensions military conflict",
        "Pakistan China CPEC Belt Road investment",
        "USA Pakistan relations aid sanctions",
        "Iran Pakistan pipeline energy deal",
        "Pakistan Afghanistan border trade",
        "Gulf remittances Pakistan overseas workers",
        "Pakistan Saudi Arabia UAE investment",
        "Russia Ukraine war commodity prices Pakistan impact",
        "US Federal Reserve rate hike emerging markets",
        "China economy slowdown Pakistan exports",
    ],

    "political_stability": [
        "Pakistan political crisis government stability",
        "Pakistan elections PTI PML PPP",
        "Pakistan army civil relations",
        "Pakistan Prime Minister economic policy",
        "Pakistan protest strike shutdown",
    ],
}


# ── RSS Fetching ──────────────────────────────────────────────────────────────

def _build_rss_url(query, after, before):
    date_q = f"{query} after:{after} before:{before}"
    return (f"{GNEWS_RSS}?q={requests.utils.quote(date_q)}"
            f"&hl=en-US&gl=PK&ceid=PK:en")


def fetch_one_chunk(query, after, before, max_per_chunk):
    url = _build_rss_url(
        query,
        after.strftime("%Y-%m-%d"),
        before.strftime("%Y-%m-%d"),
    )
    try:
        resp = requests.get(
            url, timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as exc:
        log.warning("RSS error for query '%s': %s", query[:40], exc)
        return []

    articles = []
    for entry in feed.entries[:max_per_chunk]:
        pub    = entry.get("published_parsed") or entry.get("updated_parsed")
        pub_dt = datetime(*pub[:6]) if pub else after
        title  = entry.get("title", "").strip()
        if not title:
            continue
        articles.append({
            "date":     pub_dt.strftime("%Y-%m-%d"),
            "hour":     pub_dt.hour,
            "title":    title,
            "category": "",
            "source":   entry.get("source", {}).get("title", ""),
            "url":      entry.get("link", ""),
            "_hash":    hashlib.md5(title.lower().encode()).hexdigest(),
        })
    return articles


def collect_all_articles(start, end, chunk_months, max_per_chunk, sleep_sec):
    start_dt  = datetime.strptime(start, "%Y-%m-%d")
    end_dt    = datetime.strptime(end,   "%Y-%m-%d")
    all_rows  = []
    seen_hash = set()

    total_queries = sum(len(v) for v in QUERY_CATEGORIES.values())
    log.info("Starting collection: %d categories, %d total queries",
             len(QUERY_CATEGORIES), total_queries)

    for category, queries in QUERY_CATEGORIES.items():
        log.info("=== Category: %s (%d queries) ===", category, len(queries))
        cursor = start_dt

        while cursor < end_dt:
            next_cur = min(
                cursor + relativedelta(months=chunk_months), end_dt
            )
            for query in queries:
                for art in fetch_one_chunk(
                    query, cursor, next_cur, max_per_chunk
                ):
                    h = art.pop("_hash")
                    if h not in seen_hash:
                        seen_hash.add(h)
                        art["category"] = category
                        all_rows.append(art)
                time.sleep(sleep_sec)
            cursor = next_cur

    df = pd.DataFrame(
        all_rows,
        columns=["date", "hour", "title", "category", "source", "url"]
    )
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    log.info("Total unique articles collected: %d", len(df))
    return df.reset_index(drop=True)


# ── FinBERT GPU-optimised Scoring ─────────────────────────────────────────────

class _TextDataset(Dataset):
    """Simple dataset so DataLoader can pre-fetch while GPU is busy."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


class FinBERTScorer:
    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self, model_name, batch_size=None):
        log.info("Loading FinBERT: %s", model_name)
        self.tok   = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # ── GPU setup ────────────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        if torch.cuda.is_available():
            # Use fp16 on GPU — 2x throughput, half memory
            self.model = self.model.half()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("GPU: %s  (%.1f GB VRAM)", gpu_name, gpu_mem)
            # Auto batch size: ~128 for T4/16GB, ~64 for smaller GPUs
            self.bs = batch_size or (128 if gpu_mem >= 14 else 64)
        else:
            log.warning("No GPU found — running on CPU (will be slow)")
            self.bs = batch_size or 16

        log.info("FinBERT device: %s | batch_size: %d", self.device, self.bs)

    def score(self, texts):
        if not texts:
            return pd.DataFrame(columns=["sentiment_score", "sentiment_label"])

        log.info("Tokenising %d texts ...", len(texts))
        # Tokenise ALL texts at once — much faster than per-batch tokenisation
        encodings = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=128,       # titles are short; 128 >> 512 for speed
            return_tensors="pt",
        )

        dataset    = _TextDataset(encodings)
        loader     = DataLoader(
            dataset,
            batch_size  = self.bs,
            shuffle     = False,
            pin_memory  = self.device.type == "cuda",   # faster H2D transfer
            num_workers = 0,                             # Colab: keep at 0
        )

        all_scores, all_labels = [], []

        log.info("Running FinBERT inference on %s ...", self.device)
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"FinBERT [{self.device}]",
                              unit="batch"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # fp16 on GPU, fp32 on CPU
                with torch.cuda.amp.autocast(
                    enabled=(self.device.type == "cuda")
                ):
                    logits = self.model(**batch).logits

                probs = F.softmax(logits.float(), dim=-1).cpu()

                all_scores.extend(
                    (probs[:, 0] - probs[:, 1]).tolist()
                )
                all_labels.extend(
                    [self.LABEL_MAP[i]
                     for i in probs.argmax(dim=-1).tolist()]
                )

        return pd.DataFrame({
            "sentiment_score": [round(s, 4) for s in all_scores],
            "sentiment_label": all_labels,
        })


# ── Date Alignment to Trading Days ───────────────────────────────────────────

def align_to_trading_days(df, trading_dates):
    trading_dates = pd.DatetimeIndex(sorted(trading_dates))

    def _map_date(row):
        d    = row["date"]
        hour = row["hour"]
        if d in trading_dates and hour >= 11:
            idx = trading_dates.get_loc(d)
            if idx + 1 < len(trading_dates):
                return trading_dates[idx + 1]
        future = trading_dates[trading_dates >= d]
        return future[0] if len(future) > 0 else d

    log.info("Aligning %d articles to trading days ...", len(df))
    df = df.copy()
    df["trading_date"] = df.apply(_map_date, axis=1)
    return df


# ── Daily Aggregation ─────────────────────────────────────────────────────────

def aggregate_daily_sentiment(df, start, end, trading_dates):
    daily = df.groupby("trading_date").agg(
        sentiment_score = ("sentiment_score", "mean"),
        article_count   = ("sentiment_score", "count"),
        positive_count  = ("sentiment_label",
                           lambda x: (x == "positive").sum()),
        negative_count  = ("sentiment_label",
                           lambda x: (x == "negative").sum()),
        neutral_count   = ("sentiment_label",
                           lambda x: (x == "neutral").sum()),
    ).reset_index().rename(columns={"trading_date": "date"})

    daily["sentiment_label"] = daily.apply(
        lambda r: max(
            {"positive": r.positive_count,
             "negative": r.negative_count,
             "neutral":  r.neutral_count},
            key=lambda k: {"positive": r.positive_count,
                           "negative": r.negative_count,
                           "neutral":  r.neutral_count}[k]
        ), axis=1
    )

    for cat in QUERY_CATEGORIES:
        cat_df = (df[df["category"] == cat]
                  .groupby("trading_date")["sentiment_score"]
                  .mean()
                  .reset_index()
                  .rename(columns={
                      "trading_date":    "date",
                      "sentiment_score": f"{cat}_score",
                  }))
        daily = daily.merge(cat_df, on="date", how="left")

    full_idx = pd.date_range(start=start, end=end, freq="D")
    daily    = daily.set_index("date").reindex(full_idx)
    daily.index.name = "date"

    daily["sentiment_score"] = daily["sentiment_score"].ffill().fillna(0.0)
    daily["sentiment_label"] = daily["sentiment_label"].ffill().fillna("neutral")

    for col in ("article_count", "positive_count",
                "negative_count", "neutral_count"):
        daily[col] = daily[col].fillna(0).astype(int)

    for cat in QUERY_CATEGORIES:
        daily[f"{cat}_score"] = daily[f"{cat}_score"].ffill().fillna(0.0)

    return daily.reset_index()


# ── Entry Point ───────────────────────────────────────────────────────────────

def run(cfg=None, trading_dates=None):
    """
    1. Scrapes articles        -> data/raw/news/articles_raw.csv
    2. Scores with FinBERT     -> data/raw/news/articles_scored.csv
    3. Aggregates daily        -> data/processed/news_sentiment_daily.csv
    Returns daily sentiment DataFrame.
    """
    if cfg is None:
        cfg = load_config()

    start    = cfg["data"]["start_date"]
    end      = cfg["data"]["end_date"]
    news_cfg = cfg["news"]

    # ── Resolve directories ───────────────────────────────────────────
    raw_news_dir  = _resolve(cfg["data"]["raw_news_dir"])
    processed_dir = _resolve(cfg["data"]["processed_dir"])
    os.makedirs(raw_news_dir,  exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    log.info("raw_news_dir  -> %s", raw_news_dir)
    log.info("processed_dir -> %s", processed_dir)

    # ── Step 1: Collect articles ──────────────────────────────────────
    articles_df = collect_all_articles(
        start         = start,
        end           = end,
        chunk_months  = news_cfg["chunk_months"],
        max_per_chunk = news_cfg["max_per_chunk"],
        sleep_sec     = news_cfg["sleep_sec"],
    )

    raw_articles_path = os.path.join(raw_news_dir, "articles_raw.csv")
    articles_df.to_csv(raw_articles_path, index=False)
    log.info("✓ Raw articles saved  -> %s  (%d rows)", raw_articles_path, len(articles_df))

    # ── Step 2: Score with FinBERT (GPU-optimised) ────────────────────
    scorer   = FinBERTScorer(model_name=news_cfg["finbert_model"])
    score_df = scorer.score(articles_df["title"].tolist())
    articles_df = pd.concat([articles_df.reset_index(drop=True),
                             score_df.reset_index(drop=True)], axis=1)

    scored_path = os.path.join(raw_news_dir, "articles_scored.csv")
    articles_df.to_csv(scored_path, index=False)
    log.info("✓ Scored articles saved -> %s  (%d rows)", scored_path, len(articles_df))

    # ── Step 3: Align to trading days ────────────────────────────────
    if trading_dates is None:
        trading_dates = pd.date_range(start=start, end=end, freq="B")
    articles_df = align_to_trading_days(articles_df, trading_dates)

    # ── Step 4: Aggregate to daily and save to processed ─────────────
    sentiment_daily = aggregate_daily_sentiment(
        articles_df, start, end, trading_dates
    )

    processed_path = os.path.join(processed_dir, "news_sentiment_daily.csv")
    sentiment_daily.to_csv(processed_path, index=False)

    if not os.path.exists(processed_path):
        raise RuntimeError(f"Processed sentiment file NOT saved -> {processed_path}")
    log.info(
        "✓ Daily sentiment saved -> %s  (%d rows, %.1f MB)",
        processed_path, len(sentiment_daily),
        os.path.getsize(processed_path) / 1e6
    )

    return sentiment_daily


if __name__ == "__main__":
    run()
