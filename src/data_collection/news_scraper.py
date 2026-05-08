"""
src/data_collection/news_scraper.py
=====================================
Fetches event-driven, date-aligned news for PSX.
Queries are categorised by macro, sector, and
geopolitical topics so sentiment on each trading
day reflects what actually moved the market.

Date alignment:
  - Article published on day T  → mapped to trading day T
  - Weekend / holiday articles  → forward-filled to next trading day
  - Pre-market news (before 9am)→ impacts same day T
  - After-market news           → forward-filled to T+1
"""

import os, time, logging, hashlib, warnings
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import requests, feedparser
import pandas as pd
import torch
import torch.nn.functional as F
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

# ── Categorised Query Bank ────────────────────────────────────────────────────
# Each category captures a different driver of PSX price movement.
# All queries are time-windowed so articles are date-aligned to stock data.

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


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
            "hour":     pub_dt.hour,             # ← keep hour for alignment
            "title":    title,
            "category": "",                      # filled by caller
            "source":   entry.get("source", {}).get("title", ""),
            "url":      entry.get("link", ""),
            "_hash":    hashlib.md5(title.lower().encode()).hexdigest(),
        })
    return articles


def collect_all_articles(start, end, chunk_months, max_per_chunk, sleep_sec):
    """
    Loops through all query categories and all time chunks.
    Returns one DataFrame with every unique article tagged
    with its category.
    """
    start_dt  = datetime.strptime(start, "%Y-%m-%d")
    end_dt    = datetime.strptime(end,   "%Y-%m-%d")
    all_rows  = []
    seen_hash = set()

    for category, queries in QUERY_CATEGORIES.items():
        log.info("=== Category: %s (%d queries) ===", category, len(queries))
        cursor = start_dt

        while cursor < end_dt:
            next_cur = min(
                cursor + relativedelta(months=chunk_months), end_dt
            )
            for query in queries:
                log.info("  RSS '%s' [%s -> %s]",
                         query[:45],
                         cursor.strftime("%Y-%m-%d"),
                         next_cur.strftime("%Y-%m-%d"))
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


# ── FinBERT Scoring ───────────────────────────────────────────────────────────

class FinBERTScorer:
    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self, model_name, batch_size=32):
        log.info("Loading FinBERT: %s", model_name)
        self.tok    = BertTokenizer.from_pretrained(model_name)
        self.model  = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.bs     = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        log.info("FinBERT running on: %s", self.device)

    def score(self, texts):
        scores, labels = [], []
        for i in tqdm(range(0, len(texts), self.bs),
                      desc="FinBERT scoring", unit="batch"):
            batch = texts[i: i + self.bs]
            enc   = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = F.softmax(
                    self.model(**enc).logits, dim=-1
                ).cpu()
            for p in probs.tolist():
                scores.append(round(p[0] - p[1], 4))
                labels.append(
                    self.LABEL_MAP[int(torch.tensor(p).argmax())]
                )
        return pd.DataFrame({
            "sentiment_score": scores,
            "sentiment_label": labels,
        })


# ── Date Alignment to Trading Days ───────────────────────────────────────────

def align_to_trading_days(df, trading_dates):
    """
    Maps each article to the correct trading day:
      - Article on a trading day before market close  → same day
      - Article on a trading day after market close   → next trading day
      - Article on weekend / holiday                  → next trading day

    PSX market hours: 09:30 - 15:30 PKT (UTC+5)
    We use hour >= 11 UTC (4pm PKT) as after-market threshold.
    """
    trading_dates = pd.DatetimeIndex(sorted(trading_dates))

    def _map_date(row):
        d    = row["date"]
        hour = row["hour"]
        # After-market (after 11 UTC = 4pm PKT) → next trading day
        if d in trading_dates and hour >= 11:
            idx = trading_dates.get_loc(d)
            if idx + 1 < len(trading_dates):
                return trading_dates[idx + 1]
        # Find next available trading day on or after this date
        future = trading_dates[trading_dates >= d]
        if len(future) > 0:
            return future[0]
        return d

    log.info("Aligning %d articles to trading days ...", len(df))
    df = df.copy()
    df["trading_date"] = df.apply(_map_date, axis=1)
    return df


# ── Daily Aggregation ─────────────────────────────────────────────────────────

def aggregate_daily_sentiment(df, start, end, trading_dates):
    """
    Aggregates sentiment per trading day.
    Also produces per-category sentiment so the model can
    see macro vs geopolitical vs sector sentiment separately.
    """
    # Overall daily sentiment
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

    # Per-category sentiment columns
    # e.g. macro_pakistan_score, geopolitical_global_score
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

    # Reindex to every calendar day then forward-fill gaps
    full_idx = pd.date_range(start=start, end=end, freq="D")
    daily    = daily.set_index("date").reindex(full_idx)
    daily.index.name = "date"

    # Forward-fill then neutral for very start of dataset
    daily["sentiment_score"] = daily["sentiment_score"].ffill().fillna(0.0)
    daily["sentiment_label"] = daily["sentiment_label"].ffill().fillna("neutral")

    for col in ("article_count", "positive_count",
                "negative_count", "neutral_count"):
        daily[col] = daily[col].fillna(0).astype(int)

    for cat in QUERY_CATEGORIES:
        col = f"{cat}_score"
        daily[col] = daily[col].ffill().fillna(0.0)

    return daily.reset_index()


# ── Entry Point ───────────────────────────────────────────────────────────────

def run(cfg=None, trading_dates=None):
    """
    trading_dates: list or DatetimeIndex of actual PSX trading days.
    Pass this from pipeline.py after loading psx_prices.csv so
    alignment is exact. If None, uses calendar days only.
    """
    if cfg is None:
        cfg = load_config()

    start    = cfg["data"]["start_date"]
    end      = cfg["data"]["end_date"]
    out_dir  = cfg["data"]["raw_news_dir"]
    news_cfg = cfg["news"]
    os.makedirs(out_dir, exist_ok=True)

    # Step 1 — collect
    articles_df = collect_all_articles(
        start         = start,
        end           = end,
        chunk_months  = news_cfg["chunk_months"],
        max_per_chunk = news_cfg["max_per_chunk"],
        sleep_sec     = news_cfg["sleep_sec"],
    )
    articles_df.to_csv(
        os.path.join(out_dir, "articles_raw.csv"), index=False
    )
    log.info("Raw articles saved: %d rows", len(articles_df))

    # Step 2 — score with FinBERT
    scorer      = FinBERTScorer(model_name=news_cfg["finbert_model"])
    score_df    = scorer.score(articles_df["title"].tolist())
    articles_df = pd.concat([articles_df, score_df], axis=1)
    articles_df.to_csv(
        os.path.join(out_dir, "articles_scored.csv"), index=False
    )

    # Step 3 — align to trading days
    if trading_dates is None:
        trading_dates = pd.date_range(start=start, end=end, freq="B")
    articles_df = align_to_trading_days(articles_df, trading_dates)

    # Step 4 — aggregate to daily
    sentiment_daily = aggregate_daily_sentiment(
        articles_df, start, end, trading_dates
    )
    out_path = os.path.join(out_dir, "news_sentiment_daily.csv")
    sentiment_daily.to_csv(out_path, index=False)
    log.info("Daily sentiment saved -> %s", out_path)
    return sentiment_daily


if __name__ == "__main__":
    run()
