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

import os, asyncio, logging, hashlib, warnings, sys
from datetime import datetime
from urllib.parse import quote_plus
from dateutil.relativedelta import relativedelta
import aiohttp
import feedparser
import nest_asyncio
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

GNEWS_RSS = "https://news.google.com/rss/search"

# ── Colab-safe PROJECT_ROOT ───────────────────────────────────────────────────
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    _cwd = os.path.abspath(".")
    PROJECT_ROOT = _cwd
    for _candidate in [_cwd] + [os.path.join(_cwd, d) for d in os.listdir(_cwd)
                                  if os.path.isdir(os.path.join(_cwd, d))]:
        if os.path.isfile(os.path.join(_candidate, "config.yaml")):
            PROJECT_ROOT = _candidate
            break

log.info("PROJECT_ROOT: %s", PROJECT_ROOT)


def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


def _cache_valid(path, start, end):
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path, parse_dates=["date"])
    return (str(df["date"].min().date()) <= start and
            str(df["date"].max().date()) >= end)


# ── Categorised Query Bank ────────────────────────────────────────────────────

QUERY_CATEGORIES = {

    "macro_pakistan": [
        "IMF Pakistan loan bailout package",
        "Pakistan economy GDP inflation",
        "State Bank Pakistan interest rate",
        "Pakistani rupee PKR devaluation",
        "Pakistan foreign exchange reserves",
        "Pakistan current account deficit surplus",
        "Pakistan budget deficit fiscal policy",
        "Pakistan inflation CPI SPI",
        "Pakistan remittances workers inflow",
        "Pakistan trade balance exports imports",
        "Pakistan external debt repayment",
        "Pakistan tax revenue FBR collection",
        "Pakistan monetary policy tightening easing",
        "Pakistan credit rating Moody Fitch",
        "Pakistan economic reform privatisation",
        "Pakistan FATF grey list compliance",
        "Pakistan circular debt energy sector",
        "Pakistan poverty unemployment economy",
        "Pakistan agriculture crop wheat cotton",
        "Pakistan manufacturing LSM industrial output",
    ],

    "psx_market": [
        "PSX Pakistan Stock Exchange KSE100",
        "KSE100 index bullish bearish",
        "PSX company earnings results",
        "Pakistan stock market rally sell-off",
        "KSE100 record high low points",
        "PSX foreign investor outflow inflow",
        "Pakistan stock exchange listing IPO",
        "PSX circuit breaker trading halt",
        "KSE100 market capitalization",
        "PSX brokerage TREC regulations",
        "Pakistan equity market outlook",
        "KSE100 dividend yield returns",
        "PSX SECP regulations compliance",
        "Pakistan stock market correction crash",
        "KSE AllShare index performance",
        "PSX trading volume turnover",
        "Pakistan mutual fund AUM flows",
        "PSX index rebalancing constituents",
        "Pakistan capital market development",
        "SECP Pakistan securities policy",
    ],

    "energy_oil": [
        "Pakistan oil gas OGDC PPL",
        "crude oil OPEC Pakistan impact",
        "Pakistan LNG energy crisis",
        "Pakistan petroleum fuel price",
        "Pakistan electricity tariff hike",
        "Pakistan power sector NEPRA",
        "Pakistan gas shortage winter",
        "Pakistan refinery expansion upgrade",
        "Pakistan renewable solar wind energy",
        "Pakistan nuclear power plant",
        "Pakistan petroleum levy fuel subsidy",
        "Pakistan energy mix coal furnace oil",
        "OGDC PPL exploration discovery",
        "Pakistan offshore oil gas exploration",
        "Pakistan pipeline gas import",
        "Pakistan petrol diesel price change",
        "Pakistan RLNG terminal import",
        "Pakistan IPP independent power producer",
        "Pakistan electricity load shedding",
        "Pakistan energy transition climate",
    ],

    "banking_finance": [
        "HBL MCB UBL Pakistan bank earnings",
        "Pakistan banking NPL loans",
        "Pakistan bank profit interest income",
        "State Bank Pakistan SBP policy rate",
        "Pakistan microfinance digital banking",
        "Pakistan banking sector CAR capital",
        "HBL Habib Bank international expansion",
        "MCB Bank profit dividend",
        "UBL United Bank earnings result",
        "Allied Bank ABL quarterly results",
        "Bank Alfalah BAFL performance",
        "Meezan Bank Islamic finance growth",
        "Pakistan fintech digital payments",
        "Pakistan banking NPL provisioning",
        "Pakistan credit growth private sector",
        "Pakistan T-bill PIB yields auction",
        "Pakistan banking sector merger acquisition",
        "NBP National Bank Pakistan results",
        "Pakistan insurance sector takaful",
        "Pakistan mortgage housing finance",
    ],

    "geopolitical_global": [
        "Pakistan India tensions conflict",
        "Pakistan China CPEC investment",
        "Iran Pakistan pipeline deal",
        "US Federal Reserve emerging markets",
        "Russia Ukraine commodity Pakistan",
        "Pakistan Saudi Arabia UAE investment",
        "Pakistan Afghanistan border trade",
        "Pakistan Turkey bilateral trade",
        "Pakistan US relations sanctions",
        "China Pakistan economic corridor update",
        "Pakistan Gulf remittances workers",
        "Pakistan Asia emerging market capital",
        "Global commodity prices Pakistan impact",
        "Pakistan dollar shortage forex crisis",
        "Pakistan regional connectivity trade",
        "Pakistan Iran border trade sanctions",
        "Pakistan SCO Shanghai Cooperation",
        "Pakistan IMF World Bank ADB loans",
        "Pakistan Belt Road Initiative BRI",
        "Pakistan diaspora investment bonds",
    ],

    "political_stability": [
        "Pakistan political crisis government",
        "Pakistan elections economy",
        "Pakistan Prime Minister policy",
        "Pakistan army military political",
        "Pakistan Supreme Court ruling economy",
        "Pakistan PTI PDM government policy",
        "Pakistan coalition government stability",
        "Pakistan protest strike business impact",
        "Pakistan martial law constitutional crisis",
        "Pakistan general election result",
        "Pakistan cabinet reshuffle minister",
        "Pakistan parliament budget approval",
        "Pakistan provincial government KPK Punjab Sindh",
        "Pakistan political uncertainty investor",
        "Pakistan governance reform accountability",
        "Pakistan NAB corruption case business",
        "Pakistan Senate National Assembly legislation",
        "Pakistan political party economic agenda",
        "Pakistan civil military relations",
        "Pakistan policy continuity investor confidence",
    ],

    "pakistani_media": [
        "Dawn News Pakistan economy stock",
        "Geo News Pakistan business finance",
        "ARY News Pakistan economy market",
        "The News International Pakistan stocks",
        "Express Tribune Pakistan economy PSX",
        "Business Recorder Pakistan KSE market",
        "Pakistan Observer economy inflation",
        "Daily Pakistan economy rupee",
        "Samaa News Pakistan economic crisis",
        "Dunya News Pakistan economy budget",
        "Profit Pakistan business finance",
        "Pakistan Today market economy",
        "Tribune Express Pakistan business",
        "Dawn Business Pakistan corporate results",
        "Geo Business Pakistan finance news",
    ],

    "international_media_pakistan": [
        "Reuters Pakistan economy market",
        "Bloomberg Pakistan stocks rupee",
        "Financial Times Pakistan economy",
        "Wall Street Journal Pakistan",
        "Al Jazeera Pakistan economy crisis",
        "BBC Pakistan economy inflation",
        "CNBC Pakistan emerging market",
        "The Economist Pakistan economy",
        "Associated Press Pakistan finance",
        "South China Morning Post Pakistan CPEC",
        "Nikkei Asia Pakistan economy",
        "Gulf News Pakistan economy remittances",
        "Arab News Pakistan trade investment",
        "Middle East Eye Pakistan economy",
        "AFP Pakistan economy stock market",
    ],

    "psx_official_corporate": [
        "PSX official announcement Pakistan Exchange",
        "SECP Securities Exchange Commission Pakistan order",
        "Pakistan stock exchange new listing",
        "PSX corporate disclosure financial results",
        "Pakistan company quarterly annual results",
        "PSX dividend announcement Pakistan",
        "Pakistan company rights issue bonus",
        "SECP enforcement action Pakistan company",
        "PSX trading rules regulations update",
        "Pakistan company merger acquisition PSX",
        "PSX index methodology change",
        "Pakistan Exchange Traded Fund ETF",
        "PSX market maker liquidity provider",
        "Pakistan company AGM EGM announcement",
        "SECP prospectus IPO approval Pakistan",
        "PSX corporate governance compliance",
        "Pakistan privatisation divestment PSX",
        "PSX settlement clearing NCCPL",
        "Pakistan company sukuk bond issue",
        "SECP mutual fund regulations Pakistan",
    ],

    "sector_specific": [
        "Pakistan cement sector demand prices LUCK DGKC",
        "Pakistan fertiliser sector ENGRO FFBL FATIMA",
        "Pakistan textile sector exports quota",
        "Pakistan auto sector PSMC INDU sales",
        "Pakistan pharma sector SEARL HINOON results",
        "Pakistan telecom PTCL Jazz Telenor",
        "Pakistan steel sector ISL ASTL demand",
        "Pakistan sugar sector mill crushing",
        "Pakistan chemicals sector ICI Lotte",
        "Pakistan food sector NESTLE UNILEVER FFL",
        "Pakistan real estate property DHA",
        "Pakistan technology IT exports software",
        "Pakistan media PEMRA broadcast advertising",
        "Pakistan tobacco PMI PAKT results",
        "Pakistan glass packaging Ghani AGC",
    ],
}


# ── Async RSS Scraper ─────────────────────────────────────────────────────────

def _build_jobs(start, end, chunk_months):
    jobs = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    for category, queries in QUERY_CATEGORIES.items():
        cursor = start_dt
        while cursor < end_dt:
            next_cur = min(cursor + relativedelta(months=chunk_months), end_dt)
            for query in queries:
                jobs.append({
                    "category": category,
                    "query":    query,
                    "after":    cursor.strftime("%Y-%m-%d"),
                    "before":   next_cur.strftime("%Y-%m-%d"),
                })
            cursor = next_cur
    return jobs


async def _fetch_job(session, job, semaphore, seen_hash, results, max_per_chunk):
    q, after, before, cat = job["query"], job["after"], job["before"], job["category"]
    url = (f"{GNEWS_RSS}?q={quote_plus(f'{q} after:{after} before:{before}')}"
           f"&hl=en-US&gl=PK&ceid=PK:en")
    async with semaphore:
        try:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                feed = feedparser.parse(await resp.read())
                for entry in feed.entries[:max_per_chunk]:
                    pub    = entry.get("published_parsed") or entry.get("updated_parsed")
                    pub_dt = datetime(*pub[:6]) if pub else datetime.strptime(after, "%Y-%m-%d")
                    title  = entry.get("title", "").strip()
                    if not title:
                        continue
                    h = hashlib.md5(title.lower().encode()).hexdigest()
                    if h not in seen_hash:
                        seen_hash.add(h)
                        results.append({
                            "date":     pub_dt.strftime("%Y-%m-%d"),
                            "hour":     pub_dt.hour,
                            "title":    title,
                            "category": cat,
                            "source":   entry.get("source", {}).get("title", ""),
                            "url":      entry.get("link", ""),
                        })
        except Exception:
            pass


async def _run_all_jobs(jobs, max_per_chunk, concurrency):
    seen_hash = set()
    results   = []
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=10, connect=5)) as session:
        tasks = [asyncio.ensure_future(_fetch_job(session, job, semaphore, seen_hash, results, max_per_chunk))
                 for job in jobs]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching RSS", unit="req", ncols=80):
            await coro
    return results


def collect_all_articles(start, end, chunk_months, max_per_chunk, concurrency=50):
    jobs = _build_jobs(start, end, chunk_months)
    log.info("Jobs: %d categories | %d queries | %d RSS requests",
             len(QUERY_CATEGORIES), sum(len(v) for v in QUERY_CATEGORIES.values()), len(jobs))
    nest_asyncio.apply()

    async def _run():
        return await _run_all_jobs(jobs, max_per_chunk, concurrency)

    try:
        asyncio.get_running_loop()
        results = asyncio.get_event_loop().run_until_complete(_run())
    except RuntimeError:
        results = asyncio.run(_run())

    if not results:
        log.warning("No articles collected — check internet / Google News access")
        return pd.DataFrame(columns=["date", "hour", "title", "category", "source", "url"])

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    log.info("✓ Unique articles collected: %d  (from %d requests)", len(df), len(jobs))
    return df.reset_index(drop=True)


# ── FinBERT GPU-optimised Scoring ─────────────────────────────────────────────

class _TextDataset(Dataset):
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
        self.tok    = BertTokenizer.from_pretrained(model_name)
        self.model  = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if torch.cuda.is_available():
            self.model = self.model.half()
            gpu_mem    = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("GPU: %s  (%.1f GB VRAM)", torch.cuda.get_device_name(0), gpu_mem)
            self.bs = batch_size or (128 if gpu_mem >= 14 else 64)
        else:
            log.warning("No GPU found — running on CPU (will be slow)")
            self.bs = batch_size or 16
        log.info("FinBERT device: %s | batch_size: %d", self.device, self.bs)

    def score(self, texts):
        if not texts:
            return pd.DataFrame(columns=["sentiment_score", "sentiment_label"])
        encodings = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        loader    = DataLoader(_TextDataset(encodings), batch_size=self.bs, shuffle=False,
                               pin_memory=(self.device.type == "cuda"), num_workers=0)
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"FinBERT [{self.device}]", unit="batch"):
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    logits = self.model(**batch).logits
                probs = F.softmax(logits.float(), dim=-1).cpu()
                all_scores.extend((probs[:, 0] - probs[:, 1]).tolist())
                all_labels.extend([self.LABEL_MAP[i] for i in probs.argmax(dim=-1).tolist()])
        return pd.DataFrame({"sentiment_score": [round(s, 4) for s in all_scores],
                             "sentiment_label": all_labels})


# ── Date Alignment to Trading Days ───────────────────────────────────────────

def align_to_trading_days(df, trading_dates):
    trading_dates = pd.DatetimeIndex(sorted(trading_dates))

    def _map_date(row):
        d    = pd.Timestamp(row["date"]).normalize()
        hour = row["hour"]
        if d in trading_dates and hour >= 11:
            idx = trading_dates.get_loc(d)
            if idx + 1 < len(trading_dates):
                return trading_dates[idx + 1]
        future = trading_dates[trading_dates >= d]
        return future[0] if len(future) > 0 else d

    log.info("Aligning %d articles to trading days ...", len(df))
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["trading_date"] = df.apply(_map_date, axis=1)
    return df


# ── Daily Aggregation ─────────────────────────────────────────────────────────

def aggregate_daily_sentiment(df, start, end, trading_dates):
    daily = df.groupby("trading_date").agg(
        sentiment_score = ("sentiment_score", "mean"),
        article_count   = ("sentiment_score", "count"),
        positive_count  = ("sentiment_label", lambda x: (x == "positive").sum()),
        negative_count  = ("sentiment_label", lambda x: (x == "negative").sum()),
        neutral_count   = ("sentiment_label", lambda x: (x == "neutral").sum()),
    ).reset_index().rename(columns={"trading_date": "date"})

    daily["sentiment_label"] = daily.apply(
        lambda r: max({"positive": r.positive_count, "negative": r.negative_count, "neutral": r.neutral_count},
                      key=lambda k: {"positive": r.positive_count, "negative": r.negative_count,
                                     "neutral": r.neutral_count}[k]), axis=1)

    for cat in QUERY_CATEGORIES:
        cat_df = (df[df["category"] == cat]
                  .groupby("trading_date")["sentiment_score"].mean()
                  .reset_index()
                  .rename(columns={"trading_date": "date", "sentiment_score": f"{cat}_score"}))
        daily = daily.merge(cat_df, on="date", how="left")

    full_idx = pd.date_range(start=start, end=end, freq="D")
    daily    = daily.set_index("date").reindex(full_idx)
    daily.index.name = "date"
    daily["sentiment_score"] = daily["sentiment_score"].ffill().fillna(0.0)
    daily["sentiment_label"] = daily["sentiment_label"].ffill().fillna("neutral")
    for col in ("article_count", "positive_count", "negative_count", "neutral_count"):
        daily[col] = daily[col].fillna(0).astype(int)
    for cat in QUERY_CATEGORIES:
        daily[f"{cat}_score"] = daily[f"{cat}_score"].ffill().fillna(0.0)
    return daily.reset_index()


# ── Entry Point ───────────────────────────────────────────────────────────────

def run(cfg=None, trading_dates=None):
    if cfg is None:
        cfg = load_config()

    start         = cfg["data"]["start_date"]
    end           = cfg["data"]["end_date"]
    news_cfg      = cfg["news"]
    raw_news_dir  = _resolve(cfg["data"]["raw_news_dir"])
    processed_dir = _resolve(cfg["data"]["processed_dir"])
    sentiment_path = os.path.join(processed_dir, "news_sentiment_daily.csv")

    os.makedirs(raw_news_dir,  exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # ── Cache check ───────────────────────────────────────────────────────────
    if _cache_valid(sentiment_path, start, end):
        log.info("✅ Cache valid — loading sentiment from disk")
        return pd.read_csv(sentiment_path, parse_dates=["date"])

    # ── Collect articles ──────────────────────────────────────────────────────
    articles_df = collect_all_articles(
        start         = start,
        end           = end,
        chunk_months  = news_cfg.get("chunk_months", 3),
        max_per_chunk = news_cfg.get("max_per_chunk", 5),
        concurrency   = news_cfg.get("concurrency", 50),
    )
    articles_df.to_csv(os.path.join(raw_news_dir, "articles_raw.csv"), index=False)
    log.info("✓ Raw articles saved -> %d rows", len(articles_df))

    # ── Score with FinBERT ────────────────────────────────────────────────────
    score_df    = FinBERTScorer(model_name=news_cfg["finbert_model"]).score(articles_df["title"].tolist())
    articles_df = pd.concat([articles_df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
    articles_df.to_csv(os.path.join(processed_dir, "articles_scored.csv"), index=False)
    log.info("✓ Scored articles saved -> %d rows", len(articles_df))

    # ── Align + aggregate ─────────────────────────────────────────────────────
    if trading_dates is None:
        trading_dates = pd.date_range(start=start, end=end, freq="B")
    articles_df     = align_to_trading_days(articles_df, trading_dates)
    sentiment_daily = aggregate_daily_sentiment(articles_df, start, end, trading_dates)
    sentiment_daily.to_csv(sentiment_path, index=False)
    log.info("✓ Daily sentiment saved -> %s  (%d rows, %.1f MB)",
             sentiment_path, len(sentiment_daily), os.path.getsize(sentiment_path) / 1e6)

    return sentiment_daily


if __name__ == "__main__":
    run()
