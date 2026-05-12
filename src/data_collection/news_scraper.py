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

Relevance filtering:
  - Every article title must contain at least one Pakistan/PSX anchor
  - Irrelevant articles are dropped before FinBERT scoring

Caching (3 levels):
  1. news_sentiment_daily.csv exists -> load and return immediately
  2. articles_scored.csv exists      -> skip scraping + FinBERT, re-aggregate only
  3. Neither exists                  -> full pipeline from scratch
"""

import os, asyncio, logging, hashlib, warnings, sys, subprocess
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

# ── Relevance anchors ─────────────────────────────────────────────────────────
_RELEVANCE_ANCHORS = [
    "pakistan", "psx", "kse", "pkr", "karachi", "lahore", "islamabad",
    "rupee", "sbp", "secp", "hbl", "mcb", "ubl", "ogdc", "ppl", "engro",
    "ffbl", "fatima", "luck", "dgkc", "psmc", "indu", "searl", "ptcl",
    "nccpl", "nepra", "fbr", "cpec", "imf pakistan", "pakistan stock",
    "pakistani", "islamabad", "lahore stock", "karachi stock",
]

def _is_relevant(title: str) -> bool:
    t = title.lower()
    return any(anchor in t for anchor in _RELEVANCE_ANCHORS)

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


def _cache_valid(path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            return False
        log.info("Cache hit — CSV covers %s -> %s (%d rows)",
                 df["date"].min().date(), df["date"].max().date(), len(df))
        return True
    except Exception as e:
        log.warning("Cache check failed: %s", e)
        return False


def _push_to_github(files, start, end):
    try:
        cmds = [
            ["git", "-C", PROJECT_ROOT, "add"] + files,
            ["git", "-C", PROJECT_ROOT, "commit", "-m",
             f"Update news sentiment cache {start} -> {end}"],
            ["git", "-C", PROJECT_ROOT, "push"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        log.info("✅ Pushed updated CSVs to GitHub")
    except subprocess.CalledProcessError as e:
        log.warning("⚠️ GitHub push failed: %s", e.stderr.decode())


# ── Categorised Query Bank ────────────────────────────────────────────────────

QUERY_CATEGORIES = {

    "macro_pakistan": [
        "IMF Pakistan loan bailout package KSE",
        "Pakistan economy GDP inflation PSX",
        "State Bank Pakistan interest rate PKR",
        "Pakistani rupee PKR devaluation forex",
        "Pakistan foreign exchange reserves SBP",
        "Pakistan current account deficit surplus",
        "Pakistan budget deficit fiscal policy FBR",
        "Pakistan inflation CPI SPI economy",
        "Pakistan remittances workers inflow PKR",
        "Pakistan trade balance exports imports",
        "Pakistan external debt repayment IMF",
        "Pakistan tax revenue FBR collection economy",
        "Pakistan monetary policy tightening easing SBP",
        "Pakistan credit rating Moody Fitch economy",
        "Pakistan economic reform privatisation PSX",
        "Pakistan FATF grey list compliance economy",
        "Pakistan circular debt energy economy",
        "Pakistan poverty unemployment economy KSE",
        "Pakistan agriculture crop wheat cotton economy",
        "Pakistan manufacturing LSM industrial output",
    ],

    "psx_market": [
        "PSX Pakistan Stock Exchange KSE100",
        "KSE100 index bullish bearish PSX",
        "PSX company earnings results KSE",
        "Pakistan stock market rally sell-off KSE100",
        "KSE100 record high low points PSX",
        "PSX foreign investor outflow inflow KSE",
        "Pakistan stock exchange listing IPO PSX",
        "PSX circuit breaker trading halt KSE100",
        "KSE100 market capitalization PSX",
        "PSX brokerage TREC regulations SECP",
        "Pakistan equity market outlook KSE100",
        "KSE100 dividend yield returns PSX",
        "PSX SECP regulations compliance KSE",
        "Pakistan stock market correction crash KSE100",
        "KSE AllShare index performance PSX",
        "PSX trading volume turnover KSE100",
        "Pakistan mutual fund AUM flows PSX",
        "PSX index rebalancing constituents KSE",
        "Pakistan capital market development PSX",
        "SECP Pakistan securities policy KSE100",
    ],

    "energy_oil": [
        "Pakistan oil gas OGDC PPL KSE",
        "crude oil price Pakistan PSX KSE impact",
        "Pakistan LNG energy crisis economy",
        "Pakistan petroleum fuel price economy",
        "Pakistan electricity tariff hike NEPRA",
        "Pakistan power sector NEPRA KSE",
        "Pakistan gas shortage winter economy",
        "Pakistan refinery expansion upgrade PSX",
        "Pakistan renewable solar wind energy KSE",
        "Pakistan nuclear power plant economy",
        "Pakistan petroleum levy fuel subsidy economy",
        "Pakistan energy mix coal furnace oil KSE",
        "OGDC PPL exploration discovery PSX",
        "Pakistan offshore oil gas exploration KSE",
        "Pakistan pipeline gas import economy",
        "Pakistan petrol diesel price change PKR",
        "Pakistan RLNG terminal import economy",
        "Pakistan IPP independent power producer KSE",
        "Pakistan electricity load shedding economy",
        "Pakistan energy transition climate PSX",
    ],

    "banking_finance": [
        "HBL MCB UBL Pakistan bank earnings PSX",
        "Pakistan banking NPL loans KSE",
        "Pakistan bank profit interest income PSX",
        "State Bank Pakistan SBP policy rate PKR",
        "Pakistan microfinance digital banking KSE",
        "Pakistan banking sector CAR capital PSX",
        "HBL Habib Bank profit dividend PSX",
        "MCB Bank profit dividend KSE results",
        "UBL United Bank earnings result PSX",
        "Allied Bank ABL quarterly results KSE",
        "Bank Alfalah BAFL performance PSX",
        "Meezan Bank Islamic finance growth KSE",
        "Pakistan fintech digital payments economy",
        "Pakistan banking NPL provisioning KSE",
        "Pakistan credit growth private sector SBP",
        "Pakistan T-bill PIB yields auction SBP",
        "Pakistan banking sector merger acquisition PSX",
        "NBP National Bank Pakistan results KSE",
        "Pakistan insurance sector takaful PSX",
        "Pakistan mortgage housing finance KSE",
    ],

    "geopolitical_global": [
        "Pakistan India tensions conflict economy PSX",
        "Pakistan China CPEC investment KSE",
        "Iran Pakistan pipeline deal economy",
        "US Federal Reserve rate Pakistan rupee PKR",
        "Russia Ukraine commodity Pakistan economy",
        "Pakistan Saudi Arabia UAE investment PKR",
        "Pakistan Afghanistan border trade economy",
        "Pakistan Turkey bilateral trade economy",
        "Pakistan US relations sanctions economy",
        "China Pakistan economic corridor CPEC KSE",
        "Pakistan Gulf remittances workers PKR",
        "Pakistan emerging market capital KSE100",
        "commodity prices Pakistan impact PSX",
        "Pakistan dollar shortage forex crisis PKR",
        "Pakistan regional connectivity trade economy",
        "Pakistan Iran border trade sanctions economy",
        "Pakistan SCO Shanghai Cooperation economy",
        "Pakistan IMF World Bank ADB loans economy",
        "Pakistan Belt Road Initiative BRI CPEC",
        "Pakistan diaspora investment bonds PKR",
    ],

    "political_stability": [
        "Pakistan political crisis government economy",
        "Pakistan elections economy PSX KSE",
        "Pakistan Prime Minister policy economy",
        "Pakistan army military political economy",
        "Pakistan Supreme Court ruling economy PSX",
        "Pakistan PTI PDM government policy economy",
        "Pakistan coalition government stability KSE",
        "Pakistan protest strike business PSX impact",
        "Pakistan constitutional crisis economy KSE",
        "Pakistan general election result economy",
        "Pakistan cabinet reshuffle minister economy",
        "Pakistan parliament budget approval economy",
        "Pakistan provincial government economy KSE",
        "Pakistan political uncertainty investor PSX",
        "Pakistan governance reform accountability KSE",
        "Pakistan NAB corruption case business PSX",
        "Pakistan Senate National Assembly legislation economy",
        "Pakistan political party economic agenda PSX",
        "Pakistan civil military relations economy",
        "Pakistan policy continuity investor confidence PSX",
    ],

    "pakistani_media": [
        "Dawn News Pakistan economy stock KSE",
        "Geo News Pakistan business finance PSX",
        "ARY News Pakistan economy market KSE",
        "The News International Pakistan stocks PSX",
        "Express Tribune Pakistan economy PSX KSE",
        "Business Recorder Pakistan KSE market PSX",
        "Pakistan Observer economy inflation KSE",
        "Daily Pakistan economy rupee PKR",
        "Samaa News Pakistan economic crisis KSE",
        "Dunya News Pakistan economy budget PSX",
        "Profit Pakistan business finance KSE",
        "Pakistan Today market economy PSX",
        "Tribune Express Pakistan business KSE",
        "Dawn Business Pakistan corporate results PSX",
        "Geo Business Pakistan finance news KSE",
    ],

    "international_media_pakistan": [
        "Reuters Pakistan economy market KSE PSX",
        "Bloomberg Pakistan stocks rupee PKR",
        "Financial Times Pakistan economy KSE",
        "Wall Street Journal Pakistan economy PSX",
        "Al Jazeera Pakistan economy crisis KSE",
        "BBC Pakistan economy inflation PSX",
        "CNBC Pakistan emerging market KSE100",
        "The Economist Pakistan economy PSX",
        "Associated Press Pakistan finance KSE",
        "South China Morning Post Pakistan CPEC economy",
        "Nikkei Asia Pakistan economy KSE",
        "Gulf News Pakistan economy remittances PKR",
        "Arab News Pakistan trade investment economy",
        "Middle East Eye Pakistan economy KSE",
        "AFP Pakistan economy stock market PSX",
    ],

    "psx_official_corporate": [
        "PSX official announcement Pakistan Exchange KSE100",
        "SECP Securities Exchange Commission Pakistan order",
        "Pakistan stock exchange new listing PSX",
        "PSX corporate disclosure financial results KSE",
        "Pakistan company quarterly annual results PSX",
        "PSX dividend announcement Pakistan KSE",
        "Pakistan company rights issue bonus PSX",
        "SECP enforcement action Pakistan company KSE",
        "PSX trading rules regulations update SECP",
        "Pakistan company merger acquisition PSX KSE",
        "PSX index methodology change KSE100",
        "Pakistan Exchange Traded Fund ETF PSX",
        "PSX market maker liquidity provider KSE",
        "Pakistan company AGM EGM announcement PSX",
        "SECP prospectus IPO approval Pakistan PSX",
        "PSX corporate governance compliance SECP",
        "Pakistan privatisation divestment PSX KSE",
        "PSX settlement clearing NCCPL KSE",
        "Pakistan company sukuk bond issue PSX",
        "SECP mutual fund regulations Pakistan PSX",
    ],

    "sector_specific": [
        "Pakistan cement sector LUCK DGKC PSX KSE",
        "Pakistan fertiliser ENGRO FFBL FATIMA PSX",
        "Pakistan textile sector exports KSE PSX",
        "Pakistan auto sector PSMC INDU PSX KSE",
        "Pakistan pharma SEARL HINOON results PSX",
        "Pakistan telecom PTCL Jazz Telenor KSE",
        "Pakistan steel sector ISL ASTL PSX KSE",
        "Pakistan sugar sector mill PSX KSE",
        "Pakistan chemicals ICI Lotte PSX KSE",
        "Pakistan food NESTLE UNILEVER FFL PSX",
        "Pakistan real estate property PSX KSE",
        "Pakistan technology IT exports PSX KSE",
        "Pakistan media PEMRA broadcast PSX KSE",
        "Pakistan tobacco PAKT results PSX KSE",
        "Pakistan glass packaging Ghani PSX KSE",
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
                    # ── Relevance filter ──────────────────────────────────────
                    if not _is_relevant(title):
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
        # PSX closes ~15:30 PKT = 10:30 UTC; treat hour >= 10 UTC as after-market
        if d in trading_dates and hour >= 10:
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

    # ── has_news flag before filling ─────────────────────────────────────────
    daily["has_news"] = (daily["article_count"] > 0).fillna(False)

    # forward-fill max 1 day then reset to neutral/0.0
    daily["sentiment_score"] = daily["sentiment_score"].ffill(limit=1).fillna(0.0)
    daily["sentiment_label"] = daily["sentiment_label"].ffill(limit=1).fillna("neutral")

    for col in ("article_count", "positive_count", "negative_count", "neutral_count"):
        daily[col] = daily[col].fillna(0).astype(int)
    for cat in QUERY_CATEGORIES:
        daily[f"{cat}_score"] = daily[f"{cat}_score"].ffill(limit=1).fillna(0.0)

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
    sentiment_path    = os.path.join(processed_dir, "news_sentiment_daily.csv")
    raw_articles_path = os.path.join(raw_news_dir,  "articles_raw.csv")
    scored_path       = os.path.join(processed_dir, "articles_scored.csv")

    os.makedirs(raw_news_dir,  exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if trading_dates is None:
        trading_dates = pd.date_range(start=start, end=end, freq="B")

    # ── Level 1 cache: final sentiment CSV ───────────────────────────────────
    if _cache_valid(sentiment_path):
        log.info("✅ Level 1 cache hit — loading sentiment from disk")
        return pd.read_csv(sentiment_path, parse_dates=["date"])

    # ── Level 2 cache: scored articles CSV ───────────────────────────────────
    if _cache_valid(scored_path):
        log.info("✅ Level 2 cache hit — scored articles found, skipping scrape + FinBERT")
        articles_df = pd.read_csv(scored_path, parse_dates=["date"])
    else:
        # ── Level 3: full pipeline from scratch ──────────────────────────────
        log.info("No cache found — running full pipeline")
        articles_df = collect_all_articles(
            start         = start,
            end           = end,
            chunk_months  = news_cfg.get("chunk_months", 3),
            max_per_chunk = news_cfg.get("max_per_chunk", 5),
            concurrency   = news_cfg.get("concurrency", 50),
        )
        articles_df.to_csv(raw_articles_path, index=False)
        log.info("✓ Raw articles saved -> %d rows", len(articles_df))

        score_df    = FinBERTScorer(model_name=news_cfg["finbert_model"]).score(articles_df["title"].tolist())
        articles_df = pd.concat([articles_df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
        articles_df.to_csv(scored_path, index=False)
        log.info("✓ Scored articles saved -> %d rows", len(articles_df))

    # ── Align + aggregate (always runs if sentiment CSV missing) ──────────────
    articles_df     = align_to_trading_days(articles_df, trading_dates)
    sentiment_daily = aggregate_daily_sentiment(articles_df, start, end, trading_dates)
    sentiment_daily.to_csv(sentiment_path, index=False)
    log.info("✓ Daily sentiment saved -> %s  (%d rows, %.1f MB)",
             sentiment_path, len(sentiment_daily), os.path.getsize(sentiment_path) / 1e6)

    # ── Push updated CSVs to GitHub ───────────────────────────────────────────
    _push_to_github([raw_articles_path, scored_path, sentiment_path], start, end)

    return sentiment_daily


if __name__ == "__main__":
    run()
