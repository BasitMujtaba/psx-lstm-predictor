"""
================================================================================
 File   : src/data_collection/mettis_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Scrapes financial news from Mettis Global for Pakistan market via
          JSON API endpoints. Filters, deduplicates, and saves processed CSV.

 Saves:
   data/raw/news/mettis_pakistan_raw.csv
   data/processed/news/mettis_pakistan_processed.csv

 Cache logic:
   1. If raw CSV exists -> skip scraping entirely, go straight to processing
   2. If raw CSV does not exist -> scrape from scratch
================================================================================
"""

import asyncio, csv, os, random, re, json, logging
import pandas as pd
import httpx
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


# ── Filters ───────────────────────────────────────────────────────────────────

FOREIGN_SIGNALS = (
    r"\b(european[ ]stock|europe[ ]stock|asian[ ]markets|asian[ ]stocks"
    r"|wall[ ]street|nasdaq|dow[ ]jones|s&p[ ]500|ftse|dax|nikkei|hang[ ]seng"
    r"|sensex|nifty|bse[ ]india|nse[ ]india|shanghai|shenzhen|csi[ ]300"
    r"|kospi|straits[ ]times|asx[ ]200|federal[ ]reserve|fed[ ]reserve"
    r"|ecb|bank[ ]of[ ]england|boe|rbi|pboc"
    r"|us[ ]stocks|us[ ]markets|global[ ]markets|world[ ]markets|emerging[ ]markets"
    r"|china[ ]stocks|india[ ]stocks|uk[ ]stocks|euro[ ]zone|eurozone"
    r"|yuan|renminbi|yen[ ]falls|yen[ ]rises|euro[ ]falls|euro[ ]rises"
    r"|brent[ ]falls|brent[ ]rises|crude[ ]falls|crude[ ]rises"
    r"|oil[ ]falls|oil[ ]rises|gold[ ]falls|gold[ ]rises"
    r"|dollar[ ]index|dxy|london[ ]stock|new[ ]york[ ]stock|tokyo[ ]stock"
    r"|hong[ ]kong[ ]stock|toronto[ ]stock)\b"
)

EXCLUDE_KEYWORDS = (
    r"\b(bollywood|lollywood|oscar|grammy|actor|actress|film[ ]review|movie"
    r"|drama[ ]serial|reality[ ]show|game[ ]show|fashion[ ]week|skin[ ]care"
    r"|hair[ ]care|make.?up|beauty|recipe|cooking|restaurant[ ]review"
    r"|horoscope|zodiac|astrology|travel[ ]guide|tourism|visa[ ]guide"
    r"|book[ ]review|novel|poetry|cricket[ ]score|match[ ]report|match[ ]preview"
    r"|ipl|champions[ ]trophy|football[ ]result|hockey[ ]result|tennis[ ]result"
    r"|golf[ ]result|fifa|uefa|nba|nhl|nfl|olympics|phone[ ]review|laptop[ ]review"
    r"|gadget[ ]review|gaming|video[ ]game|viral[ ]video|tiktok|instagram[ ]reel"
    r"|youtube[ ]star|influencer|meme|weather[ ]forecast|rain[ ]forecast"
    r"|murder[ ]case|robbery|kidnapping|road[ ]accident|drug[ ]haul|drug[ ]bust"
    r"|weight[ ]loss|diet[ ]plan|yoga|meditation)\b"
)

CATEGORY_PATTERNS = {
    "equities": (
        r"\b(stock|share|kse|psx|index|equit|listed|scrip|dividend|ipo"
        r"|market[ ]cap|kse.?100|allotment|bonus[ ]share|right[ ]share"
        r"|trading|bourse|nccpl|trec|rally|sell[ ]off|bull|bear"
        r"|upper[ ]lock|lower[ ]lock|circuit[ ]breaker)\b"
    ),
    "commodities": (
        r"\b(gold|silver|copper|wheat|sugar|cotton|commodity|commodities"
        r"|rice|maize|zinc|tin|palm[ ]oil|fertilizer|urea|dap"
        r"|cement[ ]dispatches|cement[ ]offtake|steel[ ]price)\b"
    ),
    "forex": (
        r"\b(rupee|pkr|dollar[ ]rate|exchange[ ]rate|interbank"
        r"|open[ ]market[ ]rate|currency|devaluation|revaluation"
        r"|kerb[ ]market|hawala|hundi|dollar[ ]shortage|remittance)\b"
    ),
    "monetary": (
        r"\b(kibor|t[ ]bill|interest[ ]rate|policy[ ]rate|sbp[ ]rate|yield"
        r"|treasury|discount[ ]rate|monetary[ ]policy|pib|repo"
        r"|rate[ ]cut|rate[ ]hike|rate[ ]unchanged|bond[ ]auction|liquidity)\b"
    ),
    "fiscal": (
        r"\b(budget|fiscal[ ]deficit|primary[ ]deficit|fbr|tax[ ]revenue"
        r"|tax[ ]collection|revenue[ ]target|revenue[ ]shortfall"
        r"|public[ ]debt|external[ ]debt|domestic[ ]debt|psdp|imf"
        r"|eurobond|sukuk|privatization|privatisation|subsidy"
        r"|circular[ ]debt|adb[ ]loan|world[ ]bank[ ]loan|ecnec|cdwp)\b"
    ),
    "energy": (
        r"\b(petrol[ ]price|diesel[ ]price|petroleum[ ]levy|fuel[ ]price"
        r"|electricity[ ]tariff|power[ ]tariff|tariff[ ]adjustment"
        r"|capacity[ ]payment|loadshedding|gas[ ]curtailment|rlng|lng"
        r"|ogdc|ppl|pso|hubco|kapco|nepra|ppib|sngpl|ssgc)\b"
    ),
    "banking": (
        r"\b(hbl|ubl|mcb|nbp|meezan|bank[ ]alfalah|askari|faysal|js[ ]bank"
        r"|non[ ]performing[ ]loan|npl|capital[ ]adequacy|deposit[ ]growth"
        r"|credit[ ]growth|banking[ ]profit|banking[ ]sector"
        r"|advance[ ]to[ ]deposit)\b"
    ),
    "corporates": (
        r"\b(engro|lucky[ ]cement|fauji|maple[ ]leaf|mlcf|dgkc|bestway"
        r"|annual[ ]result|quarterly[ ]result|eps|earnings[ ]per[ ]share"
        r"|dividend[ ]declared|dividend[ ]announced|profit[ ]after[ ]tax"
        r"|profit[ ]before[ ]tax|topline|bottomline|revenue[ ]growth)\b"
    ),
    "macro": (
        r"\b(gdp|cpi|inflation|current[ ]account|trade[ ]deficit"
        r"|trade[ ]surplus|balance[ ]of[ ]payment|bop|foreign[ ]reserve"
        r"|forex[ ]reserve|remittance|large[ ]scale[ ]manufacturing|lsm"
        r"|economic[ ]growth|recession|unemployment|exports|imports"
        r"|trade[ ]balance)\b"
    ),
    "market_political": (
        r"\b(imf[ ]condition|imf[ ]tranche|imf[ ]review|imf[ ]program"
        r"|imf[ ]board|imf[ ]approval|imf[ ]disbursement|imf[ ]bailout"
        r"|imf[ ]mission|fatf[ ]grey|fatf[ ]black|fatf[ ]plenary"
        r"|credit[ ]rating|rating[ ]downgrade|rating[ ]upgrade"
        r"|moody|fitch|cpec[ ]investment|saudi[ ]deposit|uae[ ]deposit"
        r"|bilateral[ ]swap|privatization[ ]commission|martial[ ]law)\b"
    ),
}

CATEGORIES = [
    ("economy",          "https://mettisglobal.news/Home/GetEconomylatestnews",          "Economy"),
    ("equity",           "https://mettisglobal.news/Home/GetEquitylatestnews",           "Equity"),
    ("forex",            "https://mettisglobal.news/Home/GetForexlatestnews",            "Forex"),
    ("company_analysis", "https://mettisglobal.news/Home/GetCompanyAnalysislatestnews",  "CompanyAnalysis"),
    ("technical",        "https://mettisglobal.news/Home/GetTechnicalAnalysislatestnews","TechnicalAnalysis"),
    ("mg_opinion",       "https://mettisglobal.news/Home/GetMGOpinionlatestnews",        "MGOpinion"),
    ("global_business",  "https://mettisglobal.news/Home/GetGlobalBusinesslatestnews",   "GlobalBusiness"),
    ("native",           "https://mettisglobal.news/Home/GetNativenewsSidebar",          "Native"),
    ("press_release",    "https://mettisglobal.news/Home/GetPressReleaseNewsSidebar",    "PressRelease"),
    ("analyst_briefing", "https://mettisglobal.news/Home/GetAnalystBriefingSessionlatestnews", "AnalystBriefing"),
    ("stock_picks",      "https://mettisglobal.news/Home/GetStockPicks",                 "Equity"),
]

LOAD_MORE_URL = "https://mettisglobal.news/Home/LoadMore"
FIELDNAMES    = ["id", "date", "category", "subcategory", "title", "description", "url"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def should_keep(title: str, description: str = "", api_category: str = "") -> bool:
    text = "{} {}".format(title, description).strip()
    if not title or len(title) < 15:
        return False
    if re.search(FOREIGN_SIGNALS, text, re.IGNORECASE):
        return False
    if re.search(EXCLUDE_KEYWORDS, text, re.IGNORECASE):
        return False
    if api_category == "global_business":
        return False
    for pattern in CATEGORY_PATTERNS.values():
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def get_subcategories(text: str) -> str:
    cats = [c for c, p in CATEGORY_PATTERNS.items()
            if re.search(p, text, re.IGNORECASE)]
    return "|".join(cats) if cats else "general_market"


def extract_id(item):
    return (item.get("NewsID") or item.get("newsID") or
            item.get("NewsId") or item.get("newsId"))


def parse_articles(data):
    articles = []
    if not isinstance(data, list):
        return articles
    for item in data:
        try:
            news_id  = extract_id(item)
            link     = item.get("Link")         or item.get("link")         or ""
            cat_name = item.get("CategoryName") or item.get("categoryName") or ""
            dt       = (item.get("ModifyDateTime") or item.get("modifyDateTime") or
                        item.get("PublishedTime")  or item.get("publishedTime")  or
                        item.get("CreatedDate")    or "")
            if dt: dt = dt[:10]
            heading = description = ""
            h = item.get("Headings") or item.get("headings") or {}
            if isinstance(h, dict):
                hl = h.get("Heading") or h.get("heading") or []
                if hl: heading = hl[0] if isinstance(hl, list) else hl
            d = item.get("Descriptions") or item.get("descriptions") or {}
            if isinstance(d, dict):
                dl = d.get("Description") or d.get("description") or []
                if dl: description = dl[0] if isinstance(dl, list) else dl
            if news_id and heading:
                articles.append((news_id, dt, heading, description, link, cat_name))
        except Exception:
            pass
    return articles


# ── Processing ────────────────────────────────────────────────────────────────

def process_raw(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Reads raw CSV, deduplicates on id then title, applies should_keep filter,
    refreshes subcategory column, sorts by date, saves to processed_path.
    Returns the processed DataFrame.
    """
    log.info("Processing raw file: %s", raw_path)
    df = pd.read_csv(raw_path, dtype={"id": str})
    raw_count = len(df)

    # Deduplicate on id
    before = len(df)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    log.info("ID dedup: %d -> %d rows (removed %d)", before, len(df), before - len(df))

    # Deduplicate on title
    before = len(df)
    df["_title_key"] = df["title"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["_title_key"]).drop(columns=["_title_key"]).reset_index(drop=True)
    log.info("Title dedup: %d -> %d rows (removed %d)", before, len(df), before - len(df))

    # Re-apply filter
    mask = df.apply(
        lambda r: should_keep(
            str(r.get("title", "")),
            str(r.get("description", "")),
            str(r.get("category", ""))
        ), axis=1
    )
    df = df[mask].copy()
    log.info("Filter: %d -> %d rows (removed %d)", raw_count, len(df), raw_count - len(df))

    # Refresh subcategory column
    df["subcategory"] = df.apply(
        lambda r: get_subcategories("{} {}".format(r.get("title", ""), r.get("description", ""))),
        axis=1
    )

    # Sort by date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    log.info("Saved processed file -> %s  (%d rows)", processed_path, len(df))
    return df


# ── Scraper internals ─────────────────────────────────────────────────────────

async def fetch_json(client, url, params=None):
    for attempt in range(3):
        try:
            r = await client.get(
                url,
                params=params,
                headers={
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept": "application/json, text/javascript, */*",
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": "https://mettisglobal.news/",
                },
                timeout=20.0,
            )
            if r.status_code == 200:
                return json.loads(r.text)
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2)
    return None


async def fetch_load_more(client, cat_param, cursor):
    params_list = [
        {"category": cat_param, "lastNewsId": cursor},
        {"category": cat_param, "newsId":     cursor},
        {"category": cat_param, "NewsID":     cursor},
        {"CategoryName": cat_param, "lastNewsId": cursor},
        {"lastNewsId": cursor},
    ]
    for params in params_list:
        data = await fetch_json(client, LOAD_MORE_URL, params=params)
        if not data or not isinstance(data, list) or len(data) == 0:
            continue
        returned_rowids = [a.get("rowid") for a in data if isinstance(a, dict) and a.get("rowid")]
        if returned_rowids and set(returned_rowids) != {cursor}:
            return data, params
    return None, None


def update_params(params, new_cursor):
    updated = dict(params)
    for k in ["lastNewsId", "newsId", "NewsID"]:
        if k in updated:
            updated[k] = new_cursor
    return updated


async def scrape_category(client, cat_key, seed_url, cat_param,
                           seen_ids, results, writer, csv_f, pbar):
    new_rows = 0

    data = await fetch_json(client, seed_url)
    if not data:
        log.warning("[%s] seed failed", cat_key)
        return 0

    articles = parse_articles(data)
    if not articles:
        log.warning("[%s] no articles in seed", cat_key)
        return 0

    total_rowid    = data[0].get("rowid", 0)
    cursor         = min(item.get("rowid") or 999999 for item in data if isinstance(item, dict))
    working_params = None
    log.info("[%s] total=%s | starting cursor=%s", cat_key, total_rowid, cursor)

    def write_batch(batch):
        nonlocal new_rows
        for news_id, dt, heading, description, link, cat_name in batch:
            if news_id in seen_ids:
                continue
            seen_ids.add(news_id)
            if not should_keep(heading, description, cat_key):
                continue
            url      = "https://mettisglobal.news/{}".format(link) if link else "https://mettisglobal.news/-{}".format(news_id)
            combined = "{} {}".format(heading, description)
            row = {
                "id":          news_id,
                "date":        dt,
                "category":    cat_key,
                "subcategory": get_subcategories(combined),
                "title":       heading,
                "description": description[:300],
                "url":         url,
            }
            results.append(row)
            writer.writerow(row)
            new_rows += 1
        csv_f.flush()

    write_batch(articles)

    consecutive_empty = 0
    while cursor > 1:
        if working_params:
            load_params = update_params(working_params, cursor)
            load_data   = await fetch_json(client, LOAD_MORE_URL, params=load_params)
            if not load_data or not isinstance(load_data, list) or len(load_data) == 0:
                load_data      = None
                working_params = None
        else:
            load_data, working_params = await fetch_load_more(client, cat_param, cursor)

        if not load_data:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                log.info("[%s] LoadMore exhausted at cursor=%s", cat_key, cursor)
                break
            await asyncio.sleep(2)
            continue

        consecutive_empty = 0
        new_articles      = parse_articles(load_data)
        if not new_articles:
            break

        write_batch(new_articles)

        new_cursor = min(a.get("rowid") or 999999 for a in load_data if isinstance(a, dict))
        if new_cursor >= cursor:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                log.info("[%s] no cursor progress, stopping at cursor=%s", cat_key, cursor)
                break
            await asyncio.sleep(1)
            continue

        cursor = new_cursor
        pbar.set_postfix_str("{} cursor={} rows={:,}".format(cat_key, cursor, new_rows))
        await asyncio.sleep(0.3)

    log.info("[%s] done | +%d rows | total=%d", cat_key, new_rows, len(results))
    return new_rows


async def _scrape(raw_path):
    results  = []
    seen_ids = set()

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    csv_f  = open(raw_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    if os.path.getsize(raw_path) == 0:
        writer.writeheader()

    async with httpx.AsyncClient(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    ) as client:
        pbar = tqdm(total=len(CATEGORIES), desc="Mettis", unit="cat")
        for cat_key, seed_url, cat_param in CATEGORIES:
            await scrape_category(client, cat_key, seed_url, cat_param,
                                   seen_ids, results, writer, csv_f, pbar)
            pbar.update(1)
            await asyncio.sleep(1)
        pbar.close()

    csv_f.close()
    log.info("Scraping done — %d rows -> %s", len(results), raw_path)


# ── Public API ────────────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    news_dir = _resolve(cfg["data"]["raw_news_dir"])
    raw_path = os.path.join(news_dir, "mettis_pakistan_raw.csv")

    processed_dir  = os.path.join(PROJECT_ROOT, "data", "processed", "news")
    processed_path = os.path.join(processed_dir, "mettis_pakistan_processed.csv")

    os.makedirs(news_dir, exist_ok=True)

    # ── If raw file already exists, skip scraping entirely ────────────────────
    if os.path.exists(raw_path):
        log.info("Raw file already exists at %s — skipping scrape.", raw_path)
        return process_raw(raw_path, processed_path)

    # ── Scrape from scratch ───────────────────────────────────────────────────
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(_scrape(raw_path))

    return process_raw(raw_path, processed_path)


if __name__ == "__main__":
    run()
