"""
================================================================================
 File   : src/data_collection/dawn_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Scrapes financial news headlines from Dawn for Pakistan market.
          Filters, deduplicates, and saves processed CSV.

 Saves:
   data/raw/news/dawn_pakistan_raw.csv
   data/processed/news/dawn_pakistan_processed.csv

 Cache logic:
   1. If raw CSV exists -> skip scraping entirely, go straight to processing
   2. If raw CSV does not exist -> scrape from scratch
================================================================================
"""

import asyncio, csv, os, random, re, logging
import pandas as pd
import yaml
from datetime import date, timedelta
from playwright.async_api import async_playwright
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
    r"|ecb|bank[ ]of[ ]england|rbi|pboc"
    r"|us[ ]stocks|us[ ]markets|global[ ]markets|world[ ]markets"
    r"|china[ ]stocks|india[ ]stocks|uk[ ]stocks|euro[ ]zone|eurozone"
    r"|yuan|renminbi|yen[ ]falls|yen[ ]rises|euro[ ]falls|euro[ ]rises"
    r"|brent[ ]falls|brent[ ]rises|crude[ ]falls|crude[ ]rises"
    r"|oil[ ]falls|oil[ ]rises|gold[ ]falls|gold[ ]rises"
    r"|dollar[ ]index|dxy|london[ ]stock|new[ ]york[ ]stock|tokyo[ ]stock"
    r"|hong[ ]kong[ ]stock|toronto[ ]stock)\b"
)

EXCLUDE_KEYWORDS = (
    r"\b(bollywood|lollywood|oscar|grammy|filmfare|actor|actress"
    r"|film[ ]review|movie|drama[ ]serial|television[ ]show|reality[ ]show"
    r"|game[ ]show|fashion[ ]week|skin[ ]care|hair[ ]care|make.?up|beauty"
    r"|perfume|recipe|cooking|food[ ]trend|restaurant[ ]review|cafe[ ]review"
    r"|horoscope|zodiac|astrology|numerology|travel[ ]guide|tourism"
    r"|visa[ ]guide|adventure[ ]travel|book[ ]review|novel|poetry|fiction"
    r"|cricket[ ]score|match[ ]report|match[ ]preview|ipl|champions[ ]trophy"
    r"|football[ ]result|hockey[ ]result|tennis[ ]result|golf[ ]result"
    r"|fifa|uefa|nba|nhl|nfl|olympics|commonwealth[ ]games"
    r"|phone[ ]review|laptop[ ]review|gadget[ ]review|gaming|video[ ]game"
    r"|android[ ]update|ios[ ]update|viral[ ]video|tiktok|instagram[ ]reel"
    r"|youtube[ ]star|influencer|meme|social[ ]media[ ]trend"
    r"|weather[ ]forecast|rain[ ]forecast|temperature[ ]record"
    r"|heat[ ]wave[ ]forecast|murder[ ]case|robbery|kidnapping"
    r"|road[ ]accident|drug[ ]haul|drug[ ]bust|dental|hair[ ]loss"
    r"|weight[ ]loss[ ]tip|diet[ ]plan|yoga|meditation)\b"
)

PAKISTAN_ANCHOR = (
    r"\b(pakistan|pakistani"
    r"|karachi|lahore|islamabad|rawalpindi|peshawar|quetta|multan"
    r"|faisalabad|hyderabad|sialkot|sukkur|larkana|mirpur"
    r"|sindh|punjab|balochistan|kpk|khyber"
    r"|psx|kse|nccpl|secp"
    r"|sbp|pkr|rupee|kibor"
    r"|fbr|imf|cpec|sifc"
    r"|ogdc|ppl|pso|sngpl|ssgc|hubco|kapco|nepra|ppib|wapda"
    r"|engro|fauji|mlcf|dgkc|bestway|lucky[ ]cement|maple[ ]leaf"
    r"|mcb|ubl|hbl|nbp|meezan|askari|bank[ ]alfalah"
    r"|indus[ ]motor|pak[ ]suzuki|millat|ptcl|pia"
    r"|nestle[ ]pakistan|unilever[ ]pakistan|abbott[ ]pakistan|searle"
    r"|gul[ ]ahmed|interloop|bata[ ]pakistan)\b"
)

CATEGORY_PATTERNS = {
    "equities": (
        r"\b(kse|psx|stock[ ]market|share[ ]market|stocks|shares|equit"
        r"|listed[ ]compan|scrip|ipo|initial[ ]public[ ]offer"
        r"|bonus[ ]share|right[ ]share|stock[ ]split|buy[ ]back"
        r"|market[ ]cap|index[ ]point|rally|sell[ ]off|bull[ ]run|bear[ ]market"
        r"|trading[ ]volume|circuit[ ]breaker|upper[ ]lock|lower[ ]lock|nccpl)\b"
    ),
    "macro": (
        r"\b(gdp|gross[ ]domestic|cpi|inflation|consumer[ ]price"
        r"|current[ ]account|trade[ ]deficit|trade[ ]surplus|trade[ ]balance"
        r"|balance[ ]of[ ]payment|bop|foreign[ ]reserve|forex[ ]reserve"
        r"|remittance|worker[ ]remittance|large[ ]scale[ ]manufacturing|lsm"
        r"|economic[ ]growth|economic[ ]contraction|recession|unemployment)\b"
    ),
    "monetary": (
        r"\b(sbp|state[ ]bank|monetary[ ]policy|mpc|policy[ ]rate"
        r"|discount[ ]rate|repo[ ]rate|kibor|karachi[ ]interbank"
        r"|t[ ]bill|treasury[ ]bill|pib|pakistan[ ]investment[ ]bond"
        r"|bond[ ]auction|interest[ ]rate|rate[ ]cut|rate[ ]hike"
        r"|rate[ ]unchanged|liquidity[ ]injection)\b"
    ),
    "forex": (
        r"\b(rupee|pkr|dollar[ ]rate|exchange[ ]rate|interbank[ ]rate"
        r"|open[ ]market[ ]rate|currency[ ]depreciati|currency[ ]appreciati"
        r"|devaluation|revaluation|dollar[ ]shortage|kerb[ ]market"
        r"|hawala|hundi)\b"
    ),
    "fiscal": (
        r"\b(federal[ ]budget|mini[ ]budget|supplementary[ ]budget|fbr"
        r"|tax[ ]revenue|tax[ ]collection|revenue[ ]shortfall|revenue[ ]target"
        r"|fiscal[ ]deficit|primary[ ]deficit|primary[ ]surplus"
        r"|public[ ]debt|domestic[ ]debt|external[ ]debt|debt[ ]to[ ]gdp"
        r"|eurobond|sukuk|privatization|privatisation|psdp"
        r"|subsidy[ ]removal|subsidy[ ]cut|circular[ ]debt"
        r"|adb[ ]loan|world[ ]bank[ ]loan|ecnec|cdwp)\b"
    ),
    "energy": (
        r"\b(ogdc|ppl|pso|sui[ ]northern|sui[ ]southern|sngpl|ssgc"
        r"|hubco|kapco|nepra|ppib|petroleum[ ]levy|petrol[ ]price"
        r"|diesel[ ]price|fuel[ ]price|fuel[ ]adjustment"
        r"|electricity[ ]tariff|power[ ]tariff|tariff[ ]adjustment"
        r"|capacity[ ]payment|ipp|loadshedding|rlng|gas[ ]curtailment)\b"
    ),
    "banking": (
        r"\b(hbl|ubl|mcb|nbp|meezan|bank[ ]alfalah|askari[ ]bank"
        r"|standard[ ]chartered[ ]pakistan|non[ ]performing[ ]loan|npl"
        r"|infection[ ]ratio|capital[ ]adequacy|advance[ ]to[ ]deposit|adr"
        r"|deposit[ ]growth|credit[ ]growth|banking[ ]profit|banking[ ]sector)\b"
    ),
    "corporates": (
        r"\b(engro|lucky[ ]cement|dgkc|dg[ ]khan|maple[ ]leaf|mlcf|bestway"
        r"|fauji[ ]cement|fauji[ ]fertilizer|ffbl|nestle[ ]pakistan"
        r"|unilever[ ]pakistan|ptcl|packages[ ]limited|searle"
        r"|glaxo[ ]pakistan|abbott[ ]pakistan|indus[ ]motor|pak[ ]suzuki"
        r"|honda[ ]atlas|millat[ ]tractor|annual[ ]result|quarterly[ ]result"
        r"|eps|earnings[ ]per[ ]share|dividend[ ]announced|dividend[ ]declared"
        r"|profit[ ]after[ ]tax|profit[ ]before[ ]tax|topline|bottomline)\b"
    ),
    "commodities": (
        r"\b(urea[ ]price|dap[ ]price|fertilizer[ ]price|cotton[ ]price"
        r"|cotton[ ]export|wheat[ ]procurement|wheat[ ]support[ ]price"
        r"|sugar[ ]price|sugar[ ]mill|palm[ ]oil[ ]import|edible[ ]oil[ ]pakistan"
        r"|gold[ ]price[ ]pakistan|gold[ ]rate[ ]pakistan|gold[ ]rate[ ]today"
        r"|cement[ ]dispatches|cement[ ]offtake|cement[ ]export)\b"
    ),
    "market_political": (
        r"\b(imf[ ]condition|imf[ ]demand|imf[ ]benchmark|imf[ ]deadline"
        r"|imf[ ]board|imf[ ]approval|imf[ ]disbursement|imf[ ]tranche"
        r"|imf[ ]review|imf[ ]program|imf[ ]bailout|imf[ ]mission|imf[ ]staff"
        r"|fatf[ ]grey|fatf[ ]black|fatf[ ]plenary|fatf[ ]action"
        r"|credit[ ]rating[ ]pakistan|rating[ ]downgrade|rating[ ]upgrade"
        r"|moody|fitch|cpec[ ]investment|cpec[ ]project|cpec[ ]corridor"
        r"|saudi[ ]deposit|uae[ ]deposit|china[ ]swap|bilateral[ ]swap"
        r"|privatization[ ]commission|martial[ ]law)\b"
    ),
}

STICKY_IDS     = {"1984584"}
FIELDNAMES     = ["date", "category", "title", "url"]
ARTICLE_URL_RE = re.compile(r"dawn\.com/news/(\d+)")
TOTAL_WORKERS  = 4
USER_AGENTS    = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def should_keep(title: str) -> bool:
    if not title or len(title) < 15:
        return False
    if re.search(FOREIGN_SIGNALS, title, re.IGNORECASE):
        return False
    if re.search(EXCLUDE_KEYWORDS, title, re.IGNORECASE):
        return False
    if not re.search(PAKISTAN_ANCHOR, title, re.IGNORECASE):
        return False
    for pattern in CATEGORY_PATTERNS.values():
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


def get_categories(title: str) -> str:
    cats = [c for c, p in CATEGORY_PATTERNS.items()
            if re.search(p, title, re.IGNORECASE)]
    return "|".join(cats) if cats else "general_market"


# ── Processing ────────────────────────────────────────────────────────────────

def process_raw(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Reads raw CSV, deduplicates on (url) then (date, title),
    applies should_keep filter, refreshes category column,
    sorts by date, saves to processed_path.
    Returns the processed DataFrame.
    """
    log.info("Processing raw file: %s", raw_path)
    df = pd.read_csv(raw_path)
    raw_count = len(df)

    # Deduplicate on URL
    before = len(df)
    df["url"] = df["url"].str.strip().str.rstrip("/").str.lower()
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    log.info("URL dedup: %d -> %d rows (removed %d)", before, len(df), before - len(df))

    # Deduplicate on (date, title)
    before = len(df)
    df = df.drop_duplicates(subset=["date", "title"]).reset_index(drop=True)
    log.info("Title dedup: %d -> %d rows (removed %d)", before, len(df), before - len(df))

    # Re-apply filter
    mask = df["title"].apply(should_keep)
    df   = df[mask].copy()
    log.info("Filter: %d -> %d rows (removed %d)", raw_count, len(df), raw_count - len(df))

    # Refresh category column
    df["category"] = df["title"].apply(get_categories)

    # Sort by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    log.info("Saved processed file -> %s  (%d rows)", processed_path, len(df))
    return df


# ── Scraper internals ─────────────────────────────────────────────────────────

async def new_page(browser):
    ctx = await browser.new_context(
        user_agent=random.choice(USER_AGENTS),
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
    )
    page = await ctx.new_page()
    await page.route(
        "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,eot}",
        lambda r: r.abort(),
    )
    return ctx, page


async def scrape_date(page, date_str: str, seen_urls: set, seen_lock) -> list:
    rows = []
    url  = f"https://www.dawn.com/latest-news/{date_str}"
    for attempt in range(3):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            for _ in range(8):
                t = (await page.title()).lower()
                if "just a moment" not in t and "attention required" not in t:
                    break
                await page.wait_for_timeout(4_000 + random.randint(0, 2_000))
            else:
                return rows
            try:
                await page.wait_for_function(
                    "() => document.querySelectorAll('a[href]').length > 20",
                    timeout=15_000,
                )
            except Exception:
                pass
            items = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    href: a.href || '',
                    text: (a.innerText || a.textContent || '').trim()
                }))
            """)
            for item in items:
                href  = item.get("href", "")
                title = item.get("text", "").strip()
                m     = ARTICLE_URL_RE.search(href)
                if not m:                          continue
                if m.group(1) in STICKY_IDS:       continue
                if not title or len(title) < 10:   continue
                full_url = href.split("?")[0].rstrip("/")
                async with seen_lock:
                    if full_url in seen_urls:      continue
                    seen_urls.add(full_url)
                if not should_keep(title):         continue
                rows.append({
                    "date":     date_str,
                    "category": get_categories(title),
                    "title":    title,
                    "url":      full_url,
                })
            return rows
        except Exception as e:
            if any(x in str(e) for x in ("TargetClosedError", "Target page", "closed")):
                raise
            if attempt < 2:
                await asyncio.sleep(3 + random.uniform(0, 2))
    return rows


async def worker(wid, queue, browser, seen_urls, seen_lock,
                 results, csv_f, writer, pbar):
    ctx, page = await new_page(browser)
    while True:
        date_str = await queue.get()
        if date_str is None:
            queue.task_done()
            try: await ctx.close()
            except Exception: pass
            return
        try:
            rows = await scrape_date(page, date_str, seen_urls, seen_lock)
        except Exception:
            try: await ctx.close()
            except Exception: pass
            try:
                ctx, page = await new_page(browser)
                rows = await scrape_date(page, date_str, seen_urls, seen_lock)
            except Exception:
                rows = []
        for row in rows:
            results.append(row)
            writer.writerow(row)
        csv_f.flush()
        pbar.update(1)
        pbar.set_postfix_str(f"{len(results):,} articles")
        queue.task_done()


async def _scrape(pending, raw_path, results, seen_urls):
    seen_lock = asyncio.Lock()
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    csv_f  = open(raw_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    if os.path.getsize(raw_path) == 0:
        writer.writeheader()

    pw       = await async_playwright().start()
    browsers = [await pw.firefox.launch(headless=True) for _ in range(TOTAL_WORKERS)]
    log.info("Launched %d browsers", len(browsers))

    try:
        queue = asyncio.Queue(maxsize=TOTAL_WORKERS * 4)
        pbar  = tqdm(total=len(pending), desc="Dawn", unit="date")
        tasks = [
            asyncio.create_task(
                worker(i, queue, browsers[i], seen_urls, seen_lock,
                       results, csv_f, writer, pbar)
            )
            for i in range(TOTAL_WORKERS)
        ]
        for d in pending:
            await queue.put(d)
        for _ in range(TOTAL_WORKERS):
            await queue.put(None)
        await queue.join()
        await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
    finally:
        csv_f.close()
        for b in browsers:
            try: await b.close()
            except Exception: pass
        await pw.stop()
    log.info("Scraping done — %d articles -> %s", len(results), raw_path)


# ── Public API ────────────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    start    = str(cfg["data"]["start_date"])
    end      = str(cfg["data"]["end_date"])
    news_dir = _resolve(cfg["data"]["raw_news_dir"])
    raw_path = os.path.join(news_dir, "dawn_pakistan_raw.csv")

    processed_dir  = os.path.join(PROJECT_ROOT, "data", "processed", "news")
    processed_path = os.path.join(processed_dir, "dawn_pakistan_processed.csv")

    os.makedirs(news_dir, exist_ok=True)

    # ── If raw file already exists, skip scraping entirely ────────────────────
    if os.path.exists(raw_path):
        log.info("Raw file already exists at %s — skipping scrape.", raw_path)
        return process_raw(raw_path, processed_path)

    # ── Scrape from scratch ───────────────────────────────────────────────────
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)
    pending  = [
        (start_dt + timedelta(n)).strftime("%Y-%m-%d")
        for n in range((end_dt - start_dt).days + 1)
    ]
    log.info("Dates to scrape: %d  (%s -> %s)", len(pending), start_dt, end_dt)

    results, seen_urls = [], set()

    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(_scrape(pending, raw_path, results, seen_urls))

    return process_raw(raw_path, processed_path)


if __name__ == "__main__":
    run()
