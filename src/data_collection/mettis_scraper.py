import asyncio
import httpx
import re
import csv
import os
import json
import random
from tqdm import tqdm

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

RAW_CSV       = "data/raw/news/mettis_pakistan_raw.csv"
PROGRESS_FILE = "data/mettis_progress.txt"
FINAL_CSV     = "data/processed/mettis_news_processed.csv"
FIELDNAMES    = ["id", "date", "category", "subcategory", "title", "description", "url"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
]


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


def load_done() -> set:
    done = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            for line in f:
                s = line.strip()
                if s:
                    done.add(s)
    return done


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
        print("  [{}] seed failed".format(cat_key))
        return 0

    articles = parse_articles(data)
    if not articles:
        print("  [{}] no articles in seed".format(cat_key))
        return 0

    total_rowid    = data[0].get("rowid", 0)
    cursor         = min(item.get("rowid") or 999999 for item in data if isinstance(item, dict))
    working_params = None
    print("  [{}] total={} | starting cursor={}".format(cat_key, total_rowid, cursor))

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
                print("  [{}] LoadMore exhausted at cursor={}".format(cat_key, cursor))
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
                print("  [{}] no cursor progress, stopping at cursor={}".format(cat_key, cursor))
                break
            await asyncio.sleep(1)
            continue

        cursor = new_cursor
        pbar.set_postfix_str("{} cursor={} rows={:,}".format(cat_key, cursor, new_rows))
        await asyncio.sleep(0.3)

    print("  [{}] done | +{} rows | total={:,}".format(cat_key, new_rows, len(results)))
    return new_rows


async def scrape():
    results  = []
    seen_ids = set()

    done = load_done()
    if os.path.exists(RAW_CSV):
        with open(RAW_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append(row)
                try: seen_ids.add(int(row["id"]))
                except: pass
        print("Resuming — {:,} rows already saved".format(len(results)))

    pending = [(k, s, p) for k, s, p in CATEGORIES if k not in done]
    print("Categories pending: {} / {}".format(len(pending), len(CATEGORIES)))
    if not pending:
        print("Scraping already complete.")
        return

    os.makedirs(os.path.dirname(RAW_CSV),       exist_ok=True)
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

    csv_f  = open(RAW_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    if os.path.getsize(RAW_CSV) == 0:
        writer.writeheader()
    progress_f = open(PROGRESS_FILE, "a", encoding="utf-8")

    async with httpx.AsyncClient(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    ) as client:
        pbar = tqdm(total=len(pending), desc="Scraping Mettis", unit="cat")
        for cat_key, seed_url, cat_param in pending:
            await scrape_category(client, cat_key, seed_url, cat_param,
                                   seen_ids, results, writer, csv_f, pbar)
            progress_f.write(cat_key + "\n")
            progress_f.flush()
            pbar.update(1)
            await asyncio.sleep(1)
        pbar.close()

    csv_f.close()
    progress_f.close()
    print("\nScraping done! {:,} rows -> {}".format(len(results), RAW_CSV))


def process():
    print("\n-- Processing {} --".format(RAW_CSV))
    with open(RAW_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("Loaded:            {:,} rows".format(len(rows)))

    seen_ids = set()
    deduped  = []
    for row in rows:
        aid = str(row.get("id", "")).strip()
        if aid and aid not in seen_ids:
            seen_ids.add(aid)
            deduped.append(row)
    print("After ID dedup:    {:,} rows  (removed {:,})".format(len(deduped), len(rows) - len(deduped)))

    seen_titles = set()
    title_deduped = []
    for row in deduped:
        key = row.get("title", "").strip().lower()
        if key not in seen_titles:
            seen_titles.add(key)
            title_deduped.append(row)
    print("After title dedup: {:,} rows  (removed {:,})".format(len(title_deduped), len(deduped) - len(title_deduped)))

    clean, excluded = [], []
    for row in title_deduped:
        title       = row.get("title", "")
        description = row.get("description", "")
        cat_key     = row.get("category", "")
        if should_keep(title, description, cat_key):
            combined           = "{} {}".format(title, description)
            row["subcategory"] = get_subcategories(combined)
            clean.append(row)
        else:
            excluded.append(title)

    print("After filter:      {:,} rows  (removed {:,})".format(len(clean), len(title_deduped) - len(clean)))

    if excluded:
        print("\nSample removed:")
        for t in excluded[:10]:
            print("  x  " + t)

    print("\nSample kept:")
    for r in clean[:10]:
        print("  ok  [{}]  {}".format(r["subcategory"], r["title"]))

    from collections import Counter
    print("\nSubcategory breakdown (primary label):")
    for cat, cnt in Counter(r["subcategory"].split("|")[0] for r in clean).most_common():
        print("  {:<20} {:>6,}".format(cat, cnt))

    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(clean)
    print("\nSaved -> {}  ({:,} articles)".format(FINAL_CSV, len(clean)))


if __name__ == "__main__":
    asyncio.run(scrape())
    process()
