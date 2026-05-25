"""
================================================================================
 File   : src/data_collection/news_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Merges processed news CSVs from Dawn, BRecorder, and Mettis.
          Outputs a clean news_processed.csv with columns:
            date | source | category | title
          Categories mapped to 5 standard values:
            macro | corporate | energy | forex | banking
          Rows sorted by date. Duplicates removed. Irrelevant articles filtered.

 Input:
   data/processed/news/dawn_pakistan_processed.csv
   data/processed/news/brecorder_pakistan_processed.csv
   data/processed/news/mettis_pakistan_processed.csv

 Output:
   data/processed/news/news_processed.csv

 Cache logic:
   If output already exists -> skip and return cached result
================================================================================
"""

import re, logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "news"

DAWN_PATH      = PROCESSED_DIR / "dawn_pakistan_processed.csv"
BRECORDER_PATH = PROCESSED_DIR / "brecorder_pakistan_processed.csv"
METTIS_PATH    = PROCESSED_DIR / "mettis_pakistan_processed.csv"
OUTPUT_PATH    = PROCESSED_DIR / "news_processed.csv"


# ── Category Mapping ──────────────────────────────────────────────────────────

CATEGORY_MAP = {
    # macro
    "macro"            : "macro",
    "fiscal"           : "macro",
    "monetary"         : "macro",
    "market_political" : "macro",
    "general_market"   : "macro",
    "general"          : "macro",
    "rates"            : "macro",
    "economy"          : "macro",
    # corporate
    "corporates"       : "corporate",
    "equities"         : "corporate",
    "stocks"           : "corporate",
    "corporate"        : "corporate",
    "equity"           : "corporate",
    "company_analysis" : "corporate",
    "technical"        : "corporate",
    "analyst_briefing" : "corporate",
    "stock_picks"      : "corporate",
    "press_release"    : "corporate",
    "native"           : "corporate",
    "mg_opinion"       : "corporate",
    # energy
    "energy"           : "energy",
    "commodities"      : "energy",
    # forex
    "forex"            : "forex",
    "exchange"         : "forex",
    # banking
    "banking"          : "banking",
}


# ── Relevance Filter ──────────────────────────────────────────────────────────

PK_INSTITUTIONS = [
    "pmex", "kse", "psx", "sbp", "secp", "nepra", "ogra", "fbr", "wapda",
    "ssgc", "sngpl", "ogdc", "ppl", "pso", "engro", "hbl", "mcb", "ubl",
    "fauji", "kapco", "hubc", "ptcl", "pta", "pia", "ndma", "eobi",
    "state bank of pakistan", "federal board of revenue",
    "national electric", "cpec", "circular debt", "pkr",
    "karachi", "lahore", "islamabad", "rawalpindi", "peshawar", "quetta",
    "sindh", "punjab", "balochistan", "khyber", "gilgit",
    "rs.", "rupees", "billion rupee", "million rupee",
    "national assembly", " na told", "senate pakistan",
    "pm imran", "pm shehbaz", "pm nawaz", "finance minister pakistan",
]

PK_GENERIC = ["pakistan", "pakistani"]

FOREIGN_SIGNALS = [
    "indian rupee", "indian stock", "indian market", "indian economy",
    "indian growth", "indian sugar", "indian shares", "indian firm",
    "india rupee", "india's economic", "india inflation",
    "india's growth", "india stocks", "india shares",
    "india rupee ends", "india's rupee", "india shares dip",
    "india shares fall", "india shares rise",
    "rupee treads water", "rupee slips to record low", "rupee's rough patch",
    "india hikes gold", "india raises gold",
    "modi's call for austerity", "india's economic growth",
    "india kicks off privatisation", "india will stick to fiscal",
    "india fiscal deficit target",
    "reserve bank of india", "rbi seen cutting",
    "indian central bank", "india central bank",
    "bse sensex", "bombay stock", "nifty",
    "energy supply worries to keep indian rupee",
    "indian rupee losing run seen extending",
    "india gold prices", "indian gold prices",
    r"^asia rice", "asia rice.*india", "asia rice.*rupee", r"asia rice[:\-]",
    r"^qe3 pumps",
    "asian shares", "asia stocks", "asian stocks", "asian markets",
    "nikkei", "shanghai composite", "hang seng",
    "wall street", "dow jones", "s&p 500", "nasdaq",
    "ftse", "dax", "european stocks", "european shares", "global stocks",
    "world stock markets", "global equity markets", "world shares",
    "asia shares", "asian equities", "emerging market",
    r"^world stocks", r"^us stocks", r"^wall st",
    r"world stocks.*plunge", r"world stocks.*rise", r"world stocks.*fall",
    r"world stocks.*higher", r"world stocks.*rally", r"world stocks.*oil",
    r"world stocks.*sink", r"world stocks.*mixed", r"world stocks.*struggle",
    r"world stocks.*edge", r"world stocks.*shrug", r"world stocks.*advance",
    r"world stocks.*drop", r"world stocks.*spooked", r"world stocks.*slide",
    r"world stocks.*power", r"world stocks.*soar",
    r"us stocks.*tumble", r"us stocks.*rocket", r"us stocks.*push",
    r"us stocks.*slide", r"us stocks.*mixed", r"us stocks.*end lower",
    r"us stocks.*rise", r"us stocks.*fall", r"us stocks.*close",
    r"us stocks.*plunge", r"us stocks.*dive", r"us stocks.*rally",
    r"us stocks.*drop", r"us stocks.*soar", r"us stocks.*skid",
    r"us stocks.*down", r"us stocks.*higher",
    r"fall in us stocks", r"soft landing.*us stocks",
    r"wall st.*rally", r"wall st.*edges", r"wall st.*higher", r"wall st.*lower",
    r"^global stock markets",
    r"^world economy",
    "us federal reserve", "federal reserve slashes", "federal reserve hikes",
    "federal reserve cuts", "european central bank", "ecb",
    "bank of japan", "bank of england", "bank of canada",
    "us economy", "us unemployment", "us-china trade",
    "turkiye.*rate hike", "turkey.*rate hike",
    "argentina", "erdogan",
    "hong kong", "virginia", "bangkok", "tokyo",
    "brazil", "venezuela", "myanmar", "sri lanka",
    "north korea", "turkish lira", "turkey's lira",
    "alibaba", "snapchat",
    "bitcoin falls", "bitcoin hits record high",
]

EXPLICIT_KEEP = [
    "privatisation of psl",
    "nbp to form women cricket",
    "psx in share sale talks with qatar, istanbul",
    "avanceon signs mou with pe energy to expand its footprint in nigeria",
    "stampede at pti",
    "police blame pti for deaths in multan",
    "ge and cmec mark important milestone",
    "lucky cement starts production in iraq",
    "indus motor signs export agreement with egypt",
    "indus motor company eyes import of used vehicles",
    "indus motor company begins export of vehicles",
    "pak-russia negotiating", "polish company to drill",
    "russia-ukraine crisis to hurt pakistan",
    "russian firm geared up to start feasibility",
    "pak inks agreement with azerbaijan",
    "pll negotiating lng deal with azerbaijan",
    "hbl pakistan super league",
    "pakistan super league.*season", "psl.*season",
    "world bank.*pakistan", "world bank cuts pakistan",
    "world bank trims pakistan", "world bank's latest forecast.*pakistan",
    "imf.*warns.*pakistan", "imf warns pakistan",
]

NON_ECONOMIC = [
    "blind cricket", "cricket championship", "cricket tournament",
    "cricket gold medal", "cricket gold",
    "football match", "hockey tournament",
    "world cup squad", "test match", "odi series",
    "ppfl", "premier football league", "pakistan premier football league",
    "ppl.*football", "ppl balochistan football", "balochistan football cup",
    "national hockey", "national hockey championship", "national hockey opener",
    "hockey players", "hockey gold", "hockey silver", "hockey team returns",
    "women hockey team", "wapda.*hockey", "nbp.*hockey", "ssgc.*hockey",
    "pia.*hockey", "kpt.*hold.*hbl", "hbl.*hold.*kpt",
    "nbp crowned national hockey", "nbp edge ssgc.*hockey",
    "nbp overcome ztbl", "navy stun wapda",
    "wapda.*football", "football.*wapda",
    "wapda.*cricket", "ppl cricket", "nbp cricket", "ssgc cricket",
    r"hbl almost through.*pia struggle",
    r"(hbl|pia).*(almost through|struggle).*(final|cup|semi)",
    r"(hbl|nbp|pia|ssgc|kesc|wapda|paf|krl|ztbl|kpt|pel).*\d+-\d+",
    r"\d+-\d+.*(hbl|nbp|pia|ssgc|kesc|wapda|paf|krl|ztbl|kpt|pel)",
    "gold medal", "gold medalist", "silver medal", "bronze medal",
    "wins gold", "win gold", "won gold", "grabs gold",
    "asian games.*gold", "sag.*gold", "asian beach games",
    "karate championship", "wushu trophy",
    "olympic gold medalist arshad nadeem",
    "pakistan wins gold", "pakistan win.*gold", "pakistan grabs.*gold",
    "vintage car rally", "vintage.*rally", "classic car rally",
    "car rall", "peace car rally", "car rally.*waziristan",
    "karachi chronicle",
    "drama serial", "film release", "box office",
    "recipe", "fashion week",
    "zayn malik shares a throwback",
    "taylor swift.*music is held hostage",
    r"imran ashraf shares.*film",
    "thousands defy.*rally.*qadri",
    r"pti cancels rally",
    "political rally.*killed", "killed.*political rally",
    "firing.*political rally", "political rally.*firing",
    "blast rocks zimbabwean president.*rally",
    "thousands rally in kenya against president",
    "blast at hekmatyar.*rally",
    "14 killed in bomb attack on afghan election rally",
    "suicide bomber kills 13 in election rally in afghanistan",
    "200,000 rohingya rally",
    "stampede at nigerian president.*rally kills",
    "trump brands biden.*enemy of the state.*pennsylvania rally",
    "trump mocks democrats.*insults pelosi.*campaign rally",
    "harris to rally where trump riled capitol",
    "dakar rally.*goncalves dies", "dakar rally 2012.*top shots",
    "motorcycle champion sanders.*dakar rally",
]

GOLD_TICKER_PATTERN = re.compile(
    r'gold price[s]?\s+(per tola|gains|sheds|drops|falls|rises|jumps|'
    r'increases|decreases|declines|dips|soars|remains|surges|plunges|'
    r'climbs|edges|goes|went|up by|up rs|down by|gain|shed|drop|fall|rise|'
    r'jump|stable|unchanged|flat|steady|recorded|traded|close|decreased|'
    r'increased|decrease|increase|dip|slumps|nosedives|shoots|continues|'
    r'in domestic|makes history|remain|per 12)',
    re.IGNORECASE
)

GOLD_TICKER_PATTERN2 = re.compile(
    r'(gold prices (per tola|increase rs|decline rs|increase by|decline by|'
    r'remain (stable|unchanged|largely stable|steady)|fall by|fall for|'
    r'reverse losing|in pakistan (hit|are|continue|near|remain|reach|soar)|'
    r'soar to a new|finally exhibit|reach|come down|surge by|slip|'
    r'plummet|clamber|climb|hold|edge|stable|steady|dull|decrease|'
    r'jump by|gains rs|rise from|per 12 gram|per tola gains|per tola declines))',
    re.IGNORECASE
)

GOLD_KEEP_PATTERN = re.compile(
    r'(why are gold prices|charities report.*gold prices|'
    r'stocks.*oil.*gold prices jump|stocks fall.*oil.*gold prices jump|'
    r'gold prices.*surpass rs\d|gold prices.*crosses rs\d|'
    r'gold prices.*record high in pakistan.*surpass|'
    r'gold prices in pakistan.*record rs\d|'
    r'gold prices in pakistan soar to another record|'
    r'gold prices (soar to record|hit record high in pakistan|'
    r'reach record|near all-time high|hit fresh all-time|'
    r'in pakistan reach record|in pakistan hit record))',
    re.IGNORECASE
)


def is_relevant(title: str) -> bool:
    t = str(title).lower()
    if any(re.search(sig, t) for sig in EXPLICIT_KEEP):
        return True
    if any(re.search(sig, t) for sig in NON_ECONOMIC):
        return False
    if GOLD_TICKER_PATTERN.search(t) or GOLD_TICKER_PATTERN2.search(t):
        if not GOLD_KEEP_PATTERN.search(t):
            return False
    if any(sig in t for sig in PK_INSTITUTIONS):
        return True
    if re.search(r'rs\d', t):
        return True
    if any(sig in t for sig in PK_GENERIC):
        return True
    if any(re.search(sig, t) for sig in FOREIGN_SIGNALS):
        return False
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    title = str(title).lower().strip()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def standardize_category(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map raw category values to the 5 standard categories."""
    if source == "mettis" and "subcategory" in df.columns:
        cat_col = df["subcategory"].fillna(df["category"])
    else:
        cat_col = df["category"]

    # Take first label if pipe-separated (e.g. "equities|macro" -> "equities")
    cat_col = cat_col.str.strip().str.lower().str.split("|").str[0]
    df      = df.copy()
    df["category"] = cat_col.map(CATEGORY_MAP)
    before = len(df)
    df = df.dropna(subset=["category"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.info("  [%s] dropped %d rows with unmapped category", source, dropped)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates by normalized title within the same date."""
    SOURCE_PRIORITY = {"brecorder": 0, "dawn": 1, "mettis": 2}
    before = len(df)
    df = df.copy()
    df["_norm"]     = df["title"].fillna("").apply(_normalize_title)
    df["_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(99)
    df.sort_values(["date", "_norm", "_priority"], inplace=True)
    df = df.drop_duplicates(subset=["date", "_norm"], keep="first")
    df = df.drop_duplicates(subset=["_norm"], keep="first")
    df = df.drop(columns=["_norm", "_priority"]).reset_index(drop=True)
    log.info("Dedup: %d -> %d rows (removed %d)", before, len(df), before - len(df))
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def run():
    # ── Cache check ───────────────────────────────────────────────────────────
    if OUTPUT_PATH.exists():
        log.info("Output already exists at %s — returning cached result.", OUTPUT_PATH)
        return pd.read_csv(OUTPUT_PATH)

    # ── Load sources ──────────────────────────────────────────────────────────
    source_files = {
        "dawn"      : DAWN_PATH,
        "brecorder" : BRECORDER_PATH,
        "mettis"    : METTIS_PATH,
    }
    dfs = []
    for source, path in source_files.items():
        if not path.exists():
            log.warning("Missing: %s — skipping.", path)
            continue
        raw = pd.read_csv(path)
        raw["source"] = source
        raw = standardize_category(raw, source)
        raw = raw[["date", "source", "category", "title"]].copy()
        log.info("Loaded %6d rows  <- %s", len(raw), source)
        dfs.append(raw)

    if not dfs:
        raise FileNotFoundError(
            "No processed news CSVs found. Run dawn_scraper, "
            "brecorder_scraper, and mettis_scraper first."
        )

    # ── Combine and sort ──────────────────────────────────────────────────────
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", "title"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("Combined: %d rows across %d sources", len(df), len(dfs))

    # ── Deduplicate ───────────────────────────────────────────────────────────
    df = deduplicate(df)

    # ── Relevance filter ──────────────────────────────────────────────────────
    before = len(df)
    mask   = df["title"].apply(is_relevant)
    df     = df[mask].reset_index(drop=True)
    log.info("Filter: %d -> %d rows (removed %d)", before, len(df), before - len(df))

    # ── Format date and save ──────────────────────────────────────────────────
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[["date", "source", "category", "title"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved -> %s  (%d rows)", OUTPUT_PATH, len(df))
    return df


if __name__ == "__main__":
    run()
