"""
================================================================================
 File   : src/data_collection/news_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source | sentiment_score | sentiment_label
          Categories are standardized to 4 values:
            macro | corporate | energy | forex
          Rows are sorted by date across all sources
          Only Pakistan-economy-relevant articles are kept (strict gate)
          Duplicates removed across all sources and categories
          Sentiment scored using FinBERT (GPU if available else CPU)
 Outputs:
          data/processed/news_merged.csv                   <- per-article sentiment
          data/processed/news_aggregated_flags.csv         <- flag approach
          data/processed/news_aggregated_decay_catwise.csv <- category-wise decay approach

 Filter Architecture:
   HARD GATE  : title must contain a Pakistan signal (_PAK_SIGNAL)
   BLOCK LIST : known irrelevant keywords + regex patterns dropped first
   CONTEXTUAL : foreign-country mentions allowed only with strong Pak link
   SPORTS/CRIME/DISASTER/CELEBRITY : always dropped
================================================================================
"""

import re
import subprocess
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"
RAW_NEWS  = BASE_DIR / "data" / "raw" / "news"

RAW_NEWS_FILES = {
    "dawn"      : RAW_NEWS / "dawn_pakistan_raw.csv",
    "brecorder" : RAW_NEWS / "brecorder_pakistan_raw.csv",
    "mettis"    : RAW_NEWS / "mettis_pakistan_raw.csv",
}

OUTPUT_PATH = PROCESSED / "news_merged.csv"
FLAGS_PATH  = PROCESSED / "news_aggregated_flags.csv"
DECAY_PATH  = PROCESSED / "news_aggregated_decay_catwise.csv"

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE    = 32

SENTIMENT_COLS = [
    "sentiment_corporate",
    "sentiment_energy",
    "sentiment_forex",
    "sentiment_macro",
]

DECAY_FACTORS = {
    "sentiment_corporate" : 0.7,
    "sentiment_energy"    : 0.7,
    "sentiment_forex"     : 0.8,
    "sentiment_macro"     : 0.85,
}

# ── Category Mapping ──────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "macro"                : "macro",
    "fiscal"               : "macro",
    "monetary"             : "macro",
    "market_political"     : "macro",
    "macro|monetary"       : "macro",
    "macro|fiscal"         : "macro",
    "corporates"           : "corporate",
    "banking"              : "corporate",
    "equities"             : "corporate",
    "energy|banking"       : "corporate",
    "equities|forex"       : "corporate",
    "equities|commodities" : "corporate",
    "energy"               : "energy",
    "commodities"          : "energy",
    "fiscal|energy"        : "energy",
    "forex"                : "forex",
}

# ── Pakistan Hard Gate ────────────────────────────────────────────────────────
# An article MUST match at least one token here to pass through.
# This is the primary filter — if it has no Pakistan signal, it is dropped.
_PAK_SIGNAL = re.compile(
    r"pakistan|pakist|\bsbp\b|\bpkr\b|rupee|\bkse\b|\bpsx\b"
    r"|karachi stock|islamabad|lahore|\bfbr\b|\bnepra\b|\bogra\b"
    r"|\bpia\b|\bcpec\b|remittance|\brda\b"
    r"|\becc\b|\bpso\b|\bogdc\b|\bptcl\b|\bhubco\b"
    r"|\bengro\b|\bfauji\b|lucky cement|meezan bank"
    r"|habib bank|\bmcb\b|\bnbp\b|\bubl\b|\bbahl\b"
    r"|balance of payment|\bbop\b|current account deficit"
    r"|forex reserves|foreign exchange reserve"
    r"|pak economy|privatisation.*airport|airport.*privatis"
    r"|aurangzeb|ishaq dar|miftah ismail|shaukat tarin|hafeez shaikh|reza baqir"
    r"|pakistan.*stock|stock.*pakistan|pakistan.*market|market.*pakistan"
    r"|pakistan.*inflation|pakistan.*gdp|pakistan.*budget|pakistan.*tax"
    r"|pakistan.*trade|pakistan.*export|pakistan.*import"
    r"|pakistan.*interest rate|pakistan.*policy rate"
    r"|pakistan.*debt|pakistan.*loan|pakistan.*imf|pakistan.*adb"
    r"|pakistan.*oil|pakistan.*gas|pakistan.*energy|pakistan.*power"
    r"|pakistan.*bank|pakistan.*finance|pakistan.*fiscal"
    r"|pakistan.*rupee|pakistan.*dollar|pakistan.*forex"
    r"|pakistan.*growth|pakistan.*deficit|pakistan.*surplus"
    r"|pakistan.*revenue|pakistan.*privatis|pakistan.*invest"
    r"|pakistan.*cement|pakistan.*fertilizer|pakistan.*textile"
    r"|pakistan.*refinery|pakistan.*pipeline|pakistan.*circular debt"
    r"|pakistan.*capacity payment|pakistan.*subsidy|pakistan.*tariff"
    r"|pakistan.*cotton|pakistan.*wheat|pakistan.*sugar|pakistan.*rice"
    r"|\bsecp\b|\bpra\b|sukuk|\bt-bill\b|\bpib\b"
    r"|\bkesc\b|\blesco\b|\biesco\b|\bpesco\b|\bqesco\b|\bhesco\b"
    r"|\bppl\b.*gas|gas.*\bppl\b|\bpso\b.*fuel|fuel.*\bpso\b"
    r"|\bpak.*rupee|rupee.*\bpak"
    r"|karachi|lahore|islamabad|peshawar|quetta|multan|faisalabad"
    r"|\bsindh\b|\bpunjab\b|\bkpk\b|\bbalochistan\b",
    re.IGNORECASE,
)

# ── Hard Block Keywords (always dropped, no exceptions) ───────────────────────
_BLOCK_KEYWORDS = [
    # Foreign currencies with no pak link
    "yuan", "renminbi", "ringgit", "baht", "peso", "lira", "rand",
    "ruble", "shekel", "sterling", "pound sterling",
    # Foreign indices
    "sensex", "nifty", "bse ", "nse india", "bombay stock",
    "shanghai composite", "hang seng", "nikkei", "ftse 100",
    "dow jones", "s&p 500", "nasdaq", "wall street rally",
    # Sports scores / tournaments with no financial angle
    "hat-trick", "hat trick", "wicket", "century puts",
    "innings defeat", "thrash", "outplay",
    "ppfl", "krl fc", "pia beat", "nbp beat", "hbl beat", "ztbl",
    "kpt score", "navy thrash", "paf beat", "army thrash",
    "ssgc beat", "kesc crush", "scores hat", "slams hat",
    "cricket championship", "hockey champions", "football cup",
    "blind cricket", "coaching career", "test match scorecard",
    "one-day match", "t20 match",
    # Entertainment / celebrity
    "bollywood", "film festival", "music concert",
    "morning show", "drama serial", "film review",
    "lifetime achievement award", "vintage cars", "heavy bikes",
    "shares first look", "shares heartfelt", "share screen space",
    "father's day", "mother's day", "valentine's day", "eid special",
    # Crime / social
    "booked for student", "pistol and liquor",
    "suspects arrested for", "three girls",
    # Disaster / weather (non-economic)
    "rain disrupts life", "death toll rises",
    "tanker fire kills", "plane crash kills", "train accident kills",
    "building collapse", "explosion kills", "blast kills",
    # Foreign politics with zero pak link
    "brexit", "boris johnson", "theresa may",
    "trump rally", "send her back",
    # Military sports / medals
    "army wins gold", "army wins silver", "army wins bronze",
    "exercise cambrian", "cambrian patrol",
    "pakistan army wins", "pakistan navy wins", "pakistan air force wins",
    "silver medal", "gold medal", "bronze medal",
    "wins trophy", "pride of performance",
    # Irrelevant pak exports
    "cutlery export", "surgical instrument export",
    "sports goods export", "leather goods export",
    # Pure political rallies (no economic content)
    "pdm rally", "pti rally", "pml-n rally", "ppp rally",
    "mqm rally", "sunni conference rally",
    "rally to condemn", "rally against holy",
    "rally in support of yemen", "rally in london",
    "rally in bannu", "rally in bajaur",
    # Doomsday / global environment with no pak link
    "arctic doomsday", "doomsday vault",
    "ppl balochistan football", "ppl.*football cup",
]

# ── Hard Block Regex (always dropped, no exceptions) ──────────────────────────
_BLOCK_REGEX = re.compile(
    r"\bsensex\b|\bnifty\b|\byen\b|\byuan\b"
    r"|\bsensex\b|\bnikkei\b|\bftse\b|\bdow jones\b"
    # Foreign countries (comprehensive)
    r"|\bindia\b|\bindian\b|\bmodi\b|\bnew delhi\b|\brbi\b"
    r"|\bchina\b|\bchinese\b|\bbeijing\b|\bshanghai\b"
    r"|\bjapan\b|\bjapanese\b|\btokyo\b"
    r"|\buk\b|\bbrexit\b|\bsterling\b|\bpound\b"
    r"|\beuro\b|\beuros\b|\becb\b|\beuropean central bank\b"
    r"|\bwall street\b|\bfederal reserve\b"
    r"|\bgreece\b|\bgreek\b|\bathens\b"
    r"|\bbelarus\b|\bminsk\b"
    r"|\brussia\b|\brussian\b|\bmoscow\b"
    r"|\bgermany\b|\bgerman\b|\bberlin\b|\bbundesbank\b"
    r"|\bhungary\b|\bhungarian\b|\bbudapest\b"
    r"|\bpoland\b|\bpolish\b|\bwarsaw\b"
    r"|\bukraine\b|\bukrainian\b|\bkyiv\b"
    r"|\bspain\b|\bspanish\b|\bmadrid\b"
    r"|\bportugal\b|\bportuguese\b|\blisbon\b"
    r"|\bfrench\b|\bfrance\b|\bparis\b|\bbanque de france\b"
    r"|\bitalian\b|\bitaly\b|\brome\b|\bbanca d.italia\b"
    r"|\bnetherlands\b|\bdutch\b|\bamsterdam\b"
    r"|\bbelgium\b|\bbelgian\b|\bbrussels\b"
    r"|\baustria\b|\baustrian\b|\bvienna\b"
    r"|\bsweden\b|\bswedish\b|\bstockholm\b|\briksbankb\b"
    r"|\bnorway\b|\bnorwegian\b|\boslo\b"
    r"|\bdenmark\b|\bdanish\b|\bcopenhagen\b"
    r"|\bfinland\b|\bfinnish\b|\bhelsinki\b"
    r"|\bswitzerland\b|\bswiss\b|\bzurich\b|\bsnb\b"
    r"|\baustralia\b|\baustralian\b|\bsydney\b|\brba\b|\bcanberra\b"
    r"|\bnew zealand\b|\bnz\b|\bauckland\b|\brbnz\b"
    r"|\bcanada\b|\bcanadian\b|\bbank of canada\b|\bottawa\b|\btoonto\b"
    r"|\bbrazil\b|\bbrazilian\b|\bbrasilia\b|\bbcb\b"
    r"|\bargentina\b|\bargentinian\b|\bbuenos aires\b"
    r"|\bmexico\b|\bmexican\b|\bmexico city\b|\bbanxico\b"
    r"|\bchile\b|\bchilean\b|\bsantiago\b"
    r"|\bcolombia\b|\bcolombian\b|\bbogota\b"
    r"|\bsouth africa\b|\bsouth african\b|\bsarb\b|\bjohannesburg\b"
    r"|\bkenya\b|\bkenyan\b|\bnairobi\b"
    r"|\bnigeria\b|\bnigerian\b|\babuja\b|\bcbn\b"
    r"|\begypt\b|\begyptian\b|\bcairo\b|\bcbe\b"
    r"|\bmoroco\b|\bmoroccan\b|\brabat\b"
    r"|\btunis\b|\btunisian\b|\btunis city\b"
    r"|\balgeria\b|\balgerian\b|\balgiers\b"
    r"|\bethiopia\b|\bethiopian\b|\baddis ababa\b"
    r"|\bghana\b|\bghanaian\b|\baccra\b"
    r"|\bsenegal\b|\bsenegalese\b|\bdakar\b"
    r"|\bbangladesh\b|\bbangladeshi\b|\bdhaka\b|\bbb\b"
    r"|\bsri lanka\b|\bsri lankan\b|\bcolombo\b"
    r"|\bmyanmar\b|\bburma\b|\bburmese\b|\bnaypyidaw\b"
    r"|\bvietnam\b|\bvietnamese\b|\bhanoi\b"
    r"|\bthailand\b|\bthai\b|\bbangkok\b|\bbot\b"
    r"|\bmalaysia\b|\bmalaysian\b|\bkuala lumpur\b|\bbnm\b"
    r"|\bindonesia\b|\bindonesian\b|\bjakarta\b|\bbank indonesia\b"
    r"|\bphilippines\b|\bfilipino\b|\bmanila\b|\bbsp\b"
    r"|\bsouth korea\b|\bkorean\b|\bseoul\b|\bbok\b"
    r"|\bnorth korea\b|\bnorth korean\b|\bpyongyang\b"
    r"|\btaiwan\b|\btaiwanese\b|\btaipei\b"
    r"|\bhong kong\b|\bhkma\b"
    r"|\bsingapore\b|\bsingaporean\b|\bmas\b"
    r"|\bkabul\b|\bafghan\b"
    r"|\biraq\b|\biraqi\b|\bbaghdad\b"
    r"|\bisrael\b|\bisraeli\b|\btel aviv\b|\bbank of israel\b"
    r"|\bjordan\b|\bjordanian\b|\bamman\b|\bcbj\b"
    r"|\bkuwait\b|\bkuwaiti\b|\bkuwait city\b"
    r"|\bqatar\b|\bqatari\b|\bdoha\b|\bqcb\b"
    r"|\bbahrain\b|\bbahraini\b|\bmanama\b"
    r"|\boman\b|\bomani\b|\bmuscat\b|\bcbo\b"
    r"|\bluanda\b|\bangola\b|\bangolan\b"
    r"|\bkazakhstan\b|\bkazakh\b|\bnur-sultan\b"
    r"|\bazerbaijan\b|\bazeri\b|\bbaku\b"
    r"|\bturkey\b|\bturkish\b|\bankara\b|\bcbrt\b|\berdogan\b"
    r"|\bbritain\b|\bbritish\b|\blondon\b|\bbank of england\b"
    # Entertainment / celebrity
    r"|\bbollywood\b|\bfilm festival\b|\bkriti sanon\b"
    r"|\billangovan\b|\bactor\b|\bactress\b"
    r"|\bdrama serial\b|\bmorning show\b"
    # Crime
    r"|\bmurder\b|\bkidnap\b|\brake\b"
    r"|\bterrorist attack\b|\bbomb blast\b"
    # Sports
    r"|\bt20\b|\btest match\b|\bone.day match\b"
    r"|\bwins gold\b|\bwins silver\b|\bwins bronze\b"
    r"|\bgold medal\b|\bsilver medal\b|\bbronze medal\b"
    r"|\bcambrian patrol\b|\blifts.*cup\b|\bwins.*trophy\b"
    r"|\bcoaching career\b"
    # Disaster
    r"|\bdeath toll\b|\btanker fire\b|\bplane crash\b"
    r"|\btrain accident\b|\bbuilding collapse\b"
    r"|\bearthquake hits\b|\bflood warning\b"
    # Social / lifestyle
    r"|\bfather.?s day\b|\bmother.?s day\b|\bvalentine.?s day\b"
    r"|\beid special\b",
    re.IGNORECASE,
)

# ── Contextual Allow List ─────────────────────────────────────────────────────
# For sources that slip through _BLOCK_REGEX due to partial block,
# these patterns are allowed ONLY if a Pakistan context also exists.
# Format: (foreign_pattern, required_pak_context_pattern)
_CONTEXTUAL_ALLOWS = [
    # IMF always needs pak context
    (r"\bimf\b",
     r"pakistan|pakist|programme|loan|bailout|staff.level|review|tranche"),
    # World Bank / ADB
    (r"\bworld bank\b|\badb\b|\basian development bank\b",
     r"pakistan|pakist|loan|grant|project"),
    # Saudi / UAE / Gulf (big investors in pak)
    (r"\bsaudi\b|\buae\b|\bdubai\b|\babu dhabi\b|\bgulf\b",
     r"pakistan|pakist|investment|deposit|oil|remittance|export|import|trade"),
    # Oil / crude (global commodity that affects pak)
    (r"\bcrude oil\b|\bbrent\b|\bwti\b|\bopec\b",
     r"pakistan|pakist|pkr|rupee|fuel|pso|ogra|import bill|energy|circular debt"),
    # Gold (affects pak forex reserves)
    (r"\bgold price|\bgold rally|\bgold falls|\bgold rises",
     r"pakistan|pakist|pkr|rupee|reserves|sbp|jewell"),
    # Fed rate / US monetary (affects pak dollar, capital flows)
    (r"\bfederal reserve\b|\bfed rate\b|\bfomc\b|\bfed funds\b",
     r"pakistan|pakist|pkr|rupee|dollar|gold|oil|commodity|emerging market"),
    # China (cpec)
    (r"\bchina\b|\bchinese\b",
     r"pakistan|pakist|cpec|corridor|loan|investment|import|export|trade"),
    # Russia (wheat/energy)
    (r"\brussia\b|\brussian\b",
     r"pakistan|pakist|wheat|pipeline|gas|oil|trade|investment|cpec"),
    # Germany / EU (trade)
    (r"\bgermany\b|\bgerman\b|\beu\b|\beuropean union\b",
     r"pakistan|pakist|trade|investment|loan|export|import"),
]

# ── PSL keep signal (PSL allowed only if financial context) ───────────────────
_PSL_KEEP = re.compile(
    r"kse|psx|stock|share|equity|market|invest|rupee|pkr|sbp"
    r"|pcb|sponsorship|revenue|broadcast|rights",
    re.IGNORECASE,
)

# ── Standardize Category ──────────────────────────────────────────────────────
def standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    if "subcategory" in df.columns:
        df["category"] = df["subcategory"].fillna(df["category"])
    df["category"] = df["category"].str.strip().str.lower().map(CATEGORY_MAP)
    before = len(df)
    df.dropna(subset=["category"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"   🗑️  Dropped {dropped:,} rows with unmapped category")
    return df


# ── Irrelevance Filter ────────────────────────────────────────────────────────
def filter_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    title = df["title"].fillna("").str.lower()
    drop  = pd.Series(False, index=df.index)

    # Step 1: hard block keywords
    for kw in _BLOCK_KEYWORDS:
        drop |= title.str.contains(re.escape(kw), na=False)

    # Step 2: hard block regex
    drop |= title.str.contains(_BLOCK_REGEX, na=False)

    # Step 3: PSL — keep only if financial angle
    is_psl     = title.str.contains(r"\bpsl\b", regex=True, na=False)
    psl_is_fin = title.str.contains(_PSL_KEEP, na=False)
    drop |= (is_psl & ~psl_is_fin)

    # Step 4: contextual allows — re-block if foreign but no pak context
    for foreign_pat, pak_ctx in _CONTEXTUAL_ALLOWS:
        is_foreign  = title.str.contains(foreign_pat, regex=True, na=False)
        has_pak_ctx = title.str.contains(pak_ctx,     regex=True, na=False)
        drop |= (is_foreign & ~has_pak_ctx)

    # Step 5: HARD GATE — must have Pakistan signal
    has_pak = title.str.contains(_PAK_SIGNAL, na=False)
    drop |= ~has_pak

    before  = len(df)
    df      = df[~drop].copy()
    dropped = before - len(df)
    if dropped:
        print(f"   🚫 Dropped {dropped:,} irrelevant articles")
    return df


# ── Deduplication ─────────────────────────────────────────────────────────────
def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    SOURCE_PRIORITY = {"brecorder": 0, "dawn": 1, "mettis": 2}

    df = df.copy()
    df["_norm"] = df["title"].fillna("").apply(_normalize_title)

    # Pass 1: exact normalized title same date
    df["_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(99)
    df.sort_values(["date", "_norm", "_priority"], inplace=True)
    df = df.drop_duplicates(subset=["date", "_norm"], keep="first")

    # Pass 2: near-duplicate via 3-gram overlap same date
    def ngrams(text, n=3):
        words = text.split()
        return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set(words)

    keep_idx = []
    for date, group in df.groupby("date"):
        indices = group.index.tolist()
        norms   = group["_norm"].tolist()
        titles  = group["title"].tolist()
        dropped = set()
        for i in range(len(indices)):
            if indices[i] in dropped:
                continue
            grams_i = ngrams(norms[i])
            for j in range(i + 1, len(indices)):
                if indices[j] in dropped:
                    continue
                grams_j = ngrams(norms[j])
                union   = grams_i | grams_j
                if not union:
                    continue
                overlap = len(grams_i & grams_j) / len(union)
                if overlap >= 0.60:
                    if len(titles[i]) >= len(titles[j]):
                        dropped.add(indices[j])
                    else:
                        dropped.add(indices[i])
                        break
            if indices[i] not in dropped:
                keep_idx.append(indices[i])
    df = df.loc[keep_idx]

    # Pass 3: exact normalized title across all dates keep earliest
    df.sort_values("date", inplace=True)
    df = df.drop_duplicates(subset=["_norm"], keep="first")

    df = df.drop(columns=["_norm", "_priority"])
    df.reset_index(drop=True, inplace=True)
    dropped_total = before - len(df)
    print(f"   🔁 Removed {dropped_total:,} duplicates — {len(df):,} unique articles remain")
    return df


# ── FinBERT Sentiment ─────────────────────────────────────────────────────────
def load_finbert(device):
    print(f"\n🤖 Loading FinBERT on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.to(device)
    model.eval()
    print("   ✅ FinBERT loaded")
    return tokenizer, model


def compute_sentiment(titles, tokenizer, model, device):
    scores = []
    for i in tqdm(range(0, len(titles), BATCH_SIZE), desc="   Scoring"):
        batch   = titles[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = softmax(logits, dim=1).cpu()
        scores.extend((probs[:, 0] - probs[:, 1]).tolist())
    return scores


def score_to_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    return "neutral"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_finbert(device)
    print(f"\n📊 Computing sentiment for {len(df):,} articles ...")
    titles = df["title"].fillna("").tolist()
    scores = compute_sentiment(titles, tokenizer, model, device)
    df["sentiment_score"] = [round(s, 4) for s in scores]
    df["sentiment_label"] = df["sentiment_score"].apply(score_to_label)
    print("   ✅ Sentiment scoring complete")
    print(f"   Score range : {df['sentiment_score'].min():.4f} → {df['sentiment_score'].max():.4f}")
    print(f"   Mean score  : {df['sentiment_score'].mean():.4f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df


# ── GitHub Push ───────────────────────────────────────────────────────────────
def push_to_github(files: list, message: str):
    try:
        project_root = str(BASE_DIR)
        subprocess.run(["git", "-C", project_root, "pull", "--rebase", "origin", "main"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", project_root, "add"] + files,
                       check=True, capture_output=True)
        commit = subprocess.run(
            ["git", "-C", project_root, "commit", "-m", message],
            capture_output=True, text=True
        )
        if "nothing to commit" in commit.stdout or "nothing to commit" in commit.stderr:
            print("   ℹ️  Nothing new to commit")
            return
        if commit.returncode != 0:
            print(f"   ⚠️  Commit failed: {commit.stderr.strip()}")
            return
        subprocess.run(["git", "-C", project_root, "push"],
                       check=True, capture_output=True)
        print(f"   ✅ Pushed: {message}")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  GitHub push failed: {e.stderr.decode()}")


# ── Merge ─────────────────────────────────────────────────────────────────────
def merge_news() -> pd.DataFrame:
    raw_found = {k: v for k, v in RAW_NEWS_FILES.items() if v.exists()}
    if not raw_found:
        raise FileNotFoundError(
            "No raw news CSVs found. Run dawn_scraper, brecorder_scraper, mettis_scraper first."
        )

    print(f"\n📰 Loading {len(raw_found)} raw CSVs ...")
    dfs = []
    for source, path in raw_found.items():
        df = pd.read_csv(path)
        df["source"] = source
        df = standardize_category(df)
        df = filter_irrelevant(df)
        df = df[["date", "category", "title", "source"]]
        dfs.append(df)
        print(f"   ✅ Loaded {len(df):>6,} rows  ← {source}")

    merged = pd.concat(dfs, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    before = len(merged)
    merged.dropna(subset=["date", "title"], inplace=True)
    dropped = before - len(merged)
    if dropped:
        print(f"   🗑️  Dropped {dropped:,} rows with null date/title")

    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    print("\n🔍 Deduplicating across all sources and categories ...")
    merged = deduplicate(merged)

    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged = add_sentiment(merged)
    return merged


# ── Aggregation ───────────────────────────────────────────────────────────────
def _base_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    agg = (
        df.groupby(["date", "category"])["sentiment_score"]
        .mean()
        .unstack(level="category")
        .reset_index()
    )
    agg.columns.name = None
    agg = agg.rename(columns={
        "corporate" : "sentiment_corporate",
        "energy"    : "sentiment_energy",
        "forex"     : "sentiment_forex",
        "macro"     : "sentiment_macro",
    })
    for col in SENTIMENT_COLS:
        if col not in agg.columns:
            agg[col] = float("nan")
    news_count        = df.groupby("date").size().reset_index(name="news_count")
    agg               = agg.merge(news_count, on="date", how="left")
    agg["news_count"] = agg["news_count"].fillna(0).astype(int)
    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


def aggregate_flags(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Building flag aggregation ...")
    agg = _base_aggregate(df)
    flag_map = {
        "sentiment_corporate" : "has_corporate",
        "sentiment_energy"    : "has_energy",
        "sentiment_forex"     : "has_forex",
        "sentiment_macro"     : "has_macro",
    }
    for sent_col, flag_col in flag_map.items():
        agg[flag_col] = agg[sent_col].notna().astype(float)
    agg[SENTIMENT_COLS] = agg[SENTIMENT_COLS].fillna(0.0)
    col_order = [
        "date",
        "sentiment_corporate", "has_corporate",
        "sentiment_energy",    "has_energy",
        "sentiment_forex",     "has_forex",
        "sentiment_macro",     "has_macro",
        "news_count",
    ]
    agg = agg[col_order]
    print(f"   ✅ Flags done — shape: {agg.shape}")
    return agg


def aggregate_decay(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Building decay aggregation ...")
    agg        = _base_aggregate(df)
    full_dates = pd.date_range(start=agg["date"].min(), end=agg["date"].max(), freq="D")
    agg        = agg.set_index("date").reindex(full_dates)
    agg.index.name    = "date"
    agg["news_count"] = agg["news_count"].fillna(0).astype(int)
    for col, factor in DECAY_FACTORS.items():
        values = agg[col].copy()
        for i in range(1, len(values)):
            if pd.isna(values.iloc[i]):
                values.iloc[i] = values.iloc[i - 1] * factor
        agg[col] = values
        print(f"   {col:<25} decay={factor}")
    agg[SENTIMENT_COLS] = agg[SENTIMENT_COLS].fillna(0.0).round(4)
    agg = agg.reset_index()
    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    agg = agg[["date"] + SENTIMENT_COLS + ["news_count"]]
    print(f"   ✅ Decay done — shape: {agg.shape}")
    return agg


# ── Sanity Check ──────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame) -> None:
    print("\n🔍 Sanity Check:")
    checks = {
        "India"      : r"\bindia\b",
        "China"      : r"\bchina\b",
        "UK"         : r"\buk\b",
        "Russia"     : r"\brussia\b",
        "Germany"    : r"\bgermany\b",
        "Egypt"      : r"\begypt\b",
        "Australia"  : r"\baustralia\b",
        "Canada"     : r"\bcanada\b",
        "USA"        : r"\bunited states\b|\busa\b",
        "S.Korea"    : r"\bsouth korea\b",
        "Japan"      : r"\bjapan\b",
        "Indonesia"  : r"\bindonesia\b",
        "Turkey"     : r"\bturkey\b",
        "Britain"    : r"\bbritain\b",
        "Euro"       : r"\beuro\b",
        "Greece"     : r"\bgreece\b",
        "PSL"        : r"\bpsl\b",
    }
    for label, pattern in checks.items():
        if label == "PSL":
            leaked = df[
                df["title"].str.lower().str.contains(pattern, regex=True, na=False) &
                ~df["title"].str.lower().str.contains(_PSL_KEEP, na=False)
            ]
        else:
            leaked = df[df["title"].str.lower().str.contains(pattern, regex=True, na=False)]
        status = f"⚠️  {len(leaked):,} leaked" if len(leaked) else "✅ 0"
        print(f"   {label:<12}: {status}")

    norm_titles = df["title"].fillna("").apply(_normalize_title)
    exact_dups  = norm_titles.duplicated().sum()
    print(f"   Exact dupes : {'⚠️  ' + str(exact_dups) if exact_dups else '✅ 0'}")
    cats = sorted(df["category"].unique().tolist())
    print(f"   Categories  : {cats}")
    print(f"   Total rows  : {len(df):,}")
    print(f"   Date range  : {df['date'].min()} → {df['date'].max()}")

    # Pakistan gate verification — every article must have a pak signal
    no_pak = df[~df["title"].str.contains(_PAK_SIGNAL, na=False)]
    if len(no_pak):
        print(f"   ⚠️  PAK GATE : {len(no_pak):,} articles passed without Pakistan signal!")
        print(no_pak["title"].head(10).to_string())
    else:
        print(f"   PAK GATE   : ✅ all articles have Pakistan signal")


# ── run() ─────────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  PSX News Scraper — Full Pipeline")
    print("=" * 60)

    print("\n📋 Raw CSV check:")
    for source, path in RAW_NEWS_FILES.items():
        print(f"   {source:<12}: {'✅ exists' if path.exists() else '❌ missing'}")

    df = merge_news()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved merged → {OUTPUT_PATH}  ({len(df):,} rows)")
    sanity_check(df)
    push_to_github(
        [str(OUTPUT_PATH.relative_to(BASE_DIR))],
        "Update news_merged.csv — strict pak gate + comprehensive country block"
    )

    print("\n" + "=" * 60)
    flags_df = aggregate_flags(df)
    flags_df.to_csv(FLAGS_PATH, index=False)
    print(f"💾 Saved flags  → {FLAGS_PATH}  ({len(flags_df):,} rows)")
    push_to_github(
        [str(FLAGS_PATH.relative_to(BASE_DIR))],
        "Update news_aggregated_flags.csv"
    )

    print("\n" + "=" * 60)
    decay_df = aggregate_decay(df)
    decay_df.to_csv(DECAY_PATH, index=False)
    print(f"💾 Saved decay  → {DECAY_PATH}  ({len(decay_df):,} rows)")
    push_to_github(
        [str(DECAY_PATH.relative_to(BASE_DIR))],
        "Update news_aggregated_decay_catwise.csv"
    )

    print("\n" + "=" * 60)
    print("  ✅ All done")
    print(f"    {OUTPUT_PATH.name:<40} {len(df):>7,} rows")
    print(f"    {FLAGS_PATH.name:<40} {len(flags_df):>7,} rows")
    print(f"    {DECAY_PATH.name:<40} {len(decay_df):>7,} rows")
    print("=" * 60)
    return df, flags_df, decay_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
