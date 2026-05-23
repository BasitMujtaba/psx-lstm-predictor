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
   HARD GATE        : title must contain a Pakistan signal (_PAK_SIGNAL)
   BLOCK LIST       : known irrelevant keywords + regex patterns dropped first
   POLITICAL GATE   : political headlines pass only if economic context present
   CONTEXTUAL       : foreign-country mentions allowed only with strong Pak link
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
# EXPANDED: added political economy subcategories all mapping to correct targets
CATEGORY_MAP = {
    # Macro
    "macro"                 : "macro",
    "fiscal"                : "macro",
    "monetary"              : "macro",
    "market_political"      : "macro",
    "political"             : "macro",
    "macro|monetary"        : "macro",
    "macro|fiscal"          : "macro",
    "macro|political"       : "macro",
    "imf"                   : "macro",
    "imf|macro"             : "macro",
    "fatf"                  : "macro",
    "fatf|macro"            : "macro",
    "cpec"                  : "macro",
    "cpec|macro"            : "macro",
    "budget"                : "macro",
    "budget|fiscal"         : "macro",
    "debt"                  : "macro",
    "external"              : "macro",
    "external|macro"        : "macro",
    "ratings"               : "macro",
    "ratings|macro"         : "macro",
    "election|macro"        : "macro",
    "political|macro"       : "macro",
    "privatisation"         : "macro",
    "privatization"         : "macro",
    # Corporate
    "corporates"            : "corporate",
    "banking"               : "corporate",
    "equities"              : "corporate",
    "energy|banking"        : "corporate",
    "equities|forex"        : "corporate",
    "equities|commodities"  : "corporate",
    "corporate|banking"     : "corporate",
    "privatisation|corporate" : "corporate",
    # Energy
    "energy"                : "energy",
    "commodities"           : "energy",
    "fiscal|energy"         : "energy",
    "energy|macro"          : "energy",
    "oil|gas"               : "energy",
    "circular debt"         : "energy",
    # Forex
    "forex"                 : "forex",
    "forex|macro"           : "forex",
    "external|forex"        : "forex",
    "remittance"            : "forex",
}

# ── Pakistan Hard Gate ────────────────────────────────────────────────────────
# EXPANDED: added IMF review terms, FATF, political economy figures,
#           budget/tax signals, PSX index names, credit ratings, LNG/RLNG,
#           CPEC milestones, elections/cabinet with economic link
_PAK_SIGNAL = re.compile(
    r"pakistan|pakist|sbp|pkr|rupee|kse|psx"
    r"|karachi stock|islamabad|lahore|fbr|nepra|ogra"
    r"|pia|cpec|remittance|rda"
    r"|ecc|pso|ogdc|ptcl|hubco"
    r"|engro|fauji|lucky cement|meezan bank"
    r"|habib bank|mcb|nbp|ubl|bahl"
    r"|balance of payment|bop|current account deficit"
    r"|forex reserves|foreign exchange reserve"
    r"|pak economy|privatisation.*airport|airport.*privatis"

    # ── Political Economy Figures ──────────────────────────────────────────
    r"|aurangzeb|ishaq dar|miftah ismail|shaukat tarin|hafeez shaikh|reza baqir"
    r"|muhammad aurangzeb|shamshad akhtar|tariq bajwa|yaseen anwar"
    r"|shehbaz sharif|imran khan|nawaz sharif|asif zardari|bilawal"
    r"|pm pakistan|prime minister pakistan|finance minister pakistan"
    r"|governor sbp|finance ministry pakistan|planning commission pakistan"
    r"|economic coordination committee|advisory council pakistan"

    # ── IMF / Multilateral ─────────────────────────────────────────────────
    r"|imf.*pakistan|pakistan.*imf"
    r"|imf programme|imf review|imf tranche|imf bailout|imf loan"
    r"|imf staff.level|extended fund facility|eef.*pakistan"
    r"|world bank.*pakistan|pakistan.*world bank"
    r"|adb.*pakistan|pakistan.*adb"
    r"|stand.by arrangement|article iv|imf condition"

    # ── FATF ───────────────────────────────────────────────────────────────
    r"|fatf|financial action task force|grey list|black list.*pakistan"
    r"|pakistan.*grey list|money laundering.*pakistan|terror financ.*pakistan"
    r"|aml.*pakistan|pakistan.*aml|counter financing terrorism"

    # ── Budget / Fiscal / Tax ──────────────────────────────────────────────
    r"|federal budget|mini.budget|pakistan budget|budget.*pakistan"
    r"|tax amnesty|amnesty scheme|withholding tax|super tax|income tax.*pakistan"
    r"|fiscal deficit.*pakistan|pakistan.*fiscal deficit"
    r"|pakistan.*revenue|revenue.*pakistan|fbr.*collection|tax collection.*pakistan"
    r"|sales tax|customs duty.*pakistan|pakistan.*customs"
    r"|pension reform|public debt.*pakistan|pakistan.*public debt"
    r"|pakistan.*austerity|austerity.*pakistan|expenditure cut.*pakistan"

    # ── Monetary Policy ────────────────────────────────────────────────────
    r"|policy rate|interest rate.*pakistan|pakistan.*interest rate"
    r"|monetary policy committee|mpc.*pakistan|sbp.*rate|rate.*sbp"
    r"|inflation.*pakistan|pakistan.*inflation|cpi.*pakistan"
    r"|money supply|m2.*pakistan|credit growth.*pakistan"
    r"|currency devaluation|rupee devaluation|rupee depreciation|rupee appreciation"

    # ── External Sector ────────────────────────────────────────────────────
    r"|current account.*pakistan|pakistan.*current account"
    r"|trade deficit.*pakistan|pakistan.*trade deficit"
    r"|export.*pakistan|pakistan.*export|import.*pakistan|pakistan.*import"
    r"|remittance.*pakistan|pakistan.*remittance"
    r"|foreign direct investment.*pakistan|fdi.*pakistan|pakistan.*fdi"
    r"|pakistan.*reserves|reserves.*pakistan|dollar.*reserves"

    # ── Political Events with Economic Impact ──────────────────────────────
    r"|pakistan.*election.*economy|election.*pakistan.*market"
    r"|pakistan.*cabinet.*economy|cabinet reshuffle.*pakistan"
    r"|pm.*resign.*pakistan|pakistan.*pm.*resign"
    r"|political.*instability.*pakistan|pakistan.*political.*uncertainty"
    r"|pakistan.*government.*collapse|no.confidence.*pakistan"
    r"|army.*economy.*pakistan|pakistan.*army.*economic"
    r"|martial law.*pakistan|pakistan.*martial law"
    r"|article 58.*pakistan|pakistan.*article 58"
    r"|pakistan.*political.*crisis|political crisis.*pakistan"
    r"|imran.*economy|economy.*imran"
    r"|shehbaz.*economy|economy.*shehbaz"
    r"|nawaz.*economy|economy.*nawaz"
    r"|pakistan.*regime.*change|regime.*change.*pakistan"
    r"|election.*economy.*pakistan|general election.*pakistan"

    # ── CPEC / Regional Investment ─────────────────────────────────────────
    r"|cpec|china.pakistan economic corridor"
    r"|gwadar|ml.1|orange line|special economic zone.*pakistan"
    r"|pakistan.*investment.*china|china.*investment.*pakistan"
    r"|saudi.*invest.*pakistan|uae.*invest.*pakistan"
    r"|gulf.*remittance|gulf.*worker.*pakistan"

    # ── PSX / Capital Market ───────────────────────────────────────────────
    r"|pakistan.*stock|stock.*pakistan|pakistan.*market|market.*pakistan"
    r"|kse.100|kse.30|all share index|psx.*index|bull.*psx|bear.*psx"
    r"|pakistan.*equity|equity.*pakistan|secp|mutual fund.*pakistan"
    r"|t.bill.*pakistan|pib|sukuk.*pakistan|eurobond.*pakistan"
    r"|pakistan.*bond|bond.*pakistan|credit rating.*pakistan"
    r"|moody.*pakistan|s&p.*pakistan|fitch.*pakistan"
    r"|pakistan.*gdp|pakistan.*growth|pakistan.*debt|pakistan.*loan"

    # ── Energy Sector ──────────────────────────────────────────────────────
    r"|pakistan.*oil|pakistan.*gas|pakistan.*energy|pakistan.*power"
    r"|circular debt|capacity payment|pakistan.*tariff|tariff.*pakistan"
    r"|pakistan.*fuel|fuel.*pakistan|petroleum.*pakistan|pakistan.*petroleum"
    r"|lng.*pakistan|pakistan.*lng|rlng.*pakistan"
    r"|electricity.*pakistan|pakistan.*electricity|load.?shedding"
    r"|ipp.*pakistan|independent power.*pakistan"
    r"|kesc|lesco|iesco|pesco|qesco|hesco"

    # ── Key Sectors ────────────────────────────────────────────────────────
    r"|pakistan.*bank|pakistan.*finance|pakistan.*fiscal"
    r"|pakistan.*rupee|pakistan.*dollar|pakistan.*forex"
    r"|pakistan.*deficit|pakistan.*surplus"
    r"|pakistan.*cement|pakistan.*fertilizer|pakistan.*textile"
    r"|pakistan.*refinery|pakistan.*pipeline"
    r"|pakistan.*cotton|pakistan.*wheat|pakistan.*sugar|pakistan.*rice"
    r"|ppl|ogdcl|mari.*gas"
    r"|secp|pra|t-bill|pib"
    r"|karachi|lahore|islamabad|peshawar|quetta|multan|faisalabad"
    r"|sindh|punjab|kpk|balochistan",
    re.IGNORECASE,
)

# ── Economy-Political Keep Signal ─────────────────────────────────────────────
# Political headlines are KEPT only if they also contain one of these economic
# triggers. This blocks pure rally/protest/jalsa content while keeping cabinet
# reshuffles, PM changes, no-confidence votes with market impact, policy shifts,
# and any political event that directly affects the economy or stock market.
_ECON_POLITICAL_KEEP = re.compile(
    r"budget|tax|fiscal|monetary|economy|economic|gdp|debt|deficit|surplus"
    r"|inflation|interest rate|policy rate|sbp|fbr|secp|ogra|nepra"
    r"|privatisation|privatization|stock|market|rupee|dollar|forex|pkr"
    r"|investment|investor|imf|world bank|adb|loan|bailout|tranche"
    r"|power|energy|gas|oil|circular debt|subsidy|tariff"
    r"|export|import|trade|remittance|current account|balance of payment"
    r"|cabinet|finance minister|pm resign|prime minister resign"
    r"|federal cabinet|economic advisory|planning commission"
    r"|austerity|reform|restructur|privatis"
    r"|no.confidence.*economy|no.confidence.*market|no.confidence.*rupee"
    r"|political.*uncertainty.*market|political.*crisis.*economy"
    r"|regime change.*economy|instability.*market|instability.*rupee"
    r"|election.*market|election.*economy|election.*investor"
    r"|election.*rupee|election.*stock|election.*reform"
    r"|government.*collapse.*economy|collapse.*market"
    r"|martial law.*economy|article 58.*economy"
    r"|pti.*economy|pdm.*economy|pml.*economy|ppp.*economy"
    r"|army.*economy|military.*economy|establishment.*economy",
    re.IGNORECASE,
)

# ── Hard Block Keywords (always dropped, no exceptions) ───────────────────────
_BLOCK_KEYWORDS = [
    # Foreign stock indices
    "sensex", "nifty", "bse india", "nse india", "bombay stock",
    "shanghai composite", "hang seng", "nikkei", "ftse 100",
    "dow jones", "s&p 500", "nasdaq", "wall street rally",
    # Sports — scores and tournaments only, not financial events
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
    # Purely social holidays with no economic angle
    "father's day", "mother's day", "valentine's day", "eid special",
    # Crime / social with no market angle
    "booked for student", "pistol and liquor",
    "suspects arrested for", "three girls",
    # Disaster / weather with no infrastructure or economic angle
    "rain disrupts life", "tanker fire kills",
    "plane crash kills", "train accident kills",
    "building collapse", "explosion kills", "blast kills",
    # Foreign pure-politics with no pak economic link
    "boris johnson", "theresa may", "send her back",
    # Military sports / medals
    "army wins gold", "army wins silver", "army wins bronze",
    "exercise cambrian", "cambrian patrol",
    "pakistan army wins", "pakistan navy wins", "pakistan air force wins",
    "silver medal", "gold medal", "bronze medal",
    "wins trophy", "pride of performance",
    # Pure political rallies — blocked here; re-examined via _ECON_POLITICAL_KEEP
    "rally to condemn", "rally against holy",
    "rally in support of yemen", "rally in london",
    "pdm rally in", "pti rally in", "pml-n rally in", "ppp rally in",
    # Global environment / doomsday with no pak economic link
    "arctic doomsday", "doomsday vault",
    # Irrelevant niche exports
    "cutlery export", "surgical instrument export",
    "sports goods export", "leather goods export",
]

# ── Hard Block Regex (always dropped, no exceptions) ──────────────────────────
# NOTE: Foreign countries are NOT hard-blocked here. They are handled by
# _CONTEXTUAL_ALLOWS so Pakistan-linked stories (CPEC, IMF, oil, remittance,
# FATF, credit ratings) are not lost.
_BLOCK_REGEX = re.compile(
    # Foreign stock indices
    r"sensex|nifty|nikkei|ftse|dow jones"
    r"|hang seng|shanghai composite"
    # Entertainment
    r"|bollywood|film festival|kriti sanon"
    r"|illangovan|actor|actress"
    r"|drama serial|morning show"
    # Crime
    r"|murder|kidnap|rake"
    r"|terrorist attack|bomb blast"
    # Sports scores
    r"|t20|test match|one.day match"
    r"|wins gold|wins silver|wins bronze"
    r"|gold medal|silver medal|bronze medal"
    r"|cambrian patrol|lifts.*cup|wins.*trophy"
    r"|coaching career"
    # Disaster with no economic angle
    r"|death toll|tanker fire|plane crash"
    r"|train accident|building collapse"
    r"|earthquake hits|flood warning"
    # Social / lifestyle
    r"|father.?s day|mother.?s day|valentine.?s day"
    r"|eid special",
    re.IGNORECASE,
)

# ── Contextual Allow List ─────────────────────────────────────────────────────
# Foreign country or topic mentions pass ONLY if Pakistan economic context
# also exists in the same title.
# EXPANDED: added Fed rate, FATF, credit ratings, commodity prices, GSP+
_CONTEXTUAL_ALLOWS = [
    # IMF — always needs pak context
    (r"imf",
     r"pakistan|pakist|programme|loan|bailout|staff.level|review|tranche|eff"),

    # World Bank / ADB / multilateral
    (r"world bank|adb|asian development bank|isdb|idb",
     r"pakistan|pakist|loan|grant|project|infrastructure"),

    # Saudi / UAE / Gulf (major investors and remittance sources)
    (r"saudi|uae|dubai|abu dhabi|gulf|ksa|qatar|kuwait",
     r"pakistan|pakist|investment|deposit|oil|remittance|export|import|trade|cpec|loan"),

    # Oil / crude / LNG (direct import bill impact on pak economy)
    (r"crude oil|brent|wti|opec|lng|rlng",
     r"pakistan|pakist|pkr|rupee|fuel|pso|ogra|import bill|energy|circular debt|petroleum"),

    # Gold (pak forex reserves and jewellery imports)
    (r"gold price|gold rally|gold falls|gold rises|gold hits",
     r"pakistan|pakist|pkr|rupee|reserves|sbp|jewel|import"),

    # US Fed / dollar index (affects pak capital flows and exchange rate)
    (r"federal reserve|fed rate|fomc|fed funds|dxy|dollar index",
     r"pakistan|pakist|pkr|rupee|dollar|gold|oil|commodity|emerging market|sbp"),

    # China / Beijing (CPEC, trade, debt restructuring)
    (r"china|chinese|beijing",
     r"pakistan|pakist|cpec|corridor|loan|investment|import|export|trade|debt|refinanc"),

    # Russia (wheat imports, gas pipeline, trade)
    (r"russia|russian",
     r"pakistan|pakist|wheat|pipeline|gas|oil|trade|investment"),

    # India (trade normalization, water treaty, transit trade)
    (r"india|indian",
     r"pakistan|pakist|trade|export|import|water treaty|indus|border|transit|saarc"),

    # Turkey (growing economic and trade ties)
    (r"turkey|turkish",
     r"pakistan|pakist|trade|investment|loan|bilateral|export|import"),

    # Germany / EU / GSP+ (pak textile export preferences)
    (r"germany|german|eu|european union|gsp",
     r"pakistan|pakist|trade|investment|loan|export|import|gsp|textile"),

    # UK (diaspora remittances, trade, bilateral investment)
    (r"uk|britain|british",
     r"pakistan|pakist|remittance|investment|trade|diaspora|export|import"),

    # FATF (directly affects pak banking and investment climate)
    (r"fatf|financial action task force",
     r"pakistan|pakist|grey list|black list|compliance|money laundering|aml"),

    # Credit rating agencies (pak sovereign and corporate ratings)
    (r"moody|s&p|fitch|dcr",
     r"pakistan|pakist|rating|sovereign|downgrade|upgrade|outlook|bond"),

    # Wheat / food commodities (pak food security and import bill)
    (r"wheat price|global wheat|global food",
     r"pakistan|pakist|import|food security|subsidy|flour|atta"),
]

# ── PSL Keep Signal ───────────────────────────────────────────────────────────
# PSL articles allowed only if they contain a financial/market angle
_PSL_KEEP = re.compile(
    r"kse|psx|stock|share|equity|market|invest|rupee|pkr|sbp"
    r"|pcb|sponsorship|revenue|broadcast|rights|franchise|valuation",
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
    is_psl     = title.str.contains(r"psl", regex=True, na=False)
    psl_is_fin = title.str.contains(_PSL_KEEP, na=False)
    drop |= (is_psl & ~psl_is_fin)

    # Step 4: political mentions — keep only if economic/market context present
    # Blocks pure rally/protest/jalsa content.
    # Allows: PM resignation affecting economy, no-confidence + market reaction,
    #         cabinet reshuffle with policy implications, election + investor impact.
    is_political = title.str.contains(
        r"pti|pdm|pml.n|ppp|mqm|ani"
        r"|rally|protest|dharna|long march"
        r"|jalsa|by.election|na.\d+"
        r"|political.*party|party workers",
        regex=True, na=False,
    )
    has_econ_ctx = title.str.contains(_ECON_POLITICAL_KEEP, na=False)
    drop |= (is_political & ~has_econ_ctx)

    # Step 5: contextual allows — re-block if foreign topic but no pak context
    for foreign_pat, pak_ctx in _CONTEXTUAL_ALLOWS:
        is_foreign  = title.str.contains(foreign_pat, regex=True, na=False)
        has_pak_ctx = title.str.contains(pak_ctx,     regex=True, na=False)
        drop |= (is_foreign & ~has_pak_ctx)

    # Step 6: HARD GATE — must have Pakistan signal regardless of all above
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
        "India"      : r"india",
        "China"      : r"china",
        "UK"         : r"uk",
        "Russia"     : r"russia",
        "Germany"    : r"germany",
        "Egypt"      : r"egypt",
        "Australia"  : r"australia",
        "Canada"     : r"canada",
        "USA"        : r"united states|usa",
        "S.Korea"    : r"south korea",
        "Japan"      : r"japan",
        "Indonesia"  : r"indonesia",
        "Turkey"     : r"turkey",
        "Britain"    : r"britain",
        "Euro"       : r"euro",
        "Greece"     : r"greece",
        "PSL"        : r"psl",
        "Rally/Dharna": r"rally|dharna|jalsa",
        "Election"   : r"election",
        "No-Conf"    : r"no.confidence",
    }
    for label, pattern in checks.items():
        if label == "PSL":
            leaked = df[
                df["title"].str.lower().str.contains(pattern, regex=True, na=False) &
                ~df["title"].str.lower().str.contains(_PSL_KEEP, na=False)
            ]
        elif label in ("Rally/Dharna", "Election", "No-Conf"):
            # For political checks, show how many passed AND how many have econ context
            matched = df[df["title"].str.lower().str.contains(pattern, regex=True, na=False)]
            with_econ = matched[matched["title"].str.lower().str.contains(_ECON_POLITICAL_KEEP, na=False)]
            print(f"   {label:<14}: {len(matched):,} total  |  {len(with_econ):,} with econ context ✅")
            continue
        else:
            leaked = df[df["title"].str.lower().str.contains(pattern, regex=True, na=False)]
        status = f"⚠️  {len(leaked):,} leaked" if len(leaked) else "✅ 0"
        print(f"   {label:<14}: {status}")

    norm_titles = df["title"].fillna("").apply(_normalize_title)
    exact_dups  = norm_titles.duplicated().sum()
    print(f"   Exact dupes   : {'⚠️  ' + str(exact_dups) if exact_dups else '✅ 0'}")
    cats = sorted(df["category"].unique().tolist())
    print(f"   Categories    : {cats}")
    print(f"   Total rows    : {len(df):,}")
    print(f"   Date range    : {df['date'].min()} → {df['date'].max()}")

    # Pakistan gate verification — every article must have a pak signal
    no_pak = df[~df["title"].str.contains(_PAK_SIGNAL, na=False)]
    if len(no_pak):
        print(f"   ⚠️  PAK GATE  : {len(no_pak):,} articles passed without Pakistan signal!")
        print(no_pak["title"].head(10).to_string())
    else:
        print(f"   PAK GATE      : ✅ all articles have Pakistan signal")


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
        "Update news_merged.csv — political economy gate + contextual foreign allow"
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
