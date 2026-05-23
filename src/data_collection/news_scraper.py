"""
================================================================================
 File   : src/data_collection/news_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source | sentiment_score | sentiment_label
          Categories are standardized to 4 values:
            macro | corporate | energy | forex
          Rows are sorted by date across all sources
          Irrelevant non-Pakistan articles are filtered out
          Sentiment scored using FinBERT (GPU if available else CPU)
 Outputs:
          data/processed/news_merged.csv                   <- per-article sentiment
          data/processed/news_aggregated_flags.csv         <- flag approach
          data/processed/news_aggregated_decay_catwise.csv <- category-wise decay approach

 Cache logic:
   1. Raw CSVs exist -> filter + score + save + push merged, then aggregate + push
   2. Raw CSVs missing -> raise error, run scrapers first
================================================================================
"""

import os
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

# ── Irrelevance Filter ────────────────────────────────────────────────────────
IRRELEVANT_KEYWORDS = [
    # foreign currencies
    "yuan", "renminbi", "ringgit", "baht", "peso",
    "lira", "rand", "ruble", "shekel", "sterling", "pound sterling",
    # foreign indices
    "sensex", "nifty", "bse ", "nse india", "bombay stock",
    "shanghai", "hang seng", "nikkei", "ftse", "dow jones",
    "s&p 500", "nasdaq", "wall street",
    # foreign institutions
    "us federal reserve", "european central bank", "bank of england",
    # foreign politics
    "brexit", "uk budget", "uk economy", "uk inflation",
    "theresa may", "boris johnson", "trump rally", "send her back",
    # irrelevant pak exports
    "cutlery export", "cutlery import",
    "surgical export", "surgical instrument",
    "sports goods export", "leather export",
    # foreign countries
    "bangladesh", "sri lanka", "myanmar", "vietnam",
    "latin america", "brazil ", "argentina ",
    "turkey inflation", "iran sanction",
    "afghanistan ", "african ",
    # cricket/sports scores
    "hat-trick", "hat trick", "wicket", "century puts",
    "innings", "thrash", "outplay", "ppfl", "krl",
    "pia beat", "nbp beat", "hbl beat", "ztbl", "kpt score",
    "navy thrash", "paf beat", "army thrash", "ssgc beat",
    "kesc crush", "scores hat", "slams hat",
    "cricket championship", "hockey champions", "football cup",
    "blind cricket", "coaching career",
    # rallies with no market relevance
    "rally in support of yemen", "rally in london",
    "rally in bannu", "rally in bajaur", "rally in quetta",
    "rally against holy quran", "rally to condemn",
    "rally in new york", "pro-muslim rally",
    "pdm rally", "pti rally", "pml-n rally", "ppp rally",
    "mqm rally", "sunni conference",
    # crime/social/disaster
    "booked for student", "girls burning", "pistol and liquor",
    "suspects arrested for", "three girls",
    "rain disrupts life", "flood warning for",
    "flood victims rally",
    # celebrity/entertainment
    "shares she's got covid", "fantastic experiences",
    "lifetime achievement award",
    "vintage cars", "heavy bikes",
    # foreign economic with no pak link
    "record fall in japan", "eurozone unemployment",
    "youth unemployment in europe", "europe seeking end",
    "singapore slashes growth",
]

IRRELEVANT_REGEX = [
    # foreign countries
    r"\bindia\b", r"\bindian\b", r"\bmodi\b",
    r"\bnew delhi\b", r"\brbi\b",
    r"\buk\b", r"\bbrexit\b", r"\bpound\b",
    r"\bsterling\b", r"\beuro\b", r"\beuros\b", r"\becb\b",
    r"\bfederal reserve\b", r"\bwall street\b",
    r"\bsensex\b", r"\bnifty\b", r"\byen\b",
    r"\bwon\b", r"\byuan\b", r"\bchina\b", r"\bchinese\b",
    r"\bjapan\b", r"\bjapanese\b",
    r"\bgreece\b", r"\bgreek\b",
    r"\bbelarus\b",
    # pure politics rallies
    r"\bstages? rally\b", r"\bholds? rally\b",
    r"\baddress.*rally\b", r"\brally.*against\b",
    r"\bthousands.*rally\b", r"\btens of thousands.*rally\b",
    r"\bdemonstration\b", r"\bprotest march\b",
    # crime
    r"\barrested for\b", r"\bbooked for\b",
    r"\brake\b", r"\bmurder\b", r"\bkidnap\b",
    r"\bterrorist attack\b", r"\bbomb blast\b",
    # entertainment/celebrity
    r"\bkriti sanon\b", r"\bbollywood\b",
    r"\bfilm festival\b", r"\bmusic concert\b",
    # weather/disaster
    r"\brain disrupts\b", r"\bflood warning\b",
    r"\bearthquake hits\b",
    # sports
    r"\blifts.*cup\b", r"\bwins.*trophy\b",
    r"\bcoaching career\b", r"\btest match\b",
    r"\bone.day match\b", r"\bt20\b",
]

# Pakistan market signal — keep even if foreign topic detected
_PAK_SIGNAL = (
    r"pakistan|pakist|sbp|pkr|rupee|kse|psx|karachi stock|islamabad|lahore"
    r"|fbr|nepra|ogra|pia|cpec|remittance|rda"
    r"|ecc|pso|ogdc|ptcl|hubco|engro|fauji|lucky|meezan"
    r"|habib|ubi\b|mcb|nbp|ubl|bahl"
    r"|balance of payment|bop\b|current account deficit"
    r"|forex reserves|foreign exchange reserve"
    r"|pak economy|pak.*export|privatisation.*airport"
    r"|aurangzeb|ishaq dar|miftah|shaukat tarin|hafeez shaikh|reza baqir"
    r"|stock market|share price|equity market|market capital"
    r"|interest rate|policy rate|inflation rate|gdp growth"
    r"|trade deficit|fiscal deficit|tax revenue|budget deficit"
)

# Market relevance gate — article must contain at least one of these
_MARKET_SIGNAL = (
    r"stock|share|equity|market|invest|rupee|pkr|sbp|kse|psx"
    r"|bank|finance|fiscal|monetary|budget|tax|tariff|duty"
    r"|inflation|gdp|growth|deficit|surplus|debt|loan|imf|adb|world bank"
    r"|oil|gas|energy|power|electricity|fuel|coal|solar"
    r"|export|import|trade|remittance|current account|balance of payment"
    r"|fbr|nepra|ogra|secp|pra|revenue|privatis"
    r"|interest rate|policy rate|yield|bond|sukuk|treasury"
    r"|profit|loss|earnings|dividend|ipo|listing"
    r"|cement|fertilizer|textile|automobile|pharma|chemical"
    r"|gold|silver|commodity|cotton|wheat|sugar|rice"
    r"|cpec|corridor|infrastructure|refinery|pipeline"
    r"|circular debt|capacity payment|subsidy"
)

_CONTEXTUAL_RULES = [
    (r"\bturkey\b|\berdogan\b", None),
    (r"\bbritain\b|\bbritish\b", None),
    (r"\bsaudi\b", r"pak economy|aurangzeb|deposit|investment|pakistan"),
    (r"\bfederal reserve\b|\bwall street\b|\bfed rate\b|\bfed funds\b",
     r"gold|oil|dollar|crude|commodity|pakistan|pkr|rupee"),
    (r"\bimf\b", r"pakistan|pakist|programme|loan|bailout|staff.level|review"),
    (r"\bworld bank\b|\badb\b|\basian development\b",
     r"pakistan|pakist|loan|grant|project"),
]

_PSL_KEEP_SIGNAL = (
    r"kse|psx|stock|share|equity|market|invest|rupee|pkr|sbp"
    r"|pcb|sponsorship|revenue|broadcast|rights"
)

_PPL_FOOTBALL_RE = r"ppl balochistan football|ppl.*football cup"


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
    title_lower = df["title"].str.lower()
    drop_mask   = pd.Series(False, index=df.index)

    # Stage 1: hard blacklist keywords
    for keyword in IRRELEVANT_KEYWORDS:
        drop_mask |= title_lower.str.contains(keyword, na=False)

    # Stage 2: hard blacklist regex
    for pattern in IRRELEVANT_REGEX:
        drop_mask |= title_lower.str.contains(pattern, regex=True, na=False)

    # Stage 3: contextual foreign-topic rules
    for foreign_pat, extra_signal in _CONTEXTUAL_RULES:
        pak_signal = _PAK_SIGNAL
        if extra_signal:
            pak_signal = pak_signal + r"|" + extra_signal
        is_foreign = title_lower.str.contains(foreign_pat, regex=True, na=False)
        has_pak    = title_lower.str.contains(pak_signal,  regex=True, na=False)
        drop_mask |= (is_foreign & ~has_pak)

    # Stage 4: PSL cricket
    is_psl     = title_lower.str.contains(r"\bpsl\b", regex=True, na=False)
    psl_is_fin = title_lower.str.contains(_PSL_KEEP_SIGNAL, regex=True, na=False)
    drop_mask |= (is_psl & ~psl_is_fin)

    # Stage 5: PPL football
    drop_mask |= title_lower.str.contains(_PPL_FOOTBALL_RE, regex=True, na=False)

    # Stage 6: market relevance gate
    has_market = title_lower.str.contains(_MARKET_SIGNAL, regex=True, na=False)
    drop_mask |= ~has_market

    before  = len(df)
    df      = df[~drop_mask].copy()
    dropped = before - len(df)
    if dropped:
        print(f"   🚫 Dropped {dropped:,} irrelevant articles")
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
        "India"  : r"\bindia\b", "Indian": r"\bindian\b",
        "UK"     : r"\buk\b",    "China" : r"\bchina\b",
        "Euro"   : r"\beuro\b",  "PSL"   : r"\bpsl\b",
        "Japan"  : r"\bjapan\b", "Greece": r"\bgreece\b",
    }
    for label, pattern in checks.items():
        if label == "PSL":
            leaked = df[
                df["title"].str.lower().str.contains(pattern, regex=True, na=False) &
                ~df["title"].str.lower().str.contains(_PSL_KEEP_SIGNAL, regex=True, na=False)
            ]
        else:
            leaked = df[df["title"].str.lower().str.contains(pattern, regex=True, na=False)]
        status = f"⚠️  {len(leaked):,} leaked" if len(leaked) else "✅ 0"
        print(f"   {label:<10}: {status}")
    cats = sorted(df["category"].unique().tolist())
    print(f"   Categories : {cats}")
    print(f"   Total rows : {len(df):,}")


# ── run() ─────────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  PSX News Scraper — Full Pipeline")
    print("=" * 60)

    print("\n📋 Raw CSV check:")
    for source, path in RAW_NEWS_FILES.items():
        print(f"   {source:<12}: {'✅ exists' if path.exists() else '❌ missing'}")

    # Step 1: merge + sentiment + save + push
    df = merge_news()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved merged → {OUTPUT_PATH}  ({len(df):,} rows)")
    sanity_check(df)
    push_to_github(
        [str(OUTPUT_PATH.relative_to(BASE_DIR))],
        "Update news_merged.csv — filtered + FinBERT sentiment"
    )

    # Step 2: flags + save + push
    print("\n" + "=" * 60)
    flags_df = aggregate_flags(df)
    flags_df.to_csv(FLAGS_PATH, index=False)
    print(f"💾 Saved flags  → {FLAGS_PATH}  ({len(flags_df):,} rows)")
    push_to_github(
        [str(FLAGS_PATH.relative_to(BASE_DIR))],
        "Update news_aggregated_flags.csv"
    )

    # Step 3: decay + save + push
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
