"""
================================================================================
 File   : src/data_collection/news_scraper.py
 Project: PSX LSTM Predictor
 Purpose: Merge news from Dawn, BRecorder, and Mettis into a single CSV
          keeping only: date | category | title | source | sentiment_score | sentiment_label
          Categories are standardized to 11 values:
            macro | corporate | energy | forex | banking |
            cement | fertilizer | auto | tech | pharma | textile
          Rows are sorted by date across all sources
          Duplicates removed across all sources and categories
          Irrelevant rows filtered out (foreign/entertainment/sports noise)
          Sentiment scored using FinBERT (GPU if available else CPU)
 Outputs:
          data/processed/news_merged.csv                   <- per-article sentiment (pre-filter)
          data/processed/news_filtered.csv                 <- after relevance filter
          data/processed/news_aggregated_flags.csv         <- flag approach (on filtered)
          data/processed/news_aggregated_decay_catwise.csv <- category-wise decay (on filtered)

 Cache logic:
   1. Raw CSVs exist -> merge + dedupe + score + save + push merged,
      then filter + save + push filtered,
      then aggregate + push
   2. Raw CSVs missing -> raise error, run scrapers first
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

OUTPUT_PATH   = PROCESSED / "news_merged.csv"
FILTERED_PATH = PROCESSED / "news_filtered.csv"
FLAGS_PATH    = PROCESSED / "news_aggregated_flags.csv"
DECAY_PATH    = PROCESSED / "news_aggregated_decay_catwise.csv"

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE    = 32

SENTIMENT_COLS = [
    "sentiment_macro",
    "sentiment_corporate",
    "sentiment_energy",
    "sentiment_forex",
    "sentiment_banking",
    "sentiment_cement",
    "sentiment_fertilizer",
    "sentiment_auto",
    "sentiment_tech",
    "sentiment_pharma",
    "sentiment_textile",
]

DECAY_FACTORS = {
    "sentiment_macro"      : 0.85,
    "sentiment_corporate"  : 0.70,
    "sentiment_energy"     : 0.70,
    "sentiment_forex"      : 0.80,
    "sentiment_banking"    : 0.70,
    "sentiment_cement"     : 0.70,
    "sentiment_fertilizer" : 0.70,
    "sentiment_auto"       : 0.70,
    "sentiment_tech"       : 0.70,
    "sentiment_pharma"     : 0.70,
    "sentiment_textile"    : 0.70,
}

# ── Category Mapping ──────────────────────────────────────────────────────────
# Maps raw scraper categories -> standardized internal categories
# banking kept as banking (was wrongly mapped to corporate before)
# general_market, general, rates, stocks, exchange added
CATEGORY_MAP = {
    "macro"                : "macro",
    "fiscal"               : "macro",
    "monetary"             : "macro",
    "market_political"     : "macro",
    "macro|monetary"       : "macro",
    "macro|fiscal"         : "macro",
    "general_market"       : "macro",
    "general"              : "macro",
    "rates"                : "macro",
    "corporates"           : "corporate",
    "equities"             : "corporate",
    "stocks"               : "corporate",
    "energy|banking"       : "corporate",
    "equities|forex"       : "corporate",
    "equities|commodities" : "corporate",
    "corporate"            : "corporate",
    "energy"               : "energy",
    "commodities"          : "energy",
    "fiscal|energy"        : "energy",
    "forex"                : "forex",
    "exchange"             : "forex",
    "banking"              : "banking",
}

# ── Sector Keyword Detection ──────────────────────────────────────────────────
# Applied AFTER category mapping to override category based on title keywords
# Covers sectors not present as raw categories in any scraper
SECTOR_KEYWORDS = {
    "cement"      : [
        "cement", "clinker", "lucky cement", "dgkc", "dg khan cement",
        "maple leaf cement", "pioneer cement", "pioc", "mlcf",
    ],
    "fertilizer"  : [
        "fertilizer", "fertiliser", "urea", "dap", "engro fertilizer",
        "efert", "fauji fertilizer", "ffc", "fatima fertilizer", "fatima",
    ],
    "auto"        : [
        "automobile", "automotive", "indus motor", "pak suzuki", "psmc",
        "vehicle sales", "car sales", "auto sector", "auto industry",
    ],
    "tech"        : [
        "software export", "technology sector", "trg pakistan",
        "systems limited", "avanceon", "it sector", "it industry",
        "tech sector", "it exports",
    ],
    "pharma"      : [
        "pharma", "pharmaceutical", "searle", "ferozsons",
        "drug pricing", "healthcare sector",
    ],
    "textile"     : [
        "textile", "nishat mills", "nishat chunian", "nml", "ncl",
        "spinning", "fabric", "yarn", "garment export",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# ── Relevance Filter ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

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
    "banks.*reluctance.*india.*gold", "india plan to draw out gold",
    "india continues blocking social media",
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
    r"us stocks.*(apple|push higher|slide on|rise despite)",
    r"fall in us stocks", r"soft landing.*us stocks",
    r"coronavirus crash.*world stocks",
    r"crude falls.*us stocks", r"oil rises.*us stocks",
    r"trump trade.*us stocks",
    r"wall st.*rally", r"wall st.*edges", r"wall st.*higher", r"wall st.*lower",
    r"asia markets.*wall st",
    r"^global stock markets",
    r"^corporate debt risks weighing on world",
    r"^slowing us inflation",
    r"^fed rate hike.*wrong move.*emerging",
    r"^eu cuts eurozone",
    r"^world economy",
    r"^pandemic to hit growth in asia",
    r"^un lowers global",
    r"^world bank to lower",
    r"^gold reaches a new high.*\$",
    r"^gold.*\$\d",
    r"imf.*wto.*oecd.*free trade",
    r"^imf chief says us-china",
    r"^imf warns of a.*greater global",
    r"imf to give away.*gold profits",
    r"oil price fall.*good news.*world economy.*imf",
    r"^imf chief (seeks|calls for|says|warns eu)",
    r"^imf warns eu",
    r"^imf chief seeks second term",
    r"^saudi arabia borrows",
    r"^bangladesh current account",
    r"^soaring inflation pushed.*\d+m.*poverty.*un",
    r"^turkiye delivers",
    "snap stock falls",
    "world stocks rally", "us stocks rally", "us stocks falter",
    "global stocks rally", "santa claus rally", "stocks track wall st",
    "foreigners shun thai stocks",
    "us federal reserve", "federal reserve slashes", "federal reserve hikes",
    "federal reserve cuts", "european central bank", "ecb",
    "bank of japan", "bank of england", "bank of canada", "reserve bank of australia",
    "reserve bank of new zealand",
    "us economy", "us unemployment", "us-china trade",
    "gloomy bernanke", "fed confident", "bernanke sees", "bernanke defends",
    "us shale", "shale producer", "shale output",
    "france to lead imf", "lagarde favorite", "lagarde bid for imf", "lagarde for imf",
    "shares tumble around the world",
    "imf warns.*ai", "imf warns us on debt",
    "turkiye.*rate hike", "turkey.*rate hike",
    "argentina", "erdogan",
    "hong kong", "virginia", "bangkok", "tokyo",
    "brazil", "venezuela", "myanmar", "sri lanka",
    "north korea", "turkish lira", "turkey's lira",
    "uk far-right", "far-right rally", "white supremacist rally", "pro-gun rally",
    "supporters rally in damascus", "assad supporters rally",
    "thousands rally in malaysia", "anwar's resignation",
    "kurdish referendum", "spain's right", "catalonia",
    "hundreds rally in moscow", "hundreds arrested in russian anti-putin",
    "foes and fans rally ahead of putin", "russia.*rally.*navalny",
    "thousands rally.*lebanon", "thousands rally.*support.*lebanon",
    "protesters in belarus rally",
    "presidential hopeful.*dies.*colombia.*rally",
    "libyan oil supply", "libyan supply return", "libyan supply resumption",
    "alibaba", "ali baba", "snapchat", "ge shares", "ge plunge",
    "siemens buys", "saudi aramco", "arada", "fab, sace",
    "mcdonald's shares", "e. coli outbreak tied to quarter pounder",
    "renren shares", "total shares dive",
    "nigeria: blackouts", "stampede at nigerian", "nigerian president's rally",
    "defence cuts could worsen us unemployment",
    "massive crowd, chaos preceded deadly india rally stampede",
    "us unemployment rate hits", "unemployment claims at lowest",
    "zero job growth sparks recession",
    "number of us families affected by unemployment",
    "eurozone unemployment", "eurozone escapes recession", "eurozone recession extends",
    "germany dodges recession", "germany.*first quarter growth",
    "swiss.*recession", "switzerland faces first recession",
    "despite debt deal.*europe may slide",
    "australia.*unemployment", "australian unemployment", "south african unemployment",
    "facebook ipo", "facebook stock", "facebook's mark", "zuckerberg",
    "saverin", "twitter wants to fly with billion", "twitter sets.*ipo",
    "twitter.*anti-facebook", "zynga shares",
    "netflix stock", "apple stock plunges", "apple shares rise on china mobile",
    "apple hails results but outlook knocks",
    "nokia shares", "sony shares", "nintendo.*shares",
    "samsung shares plunge", "samsung.*apple.*top market share",
    "sharp shares dive", "disney to buy back", "new philips lights up",
    "blackberry firm refuses", "google.*shares stay airborne", "gamestop",
    "ant group ipo", "uber.*ipo", "uber launches ipo",
    "pinterest.*ipo", "xiaomi plans.*ipo", "candy crush.maker.*ipo",
    "ferrari valued.*ipo", "city schools planning.*education ipo",
    "aramco ipo", "aramco declares.*valuation", "aramco prices shares",
    "aramco shares rocket", "aramco to announce ipo", "shelved aramco ipo",
    "5 things to know about saudi.*ipo",
    "questions over valuation delaying aramco",
    "aramco names first woman.*overseas",
    "international investors grab.*aramco shares",
    "bitcoin falls", "bitcoin hits record high", "crypto.*hawala alternative",
    "squash classics.*jansher", "fans rally behind afridi",
    "dhoni unimpressed", "sharma credits zaheer",
    "dravid pleased to share", "chappell to share his knowledge",
    "spieth and reed share open lead", "morikawa shares us open lead",
    "murray gets a massage from djokovic",
    "dortmund hold on despite late real rally",
    "djokovic to take stock after",
    "australia rally to restrict england", "south africa rally to 309",
    "england rally around sterling", "crouch earns stoke share of spoils",
    "rain has final say.*england.*south africa share t20",
    "girona fight back to share points", "liverpool go top.*united rally",
    "sincaraz.*women share the spotlight",
    "scheffler.*masters lead", "mcilroy falters.*masters lead",
    "morikawa.*shares.*open lead", "mcilroy shares.*masters lead",
    "bangladesh.*wi share first", "nz bowlers.*share.*wickets.*india",
    "india.*nz share points.*world cup", "zimbabwe.*bangladesh share.*t20",
    "yasir.*williamson share honours",
    "england rally after india", "india double strike", "indian rickshaw rally",
    "zayn malik shares a throwback",
    "taylor swift.*music is held hostage", "celebrity friends rally behind taylor swift",
    "ali zafar shares success of kill dil",
    "samina baig shares feelings.*top of the world",
    "sajal.*bilal.*share the big screen in khel khel mein",
    "jibran nasir shares how.*whatsapp.*hacked",
    "game of thrones cast share",
    "in muthi bhar chahat.*aagha ali.*resham share",
    "blogger anam hakeem shares.*parents.*travel",
    "tahir shah shares.*farishta", "multan sultans.*tim david shares psl",
    "i used my heart and head.*paray hut love.*asim raza",
    "adnan siddiqui shares first look.*partition",
    "arish razi says collab", "nadia afgan shares", "dananeer.*acne.*pictures",
    "gynaecologist.*dr tahira.*practices.*women.*stay alive",
    "ahmed ali akbar got married",
    "naqvi.*investigators share initial findings.*asp",
    "abhishek bachchan shares his love.*aishwarya",
    "adil omar promises.*copyright strikes",
    "priyanka congratulates sania mirza.*time.*most influential",
    "two americans and german share chemistry nobel",
    "three physicists share nobel.*black hole",
    "chappell to share his knowledge of india with australia",
    "aamir.*fakhar share limelight", "adnan.*khalil share 17 wickets.*ravi",
    "zebras.*lions share title after rain", "services earn lion.*share.*boxing finals",
    "khawaja surgery fear.*brittle aussies take stock",
    "imam.*haider share spotlight on day of mixed fortunes",
    "stampede at nigerian president.*rally kills",
    "blast rocks zimbabwean president.*rally",
    "thousands rally in kenya against president",
    "kerry warns afghanistan as thousands rally in support of abdullah",
    "cambodia police use smoke grenades.*rally",
    "blast at hekmatyar.*rally",
    "14 killed in bomb attack on afghan election rally",
    "suicide bomber kills 13 in election rally in afghanistan",
    "bombing kills 24 at afghan president.*rally",
    "200,000 rohingya rally", "rohingya youths share.*social media",
    "rohingya refugees.*rally.*go home",
    "hundreds rally in sudan", "sudan protesters.*rally.*army hq",
    "four students shot dead at sudan rally", "sudan protesters.*generals trade blame",
    "protesters storm pro-kurdish party rally",
    "kurdish party.*rally comes under attack in turkey",
    "blast at.*thai political rally", "two killed.*41 wounded.*thai protest rally",
    "thai government urges police to arrest rally leaders",
    "thai army invokes martial law", "thailand.*ruling junta lifts martial law",
    "thai pro-democracy rally",
    "30 civilians.*police injured after rally near thai king",
    "police arrest 21 at pro-democracy rally in thailand",
    "thousands rally against.*serbian government",
    "thousands in sweden rally", "riot in sweden after.*banned from rally",
    "neo-nazi rally.*stockholm", "clashes over anti-immigration rally in sweden",
    "massive rally in prague calls for czech pm",
    "thousands march in poland nationalist rally",
    "hundreds rally in french city.*iran", "france begins nationwide strike",
    "thousands march in paris.*vent anger over inflation",
    "le pen slams.*witch-hunt.*paris rally",
    "leftist leaders gather in spain.*rally against.*far right",
    "anti-racism rally in paris", "macron holds first rally.*french election",
    "thousands of poles rally", "thousands rally in warsaw",
    "south koreans rally", "tens of thousands rally for taiwan independence",
    "hk police arrest protesters",
    "nine veteran hk activists convicted over democracy rally",
    "use of facial recognition in new delhi rally",
    "huge anti-graft rally.*india govt",
    "students rally after woman threatened with rape.*india",
    "thousands.*rally.*india over nun gang-rape",
    "protester shot dead after modi rally in srinagar",
    "bomb blasts before indian opposition rally kill",
    "police say indian mujahideen.*modi rally",
    "narendra modi vows unity.*jammu-kashmir rally",
    "bihar police deny terror alert for modi rally",
    "shinde faces music.*deadly rally bombs",
    "rahul leads farmers.*anti-govt rally in delhi",
    "mass rally against inflation in new delhi",
    "india.*congress holds rally.*defiance against modi",
    "opposition stages giant joint rally to oust modi",
    "big opposition rally warns modi.*match-fixing",
    "india police ban rally.*adani port",
    "evicted wall st protesters seek rebound with rally",
    "uk muslims hold rally against extremism",
    "muslims denounce is.*barbarism.*paris rally",
    "muslims rally against extremism in germany",
    "hundreds of muslims rally.*white house.*post-nz attack",
    "religious leaders rally for american muslims",
    "thousands rally for unity in egypt.*tahrir",
    "egypt.*rally.*morsi", "pro-assad rally",
    "mass rally backs syrian president",
    "thousands rally for egypt military chief",
    "egypt islamic activists rally",
    "thousands rally in cairo.*tahrir", "cairo.*tahrir.*mass rally",
    "rally keeps up reform pressure on egypt",
    "bahrain declares martial law",
    "bahrain opposition rally", "bahrain hopes for normalcy",
    "bahrain sunnis warn government.*rally",
    "amid deadly unrest.*bahrain.*opposition calls rally",
    "hundreds rally in bahrain", "hundreds rally in moscow",
    "hundreds arrested in russian anti-putin rally",
    "moscow police clash with navalny", "tens of thousands.*anti-kremlin rally",
    "navalny supporters rally across russia",
    "moscow police arrest hundreds at banned rally",
    "several arrested at huge opposition rally in moscow",
    "foes and fans rally ahead of putin inauguration",
    "anti-racism protesters rally across uk",
    "thousands in iran rally.*against us", "iranians rally en masse.*rioting",
    "iranian president in europe to rally support", "iranian unrest pushes crude",
    "rally demands revival of arab royals.*hunting",
    "traders organise rally in support of uae leaders",
    "aswj holds rally in saudi arabia.*support",
    "students rally against.*air raids on gaza",
    "thousands rally outside white house.*call for end to israeli offensive",
    "indonesians rally against new laws",
    "fuel price protests.*nepal", "nepal police fire teargas.*fuel price protests",
    "bangladesh.*mob beats.*hasina",
    "bangladesh players rally behind.*shakib al hasan",
    "bangladesh rally.*attacks on hindus",
    "tens of thousands rally in bangladesh over attacks on hindus",
    "thousands rally in bangladesh.*ban.*former pm",
    "thousands rally in lisbon against racism",
    "trump brands biden.*enemy of the state.*pennsylvania rally",
    "trump mocks democrats.*insults pelosi.*campaign rally since",
    "trump lashes out at media in arizona rally",
    "secret service says no gun involved in trump rally",
    "chaotic scenes as trump cancels rally",
    "this sherwani-clad guy.*trump rally",
    "anti-trump protests turn violent.*new mexico rally",
    "here's what we know about thomas matthew crooks",
    "man with loaded firearm arrested at donald trump rally",
    "harris to rally where trump riled capitol",
    "trump vows to end.*american decline.*inauguration.*rally",
    "takeaways from trump's pre-inauguration rally speech",
    "big us donors rally around nikki haley",
    "trump blames.*crazed.*media for rally taunt",
    "us army veteran charged with plotting to bomb white nationalist rally",
    "muslim woman ejected from trump rally",
    "'no kings' protesters rally across us",
    "trump rips into diversity.*rally-style speech",
    "trump shares messages from france.*macron",
    "clashes in australia.*thousands hold rally against lockdown",
    "australian police use pepper spray.*anti-immigration rally",
    "south korean doctors rally", "south korean.*martial law",
    "south korea parliament rejects president",
    "south korea.*yoon impeach", "south korean.*yoon",
    "lee jae-myung rides anti-martial law wave",
    "bus crash kills.*anc supporters",
    "dakar rally.*goncalves dies", "dakar rally 2012.*top shots",
    "dakar rally in mourning", "two saudi women set to compete in dakar rally",
    "motorcycle champion sanders.*dakar rally",
    "samba financial group.*national commercial bank",
    "gloomy bernanke sees slow drop in us unemployment",
    "fed confident us unemployment", "defence cuts could worsen us unemployment",
    "south african unemployment hits 11-year high",
    "europe seeking end to.*nightmare.*youth unemployment",
    "tackling youth unemployment in europe",
    "eurozone unemployment", "european youth unemployment",
    "australia.*unemployment rate", "british official unemployment",
    "world's highest unemployment ails arab region",
]

EXPLICIT_KEEP = [
    "privatisation of psl",
    "nbp to form women cricket",
    "psx in share sale talks with qatar, istanbul",
    "avanceon signs mou with pe energy to expand its footprint in nigeria",
    "stampede at pti",
    "police blame pti for deaths in multan",
    "three-member committee constituted to investigate stampede at pti",
    "ge and cmec mark important milestone",
    "ge, harbin electric", "thal power to invest", "thal nova",
    "lucky cement starts production in iraq",
    "indus motor signs export agreement with egypt",
    "indus motor company eyes import of used vehicles",
    "indus motor company begins export of vehicles",
    "pak-russia negotiating", "polish company to drill",
    "russia-ukraine crisis to hurt pakistan",
    "russian firm geared up to start feasibility",
    "pak inks agreement with azerbaijan",
    "pll negotiating lng deal with azerbaijan",
    "negotiations underway to get lng from azerbaijan",
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
    r"\bppfl\b", "premier football league", "pakistan premier football league",
    "ppl.*football", "ppl balochistan football", "balochistan football cup",
    "national hockey", "national hockey championship", "national hockey opener",
    "hockey players", "hockey gold", "hockey silver", "hockey team returns",
    "women hockey team", "wapda.*hockey", "nbp.*hockey", "ssgc.*hockey",
    "pia.*hockey", "kpt.*hold.*hbl", "hbl.*hold.*kpt",
    "nbp crowned national hockey", "nbp edge ssgc.*hockey",
    "nbp overcome ztbl", "navy stun wapda",
    "wapda.*football", "football.*wapda",
    "wapda.*cricket", "ppl cricket", "nbp cricket", "ssgc cricket",
    "ssgc.*gas supply.*cricket", "gas supply.*cricket match",
    r"hbl almost through.*pia struggle",
    r"(hbl|pia).*(almost through|struggle).*(final|cup|semi)",
    r"(hbl|nbp|pia|ssgc|kesc|wapda|paf|krl|ztbl|kpt|pel).*\d+-\d+",
    r"\d+-\d+.*(hbl|nbp|pia|ssgc|kesc|wapda|paf|krl|ztbl|kpt|pel)",
    "prime minister awards gold medal to nbp",
    "gold medal", "gold medalist", "silver medal", "bronze medal",
    "wins gold", "win gold", "won gold", "grabs gold", "grab gold",
    "asian games.*gold", "sag.*gold", "asian beach games",
    "karate championship", "wushu trophy", "stamp exhibition.*gold",
    "olympic gold medalist arshad nadeem",
    "inam wins gold", "inam butt.*gold", "inam butt.*returns",
    "pakistan wins gold", "pakistan win.*gold", "pakistan grabs.*gold",
    "pakistan beat india.*gold", "pakistan beat india.*shootout",
    "wapda players shine.*games", "wapda.*south asian games",
    "wapda honours.*players",
    "vintage car rally", "vintage.*rally", "classic car rally",
    r"\bcar rally\b", "peace car rally", "car rally.*waziristan",
    "karachi chronicle",
    "fans share.*encounters.*cricketers",
    "cricketer.*shares.*teaser.*drama", "cricketer.*shares.*teaser.*series",
    "footballer.*shares.*mental health",
    r"foreign cricketers excited.*pakistan.*psl",
    "commentators.*analysts share.*psl experience",
    "sensational cricket product.*psl experience",
    "street footballers share their stories",
    "unemployment looms.*aussie cricketers",
    "blow to departmental cricket.*hbl disbands",
    "thousands defy.*rally.*qadri", "defy rally ban.*qadri",
    r"for aswj.*capital.*open.*rally",
    "strange bedfellows.*share stage.*pat.*rally",
    r"bilawal.*\d+-city.*rally tour", "bilawal.*election rally tour",
    "police break.*pti rally.*korangi",
    "mall road rally.*imran.*lahore",
    "nawaz takes on.*judges.*rally",
    r"pti cancels rally",
    "registration of rally attack.*ppp lawmakers",
    "rally.*bin laden", "bin laden.*rally",
    "blasphemy.*rally", "rally.*blasphemy",
    "rally.*drone", "drone.*rally",
    "rally.*target killings",
    "political rally.*killed", "killed.*political rally",
    "firing.*political rally", "political rally.*firing",
    "rally.*kashmir", "rally.*against.*us",
    "pickpockets.*political rally", "pickpockets.*rally",
    "anoushey ashraf.*pml-n.*rally",
    "twin cities.*pti rally.*cricket match",
    "karachi fishermen.*political rally",
    "dpc rally.*drone", "rocket attack.*political rally.*waziristan",
    "drama serial", "film release", "box office",
    "recipe", "fashion week", "komal rizvi",
    "zayn malik shares a throwback",
    "taylor swift.*music is held hostage",
    "worldwide movement against blasphemy",
    "pakhtuns only rally to denounce",
    r"imran ashraf shares.*film",
    r".*shares.*song.*film",
    "huda kattan.*content creators.*share",
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


def filter_news(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n🔎 Applying relevance filter to {len(df):,} articles ...")
    df   = df.copy()
    mask = df["title"].apply(is_relevant)
    df_filtered   = df[mask].reset_index(drop=True)
    removed_count = (~mask).sum()
    print(f"   ✅ Retained : {len(df_filtered):,}")
    print(f"   🗑️  Removed  : {removed_count:,}")
    print(f"   📈 Retention: {len(df_filtered) / len(df) * 100:.1f}%")
    return df_filtered


# ══════════════════════════════════════════════════════════════════════════════


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


# ── Keyword-based Sector Recategorization ─────────────────────────────────────
def recategorize_by_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    After standardize_category(), override category to a specific sector
    if the article title contains sector-specific keywords.
    This adds cement, fertilizer, auto, tech, pharma, textile categories
    which do not exist as raw scraper categories.
    More specific sectors take priority over generic corporate/macro.
    """
    df   = df.copy()
    t    = df["title"].str.lower().fillna("")

    for sector, keywords in SECTOR_KEYWORDS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        mask    = t.str.contains(pattern, na=False)
        df.loc[mask, "category"] = sector

    counts = df["category"].value_counts()
    print(f"   Category distribution after keyword recategorization:")
    for cat, cnt in counts.items():
        print(f"     {cat:<15} {cnt:>6,}")
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

    df["_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(99)
    df.sort_values(["date", "_norm", "_priority"], inplace=True)
    df = df.drop_duplicates(subset=["date", "_norm"], keep="first")

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

    # keyword recategorization before dedup so sector articles
    # get correct category going into aggregation
    print("\n🏷️  Applying keyword-based sector recategorization ...")
    merged = recategorize_by_keywords(merged)

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
        "macro"       : "sentiment_macro",
        "corporate"   : "sentiment_corporate",
        "energy"      : "sentiment_energy",
        "forex"       : "sentiment_forex",
        "banking"     : "sentiment_banking",
        "cement"      : "sentiment_cement",
        "fertilizer"  : "sentiment_fertilizer",
        "auto"        : "sentiment_auto",
        "tech"        : "sentiment_tech",
        "pharma"      : "sentiment_pharma",
        "textile"     : "sentiment_textile",
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
        "sentiment_macro"      : "has_macro",
        "sentiment_corporate"  : "has_corporate",
        "sentiment_energy"     : "has_energy",
        "sentiment_forex"      : "has_forex",
        "sentiment_banking"    : "has_banking",
        "sentiment_cement"     : "has_cement",
        "sentiment_fertilizer" : "has_fertilizer",
        "sentiment_auto"       : "has_auto",
        "sentiment_tech"       : "has_tech",
        "sentiment_pharma"     : "has_pharma",
        "sentiment_textile"    : "has_textile",
    }
    for sent_col, flag_col in flag_map.items():
        agg[flag_col] = agg[sent_col].notna().astype(float)
    agg[SENTIMENT_COLS] = agg[SENTIMENT_COLS].fillna(0.0)
    col_order = (
        ["date"] +
        [c for pair in zip(SENTIMENT_COLS, flag_map.values()) for c in pair] +
        ["news_count"]
    )
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
        print(f"   {col:<30} decay={factor}")
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
    norm_titles = df["title"].fillna("").apply(_normalize_title)
    exact_dups  = norm_titles.duplicated().sum()
    print(f"   Exact dupes : {'⚠️  ' + str(exact_dups) if exact_dups else '✅ 0'}")
    cats = sorted(df["category"].unique().tolist())
    print(f"   Categories  : {cats}")
    print(f"   Total rows  : {len(df):,}")
    print(f"   Date range  : {df['date'].min()} → {df['date'].max()}")


# ── run() ─────────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  PSX News Scraper — Full Pipeline")
    print("=" * 60)

    print("\n📋 Raw CSV check:")
    for source, path in RAW_NEWS_FILES.items():
        print(f"   {source:<12}: {'✅ exists' if path.exists() else '❌ missing'}")

    # ── Step 1: Merge, dedupe, score ──────────────────────────────────────────
    df = merge_news()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved merged → {OUTPUT_PATH}  ({len(df):,} rows)")
    sanity_check(df)
    push_to_github(
        [str(OUTPUT_PATH.relative_to(BASE_DIR))],
        "Update news_merged.csv — deduped + FinBERT + 11 sector categories"
    )

    # ── Step 2: Filter irrelevant articles ───────────────────────────────────
    print("\n" + "=" * 60)
    df_filtered = filter_news(df)
    df_filtered.to_csv(FILTERED_PATH, index=False)
    print(f"💾 Saved filtered → {FILTERED_PATH}  ({len(df_filtered):,} rows)")
    push_to_github(
        [str(FILTERED_PATH.relative_to(BASE_DIR))],
        "Update news_filtered.csv — relevance filter applied"
    )

    # ── Step 3: Aggregate on filtered data ───────────────────────────────────
    print("\n" + "=" * 60)
    flags_df = aggregate_flags(df_filtered)
    flags_df.to_csv(FLAGS_PATH, index=False)
    print(f"💾 Saved flags  → {FLAGS_PATH}  ({len(flags_df):,} rows)")
    push_to_github(
        [str(FLAGS_PATH.relative_to(BASE_DIR))],
        "Update news_aggregated_flags.csv — 11 sector categories"
    )

    print("\n" + "=" * 60)
    decay_df = aggregate_decay(df_filtered)
    decay_df.to_csv(DECAY_PATH, index=False)
    print(f"💾 Saved decay  → {DECAY_PATH}  ({len(decay_df):,} rows)")
    push_to_github(
        [str(DECAY_PATH.relative_to(BASE_DIR))],
        "Update news_aggregated_decay_catwise.csv — 11 sector categories"
    )

    print("\n" + "=" * 60)
    print("  ✅ All done")
    print(f"    {OUTPUT_PATH.name:<40} {len(df):>7,} rows")
    print(f"    {FILTERED_PATH.name:<40} {len(df_filtered):>7,} rows")
    print(f"    {FLAGS_PATH.name:<40} {len(flags_df):>7,} rows")
    print(f"    {DECAY_PATH.name:<40} {len(decay_df):>7,} rows")
    print("=" * 60)
    return df, df_filtered, flags_df, decay_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
