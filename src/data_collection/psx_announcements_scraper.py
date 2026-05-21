"""
PSX Announcements Scraper
Scrapes company announcements from https://dps.psx.com.pk/announcements
for a predefined list of KSE-100 tickers across multiple announcement types.
"""

import pandas as pd
import time
import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

TICKERS = [
    "PSO.KA", "OGDC.KA", "PPL.KA", "MARI.KA", "POL.KA",
    "HBL.KA", "MCB.KA", "UBL.KA", "NBP.KA", "BAFL.KA",
    "ENGRO.KA", "EFERT.KA", "FATIMA.KA", "FFC.KA",
    "LUCK.KA", "DGKC.KA", "MLCF.KA", "PIOC.KA",
    "NML.KA", "NCL.KA", "GATM.KA", "TRG.KA", "SYS.KA",
    "AVN.KA", "SEARL.KA", "FEROZ.KA", "INDU.KA", "PSMC.KA",
    "HUBC.KA", "KAPCO.KA"
]
YEAR_FROM  = 2010
YEAR_TO    = 2026
OUTPUT_CSV = "data/processed/psx_announcements.csv"

FETCH_TYPES = {
    "C": "Companies Announcements",
    "E": "PSX Notices",
    "D": "Dividends",
    "B": "Book Closure",
    "A": "AGM/EGM",
    "R": "Rights",
    "T": "Takeover/Acquisitions",
}

TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)$", re.IGNORECASE)

_COMPANY_NAME_SUFFIXES = re.compile(
    r"\b(Limited|Bank|Company|Mills|Factory|Corporation|Industries|"
    r"Pakistan|Energy|Energies|Cement|Petroleum|Chemicals|Textiles?|"
    r"Securities|Capital|Investments?|Modaraba|Insurance|Leasing|"
    r"Fertilizer|Fertilizers)\s*(Limited)?\s*$",
    re.IGNORECASE,
)


def make_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--window-size=1400,900")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def is_time_cell(text):
    return bool(TIME_PATTERN.match(text.strip()))


def is_junk(text):
    t = text.strip().upper()
    return t in ("VIEW PDF", "VIEW   PDF", "VIEW", "PDF", "") or is_time_cell(text.strip())


def looks_like_bare_company_name(text):
    t = text.strip()
    words = t.split()
    if len(words) <= 7 and _COMPANY_NAME_SUFFIXES.search(t):
        announcement_keywords = re.compile(
            r"\b(dividend|announce|result|profit|loss|agm|egm|merger|"
            r"acquisition|rights|bonus|notice|suspension|listing|meeting|"
            r"financial|quarterly|annual|half.year|interim|book.closure|"
            r"placement|unusual|movement|issuance|change|de-?list|"
            r"resumption|adjustment|recomposition|rebalancing)\b",
            re.IGNORECASE,
        )
        if announcement_keywords.search(t):
            return False
        return True
    return False


def extract_title_from_td(td_el, driver):
    result = driver.execute_script(
        "var td = arguments[0];"
        "var anchors = td.querySelectorAll('a');"
        "var best = '';"
        "for (var i = 0; i < anchors.length; i++) {"
        "  var a = anchors[i];"
        "  var txt = (a.innerText || '').trim();"
        "  var titleAttr = (a.getAttribute('title') || '').trim();"
        "  var dataTitle = (a.getAttribute('data-title') || '').trim();"
        "  var ariaLabel = (a.getAttribute('aria-label') || '').trim();"
        "  if (txt.length > 5 && txt.toUpperCase() !== 'VIEW PDF' &&"
        "      txt.toUpperCase() !== 'VIEW' && txt.toUpperCase() !== 'PDF') {"
        "    best = txt; break;"
        "  }"
        "  if (titleAttr.length > 5) { best = titleAttr; break; }"
        "  if (dataTitle.length > 5) { best = dataTitle; break; }"
        "  if (ariaLabel.length > 5) { best = ariaLabel; break; }"
        "}"
        "if (!best) {"
        "  var raw = (td.innerText || '').trim();"
        "  raw = raw.replace(/View\\s*PDF/gi, '').replace(/\\bView\\b/gi, '').trim();"
        "  if (raw.length > 5) best = raw;"
        "}"
        "return best;",
        td_el
    )
    return (result or "").strip()


def parse_row(driver, row):
    tds = row.find_elements(By.TAG_NAME, "td")
    if len(tds) < 2:
        return None

    date_val         = ""
    title_candidates = []

    for td in tds:
        text = td.text.strip()
        if not text or is_junk(text) or is_time_cell(text):
            continue
        if not date_val:
            try:
                dt = pd.to_datetime(text, dayfirst=False, errors="coerce")
                if not pd.isna(dt) and 2000 <= dt.year <= 2030:
                    date_val = text
                    continue
            except Exception:
                pass
        candidate = extract_title_from_td(td, driver)
        if candidate and not is_junk(candidate) and not is_time_cell(candidate):
            title_candidates.append(candidate)

    if not date_val or not title_candidates:
        return None

    filtered = [t for t in title_candidates if not looks_like_bare_company_name(t)]
    pool      = filtered if filtered else title_candidates
    title_val = max(pool, key=len)

    if looks_like_bare_company_name(title_val):
        return None

    try:
        dt = pd.to_datetime(date_val, dayfirst=False, errors="coerce")
        if pd.isna(dt):
            return None
        if not (YEAR_FROM <= dt.year <= YEAR_TO):
            return None
        return {"date": dt.strftime("%b %d, %Y"), "title": title_val}
    except Exception:
        return None


def set_type(driver, type_value):
    try:
        selects = driver.find_elements(By.CSS_SELECTOR, "select[name='type']")
        for sel_el in selects:
            opts = [o.get_attribute("value") for o in
                    sel_el.find_elements(By.TAG_NAME, "option")]
            if type_value in opts:
                Select(sel_el).select_by_value(type_value)
                return True
    except Exception:
        pass
    return False


def set_symbol(driver, wait, psx_symbol):
    if not psx_symbol:
        return None
    try:
        sym_box = wait.until(
            EC.element_to_be_clickable((By.ID, "announcementsSearch"))
        )
        sym_box.clear()
        time.sleep(0.3)
        sym_box.send_keys(psx_symbol)
        time.sleep(2)
        items = driver.find_elements(
            By.CSS_SELECTOR,
            ".autocomplete__results a, .autocomplete__result, "
            ".autocomplete-items div"
        )
        for item in items:
            if item.text.strip().upper().startswith(psx_symbol.upper()):
                item.click()
                time.sleep(0.5)
                return sym_box
        if items:
            items[0].click()
            time.sleep(0.5)
        return sym_box
    except Exception as e:
        print(f"  ERROR symbol: {e}")
        return None


def click_search_btn(driver, sym_box):
    for sel in [
        ".announcementsResults__header .form__button",
        "button.form__button",
    ]:
        try:
            btns = driver.find_elements(By.CSS_SELECTOR, sel)
            if btns:
                driver.execute_script("arguments[0].click();", btns[0])
                time.sleep(4)
                return
        except Exception:
            pass
    if sym_box:
        sym_box.send_keys(Keys.RETURN)
    time.sleep(4)


def click_next(driver):
    result = driver.execute_script(
        "var containers = [document.querySelector('.announcementsResults__header'),"
        "document.querySelector('.announcementsResults'),document.body];"
        "for (var i = 0; i < containers.length; i++) {"
        "  var container = containers[i]; if (!container) continue;"
        "  var candidates = container.querySelectorAll("
        "    'button.next, a.next, .paginate_button.next, button[class*=\"next\"], a[class*=\"next\"]');"
        "  for (var j = 0; j < candidates.length; j++) {"
        "    var btn = candidates[j];"
        "    var cls = (btn.className||'')+(btn.parentElement?btn.parentElement.className:'');"
        "    if (cls.indexOf('disabled') !== -1) continue;"
        "    if (btn.disabled) return 'disabled';"
        "    btn.click(); return 'clicked';"
        "  }"
        "  var all = container.querySelectorAll('button, a, span');"
        "  for (var k = 0; k < all.length; k++) {"
        "    var el = all[k];"
        "    if (el.innerText && el.innerText.trim() === 'Next') {"
        "      var elcls = (el.className||'')+(el.parentElement?el.parentElement.className:'');"
        "      if (elcls.indexOf('disabled') !== -1) return 'disabled';"
        "      el.click(); return 'clicked';"
        "    }"
        "  }"
        "}"
        "return 'not-found';"
    )
    return result == "clicked"


def get_rows(driver):
    for sel in [
        "#announcementsTable tbody tr",
        ".announcementsResults table tbody tr",
        ".announcementsResults tbody tr",
        "table.display tbody tr",
        "table tbody tr",
    ]:
        rows = driver.find_elements(By.CSS_SELECTOR, sel)
        if rows:
            return rows
    return []


def scrape_one(driver, yahoo_ticker, psx_symbol, type_value, type_label):
    records  = []
    wait     = WebDriverWait(driver, 20)
    page_num = 1

    driver.get("https://dps.psx.com.pk/announcements")
    time.sleep(3)

    if not set_type(driver, type_value):
        print(f"    WARNING: type '{type_value}' not found on page — skipping.")
        return records

    sym_box = set_symbol(driver, wait, psx_symbol)
    click_search_btn(driver, sym_box)

    while True:
        time.sleep(2)
        rows = get_rows(driver)
        if not rows:
            break

        page_kept = 0
        for row in rows:
            data = parse_row(driver, row)
            if not data:
                continue
            records.append({
                "DATE":   data["date"],
                "SYMBOL": yahoo_ticker,
                "TITLE":  data["title"],
                "TYPE":   type_label,
            })
            page_kept += 1

        print(f"    [{psx_symbol or 'ALL'}/{type_label[:3]}] "
              f"Page {page_num}: {len(rows)} rows → {page_kept} kept "
              f"| total: {len(records)}")

        if click_next(driver):
            page_num += 1
        else:
            print(f"    [{psx_symbol or 'ALL'}/{type_label[:3]}] "
                  f"All {page_num} pages done.")
            break

    print(f"    [{psx_symbol or 'ALL'}/{type_label[:3]}] DONE — {len(records)} records")
    return records


def run():
    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Already exists: {OUTPUT_CSV} — skipping.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_records = []
    driver      = make_driver()

    try:
        TICKER_TYPES = {k: v for k, v in FETCH_TYPES.items() if k != "E"}

        for type_value, type_label in TICKER_TYPES.items():
            print(f"\n{'='*60}")
            print(f"PART — {type_label} (Type {type_value})")
            print(f"{'='*60}")
            for yahoo_ticker in TICKERS:
                psx_symbol = yahoo_ticker.replace(".KA", "")
                print(f"\n  {yahoo_ticker}")
                recs = scrape_one(driver, yahoo_ticker, psx_symbol,
                                  type_value, type_label)
                all_records.extend(recs)
                time.sleep(2)

        print(f"\n{'='*60}")
        print("PART — PSX Notices (Type E)")
        print(f"{'='*60}")
        psx_recs = scrape_one(driver, "PSX_NOTICE", "", "E", "PSX Notices")
        all_records.extend(psx_recs)
        print(f"\n  PSX Notices total: {len(psx_recs)}")

    finally:
        driver.quit()
        print("\nBrowser closed.")

    df = pd.DataFrame(all_records, columns=["DATE", "SYMBOL", "TITLE", "TYPE"])

    junk_titles = {"VIEW PDF", "VIEW   PDF", "VIEW", "PDF", ""}
    df = df[~df["TITLE"].str.strip().str.upper().isin(junk_titles)]
    df = df[~df["TITLE"].str.strip().str.match(r"^\d{1,2}:\d{2}\s*(AM|PM)$", case=False)]
    df = df[~df["TITLE"].apply(looks_like_bare_company_name)]
    df = df[df["TITLE"].str.strip().str.len() >= 8]
    df = df.drop_duplicates(subset=["DATE", "SYMBOL", "TITLE"])
    df = df.reset_index(drop=True)

    df["_sort"] = pd.to_datetime(df["DATE"], format="%b %d, %Y", errors="coerce")
    df = (df.sort_values(["SYMBOL", "_sort"], ascending=[True, False])
            .drop(columns="_sort")
            .reset_index(drop=True))

    print(f"\n{'='*60}")
    print(f"TOTAL RECORDS: {len(df)}")
    print(f"{'='*60}")
    print(df.head(20).to_string(index=False))

    print("\nYear distribution:")
    df["year"] = pd.to_datetime(df["DATE"], format="%b %d, %Y", errors="coerce").dt.year
    print(df["year"].value_counts().sort_index().to_string())
    df = df.drop(columns=["year"])

    print("\nBreakdown by TYPE:")
    print(df.groupby("TYPE").size().to_string())

    print("\nBreakdown by SYMBOL:")
    print(df.groupby("SYMBOL").size().to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"✅ {OUTPUT_CSV} already exists. Nothing to do.")
    else:
        run()
