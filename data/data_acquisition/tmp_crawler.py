import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import random

# Whether to save raw HTML fragments instead of (or in addition to) plain text
save_html = False

# Output file
output_fname = "tombs_data.json"

# 1. Launch Chromium via Selenium
chromedriver_path = "/home/natalie/Downloads/chromedriver-linux64/chromedriver"

# options = Options()
# options.binary_location = chromedriver_path

service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)
# 2. Navigate to the main tombs listing page
base_url = "https://thebanmappingproject.com/valley-kings"
driver.get(base_url)

# 3. Grab the fully rendered HTML and parse with BS4
soup = BeautifulSoup(driver.page_source, "html.parser")

# 4. Find all <li data‑abbreviation="KV"> and extract their links
elements = driver.find_elements(By.CSS_SELECTOR, 'a[class^="map__sidebar__link"]')

links = [element.get_attribute('href') for element in elements]

# ─── UTILITY TO EXTRACT A SINGLE PAGE ─────────────────────────────────────────

def scrape_tomb(url, save_html=False):
    """Given a tomb-detail URL, return a flat dict of all <h3> → values."""
    driver.get(url)
    # small delay to let JS render
    time.sleep(1.0)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # 1) Tomb identifier by page <title>
    full_title = soup.title.string or ""
    tomb_id = full_title.split("|")[0].strip()

    data = {"tomb_id": tomb_id, "url": url}
    # 2) For every sidebar-item block
    items = soup.find_all("div", class_="tomb__details__sidebar__item")
    for item in items:
        # Extract the label
        h3 = item.find("h3")
        if not h3:
            continue
        label = h3.get_text(strip=True)

        # Gather all values: prefer .field__item, else any direct <div> or text siblings
        values = []

        # a) first look for standardized field__item elements
        for fi in item.select(".field__item"):
            values.append(fi.get_text(strip=True) if not save_html else str(fi))

        # b) if none, look for direct <div> children beyond the <h3>
        if not values:
            for child in item.find_all(recursive=False):
                if child is h3:
                    continue
                if child.name == "div":
                    text = child.get_text(strip=True)
                    if text:
                        values.append(text if not save_html else str(child))

        # c) as a last resort, grab any non-empty text nodes after the <h3>
        if not values:
            for sib in h3.next_siblings:
                txt = getattr(sib, "string", None) or (sib.get_text(strip=True) if getattr(sib, "get_text", None) else "")
                if txt and txt.strip():
                    values.append(txt.strip())

        # Only attach if we found something
        if values:
            # flatten single‐item lists to direct string
            data[label] = values if len(values) > 1 else values[0]

    return data

# ─── MAIN CRAWL LOOP ──────────────────────────────────────────────────────────

all_tombs = []
for link in links:
    try:
        tomb_data = scrape_tomb(link, save_html=save_html)
        all_tombs.append(tomb_data)
    except Exception as e:
        print(f"⚠️ Failed to scrape {link}: {e}")
    time.sleep(5 + random.uniform(0,2))

# ─── WRITE OUT JSON ──────────────────────────────────────────────────────────

with open(output_fname, "w", encoding="utf-8") as f:
    json.dump(all_tombs, f, ensure_ascii=False, indent=2)

print(f"✅ Scraped {len(all_tombs)} tombs → {output_fname}")

# ─── CLEAN UP ─────────────────────────────────────────────────────────────────

driver.quit()
