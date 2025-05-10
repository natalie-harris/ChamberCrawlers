# Author: Natalie E. G. Harris & Brenna J. Bentley
# Description: This script scrapes tomb data from the Theban Mapping Project website.
import json
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

START_URL = "https://thebanmappingproject.com/valley-kings"
OUTPUT_FILE = "tombs_data.json"

# Whether to save raw HTML fragments instead of (or in addition to) plain text
save_html = False

# ─── UTILITY TO EXTRACT A SINGLE PAGE ─────────────────────────────────────────
# Function to extract tomb data from the sidebar
def scrape_tomb(soup, url):

    # Tomb identifier by page <title>
    full_title = soup.title.string or ""
    tomb_id = full_title.split("|")[0].strip()
    data = {"tomb_id": tomb_id, "url": url}

    # For every sidebar-item block
    items = soup.find_all("div", class_="tomb__details__sidebar__item")
    for item in items:
        # Extract the label
        h3 = item.find("h3")
        if not h3:
            continue
        label = h3.get_text(strip=True)
        values = []

        # Gather all values: prefer .field__item, else any direct <div> or text siblings
        ### a) first look for standardized field__item elements
        for fi in item.select(".field__item"):
            values.append(fi.get_text(strip=True) if not save_html else str(fi))
        ### b) if none, look for direct <div> children beyond the <h3>
        if not values:
            for child in item.find_all(recursive=False):
                if child is h3:
                    continue
                if child.name == "div":
                    text = child.get_text(strip=True)
                    if text:
                        values.append(text if not save_html else str(child))
        ### c) as a last resort, grab any non-empty text nodes after the <h3>
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
def main():
    # Set up the Chrome driver with undetected_chromedriver to avoid detection by bot detection systems
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1280,800")
    # Watch scraping happen in real time
    driver = uc.Chrome(options=options, headless=False)

    all_tombs = []

    try:
        print(f"Opening: {START_URL}")
        driver.get(START_URL)

        # Click "Map"
        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.LINK_TEXT, "Map"))).click()
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "js-map-sidebar")))
        print("Map loaded.")

        # Wait for the map to load and then find all tomb links
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.map__sidebar__link')
        links = []
        for el in elements:
            href = el.get_attribute("href")
            name = el.get_attribute("data-name")
            if href and "/tombs/" in href and "#chamber" not in href:
                links.append((name.strip() if name else "Unknown", urljoin(START_URL, href)))

        print(f"Found {len(links)} tomb links.")

        # Loop through each tomb link and scrape data
        for i, (name, url) in enumerate(links):
            print(f"   ({i+1}/{len(links)}) Visiting: {name} → {url}")
            try:
                driver.get(url)
                time.sleep(1.5)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                data = scrape_tomb(soup, url)
                data["tomb_name"] = name
                all_tombs.append(data)
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
            time.sleep(15 + random.uniform(0, 10))

    finally:
        driver.quit()

    # ─── WRITE OUT JSON ──────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tombs, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(all_tombs)} tombs scraped to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
