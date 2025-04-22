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
save_html = False

def extract_sidebar_fields(soup, url):
    """Parses <div class='tomb__details__sidebar__item'> blocks into key-value pairs."""
    full_title = soup.title.string or ""
    tomb_id = full_title.split("|")[0].strip()
    data = {"tomb_id": tomb_id, "url": url}

    items = soup.find_all("div", class_="tomb__details__sidebar__item")
    for item in items:
        h3 = item.find("h3")
        if not h3:
            continue
        label = h3.get_text(strip=True)
        values = []

        for fi in item.select(".field__item"):
            values.append(fi.get_text(strip=True) if not save_html else str(fi))

        if not values:
            for child in item.find_all(recursive=False):
                if child is h3:
                    continue
                if child.name == "div":
                    text = child.get_text(strip=True)
                    if text:
                        values.append(text if not save_html else str(child))

        if not values:
            for sib in h3.next_siblings:
                txt = getattr(sib, "string", None) or (sib.get_text(strip=True) if getattr(sib, "get_text", None) else "")
                if txt and txt.strip():
                    values.append(txt.strip())

        if values:
            data[label] = values if len(values) > 1 else values[0]

    return data

def main():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1280,800")
    driver = uc.Chrome(options=options, headless=False)

    all_data = []

    try:
        print(f"🔗 Opening: {START_URL}")
        driver.get(START_URL)

        # Click "Map"
        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.LINK_TEXT, "Map"))).click()
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "js-map-sidebar")))
        print("🗺️  Map loaded.")

        elements = driver.find_elements(By.CSS_SELECTOR, 'a.map__sidebar__link')
        links = []
        for el in elements:
            href = el.get_attribute("href")
            name = el.get_attribute("data-name")
            if href and "/tombs/" in href and "#chamber" not in href:
                links.append((name.strip() if name else "Unknown", urljoin(START_URL, href)))

        print(f"🔍 Found {len(links)} tomb links.")

        for i, (name, url) in enumerate(links):
            print(f"🔸 ({i+1}/{len(links)}) Visiting: {name} → {url}")
            try:
                driver.get(url)
                time.sleep(1.5)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                data = extract_sidebar_fields(soup, url)
                data["tomb_name"] = name
                all_data.append(data)
            except Exception as e:
                print(f"⚠️ Failed to scrape {url}: {e}")
            time.sleep(15 + random.uniform(0, 10))

    finally:
        driver.quit()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Done! {len(all_data)} tombs scraped to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
