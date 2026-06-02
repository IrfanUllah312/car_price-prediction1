# ============================================================
# Name: [Your Name] | Roll No: [Your Roll No]
# Section: 2 - Data Collection (OLX Pakistan Scraper)
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scraping_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://www.olx.com.pk/cars_c84/?page={}"

def scrape_olx_cars(max_pages=20):
    """
    Scrapes used car listings from OLX Pakistan.
    Handles pagination, missing fields, and timeouts.
    """
    all_cars = []
    challenge_log = []

    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page)
        logging.info(f"Scraping OLX page {page}: {url}")
        print(f"[OLX] Scraping page {page}/{max_pages} ...")

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            msg = f"HTTP error on OLX page {page}: {e}"
            logging.warning(msg)
            challenge_log.append(msg)
            time.sleep(5)
            continue
        except requests.exceptions.ConnectionError as e:
            msg = f"Connection error on OLX page {page}: {e}"
            logging.warning(msg)
            challenge_log.append(msg)
            time.sleep(10)
            continue
        except requests.exceptions.Timeout:
            msg = f"Timeout on OLX page {page}"
            logging.warning(msg)
            challenge_log.append(msg)
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # OLX listings are inside <li> with data-aut-id="itemBox"
        listings = soup.find_all("li", {"data-aut-id": "itemBox"})

        if not listings:
            # Try alternate selector
            listings = soup.find_all("div", class_=lambda x: x and "IKo3_" in str(x))

        if not listings:
            logging.info(f"No OLX listings found on page {page}. Stopping.")
            print(f"  No more listings at page {page}. Done.")
            break

        for car in listings:
            try:
                # Title
                title_tag = car.find(attrs={"data-aut-id": "itemTitle"})
                title = title_tag.get_text(strip=True) if title_tag else None

                # Price
                price_tag = car.find(attrs={"data-aut-id": "itemPrice"})
                price_text = price_tag.get_text(strip=True) if price_tag else None

                # Location
                loc_tag = car.find(attrs={"data-aut-id": "item-location"})
                location = loc_tag.get_text(strip=True) if loc_tag else None

                # Date posted
                date_tag = car.find(attrs={"data-aut-id": "item-date"})
                date_posted = date_tag.get_text(strip=True) if date_tag else None

                # Listing URL
                link_tag = car.find("a", href=True)
                listing_url = link_tag["href"] if link_tag else None
                if listing_url and not listing_url.startswith("http"):
                    listing_url = "https://www.olx.com.pk" + listing_url

                # Some detail spans (mileage / year sometimes in subtitle)
                subtitle_tag = car.find(attrs={"data-aut-id": "itemDetails"})
                subtitle = subtitle_tag.get_text(strip=True) if subtitle_tag else None

                all_cars.append({
                    "title": title,
                    "price_pkr": price_text,
                    "location": location,
                    "details": subtitle,
                    "date_posted": date_posted,
                    "source": "OLX",
                    "source_url": listing_url,
                    "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                msg = f"Error parsing OLX listing on page {page}: {e}"
                logging.warning(msg)
                challenge_log.append(msg)
                continue

        sleep_time = random.uniform(2, 4)
        time.sleep(sleep_time)

    with open("logs/olx_challenges.txt", "w") as f:
        f.write("=== OLX Scraping Challenge Log ===\n\n")
        if challenge_log:
            for item in challenge_log:
                f.write(f"- {item}\n")
        else:
            f.write("No major challenges encountered.\n")
        f.write(f"\nTotal records scraped: {len(all_cars)}\n")

    df = pd.DataFrame(all_cars)
    logging.info(f"OLX scraping done. Total rows: {len(df)}")
    print(f"[OLX] Done. Total records: {len(df)}")
    return df


if __name__ == "__main__":
    df = scrape_olx_cars(max_pages=20)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_olx.csv", index=False)
    print("Saved: data/raw_olx.csv")
