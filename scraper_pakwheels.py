# ============================================================
# Name: [Your Name] | Roll No: [Your Roll No]
# Section: 2 - Data Collection (PakWheels Scraper)
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import os
from datetime import datetime

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scraping_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://www.pakwheels.com/used-cars/search/-/?page={}"

def scrape_pakwheels(max_pages=20):
    """
    Scrapes used car listings from PakWheels.
    Handles pagination, missing fields, and rate limiting.
    """
    all_cars = []
    challenge_log = []

    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page)
        logging.info(f"Scraping PakWheels page {page}: {url}")
        print(f"[PakWheels] Scraping page {page}/{max_pages} ...")

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            msg = f"HTTP error on page {page}: {e}"
            logging.warning(msg)
            challenge_log.append(msg)
            time.sleep(5)
            continue
        except requests.exceptions.ConnectionError as e:
            msg = f"Connection error on page {page}: {e}"
            logging.warning(msg)
            challenge_log.append(msg)
            time.sleep(10)
            continue
        except requests.exceptions.Timeout:
            msg = f"Timeout on page {page}"
            logging.warning(msg)
            challenge_log.append(msg)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        listings = soup.find_all("li", class_="classified-listing")

        if not listings:
            logging.info(f"No listings found on page {page}. Stopping.")
            print(f"  No more listings at page {page}. Done.")
            break

        for car in listings:
            try:
                # Title / Name
                title_tag = car.find("a", class_="car-name ad-detail-path")
                title = title_tag.get_text(strip=True) if title_tag else None

                # Price
                price_tag = car.find("div", class_="price-details")
                price_text = price_tag.get_text(strip=True) if price_tag else None

                # Details list (year, mileage, fuel, transmission, etc.)
                details = car.find_all("li", class_=lambda x: x and "detail-item" in x)
                detail_texts = [d.get_text(strip=True) for d in details]

                year       = detail_texts[0] if len(detail_texts) > 0 else None
                mileage    = detail_texts[1] if len(detail_texts) > 1 else None
                fuel_type  = detail_texts[2] if len(detail_texts) > 2 else None
                transmission = detail_texts[3] if len(detail_texts) > 3 else None

                # City
                city_tag = car.find("span", class_=lambda x: x and "city" in str(x).lower())
                city = city_tag.get_text(strip=True) if city_tag else None

                # Listing URL
                link_tag = car.find("a", class_="car-name ad-detail-path")
                listing_url = ("https://www.pakwheels.com" + link_tag["href"]) if link_tag else None

                all_cars.append({
                    "title": title,
                    "price_pkr": price_text,
                    "year": year,
                    "mileage_km": mileage,
                    "fuel_type": fuel_type,
                    "transmission": transmission,
                    "city": city,
                    "source": "PakWheels",
                    "source_url": listing_url,
                    "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                msg = f"Error parsing a listing on page {page}: {e}"
                logging.warning(msg)
                challenge_log.append(msg)
                continue

        # Polite rate limiting
        sleep_time = random.uniform(2, 4)
        time.sleep(sleep_time)

    # Save challenge log
    with open("logs/pakwheels_challenges.txt", "w") as f:
        f.write("=== PakWheels Scraping Challenge Log ===\n\n")
        if challenge_log:
            for item in challenge_log:
                f.write(f"- {item}\n")
        else:
            f.write("No major challenges encountered.\n")
        f.write(f"\nTotal records scraped: {len(all_cars)}\n")

    df = pd.DataFrame(all_cars)
    logging.info(f"PakWheels scraping done. Total rows: {len(df)}")
    print(f"[PakWheels] Done. Total records: {len(df)}")
    return df


if __name__ == "__main__":
    df = scrape_pakwheels(max_pages=20)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_pakwheels.csv", index=False)
    print("Saved: data/raw_pakwheels.csv")
