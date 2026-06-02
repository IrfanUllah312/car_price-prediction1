# Pakistan Used Car Price Predictor — ML Project

## Topic
**Pakistan Used Car Price Prediction**  
Scraped from PakWheels.com and OLX Pakistan (Cars section)

---

## Project Structure

```
ml_project/
├── scraper_pakwheels.py      # Section 2: PakWheels scraper
├── scraper_olx.py            # Section 2: OLX Pakistan scraper
├── merge_data.py             # Section 2: Merge + fallback synthetic data
├── eda.py                    # Section 3: Exploratory Data Analysis
├── preprocessing.py          # Section 4: Preprocessing
├── feature_engineering.py    # Section 5: Feature Engineering
├── model_training.py         # Section 6: Model Training
├── evaluation.py             # Section 7: Evaluation & Interpretation
├── app.py                    # Section 8: Streamlit Deployment App
├── requirements.txt
├── data/                     # Created at runtime
├── models/                   # Created at runtime
├── plots/                    # Created at runtime
└── logs/                     # Created at runtime
```

---

## How to Run (in order)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run scrapers (needs internet)
```bash
python scraper_pakwheels.py
python scraper_olx.py
```
> If scraping is blocked (PakWheels uses anti-bot), the next step auto-generates realistic data.

### 3. Merge data (auto-generates synthetic if scraped data < 1000 rows)
```bash
python merge_data.py
```

### 4. Run pipeline in order
```bash
python eda.py
python preprocessing.py
python feature_engineering.py
python model_training.py
python evaluation.py
```

### 5. Launch Streamlit App
```bash
streamlit run app.py
```

---

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo
2. Go to https://streamlit.io/cloud → New App
3. Select your repo → set `app.py` as main file
4. **Important:** Add `models/` folder (with .pkl files) to your repo after running training locally

---

## Features Used for Prediction
- Car Age, Mileage, Engine CC
- Transmission, Fuel Type, Condition
- Number of owners
- Sunroof, Navigation, Imported flag
- Derived: mileage_per_year, engine_age_score, is_luxury, feature_score, log_mileage

## Models Trained
1. Ridge Regression
2. Decision Tree
3. **Random Forest** ← Final Model
4. Gradient Boosting
5. K-Nearest Neighbors

---

## Note for Viva
- You can re-run `scraper_pakwheels.py` live to demonstrate scraping
- All code is documented and justifications are written in comments
- `merge_data.py` auto-fills with realistic synthetic data if scraping is blocked
