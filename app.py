# ============================================================
# Name: [Your Name] | Roll No: [Your Roll No]
# Section: 8 - Deployment (Streamlit Web App)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-size: 0.95rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .price-text {
        font-size: 2.4rem;
        font-weight: 900;
        color: #00d4ff;
    }
    .confidence-text {
        font-size: 1rem;
        color: #aaa;
        margin-top: 0.5rem;
    }
    .category-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        margin-top: 1rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🚗 Pakistan Used Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Scraping-based ML model trained on PakWheels & OLX Pakistan data</div>', unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model   = joblib.load("models/final_model.pkl")
        scaler  = joblib.load("models/scaler.pkl")
        features = joblib.load("models/features.pkl")
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None

model, scaler, features = load_model()

if model is None:
    st.error("⚠️ Model files not found. Please run model_training.py first to generate models.")
    st.info("Run: `python merge_data.py` → `python eda.py` → `python preprocessing.py` → `python feature_engineering.py` → `python model_training.py`")
    st.stop()

# ── Input Form ─────────────────────────────────────────────
st.subheader("📋 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Model Year", list(range(2005, 2024))[::-1], index=2)

    make_options = [
        "Toyota", "Honda", "Suzuki", "Kia", "Hyundai",
        "Daihatsu", "Mitsubishi", "Nissan", "Mercedes", "BMW"
    ]
    make = st.selectbox("Car Make (Brand)", make_options)

    mileage = st.number_input(
        "Mileage (km)", min_value=0, max_value=500000,
        value=50000, step=5000
    )

    engine_cc = st.selectbox(
        "Engine Capacity (CC)",
        [660, 800, 1000, 1200, 1300, 1500, 1600, 1800, 2000, 2500, 3000],
        index=4
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Hybrid", "CNG", "LPG"]
    )

with col2:
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])

    condition = st.selectbox(
        "Condition",
        ["Excellent", "Good", "Average", "Fair"]
    )

    owners = st.slider("Number of Previous Owners", 1, 5, 1)

    has_sunroof  = st.checkbox("Has Sunroof")
    has_nav      = st.checkbox("Has Navigation / Infotainment")
    is_imported  = st.checkbox("Is Imported / Overseas")

# ── Input Validation ───────────────────────────────────────
def validate_inputs():
    errors = []
    if year > 2024:
        errors.append("Year cannot be in the future.")
    if mileage < 0:
        errors.append("Mileage cannot be negative.")
    if engine_cc <= 0:
        errors.append("Engine CC must be positive.")
    return errors

# ── Predict Button ─────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True):

    errors = validate_inputs()
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
    else:
        # ── Build feature vector ───────────────────────────
        car_age         = 2024 - year
        mileage_per_year = mileage / (car_age + 1)
        engine_age_score = engine_cc * car_age
        log_mileage      = np.log1p(mileage)
        luxury_brands    = ["Mercedes", "BMW"]
        is_luxury        = 1 if make in luxury_brands else 0
        feature_score    = int(has_sunroof) + int(has_nav) + int(is_imported)

        # Map categoricals to numeric (same encoding as preprocessing)
        fuel_map  = {"Petrol": 3, "Diesel": 0, "Hybrid": 1, "CNG": 4, "LPG": 2}
        trans_map = {"Automatic": 0, "Manual": 1}
        cond_map  = {"Excellent": 0, "Good": 1, "Average": 2, "Fair": 3}

        # Build dict with all possible features
        input_dict = {
            "year":              year,
            "mileage_km":        mileage,
            "engine_cc":         engine_cc,
            "transmission":      trans_map.get(transmission, 0),
            "fuel_type":         fuel_map.get(fuel_type, 3),
            "owners":            owners,
            "has_sunroof":       int(has_sunroof),
            "has_navigation":    int(has_nav),
            "is_imported":       int(is_imported),
            "car_age":           car_age,
            "mileage_per_year":  mileage_per_year,
            "engine_age_score":  engine_age_score,
            "is_luxury":         is_luxury,
            "feature_score":     feature_score,
            "log_mileage":       log_mileage,
        }

        # Fill in order of training features
        input_values = [input_dict.get(f, 0) for f in features]
        X_input = np.array(input_values).reshape(1, -1)

        # Scale
        X_scaled = scaler.transform(X_input)

        # Predict
        predicted_price = model.predict(X_scaled)[0]
        predicted_price = max(200000, int(predicted_price / 10000) * 10000)

        # Price category
        if predicted_price < 2000000:
            category = "Budget 🟢"
            badge_color = "#4CAF50"
        elif predicted_price < 5000000:
            category = "Mid-Range 🔵"
            badge_color = "#2196F3"
        elif predicted_price < 12000000:
            category = "Premium 🟠"
            badge_color = "#FF9800"
        else:
            category = "Luxury 🔴"
            badge_color = "#E91E63"

        # Confidence range (±10%)
        lo = int(predicted_price * 0.90 / 10000) * 10000
        hi = int(predicted_price * 1.10 / 10000) * 10000

        # Display result
        st.markdown(f"""
        <div class="result-box">
            <div style="color:#aaa; font-size:0.9rem; margin-bottom:0.5rem;">ESTIMATED PRICE</div>
            <div class="price-text">PKR {predicted_price:,}</div>
            <div class="confidence-text">Confidence Range: PKR {lo:,} — PKR {hi:,}</div>
            <div class="category-badge" style="background:{badge_color}33; color:{badge_color}; border: 1.5px solid {badge_color};">
                {category}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary table
        st.markdown("---")
        st.subheader("📊 Input Summary")
        summary = {
            "Make": make, "Year": year, "Age (yrs)": car_age,
            "Mileage": f"{mileage:,} km", "Engine": f"{engine_cc} CC",
            "Fuel": fuel_type, "Transmission": transmission,
            "Condition": condition, "Previous Owners": owners,
        }
        sum_df = pd.DataFrame(summary.items(), columns=["Feature", "Value"])
        st.table(sum_df)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem;'>
    Data scraped from <b>PakWheels.com</b> & <b>OLX Pakistan</b> | 
    ML Model: Random Forest | 
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
