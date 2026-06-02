# ============================================================
# Name: [Your Name] | Roll No: [Your Roll No]
# Section: 5 - Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)

# Load preprocessed data (unscaled version from raw)
df = pd.read_csv("data/eda_data.csv")

# Re-do minimal cleaning to get numeric base
drop_cols = ["title", "source_url", "scraped_at", "model", "reg_city", "price_category"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
df.dropna(subset=["price_pkr"], inplace=True)

# Fill numeric NAs
for col in ["year", "mileage_km", "engine_cc", "owners"]:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

print(f"Base shape: {df.shape}")

# ═══════════════════════════════════════════════════════════
# NEW FEATURES (5+ meaningful combinations, not single transforms)
# ═══════════════════════════════════════════════════════════

# Feature 1: Car Age (2024 - year)
# Why: Year alone is less interpretable; age is directly meaningful for depreciation
df["car_age"] = 2024 - df["year"]
print("Feature 1 created: car_age = 2024 - year")

# Feature 2: Mileage per Year (avg annual mileage)
# Why: A car with 100k km in 10 years is less worn than 100k km in 3 years
df["mileage_per_year"] = df["mileage_km"] / (df["car_age"] + 1)
print("Feature 2 created: mileage_per_year = mileage_km / (car_age + 1)")

# Feature 3: Engine Age Score (engine_cc × car_age)
# Why: Large engines in old cars degrade faster → higher maintenance risk
df["engine_age_score"] = df["engine_cc"] * df["car_age"]
print("Feature 3 created: engine_age_score = engine_cc × car_age")

# Feature 4: Luxury Indicator
# Why: Some brands cluster into premium segment — combined indicator
luxury_makes = ["Mercedes", "BMW", "Land Cruiser"]
if "make" in df.columns:
    df["is_luxury"] = df["make"].apply(lambda x: 1 if x in luxury_makes else 0)
else:
    df["is_luxury"] = 0
print("Feature 4 created: is_luxury (1 if Mercedes/BMW/LandCruiser)")

# Feature 5: Feature Score (combined modern features)
# Why: Cars with more features tend to hold value better
feature_cols = [c for c in ["has_sunroof", "has_navigation", "is_imported"] if c in df.columns]
if feature_cols:
    df["feature_score"] = df[feature_cols].sum(axis=1)
    print(f"Feature 5 created: feature_score = sum of {feature_cols}")

# Feature 6: Price Per CC (value for money proxy — will be used to detect anomalies)
# Why: Normalizes price by engine size for fair cross-segment comparison
df["price_per_cc"] = df["price_pkr"] / (df["engine_cc"] + 1)
print("Feature 6 created: price_per_cc = price_pkr / engine_cc")

# ── Non-Linear Transformation ─────────────────────────────
# Log transform of mileage_km
# Why: Mileage effect on price diminishes at high values → log captures this
df["log_mileage"] = np.log1p(df["mileage_km"])
print("\nNon-linear transform: log_mileage = log(1 + mileage_km)")
print("  Justification: Price drops fast with first 50k km then slows → log fits better")

# ── Encode Categoricals for importance scoring ─────────────
from sklearn.preprocessing import LabelEncoder
for col in ["fuel_type", "transmission", "condition", "color", "make", "city", "source"]:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# ── Feature Importance with New Features ──────────────────
print("\n=== IMPACT OF NEW FEATURES (RF Importance) ===")

all_features = [
    "year", "mileage_km", "engine_cc", "transmission", "fuel_type",
    "owners", "car_age", "mileage_per_year", "engine_age_score",
    "is_luxury", "feature_score", "log_mileage",
    "has_sunroof", "has_navigation", "is_imported"
]
available = [f for f in all_features if f in df.columns]

X = df[available].fillna(0)
y = df["price_pkr"]

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
rf.fit(X_s, y)

importance = pd.Series(rf.feature_importances_, index=available).sort_values(ascending=False)
print(importance.to_string())

# Plot
plt.figure(figsize=(10, 7))
importance.plot(kind="barh", color=["#E91E63" if f in
    ["car_age","mileage_per_year","engine_age_score","is_luxury","feature_score","log_mileage"]
    else "#2196F3" for f in importance.index])
plt.title("Feature Importance — Original (Blue) vs New (Pink)", fontweight="bold")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/feature_engineering_importance.png", dpi=120)
plt.close()
print("\nSaved: plots/feature_engineering_importance.png")

# ── Remove Harmful Features ───────────────────────────────
print("\n=== REMOVING LOW-VALUE FEATURES ===")
threshold = 0.01
to_remove = importance[importance < threshold].index.tolist()
# Also remove price_per_cc — it's a leaky feature (uses target)
if "price_per_cc" in df.columns:
    to_remove.append("price_per_cc")
print(f"Removing (importance < {threshold} or leaky): {to_remove}")

final_features = [f for f in available if f not in to_remove]
print(f"Final feature set ({len(final_features)}): {final_features}")

# ── Save engineered data ───────────────────────────────────
df_eng = df[final_features + ["price_pkr"]].copy()
df_eng.fillna(0, inplace=True)
df_eng.to_csv("data/engineered.csv", index=False)
print(f"\nSaved: data/engineered.csv  shape={df_eng.shape}")

import json
with open("data/final_features.json", "w") as f:
    json.dump(final_features, f)
print("Saved: data/final_features.json")

print("\n=== FEATURE ENGINEERING SUMMARY ===")
print(f"""
New Features Created:
  1. car_age           — Direct age of car (more interpretable than year)
  2. mileage_per_year  — Avg annual mileage (usage intensity, not just total)
  3. engine_age_score  — Engine × Age interaction (degradation proxy)
  4. is_luxury         — Binary luxury brand flag
  5. feature_score     — Sum of modern amenities (sunroof + nav + imported)
  6. log_mileage       — Log transform of mileage (captures diminishing effect)

Features Removed: {to_remove}
Reason: Low predictive power (< {threshold} importance) or data leakage.
""")
