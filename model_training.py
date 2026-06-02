# ============================================================
# Name: [Your Name] | Roll No: [Your Roll No]
# Section: 6 - Model Training
# ============================================================

import pandas as pd
import numpy as np
import time
import tracemalloc
import json
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv("data/engineered.csv")
with open("data/final_features.json") as f:
    features = json.load(f)

features = [f for f in features if f in df.columns]
X = df[features].fillna(0).values
y = df["price_pkr"].values

print(f"Data shape: X={X.shape}, y={y.shape}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")

# ── Stratified K-Fold using price categories ───────────────
def get_price_cat(p):
    if p < 2000000:    return 0
    elif p < 5000000:  return 1
    elif p < 12000000: return 2
    else:              return 3

y_cat = np.array([get_price_cat(p) for p in y])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Model Definitions ─────────────────────────────────────
models = {
    "Ridge Regression":        Ridge(alpha=100),
    "Decision Tree":           DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42),
    "K-Nearest Neighbors":     KNeighborsRegressor(n_neighbors=7)
}

results = {}

print("\n" + "=" * 60)
print("TRAINING 5 MODELS WITH 5-FOLD STRATIFIED CV")
print("=" * 60)

for name, model in models.items():
    print(f"\n[{name}]")

    # Track memory & time
    tracemalloc.start()
    start = time.time()

    cv_scores = cross_val_score(model, X_scaled, y, cv=skf,
                                scoring="r2", n_jobs=-1)

    elapsed = time.time() - start
    _, peak_mem = tracemalloc.get_traced_memory()  # bytes
    tracemalloc.stop()

    # Fit full model for individual metrics
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    results[name] = {
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std":  cv_scores.std(),
        "MAE":        mae,
        "RMSE":       rmse,
        "R2":         r2,
        "time_sec":   elapsed,
        "memory_kb":  peak_mem / 1024
    }

    print(f"  CV R2:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  MAE:    {mae:,.0f} PKR")
    print(f"  RMSE:   {rmse:,.0f} PKR")
    print(f"  R2:     {r2:.4f}")
    print(f"  Time:   {elapsed:.2f}s  |  Memory: {peak_mem/1024:.1f} KB")

# ── Results Table ──────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)
results_df = pd.DataFrame(results).T
print(results_df.to_string())
results_df.to_csv("models/model_comparison.csv")
print("Saved: models/model_comparison.csv")

# ── Top 2 Models: Hyperparameter Tuning ───────────────────
top2 = results_df["CV_R2_mean"].nlargest(2).index.tolist()
print(f"\nTop 2 models for tuning: {top2}")

# --- Random Forest Tuning ---
if "Random Forest" in top2:
    print("\n=== GridSearchCV on Random Forest ===")
    rf_params_grid = {
        "n_estimators":  [50, 100, 200],
        "max_depth":     [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    gs_rf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0
    )
    gs_rf.fit(X_scaled, y)
    print(f"  Best params:  {gs_rf.best_params_}")
    print(f"  Best CV R2:   {gs_rf.best_score_:.4f}")

    print("\n=== RandomizedSearchCV on Random Forest ===")
    rf_params_rand = {
        "n_estimators":      [50, 100, 150, 200, 250],
        "max_depth":         [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 3, 5, 7],
        "max_features":      ["sqrt", "log2", None]
    }
    rs_rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params_rand, n_iter=20, cv=3, scoring="r2",
        random_state=42, n_jobs=-1, verbose=0
    )
    rs_rf.fit(X_scaled, y)
    print(f"  Best params:  {rs_rf.best_params_}")
    print(f"  Best CV R2:   {rs_rf.best_score_:.4f}")
    print(f"\n  Grid vs Random: Grid={gs_rf.best_score_:.4f} | Random={rs_rf.best_score_:.4f}")

    best_rf = gs_rf.best_estimator_
    joblib.dump(best_rf, "models/random_forest.pkl")
    print("Saved: models/random_forest.pkl")

# --- Gradient Boosting Tuning ---
if "Gradient Boosting" in top2:
    print("\n=== GridSearchCV on Gradient Boosting ===")
    gb_params_grid = {
        "n_estimators":  [50, 100, 150],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth":     [3, 5]
    }
    gs_gb = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0
    )
    gs_gb.fit(X_scaled, y)
    print(f"  Best params: {gs_gb.best_params_}")
    print(f"  Best CV R2:  {gs_gb.best_score_:.4f}")

    print("\n=== RandomizedSearchCV on Gradient Boosting ===")
    gb_params_rand = {
        "n_estimators":  [50, 100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "max_depth":     [2, 3, 4, 5, 6],
        "subsample":     [0.6, 0.8, 1.0]
    }
    rs_gb = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params_rand, n_iter=20, cv=3, scoring="r2",
        random_state=42, n_jobs=-1, verbose=0
    )
    rs_gb.fit(X_scaled, y)
    print(f"  Best params: {rs_gb.best_params_}")
    print(f"  Best CV R2:  {rs_gb.best_score_:.4f}")

    best_gb = gs_gb.best_estimator_
    joblib.dump(best_gb, "models/gradient_boosting.pkl")
    print("Saved: models/gradient_boosting.pkl")

# ── Save Final Model ──────────────────────────────────────
best_model_name = results_df["CV_R2_mean"].idxmax()
print(f"\n{'='*60}")
print(f"FINAL MODEL SELECTED: {best_model_name}")
print(f"Reason: Highest cross-validated R2 ({results_df.loc[best_model_name,'CV_R2_mean']:.4f})")
print(f"{'='*60}")

# Load whichever best tuned model was saved
if os.path.exists("models/random_forest.pkl"):
    final_model = joblib.load("models/random_forest.pkl")
elif os.path.exists("models/gradient_boosting.pkl"):
    final_model = joblib.load("models/gradient_boosting.pkl")
else:
    final_model = models[best_model_name]
    final_model.fit(X_scaled, y)

joblib.dump(final_model, "models/final_model.pkl")
print("Saved: models/final_model.pkl")

# ── Learning Curves Plot ───────────────────────────────────
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_scaled, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="r2", n_jobs=-1
)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#2196F3", label="Training R2")
plt.plot(train_sizes, val_scores.mean(axis=1),   "o-", color="#FF5722", label="Validation R2")
plt.fill_between(train_sizes,
                 train_scores.mean(1) - train_scores.std(1),
                 train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#2196F3")
plt.fill_between(train_sizes,
                 val_scores.mean(1) - val_scores.std(1),
                 val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="#FF5722")
plt.title(f"Learning Curves — {best_model_name}", fontweight="bold")
plt.xlabel("Training Set Size")
plt.ylabel("R2 Score")
plt.legend()
plt.tight_layout()
plt.savefig("plots/learning_curves.png", dpi=120)
plt.close()
print("Saved: plots/learning_curves.png")

print("\nModel training complete.")
