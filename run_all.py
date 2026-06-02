# ============================================================
# run_all.py — Run entire ML pipeline in sequence
# (Skip scrapers; use merge_data.py which auto-generates data)
# ============================================================

import subprocess
import sys

steps = [
    ("merge_data.py",          "Step 1/6: Generating/Merging Dataset"),
    ("eda.py",                 "Step 2/6: Exploratory Data Analysis"),
    ("preprocessing.py",       "Step 3/6: Preprocessing"),
    ("feature_engineering.py", "Step 4/6: Feature Engineering"),
    ("model_training.py",      "Step 5/6: Model Training"),
    ("evaluation.py",          "Step 6/6: Evaluation"),
]

for script, label in steps:
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error in {script}. Check output above.")
        sys.exit(1)
    print(f"✓ {script} complete.")

print("\n" + "=" * 60)
print("  ALL STEPS COMPLETE! Launch the app with:")
print("  streamlit run app.py")
print("=" * 60)
