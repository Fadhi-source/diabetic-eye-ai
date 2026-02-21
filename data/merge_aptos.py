"""
data/merge_aptos.py
One-time script: merge APTOS image labels with synthetic EHR features.

Run from project root:
    python data/merge_aptos.py

Input:
    data/aptos_train.csv   — APTOS Kaggle CSV (id_code, diagnosis)
    data/synthetic_ehr.csv — Generated synthetic EHR (2000 rows)
    data/images/           — APTOS train images (already copied here)

Output:
    data/merged.csv        — Unified dataset used by the training pipeline
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, IMAGE_DIR, SYNTHETIC_CSV, RANDOM_SEED

np.random.seed(RANDOM_SEED)

APTOS_CSV  = os.path.join(DATA_DIR, "aptos_train.csv")
MERGED_CSV = os.path.join(DATA_DIR, "merged.csv")

# ── 1. Load APTOS labels ──────────────────────────────────────────────────────
print("Loading APTOS CSV...")
aptos = pd.read_csv(APTOS_CSV)           # columns: id_code, diagnosis
print(f"  APTOS rows: {len(aptos)}")

# Binary label: DR grade >= 2 → complication risk
aptos["complication_label"] = (aptos["diagnosis"] >= 2).astype(int)

# Assign a numeric patient_id (0-indexed) so dataset.py can find images
# The image filename is kept as the id_code; we also write a lookup column
aptos = aptos.reset_index(drop=True)
aptos["patient_id"] = aptos.index          # 0, 1, 2, ...
aptos["dr_grade"]   = aptos["diagnosis"]

# Rename images to numeric IDs so dataset.py can find them by patient_id
print("\nRenaming images to numeric IDs...")
renamed, skipped = 0, 0
for _, row in aptos.iterrows():
    src = Path(IMAGE_DIR) / f"{row['id_code']}.png"
    if not src.exists():
        src = Path(IMAGE_DIR) / f"{row['id_code']}.jpg"
    dst = Path(IMAGE_DIR) / f"{int(row['patient_id'])}.jpg"

    if src.exists() and not dst.exists():
        src.rename(dst)
        renamed += 1
    elif dst.exists():
        skipped += 1  # already renamed
    else:
        skipped += 1  # image missing (shouldn't happen)

print(f"  Renamed: {renamed} | Skipped/missing: {skipped}")

# ── 2. Load synthetic EHR ─────────────────────────────────────────────────────
print("\nLoading synthetic EHR CSV...")
ehr = pd.read_csv(SYNTHETIC_CSV)          # 2000 rows, numeric patient_id
print(f"  EHR rows: {len(ehr)}")

# ── 3. Merge ──────────────────────────────────────────────────────────────────
# Strategy: for each APTOS patient, sample an EHR row with matching DR grade.
# This gives each image a realistic set of clinical features.
print("\nMerging datasets by DR grade...")

ehr_by_grade = {g: ehr[ehr["dr_grade"] == g].reset_index(drop=True)
                for g in range(5)}

merged_rows = []
for _, aptos_row in aptos.iterrows():
    grade   = int(aptos_row["dr_grade"])
    pool    = ehr_by_grade.get(grade, ehr)   # fallback to full EHR if grade pool empty
    ehr_row = pool.sample(1, random_state=int(aptos_row["patient_id"])).iloc[0]

    merged_row = {
        "patient_id":         int(aptos_row["patient_id"]),
        "dr_grade":           grade,
        "complication_label": int(aptos_row["complication_label"]),
        # Clinical features from matched EHR row
        "age":                        ehr_row["age"],
        "diabetes_duration_years":    ehr_row["diabetes_duration_years"],
        "hba1c":                      ehr_row["hba1c"],
        "fasting_blood_sugar":        ehr_row["fasting_blood_sugar"],
        "systolic_bp":                ehr_row["systolic_bp"],
        "diastolic_bp":               ehr_row["diastolic_bp"],
        "bmi":                        ehr_row["bmi"],
        "serum_creatinine":           ehr_row["serum_creatinine"],
        "ldl_cholesterol":            ehr_row["ldl_cholesterol"],
        "hdl_cholesterol":            ehr_row["hdl_cholesterol"],
        "triglycerides":              ehr_row["triglycerides"],
        "gender":                     ehr_row["gender"],
        "smoker":                     ehr_row["smoker"],
        "hypertension":               ehr_row["hypertension"],
        "on_insulin":                 ehr_row["on_insulin"],
        "family_history":             ehr_row["family_history"],
        "rural_urban":                ehr_row["rural_urban"],
    }
    merged_rows.append(merged_row)

merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv(MERGED_CSV, index=False)

# ── 4. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Merged dataset saved → {MERGED_CSV}")
print(f"  Total samples : {len(merged_df)}")
print(f"  Complication=1: {merged_df['complication_label'].sum()} "
      f"({merged_df['complication_label'].mean()*100:.1f}%)")
print(f"  Complication=0: {(merged_df['complication_label']==0).sum()}")
print(f"\n  DR Grade distribution:")
print(merged_df["dr_grade"].value_counts().sort_index().to_string())
print(f"{'='*50}\n")
print("✅ Done! Now train with:  python training/train.py --csv data/merged.csv")
