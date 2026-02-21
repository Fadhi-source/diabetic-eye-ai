"""
data/merge_aptos.py
Merges APTOS image labels with synthetic EHR features into data/merged.csv.

Usage: python data/merge_aptos.py
Input:  data/aptos_train.csv, data/synthetic_ehr.csv, data/images/
Output: data/merged.csv
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

print("Loading APTOS CSV...")
aptos = pd.read_csv(APTOS_CSV)
print(f"  APTOS rows: {len(aptos)}")

aptos["complication_label"] = (aptos["diagnosis"] >= 2).astype(int)
aptos = aptos.reset_index(drop=True)
aptos["patient_id"] = aptos.index
aptos["dr_grade"]   = aptos["diagnosis"]

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
    else:
        skipped += 1

print(f"  Renamed: {renamed} | Skipped: {skipped}")

print("\nLoading synthetic EHR CSV...")
ehr = pd.read_csv(SYNTHETIC_CSV)
print(f"  EHR rows: {len(ehr)}")

print("\nMerging by DR grade...")
ehr_by_grade = {g: ehr[ehr["dr_grade"] == g].reset_index(drop=True) for g in range(5)}

merged_rows = []
for _, aptos_row in aptos.iterrows():
    grade   = int(aptos_row["dr_grade"])
    pool    = ehr_by_grade.get(grade, ehr)
    ehr_row = pool.sample(1, random_state=int(aptos_row["patient_id"])).iloc[0]

    merged_rows.append({
        "patient_id":              int(aptos_row["patient_id"]),
        "dr_grade":                grade,
        "complication_label":      int(aptos_row["complication_label"]),
        "age":                     ehr_row["age"],
        "diabetes_duration_years": ehr_row["diabetes_duration_years"],
        "hba1c":                   ehr_row["hba1c"],
        "fasting_blood_sugar":     ehr_row["fasting_blood_sugar"],
        "systolic_bp":             ehr_row["systolic_bp"],
        "diastolic_bp":            ehr_row["diastolic_bp"],
        "bmi":                     ehr_row["bmi"],
        "serum_creatinine":        ehr_row["serum_creatinine"],
        "ldl_cholesterol":         ehr_row["ldl_cholesterol"],
        "hdl_cholesterol":         ehr_row["hdl_cholesterol"],
        "triglycerides":           ehr_row["triglycerides"],
        "gender":                  ehr_row["gender"],
        "smoker":                  ehr_row["smoker"],
        "hypertension":            ehr_row["hypertension"],
        "on_insulin":              ehr_row["on_insulin"],
        "family_history":          ehr_row["family_history"],
        "rural_urban":             ehr_row["rural_urban"],
    })

merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv(MERGED_CSV, index=False)

print(f"\n{'='*50}")
print(f"  Saved → {MERGED_CSV}")
print(f"  Total: {len(merged_df)} | Complication=1: {merged_df['complication_label'].sum()} ({merged_df['complication_label'].mean()*100:.1f}%)")
print(f"\n  DR Grade distribution:\n{merged_df['dr_grade'].value_counts().sort_index().to_string()}")
print(f"{'='*50}")
print("✅ Done! Train with: python training/train.py --csv data/merged.csv")
