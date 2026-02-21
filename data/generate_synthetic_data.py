"""
data/generate_synthetic_data.py
Generates a realistic synthetic EHR dataset of diabetic patients.

Features are statistically correlated with DR grade so the fusion model
has genuine cross-modal signal to learn from.

Usage: python data/generate_synthetic_data.py
Output: data/synthetic_ehr.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SYNTHETIC_CSV, NUM_SYNTHETIC_PATIENTS, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def generate_patient_cohort(n: int = NUM_SYNTHETIC_PATIENTS) -> pd.DataFrame:
    """
    Generate n synthetic diabetic patients with features correlated to DR grade.
    DR grade drives HbA1c, disease duration, blood pressure, and complication risk.
    """
    records = []

    for patient_id in range(n):
        dr_grade     = np.random.choice([0, 1, 2, 3, 4], p=[0.49, 0.07, 0.15, 0.10, 0.19])
        complication = int(dr_grade >= 2)
        sev          = dr_grade / 4.0  # severity scaler [0, 1]

        age                 = int(np.clip(np.random.normal(52 + sev * 8, 10), 25, 80))
        diabetes_duration   = float(np.clip(np.random.normal(7 + sev * 8, 4), 0.5, 30))
        hba1c               = float(np.clip(np.random.normal(7.5 + sev * 3.5, 1.2), 5.5, 14.0))
        fasting_blood_sugar = float(np.clip(np.random.normal(130 + sev * 80, 30), 70, 400))
        systolic_bp         = float(np.clip(np.random.normal(128 + sev * 20, 15), 90, 200))
        diastolic_bp        = float(np.clip(systolic_bp * 0.62 + np.random.normal(0, 5), 60, 120))
        bmi                 = float(np.clip(np.random.normal(26.5 + sev * 2, 4), 16, 45))
        serum_creatinine    = float(np.clip(np.random.normal(1.0 + sev * 1.2, 0.4), 0.5, 8.0))
        ldl                 = float(np.clip(np.random.normal(110 + sev * 20, 30), 40, 250))
        hdl                 = float(np.clip(np.random.normal(45 - sev * 8, 10), 20, 90))
        triglycerides       = float(np.clip(np.random.normal(150 + sev * 60, 50), 50, 600))

        gender         = int(np.random.choice([0, 1], p=[0.44, 0.56]))
        smoker         = int(np.random.choice([0, 1], p=[0.80, 0.20]))
        hypertension   = int(np.random.choice([0, 1], p=[0.45 - sev * 0.2, 0.55 + sev * 0.2]))
        on_insulin     = int(np.random.choice([0, 1], p=[0.70 - sev * 0.3, 0.30 + sev * 0.3]))
        family_history = int(np.random.choice([0, 1], p=[0.60, 0.40]))
        rural_urban    = int(np.random.choice([0, 1], p=[0.40, 0.60]))

        records.append({
            "patient_id":              patient_id,
            "dr_grade":                dr_grade,
            "complication_label":      complication,
            "age":                     age,
            "diabetes_duration_years": diabetes_duration,
            "hba1c":                   hba1c,
            "fasting_blood_sugar":     fasting_blood_sugar,
            "systolic_bp":             systolic_bp,
            "diastolic_bp":            diastolic_bp,
            "bmi":                     bmi,
            "serum_creatinine":        serum_creatinine,
            "ldl_cholesterol":         ldl,
            "hdl_cholesterol":         hdl,
            "triglycerides":           triglycerides,
            "gender":                  gender,
            "smoker":                  smoker,
            "hypertension":            hypertension,
            "on_insulin":              on_insulin,
            "family_history":          family_history,
            "rural_urban":             rural_urban,
        })

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"  Total: {len(df)} | Complication=1: {df['complication_label'].sum()} ({df['complication_label'].mean()*100:.1f}%)")
    print(f"  Missing values: {df.isnull().sum().sum()} cells")
    print(f"\n  DR Grade distribution:\n{df['dr_grade'].value_counts().sort_index().to_string()}")
    print(f"{'='*55}\n")


def main():
    print("Generating synthetic EHR data...")
    df = generate_patient_cohort(NUM_SYNTHETIC_PATIENTS)
    df.to_csv(SYNTHETIC_CSV, index=False)
    print_summary(df)
    print(f"Saved → {SYNTHETIC_CSV}")


if __name__ == "__main__":
    main()
