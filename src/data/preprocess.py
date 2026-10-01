from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET = "readmitted_30d"
ID_COLUMN = "patient_id"

FEATURE_COLUMNS = [
    "age",
    "length_of_stay",
    "prior_admissions",
    "medication_count",
    "comorbidity_count",
    "heart_rate",
    "systolic_bp",
    "oxygen_saturation",
    "creatinine",
    "glucose",
    "hemoglobin",
    "has_diabetes",
    "has_copd",
    "has_heart_failure",
    "has_renal_disease",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep this pipeline deterministic and simple for the capstone prototype.
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    return X, y
