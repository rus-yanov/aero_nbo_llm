from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Пути
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

COMMON_PATH = DATA_RAW / "common_dataset.csv"
OUTPUT_PATH = DATA_PROCESSED / "ml_training_dataset.csv"


# --- Loaders ---

def load_common():
    return pd.read_csv(COMMON_PATH)


# --- Feature engineering ---

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df[df["conversion"].notna()]
    df["conversion"] = df["conversion"].astype(int)
    return df


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["treatment_date"], errors="coerce")
    df["treatment_dow"] = dt.dt.dayofweek
    df["treatment_month"] = dt.dt.month
    return df


def filter_treatment(df: pd.DataFrame, only_treat=True) -> pd.DataFrame:
    if "treatment" not in df.columns:
        return df
    return df[df["treatment"] == 1] if only_treat else df


# --- Main builder ---

def build_ml_dataset(common_df: pd.DataFrame) -> pd.DataFrame:
    df = common_df.copy()
    df = filter_treatment(df, only_treat=True)
    df = clean(df)
    df = add_context(df)

    # Удаляем служебные поля
    if "other" in df.columns:
        df = df.drop(columns=["other"])

    # Удаляем утечки таргета
    LEAKY_COLS = ["revenue_14d"]
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])

    return df


# --- CLI ---

def main():
    common = load_common()
    ml = build_ml_dataset(common)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ml.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()