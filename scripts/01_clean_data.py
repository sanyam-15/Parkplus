"""
01_clean_data.py — Clean and prepare parking violation + event datasets.
"""
import pandas as pd
import ast
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT  = BASE / "outputs"


def clean_parking(path: Path) -> pd.DataFrame:
    """Load, filter, enrich the parking violation dataset."""
    print(f"Loading parking data from {path.name} ...")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"  Raw rows loaded: {len(df):,}")

    # Parse datetime
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    if "closed_date" in df.columns:
        df["action_taken_timestamp"] = pd.to_datetime(df["closed_date"], utc=True, errors="coerce")
    else:
        df["action_taken_timestamp"] = pd.NaT

    # Enforcement Lag (Priority 1.1)
    t1 = df["action_taken_timestamp"].dt.tz_localize(None)
    t2 = df["created_datetime"].dt.tz_localize(None)
    df["enforcement_lag_hours"] = (t1 - t2).dt.total_seconds() / 3600.0
    df["unactioned_violation"] = (df["validation_status"] == "approved") & df["action_taken_timestamp"].isna()

    # Time features
    df["hour"]        = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.day_name()
    df["day_num"]     = df["created_datetime"].dt.weekday          # Mon=0
    df["week"]        = df["created_datetime"].dt.isocalendar().week.fillna(0).astype(int)
    df["month"]       = df["created_datetime"].dt.month
    df["date"]        = df["created_datetime"].dt.date.astype(str)

    # Parse violation_type JSON string → list
    def safe_parse(val):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except Exception:
            return []

    df["violation_list"]    = df["violation_type"].astype(str).apply(safe_parse)
    df["primary_violation"] = df["violation_list"].apply(lambda x: x[0] if x else "UNKNOWN")
    df["violation_count_multi"] = df["violation_list"].apply(len)

    # Violation Severity Score (Priority 1.3)
    SEVERITY_MAP = {
        "PARKING NEAR ROAD CROSSING": 2.0,
        "PARKING AT CORNER": 1.8,
        "FOOTPATH PARKING": 1.5,
        "DOUBLE PARKING": 1.5,
        "WRONG PARKING": 1.2,
        "NO PARKING": 1.0
    }
    def compute_severity(vlist):
        if not vlist: return 1.0
        return sum(SEVERITY_MAP.get(str(v).upper().strip(), 1.0) for v in vlist)

    df["violation_severity_score"] = df["violation_list"].apply(compute_severity)

    # Drop fully null/unused columns (kept action_taken_timestamp for metrics)
    for col in ["description", "closed_datetime"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Filter to approved only (Priority 1.5: Downweight/exclude pending/rejected by explicit filter)
    df = df[df["validation_status"] == "approved"].copy()
    print(f"  Rows after approval filter: {len(df):,}")

    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Parking ---
    parking_path = DATA / "jan_to_may_police_violation_anonymized.csv"
    if not parking_path.exists():
        sys.exit(f"ERROR: {parking_path} not found")
    parking = clean_parking(parking_path)
    parking.to_csv(OUT / "parking_clean.csv", index=False)
    print(f"  Saved → outputs/parking_clean.csv  shape={parking.shape}")

    # Summary
    p_min = parking["date"].min()
    p_max = parking["date"].max()
    print(f"\n{'='*55}")
    print(f"PARKING: {len(parking):>7,} rows (approved)")
    print(f"Date range: {p_min} to {p_max}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
