"""
03_enforcement_priority.py — Rank police stations by enforcement priority.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"


def main():
    print("Loading parking_clean.csv ...")
    df = pd.read_csv(OUT / "parking_clean.csv", low_memory=False)
    print(f"  Rows: {len(df):,}")

    # ── Per station stats ────────────────────────────────────────
    if "unactioned_violation" not in df.columns:
        df["unactioned_violation"] = False
    if "enforcement_lag_hours" not in df.columns:
        df["enforcement_lag_hours"] = np.nan

    station = df.groupby("police_station").agg(
        total_violations   = ("police_station", "size"),
        unactioned_count   = ("unactioned_violation", "sum"),
        avg_lag_hours      = ("enforcement_lag_hours", "mean"),
        unique_officers    = ("created_by_id", "nunique"),
        unique_junctions   = ("junction_name", "nunique"),
        peak_hour          = ("hour", lambda x: x.mode().iloc[0] if not x.mode().empty else 0),
        peak_day           = ("day_of_week", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    ).reset_index()

    station["unactioned_pct"] = (station["unactioned_count"] / station["total_violations"] * 100).round(1)
    station["avg_lag_hours"] = station["avg_lag_hours"].round(1)

    station["violations_per_officer"] = (
        station["total_violations"] / station["unique_officers"]
    ).round(1)

    # Weekend & night ratios
    def weekend_ratio(name):
        grp = df[df["police_station"] == name]
        return grp["day_num"].isin([5, 6]).sum() / len(grp) if len(grp) else 0

    def night_ratio(name):
        grp = df[df["police_station"] == name]
        return grp["hour"].isin(list(range(22, 24)) + list(range(0, 6))).sum() / len(grp) if len(grp) else 0

    station["weekend_ratio"] = station["police_station"].apply(weekend_ratio).round(3)
    station["night_ratio"]   = station["police_station"].apply(night_ratio).round(3)

    # Priority score
    tv  = station["total_violations"]
    vpo = station["violations_per_officer"]
    uj  = station["unique_junctions"]
    wr  = station["weekend_ratio"]

    station["priority_score"] = (
        (tv / tv.max()) * 40 +
        (vpo / vpo.max()) * 30 +
        (uj / uj.max()) * 20 +
        wr * 10
    ).round(1)

    station = station.sort_values("priority_score", ascending=False).reset_index(drop=True)
    station["priority_rank"] = range(1, len(station) + 1)
    station["recommended_officers"] = np.ceil(vpo / 50).clip(upper=10).astype(int)

    station.to_csv(OUT / "enforcement_priority.csv", index=False)
    print(f"  Saved → outputs/enforcement_priority.csv  shape={station.shape}")

    # ── Hourly patterns for top 10 junctions ─────────────────────
    top_junctions = df["junction_name"].value_counts().head(10).index.tolist()
    jh = (
        df[df["junction_name"].isin(top_junctions)]
        .groupby(["junction_name", "hour"])
        .size()
        .reset_index(name="violations")
    )
    jh.to_csv(OUT / "junction_hourly.csv", index=False)
    print(f"  Saved → outputs/junction_hourly.csv  shape={jh.shape}")

    # Print
    print(f"\nEnforcement Priority Ranking (top 15):")
    cols = ["priority_rank", "police_station", "priority_score", "total_violations",
            "unactioned_pct", "avg_lag_hours", "violations_per_officer"]
    print(station[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
