"""
05c_enforcement_mode.py — Fixed-vs-Mobile Enforcement Classifier
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

def main():
    print("Loading data for enforcement mode classification ...")
    parking_path = OUT / "parking_clean.csv"
    if not parking_path.exists():
        print(f"ERROR: {parking_path} not found")
        return
        
    parking = pd.read_csv(parking_path, low_memory=False)
    
    cii_path = OUT / "cii_scores.csv"
    if not cii_path.exists():
        print(f"ERROR: {cii_path} not found")
        return
        
    cii = pd.read_csv(cii_path)
    
    total_days = parking["date"].nunique()
    
    # Calculate chronicity and time-variance per junction
    # We group by junction_name in parking
    junction_stats = parking.groupby("junction_name").agg(
        unique_days=("date", "nunique"),
        hour_std=("hour", "std")
    ).reset_index()
    
    junction_stats["chronicity"] = junction_stats["unique_days"] / total_days
    
    # Merge with cii scores
    df = cii[["junction_name", "CII"]].copy()
    df = df.merge(junction_stats, on="junction_name", how="left")
    df["hour_std"] = df["hour_std"].fillna(0)
    df["chronicity"] = df["chronicity"].fillna(0)
    
    # Determine top quartile of CII
    cii_threshold = df["CII"].quantile(0.75) if not df.empty else 0
    
    results = []
    for _, row in df.iterrows():
        j = row["junction_name"]
        chronicity = row["chronicity"]
        hr_std = row["hour_std"]
        c_score = row["CII"]
        
        # Rule-based classification
        if chronicity > 0.6 and hr_std < 4.0:
            mode = "Fixed ANPR Camera / Signage Candidate"
            reason = f"Chronic daily pattern ({chronicity*100:.0f}% days), highly clustered times (std {hr_std:.1f}h) — fixed camera more cost-effective than patrol"
        elif c_score >= cii_threshold:
            mode = "Mobile Patrol Candidate"
            reason = f"High congestion impact (CII {c_score}) but variable timing/frequency — prioritize for mobile patrol routing"
        else:
            mode = "Monitor Only"
            reason = f"Low/moderate congestion impact (CII {c_score}), low chronicity ({chronicity*100:.0f}% days) — monitor via analytics"
            
        results.append({
            "junction_name": j,
            "enforcement_mode": mode,
            "reason_string": reason
        })
        
    out_df = pd.DataFrame(results)
    out_path = OUT / "enforcement_mode.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path} shape={out_df.shape}")
    
    print("\nEnforcement Mode Summary:")
    print(out_df["enforcement_mode"].value_counts())

if __name__ == "__main__":
    main()
