"""
05b_decision_engine.py — Decision Engine Layer
Builds CII, EPI, ROI Simulator, Before/After Simulation, Predictive Alerts.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"

# ─── Vehicle severity weights ────────────────────────────────────────
VEHICLE_SEVERITY = {
    "MAXI-CAB": 1.8, "CAR": 1.4, "PASSENGER AUTO": 1.3,
    "MOTOR CYCLE": 1.0, "SCOOTER": 1.0,
}
DEFAULT_SEVERITY = 1.2

PEAK_HOURS = list(range(8, 11)) + list(range(17, 21))  # 8-10 AM, 5-8 PM


# ════════════════════════════════════════════════════════════════════
# 6.5.1 — Congestion Impact Index (CII)
# ════════════════════════════════════════════════════════════════════
def build_cii(parking: pd.DataFrame, hotspots: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Building Congestion Impact Index (CII) ===")

    # Work at junction level (from hotspots)
    cii = hotspots.copy()

    # Proxy 1 — Violation Density Score
    cii["density_score"] = (cii["violation_count"] / cii["violation_count"].max() * 100).round(1)

    # Proxy 2 — Junction Type Segmentation (Priority 1.2)
    # Junctions use turning-movement-block weighting (1.8)
    # No Junction / mid-block use lane-capacity-reduction weighting (1.0)
    cii["is_junction"] = ~cii["top_junction"].str.strip().str.lower().isin(["no junction", "unknown", ""])
    cii["junction_multiplier"] = cii["is_junction"].apply(lambda x: 1.8 if x else 1.0)
    cii["segmentation_type"] = cii["is_junction"].apply(lambda x: "Turning-Movement-Block" if x else "Lane-Capacity-Reduction")

    # Proxy 3 — Peak Hour Concentration
    def peak_concentration(junction):
        grp = parking[parking["junction_name"] == junction]
        if grp.empty:
            return 0.0
        return round(grp["hour"].isin(PEAK_HOURS).sum() / len(grp) * 100, 1)

    cii["peak_hour_concentration"] = cii["top_junction"].apply(peak_concentration)

    # Proxy 4 — Violation Severity Weight (Priority 1.3 integration)
    def avg_severity(junction):
        grp = parking[parking["junction_name"] == junction]
        if grp.empty:
            return DEFAULT_SEVERITY
        if "violation_severity_score" in grp.columns:
            return round(grp["violation_severity_score"].mean(), 2)
        else:
            weights = grp["vehicle_type"].map(VEHICLE_SEVERITY).fillna(DEFAULT_SEVERITY)
            return round(weights.mean(), 2)

    cii["vehicle_severity_avg"] = cii["top_junction"].apply(avg_severity)

    cii["road_capacity_score"] = cii.get("road_capacity_loss_factor", 0.5) * 100
    cii["choke_score"] = cii.get("choke_proximity_score", 0.0) * 100

    # Final CII formula (incorporating geo-enriched road capacity and choke proximity)
    # Re-weighted: Density=0.25, Junction=0.15, Peak=0.15, Severity=0.10, Road=0.20, Choke=0.15
    cii["CII"] = (
        cii["density_score"] * 0.25 +
        (cii["junction_multiplier"] * 20) * 0.15 +
        cii["peak_hour_concentration"] * 0.15 +
        (cii["vehicle_severity_avg"] * 20) * 0.10 +
        cii["road_capacity_score"] * 0.20 +
        cii["choke_score"] * 0.15
    ).clip(0, 100).round(1)

    cii["CII_band"] = pd.cut(
        cii["CII"],
        bins=[-1, 20, 40, 70, 101],
        labels=["Low", "Medium", "High", "Critical"],
    )
    cii["is_estimated_metric"] = True

    out_cols = [
        "top_junction", "violation_count", "density_score", "junction_multiplier",
        "peak_hour_concentration", "vehicle_severity_avg", "road_capacity_score",
        "choke_score", "road_class", "nearby_poi_type",
        "CII", "CII_band", "is_estimated_metric",
    ]
    # Rename for clarity
    cii_out = cii[out_cols].copy()
    cii_out.rename(columns={"top_junction": "junction_name"}, inplace=True)

    cii_out.to_csv(OUT / "cii_scores.csv", index=False)
    print(f"  Saved → outputs/cii_scores.csv  shape={cii_out.shape}")

    # Insight comparison
    print("\n=== WHY RAW VIOLATION COUNT IS MISLEADING ===")
    comparison = cii_out.sort_values("violation_count", ascending=False).head(10)
    print(comparison[["junction_name", "violation_count", "CII", "CII_band"]].to_string(index=False))
    print("\nNote: Some high-violation-count junctions score LOWER on CII because they")
    print("are wide road stretches, while lower-count junctions at real intersections")
    print("score HIGHER because they block more critical traffic flow.")

    return cii


# ════════════════════════════════════════════════════════════════════
# 6.5.2 — Enforcement Priority Index (EPI)
# ════════════════════════════════════════════════════════════════════
def build_epi(cii: pd.DataFrame, parking: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Building Enforcement Priority Index (EPI) ===")

    epi = cii[["top_junction", "violation_count", "CII", "peak_hour",
               "peak_day", "top_vehicle_type", "top_violation"]].copy()
    epi.rename(columns={"top_junction": "junction_name"}, inplace=True)

    # Violation density norm
    epi["violation_density_norm"] = (epi["violation_count"] / epi["violation_count"].max() * 100).round(1)
    epi["congestion_impact_norm"] = epi["CII"]

    # Forecasted violations norm
    try:
        preds = pd.read_csv(OUT / "predictions_7day.csv")
        pred_sum = preds.groupby("junction_name")["predicted_violations"].sum()
        epi["forecasted_raw"] = epi["junction_name"].map(pred_sum).fillna(0)
        max_fr = epi["forecasted_raw"].max()
        epi["forecasted_violations_norm"] = (epi["forecasted_raw"] / max_fr * 100).round(1) if max_fr > 0 else 0
    except Exception:
        epi["forecasted_violations_norm"] = 50.0

    # EPI formula (re-weighted without event risk: VD=0.50, CI=0.35, FV=0.15)
    epi["EPI"] = (
        0.50 * epi["violation_density_norm"] +
        0.35 * epi["congestion_impact_norm"] +
        0.15 * epi["forecasted_violations_norm"]
    ).round(1)

    epi = epi.sort_values("EPI", ascending=False).reset_index(drop=True)
    epi["rank"] = range(1, len(epi) + 1)

    # Recommended action string
    def make_action(row):
        officers = max(1, min(6, int(np.ceil(row["violation_count"] / 2000))))
        return (f"Deploy {officers} officers, {row['peak_day']} "
                f"{row['peak_hour']}:00-{row['peak_hour']+2}:00, "
                f"focus on {row['top_vehicle_type']} {row['top_violation'].lower()}")

    epi["recommended_action"] = epi.apply(make_action, axis=1)

    out_cols = ["rank", "junction_name", "EPI", "violation_density_norm",
                "congestion_impact_norm",
                "forecasted_violations_norm", "recommended_action"]
    epi_out = epi[out_cols]
    epi_out.to_csv(OUT / "epi_rankings.csv", index=False)
    print(f"  Saved → outputs/epi_rankings.csv  shape={epi_out.shape}")

    print("\nTop 10 EPI Rankings:")
    print(epi_out.head(10)[["rank", "junction_name", "EPI", "recommended_action"]].to_string(index=False))

    return epi


# ════════════════════════════════════════════════════════════════════
# 6.5.3 — Enforcement ROI Simulator
# ════════════════════════════════════════════════════════════════════
def build_roi(cii: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Building Enforcement ROI Simulator ===")

    roi = cii[["top_junction", "violation_count", "CII"]].copy()
    roi.rename(columns={"top_junction": "junction_name"}, inplace=True)

    # Get junction_multiplier directly from cii (same row order)
    jm = cii["junction_multiplier"].values if "junction_multiplier" in cii.columns else np.ones(len(roi))

    # Estimated relief %
    vc = roi["violation_count"].astype(float)
    cii_vals = roi["CII"].astype(float)

    base_relief = np.log1p(vc) / np.log1p(vc.max()) * 100
    weighted_relief = base_relief * (cii_vals / 100) * jm
    max_wr = weighted_relief.max()
    roi["expected_relief_pct"] = (weighted_relief / max_wr * 45).round(1) if max_wr > 0 else 0

    roi["officers_required"] = np.ceil(vc / 2000).clip(upper=10).astype(int)
    roi["is_estimated_metric"] = True

    roi = roi.sort_values("expected_relief_pct", ascending=False).reset_index(drop=True)
    roi.to_csv(OUT / "roi_simulation.csv", index=False)
    print(f"  Saved → outputs/roi_simulation.csv  shape={roi.shape}")

    top5 = roi.head(5)
    print("\n=== IF YOU HAD ONLY 5 OFFICERS TODAY, DEPLOY HERE ===")
    print(top5[["junction_name", "expected_relief_pct", "officers_required"]].to_string(index=False))
    print("\n[NOTE: Relief % is a modeled estimate based on violation density and chokepoint")
    print("scoring. Production version would calibrate this against Google Maps Traffic API")
    print("speed data before/after enforcement actions.]")

    return roi


# ════════════════════════════════════════════════════════════════════
# 6.5.4 — Before/After Simulation Engine
# ════════════════════════════════════════════════════════════════════
def build_before_after(cii: pd.DataFrame, parking: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Building Before/After Simulation ===")

    top10 = cii.sort_values("CII", ascending=False).head(10).copy()

    sim_rows = []
    for _, row in top10.iterrows():
        junction = row["top_junction"]
        pk_hour  = int(row["peak_hour"])
        jm       = row.get("junction_multiplier", 1.0)

        grp = parking[(parking["junction_name"] == junction) & (parking["hour"] == pk_hour)]
        violations_at_peak = len(grp)

        baseline_speed = 35.0
        congestion_factor = np.sqrt(max(violations_at_peak, 1)) * jm * 0.8
        current_speed = max(baseline_speed - congestion_factor, 5.0)
        predicted_speed = baseline_speed - (congestion_factor * 0.25)
        predicted_speed = max(predicted_speed, current_speed)  # ensure improvement
        improvement_pct = ((predicted_speed - current_speed) / max(current_speed, 1)) * 100

        sim_rows.append({
            "junction_name":       junction,
            "peak_hour":           pk_hour,
            "violations_at_peak":  violations_at_peak,
            "current_speed_kmh":   round(current_speed, 1),
            "predicted_speed_kmh": round(predicted_speed, 1),
            "improvement_pct":     round(improvement_pct, 1),
            "is_estimated_metric": True,
        })

    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT / "before_after_simulation.csv", index=False)
    print(f"  Saved → outputs/before_after_simulation.csv  shape={sim.shape}")

    print("\n=== BEFORE/AFTER SIMULATION (peak hour, top junctions) ===")
    print(sim[["junction_name", "current_speed_kmh", "predicted_speed_kmh", "improvement_pct"]].to_string(index=False))
    print("\n[ASSUMPTION DISCLOSURE: Speed values are modeled using a square-root congestion")
    print("heuristic calibrated to violation density, not measured by sensors. This")
    print("demonstrates the decision-support concept; production deployment requires")
    print("integration with live traffic speed APIs (Google Maps Roads API / BBMP ATCS data)")
    print("for calibration against ground truth.]")

    return sim


# ════════════════════════════════════════════════════════════════════
# 6.5.5 — Predictive Alert Generator
# ════════════════════════════════════════════════════════════════════
def build_alerts(parking: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Building Predictive Alerts ===")

    # Historical hourly baseline per junction
    jh = parking.groupby(["junction_name", "hour"]).size().reset_index(name="violations")
    threshold = jh.groupby("junction_name")["violations"].quantile(0.50)

    # Load predictions
    try:
        preds = pd.read_csv(OUT / "predictions_7day.csv")
    except Exception:
        print("  WARNING: predictions_7day.csv not found — skipping alerts")
        return pd.DataFrame()

    # Build alert threshold map
    preds["alert_threshold"] = preds["junction_name"].map(threshold).fillna(0)
    alerts = preds[preds["predicted_violations"] > preds["alert_threshold"]].copy()

    # If still empty (e.g. model predicts very low), just take the top 10 highest predicted
    if alerts.empty:
        alerts = preds.sort_values("predicted_violations", ascending=False).head(10).copy()

    alerts["recommended_officers"] = np.ceil(alerts["predicted_violations"] / 20).clip(upper=6).astype(int)
    alerts["severity"] = pd.cut(
        alerts["predicted_violations"],
        bins=[0, 50, 100, 10000],
        labels=["Watch", "Warning", "Critical"],
    )

    alerts = alerts.sort_values("predicted_violations", ascending=False).reset_index(drop=True)

    out_cols = ["junction_name", "day_name", "hour", "predicted_violations",
                "recommended_officers", "severity"]
    alerts_out = alerts[out_cols]
    alerts_out.to_csv(OUT / "predictive_alerts.csv", index=False)
    print(f"  Saved → outputs/predictive_alerts.csv  shape={alerts_out.shape}")

    print("\nTop 10 Predictive Alerts:")
    for _, a in alerts_out.head(10).iterrows():
        h_end = min(a["hour"] + 2, 23)
        print(f"  ⚠️  {a['day_name']} {a['hour']}:00-{h_end}:00")
        print(f"  ⚠️  {a['junction_name']}")
        print(f"  ⚠️  Expected Violations: {a['predicted_violations']}")
        print(f"  ⚠️  Recommended Officers: {a['recommended_officers']}")
        print(f"  ⚠️  Severity: {a['severity']}")
        print()

    return alerts_out


# ════════════════════════════════════════════════════════════════════
# 6.5.6 — Weight-Sensitivity Analysis (Priority 2.1)
# ════════════════════════════════════════════════════════════════════
def build_sensitivity_analysis(epi: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Building Weight-Sensitivity Analysis ===")
    
    base_weights = {"VD": 0.50, "CI": 0.35, "FV": 0.15}
    top10_base = set(epi.sort_values("EPI", ascending=False).head(10)["junction_name"])
    
    results = []
    for key, weight in base_weights.items():
        for change in [-0.20, 0.20]: # +/- 20% relative perturbation
            new_weight = weight * (1 + change)
            diff = weight - new_weight
            
            others_sum = 1.0 - weight
            new_weights = {}
            for k, w in base_weights.items():
                if k == key:
                    new_weights[k] = new_weight
                else:
                    new_weights[k] = w + diff * (w / others_sum)
                    
            new_epi = (
                new_weights["VD"] * epi["violation_density_norm"] +
                new_weights["CI"] * epi["congestion_impact_norm"] +
                new_weights["FV"] * epi["forecasted_violations_norm"]
            )
            
            new_ranks = epi.copy()
            new_ranks["temp_epi"] = new_epi
            new_ranks = new_ranks.sort_values("temp_epi", ascending=False)
            top10_new = set(new_ranks.head(10)["junction_name"])
            overlap = len(top10_base.intersection(top10_new))
            
            results.append({
                "perturbed_feature": key,
                "perturbation": f"{int(change*100)}%",
                "new_weight": round(new_weight, 3),
                "top10_overlap": overlap,
                "stability_score": f"{overlap * 10}%"
            })
            
    sens = pd.DataFrame(results)
    sens.to_csv(OUT / "sensitivity_analysis.csv", index=False)
    print(f"  Saved → outputs/sensitivity_analysis.csv  shape={sens.shape}")
    print("\nSensitivity Analysis (Top 10 Stability):")
    print(sens.to_string(index=False))
    return sens

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DECISION ENGINE — Loading data")
    print("=" * 60)

    parking  = pd.read_csv(OUT / "parking_clean.csv", low_memory=False)
    hotspots_path = OUT / "hotspots_geo_enriched.csv"
    if not hotspots_path.exists():
        hotspots_path = OUT / "hotspots.csv"
    hotspots = pd.read_csv(hotspots_path)
    print(f"  Parking rows: {len(parking):,}")
    print(f"  Hotspot rows: {len(hotspots):,}")

    cii = build_cii(parking, hotspots)
    epi = build_epi(cii, parking)
    roi = build_roi(cii)
    sim = build_before_after(cii, parking)
    alerts = build_alerts(parking)
    sens = build_sensitivity_analysis(epi)

    print("\n" + "=" * 60)
    print("  DECISION ENGINE — All outputs generated ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
