"""
06_generate_dashboard_data.py — Consolidate all outputs into a single JSON for the dashboard.
"""
import pandas as pd
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"

DAY_MAP = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
           4: "Friday", 5: "Saturday", 6: "Sunday"}


def safe_load(name, **kwargs):
    path = OUT / name
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"  WARNING: could not load {name}: {e}")
        return pd.DataFrame()


def safe_load_json(name):
    path = OUT / name
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: could not load {name}: {e}")
        return {}

def main():
    print("Loading all output CSVs and JSONs ...")
    parking     = safe_load("parking_clean.csv", low_memory=False)
    hotspots    = safe_load("hotspots.csv")
    enforcement = safe_load("enforcement_priority.csv")
    jh          = safe_load("junction_hourly.csv")
    preds       = safe_load("predictions_7day.csv")
    cii         = safe_load("cii_scores.csv")
    epi         = safe_load("epi_rankings.csv")
    roi         = safe_load("roi_simulation.csv")
    ba_sim      = safe_load("before_after_simulation.csv")
    alerts      = safe_load("predictive_alerts.csv")
    fi          = safe_load("feature_importance.csv")
    metrics     = safe_load_json("model_metrics.json")
    center_val  = safe_load("center_validation.csv")
    sens        = safe_load("sensitivity_analysis.csv")
    baseline    = safe_load("baseline_comparison.csv")
    enf_mode    = safe_load("enforcement_mode.csv")
    deploy      = safe_load("deployment_schedule.csv")

    # ── Meta ──────────────────────────────────────────────────────
    dates = parking["date"] if "date" in parking.columns else pd.Series()
    meta = {
        "total_violations": int(len(parking)),
        "date_range": {
            "start": str(dates.min()) if not dates.empty else "",
            "end":   str(dates.max()) if not dates.empty else "",
        },
        "total_hotspots":   int(len(hotspots)),
        "police_stations":  int(parking["police_station"].nunique()) if "police_station" in parking.columns else 0,
        "junctions":        int(parking["junction_name"].nunique()) if "junction_name" in parking.columns else 0,
    }

    # ── Hotspots ──────────────────────────────────────────────────
    hotspots_list = []
    for i, row in hotspots.head(50).iterrows():
        hotspots_list.append({
            "id":              i + 1,
            "lat":             row.get("centroid_lat", 0),
            "lon":             row.get("centroid_lon", 0),
            "junction":        row.get("top_junction", ""),
            "risk_score":      row.get("risk_score", 0),
            "violation_count": int(row.get("violation_count", 0)),
            "top_violation":   row.get("top_violation", ""),
            "top_vehicle":     row.get("top_vehicle_type", ""),
            "peak_hour":       int(row.get("peak_hour", 0)),
            "peak_day":        row.get("peak_day", ""),
            "road_class":      row.get("road_class", ""),
            "road_capacity_loss_factor": float(row.get("road_capacity_loss_factor", 0)),
            "nearby_poi_type": row.get("nearby_poi_type", ""),
            "choke_proximity_score": float(row.get("choke_proximity_score", 0)),
        })

    # ── Vehicle breakdown ──────────────────────────────────────────
    vb = {}
    if "vehicle_type" in parking.columns:
        vb = parking["vehicle_type"].value_counts().head(10).to_dict()
        vb = {k: int(v) for k, v in vb.items()}

    # ── Daily trend ───────────────────────────────────────────────
    daily = []
    if "date" in parking.columns:
        dt = parking.groupby("date").size().reset_index(name="violations")
        dt = dt.sort_values("date")
        for _, r in dt.iterrows():
            daily.append({"date": str(r["date"]), "violations": int(r["violations"])})

    # ── Enforcement priority ──────────────────────────────────────
    enf_list = []
    for _, r in enforcement.head(20).iterrows():
        enf_list.append({k: (int(v) if isinstance(v, (np.integer,)) else
                             float(v) if isinstance(v, (np.floating,)) else str(v) if pd.notnull(v) else None)
                         for k, v in r.to_dict().items()})

    # ── Hourly pattern ────────────────────────────────────────────
    hourly = []
    for _, r in jh.iterrows():
        hourly.append({
            "junction_name": r.get("junction_name", ""),
            "hour":          int(r.get("hour", 0)),
            "violations":    int(r.get("violations", 0)),
        })

    # ── Predictions ───────────────────────────────────────────────
    preds_list = []
    if not preds.empty:
        top5_j = preds["junction_name"].value_counts().head(5).index.tolist()
        for _, r in preds[preds["junction_name"].isin(top5_j)].iterrows():
            preds_list.append({
                "junction_name":       r.get("junction_name", ""),
                "date":                r.get("date", ""),
                "day_name":            r.get("day_name", ""),
                "hour":                int(r.get("hour", 0)),
                "predicted_violations": int(r.get("predicted_violations", 0)),
            })

    # ── CII scores ────────────────────────────────────────────────
    enf_mode_map = {}
    if not enf_mode.empty:
        for _, r in enf_mode.iterrows():
            enf_mode_map[r["junction_name"]] = {
                "enforcement_mode": r["enforcement_mode"],
                "reason_string": r["reason_string"]
            }

    cii_list = []
    for _, r in cii.iterrows():
        j_name = r.get("junction_name", "")
        cii_list.append({
            "junction":        j_name,
            "violation_count": int(r.get("violation_count", 0)),
            "density_score":   float(r.get("density_score", 0)),
            "junction_multiplier": float(r.get("junction_multiplier", 1.0)),
            "peak_hour_concentration": float(r.get("peak_hour_concentration", 0)),
            "vehicle_severity_avg": float(r.get("vehicle_severity_avg", 0)),
            "road_capacity_score": float(r.get("road_capacity_score", 0)),
            "choke_score":     float(r.get("choke_score", 0)),
            "road_class":      r.get("road_class", ""),
            "nearby_poi_type": r.get("nearby_poi_type", ""),
            "CII":             float(r.get("CII", 0)),
            "CII_band":        r.get("CII_band", ""),
            "enforcement_mode": enf_mode_map.get(j_name, {}).get("enforcement_mode", "Monitor Only"),
            "enforcement_reason": enf_mode_map.get(j_name, {}).get("reason_string", ""),
            "is_estimated":    True,
        })

    # ── EPI rankings ──────────────────────────────────────────────
    epi_list = []
    for _, r in epi.iterrows():
        epi_list.append({
            "rank":                      int(r.get("rank", 0)),
            "junction":                  r.get("junction_name", ""),
            "EPI":                       float(r.get("EPI", 0)),
            "violation_density_norm":    float(r.get("violation_density_norm", 0)),
            "congestion_impact_norm":    float(r.get("congestion_impact_norm", 0)),
            "forecasted_violations_norm": float(r.get("forecasted_violations_norm", 0)),
            "recommended_action":        r.get("recommended_action", ""),
        })

    # ── ROI simulation ────────────────────────────────────────────
    roi_list = []
    for _, r in roi.iterrows():
        roi_list.append({
            "junction":           r.get("junction_name", ""),
            "violation_count":    int(r.get("violation_count", 0)),
            "CII":                float(r.get("CII", 0)),
            "expected_relief_pct": float(r.get("expected_relief_pct", 0)),
            "officers_required":  int(r.get("officers_required", 0)),
            "is_estimated":       True,
        })

    # ── Before/After ──────────────────────────────────────────────
    ba_list = []
    for _, r in ba_sim.iterrows():
        ba_list.append({
            "junction":           r.get("junction_name", ""),
            "peak_hour":          int(r.get("peak_hour", 0)),
            "violations_at_peak": int(r.get("violations_at_peak", 0)),
            "current_speed_kmh":  float(r.get("current_speed_kmh", 0)),
            "predicted_speed_kmh": float(r.get("predicted_speed_kmh", 0)),
            "improvement_pct":    float(r.get("improvement_pct", 0)),
            "is_estimated":       True,
        })

    # ── Predictive alerts ─────────────────────────────────────────
    alert_list = []
    for _, r in alerts.head(20).iterrows():
        alert_list.append({
            "junction":              r.get("junction_name", ""),
            "day_name":              r.get("day_name", ""),
            "hour":                  int(r.get("hour", 0)),
            "predicted_violations":  int(r.get("predicted_violations", 0)),
            "recommended_officers":  int(r.get("recommended_officers", 0)),
            "severity":              r.get("severity", "Watch"),
        })

    # ── Feature Importance ────────────────────────────────────────
    fi_list = []
    for _, r in fi.iterrows():
        fi_list.append({
            "feature": r.get("feature", ""),
            "importance": float(r.get("importance", 0)),
        })
        
    # ── Deployment Schedule ───────────────────────────────────────
    deploy_list = []
    if not deploy.empty:
        # Sort by date, hour, then officers needed
        deploy_sorted = deploy.sort_values(["date", "hour", "officers_needed"], ascending=[True, True, False])
        for _, r in deploy_sorted.iterrows():
            if int(r.get("officers_needed", 0)) > 0:
                deploy_list.append({
                    "junction_name": r.get("junction_name", ""),
                    "date": r.get("date", ""),
                    "day_name": r.get("day_name", ""),
                    "hour": int(r.get("hour", 0)),
                    "predicted_violations": int(r.get("predicted_violations", 0)),
                    "officers_needed": int(r.get("officers_needed", 0))
                })

    # ── Priority additions ────────────────────────────────────────
    cv_list = [r.to_dict() for _, r in center_val.iterrows()] if not center_val.empty else []
    sens_list = [r.to_dict() for _, r in sens.iterrows()] if not sens.empty else []
    baseline_list = [r.to_dict() for _, r in baseline.iterrows()] if not baseline.empty else []

    # ── Assemble ──────────────────────────────────────────────────
    dashboard = {
        "meta":                  meta,
        "hotspots":              hotspots_list,
        "enforcement_priority":  enf_list,
        "hourly_pattern":        hourly,
        "predictions":           preds_list,
        "vehicle_breakdown":     vb,
        "daily_trend":           daily,
        "cii_scores":            cii_list,
        "epi_rankings":          epi_list,
        "roi_simulation":        roi_list,
        "before_after_simulation": ba_list,
        "predictive_alerts":     alert_list,
        "deployment_schedule":   deploy_list,
        "model_metrics":         metrics,
        "feature_importance":    fi_list,
        "center_validation":     cv_list,
        "sensitivity_analysis":  sens_list,
        "baseline_comparison":   baseline_list,
    }

    out_path_json = OUT / "dashboard_data.json"
    with open(out_path_json, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, default=str)
        
    out_path_js = OUT / "dashboard_data.js"
    with open(out_path_js, "w", encoding="utf-8") as f:
        f.write("window.PARKPULSE_DATA = ")
        json.dump(dashboard, f, indent=2, default=str)
        f.write(";\n")
        
    print(f"\n  Saved → {out_path_json}")
    print(f"  Saved → {out_path_js}")
    print(f"  JSON keys: {list(dashboard.keys())}")
    print(f"  Meta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    import numpy as np   # needed for type checks in enf_list
    main()
