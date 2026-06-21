"""
02_hotspot_clustering.py — Hybrid clustering: named junctions + DBSCAN for unnamed locations.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"


def main():
    print("Loading parking_clean.csv ...")
    df = pd.read_csv(OUT / "parking_clean.csv", low_memory=False)
    print(f"  Rows: {len(df):,}")

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # ── Strategy: use named junctions as natural clusters, then DBSCAN for "No Junction" ──
    named = df[df["junction_name"] != "No Junction"].copy()
    unnamed = df[df["junction_name"] == "No Junction"].copy()
    print(f"  Named junction violations: {len(named):,}")
    print(f"  Unnamed (No Junction):     {len(unnamed):,}")

    records = []
    cluster_id = 0

    # --- Named junctions: each junction = one hotspot ---
    for junction, grp in named.groupby("junction_name"):
        if len(grp) < 10:  # skip tiny junctions
            continue
        cc_mean = grp["center_count"].mean() if "center_count" in grp.columns and not grp["center_count"].isna().all() else 0
        records.append({
            "cluster_id":         cluster_id,
            "centroid_lat":       round(grp["latitude"].mean(), 6),
            "centroid_lon":       round(grp["longitude"].mean(), 6),
            "violation_count":    len(grp),
            "center_count_mean":  round(cc_mean, 2),
            "top_junction":       junction,
            "top_police_station": grp["police_station"].mode().iloc[0] if not grp["police_station"].mode().empty else "Unknown",
            "top_vehicle_type":   grp["vehicle_type"].mode().iloc[0] if not grp["vehicle_type"].mode().empty else "Unknown",
            "top_violation":      grp["primary_violation"].mode().iloc[0] if not grp["primary_violation"].mode().empty else "Unknown",
            "peak_hour":          int(grp["hour"].mode().iloc[0]) if not grp["hour"].mode().empty else 0,
            "peak_day":           grp["day_of_week"].mode().iloc[0] if not grp["day_of_week"].mode().empty else "Unknown",
        })
        cluster_id += 1

    # --- Unnamed: DBSCAN on sampled subset ---
    SAMPLE_SIZE = 15000
    if len(unnamed) > SAMPLE_SIZE:
        sample = unnamed.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        sample = unnamed.copy()

    coords_rad = np.radians(sample[["latitude", "longitude"]].values)
    EPS_500M = 0.5 / 6371
    print("Running DBSCAN on unnamed locations ...")
    db = DBSCAN(eps=EPS_500M, min_samples=10, metric="haversine", algorithm="ball_tree")
    sample["cluster"] = db.fit_predict(coords_rad)

    for cid, grp in sample[sample["cluster"] != -1].groupby("cluster"):
        if len(grp) < 10:
            continue
        # Try to find nearest named landmark via police_station
        ps = grp["police_station"].mode().iloc[0] if not grp["police_station"].mode().empty else "Unknown"
        label = f"Cluster near {ps}"
        cc_mean = grp["center_count"].mean() if "center_count" in grp.columns and not grp["center_count"].isna().all() else 0
        records.append({
            "cluster_id":         cluster_id,
            "centroid_lat":       round(grp["latitude"].mean(), 6),
            "centroid_lon":       round(grp["longitude"].mean(), 6),
            "violation_count":    len(grp),
            "center_count_mean":  round(cc_mean, 2),
            "top_junction":       label,
            "top_police_station": ps,
            "top_vehicle_type":   grp["vehicle_type"].mode().iloc[0] if not grp["vehicle_type"].mode().empty else "Unknown",
            "top_violation":      grp["primary_violation"].mode().iloc[0] if not grp["primary_violation"].mode().empty else "Unknown",
            "peak_hour":          int(grp["hour"].mode().iloc[0]) if not grp["hour"].mode().empty else 0,
            "peak_day":           grp["day_of_week"].mode().iloc[0] if not grp["day_of_week"].mode().empty else "Unknown",
        })
        cluster_id += 1

    hotspots = pd.DataFrame(records)
    max_vc = hotspots["violation_count"].max()
    hotspots["risk_score"] = (hotspots["violation_count"] / max_vc * 100).round(1)
    hotspots = hotspots.sort_values("risk_score", ascending=False).reset_index(drop=True)
    hotspots.to_csv(OUT / "hotspots.csv", index=False)
    print(f"  Total hotspots: {len(hotspots)}")
    print(f"  Saved -> outputs/hotspots.csv  shape={hotspots.shape}")

    # Center Count Validation (Priority 1.4)
    if "center_count_mean" in hotspots.columns:
        cv = hotspots[["top_junction", "violation_count", "center_count_mean"]].copy()
        # Divergence: High violation count but low center count (under-policed) vs low violation count but high center count (over-policed)
        # Normalize both to compare
        cv["density_norm"] = cv["violation_count"] / max_vc if max_vc > 0 else 0
        max_cc = cv["center_count_mean"].max()
        cv["center_norm"] = cv["center_count_mean"] / max_cc if max_cc > 0 else 0
        cv["divergence"] = cv["density_norm"] - cv["center_norm"]
        cv["status"] = "Balanced"
        cv.loc[cv["divergence"] > 0.2, "status"] = "Under-policed Hotspot"
        cv.loc[cv["divergence"] < -0.2, "status"] = "Over-policed Easy Spot"
        cv = cv.sort_values("divergence", ascending=False)
        cv.to_csv(OUT / "center_validation.csv", index=False)
        print(f"  Saved -> outputs/center_validation.csv")

    # ── Folium map ──────────────────────────────────────────────
    print("Building Folium map ...")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")

    heat_data = df[["latitude", "longitude"]].values.tolist()
    heat_layer = folium.FeatureGroup(name="Violation Heatmap")
    HeatMap(heat_data, radius=10, blur=12, max_zoom=15).add_to(heat_layer)
    heat_layer.add_to(m)

    marker_layer = folium.FeatureGroup(name="Top 15 Hotspots")
    for _, row in hotspots.head(15).iterrows():
        rs = row["risk_score"]
        color = "red" if rs > 70 else ("orange" if rs >= 40 else "yellow")
        radius = max(8, min(25, 8 + (row["violation_count"] / max_vc) * 17))
        popup_html = (
            f"<b>{row['top_junction']}</b><br>"
            f"Risk Score: {rs}<br>"
            f"Violations: {row['violation_count']:,}<br>"
            f"Top Violation: {row['top_violation']}<br>"
            f"Peak Hour: {row['peak_hour']}:00"
        )
        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(marker_layer)

    marker_layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(OUT / "hotspot_map.html"))
    print("  Saved -> outputs/hotspot_map.html")

    print(f"\nTop 10 hotspots:")
    top10 = hotspots.head(10)[["top_junction", "risk_score", "violation_count", "peak_hour"]]
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
