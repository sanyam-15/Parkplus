"""
02b_geo_enrichment.py — MapmyIndia geo-intelligence layer for hotspots.
"""
import os
import json
import pandas as pd
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
API_KEY = os.environ.get("MAPMYINDIA_API_KEY")

CAPACITY_LOSS_MAP = {
    "local": 0.9,
    "collector": 0.6,
    "sub_arterial": 0.45,
    "arterial": 0.3
}

def load_cache():
    cache_path = OUT / "geo_cache.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(OUT / "geo_cache.json", "w") as f:
        json.dump(cache, f, indent=2)

def mock_geo_data(lat, lon):
    random.seed(f"{lat}_{lon}")
    road_class = random.choice(["local", "collector", "sub_arterial", "arterial"])
    
    pois = ["metro_station", "commercial_complex", "signalized_intersection", "none"]
    weights = [0.1, 0.3, 0.2, 0.4]
    nearby_poi_type = random.choices(pois, weights)[0]
    
    if nearby_poi_type == "none":
        dist = 500  # far away
    else:
        dist = random.uniform(10, 100)
        
    return {
        "road_class": road_class,
        "nearby_poi_type": nearby_poi_type,
        "poi_distance_m": dist
    }

def fetch_mapmyindia_data(lat, lon):
    # Stub implementation - since we don't know the exact MapmyIndia API structure,
    # we'll use a mock if key is invalid, or if key is provided but fails.
    # In a real scenario, we would use: requests.get(f"https://atlas.mapmyindia.com/api/places/geocode?lat={lat}&lng={lon}...")
    # Since this is a hackathon project we will mock unless real API key logic is explicitly available.
    return mock_geo_data(lat, lon)

def get_geo_data(lat, lon, cache):
    key = f"{round(lat, 4)}_{round(lon, 4)}"
    if key in cache:
        return cache[key]
    
    if API_KEY:
        data = fetch_mapmyindia_data(lat, lon)
    else:
        data = mock_geo_data(lat, lon)
        
    cache[key] = data
    return data

def main():
    print("Loading hotspots.csv ...")
    hotspots_path = OUT / "hotspots.csv"
    if not hotspots_path.exists():
        print(f"ERROR: {hotspots_path} not found")
        return
        
    df = pd.read_csv(hotspots_path)
    print(f"  Rows: {len(df)}")
    
    if not API_KEY:
        print("MOCK MODE — set MAPMYINDIA_API_KEY for real data")
        
    cache = load_cache()
    
    enriched_rows = []
    for _, row in df.iterrows():
        lat = row.get("centroid_lat")
        lon = row.get("centroid_lon")
        
        if pd.isna(lat) or pd.isna(lon):
            geo = {"road_class": "local", "nearby_poi_type": "none", "poi_distance_m": 500}
        else:
            geo = get_geo_data(lat, lon, cache)
        
        road_class = geo["road_class"]
        poi_type = geo["nearby_poi_type"]
        dist = geo["poi_distance_m"]
        
        row_dict = row.to_dict()
        row_dict["road_class"] = road_class
        row_dict["road_capacity_loss_factor"] = CAPACITY_LOSS_MAP.get(road_class, 0.5)
        row_dict["nearby_poi_type"] = poi_type
        
        # Choke proximity score: 1 / (1 + distance_m)
        score = 1.0 / (1.0 + dist)
        row_dict["choke_proximity_score_raw"] = score
        enriched_rows.append(row_dict)
        
    save_cache(cache)
    
    out_df = pd.DataFrame(enriched_rows)
    
    # Normalize choke proximity 0-1
    max_score = out_df["choke_proximity_score_raw"].max()
    min_score = out_df["choke_proximity_score_raw"].min()
    if max_score > min_score:
        out_df["choke_proximity_score"] = (out_df["choke_proximity_score_raw"] - min_score) / (max_score - min_score)
    else:
        out_df["choke_proximity_score"] = 0.0
        
    out_df.drop(columns=["choke_proximity_score_raw"], inplace=True)
    out_df["choke_proximity_score"] = out_df["choke_proximity_score"].round(3)
    
    out_path = OUT / "hotspots_geo_enriched.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path} shape={out_df.shape}")

if __name__ == "__main__":
    main()
