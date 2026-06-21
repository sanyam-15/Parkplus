"""
04_time_forecasting.py — XGBoost forecasting of violations + deployment schedule.
Enhanced: lag features, rolling averages, interaction features, comprehensive evaluation.
"""
import pandas as pd
import numpy as np
import json
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime, timedelta

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"

PEAK_HOURS = list(range(8, 11)) + list(range(17, 21))


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE - skip zero actuals to avoid division by zero."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_features(df_raw):
    """Build rich feature set from raw parking data."""

    # Aggregate: junction x date x hour
    agg = (
        df_raw.groupby(["junction_name", "date", "hour", "day_num", "month"])
        .size()
        .reset_index(name="violations")
    )

    # Keep top 20 junctions
    top20 = df_raw["junction_name"].value_counts().head(20).index.tolist()
    agg = agg[agg["junction_name"].isin(top20)].copy()
    agg = agg.sort_values(["junction_name", "date", "hour"]).reset_index(drop=True)

    # --- Basic time features ---
    agg["is_weekend"] = (agg["day_num"] >= 5).astype(int)
    agg["is_peak_hour"] = agg["hour"].isin(PEAK_HOURS).astype(int)
    agg["time_bin"] = pd.cut(
        agg["hour"], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]
    ).astype(int)

    # --- Hour cyclical encoding (captures that hour 23 is close to hour 0) ---
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24).round(4)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24).round(4)

    # --- Day cyclical encoding ---
    agg["day_sin"] = np.sin(2 * np.pi * agg["day_num"] / 7).round(4)
    agg["day_cos"] = np.cos(2 * np.pi * agg["day_num"] / 7).round(4)

    # --- Junction encoding ---
    le = LabelEncoder()
    agg["junction_id"] = le.fit_transform(agg["junction_name"])

    # --- Target encoding: junction-level historical stats ---
    junction_stats = agg.groupby("junction_name")["violations"].agg(
        ["mean", "std", "median", "max"]
    ).rename(columns={
        "mean": "junction_avg", "std": "junction_std",
        "median": "junction_median", "max": "junction_max"
    })
    agg = agg.merge(junction_stats, on="junction_name", how="left")
    agg["junction_std"] = agg["junction_std"].fillna(0)

    # --- Target encoding: junction x hour stats ---
    jh_stats = agg.groupby(["junction_name", "hour"])["violations"].agg(
        ["mean", "std"]
    ).rename(columns={"mean": "jh_avg", "std": "jh_std"}).reset_index()
    agg = agg.merge(jh_stats, on=["junction_name", "hour"], how="left")
    agg["jh_std"] = agg["jh_std"].fillna(0)

    # --- Target encoding: junction x day_num stats ---
    jd_stats = agg.groupby(["junction_name", "day_num"])["violations"].agg(
        ["mean"]
    ).rename(columns={"mean": "jd_avg"}).reset_index()
    agg = agg.merge(jd_stats, on=["junction_name", "day_num"], how="left")

    # --- Lag features: per-junction rolling daily totals ---
    daily_totals = agg.groupby(["junction_name", "date"])["violations"].sum().reset_index()
    daily_totals = daily_totals.rename(columns={"violations": "daily_total"})
    daily_totals = daily_totals.sort_values(["junction_name", "date"])

    # Rolling means per junction
    for window in [3, 7]:
        col_name = f"rolling_{window}d_avg"
        daily_totals[col_name] = (
            daily_totals.groupby("junction_name")["daily_total"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Lag-1 day total
    daily_totals["lag_1d"] = (
        daily_totals.groupby("junction_name")["daily_total"]
        .transform(lambda x: x.shift(1))
    )

    agg = agg.merge(
        daily_totals[["junction_name", "date", "rolling_3d_avg", "rolling_7d_avg", "lag_1d"]],
        on=["junction_name", "date"], how="left"
    )

    # Fill NaN lags with junction averages
    for col in ["rolling_3d_avg", "rolling_7d_avg", "lag_1d"]:
        agg[col] = agg[col].fillna(agg["junction_avg"])

    # --- Interaction features ---
    agg["peak_weekend"] = agg["is_peak_hour"] * agg["is_weekend"]
    agg["hour_junction_interaction"] = agg["hour"] * agg["junction_id"]

    return agg, le, top20


def main():
    print("=" * 60)
    print("  VIOLATION FORECASTING ENGINE")
    print("=" * 60)

    print("\nLoading parking_clean.csv ...")
    df = pd.read_csv(OUT / "parking_clean.csv", low_memory=False)
    print(f"  Rows: {len(df):,}")

    agg, le, top20 = build_features(df)
    print(f"  Aggregated rows (top 20 junctions): {len(agg):,}")

    # --- Feature columns ---
    feature_cols = [
        "hour", "day_num", "month", "is_weekend",
        "junction_id", "is_peak_hour", "time_bin",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "junction_avg", "junction_std", "junction_median", "junction_max",
        "jh_avg", "jh_std", "jd_avg",
        "rolling_3d_avg", "rolling_7d_avg", "lag_1d",
        "peak_weekend", "hour_junction_interaction",
    ]

    X = agg[feature_cols].values
    y = agg["violations"].values

    # --- Train/test split (random stratified for fair evaluation) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # --- Train XGBoost ---
    model = XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Save Model ---
    joblib.dump(model, OUT / "xgboost_model.pkl")
    print(f"  Saved -> outputs/xgboost_model.pkl")

    # --- Evaluate ---
    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    test_mae  = mean_absolute_error(y_test, preds_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    test_r2   = r2_score(y_test, preds_test)
    test_mape = mean_absolute_percentage_error(y_test, preds_test)

    train_mae = mean_absolute_error(y_train, preds_train)
    train_r2  = r2_score(y_train, preds_train)

    accuracy_pct = max(0, 100 - test_mape)
    
    # Baseline Comparison (Priority 2.2)
    # Using 'junction_avg' as naive baseline
    j_avg_idx = feature_cols.index("junction_avg")
    baseline_naive_preds = X_test[:, j_avg_idx]
    baseline_naive_mae = mean_absolute_error(y_test, baseline_naive_preds)
    
    # Using 'rolling_7d_avg' as moving average baseline
    roll7_idx = feature_cols.index("rolling_7d_avg")
    baseline_ma_preds = X_test[:, roll7_idx]
    baseline_ma_mae = mean_absolute_error(y_test, baseline_ma_preds)
    
    # Best baseline
    best_baseline_mae = min(baseline_naive_mae, baseline_ma_mae)
    improvement_pct = max(0, (best_baseline_mae - test_mae) / best_baseline_mae * 100)

    print(f"\n{'='*60}")
    print(f"  MODEL EVALUATION METRICS (XGBoost Regressor)")
    print(f"{'='*60}")
    print(f"  Features Used:    {len(feature_cols)}")
    print(f"  ---")
    print(f"  Baseline Naive MAE: {baseline_naive_mae:.2f} violations")
    print(f"  Baseline M.A. MAE:  {baseline_ma_mae:.2f} violations")
    print(f"  ---")
    print(f"  Train MAE:        {train_mae:.2f} violations")
    print(f"  Train R2:         {train_r2:.4f}  ({train_r2*100:.1f}%)")
    print(f"  ---")
    print(f"  Test MAE:         {test_mae:.2f} violations")
    print(f"  Test RMSE:        {test_rmse:.2f} violations")
    print(f"  Test R2 Score:    {test_r2:.4f}  ({test_r2*100:.1f}%)")
    print(f"  Test MAPE:        {test_mape:.1f}%")
    print(f"  Improvement:      {improvement_pct:.1f}% over baseline")
    print(f"  Accuracy (proxy): {accuracy_pct:.1f}%")
    print(f"{'='*60}")

    # --- Save metrics to JSON ---
    metrics = {
        "model": "XGBRegressor",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "hyperparameters": {
            "n_estimators": 800,
            "max_depth": 7,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_mae": round(float(train_mae), 4),
        "train_r2": round(float(train_r2), 4),
        "test_mae": round(float(test_mae), 4),
        "test_rmse": round(float(test_rmse), 4),
        "test_r2_score": round(float(test_r2), 4),
        "test_r2_percentage": round(float(test_r2 * 100), 1),
        "test_mape": round(float(test_mape), 1),
        "accuracy_pct": round(float(accuracy_pct), 1),
        "improvement_over_baseline": {
            "baseline_naive_mae": round(float(baseline_naive_mae), 4),
            "baseline_ma_mae": round(float(baseline_ma_mae), 4),
            "best_baseline_mae": round(float(best_baseline_mae), 4),
            "baseline_features": 2,
            "improved_mae": round(float(test_mae), 4),
            "improved_r2": round(float(test_r2), 4),
            "mae_reduction_pct": round(float(improvement_pct), 1),
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUT / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved -> outputs/model_metrics.json")
    
    # Save baseline comparison explicitly for dashboard
    baseline_df = pd.DataFrame([
        {"Method": "Naive Average", "MAE": round(baseline_naive_mae, 2)},
        {"Method": "Moving Average (7d)", "MAE": round(baseline_ma_mae, 2)},
        {"Method": "XGBoost (Park+)", "MAE": round(test_mae, 2)}
    ])
    baseline_df.to_csv(OUT / "baseline_comparison.csv", index=False)
    print(f"  Saved -> outputs/baseline_comparison.csv")

    # --- Feature Importance ---
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    fi_df.to_csv(OUT / "feature_importance.csv", index=False)
    print(f"  Saved -> outputs/feature_importance.csv")

    print(f"\n  Feature Importance Ranking:")
    for _, row in fi_df.head(10).iterrows():
        bar = "#" * int(row["importance"] * 50)
        print(f"    {row['feature']:<28} {row['importance']:.4f}  {bar}")

    # --- Next 7 days prediction grid ---
    today = datetime.now()

    # Build lookup tables for predictions
    junction_stats_dict = {}
    jh_stats_dict = {}
    jd_stats_dict = {}

    for junction in top20:
        j_data = agg[agg["junction_name"] == junction]
        junction_stats_dict[junction] = {
            "avg": j_data["junction_avg"].iloc[0] if len(j_data) > 0 else 0,
            "std": j_data["junction_std"].iloc[0] if len(j_data) > 0 else 0,
            "median": j_data["junction_median"].iloc[0] if len(j_data) > 0 else 0,
            "max": j_data["junction_max"].iloc[0] if len(j_data) > 0 else 0,
            "last_lag": j_data["lag_1d"].iloc[-1] if len(j_data) > 0 else 0,
            "last_roll3": j_data["rolling_3d_avg"].iloc[-1] if len(j_data) > 0 else 0,
            "last_roll7": j_data["rolling_7d_avg"].iloc[-1] if len(j_data) > 0 else 0,
        }
        for hour in range(24):
            jh_data = j_data[j_data["hour"] == hour]
            jh_stats_dict[(junction, hour)] = {
                "avg": jh_data["jh_avg"].iloc[0] if len(jh_data) > 0 else junction_stats_dict[junction]["avg"],
                "std": jh_data["jh_std"].iloc[0] if len(jh_data) > 0 else 0,
            }
        for day in range(7):
            jd_data = j_data[j_data["day_num"] == day]
            jd_stats_dict[(junction, day)] = {
                "avg": jd_data["jd_avg"].iloc[0] if len(jd_data) > 0 else junction_stats_dict[junction]["avg"],
            }

    future_rows = []
    for junction in top20:
        jid = le.transform([junction])[0]
        js = junction_stats_dict[junction]
        for day_offset in range(1, 8):
            dt = today + timedelta(days=day_offset)
            day_num = dt.weekday()
            month   = dt.month
            is_wknd = int(day_num >= 5)
            for hour in range(24):
                is_peak = int(hour in PEAK_HOURS)
                if hour <= 6:
                    tbin = 0
                elif hour <= 12:
                    tbin = 1
                elif hour <= 18:
                    tbin = 2
                else:
                    tbin = 3
                jh = jh_stats_dict.get((junction, hour), {"avg": js["avg"], "std": 0})
                jd = jd_stats_dict.get((junction, day_num), {"avg": js["avg"]})
                future_rows.append({
                    "junction_name": junction,
                    "day_name":      dt.strftime("%A"),
                    "date":          dt.strftime("%Y-%m-%d"),
                    "hour": hour, "day_num": day_num, "month": month,
                    "is_weekend": is_wknd, "junction_id": jid,
                    "is_peak_hour": is_peak, "time_bin": tbin,
                    "hour_sin": round(np.sin(2 * np.pi * hour / 24), 4),
                    "hour_cos": round(np.cos(2 * np.pi * hour / 24), 4),
                    "day_sin": round(np.sin(2 * np.pi * day_num / 7), 4),
                    "day_cos": round(np.cos(2 * np.pi * day_num / 7), 4),
                    "junction_avg": js["avg"], "junction_std": js["std"],
                    "junction_median": js["median"], "junction_max": js["max"],
                    "jh_avg": jh["avg"], "jh_std": jh["std"],
                    "jd_avg": jd["avg"],
                    "rolling_3d_avg": js["last_roll3"],
                    "rolling_7d_avg": js["last_roll7"],
                    "lag_1d": js["last_lag"],
                    "peak_weekend": is_peak * is_wknd,
                    "hour_junction_interaction": hour * jid,
                })

    future = pd.DataFrame(future_rows)
    X_fut  = future[feature_cols].values
    future["predicted_violations"] = np.maximum(0, model.predict(X_fut)).round(0).astype(int)

    future.to_csv(OUT / "predictions_7day.csv", index=False)
    print(f"\n  Saved -> outputs/predictions_7day.csv  shape={future.shape}")

    # --- Deployment schedule for top 5 junctions ---
    top5 = top20[:5]
    sched = future[future["junction_name"].isin(top5)].copy()
    sched["officers_needed"] = np.ceil(sched["predicted_violations"] / 20).clip(upper=6).astype(int)
    sched.to_csv(OUT / "deployment_schedule.csv", index=False)
    print(f"  Saved -> outputs/deployment_schedule.csv  shape={sched.shape}")

    # Sample
    sample_junc = top5[0]
    print(f"\n  Sample predictions for {sample_junc}:")
    sample = sched[sched["junction_name"] == sample_junc].head(10)
    print(sample[["date", "day_name", "hour", "predicted_violations", "officers_needed"]].to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  FORECASTING ENGINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
