"""Model Training and Benchmarking CLI for CREW.
Trains a singular unified RandomForestRegressor across multi-domain datasets
(Energy, Water, Waste), logging cross-domain regression metrics (R², MAE, RMSE).
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

from crew.config import CAMPUS_NAME
from crew.data_generator import generate_all_datasets
from crew.detection import CampusResourceDetector


def train_singular_model(
    data_dir: str = "data",
    model_path: str = "models/detector.joblib",
) -> CampusResourceDetector:
    """Executes full training lifecycle across 3 separate 10k datasets."""
    print("=" * 70)
    print("⚡ CREW MULTI-DOMAIN SINGULAR MODEL TRAINER")
    print(f"Target Campus: {CAMPUS_NAME}")
    print("=" * 70)

    p_data = Path(data_dir)
    energy_p = p_data / "energy_data.csv"
    water_p = p_data / "water_data.csv"
    waste_p = p_data / "waste_data.csv"

    # Ensure all 3 files exist and are at 10k
    if not (energy_p.exists() and water_p.exists() and waste_p.exists()):
        print("One or more domain datasets missing. Generating 3x 10k datasets...")
        datasets = generate_all_datasets(target_per_domain=10000)
        combined_df = datasets["combined"]
    else:
        df_e = pd.read_csv(energy_p)
        df_w = pd.read_csv(water_p)
        df_waste = pd.read_csv(waste_p)
        print(f"📁 Loaded 3 Domain Files:")
        print(f"   • Energy: {len(df_e):,} records")
        print(f"   • Water:  {len(df_w):,} records")
        print(f"   • Waste:  {len(df_waste):,} records")
        combined_df = pd.concat([df_e, df_w, df_waste], ignore_index=True)

    print(f"\n📊 Total Combined Training Telemetry: {len(combined_df):,} records")
    print(f"   Facility Categories: {list(combined_df['facility_category'].unique())}")
    print(f"   Metrics: {list(combined_df['metric'].unique())}")

    start_time = time.time()
    detector = CampusResourceDetector(n_estimators=100, random_state=42)
    print("\n⏳ Training singular unified RandomForestRegressor across all records...")
    detector.fit(combined_df)
    train_duration = time.time() - start_time
    print(f"✅ Training completed in {train_duration:.2f} seconds.")

    # Save artifact
    saved_path = detector.save(model_path)
    print(f"💾 Persisted singular model to: {saved_path}")

    # Output benchmark evaluation report
    print("\n" + "-" * 70)
    print("🎯 SINGULAR MODEL ACCURACY BENCHMARKS (80/20 Test Split)")
    print("-" * 70)
    overall = detector.metrics.get("overall", {})
    print(f"  • OVERALL Singular Model:")
    print(f"      - R² Score: {overall.get('r2', 0):.4f}")
    print(f"      - MAE: {overall.get('mae', 0)} units")
    print(f"      - RMSE: {overall.get('rmse', 0)} units")

    for m in ["energy", "water", "waste"]:
        scores = detector.metrics.get(m, {})
        unit = "kWh" if m == "energy" else ("L" if m == "water" else "kg")
        print(f"  • {m.upper():<6} Domain Breakdown:")
        print(f"      - R² Score: {scores.get('r2', 0):.4f}")
        print(f"      - MAE: {scores.get('mae', 0)} {unit}")
        print(f"      - RMSE: {scores.get('rmse', 0)} {unit}")
        print(f"      - Evaluated on {scores.get('sample_count', 0):,} records")

    # Anomaly detection pass
    print("\n" + "-" * 70)
    print("🚨 MULTI-DOMAIN ANOMALY DETECTION PASS")
    print("-" * 70)
    flagged_df = detector.predict_and_flag(combined_df)
    total_flags = int(flagged_df["flagged"].sum())
    flag_rate = (total_flags / len(flagged_df)) * 100
    print(f"  • Total Anomalies Flagged: {total_flags:,} / {len(flagged_df):,} ({flag_rate:.2f}%)")

    for m in ["energy", "water", "waste"]:
        m_flags = int(flagged_df[(flagged_df["metric"] == m) & flagged_df["flagged"]].shape[0])
        print(f"      - {m.title()}: {m_flags:,} flagged anomalies")

    print("=" * 70)
    print("🚀 Singular model ready for instantaneous multi-domain UI inference.")
    print("=" * 70)

    return detector


if __name__ == "__main__":
    train_singular_model()
