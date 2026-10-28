"""Model Training and Benchmarking CLI for CREW.
Trains dual RandomForestRegressors on the 10k+ campus dataset, calculates
regression metrics (R², MAE, RMSE), and persists the trained detector artifact.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

from crew.config import CAMPUS_NAME
from crew.data_generator import generate_campus_data, save_dataset
from crew.detection import CampusResourceDetector


def train_and_benchmark(
    data_path: str = "data/campus_data.csv",
    model_path: str = "models/detector.joblib",
    days: int = 24,
) -> CampusResourceDetector:
    """Executes full training lifecycle on 10k+ records and outputs benchmark report."""
    print("=" * 65)
    print("⚡ CREW MODEL TRAINING & BENCHMARKING ENGINE")
    print(f"Target Campus: {CAMPUS_NAME}")
    print("=" * 65)

    data_file = Path(data_path)
    if not data_file.exists():
        print(f"Dataset not found at {data_path}. Generating {days}-day dataset...")
        df = generate_campus_data(days=days)
        save_dataset(df, str(data_file))
    else:
        df = pd.read_csv(data_file)
        if len(df) < 10000:
            print(f"Dataset has {len(df)} rows. Re-generating full {days}-day 10k dataset...")
            df = generate_campus_data(days=days)
            save_dataset(df, str(data_file))

    print(f"\n📊 Loaded Telemetry Dataset: {len(df):,} records ({len(df.columns)} columns)")
    print(f"   Date Span: ~{days} days | Buildings: {df['building_id'].nunique()} | Metrics: Energy & Water")

    # Initialize and fit detector
    start_time = time.time()
    detector = CampusResourceDetector(n_estimators=100, random_state=42)
    print("\n⏳ Training dual RandomForestRegressors on normal baseline...")
    detector.fit(df)
    train_duration = time.time() - start_time
    print(f"✅ Training completed in {train_duration:.2f} seconds.")

    # Save artifact
    saved_path = detector.save(model_path)
    print(f"💾 Persisted pre-trained detector to: {saved_path}")

    # Output benchmark evaluation report
    print("\n" + "-" * 65)
    print("🎯 REGRESSION ACCURACY BENCHMARKS (80/20 Baseline Test Split)")
    print("-" * 65)
    for metric_name, scores in detector.metrics.items():
        print(f"  • {metric_name.upper():<6} Regressor:")
        print(f"      - R² Score: {scores['r2']:.4f} (Baseline fit quality)")
        print(f"      - Mean Absolute Error (MAE): {scores['mae']} units")
        print(f"      - Root Mean Squared Error (RMSE): {scores['rmse']} units")
        print(f"      - Evaluated on {scores['sample_count']:,} normal telemetry records")

    # Anomaly detection pass
    print("\n" + "-" * 65)
    print("🚨 ANOMALY DETECTION INFERENCE PASS")
    print("-" * 65)
    flagged_df = detector.predict_and_flag(df)
    total_flags = int(flagged_df["flagged"].sum())
    flag_rate = (total_flags / len(flagged_df)) * 100
    print(f"  • Total Anomalies Flagged: {total_flags:,} / {len(flagged_df):,} ({flag_rate:.2f}%)")

    # Rigged anomalies recall
    rigged_total = int(flagged_df["known_inefficient"].sum())
    rigged_flagged = int(flagged_df[flagged_df["known_inefficient"]]["flagged"].sum())
    print(f"  • Rigged Anomaly Detection: {rigged_flagged:,} / {rigged_total:,} flagged during spike hours")

    print("=" * 65)
    print("🚀 Model is ready for instantaneous loading in the Streamlit UI.")
    print("=" * 65)

    return detector


if __name__ == "__main__":
    train_and_benchmark()
