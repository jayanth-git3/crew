"""Synthetic Campus Data Generator for Presidency University Bengaluru.
Matches the locked IEEE hackathon data contract:
Columns: building_id, building_name, building_type, occupancy, hour, metric, reading, known_inefficient
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from crew.config import BUILDINGS


def get_hourly_occupancy(building: Dict[str, Any], hour: int, day: int, rng: np.random.Generator) -> int:
    """Calculates realistic occupancy given building type, hour of day, weekday/weekend, and capacity."""
    b_type = building["building_type"]
    capacity = building["capacity"]
    is_weekend = (day % 7) in (5, 6)

    if b_type == "academic":
        if is_weekend:
            # Low skeleton staff / weekend study groups
            ratio = rng.uniform(0.01, 0.08) if 9 <= hour <= 16 else 0.0
        else:
            if 8 <= hour <= 16:
                ratio = rng.uniform(0.60, 0.95)
            elif hour in (17, 18):
                ratio = rng.uniform(0.15, 0.35)
            elif hour in (7, 19, 20):
                ratio = rng.uniform(0.02, 0.08)
            else:
                ratio = rng.uniform(0.0, 0.01)

    elif b_type == "lab":
        if is_weekend:
            # Occasional weekend project teams
            ratio = rng.uniform(0.05, 0.18) if 10 <= hour <= 17 else 0.0
        else:
            if 9 <= hour <= 17:
                ratio = rng.uniform(0.50, 0.85)
            elif hour in (8, 18, 19):
                ratio = rng.uniform(0.10, 0.25)
            else:
                ratio = 0.0

    elif b_type == "hostel":
        if is_weekend:
            # High occupancy throughout day on weekends
            if 0 <= hour <= 9:
                ratio = rng.uniform(0.85, 0.98)
            elif 10 <= hour <= 17:
                ratio = rng.uniform(0.55, 0.75)
            else:
                ratio = rng.uniform(0.80, 0.96)
        else:
            if 0 <= hour <= 6:
                ratio = rng.uniform(0.85, 0.98)
            elif hour in (7, 8):
                ratio = rng.uniform(0.60, 0.80)
            elif 9 <= hour <= 16:
                ratio = rng.uniform(0.05, 0.18)
            elif 17 <= hour <= 21:
                ratio = rng.uniform(0.50, 0.75)
            else:  # 22, 23
                ratio = rng.uniform(0.80, 0.95)

    elif b_type == "canteen":
        if is_weekend:
            # Shifted brunch hours on weekends
            if 9 <= hour <= 11:
                ratio = rng.uniform(0.40, 0.70)
            elif 13 <= hour <= 15:
                ratio = rng.uniform(0.50, 0.80)
            elif 18 <= hour <= 21:
                ratio = rng.uniform(0.45, 0.75)
            else:
                ratio = 0.0
        else:
            if 7 <= hour <= 9:  # Breakfast
                ratio = rng.uniform(0.35, 0.65)
            elif 12 <= hour <= 14:  # Lunch rush
                ratio = rng.uniform(0.70, 0.98)
            elif 17 <= hour <= 20:  # Evening snacks & dinner
                ratio = rng.uniform(0.40, 0.75)
            elif hour in (10, 11, 15, 16):
                ratio = rng.uniform(0.10, 0.25)
            else:
                ratio = 0.0
    else:
        ratio = rng.uniform(0.05, 0.50)

    occupancy = int(round(capacity * ratio))
    return max(0, min(capacity, occupancy))


def calculate_reading(
    building: Dict[str, Any],
    metric: str,
    occupancy: int,
    hour: int,
    rng: np.random.Generator,
) -> float:
    """Computes expected baseline reading + natural variance, with rigged anomalies."""
    if metric == "energy":
        base = building["base_energy"]
        occ_rate = building["energy_per_occ"]
        # Ambient temperature adds slight HVAC demand during daytime peaks
        time_factor = 1.15 if 11 <= hour <= 15 else 1.0
        expected = (base + (occupancy * occ_rate)) * time_factor
    else:  # water
        base = building["base_water"]
        occ_rate = building["water_per_occ"]
        # Morning shower rush and evening usage
        time_factor = 1.30 if (6 <= hour <= 9 or 18 <= hour <= 21) else 0.85
        expected = (base + (occupancy * occ_rate)) * time_factor

    # Natural stochastic variance (3% standard deviation)
    noise = rng.normal(1.0, 0.03)
    actual = max(0.5, expected * noise)

    # Apply deliberate rigged anomaly if building is known_inefficient
    if building.get("known_inefficient", False):
        anomaly_metric = building.get("anomaly_metric")
        anomaly_hours = building.get("anomaly_hours", [])
        multiplier = building.get("anomaly_multiplier", 1.0)

        if metric == anomaly_metric and hour in anomaly_hours:
            actual = actual * multiplier

    return round(float(actual), 2)


def generate_campus_data(days: int = 24, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic campus resource telemetry strictly adhering to the IEEE data contract.

    With default days=24 across 9 buildings, produces exactly:
    24 days * 24 hours * 9 buildings * 2 metrics = 10,368 rows.

    Schema:
    - building_id: str
    - building_name: str
    - building_type: str
    - occupancy: int
    - hour: int
    - metric: str
    - reading: float
    - known_inefficient: bool
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    for day in range(days):
        for hour in range(24):
            for b in BUILDINGS:
                occ = get_hourly_occupancy(b, hour, day, rng)

                # Energy record
                energy_val = calculate_reading(b, "energy", occ, hour, rng)
                records.append({
                    "building_id": b["building_id"],
                    "building_name": b["building_name"],
                    "building_type": b["building_type"],
                    "occupancy": int(occ),
                    "hour": int(hour),
                    "metric": "energy",
                    "reading": float(energy_val),
                    "known_inefficient": bool(b.get("known_inefficient", False)),
                })

                # Water record
                water_val = calculate_reading(b, "water", occ, hour, rng)
                records.append({
                    "building_id": b["building_id"],
                    "building_name": b["building_name"],
                    "building_type": b["building_type"],
                    "occupancy": int(occ),
                    "hour": int(hour),
                    "metric": "water",
                    "reading": float(water_val),
                    "known_inefficient": bool(b.get("known_inefficient", False)),
                })

    df = pd.DataFrame(records)
    # Ensure explicit data types matching the contract
    df["building_id"] = df["building_id"].astype(str)
    df["building_name"] = df["building_name"].astype(str)
    df["building_type"] = df["building_type"].astype(str)
    df["occupancy"] = df["occupancy"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["metric"] = df["metric"].astype(str)
    df["reading"] = df["reading"].astype(float)
    df["known_inefficient"] = df["known_inefficient"].astype(bool)

    return df


def save_dataset(df: pd.DataFrame, output_path: str = "data/campus_data.csv") -> str:
    """Saves DataFrame to CSV, ensuring parent directory exists."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p.resolve())


if __name__ == "__main__":
    data_df = generate_campus_data(days=24, seed=42)
    saved_to = save_dataset(data_df)
    print(f"Generated {len(data_df)} records saved to: {saved_to}")
