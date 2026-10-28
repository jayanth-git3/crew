"""Scalable Multi-Domain Telemetry Generator for Presidency University Bengaluru.
Produces 3 independent 100,000-record datasets (300,000 records total):
1. data/energy_data.csv (100k records)
2. data/water_data.csv (100k records)
3. data/waste_data.csv (100k records)
Consolidated into data/campus_data.csv (300k records).
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

    if b_type in ("academic", "engineering"):
        if is_weekend:
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

    elif b_type in ("lab", "studio"):
        if is_weekend:
            ratio = rng.uniform(0.05, 0.20) if 10 <= hour <= 17 else 0.0
        else:
            if 9 <= hour <= 17:
                ratio = rng.uniform(0.55, 0.90)
            elif hour in (8, 18, 19):
                ratio = rng.uniform(0.12, 0.30)
            else:
                ratio = rng.uniform(0.01, 0.04)

    elif b_type == "hostel":
        if is_weekend:
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
            else:
                ratio = rng.uniform(0.80, 0.95)

    elif b_type == "canteen":
        if is_weekend:
            if 9 <= hour <= 11:
                ratio = rng.uniform(0.40, 0.70)
            elif 13 <= hour <= 15:
                ratio = rng.uniform(0.50, 0.80)
            elif 18 <= hour <= 21:
                ratio = rng.uniform(0.45, 0.75)
            else:
                ratio = 0.0
        else:
            if 7 <= hour <= 9:
                ratio = rng.uniform(0.35, 0.65)
            elif 12 <= hour <= 14:
                ratio = rng.uniform(0.70, 0.98)
            elif 17 <= hour <= 20:
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
    """Computes expected baseline reading + natural variance, with injected operational anomalies."""
    if metric == "energy":
        base = building["base_energy"]
        occ_rate = building["energy_per_occ"]
        time_factor = 1.15 if 11 <= hour <= 15 else 1.0
        expected = (base + (occupancy * occ_rate)) * time_factor
    elif metric == "water":
        base = building["base_water"]
        occ_rate = building["water_per_occ"]
        time_factor = 1.30 if (6 <= hour <= 9 or 18 <= hour <= 21) else 0.85
        expected = (base + (occupancy * occ_rate)) * time_factor
    else:  # waste (kg)
        base = building["base_waste"]
        occ_rate = building["waste_per_occ"]
        time_factor = 1.45 if (13 <= hour <= 15 or 20 <= hour <= 22) else 0.80
        expected = (base + (occupancy * occ_rate)) * time_factor

    # Natural stochastic variance (3% standard deviation)
    noise = rng.normal(1.0, 0.03)
    actual = max(0.5, expected * noise)

    # Apply deliberate anomaly if building is known_inefficient
    if building.get("known_inefficient", False):
        anomaly_metric = building.get("anomaly_metric")
        anomaly_hours = building.get("anomaly_hours", [])
        multiplier = building.get("anomaly_multiplier", 1.0)

        if metric == anomaly_metric and hour in anomaly_hours:
            actual = actual * multiplier

    return round(float(actual), 2)


def generate_domain_dataset(
    metric: str,
    target_rows: int = 100000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates an independent single-domain dataset containing exactly target_rows."""
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    # Calculate days needed based on number of facilities
    records_per_day = len(BUILDINGS) * 24
    days_needed = int(np.ceil(target_rows / records_per_day))

    for day in range(days_needed):
        for hour in range(24):
            for b in BUILDINGS:
                occ = get_hourly_occupancy(b, hour, day, rng)
                reading_val = calculate_reading(b, metric, occ, hour, rng)

                floors = b.get("floors", ["Ground", "Floor 1", "Floor 2", "Floor 3"])
                floor_selected = floors[day % len(floors)]

                records.append({
                    "building_id": b["building_id"],
                    "building_name": b["building_name"],
                    "building_type": b["building_type"],
                    "facility_category": b.get("facility_category", "college"),
                    "floor_zone": floor_selected,
                    "occupancy": int(occ),
                    "hour": int(hour),
                    "metric": metric,
                    "reading": float(reading_val),
                    "known_inefficient": bool(b.get("known_inefficient", False)),
                })

                if len(records) >= target_rows:
                    break
            if len(records) >= target_rows:
                break
        if len(records) >= target_rows:
            break

    df = pd.DataFrame(records)
    df["building_id"] = df["building_id"].astype(str)
    df["building_name"] = df["building_name"].astype(str)
    df["building_type"] = df["building_type"].astype(str)
    df["facility_category"] = df["facility_category"].astype(str)
    df["floor_zone"] = df["floor_zone"].astype(str)
    df["occupancy"] = df["occupancy"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["metric"] = df["metric"].astype(str)
    df["reading"] = df["reading"].astype(float)
    df["known_inefficient"] = df["known_inefficient"].astype(bool)

    return df


def generate_all_datasets(target_per_domain: int = 100000) -> Dict[str, pd.DataFrame]:
    """Generates 3 separate 100k datasets (Energy, Water, Waste) and a consolidated 300k bridge."""
    p_data = Path("data")
    p_data.mkdir(parents=True, exist_ok=True)

    print(f"Generating Energy dataset ({target_per_domain:,} rows)...")
    energy_df = generate_domain_dataset("energy", target_rows=target_per_domain, seed=101)
    energy_df.to_csv(p_data / "energy_data.csv", index=False)

    print(f"Generating Water dataset ({target_per_domain:,} rows)...")
    water_df = generate_domain_dataset("water", target_rows=target_per_domain, seed=202)
    water_df.to_csv(p_data / "water_data.csv", index=False)

    print(f"Generating Waste dataset ({target_per_domain:,} rows)...")
    waste_df = generate_domain_dataset("waste", target_rows=target_per_domain, seed=303)
    waste_df.to_csv(p_data / "waste_data.csv", index=False)

    # Consolidated bridge (300,000 rows)
    print("Consolidating combined campus_data.csv bridge (300,000 rows)...")
    combined_df = pd.concat([energy_df, water_df, waste_df], ignore_index=True)
    combined_df.to_csv(p_data / "campus_data.csv", index=False)

    return {
        "energy": energy_df,
        "water": water_df,
        "waste": waste_df,
        "combined": combined_df,
    }


def save_dataset(df: pd.DataFrame, output_path: str = "data/campus_data.csv") -> str:
    """Saves DataFrame to CSV, ensuring parent directory exists."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p.resolve())


if __name__ == "__main__":
    datasets = generate_all_datasets(target_per_domain=100000)
    for name, d in datasets.items():
        print(f"  • {name:<10}: {len(d):,} records saved.")
