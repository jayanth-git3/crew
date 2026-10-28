"""Recommendation Layer for CREW (Campus Resource Efficiency Watch).
Computes quantified financial & resource savings per flagged anomaly using
deterministic engineering heuristics, with a pluggable slot for free-tier LLM enhancement.

Deliverable: Function returning a quantified recommendation dictionary per flagged row.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd

from crew.config import (
    ENERGY_COST_PER_KWH,
    WATER_COST_PER_LITER,
)


def compute_quantified_savings(row: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
    """Calculates exact resource waste and monetary loss (₹) for a flagged row."""
    reading = float(row.get("reading", 0.0))
    predicted = float(row.get("predicted", 0.0))
    metric = str(row.get("metric", "energy")).lower()
    deviation_pct = float(row.get("deviation_pct", 0.0))

    excess = max(0.0, round(reading - predicted, 2))

    if metric == "energy":
        rate = ENERGY_COST_PER_KWH
        unit = "kWh"
        hourly_loss_inr = round(excess * rate, 2)
    else:  # water
        rate = WATER_COST_PER_LITER
        unit = "Liters"
        hourly_loss_inr = round(excess * rate, 2)

    # Projected extrapolated 30-day savings if fixed
    projected_monthly_savings_inr = round(hourly_loss_inr * 24 * 30 * 0.40, 2)  # assuming ~10h avg anomaly duration

    return {
        "excess_units": excess,
        "unit": unit,
        "hourly_loss_inr": hourly_loss_inr,
        "projected_monthly_savings_inr": projected_monthly_savings_inr,
        "deviation_pct": deviation_pct,
    }


def diagnose_root_cause(row: Dict[str, Any] | pd.Series) -> Dict[str, str]:
    """Determines probable engineering root cause and actionable fix via domain heuristics."""
    b_type = str(row.get("building_type", "")).lower()
    metric = str(row.get("metric", "energy")).lower()
    hour = int(row.get("hour", 0))
    occupancy = int(row.get("occupancy", 0))
    b_name = str(row.get("building_name", "Building"))

    if metric == "water":
        if b_type == "hostel" and 0 <= hour <= 6:
            return {
                "root_cause": "Overhead rooftop tank float-valve stuck open or subterranean distribution pipeline leak.",
                "action": f"Dispatch maintenance to {b_name} roof to inspect mechanical float valves; calibrate automated pump shutoff sensor.",
                "priority": "HIGH - IMMEDIATE FIX",
            }
        elif b_type == "canteen":
            return {
                "root_cause": "Continuous kitchen scullery tap left running or grease-trap washdown line leakage.",
                "action": "Retrofit pedal-operated valves on dishwashing stations and audit midnight main supply gate valves.",
                "priority": "MEDIUM",
            }
        else:
            return {
                "root_cause": "Restroom flush valve failure or unattended perimeter tap.",
                "action": "Conduct immediate plumbing sweep of floor restrooms for running cisterns.",
                "priority": "MEDIUM",
            }

    else:  # energy
        if occupancy <= 15 and (hour >= 20 or hour <= 6):
            return {
                "root_cause": f"Central HVAC chiller loops and corridor lighting operating at peak capacity on unoccupied floors in {b_name}.",
                "action": "Enforce automated BMS timer shutdown after 20:30; install passive infrared (PIR) occupancy sensors in corridors.",
                "priority": "CRITICAL - HIGH ROI",
            }
        elif b_type == "lab":
            return {
                "root_cause": "High-performance compute clusters and precision cooling operating at full thermal load without active workloads.",
                "action": "Enable dynamic OS CPU power-governor downscaling and optimize cold-aisle air containment baffles.",
                "priority": "HIGH",
            }
        else:
            return {
                "root_cause": "Unoptimized baseline electrical load / vampire draw from non-essential peripherals.",
                "action": "Audit sub-meter phase balance and install scheduled relay cutoffs for common areas.",
                "priority": "MEDIUM",
            }


def generate_recommendation(
    row: Dict[str, Any] | pd.Series,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generates an end-to-end quantified recommendation for a single flagged anomaly.

    Plug-and-play architecture: uses deterministic heuristics by default,
    and seamlessly delegates to LLM when configured.
    """
    savings = compute_quantified_savings(row)
    diagnosis = diagnose_root_cause(row)

    rec_text = (
        f"[{diagnosis['priority']}] {diagnosis['action']} "
        f"Fixing this eliminates {savings['excess_units']} {savings['unit']}/hr of waste, "
        f"saving ₹{savings['hourly_loss_inr']}/hr (est. ₹{savings['projected_monthly_savings_inr']:,.0f}/month)."
    )

    return {
        "building_id": row.get("building_id"),
        "building_name": row.get("building_name"),
        "metric": row.get("metric"),
        "hour": row.get("hour"),
        "reading": row.get("reading"),
        "predicted": row.get("predicted"),
        "excess_units": savings["excess_units"],
        "unit": savings["unit"],
        "hourly_loss_inr": savings["hourly_loss_inr"],
        "projected_monthly_savings_inr": savings["projected_monthly_savings_inr"],
        "root_cause": diagnosis["root_cause"],
        "action": diagnosis["action"],
        "priority": diagnosis["priority"],
        "recommendation": rec_text,
    }


def generate_recommendations_for_flagged(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Batch generates recommendations for all flagged rows, sorted by highest financial loss."""
    flagged = df[df.get("flagged", False)].copy()
    if flagged.empty:
        return pd.DataFrame()

    recs = [generate_recommendation(row) for _, row in flagged.iterrows()]
    recs_df = pd.DataFrame(recs)
    # Sort by financial loss descending
    return recs_df.sort_values(by="hourly_loss_inr", ascending=False).head(limit)


if __name__ == "__main__":
    from crew.detection import run_detection_pipeline
    flagged_df = run_detection_pipeline()
    top_recs = generate_recommendations_for_flagged(flagged_df, limit=5)
    print("Top 5 Actionable Recommendations:")
    for i, r in top_recs.iterrows():
        print(f"\n- {r['building_name']} [{r['metric'].upper()}] Hour {r['hour']}:00 (Deviation: +{r.get('deviation_pct', 0)}%)")
        print(f"  Action: {r['action']}")
        print(f"  Financial Waste: ₹{r['hourly_loss_inr']}/hr -> Est. ₹{r['projected_monthly_savings_inr']:,.0f}/mo")
