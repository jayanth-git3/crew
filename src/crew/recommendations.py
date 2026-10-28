"""Recommendation Layer for CREW (Campus Resource Efficiency Watch).
Computes quantified financial & resource savings per flagged anomaly across
Energy (kWh), Water (Liters), and Waste (kg) using deterministic domain heuristics.

Deliverable: Function returning a quantified recommendation dictionary per flagged row.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd

from crew.config import (
    ENERGY_COST_PER_KWH,
    WATER_COST_PER_LITER,
    WASTE_COST_PER_KG,
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
    elif metric == "water":
        rate = WATER_COST_PER_LITER
        unit = "Liters"
    else:  # waste (kg)
        rate = WASTE_COST_PER_KG
        unit = "kg"

    hourly_loss_inr = round(excess * rate, 2)
    # Projected extrapolated 30-day savings assuming ~8h average daily anomaly duration
    projected_monthly_savings_inr = round(hourly_loss_inr * 8 * 30, 2)

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

    elif metric == "waste":
        if b_type == "canteen":
            return {
                "root_cause": "Bulk kitchen batch food spoilage, excess plate leftovers, and unsegregated wet/dry waste dumping.",
                "action": "Install commercial on-site biogas composter; audit kitchen meal prep portions; partner with Bengaluru food recovery NGO.",
                "priority": "CRITICAL - HIGH ESG IMPACT",
            }
        elif b_type == "hostel":
            return {
                "root_cause": "Unsegregated delivery packaging and consumer food containers accumulating in hostel waste bins.",
                "action": "Establish floor-level segregated 2-bin disposal (organic vs recyclable) and partner with BBMP dry waste recyclers.",
                "priority": "MEDIUM",
            }
        else:
            return {
                "root_cause": "Unregulated electronic packaging and component scrap in maker & computer labs.",
                "action": "Deploy dedicated electronic waste bins with an authorized KSPCB e-waste recycler.",
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
    """Generates an end-to-end quantified recommendation for a single flagged anomaly."""
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
        "facility_category": row.get("facility_category", "college"),
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
    return recs_df.sort_values(by="hourly_loss_inr", ascending=False).head(limit)
