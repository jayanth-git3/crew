"""Recommendation Layer for CREW (Campus Resource Efficiency Watch).
Computes quantified financial, resource, and carbon savings per flagged anomaly across
Energy (kWh), Water (Liters), and Waste (kg) using deterministic domain heuristics.

Deliverable: Function returning a quantified recommendation dictionary per flagged row,
campus operational health scores, and building health matrices.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd

from crew.config import (
    ENERGY_COST_PER_KWH,
    WATER_COST_PER_LITER,
    WASTE_COST_PER_KG,
)

# Utility cost benchmarks & emission factors
CO2_KG_PER_KWH = 0.82  # Central Electricity Authority (CEA) India grid baseline factor
WATER_CO2_G_PER_LITER = 0.65  # Pumping and treatment carbon factor (grams CO2/L)
WASTE_CO2_KG_PER_KG = 0.45  # Landfill methane avoidance factor (kg CO2e / kg organic waste)


def compute_quantified_savings(row: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
    """Calculates exact resource waste, monetary loss (₹), and CO2 impact for a flagged row."""
    reading = float(row.get("reading", 0.0))
    predicted = float(row.get("predicted", 0.0))
    metric = str(row.get("metric", "energy")).lower()
    deviation_pct = float(row.get("deviation_pct", 0.0))

    excess = max(0.0, round(reading - predicted, 2))

    if metric == "energy":
        rate = ENERGY_COST_PER_KWH
        unit = "kWh"
        hourly_loss_inr = round(excess * rate, 2)
        co2_kg_hourly = round(excess * CO2_KG_PER_KWH, 2)
    elif metric == "water":
        rate = WATER_COST_PER_LITER
        unit = "Liters"
        hourly_loss_inr = round(excess * rate, 2)
        co2_kg_hourly = round((excess * WATER_CO2_G_PER_LITER) / 1000.0, 3)
    else:  # waste (kg)
        rate = WASTE_COST_PER_KG
        unit = "kg"
        hourly_loss_inr = round(excess * rate, 2)
        co2_kg_hourly = round(excess * WASTE_CO2_KG_PER_KG, 2)

    # Projected extrapolated 30-day savings assuming ~10h average daily anomaly duration
    projected_monthly_savings_inr = round(hourly_loss_inr * 24 * 30 * 0.40, 2)
    projected_monthly_co2_kg = round(co2_kg_hourly * 24 * 30 * 0.40, 1)

    return {
        "excess_units": excess,
        "unit": unit,
        "hourly_loss_inr": hourly_loss_inr,
        "projected_monthly_savings_inr": projected_monthly_savings_inr,
        "co2_kg_hourly": co2_kg_hourly,
        "projected_monthly_co2_kg": projected_monthly_co2_kg,
        "deviation_pct": deviation_pct,
    }


def diagnose_root_cause(row: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
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
                "tier": "Tier 2: Targeted Maintenance",
                "capex_inr": 1500,
                "payback_days": 1.2,
                "responsible_team": "Plumbing & Hydraulics Ops",
            }
        elif b_type == "canteen":
            return {
                "root_cause": "Continuous kitchen scullery tap left running or grease-trap washdown line leakage.",
                "action": "Retrofit pedal-operated valves on dishwashing stations and audit midnight main supply gate valves.",
                "priority": "MEDIUM",
                "tier": "Tier 2: Targeted Maintenance",
                "capex_inr": 2500,
                "payback_days": 4.5,
                "responsible_team": "Canteen Facilities",
            }
        else:
            return {
                "root_cause": "Restroom flush valve failure or unattended perimeter tap.",
                "action": "Conduct immediate plumbing sweep of floor restrooms for running cisterns.",
                "priority": "MEDIUM",
                "tier": "Tier 1: Operational Quick Win",
                "capex_inr": 0,
                "payback_days": 0.1,
                "responsible_team": "General Maintenance",
            }

    elif metric == "waste":
        if b_type == "canteen":
            return {
                "root_cause": "Bulk kitchen batch food spoilage, excess plate leftovers, and unsegregated wet/dry waste dumping.",
                "action": "Install commercial on-site biogas composter; audit kitchen meal prep portions; partner with Bengaluru food recovery NGO.",
                "priority": "CRITICAL - HIGH ESG IMPACT",
                "tier": "Tier 2: Targeted Maintenance",
                "capex_inr": 8000,
                "payback_days": 7.5,
                "responsible_team": "Canteen & Waste Ops",
            }
        elif b_type == "hostel":
            return {
                "root_cause": "Unsegregated delivery packaging and consumer food containers accumulating in hostel waste bins.",
                "action": "Establish floor-level segregated 2-bin disposal (organic vs recyclable) and partner with BBMP dry waste recyclers.",
                "priority": "MEDIUM",
                "tier": "Tier 1: Operational Quick Win",
                "capex_inr": 500,
                "payback_days": 1.5,
                "responsible_team": "Hostel Facilities & BBMP Ops",
            }
        else:
            return {
                "root_cause": "Unregulated electronic packaging and component scrap in maker & computer labs.",
                "action": "Deploy dedicated electronic waste bins with an authorized KSPCB e-waste recycler.",
                "priority": "MEDIUM",
                "tier": "Tier 1: Operational Quick Win",
                "capex_inr": 0,
                "payback_days": 0.0,
                "responsible_team": "IT & Lab Asset Team",
            }

    else:  # energy
        if occupancy <= 15 and (hour >= 20 or hour <= 6):
            return {
                "root_cause": f"Central HVAC chiller loops and corridor lighting operating at peak capacity on unoccupied floors in {b_name}.",
                "action": "Enforce automated BMS timer shutdown after 20:30; install passive infrared (PIR) occupancy sensors in corridors.",
                "priority": "CRITICAL - HIGH ROI",
                "tier": "Tier 1: Operational Quick Win",
                "capex_inr": 0,
                "payback_days": 0.0,
                "responsible_team": "Electrical & BMS Automation",
            }
        elif b_type in ("lab", "engineering"):
            if "gpu" in b_name.lower() or "computer" in b_name.lower() or "socse" in b_name.lower():
                return {
                    "root_cause": "High-density GPU AI compute nodes and precision CRAC cooling running at full thermal load without active workloads.",
                    "action": "Enable dynamic OS CPU/GPU power-governor downscaling and optimize cold-aisle containment baffles.",
                    "priority": "CRITICAL - HIGH ROI",
                    "tier": "Tier 3: Systems & Workload Tuning",
                    "capex_inr": 0,
                    "payback_days": 0.0,
                    "responsible_team": "HPC & AI Computing Operations",
                }
            return {
                "root_cause": "Mechanical engineering workshop 3-phase lathe compressors and hydraulic test benches idling outside scheduled lab sessions.",
                "action": "Install contactor relay auto-cutoffs on main workshop distribution boards after 18:00.",
                "priority": "HIGH",
                "tier": "Tier 1: Operational Quick Win",
                "capex_inr": 0,
                "payback_days": 0.0,
                "responsible_team": "Workshop Electrical Operations",
            }
        elif b_type == "studio":
            return {
                "root_cause": "Broadcast television studio lighting grids and post-production edit suites left energized overnight.",
                "action": "Implement keycard master disconnect switches for studio production stages and render clusters.",
                "priority": "HIGH",
                "tier": "Tier 1: Operational Quick Win",
                "capex_inr": 800,
                "payback_days": 0.8,
                "responsible_team": "Media Technical Operations",
            }
        else:
            return {
                "root_cause": "Unoptimized baseline electrical load / vampire draw from non-essential peripherals.",
                "action": "Audit sub-meter phase balance and install scheduled relay cutoffs for common areas.",
                "priority": "MEDIUM",
                "tier": "Tier 1: Operational Quick Win",
                "capex_inr": 0,
                "payback_days": 0.0,
                "responsible_team": "Electrical Operations",
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
        f"saving ₹{savings['hourly_loss_inr']}/hr (est. ₹{savings['projected_monthly_savings_inr']:,.0f}/month) "
        f"and abating ~{savings['projected_monthly_co2_kg']} kg CO2e/month."
    )

    return {
        "building_id": row.get("building_id"),
        "building_name": row.get("building_name"),
        "building_type": row.get("building_type"),
        "facility_category": row.get("facility_category", "college"),
        "metric": row.get("metric"),
        "hour": row.get("hour"),
        "reading": row.get("reading"),
        "predicted": row.get("predicted"),
        "excess_units": savings["excess_units"],
        "unit": savings["unit"],
        "hourly_loss_inr": savings["hourly_loss_inr"],
        "projected_monthly_savings_inr": savings["projected_monthly_savings_inr"],
        "co2_kg_hourly": savings.get("co2_kg_hourly", 0.0),
        "projected_monthly_co2_kg": savings.get("projected_monthly_co2_kg", 0.0),
        "root_cause": diagnosis["root_cause"],
        "action": diagnosis["action"],
        "priority": diagnosis["priority"],
        "tier": diagnosis.get("tier", "Tier 1: Operational Quick Win"),
        "capex_inr": diagnosis.get("capex_inr", 0),
        "payback_days": diagnosis.get("payback_days", 0.0),
        "responsible_team": diagnosis.get("responsible_team", "Facilities Operations"),
        "recommendation": rec_text,
    }


def generate_recommendations_for_flagged(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Batch generates recommendations for top flagged rows, sorted by highest financial loss."""
    if "flagged" not in df.columns:
        return pd.DataFrame()
    flagged = df[df["flagged"]].copy()
    if flagged.empty:
        return pd.DataFrame()

    rates = {
        "energy": ENERGY_COST_PER_KWH,
        "water": WATER_COST_PER_LITER,
        "waste": WASTE_COST_PER_KG,
    }
    excess = (flagged["reading"] - flagged["predicted"]).clip(lower=0.0)
    metric_rates = flagged["metric"].map(rates).fillna(ENERGY_COST_PER_KWH)
    flagged["_sort_loss"] = excess * metric_rates

    top_flagged = flagged.nlargest(limit, "_sort_loss")
    recs = [generate_recommendation(row) for _, row in top_flagged.iterrows()]
    recs_df = pd.DataFrame(recs)
    if not recs_df.empty:
        return recs_df.sort_values(by="hourly_loss_inr", ascending=False).reset_index(drop=True)
    return recs_df


def calculate_campus_health_score(analyzed_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive health score, grade, and campus metrics across Energy, Water, and Waste."""
    total_records = len(analyzed_df)
    if total_records == 0:
        return {"health_score": 100, "grade": "A+", "status": "Optimal"}

    flagged_df = analyzed_df[analyzed_df["flagged"]]
    flag_count = len(flagged_df)
    flag_rate = (flag_count / total_records) * 100.0

    # Calculate net excess consumption
    excess_energy_df = flagged_df[flagged_df["metric"] == "energy"]
    excess_energy_kwh = float(max(0.0, (excess_energy_df["reading"] - excess_energy_df["predicted"]).sum())) if not excess_energy_df.empty else 0.0

    excess_water_df = flagged_df[flagged_df["metric"] == "water"]
    excess_water_liters = float(max(0.0, (excess_water_df["reading"] - excess_water_df["predicted"]).sum())) if not excess_water_df.empty else 0.0

    excess_waste_df = flagged_df[flagged_df["metric"] == "waste"]
    excess_waste_kg = float(max(0.0, (excess_waste_df["reading"] - excess_waste_df["predicted"]).sum())) if not excess_waste_df.empty else 0.0

    # Financial loss calculation (normalized to 30 days)
    # Estimate total days represented in analyzed_df
    distinct_hours = total_records / (9 * 3) if total_records > 0 else 1.0
    days_span = max(1.0, distinct_hours / 24.0)
    scaling_to_monthly = 30.0 / days_span

    total_financial_loss = (
        (excess_energy_kwh * ENERGY_COST_PER_KWH) +
        (excess_water_liters * WATER_COST_PER_LITER) +
        (excess_waste_kg * WASTE_COST_PER_KG)
    )
    est_monthly_loss_inr = total_financial_loss * scaling_to_monthly
    est_monthly_co2_kg = (
        (excess_energy_kwh * scaling_to_monthly * CO2_KG_PER_KWH) +
        ((excess_water_liters * scaling_to_monthly * WATER_CO2_G_PER_LITER) / 1000.0) +
        (excess_waste_kg * scaling_to_monthly * WASTE_CO2_KG_PER_KG)
    )

    # Health score calculation (100 base, deduction scaled to flag rate)
    score_deduction = flag_rate * 1.8
    health_score = max(35, min(100, int(round(100 - score_deduction))))

    if health_score >= 90:
        grade = "A+"
        status = "Optimal Efficiency"
        color = "#10B981"
    elif health_score >= 80:
        grade = "A"
        status = "Minor Deviations"
        color = "#3B82F6"
    elif health_score >= 70:
        grade = "B"
        status = "Moderate Inefficiencies"
        color = "#F59E0B"
    elif health_score >= 55:
        grade = "C"
        status = "Significant Waste Detected"
        color = "#F97316"
    else:
        grade = "D"
        status = "Critical Waste Alert"
        color = "#EF4444"

    return {
        "health_score": health_score,
        "grade": grade,
        "status": status,
        "color": color,
        "flag_count": flag_count,
        "flag_rate": round(flag_rate, 1),
        "excess_energy_kwh": round(excess_energy_kwh, 1),
        "excess_water_liters": round(excess_water_liters, 1),
        "excess_waste_kg": round(excess_waste_kg, 1),
        "est_monthly_loss_inr": round(est_monthly_loss_inr, 0),
        "est_monthly_co2_kg": round(est_monthly_co2_kg, 1),
        "est_annual_loss_inr": round(est_monthly_loss_inr * 12, 0),
    }


def get_building_health_matrix(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """Computes operational status and key metrics per campus building across all resources."""
    records = []
    for b_name, group in analyzed_df.groupby("building_name"):
        b_type = group["building_type"].iloc[0]
        f_cat = group["facility_category"].iloc[0] if "facility_category" in group.columns else "college"
        total_flags = group["flagged"].sum()
        total_rows = len(group)
        flag_rate = (total_flags / total_rows * 100.0) if total_rows > 0 else 0.0

        # Energy stats
        e_rows = group[group["metric"] == "energy"]
        e_actual = e_rows["reading"].sum() if not e_rows.empty else 0.0
        e_excess = (e_rows[e_rows["flagged"]]["reading"] - e_rows[e_rows["flagged"]]["predicted"]).sum() if not e_rows.empty else 0.0

        # Water stats
        w_rows = group[group["metric"] == "water"]
        w_actual = w_rows["reading"].sum() if not w_rows.empty else 0.0
        w_excess = (w_rows[w_rows["flagged"]]["reading"] - w_rows[w_rows["flagged"]]["predicted"]).sum() if not w_rows.empty else 0.0

        # Waste stats
        waste_rows = group[group["metric"] == "waste"]
        waste_actual = waste_rows["reading"].sum() if not waste_rows.empty else 0.0
        waste_excess = (waste_rows[waste_rows["flagged"]]["reading"] - waste_rows[waste_rows["flagged"]]["predicted"]).sum() if not waste_rows.empty else 0.0

        loss_inr = (
            (max(0.0, e_excess) * ENERGY_COST_PER_KWH) +
            (max(0.0, w_excess) * WATER_COST_PER_LITER) +
            (max(0.0, waste_excess) * WASTE_COST_PER_KG)
        )

        if flag_rate > 25:
            status = "🔴 Critical"
            badge = "Critical Waste"
        elif flag_rate > 5:
            status = "🟡 Warning"
            badge = "Moderate Drift"
        else:
            status = "🟢 Optimal"
            badge = "Normal Baseline"

        records.append({
            "building_name": b_name,
            "building_type": b_type,
            "facility_category": f_cat,
            "status": status,
            "badge": badge,
            "anomaly_rate_pct": round(flag_rate, 1),
            "flagged_hours": int(total_flags),
            "total_energy_kwh": round(e_actual, 1),
            "excess_energy_kwh": round(max(0.0, e_excess), 1),
            "total_water_l": round(w_actual, 1),
            "excess_water_l": round(max(0.0, w_excess), 1),
            "total_waste_kg": round(waste_actual, 1),
            "excess_waste_kg": round(max(0.0, waste_excess), 1),
            "est_monthly_loss_inr": round(loss_inr, 0),
        })

    return pd.DataFrame(records).sort_values(by="est_monthly_loss_inr", ascending=False)
