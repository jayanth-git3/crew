"""Test suite validating the heuristic diagnosis and quantified savings calculator."""
import pytest
import pandas as pd
from crew.recommendations import compute_quantified_savings, diagnose_root_cause, generate_recommendation


def test_savings_computation_energy():
    """Verify energy savings calculation at ₹8.50/kWh."""
    sample_row = {
        "building_id": "PU-ACAD-01",
        "reading": 120.0,
        "predicted": 40.0,
        "metric": "energy",
        "deviation_pct": 200.0,
    }
    savings = compute_quantified_savings(sample_row)

    assert savings["excess_units"] == 80.0
    assert savings["unit"] == "kWh"
    assert savings["hourly_loss_inr"] == 680.0  # 80 * 8.50


def test_savings_computation_water():
    """Verify water savings calculation at ₹0.15/Liter."""
    sample_row = {
        "building_id": "PU-HOSTEL-BA",
        "reading": 2500.0,
        "predicted": 500.0,
        "metric": "water",
        "deviation_pct": 400.0,
    }
    savings = compute_quantified_savings(sample_row)

    assert savings["excess_units"] == 2000.0
    assert savings["unit"] == "Liters"
    assert savings["hourly_loss_inr"] == 300.0  # 2000 * 0.15


def test_diagnose_root_cause_hostel_night_leak():
    """Verify hostel overnight water surge diagnosis."""
    sample_row = {
        "building_id": "PU-HOSTEL-BA",
        "building_name": "Boys Hostel Block A",
        "building_type": "hostel",
        "metric": "water",
        "hour": 2,
        "occupancy": 550,
    }
    diag = diagnose_root_cause(sample_row)

    assert "float-valve" in diag["root_cause"].lower() or "tank" in diag["root_cause"].lower()
    assert "HIGH" in diag["priority"]
