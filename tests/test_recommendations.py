"""Test suite validating the heuristic diagnosis and quantified savings calculator across Energy, Water, and Waste."""
import pytest
from crew.recommendations import compute_quantified_savings, diagnose_root_cause


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


def test_savings_computation_waste():
    """Verify waste savings calculation at ₹12.00/kg."""
    sample_row = {
        "building_id": "PU-CANTEEN",
        "reading": 150.0,
        "predicted": 50.0,
        "metric": "waste",
        "deviation_pct": 200.0,
    }
    savings = compute_quantified_savings(sample_row)

    assert savings["excess_units"] == 100.0
    assert savings["unit"] == "kg"
    assert savings["hourly_loss_inr"] == 1200.0  # 100 * 12.00


def test_diagnose_root_cause_canteen_waste():
    """Verify canteen food waste spoilage diagnosis."""
    sample_row = {
        "building_id": "PU-CANTEEN",
        "building_name": "Central Food Court & Canteen",
        "building_type": "canteen",
        "metric": "waste",
        "hour": 14,
        "occupancy": 600,
    }
    diag = diagnose_root_cause(sample_row)

    assert "food" in diag["root_cause"].lower() or "spoilage" in diag["root_cause"].lower()
    assert "CRITICAL" in diag["priority"]
