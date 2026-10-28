"""Configuration for CREW (Campus Resource Efficiency Watch).
Contains Presidency University Bengaluru campus layout, building profiles,
occupancy patterns, and anomaly definitions.
"""
from __future__ import annotations
from typing import Dict, Any, List

CAMPUS_NAME = "Presidency University, Bengaluru"

# Real campus building profiles
BUILDINGS: List[Dict[str, Any]] = [
    {
        "building_id": "PU-ACAD-01",
        "building_name": "Academic Block 1 (Engineering)",
        "building_type": "academic",
        "capacity": 1200,
        "base_energy": 45.0,  # kWh baseline
        "base_water": 300.0,  # Liters/hr baseline
        "energy_per_occ": 0.18,
        "water_per_occ": 1.5,
        "known_inefficient": True,
        "anomaly_reason": "Central HVAC & corridor lighting running on empty floors overnight (21:00 - 05:00)",
        "anomaly_metric": "energy",
        "anomaly_hours": list(range(21, 24)) + list(range(0, 5)),
        "anomaly_multiplier": 3.2,
    },
    {
        "building_id": "PU-ACAD-02",
        "building_name": "Academic Block 2 (Management & Law)",
        "building_type": "academic",
        "capacity": 900,
        "base_energy": 35.0,
        "base_water": 220.0,
        "energy_per_occ": 0.15,
        "water_per_occ": 1.4,
        "known_inefficient": False,
    },
    {
        "building_id": "PU-LAB-CS",
        "building_name": "Computer Science & AI Computing Center",
        "building_type": "lab",
        "capacity": 400,
        "base_energy": 60.0,
        "base_water": 80.0,
        "energy_per_occ": 0.35,
        "water_per_occ": 0.8,
        "known_inefficient": True,
        "anomaly_reason": "High-density GPU racks & precision AC cooling running idle overnight",
        "anomaly_metric": "energy",
        "anomaly_hours": list(range(22, 24)) + list(range(0, 6)),
        "anomaly_multiplier": 2.5,
    },
    {
        "building_id": "PU-LAB-ROBO",
        "building_name": "Robotics & Mechanical Workshops",
        "building_type": "lab",
        "capacity": 250,
        "base_energy": 40.0,
        "base_water": 120.0,
        "energy_per_occ": 0.25,
        "water_per_occ": 1.0,
        "known_inefficient": False,
    },
    {
        "building_id": "PU-HOSTEL-BA",
        "building_name": "Boys Hostel Block A",
        "building_type": "hostel",
        "capacity": 650,
        "base_energy": 25.0,
        "base_water": 500.0,
        "energy_per_occ": 0.12,
        "water_per_occ": 4.5,
        "known_inefficient": True,
        "anomaly_reason": "Overhead water tank float-valve failure & plumbing line leak overnight (01:00 - 05:00)",
        "anomaly_metric": "water",
        "anomaly_hours": [1, 2, 3, 4],
        "anomaly_multiplier": 4.5,
    },
    {
        "building_id": "PU-HOSTEL-BB",
        "building_name": "Boys Hostel Block B",
        "building_type": "hostel",
        "capacity": 600,
        "base_energy": 24.0,
        "base_water": 480.0,
        "energy_per_occ": 0.11,
        "water_per_occ": 4.2,
        "known_inefficient": False,
    },
    {
        "building_id": "PU-HOSTEL-G1",
        "building_name": "Girls Hostel Block",
        "building_type": "hostel",
        "capacity": 550,
        "base_energy": 22.0,
        "base_water": 450.0,
        "energy_per_occ": 0.12,
        "water_per_occ": 4.4,
        "known_inefficient": False,
    },
    {
        "building_id": "PU-CANTEEN",
        "building_name": "Central Food Court & Canteen",
        "building_type": "canteen",
        "capacity": 700,
        "base_energy": 30.0,
        "base_water": 400.0,
        "energy_per_occ": 0.08,
        "water_per_occ": 3.0,
        "known_inefficient": False,
    },
    {
        "building_id": "PU-LIB-ADMIN",
        "building_name": "Central Library & Admin Block",
        "building_type": "academic",
        "capacity": 500,
        "base_energy": 28.0,
        "base_water": 180.0,
        "energy_per_occ": 0.10,
        "water_per_occ": 1.2,
        "known_inefficient": False,
    },
]

# Utility cost benchmarks (Bangalore / BESCOM & BWSSB rates)
ENERGY_COST_PER_KWH = 8.50  # ₹/kWh commercial tariff
WATER_COST_PER_LITER = 0.15  # ₹/Liter (~₹150 per kiloliter)
DEVIATION_FLAG_THRESHOLD_PCT = 20.0  # >20% triggers anomaly flag
