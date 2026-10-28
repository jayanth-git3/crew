"""Test suite validating compliance with the locked IEEE hackathon data contract across all 3 domains."""
from pathlib import Path
import pytest
import pandas as pd
from crew.data_generator import generate_domain_dataset, generate_all_datasets


def test_schema_columns_and_contract():
    """Verify that domain datasets strictly match the locked IEEE schema."""
    df = generate_domain_dataset("energy", target_rows=100, seed=123)

    expected_columns = [
        "building_id",
        "building_name",
        "building_type",
        "facility_category",
        "occupancy",
        "hour",
        "metric",
        "reading",
        "known_inefficient",
    ]

    assert list(df.columns) == expected_columns, f"Columns mismatch. Expected {expected_columns}, got {list(df.columns)}"


def test_domain_files_row_counts_and_separation():
    """Verify that 3 independent domain files each contain 10,000 rows and combined has 30,000 rows."""
    data_dir = Path("data")
    assert (data_dir / "energy_data.csv").exists(), "energy_data.csv missing"
    assert (data_dir / "water_data.csv").exists(), "water_data.csv missing"
    assert (data_dir / "waste_data.csv").exists(), "waste_data.csv missing"
    assert (data_dir / "campus_data.csv").exists(), "campus_data.csv missing"

    df_e = pd.read_csv(data_dir / "energy_data.csv")
    df_w = pd.read_csv(data_dir / "water_data.csv")
    df_waste = pd.read_csv(data_dir / "waste_data.csv")
    df_combined = pd.read_csv(data_dir / "campus_data.csv")

    assert len(df_e) == 10000, f"Expected 10,000 energy rows, got {len(df_e)}"
    assert len(df_w) == 10000, f"Expected 10,000 water rows, got {len(df_w)}"
    assert len(df_waste) == 10000, f"Expected 10,000 waste rows, got {len(df_waste)}"
    assert len(df_combined) == 30000, f"Expected 30,000 combined rows, got {len(df_combined)}"

    # Check metrics are properly separated
    assert (df_e["metric"] == "energy").all()
    assert (df_w["metric"] == "water").all()
    assert (df_waste["metric"] == "waste").all()


def test_facility_categories_present():
    """Verify that facilities are properly categorized into college, hostel, and canteen."""
    df = pd.read_csv("data/campus_data.csv")
    expected_categories = {"college", "hostel", "canteen"}
    assert set(df["facility_category"].unique()) == expected_categories
