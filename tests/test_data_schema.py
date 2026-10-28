"""Test suite validating compliance with the locked IEEE hackathon data contract."""
import pytest
import pandas as pd
from crew.data_generator import generate_campus_data


def test_schema_columns_and_contract():
    """Verify that generated campus data matches the exact locked schema."""
    df = generate_campus_data(days=1, seed=123)

    expected_columns = [
        "building_id",
        "building_name",
        "building_type",
        "occupancy",
        "hour",
        "metric",
        "reading",
        "known_inefficient",
    ]

    assert list(df.columns) == expected_columns, f"Columns mismatch. Expected {expected_columns}, got {list(df.columns)}"


def test_schema_types_and_ranges():
    """Verify column data types and valid domain ranges."""
    df = generate_campus_data(days=1, seed=123)

    # building_id, name, type are non-empty strings
    assert df["building_id"].apply(lambda x: isinstance(x, str) and len(x) > 0).all()
    assert df["building_name"].apply(lambda x: isinstance(x, str) and len(x) > 0).all()
    assert df["building_type"].apply(lambda x: isinstance(x, str) and len(x) > 0).all()

    # occupancy is non-negative int
    assert (df["occupancy"] >= 0).all()
    assert pd.api.types.is_integer_dtype(df["occupancy"])

    # hour is 0 to 23
    assert (df["hour"] >= 0).all() and (df["hour"] <= 23).all()
    assert pd.api.types.is_integer_dtype(df["hour"])

    # metric is either 'energy' or 'water'
    valid_metrics = {"energy", "water"}
    assert set(df["metric"].unique()).issubset(valid_metrics)

    # reading is positive float
    assert (df["reading"] > 0).all()
    assert pd.api.types.is_float_dtype(df["reading"])

    # known_inefficient is boolean
    assert pd.api.types.is_bool_dtype(df["known_inefficient"])


def test_rigged_anomalies_count():
    """Verify that exactly 2 to 3 buildings are designated as known_inefficient."""
    df = generate_campus_data(days=1, seed=123)
    inefficient_buildings = df[df["known_inefficient"]]["building_id"].unique()
    assert 2 <= len(inefficient_buildings) <= 3, f"Expected 2-3 rigged buildings, found {len(inefficient_buildings)}"
