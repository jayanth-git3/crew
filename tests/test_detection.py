"""Test suite validating the RandomForestRegressor detection and flagging engine."""
import pytest
import pandas as pd
from crew.data_generator import generate_campus_data
from crew.detection import CampusResourceDetector


@pytest.fixture(scope="module")
def sample_dataset():
    """Generates a small dataset for testing."""
    return generate_campus_data(days=3, seed=99)


def test_detector_training_and_prediction(sample_dataset):
    """Verify that detector fits cleanly and adds required prediction & flag columns."""
    detector = CampusResourceDetector(n_estimators=20, random_state=42)
    detector.fit(sample_dataset)

    flagged_df = detector.predict_and_flag(sample_dataset, threshold_pct=20.0)

    # Check required deliverable columns exist
    assert "predicted" in flagged_df.columns
    assert "deviation_pct" in flagged_df.columns
    assert "flagged" in flagged_df.columns

    # Check predicted values are positive
    assert (flagged_df["predicted"] > 0).all()

    # Check that known rigged anomalies get flagged
    rigged_rows = flagged_df[flagged_df["known_inefficient"]]
    # During anomalous hours, rigged rows should have high deviation (>20%) and be flagged
    assert rigged_rows["flagged"].sum() > 0, "Detector failed to flag known inefficient rows"


def test_detector_deviation_logic():
    """Verify that deviation_pct and flagged boolean strictly follow the formula."""
    detector = CampusResourceDetector(n_estimators=10, random_state=42)
    mini_df = generate_campus_data(days=2, seed=1)
    detector.fit(mini_df)

    flagged_df = detector.predict_and_flag(mini_df, threshold_pct=20.0)

    # Verify deviation formula: ((reading - predicted) / predicted) * 100
    expected_dev = ((flagged_df["reading"] - flagged_df["predicted"]) / flagged_df["predicted"]) * 100.0
    diff = (flagged_df["deviation_pct"] - expected_dev).abs()
    assert (diff < 0.1).all(), "Deviation calculation differs from expected percentage formula"

    # Verify flag condition
    expected_flags = flagged_df["deviation_pct"] > 20.0
    assert (flagged_df["flagged"] == expected_flags).all()
