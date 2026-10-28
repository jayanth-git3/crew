"""Test suite validating the RandomForestRegressor detection and flagging engine."""
from pathlib import Path
import pytest
import pandas as pd
from crew.data_generator import generate_campus_data
from crew.detection import CampusResourceDetector


@pytest.fixture(scope="module")
def sample_dataset():
    """Generates a dataset for testing."""
    return generate_campus_data(days=4, seed=99)


def test_detector_training_and_prediction(sample_dataset):
    """Verify that detector fits cleanly and adds required prediction & flag columns."""
    detector = CampusResourceDetector(n_estimators=30, random_state=42)
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
    assert rigged_rows["flagged"].sum() > 0, "Detector failed to flag known inefficient rows"


def test_detector_accuracy_metrics(sample_dataset):
    """Verify that models achieve high R² fit quality (>= 0.90) on normal baseline data."""
    detector = CampusResourceDetector(n_estimators=50, random_state=42)
    detector.fit(sample_dataset)

    assert "energy" in detector.metrics
    assert "water" in detector.metrics

    energy_r2 = detector.metrics["energy"]["r2"]
    water_r2 = detector.metrics["water"]["r2"]

    assert energy_r2 >= 0.90, f"Energy R² {energy_r2} is below the 0.90 baseline standard"
    assert water_r2 >= 0.90, f"Water R² {water_r2} is below the 0.90 baseline standard"


def test_detector_serialization(tmp_path, sample_dataset):
    """Verify that detector model artifact can be saved and reloaded with identical predictions."""
    detector = CampusResourceDetector(n_estimators=20, random_state=42)
    detector.fit(sample_dataset)

    model_file = tmp_path / "test_detector.joblib"
    detector.save(str(model_file))
    assert model_file.exists()

    loaded_detector = CampusResourceDetector.load(str(model_file))
    preds_original = detector.predict_and_flag(sample_dataset)
    preds_loaded = loaded_detector.predict_and_flag(sample_dataset)

    assert (preds_original["predicted"] == preds_loaded["predicted"]).all()
    assert (preds_original["flagged"] == preds_loaded["flagged"]).all()


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
