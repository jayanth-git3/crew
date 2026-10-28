"""Test suite validating the singular unified RandomForestRegressor detection engine across Energy, Water, and Waste."""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from crew.data_generator import generate_domain_dataset
from crew.detection import CampusResourceDetector


@pytest.fixture(scope="module")
def multi_domain_dataset():
    """Generates a combined multi-domain dataset for testing."""
    df_e = generate_domain_dataset("energy", target_rows=300, seed=1)
    df_w = generate_domain_dataset("water", target_rows=300, seed=2)
    df_waste = generate_domain_dataset("waste", target_rows=300, seed=3)
    return pd.concat([df_e, df_w, df_waste], ignore_index=True)


def test_singular_model_training_and_prediction(multi_domain_dataset):
    """Verify that singular unified model trains across combined records and predicts all 3 metrics."""
    detector = CampusResourceDetector(n_estimators=30, random_state=42)
    detector.fit(multi_domain_dataset)

    flagged_df = detector.predict_and_flag(multi_domain_dataset, threshold_pct=20.0)

    # Check required deliverable columns exist
    assert "predicted" in flagged_df.columns
    assert "deviation_pct" in flagged_df.columns
    assert "flagged" in flagged_df.columns

    # Check predictions are positive
    assert (flagged_df["predicted"] > 0).all()

    # Check that anomalies get flagged in all 3 domains
    for m in ["energy", "water", "waste"]:
        m_rows = flagged_df[(flagged_df["metric"] == m) & flagged_df["known_inefficient"]]
        assert m_rows["flagged"].sum() > 0, f"Detector failed to flag anomalies for {m}"


def test_singular_model_r2_benchmarks(multi_domain_dataset):
    """Verify that the singular unified model achieves high R² fit quality (>= 0.90) overall."""
    detector = CampusResourceDetector(n_estimators=50, random_state=42)
    detector.fit(multi_domain_dataset)

    overall_r2 = detector.metrics["overall"]["r2"]
    assert overall_r2 >= 0.90, f"Overall R² {overall_r2} is below the 0.90 baseline standard"


def test_singular_model_serialization(tmp_path, multi_domain_dataset):
    """Verify that singular detector model artifact can be saved and reloaded with identical predictions."""
    detector = CampusResourceDetector(n_estimators=20, random_state=42)
    detector.fit(multi_domain_dataset)

    model_file = tmp_path / "test_detector.joblib"
    detector.save(str(model_file))
    assert model_file.exists()

    loaded_detector = CampusResourceDetector.load(str(model_file))
    preds_original = detector.predict_and_flag(multi_domain_dataset)
    preds_loaded = loaded_detector.predict_and_flag(multi_domain_dataset)

    max_diff = (preds_original["predicted"] - preds_loaded["predicted"]).abs().max()
    assert max_diff < 0.05, f"Loaded model predictions differ by {max_diff}"
    assert (preds_original["flagged"] == preds_loaded["flagged"]).all()
