"""Tests validating UI data contracts, recommendation optimization, and performance contracts."""
import time
import pandas as pd
import numpy as np
import pytest

from crew.config import (
    ENERGY_COST_PER_KWH,
    WATER_COST_PER_LITER,
    WASTE_COST_PER_KG,
    BUILDINGS,
)
from crew.recommendations import (
    generate_recommendations_for_flagged,
    calculate_campus_health_score,
    get_building_health_matrix,
)


def _generate_mock_analyzed_df(n_rows: int = 3000) -> pd.DataFrame:
    """Creates a mock analyzed DataFrame simulating model output."""
    np.random.seed(42)
    b_names = [b["building_name"] for b in BUILDINGS]
    metrics = ["energy", "water", "waste"]

    rows = []
    for i in range(n_rows):
        b = np.random.choice(b_names)
        m = np.random.choice(metrics)
        hour = np.random.randint(0, 24)
        pred = np.random.uniform(20.0, 100.0)
        # 25% anomaly rate
        is_flagged = np.random.random() < 0.25
        reading = pred * np.random.uniform(1.3, 2.0) if is_flagged else pred * np.random.uniform(0.9, 1.1)
        dev_pct = ((reading - pred) / pred) * 100.0

        rows.append({
            "building_id": f"BLD-{i%9:02d}",
            "building_name": b,
            "building_type": "academic" if "Academic" in b else ("hostel" if "Hostel" in b else "canteen"),
            "facility_category": "college" if "Academic" in b else ("hostel" if "Hostel" in b else "canteen"),
            "metric": m,
            "hour": hour,
            "occupancy": int(np.random.uniform(10, 500)),
            "reading": round(reading, 2),
            "predicted": round(pred, 2),
            "deviation_pct": round(dev_pct, 2),
            "flagged": is_flagged,
        })
    return pd.DataFrame(rows)


def test_recommendations_vectorization_performance():
    """Verify that generate_recommendations_for_flagged executes rapidly on large datasets."""
    df = _generate_mock_analyzed_df(n_rows=5000)
    flagged_count = df["flagged"].sum()
    assert flagged_count > 500

    start = time.perf_counter()
    recs = generate_recommendations_for_flagged(df, limit=50)
    elapsed = time.perf_counter() - start

    assert len(recs) <= 50
    assert not recs.empty
    # Must execute in under 100ms (previous iterrows took >300ms)
    assert elapsed < 0.10, f"Recommendation generation took {elapsed:.3f}s (expected < 0.10s)"

    # Verify descending sort by financial loss
    losses = recs["hourly_loss_inr"].tolist()
    assert losses == sorted(losses, reverse=True)


def test_recommendations_contract_columns():
    """Verify that generated recommendations contain all columns required by the UI table."""
    df = _generate_mock_analyzed_df(n_rows=200)
    recs = generate_recommendations_for_flagged(df, limit=10)

    required_columns = [
        "building_name",
        "facility_category",
        "metric",
        "hour",
        "reading",
        "predicted",
        "excess_units",
        "unit",
        "hourly_loss_inr",
        "priority",
        "tier",
        "action",
    ]
    for col in required_columns:
        assert col in recs.columns, f"Missing required column: {col}"
        assert not recs[col].isnull().all()


def test_campus_health_score_numeric_validity():
    """Verify campus health score and loss metrics are non-negative and bounded."""
    df = _generate_mock_analyzed_df(n_rows=500)
    stats = calculate_campus_health_score(df)

    assert 0 <= stats["health_score"] <= 100
    assert stats["grade"] in ["A+", "A", "B", "C", "D"]
    assert stats["excess_energy_kwh"] >= 0.0
    assert stats["excess_water_liters"] >= 0.0
    assert stats["excess_waste_kg"] >= 0.0
    assert stats["est_monthly_loss_inr"] >= 0.0
    assert abs(stats["est_annual_loss_inr"] - stats["est_monthly_loss_inr"] * 12.0) <= 12.0


def test_building_health_matrix_structure():
    """Verify building health matrix returns expected columns for all campus facilities."""
    df = _generate_mock_analyzed_df(n_rows=500)
    matrix = get_building_health_matrix(df)

    assert len(matrix) == len(BUILDINGS)
    expected_cols = ["building_name", "building_type", "anomaly_rate_pct", "status", "badge", "est_monthly_loss_inr"]
    for col in expected_cols:
        assert col in matrix.columns


def test_no_encoding_artifacts_in_app():
    """Verify app.py source contains no mojibake encoding artifacts."""
    from pathlib import Path
    app_path = Path("src/crew/app.py")
    if not app_path.exists():
        app_path = Path("own/crew/src/crew/app.py")
    text = app_path.read_text(encoding="utf-8")

    forbidden = ["ΓÜí", "≡ƒÅó", "Γé╣", "ΓÖ╗", "≡ƒÆº"]
    for artifact in forbidden:
        assert artifact not in text, f"Found corrupted encoding artifact: {artifact}"


def test_app_renders_without_exceptions():
    """Verify Streamlit app runs from top-to-bottom headlessly without raising unhandled exceptions."""
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    app_path = (Path(__file__).parent.parent / "src" / "crew" / "app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"App execution raised exceptions: {[e.value for e in at.exception]}"

