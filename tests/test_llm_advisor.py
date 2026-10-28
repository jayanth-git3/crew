"""Unit tests for the LLM advisory layer and its resilient deterministic fallback."""
import pytest
from crew.llm_advisor import (
    format_anomaly_prompt,
    query_llm_advisor,
    check_freellm_health,
)


def test_format_anomaly_prompt():
    """Verify prompt formatting includes all technical telemetry parameters."""
    sample_row = {
        "building_name": "Academic Block 1 (Engineering)",
        "facility_category": "college",
        "metric": "energy",
        "hour": 22,
        "occupancy": 12,
        "reading": 145.0,
        "predicted": 42.0,
        "deviation_pct": 245.2,
    }
    prompt = format_anomaly_prompt(sample_row)

    assert "Academic Block 1" in prompt
    assert "ENERGY" in prompt
    assert "22:00" in prompt
    assert "145.0 kWh" in prompt
    assert "42.0 kWh" in prompt
    assert "+245.2%" in prompt
    assert "₹" in prompt


def test_deterministic_fallback_when_offline():
    """Verify query_llm_advisor falls back cleanly to deterministic heuristics if gateway is offline."""
    sample_row = {
        "building_name": "Boys Hostel Block A",
        "building_type": "hostel",
        "facility_category": "hostel",
        "metric": "water",
        "hour": 3,
        "occupancy": 500,
        "reading": 2200.0,
        "predicted": 500.0,
        "deviation_pct": 340.0,
    }
    # Point to a closed port
    result = query_llm_advisor(
        sample_row,
        base_url="http://127.0.0.1:54321/v1",
        timeout=0.5,
    )

    assert result["status"] == "fallback"
    assert result["source"] == "deterministic_fallback"
    assert "float" in result["analysis"].lower() or "valve" in result["analysis"].lower() or "tank" in result["analysis"].lower()
    assert result["excess_units"] == 1700.0
    assert result["unit"] == "Liters"


def test_check_freellm_health_offline():
    """Verify health probe returns False for offline server without hanging."""
    is_up = check_freellm_health("http://127.0.0.1:54321/v1", timeout=0.2)
    assert is_up is False
