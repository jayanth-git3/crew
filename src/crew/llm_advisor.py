"""LLM Advisory Layer for CREW (Campus Resource Efficiency Watch).
Connects the Random Forest anomaly detection baseline to an OpenAI-compatible FreeLLMAPI gateway
(running on http://localhost:3001/v1 or configured endpoint) for deep operational engineering analysis.
Includes instant deterministic fallback to ensure zero downtime when the gateway is offline.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
import pandas as pd

from crew.config import (
    ENERGY_COST_PER_KWH,
    WATER_COST_PER_LITER,
    WASTE_COST_PER_KG,
)
from crew.recommendations import diagnose_root_cause, compute_quantified_savings

DEFAULT_FREELLM_URL = os.environ.get("FREELLM_BASE_URL", "http://localhost:3001/v1")
DEFAULT_FREELLM_KEY = os.environ.get("FREELLM_API_KEY", "")


def format_anomaly_prompt(row: Dict[str, Any] | pd.Series) -> str:
    """Formats a technical engineering prompt from Random Forest detection outputs."""
    b_name = str(row.get("building_name", "Campus Facility"))
    cat = str(row.get("facility_category", row.get("zone", "college"))).capitalize()
    metric = str(row.get("metric", "energy")).lower()
    hour = int(row.get("hour", 0))
    occ = int(row.get("occupancy", 0))
    reading = float(row.get("reading", 0.0))
    predicted = float(row.get("predicted", 0.0))
    dev_pct = float(row.get("deviation_pct", 0.0))

    unit = "kWh" if metric == "energy" else ("Liters" if metric == "water" else "kg")
    rate = ENERGY_COST_PER_KWH if metric == "energy" else (WATER_COST_PER_LITER if metric == "water" else WASTE_COST_PER_KG)
    burn_rate = max(0.0, (reading - predicted)) * rate

    prompt = (
        f"You are the Chief Campus Facilities & Decarbonization Engineer for Presidency University, Bengaluru.\n"
        f"A Random Forest ML baseline has flagged an operational anomaly with the following telemetry:\n"
        f"- Facility: {b_name} ({cat})\n"
        f"- Resource: {metric.upper()}\n"
        f"- Diurnal Time: {hour:02d}:00 (24h clock)\n"
        f"- Active Occupancy: {occ} occupants\n"
        f"- Recorded Reading: {reading:.1f} {unit}\n"
        f"- ML Baseline Prediction: {predicted:.1f} {unit}\n"
        f"- Baseline Deviation: +{dev_pct:.1f}%\n"
        f"- Financial Waste Burn Rate: ₹{burn_rate:.1f}/hour\n\n"
        f"Provide a concise, 3-part technical audit:\n"
        f"1. Probable Engineering Root Cause (specific equipment, valves, BMS timer, or occupancy factor)\n"
        f"2. Immediate Remediation Protocol (actionable directive and responsible operational team)\n"
        f"3. Payback & Decarbonization Horizon (estimated days to recover CAPEX and monthly savings)\n"
        f"Keep the response strictly practical, authoritative, and under 100 words."
    )
    return prompt


def check_freellm_health(base_url: str = DEFAULT_FREELLM_URL, timeout: float = 1.0) -> bool:
    """Probes the FreeLLMAPI gateway models endpoint to check if the server is active."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CREW-Campus-Monitor/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


def query_llm_advisor(
    row: Dict[str, Any] | pd.Series,
    base_url: str = DEFAULT_FREELLM_URL,
    api_key: str = DEFAULT_FREELLM_KEY,
    timeout: float = 4.0,
    model: str = "auto",
) -> Dict[str, Any]:
    """Queries FreeLLMAPI gateway for in-depth engineering analysis with deterministic fallback."""
    savings = compute_quantified_savings(row)
    fallback_diag = diagnose_root_cause(row)
    prompt = format_anomaly_prompt(row)

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "CREW-Campus-Monitor/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a pragmatic, expert campus facilities and decarbonization engineering advisor."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 250,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                return {
                    "source": "freellm_live",
                    "status": "success",
                    "analysis": content,
                    "root_cause": fallback_diag["root_cause"],
                    "action": fallback_diag["action"],
                    "priority": fallback_diag["priority"],
                    "tier": fallback_diag["tier"],
                    "excess_units": savings["excess_units"],
                    "unit": savings["unit"],
                    "hourly_loss_inr": savings["hourly_loss_inr"],
                }
    except Exception:
        pass

    # Seamless deterministic fallback
    summary_text = (
        f"Root Cause: {fallback_diag['root_cause']}\n\n"
        f"Remediation: {fallback_diag['action']}\n\n"
        f"Impact: Eliminates {savings['excess_units']} {savings['unit']}/hr of waste, "
        f"saving ₹{savings['hourly_loss_inr']}/hr (~₹{savings['projected_monthly_savings_inr']:,.0f}/mo) "
        f"with ~{savings['projected_monthly_co2_kg']} kg CO₂e offset."
    )

    return {
        "source": "deterministic_fallback",
        "status": "fallback",
        "analysis": summary_text,
        "root_cause": fallback_diag["root_cause"],
        "action": fallback_diag["action"],
        "priority": fallback_diag["priority"],
        "tier": fallback_diag["tier"],
        "excess_units": savings["excess_units"],
        "unit": savings["unit"],
        "hourly_loss_inr": savings["hourly_loss_inr"],
    }
