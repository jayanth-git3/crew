# CREW: Campus Resource Efficiency Watch ⚡💧

AI-powered telemetry analyzer that monitors campus energy and water consumption, flags operational anomalies, and recommends quantified, actionable interventions.

Built for the **IEEE 5-Hour Hackathon** at Presidency University, Bengaluru.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart LR
    A[Synthetic Campus Telemetry<br/>Presidency University] --> B[Dual RandomForest Regressors<br/>Energy & Water Baselines]
    B --> C{Actual vs Predicted<br/>Deviation > 20%?}
    C -- Yes (Anomaly) --> D[Heuristic Diagnostic & Savings Engine]
    C -- No (Normal) --> E[Baseline Archive]
    D --> F[Quantified Action Items<br/>₹ & CO2 Savings]
    D --> G[Streamlit Executive Command Center]
    F --> G
```

---

## 📋 Data Contract (Locked Schema)

| Column | Type | Notes |
|---|---|---|
| `building_id` | string | Unique identifier per building or hostel block (e.g. `PU-ACAD-01`) |
| `building_name` | string | Real Presidency University Bengaluru facility name |
| `building_type` | string | `academic`, `lab`, `hostel`, `canteen` |
| `occupancy` | int | Real-time / hourly headcount associated with building |
| `hour` | int | 0 to 23 |
| `metric` | string | `"energy"` or `"water"` |
| `reading` | float | Actual recorded usage (kWh for energy, Liters for water) |
| `known_inefficient` | bool | `True` for rigged anomalies (nighttime leaks, chiller waste) |

---

## 👥 Hackathon Team Layers

| Layer | Owner | Component | Deliverable |
|---|---|---|---|
| **Layer 1: Data** | Member 1 | `crew.data_generator` | `data/campus_data.csv` matching locked schema |
| **Layer 2: Detection** | Member 2 | `crew.detection` | `RandomForestRegressor` with `predicted`, `deviation_pct`, `flagged` |
| **Layer 3: Recommendation** | Member 3 | `crew.recommendations` | Quantified financial (₹), unit, and CO₂ savings per anomaly + root cause diagnosis |
| **Layer 4: Dashboard** | Member 4 | `crew.app` | Executive Command Center with Plotly heatmaps, telemetry curves & ROI simulator |

---

## 📊 Dashboard Information Architecture (Version 2)

1. **Top Executive KPI Bar**:
   - Campus Resource Health Score (0-100 & letter grade)
   - Total Energy Inefficiency (kWh)
   - Total Water Inefficiency (Liters)
   - Estimated Monthly Financial Loss (₹) & Annual Loss (₹)
   - CO₂ Abatement Potential (kg CO₂e / month)

2. **Tab 1: 🏢 Campus Health & Alert Feed**:
   - 9-Building Facility Health Matrix (Live status badges: 🟢 Optimal, 🟡 Warning, 🔴 Critical)
   - 24-Hour Spatio-Temporal Inefficiency Heatmap (Hour vs Facility deviation)
   - Segmented Alert Feed filterable by Critical, Energy, and Water

3. **Tab 2: 📈 Telemetry Diagnostics & Heatmaps**:
   - Facility Drill-down Selector with capacity and baseload metadata
   - Dual Interactive Plotly curves: Actual vs Predicted baseline with shaded excess waste fill
   - Nighttime Parasitic / Vampire Load Analysis (Midnight vs Day ratio)
   - Facility category consumption breakdown (Donut charts)

4. **Tab 3: 💡 Measurable Interventions & ROI Simulator**:
   - 3 Prioritized Action Tiers (Tier 1: Zero-Cost Quick Wins, Tier 2: Targeted Repairs, Tier 3: Compute & Workload Tuning)
   - Interactive "What-If" ROI Simulator: dynamically simulates ₹ budget saved, CO₂ avoided, and payback days
   - Prioritized Action Plan Data Table with CSV export

5. **Tab 4: 🔬 ML Model Transparency & Benchmarks**:
   - Clean baseline isolation methodology documentation
   - Feature weights (`building_type`, `occupancy`, `hour`) & pipeline audit

---

## 🚀 Quickstart

### 1. Requirements
- Python >= 3.11 (Managed via `uv`)

### 2. Generate Dataset
```bash
uv run python -m crew.data_generator
```

### 3. Run Detection Model
```bash
uv run python -m crew.detection
```

### 4. Launch Streamlit Dashboard
```bash
uv run streamlit run src/crew/app.py
```

### 5. Run Verification Tests
```bash
uv run pytest tests/
```
