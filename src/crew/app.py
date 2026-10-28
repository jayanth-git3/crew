"""CREW: Campus Resource Efficiency Watch — Streamlit Dashboard.
Interactive 3-Tab Operational Dashboard for the IEEE 5-Hour Hackathon.

Tab 1: Executive Overview & Real-Time Alerts (KPI cards, system architecture, alert feed)
Tab 2: Deep Dive & Inefficiency Analysis (Actual vs Predicted curves, building breakdown)
Tab 3: Actionable Recommendations & Projected Savings (Quantified heuristics & action items)
"""
from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import numpy as np

from crew.config import (
    CAMPUS_NAME,
    ENERGY_COST_PER_KWH,
    WATER_COST_PER_LITER,
    DEVIATION_FLAG_THRESHOLD_PCT,
)
from crew.data_generator import generate_campus_data, save_dataset
from crew.detection import CampusResourceDetector
from crew.recommendations import generate_recommendations_for_flagged

# Streamlit Page Setup
st.set_page_config(
    page_title="CREW | Campus Resource Efficiency Watch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #0d6efd;
        margin-bottom: 12px;
    }
    .stAlert {
        border-radius: 8px;
    }
    .badge-critical {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-high {
        background-color: #fd7e14;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_or_generate_data() -> pd.DataFrame:
    """Loads cached data or generates synthetic campus telemetry."""
    data_file = "data/campus_data.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
    else:
        df = generate_campus_data(days=7, seed=42)
        save_dataset(df, data_file)
    return df


@st.cache_resource
def get_trained_detector(df: pd.DataFrame) -> CampusResourceDetector:
    """Trains or loads the dual RandomForestRegressor models."""
    detector = CampusResourceDetector()
    detector.fit(df)
    return detector


def main():
    # Header Banner
    st.title("⚡ CREW: Campus Resource Efficiency Watch")
    st.caption(f"**Campus Target:** {CAMPUS_NAME} | **Engine:** Dual RandomForest Baseline + Heuristic Diagnostics")

    # Load telemetry and run model
    raw_df = load_or_generate_data()
    detector = get_trained_detector(raw_df)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Telemetry Controls")
        st.success("🟢 System Status: Model Active")

        # Metric Filter
        metric_choice = st.radio(
            "Select Metric:",
            ["All Metrics", "Energy (kWh)", "Water (Liters)"],
            index=0,
        )

        # Deviation Threshold Slider
        threshold = st.slider(
            "Anomaly Deviation Threshold (%)",
            min_value=10,
            max_value=50,
            value=int(DEVIATION_FLAG_THRESHOLD_PCT),
            step=5,
            help="Flag readings that exceed model predictions by more than this percentage.",
        )

        # Building Filter
        all_buildings = sorted(raw_df["building_name"].unique())
        selected_buildings = st.multiselect(
            "Filter Buildings:",
            options=all_buildings,
            default=all_buildings,
        )

        # Hour Filter
        hour_filter = st.slider(
            "Filter Hour of Day (24h):",
            min_value=0,
            max_value=23,
            value=(0, 23),
        )

        st.divider()
        st.markdown("**IEEE 5-Hour Hackathon**")
        st.markdown("🏆 *Round 1 Deliverable: Live Skeleton & Architecture*")

        if st.button("🔄 Regenerate Fresh Telemetry"):
            st.cache_data.clear()
            st.cache_resource.clear()
            fresh_df = generate_campus_data(days=7)
            save_dataset(fresh_df)
            st.rerun()

    # Run detection on filtered threshold
    analyzed_df = detector.predict_and_flag(raw_df, threshold_pct=float(threshold))

    # Apply UI filters
    filtered_df = analyzed_df[
        (analyzed_df["building_name"].isin(selected_buildings)) &
        (analyzed_df["hour"] >= hour_filter[0]) &
        (analyzed_df["hour"] <= hour_filter[1])
    ]

    if metric_choice == "Energy (kWh)":
        filtered_df = filtered_df[filtered_df["metric"] == "energy"]
    elif metric_choice == "Water (Liters)":
        filtered_df = filtered_df[filtered_df["metric"] == "water"]

    flagged_df = filtered_df[filtered_df["flagged"]]

    # 3-Tab Operational Layout
    tab1, tab2, tab3 = st.tabs([
        "📊 Tab 1: Executive Overview & Real-Time Alerts",
        "📈 Tab 2: Deep Dive & Inefficiency Analysis",
        "💡 Tab 3: Actionable Recommendations & Savings",
    ])

    # ---------------- TAB 1: EXECUTIVE OVERVIEW ----------------
    with tab1:
        st.subheader("Campus Resource Health Overview")

        # Top KPIs
        total_records = len(filtered_df)
        total_anomalies = len(flagged_df)
        anomaly_rate = (total_anomalies / total_records * 100) if total_records > 0 else 0

        # Calculate excess consumption
        excess_energy = flagged_df[flagged_df["metric"] == "energy"]
        excess_energy_val = (excess_energy["reading"] - excess_energy["predicted"]).sum()
        excess_water = flagged_df[flagged_df["metric"] == "water"]
        excess_water_val = (excess_water["reading"] - excess_water["predicted"]).sum()

        est_waste_inr = (excess_energy_val * ENERGY_COST_PER_KWH) + (excess_water_val * WATER_COST_PER_LITER)
        est_monthly_loss_inr = est_waste_inr * 4.3  # approx 1 week sample scaled to month

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Telemetry Records", f"{total_records:,}")
        col2.metric("Flagged Inefficiencies", f"{total_anomalies:,}", f"{anomaly_rate:.1f}% flag rate", delta_color="inverse")
        col3.metric("Excess Resource Waste", f"{excess_energy_val:,.0f} kWh | {excess_water_val:,.0f} L")
        col4.metric("Est. Monthly Financial Loss", f"₹{est_monthly_loss_inr:,.0f}", delta="Action Required", delta_color="inverse")

        st.divider()

        # Architecture diagram & Active alerts in two columns
        col_diag, col_alerts = st.columns([1, 1])

        with col_diag:
            st.markdown("#### 🏗️ System Architecture Flow")
            st.markdown("""
```mermaid
flowchart TD
    A[Presidency University Sensors / Telemetry] --> B[Dual RandomForestRegressor Baseline]
    B --> C{Actual vs Predicted > 20%?}
    C -- Flagged Anomaly --> D[Heuristic Diagnostic & Savings Engine]
    C -- Normal Load --> E[Operational Baseline Log]
    D --> F[Quantified Fix Action Items]
    D --> G[Streamlit Executive Dashboard]
    F --> G
```
            """)

        with col_alerts:
            st.markdown("#### 🚨 Top Active Inefficiency Alerts")
            if flagged_df.empty:
                st.success("No anomalies detected for current filter settings.")
            else:
                top_flags = flagged_df.sort_values(by="deviation_pct", ascending=False).head(4)
                for _, alert in top_flags.iterrows():
                    metric_icon = "⚡" if alert["metric"] == "energy" else "💧"
                    unit = "kWh" if alert["metric"] == "energy" else "L"
                    st.warning(
                        f"**{metric_icon} {alert['building_name']}** — Hour {alert['hour']:02d}:00\n\n"
                        f"Actual: **{alert['reading']:.1f} {unit}** vs Predicted: **{alert['predicted']:.1f} {unit}** "
                        f"(**+{alert['deviation_pct']:.1f}% deviation**)"
                    )

    # ---------------- TAB 2: DEEP DIVE & ANALYSIS ----------------
    with tab2:
        st.subheader("Inefficiency Deep Dive & Telemetry Curves")

        # Building Selector for drill-down
        drill_building = st.selectbox(
            "Select Building for Hourly Consumption Curves:",
            options=selected_buildings,
            index=0 if selected_buildings else 0,
        )

        if drill_building:
            b_data = filtered_df[filtered_df["building_name"] == drill_building]
            b_hourly = b_data.groupby(["hour", "metric"])[["reading", "predicted"]].mean().reset_index()

            col_e, col_w = st.columns(2)

            with col_e:
                st.markdown(f"**⚡ Energy Curve (kWh): Actual vs Predicted — {drill_building}**")
                e_hourly = b_hourly[b_hourly["metric"] == "energy"]
                if not e_hourly.empty:
                    chart_data = e_hourly.set_index("hour")[["reading", "predicted"]]
                    chart_data.columns = ["Actual (kWh)", "Predicted Baseline (kWh)"]
                    st.line_chart(chart_data)
                else:
                    st.info("No energy data for selected building.")

            with col_w:
                st.markdown(f"**💧 Water Curve (Liters): Actual vs Predicted — {drill_building}**")
                w_hourly = b_hourly[b_hourly["metric"] == "water"]
                if not w_hourly.empty:
                    chart_data = w_hourly.set_index("hour")[["reading", "predicted"]]
                    chart_data.columns = ["Actual (Liters)", "Predicted Baseline (Liters)"]
                    st.line_chart(chart_data)
                else:
                    st.info("No water data for selected building.")

        st.divider()

        # Inefficiency summary across all buildings
        st.markdown("#### 🏢 Building Deviation Comparison")
        b_dev = filtered_df.groupby("building_name")[["deviation_pct"]].mean().sort_values(by="deviation_pct", ascending=False)
        st.bar_chart(b_dev)

    # ---------------- TAB 3: RECOMMENDATIONS & SAVINGS ----------------
    with tab3:
        st.subheader("💡 Actionable Recommendations & Projected Savings")
        st.write("Quantified operational interventions generated per detected anomaly to eliminate waste and restore normal baseline.")

        recs_df = generate_recommendations_for_flagged(filtered_df, limit=30)

        if recs_df.empty:
            st.info("No flagged anomalies to generate recommendations for.")
        else:
            # Summary Metrics for recommendations
            total_monthly_savings = recs_df["projected_monthly_savings_inr"].sum()
            st.success(f"💰 **Total Potential Monthly Savings Identified: ₹{total_monthly_savings:,.0f}** across {len(recs_df)} prioritized actions.")

            # Display recommendation cards
            for _, r in recs_df.head(6).iterrows():
                badge_class = "badge-critical" if "CRITICAL" in r["priority"] else "badge-high"
                metric_label = "⚡ Energy" if r["metric"] == "energy" else "💧 Water"

                with st.expander(f"{r['building_name']} — {metric_label} at Hour {r['hour']:02d}:00 (Save ₹{r['hourly_loss_inr']:.0f}/hr)", expanded=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Root Cause:** {r['root_cause']}")
                        st.markdown(f"**Action Required:** `{r['action']}`")
                        st.markdown(f"**Detailed Recommendation:** {r['recommendation']}")
                    with c2:
                        st.metric("Excess Usage", f"{r['excess_units']:.1f} {r['unit']}/hr")
                        st.metric("Est. Monthly Fix ROI", f"₹{r['projected_monthly_savings_inr']:,.0f}")

            st.divider()
            st.markdown("#### 📋 Prioritized Action Items Table")
            display_cols = [
                "building_name", "metric", "hour", "reading", "predicted",
                "excess_units", "unit", "hourly_loss_inr", "priority", "action"
            ]
            st.dataframe(recs_df[display_cols], use_container_width=True)

            # CSV Download Button
            csv_data = recs_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Actionable Fixes (CSV)",
                data=csv_data,
                file_name="crew_campus_action_items.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
