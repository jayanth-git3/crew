"""CREW: Campus Resource Efficiency Watch — Streamlit Dashboard.
Interactive 3-Tab Operational Dashboard for the IEEE 5-Hour Hackathon.

Multi-Domain Analytics across Energy (kWh), Water (Liters), and Waste (kg)
powered by a Singular Unified RandomForest Machine Learning Model.
"""
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from crew.config import (
    CAMPUS_NAME,
    ENERGY_COST_PER_KWH,
    WATER_COST_PER_LITER,
    WASTE_COST_PER_KG,
    DEVIATION_FLAG_THRESHOLD_PCT,
)
from crew.data_generator import generate_all_datasets
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
    """Loads 3 domain datasets (Energy, Water, Waste) or generates 3x 10k files."""
    data_dir = Path("data")
    energy_f = data_dir / "energy_data.csv"
    water_f = data_dir / "water_data.csv"
    waste_f = data_dir / "waste_data.csv"

    if energy_f.exists() and water_f.exists() and waste_f.exists():
        df_e = pd.read_csv(energy_f)
        df_w = pd.read_csv(water_f)
        df_waste = pd.read_csv(waste_f)
        combined_df = pd.concat([df_e, df_w, df_waste], ignore_index=True)
    else:
        datasets = generate_all_datasets(target_per_domain=10000)
        combined_df = datasets["combined"]

    return combined_df


@st.cache_resource
def get_trained_detector(df: pd.DataFrame) -> CampusResourceDetector:
    """Loads cached pre-trained singular model artifact or fits if absent."""
    model_file = "models/detector.joblib"
    if os.path.exists(model_file):
        try:
            return CampusResourceDetector.load(model_file)
        except Exception:
            pass
    detector = CampusResourceDetector()
    detector.fit(df)
    detector.save(model_file)
    return detector


def main():
    # Header Banner
    st.title("⚡ CREW: Campus Resource Efficiency Watch")
    st.caption(f"**Campus Target:** {CAMPUS_NAME} | **Multi-Domain Intelligence:** Energy • Water • Waste | **Engine:** Singular Unified RandomForest")

    # Load telemetry and model
    raw_df = load_or_generate_data()
    detector = get_trained_detector(raw_df)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Telemetry Controls")
        st.success("🟢 Singular ML Model: Active")

        if detector.metrics:
            st.markdown("##### 🎯 Model Accuracy (R² Benchmark)")
            overall_r2 = detector.metrics.get("overall", {}).get("r2", 0.99)
            st.caption(f"Unified Model: **{overall_r2 * 100:.1f}%** fit across 30k records")
            with st.expander("Domain-Specific R² Scores"):
                for m in ["energy", "water", "waste"]:
                    r2_score = detector.metrics.get(m, {}).get("r2", 0.99)
                    icon = "⚡" if m == "energy" else ("💧" if m == "water" else "♻️")
                    st.write(f"{icon} {m.title()}: **{r2_score * 100:.2f}%**")

        # Metric Filter
        metric_choice = st.radio(
            "Select Resource Domain:",
            ["All Metrics", "Energy (kWh)", "Water (Liters)", "Waste (kg)"],
            index=0,
        )

        # Facility Category Filter
        facility_choice = st.selectbox(
            "Filter Facility Category:",
            ["All Facilities", "College Blocks", "Hostels", "Canteen"],
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
        st.markdown(f"📁 *Dataset Size: {len(raw_df):,} Total Telemetry Records (3x 10k Files)*")

        if st.button("🔄 Regenerate & Retrain (30k Rows)"):
            st.cache_data.clear()
            st.cache_resource.clear()
            fresh_datasets = generate_all_datasets(target_per_domain=10000)
            fresh_df = fresh_datasets["combined"]
            detector = CampusResourceDetector()
            detector.fit(fresh_df)
            detector.save("models/detector.joblib")
            st.rerun()

    # Run singular model detection
    analyzed_df = detector.predict_and_flag(raw_df, threshold_pct=float(threshold))

    # Apply UI filters
    filtered_df = analyzed_df[
        (analyzed_df["building_name"].isin(selected_buildings)) &
        (analyzed_df["hour"] >= hour_filter[0]) &
        (analyzed_df["hour"] <= hour_filter[1])
    ]

    # Facility Category filter
    if facility_choice == "College Blocks":
        filtered_df = filtered_df[filtered_df["facility_category"] == "college"]
    elif facility_choice == "Hostels":
        filtered_df = filtered_df[filtered_df["facility_category"] == "hostel"]
    elif facility_choice == "Canteen":
        filtered_df = filtered_df[filtered_df["facility_category"] == "canteen"]

    # Metric filter
    if metric_choice == "Energy (kWh)":
        filtered_df = filtered_df[filtered_df["metric"] == "energy"]
    elif metric_choice == "Water (Liters)":
        filtered_df = filtered_df[filtered_df["metric"] == "water"]
    elif metric_choice == "Waste (kg)":
        filtered_df = filtered_df[filtered_df["metric"] == "waste"]

    flagged_df = filtered_df[filtered_df["flagged"]]

    # 3-Tab Operational Layout
    tab1, tab2, tab3 = st.tabs([
        "📊 Tab 1: Executive Overview & Real-Time Alerts",
        "📈 Tab 2: Deep Dive & Multi-Domain Curves",
        "💡 Tab 3: Actionable Recommendations & Projected Savings",
    ])

    # ---------------- TAB 1: EXECUTIVE OVERVIEW ----------------
    with tab1:
        st.subheader("Campus Resource Health & Sustainability Overview")

        total_records = len(filtered_df)
        total_anomalies = len(flagged_df)
        anomaly_rate = (total_anomalies / total_records * 100) if total_records > 0 else 0

        # Calculate excess consumption per domain
        excess_energy_val = 0.0
        excess_water_val = 0.0
        excess_waste_val = 0.0

        if not flagged_df.empty:
            e_sub = flagged_df[flagged_df["metric"] == "energy"]
            if not e_sub.empty:
                excess_energy_val = (e_sub["reading"] - e_sub["predicted"]).sum()

            w_sub = flagged_df[flagged_df["metric"] == "water"]
            if not w_sub.empty:
                excess_water_val = (w_sub["reading"] - w_sub["predicted"]).sum()

            waste_sub = flagged_df[flagged_df["metric"] == "waste"]
            if not waste_sub.empty:
                excess_waste_val = (waste_sub["reading"] - waste_sub["predicted"]).sum()

        est_financial_waste_inr = (
            (excess_energy_val * ENERGY_COST_PER_KWH) +
            (excess_water_val * WATER_COST_PER_LITER) +
            (excess_waste_val * WASTE_COST_PER_KG)
        )
        est_monthly_loss_inr = (est_financial_waste_inr / 46.0) * 30.0  # Normalized to 30 days

        # 4 KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Telemetry Analyzed", f"{total_records:,}")
        col2.metric("Flagged Inefficiencies", f"{total_anomalies:,}", f"{anomaly_rate:.1f}% anomaly rate", delta_color="inverse")
        col3.metric("Excess Resource Loss", f"⚡ {excess_energy_val:,.0f} kWh | 💧 {excess_water_val:,.0f} L | ♻️ {excess_waste_val:,.0f} kg")
        col4.metric("Est. Monthly Financial Loss", f"₹{est_monthly_loss_inr:,.0f}", delta="Action Required", delta_color="inverse")

        st.divider()

        # Architecture diagram & Active alerts in two columns
        col_diag, col_alerts = st.columns([1, 1])

        with col_diag:
            st.markdown("#### 🏗️ Singular Model Multi-Domain Flow")
            st.markdown("""
```mermaid
flowchart TD
    A[3x 10k Bengaluru Datasets<br/>Energy, Water, Waste] --> B[Singular Unified RandomForest Regressor]
    B --> C{Actual vs Predicted > 20%?}
    C -- Inefficiency Flagged --> D[Domain Diagnostic & Savings Engine]
    C -- Normal Load --> E[Operational Baseline Log]
    D --> F[Quantified Action Items & ROI]
    D --> G[Streamlit Multi-Domain Dashboard]
    F --> G
```
            """)

        with col_alerts:
            st.markdown("#### 🚨 Top Active Anomaly Alerts")
            if flagged_df.empty:
                st.success("No anomalies detected for current filter settings.")
            else:
                top_flags = flagged_df.sort_values(by="deviation_pct", ascending=False).head(4)
                for _, alert in top_flags.iterrows():
                    metric_str = alert["metric"]
                    metric_icon = "⚡" if metric_str == "energy" else ("💧" if metric_str == "water" else "♻️")
                    unit = "kWh" if metric_str == "energy" else ("L" if metric_str == "water" else "kg")
                    st.warning(
                        f"**{metric_icon} {alert['building_name']}** ({alert.get('facility_category', '').title()}) — Hour {alert['hour']:02d}:00\n\n"
                        f"Actual: **{alert['reading']:.1f} {unit}** vs Predicted: **{alert['predicted']:.1f} {unit}** "
                        f"(**+{alert['deviation_pct']:.1f}% deviation**)"
                    )

    # ---------------- TAB 2: DEEP DIVE & ANALYSIS ----------------
    with tab2:
        st.subheader("Multi-Domain Inefficiency Deep Dive")

        drill_building = st.selectbox(
            "Select Facility for Hourly Resource Curves:",
            options=selected_buildings,
            index=0 if selected_buildings else 0,
        )

        if drill_building:
            b_data = filtered_df[filtered_df["building_name"] == drill_building]
            b_hourly = b_data.groupby(["hour", "metric"])[["reading", "predicted"]].mean().reset_index()

            col_e, col_w, col_waste = st.columns(3)

            with col_e:
                st.markdown(f"**⚡ Energy (kWh)**")
                e_hourly = b_hourly[b_hourly["metric"] == "energy"]
                if not e_hourly.empty:
                    chart_data = e_hourly.set_index("hour")[["reading", "predicted"]]
                    chart_data.columns = ["Actual", "Predicted"]
                    st.line_chart(chart_data)
                else:
                    st.info("No energy data.")

            with col_w:
                st.markdown(f"**💧 Water (Liters)**")
                w_hourly = b_hourly[b_hourly["metric"] == "water"]
                if not w_hourly.empty:
                    chart_data = w_hourly.set_index("hour")[["reading", "predicted"]]
                    chart_data.columns = ["Actual", "Predicted"]
                    st.line_chart(chart_data)
                else:
                    st.info("No water data.")

            with col_waste:
                st.markdown(f"**♻️ Waste (kg)**")
                waste_hourly = b_hourly[b_hourly["metric"] == "waste"]
                if not waste_hourly.empty:
                    chart_data = waste_hourly.set_index("hour")[["reading", "predicted"]]
                    chart_data.columns = ["Actual", "Predicted"]
                    st.line_chart(chart_data)
                else:
                    st.info("No waste data.")

        st.divider()

        # Inefficiency summary across buildings and categories
        col_bar1, col_bar2 = st.columns(2)
        with col_bar1:
            st.markdown("#### 🏢 Average Inefficiency Deviation by Facility")
            b_dev = filtered_df.groupby("building_name")[["deviation_pct"]].mean().sort_values(by="deviation_pct", ascending=False)
            st.bar_chart(b_dev)

        with col_bar2:
            st.markdown("#### 📍 Inefficiency by Facility Category")
            cat_dev = filtered_df.groupby("facility_category")[["deviation_pct"]].mean().sort_values(by="deviation_pct", ascending=False)
            st.bar_chart(cat_dev)

    # ---------------- TAB 3: RECOMMENDATIONS & SAVINGS ----------------
    with tab3:
        st.subheader("💡 Multi-Domain Recommendations & Projected ROI")
        st.write("Quantified operational interventions generated per detected anomaly across Energy, Water, and Waste vectors.")

        recs_df = generate_recommendations_for_flagged(filtered_df, limit=30)

        if recs_df.empty:
            st.info("No flagged anomalies to generate recommendations for.")
        else:
            total_monthly_savings = recs_df["projected_monthly_savings_inr"].sum()
            st.success(f"💰 **Total Potential Monthly Savings Identified: ₹{total_monthly_savings:,.0f}** across {len(recs_df)} prioritized actions.")

            for _, r in recs_df.head(6).iterrows():
                badge_class = "badge-critical" if "CRITICAL" in r["priority"] else "badge-high"
                metric_str = r["metric"]
                metric_icon = "⚡ Energy" if metric_str == "energy" else ("💧 Water" if metric_str == "water" else "♻️ Waste")

                with st.expander(f"{r['building_name']} — {metric_icon} at Hour {r['hour']:02d}:00 (Save ₹{r['hourly_loss_inr']:.0f}/hr)", expanded=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Root Cause:** {r['root_cause']}")
                        st.markdown(f"**Action Required:** `{r['action']}`")
                        st.markdown(f"**Detailed Recommendation:** {r['recommendation']}")
                    with c2:
                        st.metric("Excess Waste", f"{r['excess_units']:.1f} {r['unit']}/hr")
                        st.metric("Est. Monthly ROI", f"₹{r['projected_monthly_savings_inr']:,.0f}")

            st.divider()
            st.markdown("#### 📋 Prioritized Action Items Table")
            display_cols = [
                "building_name", "facility_category", "metric", "hour", "reading", "predicted",
                "excess_units", "unit", "hourly_loss_inr", "priority", "action"
            ]
            st.dataframe(recs_df[display_cols], use_container_width=True)

            csv_data = recs_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Multi-Domain Action Items (CSV)",
                data=csv_data,
                file_name="crew_campus_action_items.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
