
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EV Charging Intelligence",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================

BASE_PATH = "./"
APP_PATH = BASE_PATH + "EV_Charging_Dashboard/"

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    hourly = pd.read_csv(
        BASE_PATH + "caltech_hourly_demand.csv"
    )

    hourly["hour_timestamp"] = pd.to_datetime(
        hourly["hour_timestamp"],
        utc=True,
        format="mixed"
    )

    final_analysis = pd.read_csv(
        BASE_PATH + "final_site_analysis.csv"
    )

    future_capacity = pd.read_csv(
        BASE_PATH + "future_capacity_scenarios.csv"
    )

    stress = pd.read_csv(
        BASE_PATH + "site_stress_analysis.csv"
    )

    capacity = pd.read_csv(
        BASE_PATH + "capacity_scenario_analysis.csv"
    )

    return (
        hourly,
        final_analysis,
        future_capacity,
        stress,
        capacity
    )


hourly, final_analysis, future_capacity, stress, capacity = load_data()


# =========================================================
# LOAD ML MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        APP_PATH + "best_demand_model.pkl"
    )


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("⚡ EV Charging Intelligence Dashboard")

st.caption(
    "Machine Learning-Based EV Charging Demand "
    "and Infrastructure Capacity Analysis"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Demand Analysis",
        "AI Demand Forecast",
        "Infrastructure Stress",
        "Capacity Simulator",
        "Final Recommendations"
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.header("📊 System Overview")

    total_sessions = hourly["charging_sessions"].sum()

    avg_demand = hourly[
        "charging_sessions"
    ].mean()

    peak_demand = hourly[
        "charging_sessions"
    ].max()

    busiest_site = (
        hourly.groupby("station_id")
        ["charging_sessions"]
        .sum()
        .idxmax()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Charging Sessions",
        f"{int(total_sessions):,}"
    )

    col2.metric(
        "Average Hourly Demand",
        f"{avg_demand:.2f}"
    )

    col3.metric(
        "Peak Hourly Demand",
        f"{int(peak_demand)}"
    )

    col4.metric(
        "Busiest Site",
        busiest_site
    )

    st.divider()

    st.subheader("Charging Demand Over Time")

    trend = (
        hourly.groupby("hour_timestamp")
        ["charging_sessions"]
        .sum()
        .reset_index()
    )

    st.line_chart(
        trend.set_index("hour_timestamp")
    )

    st.subheader("Site-wise Charging Activity")

    site_summary = (
        hourly.groupby("station_id")
        .agg(
            total_sessions=("charging_sessions", "sum"),
            average_sessions=("charging_sessions", "mean"),
            total_energy=("total_energy_kwh", "sum")
        )
        .sort_values(
            "total_sessions",
            ascending=False
        )
    )

    st.dataframe(
        site_summary,
        use_container_width=True
    )


# =========================================================
# DEMAND ANALYSIS
# =========================================================

elif page == "Demand Analysis":

    st.header("📈 EV Charging Demand Analysis")

    selected_site = st.selectbox(
        "Select Site",
        sorted(
            hourly["station_id"].unique()
        )
    )

    site_data = hourly[
        hourly["station_id"] == selected_site
    ].copy()

    st.subheader("Average Demand by Hour")

    hourly_profile = (
        site_data.groupby("hour")
        ["charging_sessions"]
        .mean()
    )

    st.line_chart(hourly_profile)

    st.subheader("Average Demand by Day")

    day_profile = (
        site_data.groupby("day_num")
        ["charging_sessions"]
        .mean()
    )

    st.bar_chart(day_profile)

    st.subheader("Average Energy Demand by Hour")

    energy_profile = (
        site_data.groupby("hour")
        ["total_energy_kwh"]
        .mean()
    )

    st.line_chart(energy_profile)

    st.subheader("Site Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Sessions / Hour",
        f"{site_data['charging_sessions'].mean():.2f}"
    )

    col2.metric(
        "Peak Sessions / Hour",
        f"{int(site_data['charging_sessions'].max())}"
    )

    col3.metric(
        "Average Energy / Hour",
        f"{site_data['total_energy_kwh'].mean():.2f} kWh"
    )


# =========================================================
# AI DEMAND FORECAST
# =========================================================

elif page == "AI Demand Forecast":

    st.header("🤖 AI Charging Demand Forecast")

    st.write(
        "The Gradient Boosting model predicts hourly charging "
        "session demand using temporal and historical demand features."
    )

    selected_site = st.selectbox(
        "Select Site",
        sorted(
            hourly["station_id"].unique()
        ),
        key="forecast_site"
    )

    site_data = hourly[
        hourly["station_id"] == selected_site
    ].copy()

    # -----------------------------------------------------
    # BUILD COMPLETE TIMELINE
    # -----------------------------------------------------

    site_data = site_data.sort_values(
        "hour_timestamp"
    ).set_index("hour_timestamp")

    full_index = pd.date_range(
        start=site_data.index.min(),
        end=site_data.index.max(),
        freq="h",
        tz="UTC"
    )

    site_data = site_data.reindex(
        full_index
    )

    site_data["station_id"] = selected_site

    site_data["charging_sessions"] = (
        site_data["charging_sessions"]
        .fillna(0)
    )

    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------

    site_data["hour"] = site_data.index.hour

    site_data["day_num"] = (
        site_data.index.dayofweek
    )

    site_data["is_weekend"] = (
        site_data["day_num"] >= 5
    ).astype(int)

    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------

    site_data["lag_1"] = (
        site_data["charging_sessions"]
        .shift(1)
    )

    site_data["lag_2"] = (
        site_data["charging_sessions"]
        .shift(2)
    )

    site_data["lag_3"] = (
        site_data["charging_sessions"]
        .shift(3)
    )

    site_data["lag_24"] = (
        site_data["charging_sessions"]
        .shift(24)
    )

    site_data["lag_168"] = (
        site_data["charging_sessions"]
        .shift(168)
    )

    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------

    site_data["rolling_3h"] = (
        site_data["charging_sessions"]
        .rolling(3)
        .mean()
    )

    site_data["rolling_24h"] = (
        site_data["charging_sessions"]
        .rolling(24)
        .mean()
    )

    feature_columns = [
        "hour",
        "day_num",
        "is_weekend",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_168",
        "rolling_3h",
        "rolling_24h"
    ]

    prediction_data = site_data.dropna(
        subset=feature_columns
    ).copy()

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    prediction_data["predicted_demand"] = (
        model.predict(
            prediction_data[feature_columns]
        )
    )

    prediction_data["predicted_demand"] = (
        prediction_data["predicted_demand"]
        .clip(lower=0)
    )

    # -----------------------------------------------------
    # PERFORMANCE DISPLAY
    # -----------------------------------------------------

    st.subheader("Model")

    st.info(
        "Selected Model: Gradient Boosting Regressor"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        "0.5173"
    )

    col2.metric(
        "RMSE",
        "1.2265"
    )

    col3.metric(
        "R²",
        "0.2302"
    )

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader(
        "Actual vs Predicted Charging Demand"
    )

    comparison = prediction_data[
        [
            "charging_sessions",
            "predicted_demand"
        ]
    ].rename(
        columns={
            "charging_sessions": "Actual",
            "predicted_demand": "Predicted"
        }
    )

    st.line_chart(
        comparison
    )

    # -----------------------------------------------------
    # RECENT FORECAST
    # -----------------------------------------------------

    st.subheader("Recent Predictions")

    recent = prediction_data[
        [
            "hour",
            "day_num",
            "charging_sessions",
            "predicted_demand"
        ]
    ].tail(24).copy()

    recent.columns = [
        "Hour",
        "Day",
        "Actual Demand",
        "Predicted Demand"
    ]

    st.dataframe(
        recent,
        use_container_width=True
    )


# =========================================================
# INFRASTRUCTURE STRESS
# =========================================================

elif page == "Infrastructure Stress":

    st.header("🏗️ Infrastructure Stress Analysis")

    st.dataframe(
        stress.sort_values(
            "p95_occupancy",
            ascending=False
        ),
        use_container_width=True
    )

    st.subheader(
        "Connected EV Occupancy by Site"
    )

    chart_data = stress.set_index(
        "station_id"
    )[
        [
            "average_occupancy",
            "p95_occupancy",
            "maximum_occupancy"
        ]
    ]

    st.bar_chart(chart_data)

    selected_site = st.selectbox(
        "Analyze Site",
        sorted(
            stress["station_id"].unique()
        ),
        key="stress_site"
    )

    site_stress = stress[
        stress["station_id"] == selected_site
    ].iloc[0]

    st.subheader(
        f"Stress Profile — {selected_site}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Occupancy",
        f"{site_stress['average_occupancy']:.1f}"
    )

    col2.metric(
        "P95 Occupancy",
        f"{site_stress['p95_occupancy']:.1f}"
    )

    col3.metric(
        "Maximum Occupancy",
        f"{int(site_stress['maximum_occupancy'])}"
    )

    col4.metric(
        "Stress Level",
        site_stress["stress_level"]
    )


# =========================================================
# CAPACITY SIMULATOR
# =========================================================

elif page == "Capacity Simulator":

    st.header("🔌 Charging Capacity Simulator")

    selected_site = st.selectbox(
        "Select Site",
        sorted(
            capacity["station_id"].unique()
        ),
        key="capacity_site"
    )

    site_capacity = capacity[
        capacity["station_id"] == selected_site
    ].sort_values("capacity")

    st.subheader(
        "Capacity vs Overload Exposure"
    )

    chart_data = site_capacity.set_index(
        "capacity"
    )["overload_percent"]

    st.line_chart(chart_data)

    st.subheader(
        "Marginal Benefit of Additional Capacity"
    )

    marginal_data = site_capacity.set_index(
        "capacity"
    )["marginal_benefit"]

    st.bar_chart(marginal_data)

    st.subheader(
        "Capacity Scenario Results"
    )

    st.dataframe(
        site_capacity,
        use_container_width=True
    )

    st.info(
        "Marginal benefit represents the reduction in "
        "overload exposure obtained by increasing assumed capacity."
    )


# =========================================================
# FINAL RECOMMENDATIONS
# =========================================================

elif page == "Final Recommendations":

    st.header(
        "🤖 Infrastructure Recommendations"
    )

    st.dataframe(
        final_analysis,
        use_container_width=True
    )

    st.subheader(
        "Recommended Infrastructure Actions"
    )

    for _, row in final_analysis.iterrows():

        site = row["station_id"]

        recommendation = (
            row["recommendation"]
        )

        if recommendation == "Expand Capacity":

            st.error(
                f"🔴 {site}: {recommendation}"
            )

        elif recommendation == "Plan Moderate Expansion":

            st.warning(
                f"🟠 {site}: {recommendation}"
            )

        else:

            st.success(
                f"🟢 {site}: {recommendation}"
            )

    st.subheader(
        "Future Capacity Requirements"
    )

    pivot = future_capacity.pivot(
        index="station_id",
        columns="scenario",
        values="recommended_capacity"
    )

    st.dataframe(
        pivot,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EV Charging Intelligence | "
    "ACN-Data | "
    "Gradient Boosting | "
    "Infrastructure Scenario Analysis"
)
