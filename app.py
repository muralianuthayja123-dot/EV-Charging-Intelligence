import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EV Charging Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EV Charging Intelligence Dashboard")

st.caption(
    "Machine Learning-Based EV Charging Demand and "
    "Infrastructure Capacity Analysis"
)


# =========================================================
# LOAD DATA
# =========================================================

BASE_PATH = "./"


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


# =========================================================
# CREATE COMPLETE HOURLY DATA
# =========================================================

@st.cache_data
def create_complete_data(data):

    complete_sites = []

    for site in data["station_id"].unique():

        site_df = data[
            data["station_id"] == site
        ].copy()

        site_df = site_df.sort_values(
            "hour_timestamp"
        )

        site_df = site_df.set_index(
            "hour_timestamp"
        )

        full_index = pd.date_range(
            start=site_df.index.min(),
            end=site_df.index.max(),
            freq="h",
            tz="UTC"
        )

        site_df = site_df.reindex(
            full_index
        )

        site_df["station_id"] = site

        site_df["charging_sessions"] = (
            site_df["charging_sessions"]
            .fillna(0)
        )

        site_df["total_energy_kwh"] = (
            site_df["total_energy_kwh"]
            .fillna(0)
        )

        complete_sites.append(
            site_df
        )

    complete_data = pd.concat(
        complete_sites
    )

    complete_data.index.name = (
        "hour_timestamp"
    )

    return complete_data


ml_complete = create_complete_data(
    hourly
)


# =========================================================
# TRAIN GRADIENT BOOSTING MODEL
# =========================================================

@st.cache_resource
def train_model(data):

    model_data = data.copy()

    # ---------------------------------------------
    # TIME FEATURES
    # ---------------------------------------------

    model_data["hour"] = (
        model_data.index.hour
    )

    # SAME DEFINITION AS TRAINING
    model_data["day_num"] = (
        model_data.index.dayofweek + 1
    )

    model_data["is_weekend"] = (
        model_data["day_num"] >= 6
    ).astype(int)


    # ---------------------------------------------
    # LAG FEATURES
    # ---------------------------------------------

    model_data["lag_1"] = (
        model_data
        .groupby("station_id")["charging_sessions"]
        .shift(1)
    )

    model_data["lag_2"] = (
        model_data
        .groupby("station_id")["charging_sessions"]
        .shift(2)
    )

    model_data["lag_3"] = (
        model_data
        .groupby("station_id")["charging_sessions"]
        .shift(3)
    )

    model_data["lag_24"] = (
        model_data
        .groupby("station_id")["charging_sessions"]
        .shift(24)
    )

    model_data["lag_168"] = (
        model_data
        .groupby("station_id")["charging_sessions"]
        .shift(168)
    )


    # ---------------------------------------------
    # ROLLING FEATURES
    # ---------------------------------------------

    model_data["rolling_3h"] = (
        model_data
        .groupby("station_id")["charging_sessions"]
        .transform(
            lambda x: x.rolling(3).mean()
        )
    )

    model_data["rolling_24h"] = (
        model_data
        .groupby("station_id")["charging_sessions"]
        .transform(
            lambda x: x.rolling(24).mean()
        )
    )


    # ---------------------------------------------
    # FEATURES
    # ---------------------------------------------

    features = [
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


    model_data = model_data.dropna(
        subset=features
    )


    # ---------------------------------------------
    # CHRONOLOGICAL 80% TRAINING DATA
    # ---------------------------------------------

    split_date = (
        model_data.index
        .to_series()
        .quantile(0.80)
    )

    train_data = model_data[
        model_data.index <= split_date
    ]


    X_train = train_data[
        features
    ]

    y_train = train_data[
        "charging_sessions"
    ]


    # ---------------------------------------------
    # GRADIENT BOOSTING MODEL
    # ---------------------------------------------

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    return model


model = train_model(
    ml_complete
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

    st.header(
        "📊 System Overview"
    )


    total_sessions = (
        hourly["charging_sessions"].sum()
    )


    avg_demand = (
        hourly["charging_sessions"].mean()
    )


    peak_demand = (
        hourly["charging_sessions"].max()
    )


    busiest_site = (
        hourly
        .groupby("station_id")[
            "charging_sessions"
        ]
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


    st.subheader(
        "Charging Demand Over Time"
    )


    trend = (
        hourly
        .groupby("hour_timestamp")[
            "charging_sessions"
        ]
        .sum()
        .reset_index()
    )


    st.line_chart(
        trend.set_index(
            "hour_timestamp"
        )
    )


    st.subheader(
        "Site-wise Charging Activity"
    )


    site_summary = (
        hourly
        .groupby("station_id")
        .agg(
            total_sessions=(
                "charging_sessions",
                "sum"
            ),
            average_sessions=(
                "charging_sessions",
                "mean"
            ),
            total_energy=(
                "total_energy_kwh",
                "sum"
            )
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

    st.header(
        "📈 EV Charging Demand Analysis"
    )


    selected_site = st.selectbox(
        "Select Site",
        sorted(
            hourly["station_id"]
            .unique()
        )
    )


    site_data = hourly[
        hourly["station_id"] ==
        selected_site
    ].copy()


    st.subheader(
        "Hourly Demand"
    )


    hourly_profile = (
        site_data
        .groupby("hour")[
            "charging_sessions"
        ]
        .mean()
    )


    st.line_chart(
        hourly_profile
    )


    st.subheader(
        "Demand by Day of Week"
    )


    day_profile = (
        site_data
        .groupby("day_num")[
            "charging_sessions"
        ]
        .mean()
    )


    st.bar_chart(
        day_profile
    )


    st.subheader(
        "Charging Energy"
    )


    energy_profile = (
        site_data
        .groupby("hour")[
            "total_energy_kwh"
        ]
        .mean()
    )


    st.line_chart(
        energy_profile
    )


    st.subheader(
        "Demand Statistics"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Average Sessions/Hour",
        f"{site_data['charging_sessions'].mean():.2f}"
    )


    col2.metric(
        "Peak Sessions/Hour",
        f"{int(site_data['charging_sessions'].max())}"
    )


    col3.metric(
        "Average Energy/Hour",
        f"{site_data['total_energy_kwh'].mean():.2f} kWh"
    )


# =========================================================
# AI DEMAND FORECAST
# =========================================================

elif page == "AI Demand Forecast":

    st.header(
        "🤖 AI-Based EV Charging Demand Forecast"
    )


    st.write(
        "Gradient Boosting is used to forecast hourly "
        "charging demand using temporal, lag and "
        "rolling-demand features."
    )


    selected_site = st.selectbox(
        "Select Site",
        sorted(
            ml_complete["station_id"]
            .unique()
        ),
        key="forecast_site"
    )


    # ---------------------------------------------
    # GET COMPLETE SITE DATA
    # ---------------------------------------------

    site_data = ml_complete[
        ml_complete["station_id"] ==
        selected_site
    ].copy()


    site_data = site_data.sort_index()


    # ---------------------------------------------
    # TIME FEATURES
    # ---------------------------------------------

    site_data["hour"] = (
        site_data.index.hour
    )


    # SAME DEFINITION AS MODEL TRAINING
    site_data["day_num"] = (
        site_data.index.dayofweek + 1
    )


    site_data["is_weekend"] = (
        site_data["day_num"] >= 6
    ).astype(int)


    # ---------------------------------------------
    # LAG FEATURES
    # ---------------------------------------------

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


    # ---------------------------------------------
    # ROLLING FEATURES
    # ---------------------------------------------

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


    # ---------------------------------------------
    # MODEL FEATURES
    # ---------------------------------------------

    feature_cols = [
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


    prediction_data = (
        site_data
        .dropna(
            subset=feature_cols
        )
        .copy()
    )


    if len(prediction_data) > 0:

        X = prediction_data[
            feature_cols
        ]


        # -----------------------------------------
        # PREDICTION
        # -----------------------------------------

        raw_predictions = model.predict(
            X
        )


        prediction_data[
            "predicted_demand"
        ] = np.clip(
            raw_predictions,
            0,
            None
        )


        # -----------------------------------------
        # LATEST PREDICTION
        # -----------------------------------------

        latest_prediction = (
            prediction_data[
                "predicted_demand"
            ].iloc[-1]
        )


        latest_actual = (
            prediction_data[
                "charging_sessions"
            ].iloc[-1]
        )


        difference = (
            latest_prediction -
            latest_actual
        )


        st.subheader(
            "Latest Demand Forecast"
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Predicted Demand",
            f"{latest_prediction:.2f} sessions"
        )


        col2.metric(
            "Actual Demand",
            f"{latest_actual:.2f} sessions"
        )


        col3.metric(
            "Prediction Difference",
            f"{difference:+.2f}"
        )


        # -----------------------------------------
        # ACTUAL VS PREDICTED
        # -----------------------------------------

        st.subheader(
            "Actual vs Predicted Demand"
        )


        chart = prediction_data[
            [
                "charging_sessions",
                "predicted_demand"
            ]
        ].tail(200)


        chart.columns = [
            "Actual Demand",
            "Predicted Demand"
        ]


        st.line_chart(
            chart
        )


        # -----------------------------------------
        # FORECAST DATA TABLE
        # -----------------------------------------

        st.subheader(
            "Recent Prediction Data"
        )


        display_data = prediction_data[
            [
                "charging_sessions",
                "predicted_demand",
                "hour",
                "day_num",
                "lag_1",
                "lag_24",
                "lag_168",
                "rolling_3h",
                "rolling_24h"
            ]
        ].tail(20)


        st.dataframe(
            display_data,
            use_container_width=True
        )


        st.info(
            "The Gradient Boosting model uses historical "
            "charging demand, temporal features, lag features "
            "and rolling averages. Negative predictions are "
            "clipped to zero because charging demand cannot "
            "be negative."
        )


    else:

        st.warning(
            "Not enough historical observations are "
            "available to generate the required features."
        )


# =========================================================
# INFRASTRUCTURE STRESS
# =========================================================

elif page == "Infrastructure Stress":

    st.header(
        "🏗️ Infrastructure Stress Analysis"
    )


    st.dataframe(
        stress.sort_values(
            "p95_occupancy",
            ascending=False
        ),
        use_container_width=True
    )


    st.subheader(
        "Connected EV Occupancy"
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


    st.bar_chart(
        chart_data
    )


    selected_site = st.selectbox(
        "Analyze Site",
        sorted(
            stress["station_id"]
            .unique()
        ),
        key="stress_site"
    )


    site_stress = stress[
        stress["station_id"] ==
        selected_site
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

    st.header(
        "🔌 Charging Capacity Simulator"
    )


    selected_site = st.selectbox(
        "Select Site",
        sorted(
            capacity["station_id"]
            .unique()
        ),
        key="capacity_site"
    )


    site_capacity = capacity[
        capacity["station_id"] ==
        selected_site
    ].sort_values(
        "capacity"
    )


    st.subheader(
        "Capacity vs Overload Exposure"
    )


    chart_data = (
        site_capacity
        .set_index("capacity")
        ["overload_percent"]
    )


    st.line_chart(
        chart_data
    )


    st.subheader(
        "Marginal Benefit of Additional Capacity"
    )


    marginal_data = (
        site_capacity
        .set_index("capacity")
        ["marginal_benefit"]
    )


    st.bar_chart(
        marginal_data
    )


    st.subheader(
        "Capacity Scenario Table"
    )


    st.dataframe(
        site_capacity,
        use_container_width=True
    )


    st.info(
        "Marginal benefit represents the reduction "
        "in overload exposure obtained by increasing "
        "assumed charging capacity."
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

        site = row[
            "station_id"
        ]

        recommendation = row[
            "recommendation"
        ]


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
    "ACN-Data based analysis | "
    "Machine Learning + Infrastructure Scenario Analysis"
)
