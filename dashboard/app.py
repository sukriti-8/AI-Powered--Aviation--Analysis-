import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.preprocessing import preprocess_data


st.set_page_config(
    page_title="AI-Powered Aviation Analytics",
    page_icon="✈️",
    layout="wide",
)


@st.cache_data
def load_processed_data():
    """
    Use the existing project preprocessing pipeline.

    No new preprocessing logic is created here.
    """
    return preprocess_data()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("AI-POWERED AVIATION ANALYTICS")
st.subheader("Aircraft Engine Predictive Maintenance")

st.write(
    "This dashboard estimates the Remaining Useful Life (RUL) "
    "of aircraft engines using NASA C-MAPSS FD001 sensor data "
    "and trained machine learning models."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

with st.spinner("Loading NASA C-MAPSS FD001 data..."):

    (
        train,
        test,
        rul,
        X_lstm,
        y_lstm,
        lstm_units
    ) = load_processed_data()


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

st.success("NASA C-MAPSS FD001 data loaded successfully.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Test Engines",
        test["unit"].nunique()
    )

with col2:
    st.metric(
        "LSTM Sequence",
        "30 cycles"
    )

with col3:
    st.metric(
        "LSTM Features",
        X_lstm.shape[2]
    )

with col4:
    st.metric(
        "Test Records",
        len(test)
    )


# ---------------------------------------------------------
# ENGINE SELECTION
# ---------------------------------------------------------

st.header("Engine Analysis")

engine_ids = sorted(
    test["unit"].unique()
)

selected_engine = st.selectbox(
    "Select an aircraft engine",
    engine_ids
)


# ---------------------------------------------------------
# SELECT ENGINE DATA
# ---------------------------------------------------------

engine_data = (
    test[
        test["unit"] == selected_engine
    ]
    .sort_values("cycle")
    .copy()
)

latest_cycle = int(
    engine_data["cycle"].max()
)

actual_rul = float(
    rul.iloc[selected_engine - 1]["RUL"]
)


# ---------------------------------------------------------
# CURRENT ENGINE INFORMATION
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Engine",
        f"Engine {selected_engine}"
    )

with col2:
    st.metric(
        "Latest Observed Cycle",
        latest_cycle
    )

with col3:
    st.metric(
        "Benchmark Actual RUL",
        f"{actual_rul:.0f} cycles"
    )


st.info(
    "The actual RUL shown here is available because this is a "
    "benchmark dataset. In a real aircraft system, future RUL "
    "would not be known in advance."
)


# ---------------------------------------------------------
# SENSOR DATA
# ---------------------------------------------------------

st.header("Sensor Trends")

sensor_columns = [
    column
    for column in engine_data.columns
    if column.startswith("sensor_")
    and not column.endswith("_smooth")
    and "_rolling_" not in column
]

selected_sensor = st.selectbox(
    "Select sensor",
    sensor_columns
)

sensor_chart_data = engine_data[
    ["cycle", selected_sensor]
].set_index("cycle")

st.line_chart(
    sensor_chart_data
)


# ---------------------------------------------------------
# DEBUG / PIPELINE INFORMATION
# ---------------------------------------------------------

with st.expander("Pipeline Information"):

    st.write(
        "The dashboard is using the existing project "
        "preprocessing pipeline."
    )

    st.write(
        f"Processed test shape: {test.shape}"
    )

    st.write(
        f"LSTM input shape: {X_lstm.shape}"
    )

    st.write(
        f"Number of LSTM features: {X_lstm.shape[2]}"
    )

    st.write(
        "LSTM sequence length: 30 cycles"
    )

    st.write(
        "RUL training cap: 125 cycles"
    )

    st.write(
        "Sensor smoothing window: 5 cycles"
    )

    st.write(
        "Random Forest rolling-feature window: 10 cycles"
    )