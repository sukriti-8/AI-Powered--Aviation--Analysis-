import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from src.preprocessing import preprocess_data


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Powered Aviation Analytics",
    page_icon="✈️",
    layout="wide",
)


# =========================================================
# LOAD EXISTING PREPROCESSED DATA
# =========================================================

@st.cache_data
def load_processed_data():
    """
    Reuse the existing project preprocessing pipeline.

    No new preprocessing logic is created here.
    """
    return preprocess_data()


# =========================================================
# LOAD SAVED LSTM MODEL
# =========================================================

@st.cache_resource
def load_lstm_model():
    """
    Load the already-trained LSTM model.

    The model is NOT retrained when the dashboard starts.
    """
    model_path = PROJECT_ROOT / "models" / "lstm_final.keras"

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


# =========================================================
# LOAD SAVED RANDOM FOREST MODEL
# =========================================================

@st.cache_resource
def load_rf_model():
    """
    Load the already-trained Random Forest model.

    The model is NOT retrained when the dashboard starts.
    """
    model_path = PROJECT_ROOT / "models" / "random_forest.pkl"

    return joblib.load(model_path)


# =========================================================
# CREATE LSTM TEST SEQUENCE
# =========================================================

def create_engine_sequence(
    engine_data,
    lstm_features,
    window=30
):
    """
    Create the latest 30-cycle input sequence
    for one test engine.

    Output shape:
        (1, 30, 18)
    """

    engine_data = (
        engine_data
        .sort_values("cycle")
        .copy()
    )

    features = engine_data[
        lstm_features
    ].to_numpy()

    # Handle engines with fewer than 30 cycles.
    if len(features) < window:

        padding = np.repeat(
            features[:1],
            window - len(features),
            axis=0
        )

        features = np.vstack(
            [padding, features]
        )

    # Take the latest 30 observed cycles.
    sequence = features[-window:]

    # Add batch dimension.
    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    return sequence


# =========================================================
# MAINTENANCE STATUS
# =========================================================

def get_maintenance_status(predicted_rul):
    """
    Convert predicted RUL into a simple dashboard
    maintenance status.

    These are project interpretation rules and are
    NOT official aviation maintenance thresholds.
    """

    if predicted_rul > 50:
        return "Healthy"

    elif predicted_rul >= 30:
        return "Monitor"

    elif predicted_rul >= 10:
        return "Maintenance Recommended"

    else:
        return "Critical"


# =========================================================
# HEADER
# =========================================================

st.title(
    "AI-POWERED AVIATION ANALYTICS"
)

st.subheader(
    "Aircraft Engine Predictive Maintenance"
)

st.write(
    "This dashboard estimates the Remaining Useful Life (RUL) "
    "of aircraft engines using NASA C-MAPSS FD001 sensor data "
    "and trained machine learning models."
)


# =========================================================
# LOAD DATA AND MODELS
# =========================================================

with st.spinner(
    "Loading project data and trained models..."
):

    (
        train,
        test,
        rul,
        X_lstm,
        y_lstm,
        lstm_units
    ) = load_processed_data()

    lstm_model = load_lstm_model()
    rf_model = load_rf_model()


# =========================================================
# FIND EXACT LSTM FEATURES
# =========================================================

lstm_features = [
    column
    for column in train.columns
    if column.startswith("setting_")
    or (
        column.startswith("sensor_")
        and not column.endswith("_smooth")
        and "_rolling_" not in column
    )
]


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.success(
    "NASA C-MAPSS FD001 data and saved trained models loaded successfully."
)

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
        len(lstm_features)
    )

with col4:
    st.metric(
        "Primary Model",
        "LSTM"
    )


# =========================================================
# ENGINE SELECTION
# =========================================================

st.header("Engine Analysis")

engine_ids = sorted(
    test["unit"].unique()
)

selected_engine = st.selectbox(
    "Select an aircraft engine",
    engine_ids
)

selected_model = st.selectbox(
    "Select prediction model",
    ["LSTM", "Random Forest"]
)


# =========================================================
# SELECT ENGINE DATA
# =========================================================

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


# =========================================================
# GENERATE MODEL PREDICTION
# =========================================================

if selected_model == "LSTM":

    # -----------------------------------------------------
    # LSTM: latest 30 cycles × 18 features
    # -----------------------------------------------------

    X_engine = create_engine_sequence(
        engine_data,
        lstm_features,
        window=30
    )

    if X_engine.shape != (1, 30, 18):

        st.error(
            f"Unexpected LSTM input shape: {X_engine.shape}. "
            "Expected (1, 30, 18)."
        )

        st.stop()

    prediction = lstm_model.predict(
        X_engine,
        verbose=0
    )

    predicted_rul = float(
        prediction.flatten()[0]
    )

    prediction_input_description = (
        "Latest 30 cycles × 18 LSTM features"
    )


else:

    # -----------------------------------------------------
    # RANDOM FOREST: existing engineered RF features
    # -----------------------------------------------------

    test_last = (
        test
        .sort_values(["unit", "cycle"])
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
    )

    X_test_rf = test_last.drop(
        columns=["unit"]
    )

    rf_prediction = rf_model.predict(
        X_test_rf
    )

    engine_position = (
        sorted(test["unit"].unique())
        .index(selected_engine)
    )

    predicted_rul = float(
        rf_prediction[engine_position]
    )

    prediction_input_description = (
        "Existing engineered Random Forest features"
    )


# =========================================================
# ACTUAL RUL / BENCHMARK RUL
# =========================================================

# RUL_FD001.txt contains one RUL value for each test engine.
# Map the values explicitly to sorted engine IDs.

sorted_engine_ids = sorted(
    test["unit"].unique()
)

actual_rul_map = dict(
    zip(
        sorted_engine_ids,
        rul["RUL"].to_numpy()
    )
)

actual_rul = float(
    actual_rul_map[selected_engine]
)


# =========================================================
# PREDICTION ERROR
# =========================================================

prediction_error = (
    predicted_rul - actual_rul
)

absolute_error = abs(
    prediction_error
)


# =========================================================
# MAINTENANCE STATUS
# =========================================================

maintenance_status = get_maintenance_status(
    predicted_rul
)


# =========================================================
# ENGINE INFORMATION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

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
        "Predicted RUL",
        f"{predicted_rul:.2f} cycles"
    )

with col4:

    st.metric(
        "Maintenance Status",
        maintenance_status
    )


# =========================================================
# ACTUAL VS PREDICTED RUL
# =========================================================

st.subheader(
    "RUL Comparison"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.2f} cycles"
    )

with col2:

    st.metric(
        "Actual RUL",
        f"{actual_rul:.2f} cycles"
    )

with col3:

    st.metric(
        "Absolute Prediction Error",
        f"{absolute_error:.2f} cycles"
    )


if prediction_error > 0:

    st.info(
        f"The model overestimated the RUL by "
        f"{prediction_error:.2f} cycles."
    )

elif prediction_error < 0:

    st.info(
        f"The model underestimated the RUL by "
        f"{abs(prediction_error):.2f} cycles."
    )

else:

    st.success(
        "The predicted RUL exactly matches the benchmark RUL."
    )


st.caption(
    "Actual RUL is available here because this is a benchmark "
    "test dataset. In a real deployment, future RUL would not "
    "be known at prediction time."
)


# =========================================================
# MAINTENANCE STATUS MESSAGE
# =========================================================

if maintenance_status == "Healthy":

    st.success(
        f"Engine {selected_engine} is currently classified as "
        "Healthy based on the predicted RUL."
    )

elif maintenance_status == "Monitor":

    st.warning(
        f"Engine {selected_engine} is classified as Monitor. "
        "The predicted RUL suggests continued observation."
    )

elif maintenance_status == "Maintenance Recommended":

    st.warning(
        f"Engine {selected_engine} is classified as "
        "Maintenance Recommended based on the predicted RUL."
    )

else:

    st.error(
        f"Engine {selected_engine} is classified as Critical. "
        "The predicted RUL is below 10 cycles."
    )


st.caption(
    "Note: These status categories are dashboard interpretation "
    "rules for this project. They are not official aviation "
    "maintenance limits or regulatory thresholds."
)


# =========================================================
# PREDICTION EXPLANATION
# =========================================================

st.info(
    f"The {selected_model} estimates that Engine "
    f"{selected_engine} has approximately "
    f"{predicted_rul:.2f} operating cycles of useful life "
    "remaining based on the project's trained model and "
    "existing preprocessing pipeline."
)


# =========================================================
# DEBUG / PIPELINE INFORMATION
# =========================================================

with st.expander(
    "Prediction Details"
):

    st.write(
        f"Selected engine: {selected_engine}"
    )

    st.write(
        f"Latest observed cycle: {latest_cycle}"
    )

    st.write(
        f"Selected model: {selected_model}"
    )

    st.write(
        f"Prediction input: {prediction_input_description}"
    )

    st.write(
        f"Benchmark actual RUL: {actual_rul:.2f} cycles"
    )

    st.write(
        f"Prediction error: {prediction_error:.2f} cycles"
    )

    if selected_model == "LSTM":

        st.write(
            f"Number of LSTM features: {len(lstm_features)}"
        )

        st.write(
            f"LSTM input shape: {X_engine.shape}"
        )

        st.write(
            "Model file: models/lstm_final.keras"
        )

        st.write(
            "Prediction generated using the saved trained LSTM model."
        )

    else:

        st.write(
            f"Random Forest input shape: {X_test_rf.shape}"
        )

        st.write(
            "Model file: models/random_forest.pkl"
        )

        st.write(
            "Prediction generated using the saved trained Random Forest model."
        )