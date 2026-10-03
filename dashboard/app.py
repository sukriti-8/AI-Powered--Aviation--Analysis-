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
#rf
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
# LOAD DATA AND MODEL
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
    "NASA C-MAPSS FD001 data and saved LSTM model loaded successfully."
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
        "Model",
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
# ENGINE INFORMATION
# =========================================================

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
        "Predicted RUL",
        f"{predicted_rul:.2f} cycles"
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
    "LSTM Prediction Details"
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

else:

    st.write(
        f"Random Forest input shape: {X_test_rf.shape}"
    )

    st.write(
        "Model file: models/random_forest.pkl"
    )

    st.write(
        "Prediction generated using the saved trained model."
    )