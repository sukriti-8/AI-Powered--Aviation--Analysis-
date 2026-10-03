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
import plotly.express as px

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
# SENSOR LIST
# =========================================================

sensor_columns = [
    column
    for column in test.columns
    if column.startswith("sensor_")
    and not column.endswith("_smooth")
    and "_rolling_" not in column
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
# SENSOR TRENDS
# =========================================================

st.header("Sensor Trends")

st.write(
    "Explore how an individual sensor changes across the "
    "operating cycles of the selected engine."
)

selected_sensor = st.selectbox(
    "Select a sensor",
    sensor_columns
)

sensor_plot_data = engine_data[
    ["cycle", selected_sensor]
].copy()

sensor_plot = px.line(
    sensor_plot_data,
    x="cycle",
    y=selected_sensor,
    markers=True,
    title=f"{selected_sensor} Trend — Engine {selected_engine}",
    labels={
        "cycle": "Operating Cycle",
        selected_sensor: "Sensor Value"
    }
)

sensor_plot.update_layout(
    hovermode="x unified"
)

st.plotly_chart(
    sensor_plot,
    use_container_width=True
)

# =========================================================
# MODEL PERFORMANCE COMPARISON
# =========================================================

st.header("Model Performance Comparison")

st.write(
    "The following results are the documented evaluation results "
    "of the trained models on the NASA C-MAPSS FD001 test set."
)

performance_data = pd.DataFrame(
    {
        "Model": [
            "Random Forest",
            "LSTM"
        ],
        "MAE": [
            13.63,
            10.63
        ],
        "RMSE": [
            18.82,
            15.24
        ],
        "PHM08 Score": [
            662.92,
            355.81
        ]
    }
)

st.dataframe(
    performance_data,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# PERFORMANCE METRICS
# =========================================================

st.subheader("Documented Evaluation Metrics")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "LSTM MAE",
        "10.63"
    )

    st.caption(
        "Mean Absolute Error"
    )

with col2:

    st.metric(
        "LSTM RMSE",
        "15.24"
    )

    st.caption(
        "Root Mean Squared Error"
    )

with col3:

    st.metric(
        "LSTM PHM08",
        "355.81"
    )

    st.caption(
        "NASA/PHM08 scoring metric"
    )


# =========================================================
# MODEL COMPARISON EXPLANATION
# =========================================================

st.info(
    "On the documented NASA C-MAPSS FD001 test-set evaluation, "
    "the LSTM produced lower MAE, RMSE, and PHM08 values than "
    "the Random Forest baseline. These results describe this "
    "specific FD001 evaluation and do not imply that LSTM will "
    "always outperform Random Forest on every dataset."
)
# =========================================================
# OVERALL ACTUAL VS PREDICTED RUL
# =========================================================

st.header("Overall Actual vs Predicted RUL")

st.write(
    "This plot compares the predicted RUL with the benchmark "
    "actual RUL for all 100 test engines."
)


# =========================================================
# GENERATE PREDICTIONS FOR ALL TEST ENGINES
# =========================================================

if selected_model == "LSTM":

    all_engine_sequences = []

    for engine_id in sorted_engine_ids:

        current_engine_data = (
            test[
                test["unit"] == engine_id
            ]
            .sort_values("cycle")
            .copy()
        )

        current_sequence = create_engine_sequence(
            current_engine_data,
            lstm_features,
            window=30
        )

        all_engine_sequences.append(
            current_sequence[0]
        )

    X_all_lstm = np.stack(
        all_engine_sequences,
        axis=0
    )

    if X_all_lstm.shape != (
        len(sorted_engine_ids),
        30,
        18
    ):

        st.error(
            f"Unexpected overall LSTM input shape: "
            f"{X_all_lstm.shape}. "
            f"Expected "
            f"({len(sorted_engine_ids)}, 30, 18)."
        )

        st.stop()

    all_predictions = (
        lstm_model.predict(
            X_all_lstm,
            verbose=0
        )
        .flatten()
    )


else:

    # -----------------------------------------------------
    # RANDOM FOREST PREDICTIONS FOR ALL TEST ENGINES
    # -----------------------------------------------------

    all_predictions = (
        rf_model.predict(
            X_test_rf
        )
        .flatten()
    )


# =========================================================
# CREATE COMPARISON DATA
# =========================================================

overall_comparison = pd.DataFrame(
    {
        "Engine": sorted_engine_ids,
        "Actual RUL": [
            actual_rul_map[engine_id]
            for engine_id in sorted_engine_ids
        ],
        "Predicted RUL": all_predictions
    }
)


# =========================================================
# CREATE SCATTER PLOT
# =========================================================

max_rul_value = max(
    overall_comparison["Actual RUL"].max(),
    overall_comparison["Predicted RUL"].max()
)

comparison_plot = px.scatter(
    overall_comparison,
    x="Actual RUL",
    y="Predicted RUL",
    hover_data=["Engine"],
    title=f"{selected_model}: Actual vs Predicted RUL",
    labels={
        "Actual RUL": "Actual RUL (cycles)",
        "Predicted RUL": "Predicted RUL (cycles)"
    }
)


# Add ideal prediction line:
# Predicted RUL = Actual RUL

comparison_plot.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=max_rul_value,
    y1=max_rul_value,
    line=dict(
        dash="dash"
    )
)


comparison_plot.update_layout(
    hovermode="closest"
)


st.plotly_chart(
    comparison_plot,
    use_container_width=True
)


st.caption(
    "Points closer to the diagonal line indicate predictions "
    "closer to the benchmark actual RUL."
)
# =========================================================
# PREDICTION DETAILS
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