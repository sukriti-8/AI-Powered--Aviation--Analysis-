import sys
from pathlib import Path

# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

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
# LOAD EXISTING PROJECT DATA
# =========================================================

@st.cache_data
def load_processed_data():
    """
    Reuse the existing project preprocessing pipeline.

    The dashboard does not create a separate preprocessing
    pipeline and does not retrain any model.
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

    model_path = (
        PROJECT_ROOT
        / "models"
        / "lstm_final.keras"
    )

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

    model_path = (
        PROJECT_ROOT
        / "models"
        / "random_forest.pkl"
    )

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

    Expected output shape:

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

    These are project interpretation rules.
    They are NOT official aviation maintenance thresholds.
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
    "✈️ AI-POWERED AVIATION ANALYTICS"
)

st.subheader(
    "Aircraft Engine Predictive Maintenance"
)

st.write(
    "Estimate an aircraft engine's Remaining Useful Life (RUL) "
    "from historical sensor data and understand its current "
    "maintenance status."
)

st.caption(
    "Project benchmark: NASA C-MAPSS FD001"
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
# ENGINE IDS
# =========================================================

engine_ids = sorted(
    test["unit"].unique()
)


# =========================================================
# ACTUAL RUL MAPPING
# =========================================================

actual_rul_map = dict(
    zip(
        engine_ids,
        rul["RUL"].to_numpy()
    )
)


# =========================================================
# FLEET-WIDE LSTM PREDICTIONS
# =========================================================

# The fleet overview uses the primary LSTM model.
# These predictions are generated from the saved model
# using the same 30-cycle / 18-feature input used above.

all_engine_sequences = []

for engine_id in engine_ids:

    engine_data_for_fleet = (
        test[test["unit"] == engine_id]
        .sort_values("cycle")
        .copy()
    )

    sequence = create_engine_sequence(
        engine_data_for_fleet,
        lstm_features,
        window=30
    )

    all_engine_sequences.append(
        sequence[0]
    )


X_all_lstm = np.stack(
    all_engine_sequences,
    axis=0
)


# Safety check: expected shape = (100, 30, 18)
if X_all_lstm.shape[1:] != (30, 18):

    st.error(
        f"Unexpected fleet LSTM input shape: "
        f"{X_all_lstm.shape}. "
        "Expected (number_of_engines, 30, 18)."
    )

    st.stop()


all_lstm_predictions = (
    lstm_model.predict(
        X_all_lstm,
        verbose=0
    )
    .flatten()
)


fleet_status_df = pd.DataFrame({
    "Engine": engine_ids,
    "Predicted RUL": all_lstm_predictions
})


fleet_status_df["Status"] = (
    fleet_status_df["Predicted RUL"]
    .apply(get_maintenance_status)
)


# =========================================================
# DATA SOURCE
# =========================================================

st.header("Data Source")

st.info(
    "This demonstration uses NASA C-MAPSS FD001, a benchmark "
    "dataset containing simulated aircraft engine sensor data. "
    "The current trained models are specifically built for this "
    "dataset and its preprocessing pipeline."
)


# =========================================================
# FLEET HEALTH OVERVIEW
# =========================================================

st.header("Fleet Health Overview")

st.caption(
    "Fleet classification is based on the primary LSTM model and "
    "the project-defined RUL interpretation rules."
)


# =========================================================
# COUNT ENGINES IN EACH CATEGORY
# =========================================================

healthy_count = int(
    (
        fleet_status_df["Status"]
        == "Healthy"
    ).sum()
)

monitor_count = int(
    (
        fleet_status_df["Status"]
        == "Monitor"
    ).sum()
)

maintenance_count = int(
    (
        fleet_status_df["Status"]
        == "Maintenance Recommended"
    ).sum()
)

critical_count = int(
    (
        fleet_status_df["Status"]
        == "Critical"
    ).sum()
)


# =========================================================
# DISPLAY FLEET STATUS COUNTS
# =========================================================

fleet_col1, fleet_col2, fleet_col3, fleet_col4 = st.columns(4)

with fleet_col1:

    st.metric(
        "🟢 Healthy",
        healthy_count
    )

with fleet_col2:

    st.metric(
        "🟡 Monitor",
        monitor_count
    )

with fleet_col3:

    st.metric(
        "🟠 Maintenance Recommended",
        maintenance_count
    )

with fleet_col4:

    st.metric(
        "🔴 Critical",
        critical_count
    )


# =========================================================
# ENGINES REQUIRING ATTENTION
# =========================================================

attention_df = (
    fleet_status_df[
        fleet_status_df["Status"] != "Healthy"
    ]
    .sort_values("Predicted RUL")
    .copy()
)


if not attention_df.empty:

    st.subheader(
        "Engines Requiring Attention"
    )

    attention_display = attention_df.copy()

    attention_display["Engine"] = (
        attention_display["Engine"]
        .apply(
            lambda x: f"Engine {x}"
        )
    )

    attention_display["Predicted RUL"] = (
        attention_display["Predicted RUL"]
        .round(2)
    )

    attention_display = attention_display[
        [
            "Engine",
            "Predicted RUL",
            "Status"
        ]
    ]

    st.dataframe(
        attention_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No engines fall into the Monitor, Maintenance Recommended, "
        "or Critical categories under the current project rules."
    )


# =========================================================
# FIND ENGINE BY STATUS
# =========================================================

st.subheader(
    "Find an Engine"
)

status_filter = st.selectbox(
    "Filter engines by status",
    [
        "All Engines",
        "Healthy",
        "Monitor",
        "Maintenance Recommended",
        "Critical"
    ]
)


if status_filter == "All Engines":

    filtered_fleet = (
        fleet_status_df.copy()
    )

else:

    filtered_fleet = fleet_status_df[
        fleet_status_df["Status"]
        == status_filter
    ].copy()


if filtered_fleet.empty:

    st.info(
        f"No engines are currently classified as "
        f"'{status_filter}'."
    )

    st.stop()


engine_options = (
    filtered_fleet["Engine"]
    .tolist()
)


selected_engine = st.selectbox(
    "Select an aircraft engine",
    engine_options,
    format_func=lambda x: f"Engine {x}"
)


selected_fleet_row = fleet_status_df[
    fleet_status_df["Engine"]
    == selected_engine
].iloc[0]


fleet_selected_prediction = float(
    selected_fleet_row["Predicted RUL"]
)


fleet_selected_status = (
    selected_fleet_row["Status"]
)


st.caption(
    f"Fleet LSTM classification: "
    f"**{fleet_selected_status}** "
    f"with estimated RUL of "
    f"**{fleet_selected_prediction:.2f} cycles**."
)


# =========================================================
# SELECTED ENGINE DATA
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
# PRIMARY MODEL
# =========================================================

# LSTM is the primary model shown to normal users.

X_engine = create_engine_sequence(
    engine_data,
    lstm_features,
    window=30
)


# =========================================================
# VALIDATE EXPECTED INPUT SHAPE
# =========================================================

if X_engine.shape != (
    1,
    30,
    18
):

    st.error(
        f"Unexpected LSTM input shape: "
        f"{X_engine.shape}. "
        "Expected (1, 30, 18)."
    )

    st.stop()


# =========================================================
# LSTM PREDICTION
# =========================================================

prediction = lstm_model.predict(
    X_engine,
    verbose=0
)


predicted_rul = float(
    prediction.flatten()[0]
)


# =========================================================
# MAINTENANCE STATUS
# =========================================================

maintenance_status = (
    get_maintenance_status(
        predicted_rul
    )
)


# =========================================================
# MAIN HEALTH CARDS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Aircraft Engine",
        f"Engine {selected_engine}"
    )


with col2:

    st.metric(
        "Latest Observed Cycle",
        latest_cycle
    )


with col3:

    st.metric(
        "Estimated Remaining Useful Life",
        f"{predicted_rul:.2f} cycles"
    )


# =========================================================
# HEALTH STATUS
# =========================================================

if maintenance_status == "Healthy":

    st.success(
        f"### 🟢 {maintenance_status}\n\n"
        f"Engine {selected_engine} is currently classified "
        f"as Healthy based on the project's RUL interpretation "
        f"rules."
    )


elif maintenance_status == "Monitor":

    st.warning(
        f"### 🟡 {maintenance_status}\n\n"
        f"Engine {selected_engine} is classified as Monitor. "
        f"The predicted RUL suggests that continued observation "
        f"is appropriate within this project."
    )


elif maintenance_status == "Maintenance Recommended":

    st.warning(
        f"### 🟠 {maintenance_status}\n\n"
        f"Engine {selected_engine} has a lower predicted RUL "
        f"and is classified as Maintenance Recommended by "
        f"this project's dashboard rules."
    )


else:

    st.error(
        f"### 🔴 {maintenance_status}\n\n"
        f"Engine {selected_engine} has a predicted RUL below "
        f"10 cycles and is classified as Critical by this "
        f"project's dashboard rules."
    )


st.caption(
    "These status categories are project-defined interpretation "
    "rules, not official aviation maintenance limits or "
    "regulatory thresholds."
)


# =========================================================
# WHAT DOES THIS MEAN?
# =========================================================

st.subheader(
    "What does this mean?"
)

st.write(
    f"The model estimates approximately "
    f"**{predicted_rul:.2f} operating cycles** remaining "
    f"for Engine {selected_engine}, based on the latest "
    f"30 observed cycles of sensor data."
)

st.write(
    "RUL means Remaining Useful Life — an estimate of how "
    "many operating cycles remain before the engine reaches "
    "the degradation endpoint represented in the training data."
)


# =========================================================
# SENSOR TRENDS
# =========================================================

st.header(
    "Sensor Trends"
)

st.write(
    "Explore how an individual engine sensor changes across "
    "its observed operating cycles."
)


selected_sensor = st.selectbox(
    "Select a sensor",
    sensor_columns
)


sensor_plot_data = engine_data[
    [
        "cycle",
        selected_sensor
    ]
].copy()


sensor_plot = px.line(
    sensor_plot_data,
    x="cycle",
    y=selected_sensor,
    markers=True,
    title=(
        f"{selected_sensor} Trend — "
        f"Engine {selected_engine}"
    ),
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
# BENCHMARK DETAILS
# =========================================================

with st.expander(
    "🔍 Benchmark Details — Actual RUL & Prediction Error"
):

    actual_rul = float(
        actual_rul_map[selected_engine]
    )

    prediction_error = (
        predicted_rul
        - actual_rul
    )

    absolute_error = abs(
        prediction_error
    )

    st.write(
        "Because NASA C-MAPSS FD001 is a benchmark dataset, "
        "the actual RUL for the selected test engine is available "
        "for evaluation."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted RUL",
            f"{predicted_rul:.2f} cycles"
        )


    with col2:

        st.metric(
            "Benchmark Actual RUL",
            f"{actual_rul:.2f} cycles"
        )


    with col3:

        st.metric(
            "Absolute Prediction Error",
            f"{absolute_error:.2f} cycles"
        )


    if prediction_error > 0:

        st.info(
            f"The model overestimated the benchmark RUL "
            f"by {prediction_error:.2f} cycles."
        )


    elif prediction_error < 0:

        st.info(
            f"The model underestimated the benchmark RUL "
            f"by {abs(prediction_error):.2f} cycles."
        )


    else:

        st.success(
            "The predicted RUL exactly matches the benchmark RUL."
        )


    st.caption(
        "In a real deployment, future RUL would not be known "
        "at prediction time. Actual RUL is shown here only "
        "because this is a benchmark test dataset."
    )


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

with st.expander(
    "⚙️ Technical Analysis"
):

    st.subheader(
        "Prediction Model"
    )

    st.write(
        "The dashboard uses the trained LSTM model as the "
        "primary prediction model. The Random Forest model "
        "is retained as a baseline for comparison."
    )

    st.write(
        "**LSTM input:** Latest 30 operating cycles × "
        f"{len(lstm_features)} features"
    )

    st.write(
        "**Saved model:** `models/lstm_final.keras`"
    )

    st.divider()

    st.subheader(
        "Model Performance Comparison"
    )

    st.write(
        "These are the documented evaluation results of the "
        "trained models on the NASA C-MAPSS FD001 test set."
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

    st.subheader(
        "LSTM Evaluation Metrics"
    )

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
            "PHM08 scoring metric"
        )


    st.info(
        "On the documented NASA C-MAPSS FD001 test-set "
        "evaluation, the LSTM produced lower MAE, RMSE, "
        "and PHM08 values than the Random Forest baseline. "
        "These results describe this specific FD001 evaluation "
        "and do not imply that LSTM will always outperform "
        "Random Forest on every dataset."
    )


# =========================================================
# OVERALL MODEL EVALUATION
# =========================================================

with st.expander(
    "📊 Overall Model Evaluation — All 100 Test Engines"
):

    st.write(
        "This visualization compares predicted RUL with "
        "benchmark actual RUL for all 100 NASA C-MAPSS FD001 "
        "test engines."
    )


    # -----------------------------------------------------
    # REUSE FLEET-WIDE LSTM PREDICTIONS
    # -----------------------------------------------------

    all_lstm_predictions = (
        fleet_status_df[
            "Predicted RUL"
        ].to_numpy()
    )


    # -----------------------------------------------------
    # RANDOM FOREST PREDICTIONS FOR ALL ENGINES
    # -----------------------------------------------------

    test_last = (
        test
        .sort_values(
            ["unit", "cycle"]
        )
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
    )


    X_test_rf = test_last.drop(
        columns=["unit"]
    )


    all_rf_predictions = (
        rf_model.predict(
            X_test_rf
        )
        .flatten()
    )


    # -----------------------------------------------------
    # MODEL SELECTION FOR OVERALL PLOT
    # -----------------------------------------------------

    evaluation_model = st.radio(
        "Select model for overall evaluation",
        [
            "LSTM",
            "Random Forest"
        ],
        horizontal=True
    )


    if evaluation_model == "LSTM":

        all_predictions = (
            all_lstm_predictions
        )

    else:

        all_predictions = (
            all_rf_predictions
        )


    # -----------------------------------------------------
    # CREATE COMPARISON DATA
    # -----------------------------------------------------

    overall_comparison = pd.DataFrame(
        {
            "Engine": engine_ids,

            "Actual RUL": [
                actual_rul_map[engine_id]
                for engine_id in engine_ids
            ],

            "Predicted RUL": all_predictions
        }
    )


    # -----------------------------------------------------
    # CREATE SCATTER PLOT
    # -----------------------------------------------------

    max_rul_value = max(
        overall_comparison[
            "Actual RUL"
        ].max(),

        overall_comparison[
            "Predicted RUL"
        ].max()
    )


    comparison_plot = px.scatter(
        overall_comparison,
        x="Actual RUL",
        y="Predicted RUL",
        hover_data=["Engine"],
        title=(
            f"{evaluation_model}: "
            "Actual vs Predicted RUL"
        ),
        labels={
            "Actual RUL": "Actual RUL (cycles)",
            "Predicted RUL": "Predicted RUL (cycles)"
        }
    )


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
        "Points closer to the diagonal line represent "
        "predictions closer to the benchmark actual RUL."
    )


# =========================================================
# MODEL SELECTOR / TECHNICAL SETTINGS
# =========================================================

with st.expander(
    "🧪 Advanced Prediction Settings"
):

    st.write(
        "The LSTM is the primary model used in the main "
        "Engine Health Overview."
    )


    selected_model = st.radio(
        "Choose prediction model for technical inspection",
        [
            "LSTM",
            "Random Forest"
        ],
        horizontal=True
    )


    if selected_model == "Random Forest":

        test_last = (
            test
            .sort_values(
                ["unit", "cycle"]
            )
            .groupby("unit")
            .tail(1)
            .sort_values("unit")
        )


        X_test_rf = test_last.drop(
            columns=["unit"]
        )


        engine_position = (
            engine_ids.index(
                selected_engine
            )
        )


        rf_prediction = rf_model.predict(
            X_test_rf
        )


        rf_selected_prediction = float(
            rf_prediction[
                engine_position
            ]
        )


        st.metric(
            "Random Forest Predicted RUL",
            f"{rf_selected_prediction:.2f} cycles"
        )


        st.write(
            "Random Forest uses the existing engineered "
            "tabular features from the project preprocessing "
            "pipeline."
        )


        st.write(
            f"Random Forest input shape: "
            f"{X_test_rf.shape}"
        )


        st.write(
            "Model file: `models/random_forest.pkl`"
        )


    else:

        st.metric(
            "LSTM Predicted RUL",
            f"{predicted_rul:.2f} cycles"
        )


        st.write(
            "The LSTM receives the latest 30 observed cycles "
            "and the exact 18 features defined by the existing "
            "project preprocessing pipeline."
        )


        st.write(
            f"LSTM input shape: {X_engine.shape}"
        )


        st.write(
            "Model file: `models/lstm_final.keras`"
        )


# =========================================================
# USING YOUR OWN ENGINE DATA
# =========================================================

with st.expander(
    "✈️ Using Your Own Engine Data"
):

    st.subheader(
        "Future Real-World Use"
    )


    st.write(
        "In a real aviation predictive-maintenance system, "
        "engine sensor data could be supplied continuously "
        "or uploaded for analysis."
    )


    st.write(
        "However, the current models in this project were "
        "trained specifically on NASA C-MAPSS FD001 data. "
        "They cannot automatically accept any arbitrary "
        "aircraft dataset."
    )


    st.write(
        "For another aircraft or a different sensor "
        "configuration, the data would first need to be "
        "checked for compatibility with the model's expected "
        "features, preprocessing and input structure."
    )


    st.info(
        "Current project scope: NASA C-MAPSS FD001 benchmark. "
        "Compatible real-world data support would require "
        "additional data validation and, depending on the "
        "aircraft and sensor configuration, model adaptation "
        "or retraining."
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

with st.expander(
    "📘 Project Information"
):

    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Dataset**"
        )

        st.write(
            "NASA C-MAPSS FD001"
        )


        st.write(
            "**Problem**"
        )

        st.write(
            "Aircraft Engine Remaining Useful Life Prediction"
        )


        st.write(
            "**Purpose**"
        )

        st.write(
            "Predictive Maintenance"
        )


        st.write(
            "**Test Engines**"
        )

        st.write(
            f"{test['unit'].nunique()}"
        )


    with col2:

        st.write(
            "**Models**"
        )

        st.write(
            "Random Forest + LSTM"
        )


        st.write(
            "**Primary Model**"
        )

        st.write(
            "LSTM"
        )


        st.write(
            "**LSTM Input**"
        )

        st.write(
            "30 cycles × 18 features"
        )


        st.write(
            "**Dashboard**"
        )

        st.write(
            "Streamlit"
        )


    st.divider()


    st.write(
        "The dashboard integrates the existing preprocessing "
        "pipeline and saved trained models. It does not retrain "
        "the models when the dashboard is opened."
    )


# =========================================================
# PREDICTION DETAILS
# =========================================================

with st.expander(
    "🧠 Prediction Details"
):

    actual_rul = float(
        actual_rul_map[selected_engine]
    )


    prediction_error = (
        predicted_rul
        - actual_rul
    )


    st.write(
        f"Selected engine: {selected_engine}"
    )


    st.write(
        f"Latest observed cycle: {latest_cycle}"
    )


    st.write(
        "Primary prediction model: LSTM"
    )


    st.write(
        "Prediction input: Latest 30 cycles × 18 features"
    )


    st.write(
        f"LSTM input shape: {X_engine.shape}"
    )


    st.write(
        f"Predicted RUL: {predicted_rul:.2f} cycles"
    )


    st.write(
        f"Benchmark actual RUL: {actual_rul:.2f} cycles"
    )


    st.write(
        f"Prediction error: {prediction_error:.2f} cycles"
    )


    st.write(
        "Model file: models/lstm_final.keras"
    )


    st.write(
        "Prediction generated using the saved trained "
        "LSTM model and the existing project preprocessing."
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "Academic/project demonstration only. "
    "This dashboard is not a certified aviation maintenance "
    "system and should not be used to make real-world aircraft "
    "maintenance or safety decisions."
)