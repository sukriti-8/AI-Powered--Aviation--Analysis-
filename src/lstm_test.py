import os
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing import preprocess_data

from src.models import (
    split_lstm_data,
    build_lstm_model,
    train_lstm_model
)

from src.evaluate import calculate_phm08_score

from src.visualization import show_project_visualizations


# ------------------------------------------------------------
# Create the final 30-cycle sequence for each test engine
# ------------------------------------------------------------
def create_test_sequences(test, lstm_features, window=30):

    sequences = []

    # Process engines in numerical order so that
    # predictions match the order in RUL_FD001.txt
    for unit in sorted(test["unit"].unique()):

        engine_data = test[
            test["unit"] == unit
        ].sort_values("cycle")

        features = engine_data[
            lstm_features
        ].to_numpy()

        # If an engine has fewer than 30 cycles,
        # repeat the first recorded row to pad it.
        if len(features) < window:

            padding = np.repeat(
                features[:1],
                window - len(features),
                axis=0
            )

            features = np.vstack(
                [padding, features]
            )

        # Use the last 30 observed cycles
        sequence = features[-window:]

        sequences.append(sequence)

    return np.array(sequences)


if __name__ == "__main__":

    # ========================================================
    # 1. PREPROCESS DATA
    # ========================================================

    (
        train,
        test,
        rul,
        X_lstm,
        y_lstm,
        lstm_units
    ) = preprocess_data()


    # ========================================================
    # 2. CREATE ENGINE-BASED TRAIN/VALIDATION SPLIT
    # ========================================================

    (
        X_train_lstm,
        X_val_lstm,
        y_train_lstm,
        y_val_lstm,
        _
    ) = split_lstm_data(
        X_lstm,
        y_lstm,
        lstm_units
    )


    # ========================================================
    # 3. BUILD FINAL MSE LSTM
    # ========================================================

    model = build_lstm_model(
        (30, 18),
        loss_type="mse"
    )


    # ========================================================
    # 4. TRAIN USING EARLY STOPPING
    # ========================================================

    history = train_lstm_model(
        model,
        X_train_lstm,
        y_train_lstm,
        X_val_lstm,
        y_val_lstm
    )


    # ========================================================
    # 5. IDENTIFY THE SAME 18 FEATURES USED BY LSTM
    # ========================================================

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


    # ========================================================
    # 6. CREATE ONE FINAL 30-CYCLE WINDOW PER TEST ENGINE
    # ========================================================

    X_test_lstm = create_test_sequences(
        test,
        lstm_features,
        window=30
    )


    # ========================================================
    # 7. GET TRUE TEST RUL
    # ========================================================

    actual_rul = rul["RUL"].to_numpy()


    # ========================================================
    # 8. CHECK TEST DATA ALIGNMENT
    # ========================================================

    if len(X_test_lstm) != len(actual_rul):

        raise ValueError(
            "Number of test sequences does not match "
            "number of RUL values."
        )


    # ========================================================
    # 9. PREDICT RUL
    # ========================================================

    predictions = model.predict(
        X_test_lstm,
        verbose=0
    ).flatten()


    # ========================================================
    # 10. SHOW ALL THREE PROJECT VISUALIZATIONS
    #
    # Graph 1: RUL distribution
    # Graph 2: LSTM training vs validation loss
    # Graph 3: Actual vs predicted RUL
    # ========================================================

    show_project_visualizations(
        train,
        history,
        actual_rul,
        predictions
    )


    # ========================================================
    # 11. CALCULATE MAE
    # ========================================================

    mae = mean_absolute_error(
        actual_rul,
        predictions
    )


    # ========================================================
    # 12. CALCULATE RMSE
    # ========================================================

    rmse = np.sqrt(
        mean_squared_error(
            actual_rul,
            predictions
        )
    )


    # ========================================================
    # 13. CALCULATE PHM08 SCORE
    # ========================================================

    phm08 = calculate_phm08_score(
        actual_rul,
        predictions
    )


    # ========================================================
    # 14. SAVE FINAL LSTM MODEL
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = "models/lstm_final.keras"

    model.save(model_path)


    # ========================================================
    # 15. FINAL RESULTS
    # ========================================================

    print("\n===== FINAL TEST LSTM =====")

    print(
        "Test sequences:",
        X_test_lstm.shape
    )

    print(
        "Actual RUL:",
        actual_rul.shape
    )

    print(
        "Predictions:",
        predictions.shape
    )


    print("\n===== FINAL TEST METRICS =====")

    print(
        "MAE:",
        mae
    )

    print(
        "RMSE:",
        rmse
    )

    print(
        "PHM08 score:",
        phm08
    )


    print("\n===== FIRST 10 TEST PREDICTIONS =====")

    for i in range(10):

        print(
            f"Engine {i + 1}: "
            f"Actual RUL = {actual_rul[i]}, "
            f"Predicted RUL = {predictions[i]:.2f}"
        )


    print(
        "\nBest validation epoch:",
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )


    print(
        "\nFinal LSTM saved to:",
        model_path
    )