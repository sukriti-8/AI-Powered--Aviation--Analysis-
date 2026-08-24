import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing import preprocess_data
from src.models import load_model
from src.evaluate import calculate_phm08_score


if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Preprocess the data
    # --------------------------------------------------------

    (
        train,
        test,
        rul,
        X_lstm,
        y_lstm,
        lstm_units
    ) = preprocess_data()

    # --------------------------------------------------------
    # 2. Load the already-trained tuned Random Forest
    # --------------------------------------------------------

    model = load_model(
        "models/random_forest.pkl"
    )

    # --------------------------------------------------------
    # 3. Select the final observed row of each test engine
    # --------------------------------------------------------

    test_last = (
        test
        .sort_values(["unit", "cycle"])
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
    )

    # --------------------------------------------------------
    # 4. Prepare RF input features
    # --------------------------------------------------------

    X_test_rf = test_last.drop(
        columns=["unit"]
    )

    # The test set does not contain RUL,
    # so all remaining columns are model inputs.
    actual_rul = rul["RUL"].to_numpy()

    # --------------------------------------------------------
    # 5. Check alignment
    # --------------------------------------------------------

    if len(X_test_rf) != len(actual_rul):

        raise ValueError(
            "Number of RF test rows does not match "
            "number of RUL values."
        )

    # --------------------------------------------------------
    # 6. Predict RUL
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_rf
    )

    # --------------------------------------------------------
    # 7. Calculate MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_rul,
        predictions
    )

    # --------------------------------------------------------
    # 8. Calculate RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            actual_rul,
            predictions
        )
    )

    # --------------------------------------------------------
    # 9. Calculate PHM08 score
    # --------------------------------------------------------

    phm08 = calculate_phm08_score(
        actual_rul,
        predictions
    )

    # --------------------------------------------------------
    # 10. Print results
    # --------------------------------------------------------

    print("\n===== FINAL TEST RANDOM FOREST =====")

    print(
        "Test rows:",
        X_test_rf.shape
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

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("PHM08 score:", phm08)

    print("\n===== FIRST 10 TEST PREDICTIONS =====")

    for i in range(10):

        print(
            f"Engine {i + 1}: "
            f"Actual RUL = {actual_rul[i]}, "
            f"Predicted RUL = {predictions[i]:.2f}"
        )