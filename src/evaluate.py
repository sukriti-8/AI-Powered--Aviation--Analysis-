import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Evaluate LSTM predictions using MAE and RMSE
def evaluate_lstm(model, X_validation, y_validation):

    # Predict RUL for validation sequences
    predictions = model.predict(
        X_validation,
        verbose=0
    ).flatten()

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    # Calculate Root Mean Squared Error
    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions
        )
    )

    return mae, rmse, predictions