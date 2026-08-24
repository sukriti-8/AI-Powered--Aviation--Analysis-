from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_model(model, X_validation, y_validation):

    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions
        )
    )

    return mae, rmse