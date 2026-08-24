import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



# RANDOM FOREST
def split_data(train):

    X = train.drop(columns=["RUL", "unit"])
    y = train["RUL"]

    train_units, validation_units = train_test_split(
        train["unit"].unique(),  # unit = engine number
        test_size=0.2,           # 80% train, 20% validation
        random_state=42          # same engine split every time
    )

    # Create two datasets
    train_data = train[train["unit"].isin(train_units)]
    validation_data = train[train["unit"].isin(validation_units)]

    # X = input features, y = target
    X_train = train_data.drop(columns=["RUL", "unit"])
    y_train = train_data["RUL"]

    X_validation = validation_data.drop(columns=["RUL", "unit"])
    y_validation = validation_data["RUL"]

    return X_train, X_validation, y_train, y_validation


# Baseline Random Forest
def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=100,       # build 100 decision trees
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# Random Forest hyperparameter tuning
def tune_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    parameters = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5]
    }

    # GridSearchCV trains several versions of RF
    # using different hyperparameter combinations
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=3,                              # 3-fold cross-validation
        scoring="neg_mean_absolute_error", # choose using MAE
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


# Save Random Forest model
def save_model(model, path="models/random_forest.pkl"):

    joblib.dump(model, path)

    print(f"Model saved to: {path}")


# Load Random Forest model
def load_model(path="models/random_forest.pkl"):

    return joblib.load(path)


# Predict RUL using Random Forest
def predict_rul(model, X):

    predictions = model.predict(X)

    return predictions


# LSTM
# Split LSTM sequences by engine
# This prevents sequences from the same engine
# appearing in both training and validation data.
def split_lstm_data(X_lstm, y_lstm, lstm_units):

    # Get all unique engine numbers
    unique_units = np.unique(lstm_units)

    # Split engines into 80% training and 20% validation
    train_units, validation_units = train_test_split(
        unique_units,
        test_size=0.2,
        random_state=42
    )

    # Find sequences belonging to training engines
    train_mask = np.isin(
        lstm_units,
        train_units
    )

    # Find sequences belonging to validation engines
    validation_mask = np.isin(
        lstm_units,
        validation_units
    )

    # Create training sequences
    X_train_lstm = X_lstm[train_mask]
    y_train_lstm = y_lstm[train_mask]

    # Create validation sequences
    X_val_lstm = X_lstm[validation_mask]
    y_val_lstm = y_lstm[validation_mask]

    return (
        X_train_lstm,
        X_val_lstm,
        y_train_lstm,
        y_val_lstm
    )


# Build baseline LSTM model
#
# Input:
# 30 time steps × 18 features
#
# Output:
# 1 predicted RUL value
def build_lstm_model(input_shape):

    model = Sequential([

        # LSTM learns patterns across the 30 cycles
        LSTM(
            64,
            input_shape=input_shape
        ),

        # One output = predicted RUL
        Dense(1)
    ])

    # Adam updates the model's weights during training
    # MSE measures the prediction error
    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model