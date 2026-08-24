import joblib
from sklearn.model_selection import train_test_split


def split_data(train):

    X = train.drop(columns=["RUL", "unit"])
    y = train["RUL"]

    train_units, validation_units = train_test_split(
        train["unit"].unique(),
        test_size=0.2,
        random_state=42
    )

    train_data = train[train["unit"].isin(train_units)]
    validation_data = train[train["unit"].isin(validation_units)]

    X_train = train_data.drop(columns=["RUL", "unit"])
    y_train = train_data["RUL"]

    X_validation = validation_data.drop(columns=["RUL", "unit"])
    y_validation = validation_data["RUL"]

    return X_train, X_validation, y_train, y_validation

from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

from sklearn.model_selection import GridSearchCV


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

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, path="models/random_forest.pkl"):

    joblib.dump(model, path)

    print(f"Model saved to: {path}")