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