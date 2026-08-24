import joblib
from sklearn.model_selection import train_test_split


def split_data(train):

    X = train.drop(columns=["RUL", "unit"])
    y = train["RUL"]

    train_units, validation_units = train_test_split(
        train["unit"].unique(), #unit == engine number 
        test_size=0.2, #80% train, 20% validation 
        random_state=42 #same engine split 
    )
#create 2 datasets
    train_data = train[train["unit"].isin(train_units)]
    validation_data = train[train["unit"].isin(validation_units)]
#x = input features, y = target 
    X_train = train_data.drop(columns=["RUL", "unit"])
    y_train = train_data["RUL"]

    X_validation = validation_data.drop(columns=["RUL", "unit"])
    y_validation = validation_data["RUL"]

    return X_train, X_validation, y_train, y_validation

from sklearn.ensemble import RandomForestRegressor

#baseline rf 
def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=100, #built 100 decision tress 
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

from sklearn.model_selection import GridSearchCV

#another rf 
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
#grid seacrh CV - train several versions of rf using diff hyperparameters combination nd compare them 
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=3, #evaluates teh different parameter using cross validation in tuning process
        scoring="neg_mean_absolute_error", #to select the best model based on score
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, path="models/random_forest.pkl"):

    joblib.dump(model, path)

    print(f"Model saved to: {path}")
def load_model(path="models/random_forest.pkl"):

    return joblib.load(path)


def predict_rul(model, X):

    predictions = model.predict(X)

    return predictions