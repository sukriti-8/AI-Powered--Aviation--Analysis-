import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# File paths
TRAIN_PATH = "data/raw/train_FD001.txt"
TEST_PATH = "data/raw/test_FD001.txt"
RUL_PATH = "data/raw/RUL_FD001.txt"



# Column names for NASA C-MAPSS FD001
COLUMN_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]



# Load datasets
def load_data():

    train = pd.read_csv(
        TRAIN_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    test = pd.read_csv(
        TEST_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    rul = pd.read_csv(
        RUL_PATH,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    return train, test, rul


# M7.1: Sensor smoothing
def smooth_sensors(df, sensor_columns, window=5):

    df = df.copy()

    for sensor in sensor_columns:

        df[f"{sensor}_smooth"] = (
            df.groupby("unit")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

    return df


# M7.2: Random Forest feature generation
def create_rf_features(df, sensor_columns, window=10):

    df = df.copy()

    for sensor in sensor_columns:

        grouped = df.groupby("unit")[sensor]

        # Rolling mean
        df[f"{sensor}_rolling_mean"] = (
            grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

        # Rolling standard deviation
        df[f"{sensor}_rolling_std"] = (
            grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1
                ).std().fillna(0)
            )
        )

        # Rolling slope
        df[f"{sensor}_rolling_slope"] = (
            grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=2
                ).apply(
                    lambda y: np.polyfit(
                        range(len(y)),
                        y,
                        1
                    )[0],
                    raw=True
                ).fillna(0)
            )
        )

    return df

def create_lstm_sequences(df, feature_columns, window=30): #sequence function (for 100 cycles a window of 30 cycle is created)
    sequences = []
    targets = []

    for unit, engine_data in df.groupby("unit"):

        engine_data = engine_data.sort_values("cycle")

        features = engine_data[feature_columns].to_numpy()
        rul = engine_data["RUL"].to_numpy()

        for i in range(len(engine_data) - window + 1):
            sequences.append(
                features[i:i + window]
            )

            targets.append(
                rul[i + window - 1]
            )

    return np.array(sequences), np.array(targets)

# Main pipeline
if __name__ == "__main__":

    train, test, rul = load_data()


    
    # M4: Remove constant sensors
    sensor_columns = [
        column
        for column in train.columns
        if column.startswith("sensor_")
    ]

    sensor_variance = train[sensor_columns].var()

    print("\n===== SENSOR VARIANCE =====")
    print(sensor_variance.sort_values())

    low_variance_sensors = sensor_variance[
        sensor_variance < 1e-5
    ].index

    print("\n===== LOW-VARIANCE SENSOR DETAILS =====")

    for sensor in low_variance_sensors:

        print(f"\n{sensor}")
        print(f"  Variance: {train[sensor].var()}")
        print(f"  Unique values: {train[sensor].nunique()}")
        print(f"  Minimum: {train[sensor].min()}")
        print(f"  Maximum: {train[sensor].max()}")
        print(f"  Range: {train[sensor].max() - train[sensor].min()}")

    print("\n===== CONSTANT / NEAR-CONSTANT SENSORS =====")

    for sensor, variance in sensor_variance.sort_values().items():

        if variance == 0:
            print(f"{sensor}: variance = {variance}")

    constant_sensors = [
        sensor
        for sensor in sensor_columns
        if train[sensor].nunique() == 1
    ]

    print("\n===== REMOVING CONSTANT SENSORS =====")
    print(constant_sensors)

    train = train.drop(columns=constant_sensors)
    test = test.drop(columns=constant_sensors)


    
    # M5: Normalization
   
    feature_columns = [
        column
        for column in train.columns
        if column.startswith("setting_")
        or column.startswith("sensor_")
    ]

    scaler = StandardScaler()

    scaler.fit(train[feature_columns])

    train[feature_columns] = scaler.transform(
        train[feature_columns]
    )

    test[feature_columns] = scaler.transform(
        test[feature_columns]
    )

    print("\n===== NORMALIZATION =====")
    print("Features normalized:", feature_columns)

    print("\n===== NORMALIZED TRAIN DATA =====")
    print(train.head())

    print("\n===== NORMALIZED TRAIN MEAN =====")
    print(train[feature_columns].mean())

    print("\n===== NORMALIZED TRAIN STANDARD DEVIATION =====")
    print(train[feature_columns].std())

    print("\n===== SHAPES AFTER SENSOR REMOVAL =====")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)


   
    # Column consistency check
    print("\n===== COLUMN CONSISTENCY CHECK =====")

    train_columns = set(train.columns)
    test_columns = set(test.columns)

    print("Columns only in train:", train_columns - test_columns)
    print("Columns only in test:", test_columns - train_columns)
    print("Columns match:", train_columns == test_columns)


    # M6: RUL target generation
    
    RUL_MAX = 125

    failure_cycles = train.groupby("unit")["cycle"].max()

    train["RUL"] = (
        train["unit"].map(failure_cycles)
        - train["cycle"]
    ).clip(upper=RUL_MAX)

    print("\n===== RUL GENERATION =====")
    print(train[["unit", "cycle", "RUL"]].head(10))

    print("\n===== RUL RANGE =====")
    print("Minimum RUL:", train["RUL"].min())
    print("Maximum RUL:", train["RUL"].max())

    print("\n===== RUL DISTRIBUTION CHECK =====")
    print(
        train["RUL"]
        .value_counts()
        .sort_index()
        .head(10)
    )


  
    # M7.1: Sensor smoothing
    sensor_columns = [
        column
        for column in train.columns
        if column.startswith("sensor_")
        and not column.endswith("_smooth")
    ]

    train = smooth_sensors(
        train,
        sensor_columns,
        window=5
    )

    test = smooth_sensors(
        test,
        sensor_columns,
        window=5
    )

    print("\n===== M7.1 SENSOR SMOOTHING =====")
    print("Smoothing window: 5 cycles")
    print("Sensors smoothed:", sensor_columns)

    print("\n===== RAW VS SMOOTHED =====")

    print(
        train[
            [
                "unit",
                "cycle",
                "sensor_2",
                "sensor_2_smooth"
            ]
        ].head(10)
    )


    # M7.2: Random Forest features
    train = create_rf_features(
        train,
        sensor_columns,
        window=10
    )

    test = create_rf_features(
        test,
        sensor_columns,
        window=10
    )

    print("\n===== M7.2 RANDOM FOREST FEATURES =====")
    print("Rolling window: 10 cycles")

    print("\n===== RF FEATURE EXAMPLE =====")

    print(
        train[
            [
                "unit",
                "cycle",
                "sensor_2",
                "sensor_2_rolling_mean",
                "sensor_2_rolling_std",
                "sensor_2_rolling_slope"
            ]
        ].head(15)
    )


    # M7.3: LSTM SEQUENCES
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

    X_lstm, y_lstm = create_lstm_sequences(
        train,
        lstm_features,
        window=30
    )

    print("\n===== M7.3 LSTM SEQUENCES =====")
    print("Sequence window: 30 cycles")
    print("LSTM features:", lstm_features)

    print("\n===== LSTM SHAPES =====")
    print("X_lstm shape:", X_lstm.shape)
    print("y_lstm shape:", y_lstm.shape)

    print("\n===== FIRST LSTM TARGET =====")
    print("Target RUL:", y_lstm[0])


    
    # Final processed data inspection
    print("\n===== TRAIN DATA =====")
    print(train.head())

    print("\n===== TRAIN SHAPE =====")
    print(train.shape)

    print("\n===== TRAIN COLUMNS =====")
    print(train.columns.tolist())

    print("\n===== TEST DATA =====")
    print(test.head())

    print("\n===== TEST SHAPE =====")
    print(test.shape)

    print("\n===== RUL DATA =====")
    print(rul.head())

    print("\n===== RUL SHAPE =====")
    print(rul.shape)