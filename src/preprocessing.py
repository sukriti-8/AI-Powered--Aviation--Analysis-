import pandas as pd
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



# Load the datasets


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



# Test the data loading

if __name__ == "__main__":

    train, test, rul = load_data()

   
    # Sensor variability analysis
  

    sensor_columns = [
        column
        for column in train.columns
        if column.startswith("sensor_")
    ]

    sensor_variance = train[sensor_columns].var()

    print("\n===== SENSOR VARIANCE =====")
    print(sensor_variance.sort_values())

  
    # Inspect sensors with very low variance
  

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

   
    # Identify constant sensors
    

    print("\n===== CONSTANT / NEAR-CONSTANT SENSORS =====")

    for sensor, variance in sensor_variance.sort_values().items():
        if variance == 0:
            print(f"{sensor}: variance = {variance}")

    
    # Remove sensors with only one unique value
    

    constant_sensors = [
        sensor
        for sensor in sensor_columns
        if train[sensor].nunique() == 1
    ]

    print("\n===== REMOVING CONSTANT SENSORS =====")
    print(constant_sensors)

    train = train.drop(columns=constant_sensors)
    test = test.drop(columns=constant_sensors)
        # ========================================================
    # M5: NORMALIZATION
    # ========================================================

    feature_columns = [
        column
        for column in train.columns
        if column.startswith("setting_") or column.startswith("sensor_")
    ]

    scaler = StandardScaler()

    scaler.fit(train[feature_columns])

    train[feature_columns] = scaler.transform(train[feature_columns])
    test[feature_columns] = scaler.transform(test[feature_columns])

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

   
    # Verify train and test columns match
    

    print("\n===== COLUMN CONSISTENCY CHECK =====")

    train_columns = set(train.columns)
    test_columns = set(test.columns)

    print("Columns only in train:", train_columns - test_columns)
    print("Columns only in test:", test_columns - train_columns)
    print("Columns match:", train_columns == test_columns)

    
    # Display processed data
    

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