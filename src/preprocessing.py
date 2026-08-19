import pandas as pd


# ============================================================
# File paths
# ============================================================

TRAIN_PATH = "data/raw/train_FD001.txt"
TEST_PATH = "data/raw/test_FD001.txt"
RUL_PATH = "data/raw/RUL_FD001.txt"


# ============================================================
# Column names for NASA C-MAPSS FD001
# ============================================================

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


# ============================================================
# Load the datasets
# ============================================================

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


# ============================================================
# Test the data loading
# ============================================================

if __name__ == "__main__":

    train, test, rul = load_data()

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