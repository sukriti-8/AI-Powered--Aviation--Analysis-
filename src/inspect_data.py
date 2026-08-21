import pandas as pd


# Change these paths if your files are somewhere else
TRAIN_PATH = "data/raw/train_FD001.txt"
TEST_PATH = "data/raw/test_FD001.txt"
RUL_PATH = "data/raw/RUL_FD001.txt"


# Load raw files
train = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None)
test = pd.read_csv(TEST_PATH, sep=r"\s+", header=None)
rul = pd.read_csv(RUL_PATH, sep=r"\s+", header=None)


print("\n===== TRAIN DATA =====")
print("Shape:", train.shape)
print(train.head())

print("\n===== TEST DATA =====")
print("Shape:", test.shape)
print(test.head())

print("\n===== RUL DATA =====")
print("Shape:", rul.shape)
print(rul.head())


print("\n===== TRAIN INFO =====")
print(train.info())

print("\n===== MISSING VALUES =====")
print(train.isnull().sum())

print("\n===== UNIQUE ENGINES =====")
print("Train engines:", train[0].nunique())
print("Test engines:", test[0].nunique())