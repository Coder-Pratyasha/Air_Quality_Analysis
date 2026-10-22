import pandas as pd


# --------------------------------------------------
# 1. Load model dataset
# --------------------------------------------------

df = pd.read_csv("data/model_data.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])

# Sort chronologically
df = df.sort_values("Datetime").reset_index(drop=True)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Find time-based cutoff
# --------------------------------------------------

# Get all unique timestamps
unique_dates = sorted(df["Datetime"].unique())

# Use the first 80% of time for training
cutoff_index = int(len(unique_dates) * 0.8)

cutoff_date = unique_dates[cutoff_index]

print("\nCutoff date:", cutoff_date)


# --------------------------------------------------
# 3. Create Train/Test Sets
# --------------------------------------------------

train = df[df["Datetime"] < cutoff_date].copy()

test = df[df["Datetime"] >= cutoff_date].copy()


# --------------------------------------------------
# 4. Separate Features and Target
# --------------------------------------------------

X_train = train.drop(
    columns=["target_pm25", "Datetime"]
)

y_train = train["target_pm25"]

X_test = test.drop(
    columns=["target_pm25", "Datetime"]
)

y_test = test["target_pm25"]


# --------------------------------------------------
# 5. Display Split Information
# --------------------------------------------------

print("\n" + "=" * 60)
print("TIME-BASED TRAIN TEST SPLIT")
print("=" * 60)

print("\nTraining data:")
print("Shape:", X_train.shape)
print("Start:", train["Datetime"].min())
print("End:", train["Datetime"].max())

print("\nTesting data:")
print("Shape:", X_test.shape)
print("Start:", test["Datetime"].min())
print("End:", test["Datetime"].max())


# --------------------------------------------------
# 6. Save Train/Test Data
# --------------------------------------------------

train.to_csv(
    "data/train.csv",
    index=False
)

test.to_csv(
    "data/test.csv",
    index=False
)

print("\nTrain data saved to: data/train.csv")
print("Test data saved to: data/test.csv")