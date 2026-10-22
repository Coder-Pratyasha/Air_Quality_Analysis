import pandas as pd


# --------------------------------------------------
# 1. Load cleaned data
# --------------------------------------------------

df = pd.read_csv("data/cleaned_air_quality.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])

# Sort data city-wise and chronologically
df = df.sort_values(["City", "Datetime"]).reset_index(drop=True)

print("Dataset loaded:", df.shape)


# --------------------------------------------------
# 2. Create Time-Based Features
# --------------------------------------------------

df["hour"] = df["Datetime"].dt.hour

df["day_of_week"] = df["Datetime"].dt.dayofweek

df["month"] = df["Datetime"].dt.month

df["day"] = df["Datetime"].dt.day

# 1 = Weekend, 0 = Weekday
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# --------------------------------------------------
# 3. Create PM2.5 Lag Features
# --------------------------------------------------

# PM2.5 from previous 1 hour
df["pm25_lag_1"] = (
    df.groupby("City")["PM2.5"]
    .shift(1)
)

# PM2.5 from previous 3 hours
df["pm25_lag_3"] = (
    df.groupby("City")["PM2.5"]
    .shift(3)
)

# PM2.5 from previous 6 hours
df["pm25_lag_6"] = (
    df.groupby("City")["PM2.5"]
    .shift(6)
)

# PM2.5 from previous 12 hours
df["pm25_lag_12"] = (
    df.groupby("City")["PM2.5"]
    .shift(12)
)

# PM2.5 from previous 24 hours
df["pm25_lag_24"] = (
    df.groupby("City")["PM2.5"]
    .shift(24)
)


# --------------------------------------------------
# 4. Create Rolling Features
# --------------------------------------------------

# Average PM2.5 over previous 6 hours
df["pm25_rolling_6h"] = (
    df.groupby("City")["PM2.5"]
    .transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
)

# Average PM2.5 over previous 24 hours
df["pm25_rolling_24h"] = (
    df.groupby("City")["PM2.5"]
    .transform(
        lambda x: x.shift(1).rolling(24).mean()
    )
)


# --------------------------------------------------
# 5. Create Target Variable
# --------------------------------------------------

# We want to predict PM2.5 for the NEXT hour.

df["target_pm25"] = (
    df.groupby("City")["PM2.5"]
    .shift(-1)
)


# --------------------------------------------------
# 6. Select Features
# --------------------------------------------------

feature_columns = [
    "hour",
    "day_of_week",
    "month",
    "day",
    "is_weekend",

    # Other pollutants
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",

    # PM2.5 lag features
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24",

    # Rolling features
    "pm25_rolling_6h",
    "pm25_rolling_24h"
]


# --------------------------------------------------
# 7. Create Final Model Dataset
# --------------------------------------------------

df_model = df[
    [
        "City",
        "Datetime"
    ]
    + feature_columns
    + ["target_pm25"]
].dropna()


# --------------------------------------------------
# 8. Save Model Dataset
# --------------------------------------------------

output_path = "data/model_data.csv"

df_model.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 9. Display Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("Final shape:", df_model.shape)

print("\nFeatures:")
print(feature_columns)

print("\nFinal columns:")
print(df_model.columns.tolist())

print("\nFirst 5 rows:")
print(df_model.head())

print("\nTarget statistics:")
print(df_model["target_pm25"].describe())

print("\nSaved to:")
print(output_path)