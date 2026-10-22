import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/city_hour.csv")

print("=" * 60)
print("INITIAL DATASET")
print("=" * 60)

print("Shape:", df.shape)


# ============================================================
# 2. CONVERT DATETIME
# ============================================================

df["Datetime"] = pd.to_datetime(df["Datetime"])

print("\nDatetime converted successfully.")


# ============================================================
# 3. SORT DATA
# ============================================================

df = df.sort_values(
    by=["City", "Datetime"]
).reset_index(drop=True)

print("Data sorted by City and Datetime.")


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows found:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print("Shape after duplicate removal:", df.shape)


# ============================================================
# 5. CHECK MISSING VALUES BEFORE CLEANING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)

missing_before = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean() * 100
})

print(
    missing_before.sort_values(
        by="Missing_Percentage",
        ascending=False
    )
)


# ============================================================
# 6. DROP COLUMNS WITH VERY HIGH MISSINGNESS
# ============================================================

# Xylene has more than 60% missing values.
# It is removed because reliable imputation would be difficult.

df = df.drop(columns=["Xylene"])

print("\nDropped column: Xylene")


# ============================================================
# 7. DEFINE POLLUTANT COLUMNS
# ============================================================

pollutant_columns = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene"
]


# ============================================================
# 8. HANDLE INFINITE VALUES
# ============================================================

df[pollutant_columns] = df[pollutant_columns].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 9. HANDLE SUSPICIOUS EXTREME VALUES
# ============================================================

# Some observations contain unrealistic extreme values.
# These are treated as invalid measurements and converted
# to NaN before interpolation.

upper_limits = {
    "PM2.5": 500,
    "PM10": 500,
    "NO": 200,
    "NO2": 200,
    "NOx": 300,
    "NH3": 200,
    "CO": 100,
    "SO2": 200,
    "O3": 200,
    "Benzene": 50,
    "Toluene": 50
}

print("\n" + "=" * 60)
print("SUSPICIOUS VALUES REMOVED")
print("=" * 60)

for column in pollutant_columns:

    invalid_count = (
        df[column] > upper_limits[column]
    ).sum()

    print(
        f"{column}: {invalid_count} values above "
        f"{upper_limits[column]}"
    )

    df.loc[
        df[column] > upper_limits[column],
        column
    ] = np.nan


# ============================================================
# 10. CITY-WISE TIME-BASED INTERPOLATION
# ============================================================

# Missing values are interpolated using nearby observations
# from the same city.

for column in pollutant_columns:

    df[column] = (
        df.groupby("City")[column]
        .transform(
            lambda x: x.interpolate(
                method="linear",
                limit_direction="both"
            )
        )
    )


# ============================================================
# 11. CITY-WISE MEDIAN IMPUTATION
# ============================================================

# If interpolation cannot fill a value, use the median
# of that pollutant within the same city.

for column in pollutant_columns:

    df[column] = (
        df.groupby("City")[column]
        .transform(
            lambda x: x.fillna(x.median())
        )
    )


# ============================================================
# 12. GLOBAL MEDIAN FOR REMAINING MISSING VALUES
# ============================================================

# If a city has no valid value for a particular pollutant,
# use the overall median.

for column in pollutant_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


# ============================================================
# 13. DROP AQI COLUMNS
# ============================================================

# AQI is calculated from pollutant concentrations.
# Including it could introduce target leakage into our
# PM2.5 forecasting model.

df = df.drop(
    columns=["AQI", "AQI_Bucket"]
)

print("\nDropped AQI and AQI_Bucket.")


# ============================================================
# 14. CHECK MISSING VALUES AFTER CLEANING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

missing_after = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean() * 100
})

print(
    missing_after.sort_values(
        by="Missing_Percentage",
        ascending=False
    )
)


# ============================================================
# 15. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES AFTER CLEANING")
print("=" * 60)

print(df.dtypes)


# ============================================================
# 16. CHECK MAXIMUM VALUES AFTER CLEANING
# ============================================================

print("\n" + "=" * 60)
print("MAXIMUM POLLUTANT VALUES AFTER CLEANING")
print("=" * 60)

print(
    df[pollutant_columns]
    .max()
    .sort_values(ascending=False)
)


# ============================================================
# 17. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 18. SAVE CLEANED DATA
# ============================================================

output_path = "data/cleaned_air_quality.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nCleaned dataset saved to:")
print(output_path)
