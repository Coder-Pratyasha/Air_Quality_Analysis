import pandas as pd

# Load dataset
df = pd.read_csv("data/city_hour.csv")

# -------------------------------
# 1. Basic Information
# -------------------------------

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# -------------------------------
# 2. Sample Records
# -------------------------------

print("\n" + "=" * 60)
print("SAMPLE RECORDS")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# -------------------------------
# 3. Duplicate Analysis
# -------------------------------

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

print("Duplicate rows:", df.duplicated().sum())


# -------------------------------
# 4. Missing Value Analysis
# -------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean() * 100
})

missing = missing.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print(missing)


# -------------------------------
# 5. Date-Time Analysis
# -------------------------------

print("\n" + "=" * 60)
print("DATE-TIME ANALYSIS")
print("=" * 60)

df["Datetime"] = pd.to_datetime(df["Datetime"])

print("Start date:", df["Datetime"].min())
print("End date:", df["Datetime"].max())


# -------------------------------
# 6. City Analysis
# -------------------------------

print("\n" + "=" * 60)
print("CITY ANALYSIS")
print("=" * 60)

print("Number of cities:", df["City"].nunique())

print("\nCities:")
print(sorted(df["City"].unique()))


# -------------------------------
# 7. Statistical Summary
# -------------------------------

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe().T)