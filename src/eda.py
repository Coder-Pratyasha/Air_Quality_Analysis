import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------

df = pd.read_csv("data/cleaned_air_quality.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values(["City", "Datetime"])

print("Dataset Shape:", df.shape)


# --------------------------------------------------
# 2. Basic Statistical Summary
# --------------------------------------------------

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())


# --------------------------------------------------
# 3. Average PM2.5 by City
# --------------------------------------------------

city_pm25 = (
    df.groupby("City")["PM2.5"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== AVERAGE PM2.5 BY CITY =====")
print(city_pm25)


plt.figure(figsize=(12, 6))

city_pm25.plot(kind="bar")

plt.title("Average PM2.5 by City")
plt.xlabel("City")
plt.ylabel("Average PM2.5")
plt.xticks(rotation=60)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. Monthly PM2.5 Trend
# --------------------------------------------------

monthly_pm25 = (
    df.set_index("Datetime")["PM2.5"]
    .resample("ME")
    .mean()
)

print("\n===== MONTHLY PM2.5 =====")
print(monthly_pm25.head(12))


plt.figure(figsize=(14, 6))

plt.plot(monthly_pm25.index, monthly_pm25.values)

plt.title("Monthly Average PM2.5 Over Time")
plt.xlabel("Date")
plt.ylabel("Average PM2.5")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Hour-of-Day Analysis
# --------------------------------------------------

df["Hour"] = df["Datetime"].dt.hour

hourly_pm25 = (
    df.groupby("Hour")["PM2.5"]
    .mean()
)

print("\n===== AVERAGE PM2.5 BY HOUR =====")
print(hourly_pm25)


plt.figure(figsize=(10, 5))

plt.plot(hourly_pm25.index, hourly_pm25.values, marker="o")

plt.title("Average PM2.5 by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average PM2.5")
plt.xticks(range(24))

plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Day-of-Week Analysis
# --------------------------------------------------

df["DayOfWeek"] = df["Datetime"].dt.day_name()

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_pm25 = (
    df.groupby("DayOfWeek")["PM2.5"]
    .mean()
    .reindex(day_order)
)

print("\n===== AVERAGE PM2.5 BY DAY =====")
print(daily_pm25)


plt.figure(figsize=(10, 5))

daily_pm25.plot(kind="bar")

plt.title("Average PM2.5 by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average PM2.5")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Monthly Seasonality
# --------------------------------------------------

df["Month"] = df["Datetime"].dt.month

monthly_seasonality = (
    df.groupby("Month")["PM2.5"]
    .mean()
)

print("\n===== AVERAGE PM2.5 BY MONTH =====")
print(monthly_seasonality)


plt.figure(figsize=(10, 5))

plt.plot(
    monthly_seasonality.index,
    monthly_seasonality.values,
    marker="o"
)

plt.title("Seasonal PM2.5 Pattern")
plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.xticks(range(1, 13))

plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Pollutant Correlation
# --------------------------------------------------

pollutants = [
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

correlation = df[pollutants].corr()

print("\n===== POLLUTANT CORRELATION WITH PM2.5 =====")

print(
    correlation["PM2.5"]
    .sort_values(ascending=False)
)


# --------------------------------------------------
# 9. Highest PM2.5 Observations
# --------------------------------------------------

top_pollution = (
    df[
        [
            "City",
            "Datetime",
            "PM2.5"
        ]
    ]
    .sort_values("PM2.5", ascending=False)
    .head(10)
)

print("\n===== TOP 10 PM2.5 OBSERVATIONS =====")
print(top_pollution)


print("\n===== EDA COMPLETED =====")