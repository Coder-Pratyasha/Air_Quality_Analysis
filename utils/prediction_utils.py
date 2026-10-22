import pandas as pd


def get_city_timestamp_data(df, city, timestamp):
    """
    Get the historical pollutant values closest to the
    selected timestamp for the selected city.
    """

    city_df = df[
        df["City"] == city
    ].sort_values("Datetime").copy()

    if city_df.empty:
        raise ValueError(
            f"No historical data available for {city}."
        )

    # Find the observation closest to the selected timestamp
    city_df["time_difference"] = (
        city_df["Datetime"] - timestamp
    ).abs()

    closest_row = city_df.loc[
        city_df["time_difference"].idxmin()
    ]

    return closest_row


def get_historical_features(df, city, timestamp):
    """
    Calculate PM2.5 lag and rolling features using
    observations before the selected timestamp.
    """

    city_df = df[
        (df["City"] == city) &
        (df["Datetime"] < timestamp)
    ].sort_values("Datetime")

    if len(city_df) < 24:
        raise ValueError(
            "Not enough historical PM2.5 data available "
            "before the selected time."
        )

    pm25 = city_df["PM2.5"].tolist()

    return {
        "pm25_lag_1": pm25[-1],
        "pm25_lag_3": pm25[-3],
        "pm25_lag_6": pm25[-6],
        "pm25_lag_12": pm25[-12],
        "pm25_lag_24": pm25[-24],

        "pm25_rolling_6h":
            sum(pm25[-6:]) / 6,

        "pm25_rolling_24h":
            sum(pm25[-24:]) / 24
    }


def create_prediction_input(
    city,
    timestamp,
    pollutant_values,
    historical_features
):

    input_data = pd.DataFrame([{

        "City": city,

        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "month": timestamp.month,
        "day": timestamp.day,

        "is_weekend":
            int(timestamp.dayofweek >= 5),

        "PM10":
            pollutant_values["PM10"],

        "NO":
            pollutant_values["NO"],

        "NO2":
            pollutant_values["NO2"],

        "NOx":
            pollutant_values["NOx"],

        "NH3":
            pollutant_values["NH3"],

        "CO":
            pollutant_values["CO"],

        "SO2":
            pollutant_values["SO2"],

        "O3":
            pollutant_values["O3"],

        "Benzene":
            pollutant_values["Benzene"],

        "Toluene":
            pollutant_values["Toluene"],

        "pm25_lag_1":
            historical_features["pm25_lag_1"],

        "pm25_lag_3":
            historical_features["pm25_lag_3"],

        "pm25_lag_6":
            historical_features["pm25_lag_6"],

        "pm25_lag_12":
            historical_features["pm25_lag_12"],

        "pm25_lag_24":
            historical_features["pm25_lag_24"],

        "pm25_rolling_6h":
            historical_features["pm25_rolling_6h"],

        "pm25_rolling_24h":
            historical_features["pm25_rolling_24h"]
    }])

    return input_data


def make_prediction(model, input_data):

    prediction = model.predict(
        input_data
    )[0]

    return max(0, prediction)