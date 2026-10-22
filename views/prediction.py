
import pandas as pd
import streamlit as st

from utils.prediction_utils import (
    get_city_timestamp_data,
    get_historical_features,
    create_prediction_input,
    make_prediction
)


def show_prediction(df, model):

    st.markdown(
        '<div class="section-title">PM2.5 Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Predict the next-hour PM2.5 level using historical "
        "air-quality data or your own pollutant measurements."
    )

    # ========================================================
    # CITY
    # ========================================================

    city = st.selectbox(
        "Select City",
        sorted(df["City"].unique())
    )

    city_df = df[
        df["City"] == city
    ].sort_values("Datetime")

    # ========================================================
    # DATE & TIME
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        selected_date = st.date_input(
            "Date",
            value=city_df["Datetime"].max().date(),
            min_value=city_df["Datetime"].min().date(),
            max_value=city_df["Datetime"].max().date()
        )

    with col2:

        selected_time = st.time_input(
            "Time",
            value=city_df["Datetime"].max().time()
        )

    selected_timestamp = (
        pd.Timestamp(selected_date)
        + pd.Timedelta(
            hours=selected_time.hour,
            minutes=selected_time.minute
        )
    )

    st.markdown("---")

    # ========================================================
    # PREDICTION MODE
    # ========================================================

    st.markdown("### Prediction Input")

    prediction_mode = st.radio(
        "How should pollutant values be obtained?",
        [
            "Use historical data automatically",
            "Enter pollutant values manually"
        ]
    )

    # ========================================================
    # HISTORICAL DATA MODE
    # ========================================================

    if prediction_mode == "Use historical data automatically":

        try:

            historical_row = get_city_timestamp_data(
                df,
                city,
                selected_timestamp
            )

            historical_features = get_historical_features(
                df,
                city,
                selected_timestamp
            )

            st.info(
                "Pollutant values will be taken automatically "
                "from the historical observation closest to "
                "the selected date and time."
            )

            st.markdown("### Historical Pollutant Values")

            pollutants = [
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

            col1, col2, col3 = st.columns(3)

            for i, pollutant in enumerate(pollutants):

                column = [
                    col1,
                    col2,
                    col3
                ][i % 3]

                with column:

                    st.metric(
                        pollutant,
                        f"{historical_row[pollutant]:.2f}"
                    )

            pollutant_values = {
                pollutant:
                    historical_row[pollutant]
                for pollutant in pollutants
            }

        except ValueError as e:

            st.warning(str(e))
            return

    # ========================================================
    # MANUAL DATA MODE
    # ========================================================

    else:

        st.info(
            "Enter the pollutant measurements for the "
            "selected date and time."
        )

        st.markdown("### Pollutant Values")

        col1, col2, col3 = st.columns(3)

        with col1:

            pm10 = st.number_input(
                "PM10",
                min_value=0.0,
                value=100.0,
                step=0.1
            )

            no = st.number_input(
                "NO",
                min_value=0.0,
                value=20.0,
                step=0.1
            )

            no2 = st.number_input(
                "NO₂",
                min_value=0.0,
                value=40.0,
                step=0.1
            )

            nox = st.number_input(
                "NOx",
                min_value=0.0,
                value=50.0,
                step=0.1
            )

        with col2:

            nh3 = st.number_input(
                "NH₃",
                min_value=0.0,
                value=20.0,
                step=0.1
            )

            co = st.number_input(
                "CO",
                min_value=0.0,
                value=1.0,
                step=0.1
            )

            so2 = st.number_input(
                "SO₂",
                min_value=0.0,
                value=20.0,
                step=0.1
            )

            o3 = st.number_input(
                "O₃",
                min_value=0.0,
                value=30.0,
                step=0.1
            )

        with col3:

            benzene = st.number_input(
                "Benzene",
                min_value=0.0,
                value=2.0,
                step=0.1
            )

            toluene = st.number_input(
                "Toluene",
                min_value=0.0,
                value=5.0,
                step=0.1
            )

        pollutant_values = {
            "PM10": pm10,
            "NO": no,
            "NO2": no2,
            "NOx": nox,
            "NH3": nh3,
            "CO": co,
            "SO2": so2,
            "O3": o3,
            "Benzene": benzene,
            "Toluene": toluene
        }

        try:

            historical_features = get_historical_features(
                df,
                city,
                selected_timestamp
            )

        except ValueError as e:

            st.warning(str(e))
            return

    # ========================================================
    # HISTORICAL FEATURES
    # ========================================================

    with st.expander(
        "View PM2.5 historical features used by the model"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"Previous hour: "
                f"**{historical_features['pm25_lag_1']:.2f}**"
            )

            st.write(
                f"3 hours ago: "
                f"**{historical_features['pm25_lag_3']:.2f}**"
            )

            st.write(
                f"6 hours ago: "
                f"**{historical_features['pm25_lag_6']:.2f}**"
            )

            st.write(
                f"12 hours ago: "
                f"**{historical_features['pm25_lag_12']:.2f}**"
            )

        with col2:

            st.write(
                f"24 hours ago: "
                f"**{historical_features['pm25_lag_24']:.2f}**"
            )

            st.write(
                f"6-hour average: "
                f"**{historical_features['pm25_rolling_6h']:.2f}**"
            )

            st.write(
                f"24-hour average: "
                f"**{historical_features['pm25_rolling_24h']:.2f}**"
            )

    # ========================================================
    # PREDICT
    # ========================================================

    if st.button(
        "🔮 Predict Next-Hour PM2.5",
        type="primary",
        use_container_width=True
    ):

        try:

            input_data = create_prediction_input(
                city=city,
                timestamp=selected_timestamp,
                pollutant_values=pollutant_values,
                historical_features=historical_features
            )

            prediction = make_prediction(
                model,
                input_data
            )

            # =================================================
            # PREDICTED TIMESTAMP
            # =================================================

            predicted_timestamp = (
                selected_timestamp
                + pd.Timedelta(hours=1)
            )

            # =================================================
            # RESULT
            # =================================================

            st.success(
                "Prediction completed successfully!"
            )

            st.markdown(
                "### 🌫️ Predicted Next-Hour PM2.5"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted PM2.5",
                    f"{prediction:.2f} µg/m³"
                )

            with col2:

                st.metric(
                    "Predicted For",
                    predicted_timestamp.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                )

            st.info(
                f"📍 {city} • Prediction for the hour "
                f"following the selected observation."
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )
