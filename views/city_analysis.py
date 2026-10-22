import streamlit as st
import plotly.express as px


def show_city_analysis(df):

    st.markdown(
        '<div class="section-title">City Air Quality Analysis</div>',
        unsafe_allow_html=True
    )

    city = st.selectbox(
        "Select a city",
        sorted(df["City"].unique())
    )

    city_df = df[
        df["City"] == city
    ].copy()

    # Metrics
    avg_pm25 = city_df["PM2.5"].mean()
    max_pm25 = city_df["PM2.5"].max()
    min_pm25 = city_df["PM2.5"].min()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average PM2.5",
            f"{avg_pm25:.2f} µg/m³"
        )

    with col2:
        st.metric(
            "Maximum PM2.5",
            f"{max_pm25:.2f} µg/m³"
        )

    with col3:
        st.metric(
            "Minimum PM2.5",
            f"{min_pm25:.2f} µg/m³"
        )

    st.markdown("---")

    # Monthly trend
    city_monthly = (
        city_df.assign(
            Month=city_df["Datetime"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")["PM2.5"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        city_monthly,
        x="Month",
        y="PM2.5",
        title=f"{city} — Monthly PM2.5 Trend",
        labels={
            "Month": "Month",
            "PM2.5": "Average PM2.5"
        }
    )

    fig.update_layout(
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Hourly pattern
    city_hourly = (
        city_df.assign(
            Hour=city_df["Datetime"].dt.hour
        )
        .groupby("Hour")["PM2.5"]
        .mean()
        .reset_index()
    )

    fig_hour = px.line(
        city_hourly,
        x="Hour",
        y="PM2.5",
        markers=True,
        title=f"{city} — Average PM2.5 by Hour",
        labels={
            "Hour": "Hour of Day",
            "PM2.5": "Average PM2.5"
        }
    )

    fig_hour.update_layout(
        height=400,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    # Pollutant statistics
    st.markdown(
        '<div class="section-title">Pollutant Statistics</div>',
        unsafe_allow_html=True
    )

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

    pollutant_avg = (
        city_df[pollutants]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    pollutant_avg.columns = [
        "Pollutant",
        "Average"
    ]

    st.dataframe(
        pollutant_avg,
        use_container_width=True,
        hide_index=True
    )