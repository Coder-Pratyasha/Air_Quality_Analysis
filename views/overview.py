import streamlit as st
import plotly.express as px


def show_overview(df):

    st.markdown(
        '<div class="section-title">Air Quality Overview</div>',
        unsafe_allow_html=True
    )

    avg_pm25 = df["PM2.5"].mean()
    max_pm25 = df["PM2.5"].max()
    city_count = df["City"].nunique()
    record_count = len(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average PM2.5",
            f"{avg_pm25:.2f} µg/m³"
        )

    with col2:
        st.metric(
            "Highest PM2.5",
            f"{max_pm25:.2f} µg/m³"
        )

    with col3:
        st.metric(
            "Cities Covered",
            city_count
        )

    with col4:
        st.metric(
            "Total Records",
            f"{record_count:,}"
        )

    st.markdown("---")

    # City ranking
    city_avg = (
        df.groupby("City")["PM2.5"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_city = px.bar(
        city_avg,
        x="PM2.5",
        y="City",
        orientation="h",
        title="Average PM2.5 by City",
        labels={
            "PM2.5": "Average PM2.5",
            "City": ""
        }
    )

    fig_city.update_layout(
        height=650,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_city,
        use_container_width=True
    )

    # Monthly trend
    monthly = (
        df.assign(
            Month=df["Datetime"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")["PM2.5"]
        .mean()
        .reset_index()
    )

    fig_month = px.line(
        monthly,
        x="Month",
        y="PM2.5",
        title="Monthly PM2.5 Trend",
        labels={
            "Month": "Month",
            "PM2.5": "Average PM2.5"
        }
    )

    fig_month.update_layout(
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )