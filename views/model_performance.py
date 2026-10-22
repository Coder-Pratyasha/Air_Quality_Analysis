import streamlit as st
import pandas as pd


def show_model_performance():

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    results = pd.read_csv(
        "data/model_results.csv"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    best_model = results.sort_values(
        "RMSE"
    ).iloc[0]

    st.success(
        f"Best model: {best_model.iloc[0]} "
        f"with RMSE = {best_model['RMSE']:.2f}"
    )

    st.markdown(
        '<div class="section-title">'
        'Actual vs Predicted PM2.5'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        "data/actual_vs_predicted.png",
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Prediction Error'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        "data/prediction_error.png",
        use_container_width=True
    )