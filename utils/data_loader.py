import pandas as pd
import streamlit as st


@st.cache_data
def load_data():

    df = pd.read_csv("data/cleaned_air_quality.csv")

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df = df.sort_values(
        ["City", "Datetime"]
    ).reset_index(drop=True)

    return df