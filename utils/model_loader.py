import joblib
import streamlit as st


@st.cache_resource
def load_model():

    model = joblib.load(
        "models/pm25_model.pkl"
    )

    return model