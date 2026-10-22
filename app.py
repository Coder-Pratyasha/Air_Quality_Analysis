import streamlit as st

from utils.data_loader import load_data
from utils.model_loader import load_model

from views.overview import show_overview
from views.city_analysis import show_city_analysis
from views.prediction import show_prediction
from views.model_performance import show_model_performance
from views.sql_insights import show_sql_insights


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Air Quality Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #12343b;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        color: #12343b;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .prediction-card {
        background: linear-gradient(
            135deg,
            #12343b,
            #1f6f78
        );

        padding: 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 750;
    }

    .prediction-label {
        font-size: 16px;
        opacity: 0.9;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD RESOURCES
# ============================================================

df = load_data()
model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h2 style="color:#12343b;">
        🌫️ Air Quality
    </h2>

    <p style="color:#64748b;">
        Indian Air Quality Intelligence
    </p>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔎 City Analysis",
        "🗄️ SQL Insights",
        "🤖 PM2.5 Prediction",
        "📈 Model Performance"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Indian Air Quality Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore pollution patterns across Indian cities '
    'and predict PM2.5 levels.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "🏠 Overview":

    show_overview(df)

elif page == "🔎 City Analysis":

    show_city_analysis(df)

elif page == "🗄️ SQL Insights":

    show_sql_insights()
    
elif page == "🤖 PM2.5 Prediction":

    show_prediction(
        df,
        model
    )

elif page == "📈 Model Performance":

    show_model_performance()