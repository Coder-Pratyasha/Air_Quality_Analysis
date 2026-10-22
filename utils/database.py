import os

import streamlit as st
from sqlalchemy import create_engine, URL
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


@st.cache_resource
def get_database_connection():

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        database=os.getenv("MYSQL_DATABASE")
    )

    engine = create_engine(connection_url)

    return engine