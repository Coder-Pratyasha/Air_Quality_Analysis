import os

import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# 1. Load cleaned dataset
file_path = "data/cleaned_air_quality.csv"
df = pd.read_csv(file_path)

print("CSV loaded:", df.shape)


# 2. Convert datetime
df["Datetime"] = pd.to_datetime(df["Datetime"])


# 3. Rename columns to match MySQL table
df = df.rename(columns={
    "City": "city",
    "Datetime": "recorded_at",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO": "no_val",
    "NO2": "no2",
    "NOx": "nox",
    "NH3": "nh3",
    "CO": "co",
    "SO2": "so2",
    "O3": "o3",
    "Benzene": "benzene",
    "Toluene": "toluene"
})


# 4. Get MySQL credentials from .env
username = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT")
database = os.getenv("MYSQL_DATABASE")


# 5. Create MySQL connection
connection_url = (
    f"mysql+pymysql://{username}:"
    f"{quote_plus(password)}@"
    f"{host}:{port}/{database}"
)

engine = create_engine(connection_url)


# 6. Load data into MySQL in chunks
print("Loading data into MySQL...")

df.to_sql(
    "air_quality",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=5000
)

print("Data successfully loaded into MySQL!")