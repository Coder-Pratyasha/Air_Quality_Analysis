from utils.database import get_database_connection
from sqlalchemy import text

try:
    engine = get_database_connection()

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT COUNT(*) FROM air_quality")
        )

        count = result.scalar()

    print("Database connection successful!")
    print("Rows in air_quality:", count)

except Exception as e:
    print("Database connection failed!")
    print(e)