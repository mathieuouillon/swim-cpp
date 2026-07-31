from sqlalchemy import create_engine, text
import pandas as pd
import sqlite3

mysql_engine = create_engine(
    "mysql+mysqlconnector://rcdb@clasdb.jlab.org/rcdb"
)

sqlite_conn = sqlite3.connect("rcdb.sqlite")

with mysql_engine.connect() as conn:
    tables = conn.execute(text("SHOW TABLES")).fetchall()

for (table,) in tables:
    print(f"Copying {table}")
    df = pd.read_sql(f"SELECT * FROM `{table}`", mysql_engine)
    df.to_sql(table, sqlite_conn, if_exists="replace", index=False)

sqlite_conn.close()