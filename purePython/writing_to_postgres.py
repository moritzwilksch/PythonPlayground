#%%
import dotenv
from rich import print

dotenv.load_dotenv()
import os

import duckdb
import pandas as pd
import polars as pl
from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, create_engine

PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWD = os.getenv("POSTGRES_PASSWD")
PG_HOST = os.getenv("POSTGRES_HOST")
_engine = create_engine(f"postgresql://{PG_USER}:{PG_PASSWD}@{PG_HOST}:5432/playground")
_metadata = MetaData(_engine)


users = Table(
    "users",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("fullname", String),
)

_metadata.create_all()


with _engine.connect() as conn:
    conn.execute(
        users.insert(),
        [
            {"name": "a", "fullname": "fna"},
            {"name": "b", "fullname": "fnb"},
        ],
    )

print("done")


#%%
# df = pd.read_parquet("data/yellow_tripdata_2021-01.parquet")
df = pd.read_parquet("data/berlin.parquet")


#%%
for word in ["eigentum", "miet"]:
    s = duckdb.query(
        f"SELECT regexp_matches(lower(title), '{word}') AS contains_{word} FROM df"
    ).to_df()
    df = df.assign(**{f"contains_{word}": s})


#%%
duckdb.query(
    "SELECT zip_code, length(List(title)) FROM df GROUP BY zip_code USING SAMPLE 100%;"
).df()

#%%
conn = duckdb.connect()
rel = conn.df(df)

#%%
rel.aggregate("length(List(title))").project("*")

#%%

#%%
df = duckdb.query("SELECT * FROM read_csv_auto('data/WIKI-AAPL.csv')").df()

#%%
duckdb.query(
    """
    SELECT *,
        (
            CASE
            WHEN row_number() OVER my_window > 7 
            THEN AVG(Open) OVER my_window
            END
        ) as MA_7
    FROM df
    WINDOW my_window AS (ORDER BY Date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW);
    """
).df()

#%%
conn = duckdb.connect()
rel = conn.df(df)

#%%
duckdb.query(
    """ 
    WITH returns AS (
        SELECT *, (Open - lag(Open, 1) OVER empty) / lag(Open, 1) OVER empty as return
        FROM df
        WINDOW empty AS ()
    )
    SELECT EXTRACT (WEEKDAY from Date) as wd, AVG(return)
    FROM returns
    GROUP BY 1
    """
).df()

#%%
