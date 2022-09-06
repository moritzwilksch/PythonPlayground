import dotenv
from rich import print

dotenv.load_dotenv()
import os

from sqlalchemy import (Column, ForeignKey, Integer, MetaData, String, Table,
                        create_engine)

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
