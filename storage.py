"""SQLite persistence. One row per day, keyed on date (re-saving a day overwrites it)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from calc import INPUT_COLUMNS

DB_PATH = Path(__file__).with_name("plant.db")


def _conn(path=DB_PATH):
    return sqlite3.connect(path)


def init(path=DB_PATH):
    with _conn(path) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS daily (
                date TEXT PRIMARY KEY,
                tonnes_milled REAL, mill_hours REAL, scheduled_hours REAL,
                head_grade_gpt REAL, tails_grade_gpt REAL,
                gold_poured_g REAL, inventory_change_g REAL,
                nacn_kg REAL, lime_kg REAL, power_kwh REAL,
                model_grade_gpt REAL)"""
        )


def load(path=DB_PATH) -> pd.DataFrame:
    init(path)
    with _conn(path) as c:
        df = pd.read_sql_query("SELECT * FROM daily ORDER BY date", c)
    df["date"] = pd.to_datetime(df["date"])
    return df


def upsert(df: pd.DataFrame, path=DB_PATH) -> int:
    """Insert or replace rows. Returns number of rows written."""
    init(path)
    d = df.copy()
    for col in INPUT_COLUMNS:
        if col not in d.columns:
            d[col] = None
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d = d.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in d[INPUT_COLUMNS].itertuples(index=False)]
    with _conn(path) as c:
        c.executemany(
            f"INSERT OR REPLACE INTO daily ({','.join(INPUT_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(INPUT_COLUMNS))})",
            rows,
        )
    return len(rows)


def delete(date_str: str, path=DB_PATH):
    init(path)
    with _conn(path) as c:
        c.execute("DELETE FROM daily WHERE date = ?", (date_str,))


def clear(path=DB_PATH):
    init(path)
    with _conn(path) as c:
        c.execute("DELETE FROM daily")
