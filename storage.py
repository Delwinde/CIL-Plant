"""SQLite persistence for all plant data.

Tables
    daily       one row per day (key: date)                    - production, grades, reagents
    tanks       one row per day per CIL tank (key: date,tank)  - leach and carbon readings
    carbon_log  carbon events: elution, regeneration, addition, activity tests
    qaqc        lab QA/QC samples: standards, blanks, duplicates, check assays
    pours       doré bars poured
    settings    key/value JSON (costs, limits, targets, tank volumes)

The database file lives next to the app unless PLANT_DB_PATH is set. Point that variable
at a persistent disk when hosting, because many cloud hosts wipe local files on restart.
Older databases are upgraded automatically: missing columns are added on first use.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from calc import INPUT_COLUMNS, TEXT_INPUTS

DB_PATH = Path(os.environ.get("PLANT_DB_PATH", Path(__file__).with_name("plant.db")))

SCHEMAS: dict[str, dict] = {
    "daily": {
        "key": ["date"],
        "cols": [(c, "TEXT" if c in ("date", *TEXT_INPUTS) else "REAL") for c in INPUT_COLUMNS],
    },
    "tanks": {
        "key": ["date", "tank"],
        "cols": [("date", "TEXT"), ("tank", "INTEGER"), ("volume_m3", "REAL"), ("pct_solids", "REAL"),
                 ("free_cn_ppm", "REAL"), ("ph", "REAL"), ("do_ppm", "REAL"),
                 ("carbon_conc_gpl", "REAL"), ("carbon_loading_gpt", "REAL"),
                 ("solution_au_mgl", "REAL")],
    },
    "carbon_log": {
        "id": True,
        "cols": [("date", "TEXT"), ("event", "TEXT"), ("kg", "REAL"), ("activity_pct", "REAL"),
                 ("note", "TEXT")],
    },
    "qaqc": {
        "id": True,
        "cols": [("date", "TEXT"), ("sample_id", "TEXT"), ("sample_type", "TEXT"), ("lab", "TEXT"),
                 ("assay_gpt", "REAL"), ("expected_gpt", "REAL"), ("sd_gpt", "REAL"),
                 ("original_gpt", "REAL"), ("note", "TEXT")],
    },
    "pours": {
        "id": True,
        "cols": [("date", "TEXT"), ("bar_no", "TEXT"), ("gross_g", "REAL"), ("au_pct", "REAL"),
                 ("ag_pct", "REAL"), ("slag_au_g", "REAL")],
    },
}
DATA_TABLES = list(SCHEMAS)

DEFAULT_SETTINGS = {
    "plant_name": "Gold CIL Plant",
    "costs": {"nacn_per_kg": 3.2, "lime_per_kg": 0.25, "power_per_kwh": 0.12},
    "thresholds": {"recovery_drop_pts": 2.0, "unaccounted_warn_pct": 3.0, "unaccounted_crit_pct": 5.0,
                   "head_gap_pct": 5.0, "reagent_spike_factor": 1.25, "min_availability_pct": 85.0,
                   "solution_loss_pct": 25.0},
    "circuit": {"min_ph": 10.0, "min_do_ppm": 6.0, "min_free_cn_ppm": 100.0,
                "carbon_conc_min_gpl": 8.0, "carbon_conc_max_gpl": 25.0, "min_activity_pct": 70.0,
                "solids_sg": 2.7, "default_pct_solids": 40.0, "tank_volumes_m3": [800.0] * 6},
    "qaqc": {"z_warn": 2.0, "z_fail": 3.0, "blank_limit_gpt": 0.05, "dup_rd_pct": 10.0},
    "targets": {"recovery_pct": 92.0, "head_gpt": 2.5, "tails_gpt": 0.2, "throughput_tph": 165.0,
                "availability_pct": 90.0, "nacn_kgpt": 0.55, "power_kwhpt": 22.0, "unaccounted_pct": 1.0},
    "spc": {"baseline_days": 30},
}


def _conn(path=None):
    return sqlite3.connect(path or DB_PATH)


def _py(v):
    """Make a value safe for sqlite3 (numpy scalars -> python, NaN/NaT -> None)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def init(path=None):
    """Create tables and add any columns missing from an older database."""
    with _conn(path) as c:
        for name, spec in SCHEMAS.items():
            cols = ", ".join(f"{n} {t}" for n, t in spec["cols"])
            if spec.get("id"):
                c.execute(f"CREATE TABLE IF NOT EXISTS {name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
            else:
                pk = ", ".join(spec["key"])
                c.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols}, PRIMARY KEY ({pk}))")
            have = {r[1] for r in c.execute(f"PRAGMA table_info({name})")}
            for n, t in spec["cols"]:
                if n not in have:
                    c.execute(f"ALTER TABLE {name} ADD COLUMN {n} {t}")
        c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")


def _names(table):
    return [n for n, _ in SCHEMAS[table]["cols"]]


def _clean(table: str, df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in _names(table):
        if col not in d.columns:
            d[col] = None
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d = d.dropna(subset=["date"])
    return d


# ---------------------------------------------------------------- generic table API
def load_table(table: str, path=None) -> pd.DataFrame:
    init(path)
    cols = (["id"] if SCHEMAS[table].get("id") else []) + _names(table)
    order = "date, tank" if table == "tanks" else "date" + (", id" if SCHEMAS[table].get("id") else "")
    with _conn(path) as c:
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM {table} ORDER BY {order}", c)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_rows(table: str, df: pd.DataFrame, path=None) -> int:
    """Keyed tables: insert or replace. Id tables: append (new ids)."""
    init(path)
    spec = SCHEMAS[table]
    d = _clean(table, df)
    if spec.get("key"):
        d = d.drop_duplicates(spec["key"], keep="last")
    names = _names(table)
    rows = [tuple(_py(v) for v in r) for r in d[names].itertuples(index=False)]
    verb = "INSERT OR REPLACE" if spec.get("key") else "INSERT"
    with _conn(path) as c:
        c.executemany(f"{verb} INTO {table} ({','.join(names)}) VALUES ({','.join('?' * len(names))})", rows)
    return len(rows)


def replace_day(table: str, date_str: str, df: pd.DataFrame, path=None) -> int:
    """Delete every row of a keyed table for one date, then insert df (used for tank readings)."""
    init(path)
    with _conn(path) as c:
        c.execute(f"DELETE FROM {table} WHERE date = ?", (date_str,))
    return save_rows(table, df, path) if len(df) else 0


def delete_ids(table: str, ids, path=None):
    init(path)
    with _conn(path) as c:
        c.executemany(f"DELETE FROM {table} WHERE id = ?", [(int(i),) for i in ids])


def clear_table(table: str, path=None):
    init(path)
    with _conn(path) as c:
        c.execute(f"DELETE FROM {table}")


def clear_all(path=None):
    for t in DATA_TABLES:
        clear_table(t, path)


# ---------------------------------------------------------------- daily table (original API)
def load(path=None) -> pd.DataFrame:
    return load_table("daily", path)


def upsert(df: pd.DataFrame, path=None) -> int:
    return save_rows("daily", df, path)


def delete(date_str: str, path=None):
    init(path)
    with _conn(path) as c:
        c.execute("DELETE FROM daily WHERE date = ?", (date_str,))
        c.execute("DELETE FROM tanks WHERE date = ?", (date_str,))


def clear(path=None):
    clear_table("daily", path)


# ---------------------------------------------------------------- settings
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def get_settings(path=None) -> dict:
    """Saved settings laid over the defaults, so new settings appear after an upgrade."""
    init(path)
    with _conn(path) as c:
        rows = c.execute("SELECT key, value FROM settings").fetchall()
    saved = {k: json.loads(v) for k, v in rows}
    return _merge(DEFAULT_SETTINGS, saved)


def save_settings(settings: dict, path=None):
    init(path)
    with _conn(path) as c:
        c.executemany("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                      [(k, json.dumps(v)) for k, v in settings.items()])


# ---------------------------------------------------------------- backup / restore
def backup_zip(path=None) -> bytes:
    """Every data table as a CSV, plus settings.json, in one zip file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for t in DATA_TABLES:
            df = load_table(t, path)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            z.writestr(f"{t}.csv", df.to_csv(index=False))
        z.writestr("settings.json", json.dumps(get_settings(path), indent=2))
    return buf.getvalue()


def restore_zip(data: bytes, path=None, replace: bool = True) -> dict:
    """Restore from backup_zip output. Returns {table: rows_restored}."""
    init(path)
    out = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        for t in DATA_TABLES:
            if f"{t}.csv" not in names:
                continue
            df = pd.read_csv(z.open(f"{t}.csv"))
            if replace:
                clear_table(t, path)
            if "id" in df.columns:
                df = df.drop(columns="id")
            out[t] = save_rows(t, df, path) if len(df) else 0
        if "settings.json" in names:
            save_settings(json.loads(z.read("settings.json")), path)
            out["settings"] = 1
    return out
