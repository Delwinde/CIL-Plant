"""Data validation for plant records.

Errors   = physically impossible or unusable; excluded from KPIs by default.
Warnings = suspicious but possible; kept in KPIs, shown to the user.

Every validate_* function returns a DataFrame with columns
row, date, field, severity, message (row = index in the frame that was checked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calc import REQUIRED_COLUMNS, resolve_tails
from circuit import CARBON_EVENTS
from qaqc import SAMPLE_TYPES

ISSUE_COLUMNS = ["row", "date", "field", "severity", "message"]

# Plausible operating windows for a CIL plant. Tune per site.
LIMITS = {
    "tonnes_milled": (0, 40000),
    "mill_hours": (0, 24),
    "head_grade_gpt": (0.05, 60),
    "tails_grade_gpt": (0.0, 5),
    "tph": (20, 1500),
    "nacn_kgpt": (0.1, 3.0),
    "lime_kgpt": (0.2, 8.0),
    "power_kwhpt": (5, 60),
    "tails_solution_mgl": (0, 2.0),
    "tails_pct_solids": (15, 60),
}
POUR_TOLERANCE = 1.02  # recovered gold may not exceed feed gold by more than 2%

TANK_LIMITS = {
    "pct_solids": (10, 60),
    "free_cn_ppm": (0, 1500),
    "ph": (7, 13),
    "do_ppm": (0, 30),
    "carbon_conc_gpl": (0, 60),
    "carbon_loading_gpt": (0, 20000),
    "solution_au_mgl": (0, 100),
}


def _issues(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate daily production records."""
    issues: list[dict] = []

    def add(i, date, field, sev, msg):
        issues.append({"row": i, "date": date, "field": field, "severity": sev, "message": msg})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if "tails_grade_gpt" not in df.columns and "tails_solid_gpt" not in df.columns:
        missing.append("tails_grade_gpt")
    if missing:
        add(None, None, ",".join(missing), "error", f"Missing required column(s): {', '.join(missing)}")
        return _issues(issues)

    d = resolve_tails(df)
    d["_date"] = pd.to_datetime(d["date"], errors="coerce")
    nums = [c for c in REQUIRED_COLUMNS if c != "date"] + ["tails_grade_gpt", "inventory_change_g"]
    optional = ["tails_solid_gpt", "tails_solution_mgl", "tails_pct_solids", "other_inventory_g"]
    for c in nums + optional:
        if c not in d.columns:
            d[c] = np.nan
        d[f"_{c}"] = pd.to_numeric(d[c], errors="coerce")

    # --- dates ---
    dup = d["_date"].duplicated(keep=False) & d["_date"].notna()
    for i in d.index[dup]:
        add(i, d.at[i, "_date"], "date", "error", "Duplicate date; keep one record per day.")

    valid_dates = d["_date"].dropna().sort_values()
    if len(valid_dates) > 1:
        full = pd.date_range(valid_dates.iloc[0], valid_dates.iloc[-1], freq="D")
        for m in full.difference(valid_dates):
            add(None, m, "date", "warning", "No record for this day (gap in data).")

    for i, r in d.iterrows():
        dt = r["_date"]
        if pd.isna(dt):
            add(i, None, "date", "error", f"Unreadable date: {r['date']!r}")
            continue
        if dt > pd.Timestamp.today().normalize() + pd.Timedelta(days=1):
            add(i, dt, "date", "error", "Date is in the future.")

        # --- missing / non-numeric / negative ---
        bad_row = False
        for c in nums:
            v = r[f"_{c}"]
            if pd.isna(v):
                if c == "inventory_change_g":
                    continue  # treated as 0 (or calculated from tank readings)
                add(i, dt, c, "error", f"{c} is missing or not a number.")
                bad_row = True
            elif v < 0 and c != "inventory_change_g":
                add(i, dt, c, "error", f"{c} is negative ({v}).")
                bad_row = True
        for c in optional:
            v = r[f"_{c}"]
            if pd.notna(v) and v < 0 and c != "other_inventory_g":
                add(i, dt, c, "error", f"{c} is negative ({v}).")
                bad_row = True
            elif pd.notna(v) and c == "other_inventory_g" and v < 0:
                add(i, dt, c, "error", f"{c} is negative ({v}).")
                bad_row = True
        if bad_row:
            continue

        t, h, tl = r["_tonnes_milled"], r["_head_grade_gpt"], r["_tails_grade_gpt"]
        hrs, sched = r["_mill_hours"], r["_scheduled_hours"]

        # --- hard physical checks ---
        if hrs > 24:
            add(i, dt, "mill_hours", "error", f"Mill hours {hrs} exceeds 24 h in a day.")
        if sched > 24:
            add(i, dt, "scheduled_hours", "error", f"Scheduled hours {sched} exceeds 24 h.")
        if hrs > sched + 1e-9:
            add(i, dt, "mill_hours", "error", f"Mill hours {hrs} exceed scheduled hours {sched}.")
        if t > 0 and hrs == 0:
            add(i, dt, "mill_hours", "error", "Tonnes milled but zero mill hours.")
        if tl > h:
            add(i, dt, "tails_grade_gpt", "error",
                f"Tails grade {tl:.3g} g/t is higher than head grade {h} g/t (negative recovery). "
                "Likely a typo or swapped values.")
        feed = t * h
        inv = r["_inventory_change_g"]
        recovered = r["_gold_poured_g"] + (0 if pd.isna(inv) else inv)
        if feed > 0 and recovered > feed * POUR_TOLERANCE:
            add(i, dt, "gold_poured_g", "error",
                f"Gold recovered ({recovered:.0f} g) exceeds feed gold ({feed:.0f} g). "
                "Check pour weight units or inventory sign.")

        # --- tails detail consistency ---
        if pd.notna(r["_tails_solution_mgl"]) and pd.isna(r["_tails_pct_solids"]):
            add(i, dt, "tails_pct_solids", "warning",
                "Solution tails given without % solids, so the dissolved gold loss is not included in the total tails.")
        if pd.notna(r["_tails_solution_mgl"]) and pd.isna(r["_tails_solid_gpt"]):
            add(i, dt, "tails_solid_gpt", "warning",
                "Solution tails given without solid tails; the typed total tails grade is being used.")

        # --- soft range checks ---
        for c, (lo, hi) in LIMITS.items():
            if c == "tph":
                v = t / hrs if hrs > 0 else np.nan
            elif c in ("nacn_kgpt", "lime_kgpt", "power_kwhpt"):
                src = {"nacn_kgpt": "nacn_kg", "lime_kgpt": "lime_kg", "power_kwhpt": "power_kwh"}[c]
                v = r[f"_{src}"] / t if t > 0 else np.nan
            else:
                v = r[f"_{c}"]
            if pd.notna(v) and not (lo <= v <= hi) and not (c == "tonnes_milled" and v == 0):
                add(i, dt, c, "warning", f"{c} = {v:.3g} is outside the usual range {lo}–{hi}.")

        if t == 0 and hrs == 0:
            add(i, dt, "tonnes_milled", "warning", "Plant did not run this day (zero tonnes).")

    return _issues(issues)


def error_rows(issues: pd.DataFrame) -> set:
    """Row indices that carry at least one error."""
    if issues.empty:
        return set()
    e = issues[(issues["severity"] == "error") & issues["row"].notna()]
    return set(e["row"].astype(int))


def validate_tanks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate tank readings (one row per date and tank)."""
    rows = []

    def add(i, date, field, sev, msg):
        rows.append({"row": i, "date": date, "field": field, "severity": sev, "message": msg})

    if df.empty:
        return _issues(rows)
    for i, r in df.iterrows():
        dt = pd.to_datetime(r.get("date"), errors="coerce")
        tank = r.get("tank")
        tag = f"Tank {int(tank)}" if pd.notna(tank) else "Tank ?"
        if pd.isna(tank) or int(tank) < 1:
            add(i, dt, "tank", "error", "Tank number must be 1 or higher.")
        vol = pd.to_numeric(r.get("volume_m3"), errors="coerce")
        if pd.notna(vol) and vol <= 0:
            add(i, dt, "volume_m3", "error", f"{tag}: volume must be above zero.")
        for c, (lo, hi) in TANK_LIMITS.items():
            v = pd.to_numeric(r.get(c), errors="coerce")
            if pd.isna(v):
                continue
            if v < 0:
                add(i, dt, c, "error", f"{tag}: {c} is negative ({v}).")
            elif c == "ph" and v > 14:
                add(i, dt, c, "error", f"{tag}: pH {v} is impossible (above 14).")
            elif not (lo <= v <= hi):
                add(i, dt, c, "warning", f"{tag}: {c} = {v:.3g} is outside the usual range {lo}–{hi}.")
    return _issues(rows)


def validate_pours(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(i, date, field, sev, msg):
        rows.append({"row": i, "date": date, "field": field, "severity": sev, "message": msg})

    for i, r in df.iterrows():
        dt = pd.to_datetime(r.get("date"), errors="coerce")
        g = pd.to_numeric(r.get("gross_g"), errors="coerce")
        au = pd.to_numeric(r.get("au_pct"), errors="coerce")
        ag = pd.to_numeric(r.get("ag_pct"), errors="coerce")
        slag = pd.to_numeric(r.get("slag_au_g"), errors="coerce")
        if pd.isna(g) or g <= 0:
            add(i, dt, "gross_g", "error", "Gross bar weight must be above zero.")
        if pd.isna(au) or not (0 < au <= 100):
            add(i, dt, "au_pct", "error", "Gold fineness must be between 0 and 100 %.")
        elif au < 50:
            add(i, dt, "au_pct", "warning", f"Gold fineness {au:.1f}% is low for a doré bar. Check the assay.")
        if pd.notna(ag) and (ag < 0 or ag > 100):
            add(i, dt, "ag_pct", "error", "Silver fineness must be between 0 and 100 %.")
        if pd.notna(au) and pd.notna(ag) and au + ag > 100.5:
            add(i, dt, "ag_pct", "error", f"Gold + silver fineness is {au + ag:.1f}%, above 100%.")
        if pd.notna(slag) and slag < 0:
            add(i, dt, "slag_au_g", "error", "Slag gold cannot be negative.")
    return _issues(rows)


def validate_qaqc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(i, date, field, sev, msg):
        rows.append({"row": i, "date": date, "field": field, "severity": sev, "message": msg})

    for i, r in df.iterrows():
        dt = pd.to_datetime(r.get("date"), errors="coerce")
        a = pd.to_numeric(r.get("assay_gpt"), errors="coerce")
        if r.get("sample_type") not in SAMPLE_TYPES:
            add(i, dt, "sample_type", "error", f"Sample type must be one of {', '.join(SAMPLE_TYPES)}.")
        if pd.isna(a) or a < 0:
            add(i, dt, "assay_gpt", "error", "Assay result is missing or negative.")
    return _issues(rows)


def validate_carbon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(i, date, field, sev, msg):
        rows.append({"row": i, "date": date, "field": field, "severity": sev, "message": msg})

    for i, r in df.iterrows():
        dt = pd.to_datetime(r.get("date"), errors="coerce")
        kg = pd.to_numeric(r.get("kg"), errors="coerce")
        act = pd.to_numeric(r.get("activity_pct"), errors="coerce")
        if r.get("event") not in CARBON_EVENTS:
            add(i, dt, "event", "error", f"Event must be one of {', '.join(CARBON_EVENTS)}.")
        if pd.notna(kg) and kg < 0:
            add(i, dt, "kg", "error", "Carbon quantity cannot be negative.")
        if pd.notna(act) and not (0 <= act <= 150):
            add(i, dt, "activity_pct", "error", "Carbon activity must be between 0 and 150 % of fresh carbon.")
        if r.get("event") == "Activity test" and pd.isna(act):
            add(i, dt, "activity_pct", "error", "An activity test needs a result.")
    return _issues(rows)
