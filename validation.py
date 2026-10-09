"""Data validation for daily plant records.

Errors  = physically impossible or unusable; excluded from KPIs by default.
Warnings = suspicious but possible; kept in KPIs, shown to the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calc import G_PER_OZ, REQUIRED_COLUMNS  # noqa: F401

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
}
POUR_TOLERANCE = 1.02  # recovered gold may not exceed feed gold by more than 2%


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return issues: row (index in df), date, field, severity, message."""
    issues: list[dict] = []

    def add(i, date, field, sev, msg):
        issues.append({"row": i, "date": date, "field": field, "severity": sev, "message": msg})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        add(None, None, ",".join(missing), "error", f"Missing required column(s): {', '.join(missing)}")
        return pd.DataFrame(issues)

    d = df.copy()
    d["_date"] = pd.to_datetime(d["date"], errors="coerce")
    nums = [c for c in REQUIRED_COLUMNS if c != "date"]
    for c in nums:
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
                    continue  # treated as 0
                add(i, dt, c, "error", f"{c} is missing or not a number.")
                bad_row = True
            elif v < 0 and c != "inventory_change_g":
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
                f"Tails grade {tl} g/t is higher than head grade {h} g/t (negative recovery). "
                "Likely a typo or swapped values.")
        feed = t * h
        recovered = r["_gold_poured_g"] + (0 if pd.isna(r["_inventory_change_g"]) else r["_inventory_change_g"])
        if feed > 0 and recovered > feed * POUR_TOLERANCE:
            add(i, dt, "gold_poured_g", "error",
                f"Gold recovered ({recovered:.0f} g) exceeds feed gold ({feed:.0f} g). "
                "Check pour weight units or inventory sign.")

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

    cols = ["row", "date", "field", "severity", "message"]
    return pd.DataFrame(issues, columns=cols)


def error_rows(issues: pd.DataFrame) -> set:
    """Row indices that carry at least one error."""
    if issues.empty:
        return set()
    e = issues[(issues["severity"] == "error") & issues["row"].notna()]
    return set(e["row"].astype(int))
