"""Gold room: doré bars, fine gold, fineness and reconciliation with the plant's daily gold figures.

    fine gold (g)       = gross bar weight x Au fineness / 100
    fine silver (g)     = gross bar weight x Ag fineness / 100
    smelt recovery (%)  = fine gold in bar / (fine gold in bar + gold lost in slag) x 100
The plant's daily gold_poured_g should be FINE gold, so the monthly total of fine gold from the bars
should match it, apart from timing differences at month ends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calc import G_PER_OZ


def prepare(pours: pd.DataFrame) -> pd.DataFrame:
    p = pours.copy()
    p["date"] = pd.to_datetime(p["date"])
    for c in ("gross_g", "au_pct", "ag_pct", "slag_au_g"):
        if c not in p.columns:
            p[c] = np.nan
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["fine_au_g"] = p["gross_g"] * p["au_pct"] / 100
    p["fine_ag_g"] = p["gross_g"] * p["ag_pct"] / 100
    p["fine_au_oz"] = p["fine_au_g"] / G_PER_OZ
    p["ag_to_au"] = (p["ag_pct"] / p["au_pct"]).where(p["au_pct"] > 0)
    tot = p["fine_au_g"] + p["slag_au_g"]
    p["smelt_recovery_pct"] = (p["fine_au_g"] / tot * 100).where(p["slag_au_g"].notna() & (tot > 0))
    return p.sort_values("date").reset_index(drop=True)


def monthly(p: pd.DataFrame) -> pd.DataFrame:
    """Per-month totals; fineness is weighted by bar weight, not averaged."""
    if p.empty:
        return pd.DataFrame(columns=["period", "bars", "gross_kg", "fine_au_g", "fine_au_oz",
                                     "au_fineness_pct", "ag_to_au", "smelt_recovery_pct"])
    x = p.copy()
    x["period"] = x["date"].dt.to_period("M").astype(str)
    rows = []
    for period, g in x.groupby("period", sort=True):
        gross, fine = g["gross_g"].sum(), g["fine_au_g"].sum()
        s = g[g["slag_au_g"].notna()]
        sm = s["fine_au_g"].sum() / (s["fine_au_g"].sum() + s["slag_au_g"].sum()) * 100 if len(s) else np.nan
        rows.append({"period": period, "bars": len(g), "gross_kg": gross / 1000, "fine_au_g": fine,
                     "fine_au_oz": fine / G_PER_OZ,
                     "au_fineness_pct": fine / gross * 100 if gross else np.nan,
                     "ag_to_au": g["fine_ag_g"].sum() / fine if fine else np.nan,
                     "smelt_recovery_pct": sm})
    return pd.DataFrame(rows)


def reconcile(p: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Fine gold from bars vs the gold_poured_g typed in the daily records, per month."""
    if p.empty or daily.empty:
        return pd.DataFrame(columns=["period", "bars_fine_g", "daily_poured_g", "difference_g", "difference_pct"])
    a = p.assign(period=p["date"].dt.to_period("M").astype(str)).groupby("period")["fine_au_g"].sum()
    b = daily.assign(period=pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)).groupby("period")["gold_poured_g"].sum()
    out = pd.concat([a.rename("bars_fine_g"), b.rename("daily_poured_g")], axis=1).dropna().reset_index()
    out = out.rename(columns={"index": "period"})
    out["difference_g"] = out["bars_fine_g"] - out["daily_poured_g"]
    out["difference_pct"] = out["difference_g"] / out["daily_poured_g"] * 100
    # a bar poured on the 1st holds gold from the end of last month: timing differences reverse the
    # following month, so the running total is the number to judge
    out["cumulative_difference_g"] = out["difference_g"].cumsum()
    return out
