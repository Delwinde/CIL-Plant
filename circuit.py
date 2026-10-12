"""CIL circuit calculations: tank readings, in-circuit gold inventory, residence time, carbon.

Key formulas
    carbon in a tank (kg)      = carbon concentration (g/L slurry) x tank volume (m3)        [g/L = kg/m3]
    gold on carbon (g)         = carbon (kg) x loading (g Au per t carbon) / 1000
    liquid fraction of slurry  = (1 - w) x slurry density,  w = mass fraction solids
    slurry density (t/m3)      = 1 / (w / SG_solids + (1 - w))
    gold in solution (g)       = solution grade (mg/L = g/m3) x liquid volume (m3)
    in-circuit inventory (g)   = gold on carbon + gold in solution (all tanks) + other_inventory_g
    inventory change (g)       = today's inventory - yesterday's inventory  (positive = build-up)
    residence time (h)         = total tank volume / slurry flow,  flow (m3/h) = tph / (w x slurry density)

Gold locked in the ore solids inside the tanks is not counted (it is small and changes slowly when the
plant is steady). If your plant needs it, add it to other_inventory_g.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CARBON_EVENTS = ["Elution", "Regeneration", "Carbon addition", "Carbon removed", "Activity test"]
TANK_COLUMNS = ["tank", "volume_m3", "pct_solids", "free_cn_ppm", "ph", "do_ppm",
                "carbon_conc_gpl", "carbon_loading_gpt", "solution_au_mgl"]
ALERT_COLUMNS = ["date", "severity", "type", "message"]


@dataclass
class CircuitLimits:
    min_ph: float = 10.0
    min_do_ppm: float = 6.0
    min_free_cn_ppm: float = 100.0
    carbon_conc_min_gpl: float = 8.0
    carbon_conc_max_gpl: float = 25.0
    min_activity_pct: float = 70.0
    solids_sg: float = 2.7
    default_pct_solids: float = 40.0


def slurry_density(pct_solids, sg=2.7):
    """t/m3 for a slurry of the given % solids by mass (liquid taken as water, 1 t/m3)."""
    w = np.asarray(pct_solids, dtype=float) / 100.0
    return 1.0 / (w / sg + (1.0 - w))


def liquid_fraction(pct_solids, sg=2.7):
    """Fraction of the slurry volume that is liquid."""
    w = np.asarray(pct_solids, dtype=float) / 100.0
    return (1.0 - w) * slurry_density(pct_solids, sg)


def tank_calcs(tanks: pd.DataFrame, lim: CircuitLimits | None = None) -> pd.DataFrame:
    """Add carbon mass, gold on carbon, gold in solution, inventory and adsorption efficiency."""
    lim = lim or CircuitLimits()
    t = tanks.copy()
    t["date"] = pd.to_datetime(t["date"])
    for c in TANK_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.sort_values(["date", "tank"]).reset_index(drop=True)

    ps = t["pct_solids"].fillna(lim.default_pct_solids)
    t["carbon_kg"] = t["carbon_conc_gpl"] * t["volume_m3"]
    t["gold_on_carbon_g"] = t["carbon_kg"] * t["carbon_loading_gpt"] / 1000.0
    t["liquid_m3"] = t["volume_m3"] * liquid_fraction(ps, lim.solids_sg)
    t["gold_in_solution_g"] = t["solution_au_mgl"] * t["liquid_m3"]
    # a tank only counts towards the inventory when its carbon figures are present
    has_carbon = t["gold_on_carbon_g"].notna()
    t["inventory_g"] = np.where(has_carbon, t["gold_on_carbon_g"] + t["gold_in_solution_g"].fillna(0), np.nan)

    prev = t.groupby("date")["solution_au_mgl"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = (prev - t["solution_au_mgl"]) / prev * 100
    t["adsorption_eff_pct"] = eff.where(prev > 0)
    return t


def daily_inventory(tcalc: pd.DataFrame, n_tanks: int) -> pd.DataFrame:
    """Tank inventory per day. 'complete' means every tank has carbon figures."""
    if tcalc.empty:
        return pd.DataFrame(columns=["date", "tanks_reported", "tank_inventory_g", "carbon_kg", "complete"])
    g = tcalc.groupby("date")
    out = pd.DataFrame({
        "tanks_reported": g["inventory_g"].count(),
        "tank_inventory_g": g["inventory_g"].sum(min_count=1),
        "carbon_kg": g["carbon_kg"].sum(min_count=1),
    }).reset_index()
    out["complete"] = out["tanks_reported"] >= n_tanks
    return out


def apply_inventory(daily: pd.DataFrame, tanks: pd.DataFrame, lim: CircuitLimits | None = None,
                    n_tanks: int = 6):
    """Replace the typed inventory change with the one calculated from tank readings.

    A day uses the automatic value only when that day and the day before both have complete tank
    readings. Otherwise the typed inventory_change_g (or zero) is kept.
    Returns (daily_with_columns, notes) where notes explains days that could not use the calculation.
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    if "inventory_change_g" not in d.columns:
        d["inventory_change_g"] = np.nan
    d["inventory_change_g"] = pd.to_numeric(d["inventory_change_g"], errors="coerce")
    d["inventory_source"] = np.where(d["inventory_change_g"].notna(), "manual", "none")
    d["circuit_inventory_g"] = np.nan
    notes = []
    if tanks is None or tanks.empty:
        return d, pd.DataFrame(notes, columns=["date", "severity", "message"])

    tcalc = tank_calcs(tanks, lim)
    inv = daily_inventory(tcalc, n_tanks)
    other = pd.to_numeric(d.get("other_inventory_g"), errors="coerce") if "other_inventory_g" in d else pd.Series(np.nan, index=d.index)
    other = pd.Series(other.fillna(0).values, index=d["date"].values)
    other = other[~other.index.duplicated(keep="last")]

    for _, r in inv[~inv["complete"]].iterrows():
        notes.append({"date": r["date"], "severity": "warning",
                      "message": f"Only {int(r['tanks_reported'])} of {n_tanks} tanks have carbon readings, "
                                 "so in-circuit inventory was not calculated for this day."})

    ok = inv[inv["complete"]].set_index("date")["tank_inventory_g"]
    total = ok + other.reindex(ok.index).fillna(0)
    one_day = pd.Timedelta(days=1)
    for i, r in d.iterrows():
        dt = r["date"]
        if dt in total.index:
            d.at[i, "circuit_inventory_g"] = total[dt]
            if (dt - one_day) in total.index:
                d.at[i, "inventory_change_g"] = total[dt] - total[dt - one_day]
                d.at[i, "inventory_source"] = "auto"
    return d, pd.DataFrame(notes, columns=["date", "severity", "message"])


def add_residence_time(daily: pd.DataFrame, tcalc: pd.DataFrame, default_volumes,
                       lim: CircuitLimits | None = None) -> pd.DataFrame:
    """Add circuit_volume_m3 and residence_time_h. Needs tph in the daily frame."""
    lim = lim or CircuitLimits()
    d = daily.copy()
    default_total = float(np.nansum(default_volumes))
    if tcalc is not None and not tcalc.empty:
        g = tcalc.groupby("date").agg(vol=("volume_m3", "sum"), ps=("pct_solids", "mean"))
        d = d.merge(g, left_on="date", right_index=True, how="left")
    else:
        d["vol"], d["ps"] = np.nan, np.nan
    d["circuit_volume_m3"] = d["vol"].where(d["vol"] > 0, default_total)
    ps = d["ps"].fillna(lim.default_pct_solids)
    w = ps / 100.0
    flow = d["tph"] / (w * slurry_density(ps, lim.solids_sg))
    d["residence_time_h"] = (d["circuit_volume_m3"] / flow).where(flow > 0)
    return d.drop(columns=["vol", "ps"])


def circuit_alerts(tcalc: pd.DataFrame, lim: CircuitLimits | None = None) -> pd.DataFrame:
    """One alert per day and problem, listing the tanks affected."""
    lim = lim or CircuitLimits()
    if tcalc.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    found: dict[tuple, list] = {}

    def hit(date, typ, sev, tank, val):
        found.setdefault((date, typ, sev), []).append((int(tank), val))

    for _, r in tcalc.iterrows():
        dt, tk = r["date"], r["tank"]
        if pd.notna(r["ph"]) and r["ph"] < lim.min_ph:
            hit(dt, "Low pH", "critical" if r["ph"] < lim.min_ph - 0.5 else "warning", tk, r["ph"])
        if pd.notna(r["do_ppm"]) and r["do_ppm"] < lim.min_do_ppm:
            hit(dt, "Low dissolved oxygen", "warning", tk, r["do_ppm"])
        if pd.notna(r["free_cn_ppm"]) and r["free_cn_ppm"] < lim.min_free_cn_ppm:
            hit(dt, "Low free cyanide", "warning", tk, r["free_cn_ppm"])
        if pd.notna(r["carbon_conc_gpl"]) and r["carbon_conc_gpl"] < lim.carbon_conc_min_gpl:
            hit(dt, "Low carbon concentration", "warning", tk, r["carbon_conc_gpl"])
        if pd.notna(r["carbon_conc_gpl"]) and r["carbon_conc_gpl"] > lim.carbon_conc_max_gpl:
            hit(dt, "High carbon concentration", "info", tk, r["carbon_conc_gpl"])
    texts = {
        "Low pH": (lambda: f"pH below {lim.min_ph:g}", "Below pH 10 cyanide is lost as HCN gas."),
        "Low dissolved oxygen": (lambda: f"Dissolved oxygen below {lim.min_do_ppm:g} mg/L",
                                 "Leaching slows without oxygen."),
        "Low free cyanide": (lambda: f"Free cyanide below {lim.min_free_cn_ppm:g} ppm",
                             "Gold dissolves slowly when cyanide is short."),
        "Low carbon concentration": (lambda: f"Carbon concentration below {lim.carbon_conc_min_gpl:g} g/L",
                                     "Not enough carbon to capture the dissolved gold."),
        "High carbon concentration": (lambda: f"Carbon concentration above {lim.carbon_conc_max_gpl:g} g/L",
                                      "Too much carbon raises carbon losses and pumping problems."),
    }
    rows = []
    for (dt, typ, sev), items in sorted(found.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        head, hint = texts[typ]
        where = ", ".join(f"tank {tk} ({v:.3g})" for tk, v in sorted(items))
        rows.append({"date": dt, "severity": sev, "type": typ, "message": f"{head()} in {where}. {hint}"})
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def carbon_stats(log: pd.DataFrame, tonnes_milled: float, end_date, lim: CircuitLimits | None = None) -> dict:
    """Carbon management figures for a period (log should already be filtered to the period)."""
    lim = lim or CircuitLimits()
    out = {"carbon_added_kg": 0.0, "carbon_consumption_gpt": np.nan, "elutions": 0, "regenerations": 0,
           "days_since_elution": np.nan, "days_since_regen": np.nan,
           "latest_activity_pct": np.nan, "latest_activity_date": pd.NaT}
    if log is None or log.empty:
        return out
    L = log.copy()
    L["date"] = pd.to_datetime(L["date"])
    end = pd.to_datetime(end_date)
    added = L.loc[L["event"] == "Carbon addition", "kg"].sum()
    out["carbon_added_kg"] = float(added)
    if tonnes_milled and tonnes_milled > 0:
        out["carbon_consumption_gpt"] = added * 1000.0 / tonnes_milled
    out["elutions"] = int((L["event"] == "Elution").sum())
    out["regenerations"] = int((L["event"] == "Regeneration").sum())
    for ev, key in (("Elution", "days_since_elution"), ("Regeneration", "days_since_regen")):
        sub = L[L["event"] == ev]
        if not sub.empty:
            out[key] = float((end - sub["date"].max()).days)
    act = L[(L["event"] == "Activity test") & L["activity_pct"].notna()].sort_values("date")
    if not act.empty:
        out["latest_activity_pct"] = float(act.iloc[-1]["activity_pct"])
        out["latest_activity_date"] = act.iloc[-1]["date"]
    return out


def carbon_alerts(log: pd.DataFrame, lim: CircuitLimits | None = None) -> pd.DataFrame:
    lim = lim or CircuitLimits()
    rows = []
    if log is not None and not log.empty:
        for _, r in log[(log["event"] == "Activity test") & log["activity_pct"].notna()].iterrows():
            if r["activity_pct"] < lim.min_activity_pct:
                sev = "critical" if r["activity_pct"] < lim.min_activity_pct - 15 else "warning"
                rows.append({"date": pd.to_datetime(r["date"]), "severity": sev, "type": "Low carbon activity",
                             "message": f"Carbon activity {r['activity_pct']:.0f}% of fresh is below "
                                        f"{lim.min_activity_pct:g}%. Consider regeneration or replacing carbon; "
                                        "fouled carbon lets gold escape in the tails solution."})
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)
