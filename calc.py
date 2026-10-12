"""Metallurgical accounting engine for a gold CIL/CIP plant.

All inputs are daily plant figures. Units:
    tonnes_milled        dry tonnes
    head_grade_gpt       g/t (assayed feed grade)
    tails_grade_gpt      g/t total residue grade (solids + dissolved gold lost with the water)
    tails_solid_gpt      g/t gold in the residue solids (optional, replaces tails_grade_gpt)
    tails_solution_mgl   mg/L gold in the residue solution (optional)
    tails_pct_solids     % solids by mass of the residue slurry (needed for the solution loss)
    gold_poured_g        grams of FINE gold produced that day (gross doré weight x Au fineness)
    inventory_change_g   change in in-circuit gold (carbon, tanks, elution) in g;
                         positive = inventory built up (counts as gold recovered).
                         Calculated automatically from tank readings when those are recorded.
    other_inventory_g    gold held outside the tanks (elution column, cells, kiln carbon), in g
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

G_PER_OZ = 31.1035

NUMERIC_INPUTS = [
    "tonnes_milled", "mill_hours", "scheduled_hours", "head_grade_gpt", "tails_grade_gpt",
    "gold_poured_g", "inventory_change_g", "nacn_kg", "lime_kg", "power_kwh",
    "model_grade_gpt",      # optional: geological/grade-control model prediction
    "tails_solid_gpt", "tails_solution_mgl", "tails_pct_solids", "other_inventory_g",
]
TEXT_INPUTS = ["ore_type"]
INPUT_COLUMNS = ["date"] + NUMERIC_INPUTS + TEXT_INPUTS
REQUIRED_COLUMNS = ["date", "tonnes_milled", "mill_hours", "scheduled_hours", "head_grade_gpt",
                    "gold_poured_g", "nacn_kg", "lime_kg", "power_kwh"]


@dataclass
class Costs:
    nacn_per_kg: float = 3.2
    lime_per_kg: float = 0.25
    power_per_kwh: float = 0.12


@dataclass
class AlertThresholds:
    recovery_drop_pts: float = 2.0       # below the trailing 7-day mean
    unaccounted_warn_pct: float = 3.0    # % of feed gold
    unaccounted_crit_pct: float = 5.0
    head_gap_pct: float = 5.0            # |assay head - calculated head|
    reagent_spike_factor: float = 1.25   # vs trailing 7-day mean
    min_availability_pct: float = 85.0
    solution_loss_pct: float = 25.0      # dissolved gold as % of tails gold


def two_product_recovery(f, c, t):
    """Classic two-product formula, R = c(f - t) / (f(c - t)) * 100.

    f, c, t = feed, concentrate (or pregnant product) and tailings grades in the
    same units. Returns NaN where c == t or f == 0 (formula undefined).
    """
    f, c, t = (np.asarray(x, dtype=float) for x in (f, c, t))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = c * (f - t) / (f * (c - t)) * 100.0
    return np.where((c == t) | (f == 0), np.nan, r)


def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.where(b > 0)


def resolve_tails(df: pd.DataFrame) -> pd.DataFrame:
    """Work out the total tails grade from its parts when they are supplied.

    Solution loss (g per tonne of solids) = solution grade (mg/L = g/m3) x water per tonne of
    solids (m3/t) = tails_solution_mgl x (100 - %solids) / %solids.
    Total tails = solid tails + solution loss. When tails_solid_gpt is given it takes priority
    over a typed total. Adds solution_loss_gpt and solution_loss_pct_of_tails.
    """
    d = df.copy()
    for c in ("tails_grade_gpt", "tails_solid_gpt", "tails_solution_mgl", "tails_pct_solids"):
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ps = d["tails_pct_solids"]
    ratio = ((100 - ps) / ps).where((ps > 0) & (ps < 100))
    d["solution_loss_gpt"] = d["tails_solution_mgl"] * ratio
    has_solid = d["tails_solid_gpt"].notna()
    parts = d["tails_solid_gpt"] + d["solution_loss_gpt"].fillna(0)
    d["tails_grade_gpt"] = d["tails_grade_gpt"].where(~has_solid, parts)
    d["solution_loss_pct_of_tails"] = _safe_div(d["solution_loss_gpt"], d["tails_grade_gpt"]) * 100
    return d


def compute_daily(df: pd.DataFrame, costs: Costs | None = None) -> pd.DataFrame:
    """Add all derived metallurgical columns. Does not drop or alter inputs."""
    costs = costs or Costs()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date", kind="stable").reset_index(drop=True)
    for col in NUMERIC_INPUTS:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in TEXT_INPUTS:
        if col not in d.columns:
            d[col] = None
    d = resolve_tails(d)

    t = d["tonnes_milled"]
    # --- throughput / time ---
    d["tph"] = _safe_div(t, d["mill_hours"])
    d["utilisation_pct"] = d["mill_hours"] / 24.0 * 100
    d["availability_pct"] = _safe_div(d["mill_hours"], d["scheduled_hours"]) * 100

    # --- contained gold ---
    d["feed_gold_g"] = t * d["head_grade_gpt"]
    d["tails_gold_g"] = t * d["tails_grade_gpt"]
    d["gold_recovered_g"] = d["gold_poured_g"] + d["inventory_change_g"].fillna(0)

    # --- recovery, two independent ways ---
    d["recovery_grade_pct"] = _safe_div(
        d["head_grade_gpt"] - d["tails_grade_gpt"], d["head_grade_gpt"]
    ) * 100
    d["recovery_accounted_pct"] = _safe_div(d["gold_recovered_g"], d["feed_gold_g"]) * 100

    # --- reconciliation ---
    d["calc_head_gpt"] = _safe_div(d["gold_recovered_g"] + d["tails_gold_g"], t)
    d["head_gap_pct"] = _safe_div(d["head_grade_gpt"] - d["calc_head_gpt"], d["head_grade_gpt"]) * 100
    d["unaccounted_g"] = d["feed_gold_g"] - d["tails_gold_g"] - d["gold_recovered_g"]
    d["unaccounted_pct"] = _safe_div(d["unaccounted_g"], d["feed_gold_g"]) * 100

    # --- consumptions & cost ---
    d["nacn_kgpt"] = _safe_div(d["nacn_kg"], t)
    d["lime_kgpt"] = _safe_div(d["lime_kg"], t)
    d["power_kwhpt"] = _safe_div(d["power_kwh"], t)
    d["opex_per_t"] = (
        d["nacn_kgpt"] * costs.nacn_per_kg
        + d["lime_kgpt"] * costs.lime_per_kg
        + d["power_kwhpt"] * costs.power_per_kwh
    )
    d["gold_recovered_oz"] = d["gold_recovered_g"] / G_PER_OZ
    d["process_cost_per_oz"] = _safe_div(d["opex_per_t"] * t, d["gold_recovered_oz"])

    # --- trailing 7-day context (previous 7 days, excluding today) ---
    d["rec_7d_prev"] = d["recovery_grade_pct"].shift(1).rolling(7, min_periods=3).mean()
    d["nacn_7d_prev"] = d["nacn_kgpt"].shift(1).rolling(7, min_periods=3).mean()
    return d


def build_alerts(d: pd.DataFrame, th: AlertThresholds | None = None) -> pd.DataFrame:
    """Return one row per alert: date, severity, type, message."""
    th = th or AlertThresholds()
    out = []

    def add(row, sev, typ, msg):
        out.append({"date": row["date"], "severity": sev, "type": typ, "message": msg})

    for _, r in d.iterrows():
        if pd.notna(r["rec_7d_prev"]) and pd.notna(r["recovery_grade_pct"]):
            drop = r["rec_7d_prev"] - r["recovery_grade_pct"]
            if drop > th.recovery_drop_pts:
                add(r, "critical" if drop > 2 * th.recovery_drop_pts else "warning",
                    "Recovery drop",
                    f"Recovery {r['recovery_grade_pct']:.1f}% is {drop:.1f} pts below the "
                    f"trailing 7-day mean ({r['rec_7d_prev']:.1f}%).")
        if pd.notna(r["unaccounted_pct"]):
            u = r["unaccounted_pct"]
            if abs(u) >= th.unaccounted_crit_pct:
                add(r, "critical", "Unaccounted gold",
                    f"{u:.1f}% of feed gold unaccounted ({r['unaccounted_g']:.0f} g). "
                    "Check sampling, inventory, pour weights.")
            elif abs(u) >= th.unaccounted_warn_pct:
                add(r, "warning", "Unaccounted gold",
                    f"{u:.1f}% of feed gold unaccounted ({r['unaccounted_g']:.0f} g).")
        if pd.notna(r["head_gap_pct"]) and abs(r["head_gap_pct"]) > th.head_gap_pct:
            add(r, "warning", "Head grade mismatch",
                f"Assay head {r['head_grade_gpt']:.2f} g/t vs calculated head "
                f"{r['calc_head_gpt']:.2f} g/t ({r['head_gap_pct']:+.1f}%).")
        if (pd.notna(r["nacn_kgpt"]) and pd.notna(r["nacn_7d_prev"])
                and r["nacn_kgpt"] > r["nacn_7d_prev"] * th.reagent_spike_factor):
            add(r, "warning", "Cyanide spike",
                f"NaCN {r['nacn_kgpt']:.3f} kg/t vs 7-day mean {r['nacn_7d_prev']:.3f} kg/t.")
        if pd.notna(r["availability_pct"]) and r["availability_pct"] < th.min_availability_pct:
            add(r, "info", "Low availability",
                f"Mill availability {r['availability_pct']:.0f}% of scheduled hours.")
        sl = r.get("solution_loss_pct_of_tails")
        if pd.notna(sl) and sl > th.solution_loss_pct:
            add(r, "warning", "Dissolved gold loss",
                f"{sl:.0f}% of the tails gold is dissolved in the solution "
                f"({r['solution_loss_gpt']:.3f} g/t). Gold is leaching but not adsorbing: "
                "check carbon activity, carbon concentration and cyanide/oxygen in the last tanks.")
    cols = ["date", "severity", "type", "message"]
    return pd.DataFrame(out, columns=cols)


def _weighted(g: pd.DataFrame) -> pd.Series:
    t = g["tonnes_milled"].sum()
    feed = g["feed_gold_g"].sum()
    tails = g["tails_gold_g"].sum()
    rec = g["gold_recovered_g"].sum()
    hours = g["mill_hours"].sum()
    res = {
        "days": len(g),
        "tonnes_milled": t,
        "mill_hours": hours,
        "tph": t / hours if hours else np.nan,
        "head_grade_gpt": feed / t if t else np.nan,
        "tails_grade_gpt": tails / t if t else np.nan,
        "recovery_grade_pct": (feed - tails) / feed * 100 if feed else np.nan,
        "recovery_accounted_pct": rec / feed * 100 if feed else np.nan,
        "gold_recovered_oz": rec / G_PER_OZ,
        "unaccounted_g": g["unaccounted_g"].sum(),
        "unaccounted_pct": g["unaccounted_g"].sum() / feed * 100 if feed else np.nan,
        "nacn_kgpt": g["nacn_kg"].sum() / t if t else np.nan,
        "lime_kgpt": g["lime_kg"].sum() / t if t else np.nan,
        "power_kwhpt": g["power_kwh"].sum() / t if t else np.nan,
        "availability_pct": (hours / g["scheduled_hours"].sum() * 100
                             if g["scheduled_hours"].sum() else np.nan),
    }
    if "solution_loss_gpt" in g and g["solution_loss_gpt"].notna().any():
        m = g[g["solution_loss_gpt"].notna()]
        sol_gold = (m["tonnes_milled"] * m["solution_loss_gpt"]).sum()
        res["solution_loss_pct_of_tails"] = sol_gold / m["tails_gold_g"].sum() * 100 if m["tails_gold_g"].sum() else np.nan
    else:
        res["solution_loss_pct_of_tails"] = np.nan
    if "model_grade_gpt" in g and g["model_grade_gpt"].notna().any():
        m = g.loc[g["model_grade_gpt"].notna()]
        model_metal = (m["tonnes_milled"] * m["model_grade_gpt"]).sum()
        res["mine_call_factor"] = m["feed_gold_g"].sum() / model_metal if model_metal else np.nan
    else:
        res["mine_call_factor"] = np.nan
    return pd.Series(res)


def overall(d: pd.DataFrame) -> pd.Series:
    """Tonnage-weighted totals/ratios for the whole frame (e.g. selected date range)."""
    return _weighted(d)


def summarise(d: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Tonnage-weighted summary by period ('M' month, 'W' week). Never averages
    grades or recoveries arithmetically; it recomputes from contained metal."""
    d = d.copy()
    d["period"] = d["date"].dt.to_period(freq).astype(str)
    return d.groupby("period", sort=True).apply(_weighted, include_groups=False).reset_index()


def by_ore_type(d: pd.DataFrame) -> pd.DataFrame:
    """Tonnage-weighted performance for each ore type, plus each type's share of the tonnes."""
    x = d.copy()
    x["ore_type"] = x["ore_type"].fillna("Unspecified").astype(str).replace("", "Unspecified")
    out = x.groupby("ore_type", sort=True).apply(_weighted, include_groups=False).reset_index()
    total = out["tonnes_milled"].sum()
    out["share_pct"] = out["tonnes_milled"] / total * 100 if total else np.nan
    return out


# metric label, targets key, actual column in overall(), direction, use absolute value?
_PLAN_ROWS = [
    ("Recovery (%)", "recovery_pct", "recovery_grade_pct", "higher", False),
    ("Head grade (g/t)", "head_gpt", "head_grade_gpt", "higher", False),
    ("Tails grade (g/t)", "tails_gpt", "tails_grade_gpt", "lower", False),
    ("Throughput (t/h)", "throughput_tph", "tph", "higher", False),
    ("Availability (%)", "availability_pct", "availability_pct", "higher", False),
    ("NaCN (kg/t)", "nacn_kgpt", "nacn_kgpt", "lower", False),
    ("Power (kWh/t)", "power_kwhpt", "power_kwhpt", "lower", False),
    ("Unaccounted gold (%)", "unaccounted_pct", "unaccounted_pct", "lower", True),
]


def plan_vs_actual(actual: pd.Series, targets: dict) -> pd.DataFrame:
    """Compare period results with targets.

    Status: On target (at or better than target, allowing 0.5 % of rounding), Slightly behind
    (up to 2 % short on a 'higher is better' measure, or up to 5 % over on a 'lower is better' one),
    Behind (anything worse).
    """
    rows = []
    for label, tkey, acol, direction, use_abs in _PLAN_ROWS:
        tgt = targets.get(tkey)
        val = actual.get(acol, np.nan)
        if tgt is None or pd.isna(val) or not tgt:
            continue
        v = abs(val) if use_abs else val
        if direction == "higher":
            status = "On target" if v >= 0.995 * tgt else "Slightly behind" if v >= 0.98 * tgt else "Behind"
        else:
            status = "On target" if v <= 1.005 * tgt else "Slightly behind" if v <= 1.05 * tgt else "Behind"
        rows.append({"metric": label, "target": float(tgt), "actual": float(val),
                     "variance": float(val - tgt), "status": status})
    return pd.DataFrame(rows, columns=["metric", "target", "actual", "variance", "status"])


def mine_call_factor(plant_metal_g: float, model_metal_g: float) -> float:
    """MCF = metal the plant sees in its feed / metal the model predicted."""
    return plant_metal_g / model_metal_g if model_metal_g else float("nan")
