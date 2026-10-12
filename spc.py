"""Statistical process control (individuals chart) for plant results.

Limits come from a baseline period of normal operation:
    centre = mean of the baseline
    sigma  = mean moving range / 1.128     (moving range = |today - yesterday|)
    limits = centre +/- 3 sigma
A point is a signal when it falls outside the limits, when 2 of 3 consecutive points are beyond
2 sigma on the same side, or when 8 consecutive points sit on the same side of the centre
(the process mean has shifted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# column -> (label, which side is bad)
SPC_METRICS = {
    "recovery_grade_pct": ("Recovery (%)", "low"),
    "tails_grade_gpt": ("Tails grade (g/t)", "high"),
    "head_grade_gpt": ("Head grade (g/t)", "either"),
    "nacn_kgpt": ("NaCN (kg/t)", "high"),
    "unaccounted_pct": ("Unaccounted gold (%)", "high"),
    "tph": ("Throughput (t/h)", "low"),
}
ALERT_METRICS = ("recovery_grade_pct", "tails_grade_gpt", "nacn_kgpt", "tph")
MIN_BASELINE = 8


def individuals_limits(x: pd.Series, baseline_n: int | None = None) -> dict:
    v = pd.to_numeric(x, errors="coerce").dropna()
    base = v.iloc[: int(baseline_n)] if baseline_n else v
    if len(base) < MIN_BASELINE:
        base = v
    if len(base) < 2:
        return {"center": np.nan, "sigma": np.nan, "ucl": np.nan, "lcl": np.nan, "n_base": len(base)}
    mr = base.diff().abs().dropna()
    sigma = float(mr.mean() / 1.128)
    c = float(base.mean())
    return {"center": c, "sigma": sigma, "ucl": c + 3 * sigma, "lcl": c - 3 * sigma, "n_base": len(base)}


def control_table(d: pd.DataFrame, col: str, baseline_n: int | None = 30) -> tuple[pd.DataFrame, dict]:
    """Return (table, limits). Table columns: date, value, z, flag."""
    s = d[["date", col]].dropna().reset_index(drop=True)
    s = s.rename(columns={col: "value"})
    lim = individuals_limits(s["value"], baseline_n)
    if not np.isfinite(lim["sigma"]) or lim["sigma"] <= 0:
        s["z"] = np.nan
        s["flag"] = ""
        return s, lim
    s["z"] = (s["value"] - lim["center"]) / lim["sigma"]
    z = s["z"].to_numpy()
    side = np.sign(z)
    flags = [[] for _ in z]
    run = 0
    for i in range(len(z)):
        if abs(z[i]) > 3:
            flags[i].append("beyond 3σ")
        if i >= 2:
            w = z[i - 2: i + 1]
            same = side[i - 2: i + 1] == side[i]
            if abs(z[i]) > 2 and ((np.abs(w) > 2) & same).sum() >= 2:
                flags[i].append("2 of 3 beyond 2σ")
        run = run + 1 if i > 0 and side[i] == side[i - 1] and side[i] != 0 else 1
        if run == 8:
            flags[i].append("8 in a row on one side")
    s["flag"] = [", ".join(f) for f in flags]
    return s, lim


def spc_alerts(d: pd.DataFrame, baseline_n: int | None = 30,
               metrics: dict | None = None) -> pd.DataFrame:
    """Signals on the bad side of the key metrics, as alert rows."""
    # unaccounted gold already has its own threshold alert, and head grade has no 'bad' side
    metrics = metrics or {k: v for k, v in SPC_METRICS.items() if k in ALERT_METRICS}
    rows = []
    for col, (label, bad) in metrics.items():
        if col not in d.columns:
            continue
        t, lim = control_table(d, col, baseline_n)
        strong = t["flag"].str.contains("beyond 3σ|8 in a row", regex=True)   # a lone '2 of 3' stays on the chart only
        for _, r in t[strong].iterrows():
            side = "below" if r["z"] < 0 else "above"
            if bad == "low" and r["z"] > 0 or bad == "high" and r["z"] < 0:
                continue  # signal on the good side
            shift = "8 in a row" in r["flag"]
            rows.append({"date": r["date"], "severity": "info" if shift and "σ" not in r["flag"] else "warning",
                         "type": "SPC signal",
                         "message": f"{label} = {r['value']:.3g} is {side} the normal range "
                                    f"({lim['lcl']:.3g} to {lim['ucl']:.3g}); {r['flag']}."})
    return pd.DataFrame(rows, columns=["date", "severity", "type", "message"])
