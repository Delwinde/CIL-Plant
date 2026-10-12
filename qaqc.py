"""Laboratory QA/QC: standards, blanks, duplicates and check assays.

Standards (certified reference material, CRM)
    z = (assay - certified value) / certified standard deviation
    |z| > 2 warning, |z| > 3 fail. Control-chart rules (Westgard) catch drifts and bias:
        1_3s  one point beyond 3 SD
        2_2s  two consecutive points beyond 2 SD on the same side
        4_1s  four consecutive points beyond 1 SD on the same side
        8x    eight consecutive points on the same side of the certified value
Blanks      pass when the assay is at or below the blank limit (otherwise contamination).
Duplicates and check assays
    relative difference RD % = |a - b| / mean(a, b) x 100. Warning above the limit, fail above 2x the limit.
    Below 0.2 g/t the absolute difference is used instead (tolerance 0.05 g/t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SAMPLE_TYPES = ["Standard", "Blank", "Duplicate", "Check assay"]


@dataclass
class QAQCLimits:
    z_warn: float = 2.0
    z_fail: float = 3.0
    blank_limit_gpt: float = 0.05
    dup_rd_pct: float = 10.0


def _num(r, col):
    return pd.to_numeric(r.get(col), errors="coerce")


def evaluate(df: pd.DataFrame, lim: QAQCLimits | None = None) -> pd.DataFrame:
    """Add z, bias_pct, rd_pct, status (Pass/Warning/Fail/Incomplete) and reason to each sample."""
    lim = lim or QAQCLimits()
    e = df.copy()
    e["date"] = pd.to_datetime(e["date"])
    for c in ("assay_gpt", "expected_gpt", "sd_gpt", "original_gpt"):
        if c not in e.columns:
            e[c] = np.nan
        e[c] = pd.to_numeric(e[c], errors="coerce")
    z, bias, rd, status, reason = [], [], [], [], []
    for _, r in e.iterrows():
        a, exp, sd, orig = r["assay_gpt"], r["expected_gpt"], r["sd_gpt"], r["original_gpt"]
        zi = bi = rdi = np.nan
        st, why = "Pass", ""
        typ = r.get("sample_type")
        if pd.isna(a):
            st, why = "Incomplete", "No assay result."
        elif typ == "Standard":
            if pd.isna(exp) or pd.isna(sd) or sd <= 0:
                st, why = "Incomplete", "Needs the certified value and its standard deviation."
            else:
                zi = (a - exp) / sd
                bi = (a - exp) / exp * 100 if exp else np.nan
                if abs(zi) > lim.z_fail:
                    st, why = "Fail", f"z = {zi:+.1f}: beyond {lim.z_fail:g} SD of the certified value."
                elif abs(zi) > lim.z_warn:
                    st, why = "Warning", f"z = {zi:+.1f}: beyond {lim.z_warn:g} SD of the certified value."
        elif typ == "Blank":
            if a > lim.blank_limit_gpt:
                st, why = "Fail", f"Blank reads {a:.3f} g/t, above the {lim.blank_limit_gpt:g} g/t limit: possible contamination."
        elif typ in ("Duplicate", "Check assay"):
            if pd.isna(orig):
                st, why = "Incomplete", "Needs the original assay to compare with."
            else:
                mean = (a + orig) / 2
                rdi = abs(a - orig) / mean * 100 if mean > 0 else 0.0
                bi = (a - orig) / orig * 100 if orig else np.nan
                if mean < 0.2:
                    if abs(a - orig) > 0.05:
                        st, why = "Warning", f"Low-grade pair differs by {abs(a - orig):.3f} g/t (tolerance 0.05)."
                elif rdi > 2 * lim.dup_rd_pct:
                    st, why = "Fail", f"Relative difference {rdi:.1f}% is more than twice the {lim.dup_rd_pct:g}% limit."
                elif rdi > lim.dup_rd_pct:
                    st, why = "Warning", f"Relative difference {rdi:.1f}% is above the {lim.dup_rd_pct:g}% limit."
        else:
            st, why = "Incomplete", "Unknown sample type."
        z.append(zi); bias.append(bi); rd.append(rdi); status.append(st); reason.append(why)
    e["z"], e["bias_pct"], e["rd_pct"], e["status"], e["reason"] = z, bias, rd, status, reason
    return e


def summary(ev: pd.DataFrame) -> pd.DataFrame:
    """Pass rates and mean bias per sample type."""
    if ev.empty:
        return pd.DataFrame(columns=["sample_type", "n", "pass_pct", "warnings", "fails", "mean_bias_pct"])
    rows = []
    for typ in SAMPLE_TYPES:
        s = ev[(ev["sample_type"] == typ) & (ev["status"] != "Incomplete")]
        if s.empty:
            continue
        rows.append({"sample_type": typ, "n": len(s),
                     "pass_pct": (s["status"] == "Pass").mean() * 100,
                     "warnings": int((s["status"] == "Warning").sum()),
                     "fails": int((s["status"] == "Fail").sum()),
                     "mean_bias_pct": s["bias_pct"].mean() if typ in ("Standard", "Check assay") else np.nan})
    return pd.DataFrame(rows)


def westgard(z: pd.Series) -> pd.Series:
    """Control-chart rule violated at each point (comma-separated), '' when none. z must be in time order."""
    v = z.to_numpy(dtype=float)
    out = [[] for _ in v]
    side = np.sign(v)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        if abs(v[i]) > 3:
            out[i].append("1_3s")
        if i >= 1 and abs(v[i]) > 2 and abs(v[i - 1]) > 2 and side[i] == side[i - 1]:
            out[i].append("2_2s")
        if i >= 3 and all(abs(v[i - k]) > 1 and side[i - k] == side[i] for k in range(4)):
            out[i].append("4_1s")
        if i >= 7 and all(side[i - k] == side[i] and side[i] != 0 for k in range(8)):
            out[i].append("8x")
    return pd.Series([",".join(x) for x in out], index=z.index)


def standards_chart(ev: pd.DataFrame) -> pd.DataFrame:
    """Standards in date order with their z-scores and the control-chart rule hit at each point."""
    s = ev[(ev["sample_type"] == "Standard") & ev["z"].notna()].sort_values(["date", "sample_id"]).reset_index(drop=True)
    s["rule"] = westgard(s["z"]) if len(s) else ""
    return s


def check_bias(ev: pd.DataFrame) -> dict:
    """Mean bias of check assays against the original lab, with a 95 % interval."""
    c = ev[(ev["sample_type"] == "Check assay") & ev["bias_pct"].notna()]
    n = len(c)
    if n == 0:
        return {"n": 0, "mean_bias_pct": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    m = float(c["bias_pct"].mean())
    half = 1.96 * float(c["bias_pct"].std(ddof=1)) / np.sqrt(n) if n > 1 else np.nan
    return {"n": n, "mean_bias_pct": m, "ci_low": m - half, "ci_high": m + half}


def qaqc_alerts(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ev[ev["status"].isin(["Fail"])].iterrows():
        rows.append({"date": r["date"], "severity": "warning", "type": f"QA/QC {r['sample_type']} fail",
                     "message": f"{r.get('sample_id', '')}: {r['reason']}"})
    s = standards_chart(ev)
    for _, r in s[s["rule"] != ""].iterrows():
        rows.append({"date": r["date"], "severity": "warning", "type": "QA/QC control rule",
                     "message": f"Standard {r.get('sample_id', '')}: rule {r['rule']} broken (z = {r['z']:+.1f}). "
                                "Assays may be drifting or biased."})
    return pd.DataFrame(rows, columns=["date", "severity", "type", "message"])
