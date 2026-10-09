"""Gold CIL plant: metallurgical accounting & performance dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import storage
from calc import (INPUT_COLUMNS, AlertThresholds, Costs, build_alerts, compute_daily,
                  overall, summarise)
from sample_data import generate
from validation import error_rows, validate

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SEV_LABEL = {"critical": "🔴 critical", "warning": "🟠 warning", "info": "🔵 info", "error": "🔴 error"}

st.set_page_config(page_title="Plant Performance", page_icon="⛏️", layout="wide")


# ───────────────────────── helpers ─────────────────────────
def read_upload(f) -> pd.DataFrame:
    df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def line_chart(d, series, ytitle, height=300, markers=None):
    """series = [(column, label, colour)]. One y-axis, thin lines, legend when >= 2 series."""
    fig = go.Figure()
    for col, label, colour in series:
        fig.add_trace(go.Scatter(x=d["date"], y=d[col], name=label, mode="lines",
                                 line=dict(width=2, color=colour),
                                 hovertemplate="%{y:.2f}"))
    if markers is not None and not markers.empty:
        fig.add_trace(go.Scatter(x=markers["date"], y=markers["y"], mode="markers", name="Alert day",
                                 marker=dict(size=9, color=RED, line=dict(width=2, color="white")),
                                 hovertemplate="%{y:.2f}"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                      yaxis_title=ytitle, showlegend=len(series) + (markers is not None) >= 2,
                      legend=dict(orientation="h", y=1.15, x=0))
    fig.update_xaxes(showgrid=False)
    return fig


def bar_chart(d, col, ytitle, colour=BLUE, height=280):
    fig = go.Figure(go.Bar(x=d["date"], y=d[col], marker_color=colour, name=ytitle,
                           hovertemplate="%{y:,.0f}"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=ytitle,
                      hovermode="x unified", bargap=0.35)
    fig.update_xaxes(showgrid=False)
    return fig


def rnd(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.round({c: n for c in num})


def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            out = frame.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.strftime("%Y-%m-%d")
            out.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ───────────────────────── sidebar ─────────────────────────
st.sidebar.title("⛏️ Plant settings")
plant_name = st.sidebar.text_input("Plant name", "Gold CIL Plant")

with st.sidebar.expander("Unit costs (US$)", expanded=False):
    costs = Costs(
        nacn_per_kg=st.number_input("NaCN per kg", 0.0, 50.0, 3.2, 0.1),
        lime_per_kg=st.number_input("Lime per kg", 0.0, 10.0, 0.25, 0.05),
        power_per_kwh=st.number_input("Power per kWh", 0.0, 2.0, 0.12, 0.01),
    )
with st.sidebar.expander("Alert thresholds", expanded=False):
    th = AlertThresholds(
        recovery_drop_pts=st.number_input("Recovery drop vs 7-day mean (pts)", 0.5, 20.0, 2.0, 0.5),
        unaccounted_warn_pct=st.number_input("Unaccounted gold warning (% of feed)", 0.5, 20.0, 3.0, 0.5),
        unaccounted_crit_pct=st.number_input("Unaccounted gold critical (% of feed)", 0.5, 30.0, 5.0, 0.5),
        head_gap_pct=st.number_input("Head grade mismatch (%)", 1.0, 30.0, 5.0, 0.5),
        reagent_spike_factor=st.number_input("Cyanide spike factor (× 7-day mean)", 1.05, 3.0, 1.25, 0.05),
        min_availability_pct=st.number_input("Minimum availability (%)", 50.0, 100.0, 85.0, 1.0),
    )
exclude_errors = st.sidebar.checkbox("Exclude rows with validation errors from KPIs", value=True,
                                     help="Recommended. Impossible values (e.g. tails > head) would corrupt recovery.")

st.sidebar.divider()
st.sidebar.subheader("Data")
up = st.sidebar.file_uploader("Import CSV / Excel", type=["csv", "xlsx"])
if up is not None and st.sidebar.button("Import valid rows"):
    new = read_upload(up)
    iss = validate(new)
    bad = error_rows(iss)
    if "date" not in new.columns or (not iss.empty and iss["row"].isna().any()
                                     and iss[iss["row"].isna()]["severity"].eq("error").any()):
        st.sidebar.error("File is missing required columns:\n" + "\n".join(iss["message"].head(3)))
    else:
        good = new[~new.index.isin(bad)]
        n = storage.upsert(good)
        st.sidebar.success(f"Imported {n} rows. Skipped {len(bad)} with errors.")
        st.rerun()

template = pd.DataFrame(columns=INPUT_COLUMNS).to_csv(index=False)
st.sidebar.download_button("Download blank CSV template", template, "plant_template.csv", "text/csv")

c1, c2 = st.sidebar.columns(2)
if c1.button("Load demo data"):
    storage.upsert(generate())
    st.rerun()
if c2.button("Clear all", type="secondary"):
    st.session_state["confirm_clear"] = True
if st.session_state.get("confirm_clear"):
    st.sidebar.warning("Delete ALL records? This cannot be undone.")
    if st.sidebar.button("Yes, delete everything"):
        storage.clear()
        st.session_state["confirm_clear"] = False
        st.rerun()

# ───────────────────────── data pipeline ─────────────────────────
raw = storage.load().sort_values("date", kind="stable").reset_index(drop=True)
st.title(f"{plant_name}: Metallurgical Accounting")

if raw.empty:
    st.info("No data yet. Add a day in **Daily entry**, import a CSV/Excel file, or click "
            "**Load demo data** in the sidebar.")

issues = validate(raw) if not raw.empty else pd.DataFrame(columns=["row", "date", "field", "severity", "message"])
bad_rows = error_rows(issues)
kept = raw[~raw.index.isin(bad_rows)] if exclude_errors else raw
daily_all = compute_daily(kept.reset_index(drop=True), costs) if not kept.empty else pd.DataFrame()

if not daily_all.empty:
    lo, hi = daily_all["date"].min().date(), daily_all["date"].max().date()
    rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi))
    d = daily_all[(daily_all["date"].dt.date >= start) & (daily_all["date"].dt.date <= end)].reset_index(drop=True)
    alerts_all = build_alerts(daily_all, th)
    alerts = alerts_all[(alerts_all["date"].dt.date >= start) & (alerts_all["date"].dt.date <= end)]
else:
    d, alerts = pd.DataFrame(), pd.DataFrame(columns=["date", "severity", "type", "message"])

tab_dash, tab_entry, tab_rec, tab_reag, tab_alerts, tab_dq, tab_export = st.tabs(
    ["📊 Dashboard", "✍️ Daily entry", "⚖️ Reconciliation", "🧪 Reagents & energy",
     "🚨 Alerts", "✅ Data quality", "📤 Export"])

# ───────────────────────── dashboard ─────────────────────────
with tab_dash:
    if d.empty:
        st.write("Nothing to show yet.")
    else:
        s = overall(d)
        last7 = overall(d.tail(7))
        k = st.columns(6)
        k[0].metric("Tonnes milled", f"{s['tonnes_milled']:,.0f}")
        k[1].metric("Head grade", f"{s['head_grade_gpt']:.2f} g/t")
        k[2].metric("Recovery (grade)", f"{s['recovery_grade_pct']:.1f}%",
                    f"{last7['recovery_grade_pct'] - s['recovery_grade_pct']:+.1f} pts last 7 d")
        k[3].metric("Gold recovered", f"{s['gold_recovered_oz']:,.0f} oz")
        k[4].metric("Unaccounted", f"{s['unaccounted_pct']:.1f}%", f"{s['unaccounted_g']:,.0f} g",
                    delta_color="inverse")
        k[5].metric("Availability", f"{s['availability_pct']:.1f}%")

        crit = alerts[alerts["severity"] == "critical"]
        if len(crit):
            st.error(f"{len(crit)} critical alert(s) in this period. See the Alerts tab.")

        mk = d[d["date"].isin(alerts[alerts["type"] == "Recovery drop"]["date"])][["date", "recovery_grade_pct"]]
        mk = mk.rename(columns={"recovery_grade_pct": "y"})
        d_plot = d.assign(rec_7d=d["recovery_grade_pct"].rolling(7, min_periods=3).mean())
        left, right = st.columns(2)
        with left:
            st.subheader("Recovery")
            st.plotly_chart(line_chart(d_plot, [("recovery_grade_pct", "Daily recovery", BLUE),
                                               ("rec_7d", "7-day average", ORANGE)], "Recovery (%)", markers=mk),
                            width="stretch")
        with right:
            st.subheader("Grades")
            st.plotly_chart(line_chart(d, [("head_grade_gpt", "Head grade", BLUE),
                                           ("tails_grade_gpt", "Tails grade", ORANGE)], "Grade (g/t)"),
                            width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Throughput")
            st.plotly_chart(line_chart(d, [("tph", "Tonnes per hour", BLUE)], "t/h"), width="stretch")
        with right:
            st.subheader("Gold recovered per day")
            st.plotly_chart(bar_chart(d, "gold_recovered_oz", "Gold recovered (oz)"), width="stretch")

        st.subheader("Monthly summary (tonnage-weighted)")
        m = summarise(d, "M")
        st.dataframe(m.round(2), width="stretch", hide_index=True)

# ───────────────────────── daily entry ─────────────────────────
with tab_entry:
    st.subheader("Enter or correct a day")
    st.caption("The form checks your numbers before saving. Saving a date that already exists overwrites it.")
    with st.form("entry"):
        a, b, c = st.columns(3)
        date = a.date_input("Date", pd.Timestamp.today().date())
        tonnes = b.number_input("Tonnes milled (dry)", 0.0, 100000.0, 0.0, 10.0)
        mh = c.number_input("Mill running hours", 0.0, 24.0, 0.0, 0.1)
        a, b, c = st.columns(3)
        sh = a.number_input("Scheduled hours", 0.0, 24.0, 24.0, 0.5)
        head = b.number_input("Head grade (g/t)", 0.0, 100.0, 0.0, 0.01)
        tails = c.number_input("Tails grade (g/t)", 0.0, 100.0, 0.0, 0.001, format="%.3f")
        a, b, c = st.columns(3)
        pour = a.number_input("Gold poured (g)", 0.0, 1e7, 0.0, 10.0)
        inv = b.number_input("In-circuit inventory change (g)", -1e6, 1e6, 0.0, 10.0,
                             help="Positive = gold built up in carbon/tanks (counts as recovered).")
        model = c.number_input("Model / grade-control grade (g/t), optional", 0.0, 100.0, 0.0, 0.01)
        a, b, c = st.columns(3)
        nacn = a.number_input("NaCN used (kg)", 0.0, 1e6, 0.0, 10.0)
        lime = b.number_input("Lime used (kg)", 0.0, 1e7, 0.0, 10.0)
        power = c.number_input("Power used (kWh)", 0.0, 1e8, 0.0, 100.0)
        confirm = st.checkbox("I have checked any warnings below and want to save anyway")
        submitted = st.form_submit_button("Validate & save")

    if submitted:
        row = pd.DataFrame([dict(date=pd.Timestamp(date), tonnes_milled=tonnes, mill_hours=mh,
                                 scheduled_hours=sh, head_grade_gpt=head, tails_grade_gpt=tails,
                                 gold_poured_g=pour, inventory_change_g=inv, nacn_kg=nacn,
                                 lime_kg=lime, power_kwh=power,
                                 model_grade_gpt=(model if model > 0 else None))])
        iss = validate(row)
        iss = iss[iss["field"] != "date"] if not iss.empty else iss
        errs, warns = iss[iss["severity"] == "error"], iss[iss["severity"] == "warning"]
        for msg in errs["message"]:
            st.error(msg)
        for msg in warns["message"]:
            st.warning(msg)
        if len(errs):
            st.error("Not saved. Fix the errors above.")
        elif len(warns) and not confirm:
            st.info("Tick the confirmation box to save with warnings.")
        else:
            storage.upsert(row)
            st.success(f"Saved {date}.")
            st.rerun()

    if not raw.empty:
        with st.expander("Delete a day"):
            pick = st.selectbox("Date", raw["date"].dt.strftime("%Y-%m-%d").tolist()[::-1])
            if st.button("Delete this day"):
                storage.delete(pick)
                st.rerun()

# ───────────────────────── reconciliation ─────────────────────────
with tab_rec:
    if d.empty:
        st.write("Nothing to show yet.")
    else:
        st.subheader("Where did the gold go?")
        st.caption("Feed gold = tails + recovered + unaccounted. A persistent gap points to sampling error, "
                   "unmeasured inventory, pour/refinery differences or loss.")
        s = overall(d)
        total_feed = (d["feed_gold_g"]).sum()
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Feed gold", "Tails (lost)", "Unaccounted", "Recovered"],
            y=[total_feed / 1000, -d["tails_gold_g"].sum() / 1000, -d["unaccounted_g"].sum() / 1000, 0],
            increasing=dict(marker=dict(color=BLUE)), decreasing=dict(marker=dict(color=ORANGE)),
            totals=dict(marker=dict(color=BLUE)),
            hovertemplate="%{y:,.1f} kg"))
        wf.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Gold (kg)", showlegend=False)
        st.plotly_chart(wf, width="stretch")

        left, right = st.columns(2)
        with left:
            st.subheader("Unaccounted gold per day")
            st.plotly_chart(bar_chart(d, "unaccounted_g", "Unaccounted (g)", ORANGE), width="stretch")
        with right:
            st.subheader("Assay head vs calculated head")
            st.plotly_chart(line_chart(d, [("head_grade_gpt", "Assay head", BLUE),
                                           ("calc_head_gpt", "Calculated head", ORANGE)], "Grade (g/t)"),
                            width="stretch")
        st.subheader("Recovery two ways")
        st.plotly_chart(line_chart(d, [("recovery_grade_pct", "From assays (head − tails)", BLUE),
                                       ("recovery_accounted_pct", "From gold produced", ORANGE)], "Recovery (%)"),
                        width="stretch")
        m = summarise(d, "M")
        if m["mine_call_factor"].notna().any():
            st.subheader("Mine call factor")
            st.caption("Plant feed gold ÷ model-predicted gold. Below 1.00 means the plant sees less than the model promised.")
            st.dataframe(m[["period", "mine_call_factor"]].round(3), hide_index=True)

# ───────────────────────── reagents ─────────────────────────
with tab_reag:
    if d.empty:
        st.write("Nothing to show yet.")
    else:
        s = overall(d)
        k = st.columns(4)
        k[0].metric("NaCN", f"{s['nacn_kgpt']:.3f} kg/t")
        k[1].metric("Lime", f"{s['lime_kgpt']:.2f} kg/t")
        k[2].metric("Power", f"{s['power_kwhpt']:.1f} kWh/t")
        k[3].metric("Process cost", f"${(d['opex_per_t'] * d['tonnes_milled']).sum() / d['tonnes_milled'].sum():.2f}/t")
        left, right = st.columns(2)
        with left:
            st.subheader("Cyanide")
            st.plotly_chart(line_chart(d, [("nacn_kgpt", "NaCN (kg/t)", BLUE)], "kg/t"), width="stretch")
        with right:
            st.subheader("Lime")
            st.plotly_chart(line_chart(d, [("lime_kgpt", "Lime (kg/t)", BLUE)], "kg/t"), width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Power")
            st.plotly_chart(line_chart(d, [("power_kwhpt", "Power (kWh/t)", BLUE)], "kWh/t"), width="stretch")
        with right:
            st.subheader("Process cost per ounce")
            st.plotly_chart(line_chart(d, [("process_cost_per_oz", "US$ per oz", BLUE)], "US$/oz"),
                            width="stretch")

# ───────────────────────── alerts ─────────────────────────
with tab_alerts:
    if alerts.empty:
        st.success("No alerts in this period.")
    else:
        sev = st.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning"])
        typ = st.multiselect("Type", sorted(alerts["type"].unique()), default=sorted(alerts["type"].unique()))
        a = alerts[alerts["severity"].isin(sev) & alerts["type"].isin(typ)].sort_values("date", ascending=False)
        a = a.assign(date=a["date"].dt.strftime("%Y-%m-%d"), severity=a["severity"].map(SEV_LABEL))
        st.caption(f"{len(a)} alert(s)")
        st.dataframe(a, width="stretch", hide_index=True)

# ───────────────────────── data quality ─────────────────────────
with tab_dq:
    n_err = int((issues["severity"] == "error").sum()) if not issues.empty else 0
    n_warn = int((issues["severity"] == "warning").sum()) if not issues.empty else 0
    k = st.columns(3)
    k[0].metric("Records", len(raw))
    k[1].metric("Errors", n_err)
    k[2].metric("Warnings", n_warn)
    if issues.empty:
        st.success("No data issues found.")
    else:
        if exclude_errors and bad_rows:
            st.warning(f"{len(bad_rows)} record(s) with errors are excluded from all KPIs and charts.")
        v = issues.copy()
        v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
        v["severity"] = v["severity"].map(SEV_LABEL)
        st.dataframe(v.drop(columns="row"), width="stretch", hide_index=True)

# ───────────────────────── export ─────────────────────────
with tab_export:
    if d.empty:
        st.write("Nothing to export yet.")
    else:
        st.write("Download the calculated results for management reports.")
        sheets = {"Daily": rnd(d.drop(columns=["rec_7d_prev", "nacn_7d_prev"])),
                  "Monthly": rnd(summarise(d, "M")),
                  "Alerts": alerts,
                  "Data issues": issues.drop(columns="row") if not issues.empty else issues}
        st.download_button("⬇️ Excel report (.xlsx)", to_excel(sheets), "plant_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("⬇️ Daily results (.csv)", rnd(d).to_csv(index=False), "daily_results.csv", "text/csv")
