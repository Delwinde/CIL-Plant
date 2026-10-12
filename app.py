"""Gold CIL plant: metallurgical accounting & performance app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import circuit
import goldroom
import qaqc
import report
import scanner
import sample_data
import spc
import storage
from calc import (INPUT_COLUMNS, AlertThresholds, Costs, build_alerts, by_ore_type, compute_daily, overall,
                  plan_vs_actual, summarise)
from circuit import CARBON_EVENTS, TANK_COLUMNS, CircuitLimits
from qaqc import SAMPLE_TYPES, QAQCLimits
from validation import (error_rows, validate, validate_carbon, validate_pours, validate_qaqc, validate_tanks)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, RED, GREY = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[7], "#8a8983"
SEV_LABEL = {"critical": "🔴 critical", "warning": "🟠 warning", "info": "🔵 info", "error": "🔴 error"}
STATUS_LABEL = {"On target": "✅ On target", "Slightly behind": "🟠 Slightly behind", "Behind": "🔴 Behind"}
QA_LABEL = {"Pass": "✅ Pass", "Warning": "🟠 Warning", "Fail": "🔴 Fail", "Incomplete": "⚪ Incomplete"}
PRETTY = {
    "period": "Period", "days": "Days", "tonnes_milled": "Tonnes", "mill_hours": "Mill hours", "tph": "t/h",
    "head_grade_gpt": "Head g/t", "tails_grade_gpt": "Tails g/t", "recovery_grade_pct": "Recovery % (assays)",
    "recovery_accounted_pct": "Recovery % (gold produced)", "gold_recovered_oz": "Gold oz", "unaccounted_g": "Unaccounted g",
    "unaccounted_pct": "Unaccounted %", "nacn_kgpt": "NaCN kg/t", "lime_kgpt": "Lime kg/t", "power_kwhpt": "kWh/t",
    "availability_pct": "Availability %", "solution_loss_pct_of_tails": "Dissolved loss % of tails",
    "mine_call_factor": "Mine call factor", "ore_type": "Ore type", "share_pct": "Share of tonnes %",
}

st.set_page_config(page_title="Plant Performance", page_icon="⛏️", layout="wide")


# ───────────────────────── helpers ─────────────────────────
def read_upload(f) -> pd.DataFrame:
    df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def rnd(df: pd.DataFrame, nd: int = 3) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.round({c: nd for c in num})


def pretty(df: pd.DataFrame, nd: int = 2) -> pd.DataFrame:
    return rnd(df, nd).rename(columns=PRETTY)


def day_str(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def style_fig(fig, height=300, ytitle=None, legend=True):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                      yaxis_title=ytitle, showlegend=legend, legend=dict(orientation="h", y=1.15, x=0))
    fig.update_xaxes(showgrid=False)
    return fig


def add_hlines(fig, hlines, position="top left"):
    for y, label in (hlines or []):
        if y is not None and not pd.isna(y):
            fig.add_hline(y=y, line=dict(width=1, dash="dash", color=GREY), annotation_text=label,
                          annotation_position=position, annotation_font_size=10)


def line_chart(d, series, ytitle, height=300, markers=None, hlines=None, x="date"):
    """series = [(column, label, colour)]. One y-axis, thin lines, legend when >= 2 series."""
    fig = go.Figure()
    for col, label, colour in series:
        fig.add_trace(go.Scatter(x=d[x], y=d[col], name=label, mode="lines", line=dict(width=2, color=colour),
                                 hovertemplate="%{y:.2f}"))
    if markers is not None and not markers.empty:
        fig.add_trace(go.Scatter(x=markers["date"], y=markers["y"], mode="markers", name="Alert day",
                                 marker=dict(size=9, color=RED, line=dict(width=2, color="white")),
                                 hovertemplate="%{y:.2f}"))
    add_hlines(fig, hlines)
    return style_fig(fig, height, ytitle, legend=len(series) + (markers is not None) >= 2)


def bar_chart(d, col, ytitle, colour=BLUE, height=280, x="date", fmt="%{y:,.0f}"):
    fig = go.Figure(go.Bar(x=d[x], y=d[col], marker_color=colour, name=ytitle, hovertemplate=fmt))
    fig.update_layout(bargap=0.35)
    return style_fig(fig, height, ytitle, legend=False)


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


def empty_note(msg="Nothing to show yet."):
    st.info(msg)


def in_range(df, start, end, col="date"):
    if df is None or df.empty:
        return df
    dts = pd.to_datetime(df[col]).dt.date
    return df[(dts >= start) & (dts <= end)]


def numfield(label, value, lo, hi, step, key, fmt=None, help=None):
    cast = int if fmt and fmt.endswith("d") else float      # whole-number fields must not mix ints and floats
    return st.number_input(label, cast(lo), cast(hi), cast(value), cast(step), key=key, format=fmt, help=help)


# ───────────────────────── settings & objects ─────────────────────────
S = storage.get_settings()
costs = Costs(**S["costs"])
th = AlertThresholds(**S["thresholds"])
C = S["circuit"]
clim = CircuitLimits(**{k: v for k, v in C.items() if k != "tank_volumes_m3"})
qlim = QAQCLimits(**S["qaqc"])
targets = S["targets"]
tank_vols = [float(v) for v in C["tank_volumes_m3"]]
n_tanks = len(tank_vols)
baseline_days = int(S["spc"]["baseline_days"])
plant_name = S["plant_name"]

# ───────────────────────── sidebar ─────────────────────────
st.sidebar.title(f"⛏️ {plant_name}")
exclude_errors = st.sidebar.checkbox("Exclude rows with validation errors from KPIs", value=True,
                                     help="Recommended. Impossible values (e.g. tails > head) would corrupt recovery.")
st.sidebar.divider()
st.sidebar.subheader("Data")
up = st.sidebar.file_uploader("Import daily records (CSV / Excel)", type=["csv", "xlsx"])
if up is not None and st.sidebar.button("Import valid rows"):
    new = read_upload(up)
    iss = validate(new)
    bad = error_rows(iss)
    missing_cols = (not iss.empty) and iss["row"].isna().any() and iss[iss["row"].isna()]["severity"].eq("error").any()
    if "date" not in new.columns or missing_cols:
        st.sidebar.error("File is missing required columns:\n" + "\n".join(iss["message"].head(3)))
    else:
        n_saved = storage.upsert(new[~new.index.isin(bad)])
        st.sidebar.success(f"Imported {n_saved} rows. Skipped {len(bad)} with errors.")
        st.rerun()

st.sidebar.download_button("Download blank CSV template", pd.DataFrame(columns=INPUT_COLUMNS).to_csv(index=False),
                           "plant_template.csv", "text/csv")

# ───────────────────────── data pipeline ─────────────────────────
raw = storage.load().sort_values("date", kind="stable").reset_index(drop=True)
tanks_raw = storage.load_table("tanks")
carbon_log = storage.load_table("carbon_log")
qa_raw = storage.load_table("qaqc")
pours_raw = storage.load_table("pours")

if raw.empty:
    raw_inv, inv_notes = raw, pd.DataFrame(columns=["date", "severity", "message"])
else:
    raw_inv, inv_notes = circuit.apply_inventory(raw, tanks_raw, clim, n_tanks)

EMPTY_ISSUES = pd.DataFrame(columns=["row", "date", "field", "severity", "message"])
issues = validate(raw_inv) if not raw_inv.empty else EMPTY_ISSUES
bad_rows = error_rows(issues)
kept = raw_inv[~raw_inv.index.isin(bad_rows)] if exclude_errors else raw_inv

tcalc = circuit.tank_calcs(tanks_raw, clim) if not tanks_raw.empty else tanks_raw
if kept.empty:
    daily_all = pd.DataFrame()
else:
    daily_all = compute_daily(kept.reset_index(drop=True), costs)
    daily_all = circuit.add_residence_time(daily_all, tcalc, tank_vols, clim)

ev_all = qaqc.evaluate(qa_raw, qlim) if not qa_raw.empty else pd.DataFrame()

if not daily_all.empty:
    lo, hi = daily_all["date"].min().date(), daily_all["date"].max().date()
    rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi))
    d = daily_all[(daily_all["date"].dt.date >= start) & (daily_all["date"].dt.date <= end)].reset_index(drop=True)
else:
    lo = hi = start = end = pd.Timestamp.today().date()
    d = pd.DataFrame()

parts = [build_alerts(daily_all, th)] if not daily_all.empty else []
if not tcalc.empty:
    parts.append(circuit.circuit_alerts(tcalc, clim))
parts.append(circuit.carbon_alerts(carbon_log, clim))
if not ev_all.empty:
    parts.append(qaqc.qaqc_alerts(ev_all))
if not daily_all.empty:
    parts.append(spc.spc_alerts(daily_all, baseline_days))
parts = [p for p in parts if p is not None and not p.empty]
alerts_all = (pd.concat(parts, ignore_index=True).sort_values("date") if parts
              else pd.DataFrame(columns=["date", "severity", "type", "message"]))
alerts = in_range(alerts_all, start, end) if not alerts_all.empty else alerts_all

st.sidebar.divider()
c1, c2 = st.sidebar.columns(2)
has_data = not raw.empty or not tanks_raw.empty or not qa_raw.empty or not pours_raw.empty or not storage.load_table("scans").empty
if c1.button("Load demo data"):
    if has_data:
        st.session_state["confirm_demo"] = True
    else:
        for t, df_ in sample_data.generate_all().items():
            storage.save_rows(t, df_)
        st.rerun()
if c2.button("Clear all", type="secondary"):
    st.session_state["confirm_clear"] = True
if st.session_state.get("confirm_demo"):
    st.sidebar.warning("Loading demo data replaces ALL your current records. Back them up first (Reports & backup tab).")
    if st.sidebar.button("Yes, replace with demo data"):
        storage.clear_all()
        for t, df_ in sample_data.generate_all().items():
            storage.save_rows(t, df_)
        st.session_state["confirm_demo"] = False
        st.rerun()
if st.session_state.get("confirm_clear"):
    st.sidebar.warning("Delete ALL records? This cannot be undone. Back up first.")
    if st.sidebar.button("Yes, delete everything"):
        storage.clear_all()
        st.session_state["confirm_clear"] = False
        st.rerun()

# ───────────────────────── page ─────────────────────────
st.title(f"{plant_name}: Metallurgical Accounting")
if raw.empty:
    st.info("No data yet. Add a day in **Daily entry**, import a CSV/Excel file, restore a backup, or click "
            "**Load demo data** in the sidebar.")

(tab_dash, tab_entry, tab_circ, tab_rec, tab_reag, tab_ore, tab_scan, tab_qa, tab_gold, tab_spc, tab_alerts, tab_dq,
 tab_rep, tab_set) = st.tabs(["📊 Dashboard", "✍️ Daily entry", "🧪 Circuit & carbon", "⚖️ Reconciliation",
                              "⚡ Reagents", "🪨 Ore types", "📷 Ore scanner", "🔬 Lab QA/QC", "🪙 Gold room",
                              "📈 Control charts", "🚨 Alerts", "✅ Data quality", "📄 Reports", "⚙️ Settings"])

# ───────────────────────── dashboard ─────────────────────────
with tab_dash:
    if d.empty:
        empty_note()
    else:
        s = overall(d)
        last7 = overall(d.tail(7))
        k = st.columns(6)
        k[0].metric("Tonnes milled", f"{s['tonnes_milled']:,.0f}")
        k[1].metric("Head grade", f"{s['head_grade_gpt']:.2f} g/t")
        k[2].metric("Recovery (grade)", f"{s['recovery_grade_pct']:.1f}%",
                    f"{last7['recovery_grade_pct'] - s['recovery_grade_pct']:+.1f} pts last 7 d")
        k[3].metric("Gold recovered", f"{s['gold_recovered_oz']:,.0f} oz")
        k[4].metric("Unaccounted", f"{s['unaccounted_pct']:.1f}%", f"{s['unaccounted_g']:,.0f} g", delta_color="inverse")
        k[5].metric("Availability", f"{s['availability_pct']:.1f}%")

        crit = alerts[alerts["severity"] == "critical"] if not alerts.empty else alerts
        if len(crit):
            st.error(f"{len(crit)} critical alert(s) in this period. See the Alerts tab.")

        plan = plan_vs_actual(s, targets)
        if not plan.empty:
            with st.expander("Performance against targets", expanded=True):
                show = plan.assign(status=plan["status"].map(STATUS_LABEL))
                st.dataframe(rnd(show, 2).rename(columns={"metric": "Measure", "target": "Target", "actual": "Actual",
                                                          "variance": "Variance", "status": "Status"}),
                             width="stretch", hide_index=True)
                st.caption("Targets are set in the Settings tab.")

        mk = d[d["date"].isin(alerts[alerts["type"] == "Recovery drop"]["date"])][["date", "recovery_grade_pct"]] \
            if not alerts.empty else pd.DataFrame()
        mk = mk.rename(columns={"recovery_grade_pct": "y"})
        d_plot = d.assign(rec_7d=d["recovery_grade_pct"].rolling(7, min_periods=3).mean())
        left, right = st.columns(2)
        with left:
            st.subheader("Recovery")
            st.plotly_chart(line_chart(d_plot, [("recovery_grade_pct", "Daily recovery", BLUE),
                                               ("rec_7d", "7-day average", ORANGE)], "Recovery (%)", markers=mk,
                                       hlines=[(targets.get("recovery_pct"), "Target")]), width="stretch")
        with right:
            st.subheader("Grades")
            st.plotly_chart(line_chart(d, [("head_grade_gpt", "Head grade", BLUE), ("tails_grade_gpt", "Tails grade", ORANGE)],
                                       "Grade (g/t)"), width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Throughput")
            st.plotly_chart(line_chart(d, [("tph", "Tonnes per hour", BLUE)], "t/h",
                                       hlines=[(targets.get("throughput_tph"), "Target")]), width="stretch")
        with right:
            st.subheader("Gold recovered per day")
            st.plotly_chart(bar_chart(d, "gold_recovered_oz", "Gold recovered (oz)"), width="stretch")

        st.subheader("Monthly summary (tonnage-weighted)")
        cols = ["period", "days", "tonnes_milled", "head_grade_gpt", "tails_grade_gpt", "recovery_grade_pct",
                "gold_recovered_oz", "unaccounted_pct", "nacn_kgpt", "availability_pct"]
        st.dataframe(pretty(summarise(d, "M")[cols]), width="stretch", hide_index=True)

# ───────────────────────── daily entry ─────────────────────────
with tab_entry:
    st.subheader("Enter or correct a day")
    st.caption("The form checks your numbers before saving. Saving a date that already exists overwrites it. "
               "Leave optional fields at 0 if you did not measure them.")
    existing_ores = sorted({str(o) for o in raw.get("ore_type", pd.Series(dtype=object)).dropna() if str(o).strip()}
                           | {"Oxide", "Transition", "Fresh"})
    with st.form("entry"):
        a, b, c = st.columns(3)
        date = a.date_input("Date", pd.Timestamp.today().date())
        tonnes = b.number_input("Tonnes milled (dry)", 0.0, 100000.0, 0.0, 10.0)
        mh = c.number_input("Mill running hours", 0.0, 24.0, 0.0, 0.1)
        a, b, c = st.columns(3)
        sh = a.number_input("Scheduled hours", 0.0, 24.0, 24.0, 0.5)
        head = b.number_input("Head grade (g/t)", 0.0, 100.0, 0.0, 0.01)
        tails = c.number_input("Tails grade, total (g/t)", 0.0, 100.0, 0.0, 0.001, format="%.3f",
                               help="Leave at 0 if you fill in the solid and solution tails below.")
        with st.expander("Tails detail: solid and solution (optional, recommended)"):
            st.caption("Total tails = solid tails + gold dissolved in the water. This shows whether losses are "
                       "undissolved gold (leach problem) or dissolved gold (adsorption problem).")
            a, b, c = st.columns(3)
            t_solid = a.number_input("Solid tails (g/t)", 0.0, 100.0, 0.0, 0.001, format="%.3f")
            t_sol = b.number_input("Solution tails (mg/L)", 0.0, 100.0, 0.0, 0.001, format="%.3f")
            t_ps = c.number_input("Residue slurry % solids", 0.0, 100.0, 0.0, 0.5)
        a, b, c = st.columns(3)
        pour = a.number_input("Fine gold poured (g)", 0.0, 1e7, 0.0, 10.0,
                              help="Gross bar weight x gold fineness. Not the doré weight.")
        inv = b.number_input("Typed inventory change (g)", -1e6, 1e6, 0.0, 10.0,
                             help="Only used when tank readings are missing for today or yesterday. Positive = gold built up in carbon/tanks.")
        other_inv = c.number_input("Gold outside the tanks (g)", 0.0, 1e7, 0.0, 10.0,
                                   help="Elution column, cells and kiln carbon. Added to the tank inventory.")
        a, b, c = st.columns(3)
        nacn = a.number_input("NaCN used (kg)", 0.0, 1e6, 0.0, 10.0)
        lime = b.number_input("Lime used (kg)", 0.0, 1e7, 0.0, 10.0)
        power = c.number_input("Power used (kWh)", 0.0, 1e8, 0.0, 100.0)
        a, b, c = st.columns(3)
        model = a.number_input("Model grade (g/t), optional", 0.0, 100.0, 0.0, 0.01)
        ore_pick = b.selectbox("Ore type", ["(not recorded)"] + existing_ores)
        ore_new = c.text_input("…or a new ore type")
        confirm = st.checkbox("I have checked any warnings below and want to save anyway")
        submitted = st.form_submit_button("Validate & save")

    if submitted:
        ore_val = ore_new.strip() or (None if ore_pick == "(not recorded)" else ore_pick)
        row = pd.DataFrame([dict(
            date=pd.Timestamp(date), tonnes_milled=tonnes, mill_hours=mh, scheduled_hours=sh, head_grade_gpt=head,
            tails_grade_gpt=(None if t_solid > 0 else tails), gold_poured_g=pour, inventory_change_g=inv,
            nacn_kg=nacn, lime_kg=lime, power_kwh=power, model_grade_gpt=(model if model > 0 else None),
            tails_solid_gpt=(t_solid if t_solid > 0 else None), tails_solution_mgl=(t_sol if t_solid > 0 else None),
            tails_pct_solids=(t_ps if t_ps > 0 else None), other_inventory_g=(other_inv if other_inv > 0 else None),
            ore_type=ore_val)])
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
            st.success(f"Saved {date}. Next: enter the tank readings in the Circuit & carbon tab.")
            st.rerun()

    if not raw.empty:
        with st.expander("Delete a day"):
            pick = st.selectbox("Date", raw["date"].dt.strftime("%Y-%m-%d").tolist()[::-1], key="del_day")
            st.caption("This also deletes that day's tank readings.")
            if st.button("Delete this day"):
                storage.delete(pick)
                st.rerun()

# ───────────────────────── circuit & carbon ─────────────────────────
with tab_circ:
    sub_entry, sub_prof, sub_inv, sub_carbon = st.tabs(["Enter tank readings", "Leach profile", "Inventory & residence time",
                                                        "Carbon management"])

    # ---- enter tank readings
    with sub_entry:
        st.caption(f"One row per tank ({n_tanks} tanks, set in Settings). When every tank has carbon concentration and "
                   "loading for two days in a row, the in-circuit inventory change is worked out automatically and "
                   "replaces the typed value. Leave a row blank if that tank was not sampled.")
        default_date = (raw["date"].max().date() if not raw.empty else pd.Timestamp.today().date())
        sel = st.date_input("Date of readings", default_date, key="tank_date")
        have = tanks_raw[tanks_raw["date"] == pd.Timestamp(sel)] if not tanks_raw.empty else tanks_raw
        if len(have):
            base = have[TANK_COLUMNS].sort_values("tank").reset_index(drop=True)
        else:
            base = pd.DataFrame({"tank": range(1, n_tanks + 1), "volume_m3": tank_vols})
            for c_ in TANK_COLUMNS[2:]:
                base[c_] = np.nan
        edited = st.data_editor(
            base, hide_index=True, num_rows="fixed", width="stretch", key=f"tank_editor_{sel}_{n_tanks}_{len(have)}",
            disabled=["tank"],
            column_config={
                "tank": st.column_config.NumberColumn("Tank", format="%d", width="small"),
                "volume_m3": st.column_config.NumberColumn("Vol (m³)", min_value=0, format="%.0f", width="small",
                                                           help="Working volume of the tank"),
                "pct_solids": st.column_config.NumberColumn("% solids", min_value=0, max_value=100, format="%.1f", width="small",
                                                            help="Slurry density, % solids by mass"),
                "free_cn_ppm": st.column_config.NumberColumn("Free CN", min_value=0, format="%.0f", width="small",
                                                             help="Free cyanide in ppm"),
                "ph": st.column_config.NumberColumn("pH", min_value=0, max_value=14, format="%.2f", width="small"),
                "do_ppm": st.column_config.NumberColumn("DO", min_value=0, format="%.1f", width="small",
                                                        help="Dissolved oxygen in mg/L"),
                "carbon_conc_gpl": st.column_config.NumberColumn("Carbon g/L", min_value=0, format="%.2f", width="small",
                                                                 help="Carbon concentration, grams of carbon per litre of slurry"),
                "carbon_loading_gpt": st.column_config.NumberColumn("Loading g/t", min_value=0, format="%.0f", width="small",
                                                                    help="Grams of gold per tonne of carbon"),
                "solution_au_mgl": st.column_config.NumberColumn("Solution Au", min_value=0, format="%.3f", width="small",
                                                                 help="Gold in the solution, mg/L"),
            })
        if st.button("Save tank readings"):
            reading_cols = ["free_cn_ppm", "ph", "do_ppm", "carbon_conc_gpl", "carbon_loading_gpt", "solution_au_mgl"]
            rows_ = edited[edited[reading_cols].notna().any(axis=1)].copy()
            rows_["date"] = pd.Timestamp(sel)
            if rows_.empty:
                st.warning("Nothing to save: no readings were entered.")
            else:
                iss_t = validate_tanks(rows_.reset_index(drop=True))
                errs = iss_t[iss_t["severity"] == "error"]
                for m_ in errs["message"]:
                    st.error(m_)
                if len(errs):
                    st.error("Not saved. Fix the errors above.")
                else:
                    storage.replace_day("tanks", day_str(sel), rows_)
                    for m_ in iss_t[iss_t["severity"] == "warning"]["message"]:
                        st.warning("Saved with a warning: " + m_)
                    done = rows_["carbon_loading_gpt"].notna().sum()
                    if done < n_tanks:
                        st.info(f"Saved {len(rows_)} tank rows. Inventory needs carbon figures for all {n_tanks} tanks.")
                    else:
                        st.success(f"Saved {len(rows_)} tank rows for {sel}.")
                    st.rerun()

    # ---- leach profile
    with sub_prof:
        if tcalc.empty:
            empty_note("No tank readings yet. Enter them in the first sub-tab.")
        else:
            dates_avail = sorted(tcalc["date"].dt.date.unique(), reverse=True)
            pick = st.selectbox("Day", dates_avail, key="prof_day")
            day_t = tcalc[tcalc["date"].dt.date == pick].sort_values("tank")
            labels = [f"Tank {int(t)}" for t in day_t["tank"]]

            def tank_bar(col, title, limit=None, limit_label="Limit", colour=BLUE, fmt="%{y:.2f}"):
                """A profile across the tanks: points joined by a line, with the limit kept in view."""
                y = day_t[col]
                fig = go.Figure(go.Scatter(x=labels, y=y, mode="lines+markers", name=title, hovertemplate=fmt,
                                           line=dict(width=2, color=colour), marker=dict(size=8, color=colour)))
                if limit is not None:
                    add_hlines(fig, [(limit, limit_label)], position="bottom right")
                    vals = [v for v in y.dropna()] + [limit]
                    pad = (max(vals) - min(vals)) * 0.2 or 1.0
                    fig.update_yaxes(range=[min(vals) - pad, max(vals) + pad])
                return style_fig(fig, 250, None, legend=False)

            r1 = st.columns(3)
            with r1[0]:
                st.markdown("**pH**")
                st.plotly_chart(tank_bar("ph", "pH", clim.min_ph, "Minimum"), width="stretch")
            with r1[1]:
                st.markdown("**Dissolved oxygen (mg/L)**")
                st.plotly_chart(tank_bar("do_ppm", "DO", clim.min_do_ppm, "Minimum", fmt="%{y:.1f}"), width="stretch")
            with r1[2]:
                st.markdown("**Free cyanide (ppm)**")
                st.plotly_chart(tank_bar("free_cn_ppm", "Free CN", clim.min_free_cn_ppm, "Minimum", fmt="%{y:.0f}"), width="stretch")
            r2 = st.columns(3)
            with r2[0]:
                st.markdown("**Carbon concentration (g/L)**")
                st.plotly_chart(tank_bar("carbon_conc_gpl", "Carbon", clim.carbon_conc_min_gpl, "Minimum", fmt="%{y:.1f}"), width="stretch")
            with r2[1]:
                st.markdown("**Carbon loading (g Au/t carbon)**")
                st.plotly_chart(tank_bar("carbon_loading_gpt", "Loading", fmt="%{y:,.0f}"), width="stretch")
            with r2[2]:
                st.markdown("**Solution gold (mg/L)**")
                st.plotly_chart(tank_bar("solution_au_mgl", "Solution Au", colour=ORANGE, fmt="%{y:.3f}"), width="stretch")
            st.caption("A healthy CIL train shows loading falling and solution gold dropping steeply from the first tank to the last. "
                       "A flat solution profile means the carbon is not adsorbing well.")
            tbl = day_t[["tank", "carbon_kg", "gold_on_carbon_g", "gold_in_solution_g", "inventory_g", "adsorption_eff_pct"]].copy()
            st.dataframe(rnd(tbl, 1).rename(columns={
                "tank": "Tank", "carbon_kg": "Carbon (kg)", "gold_on_carbon_g": "Gold on carbon (g)",
                "gold_in_solution_g": "Gold in solution (g)", "inventory_g": "Tank inventory (g)",
                "adsorption_eff_pct": "Adsorption efficiency vs previous tank (%)"}), width="stretch", hide_index=True)

            st.markdown("**Trend by tank**")
            mcol, rcol = st.columns([2, 3])
            measure = mcol.selectbox("Measure", ["ph", "do_ppm", "free_cn_ppm", "carbon_conc_gpl", "carbon_loading_gpt",
                                                 "solution_au_mgl"],
                                     format_func=lambda m: {"ph": "pH", "do_ppm": "Dissolved oxygen (mg/L)",
                                                            "free_cn_ppm": "Free cyanide (ppm)", "carbon_conc_gpl": "Carbon (g/L)",
                                                            "carbon_loading_gpt": "Carbon loading (g/t)",
                                                            "solution_au_mgl": "Solution gold (mg/L)"}[m])
            tr = in_range(tcalc, start, end)
            fig = go.Figure()
            for i_, (tk, g_) in enumerate(tr.groupby("tank")):
                fig.add_trace(go.Scatter(x=g_["date"], y=g_[measure], mode="lines", name=f"Tank {int(tk)}",
                                         line=dict(width=1.8, color=PALETTE[i_ % len(PALETTE)])))
            lim_map = {"ph": (clim.min_ph, "Minimum"), "do_ppm": (clim.min_do_ppm, "Minimum"),
                       "free_cn_ppm": (clim.min_free_cn_ppm, "Minimum"), "carbon_conc_gpl": (clim.carbon_conc_min_gpl, "Minimum")}
            if measure in lim_map:
                add_hlines(fig, [lim_map[measure]])
            st.plotly_chart(style_fig(fig, 320, None), width="stretch")

    # ---- inventory & residence time
    with sub_inv:
        if d.empty:
            empty_note()
        else:
            src = d["inventory_source"].value_counts() if "inventory_source" in d else pd.Series(dtype=int)
            m = st.columns(4)
            m[0].metric("Days with calculated inventory", int(src.get("auto", 0)))
            m[1].metric("Days using typed value", int(src.get("manual", 0)))
            m[2].metric("Days with no inventory data", int(src.get("none", 0)))
            if d["residence_time_h"].notna().any():
                m[3].metric("Mean residence time", f"{d['residence_time_h'].mean():.1f} h")
            inv_d = d[d["circuit_inventory_g"].notna()].assign(inventory_kg=lambda x: x["circuit_inventory_g"] / 1000)
            left, right = st.columns(2)
            with left:
                st.subheader("In-circuit gold inventory")
                if inv_d.empty:
                    empty_note("Enter carbon readings for all tanks to see the inventory.")
                else:
                    st.plotly_chart(line_chart(inv_d, [("inventory_kg", "Inventory (kg Au)", BLUE)], "kg gold"), width="stretch")
            with right:
                st.subheader("Residence time")
                st.plotly_chart(line_chart(d, [("residence_time_h", "CIL residence time (h)", BLUE)], "hours"), width="stretch")
            st.caption("Inventory = gold on carbon + gold in solution (all tanks) + gold outside the tanks. "
                       "Residence time = total tank volume ÷ slurry flow. Longer residence gives more leaching time; "
                       "it falls when throughput rises or a tank is out of service.")
            if not inv_notes.empty:
                st.subheader("Days where inventory could not be calculated")
                n_ = inv_notes.assign(date=inv_notes["date"].dt.strftime("%Y-%m-%d"), severity=inv_notes["severity"].map(SEV_LABEL))
                st.dataframe(n_, width="stretch", hide_index=True)

    # ---- carbon management
    with sub_carbon:
        log_sel = in_range(carbon_log, start, end) if not carbon_log.empty else carbon_log
        tonnes_sel = float(d["tonnes_milled"].sum()) if not d.empty else 0.0
        cs = circuit.carbon_stats(log_sel, tonnes_sel, end, clim)
        m = st.columns(5)
        m[0].metric("Carbon added", f"{cs['carbon_added_kg']:,.0f} kg")
        m[1].metric("Carbon consumption", "–" if pd.isna(cs["carbon_consumption_gpt"]) else f"{cs['carbon_consumption_gpt']:.1f} g/t ore")
        m[2].metric("Elutions / regenerations", f"{cs['elutions']} / {cs['regenerations']}")
        m[3].metric("Days since last elution", "–" if pd.isna(cs["days_since_elution"]) else f"{cs['days_since_elution']:.0f}")
        m[4].metric("Latest activity", "–" if pd.isna(cs["latest_activity_pct"]) else f"{cs['latest_activity_pct']:.0f}% of fresh")
        act = log_sel[(log_sel["event"] == "Activity test") & log_sel["activity_pct"].notna()] if not log_sel.empty else log_sel
        if not act.empty:
            st.subheader("Carbon activity")
            st.plotly_chart(line_chart(act, [("activity_pct", "Activity (% of fresh carbon)", BLUE)], "% of fresh",
                                       height=260, hlines=[(clim.min_activity_pct, "Minimum")]), width="stretch")
            st.caption("Fouled carbon (oils, scale, organics) loses activity. When it drops below the limit, regenerate or replace carbon; "
                       "otherwise dissolved gold escapes in the tails solution.")
        st.subheader("Log a carbon event")
        with st.form("carbon_form", clear_on_submit=True):
            a, b, c = st.columns(3)
            c_date = a.date_input("Date", pd.Timestamp.today().date(), key="c_date")
            c_event = b.selectbox("Event", CARBON_EVENTS)
            c_kg = c.number_input("Quantity (kg of carbon)", 0.0, 1e6, 0.0, 10.0)
            a, b = st.columns([1, 2])
            c_act = a.number_input("Activity test result (% of fresh)", 0.0, 150.0, 0.0, 1.0)
            c_note = b.text_input("Note")
            if st.form_submit_button("Save event"):
                row = pd.DataFrame([dict(date=pd.Timestamp(c_date), event=c_event, kg=(c_kg if c_kg > 0 else None),
                                         activity_pct=(c_act if c_act > 0 else None), note=c_note)])
                iss_c = validate_carbon(row)
                if len(iss_c):
                    for m_ in iss_c["message"]:
                        st.error(m_)
                else:
                    storage.save_rows("carbon_log", row)
                    st.rerun()
        if not carbon_log.empty:
            shown = carbon_log.sort_values("date", ascending=False).head(60)
            st.dataframe(shown.assign(date=shown["date"].dt.strftime("%Y-%m-%d")).rename(columns={
                "id": "ID", "date": "Date", "event": "Event", "kg": "kg", "activity_pct": "Activity %", "note": "Note"}),
                width="stretch", hide_index=True)
            with st.expander("Delete a carbon event"):
                del_id = st.selectbox("Event", shown["id"].tolist(), key="del_carbon",
                                      format_func=lambda i_: f"{i_}: {day_str(shown.set_index('id').loc[i_, 'date'])} "
                                                            f"{shown.set_index('id').loc[i_, 'event']}")
                if st.button("Delete event"):
                    storage.delete_ids("carbon_log", [del_id])
                    st.rerun()

# ───────────────────────── reconciliation ─────────────────────────
with tab_rec:
    if d.empty:
        empty_note()
    else:
        st.subheader("Where did the gold go?")
        st.caption("Feed gold = tails + recovered + unaccounted. A persistent gap points to sampling error, "
                   "unmeasured inventory, pour/refinery differences or loss.")
        total_feed = d["feed_gold_g"].sum()
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=["Feed gold", "Tails (lost)", "Unaccounted", "Recovered"],
            y=[total_feed / 1000, -d["tails_gold_g"].sum() / 1000, -d["unaccounted_g"].sum() / 1000, 0],
            increasing=dict(marker=dict(color=BLUE)), decreasing=dict(marker=dict(color=ORANGE)),
            totals=dict(marker=dict(color=BLUE)), hovertemplate="%{y:,.1f} kg"))
        wf.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Gold (kg)", showlegend=False)
        st.plotly_chart(wf, width="stretch")

        if "inventory_source" in d:
            share_auto = (d["inventory_source"] == "auto").mean() * 100
            if share_auto < 50:
                st.warning(f"Only {share_auto:.0f}% of days use a calculated in-circuit inventory. Typed or missing inventory "
                           "shows up here as unaccounted gold. Enter the tank readings in Circuit & carbon.")
        left, right = st.columns(2)
        with left:
            st.subheader("Unaccounted gold per day")
            st.plotly_chart(bar_chart(d, "unaccounted_g", "Unaccounted (g)", ORANGE), width="stretch")
        with right:
            st.subheader("Assay head vs calculated head")
            st.plotly_chart(line_chart(d, [("head_grade_gpt", "Assay head", BLUE), ("calc_head_gpt", "Calculated head", ORANGE)],
                                       "Grade (g/t)"), width="stretch")
        st.subheader("Recovery two ways")
        st.plotly_chart(line_chart(d, [("recovery_grade_pct", "From assays (head − tails)", BLUE),
                                       ("recovery_accounted_pct", "From gold produced", ORANGE)], "Recovery (%)"), width="stretch")

        split = d[d["solution_loss_gpt"].notna()].assign(solid_part=lambda x: x["tails_grade_gpt"] - x["solution_loss_gpt"])
        if not split.empty:
            st.subheader("Tails: solid gold vs dissolved gold")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=split["date"], y=split["solid_part"], name="Solid tails", marker_color=BLUE,
                                 hovertemplate="%{y:.3f}"))
            fig.add_trace(go.Bar(x=split["date"], y=split["solution_loss_gpt"], name="Dissolved in solution", marker_color=ORANGE,
                                 hovertemplate="%{y:.3f}"))
            fig.update_layout(barmode="stack", bargap=0.3)
            st.plotly_chart(style_fig(fig, 280, "Tails grade (g/t)"), width="stretch")
            st.caption("Solid tails = gold that never dissolved (grind, leach time, cyanide, oxygen). "
                       "Dissolved tails = gold that dissolved but was not adsorbed (carbon activity, carbon concentration, tank volume).")
            st.plotly_chart(line_chart(split, [("solution_loss_pct_of_tails", "Dissolved gold, % of tails gold", ORANGE)],
                                       "% of tails gold", height=240, hlines=[(th.solution_loss_pct, "Alert level")]), width="stretch")
        m = summarise(d, "M")
        if m["mine_call_factor"].notna().any():
            st.subheader("Mine call factor")
            st.caption("Plant feed gold ÷ model-predicted gold. Below 1.00 means the plant sees less than the model promised.")
            st.dataframe(m[["period", "mine_call_factor"]].round(3), hide_index=True)

# ───────────────────────── reagents ─────────────────────────
with tab_reag:
    if d.empty:
        empty_note()
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
            st.plotly_chart(line_chart(d, [("nacn_kgpt", "NaCN (kg/t)", BLUE)], "kg/t",
                                       hlines=[(targets.get("nacn_kgpt"), "Target")]), width="stretch")
        with right:
            st.subheader("Lime")
            st.plotly_chart(line_chart(d, [("lime_kgpt", "Lime (kg/t)", BLUE)], "kg/t"), width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Power")
            st.plotly_chart(line_chart(d, [("power_kwhpt", "Power (kWh/t)", BLUE)], "kWh/t",
                                       hlines=[(targets.get("power_kwhpt"), "Target")]), width="stretch")
        with right:
            st.subheader("Process cost per ounce")
            st.plotly_chart(line_chart(d, [("process_cost_per_oz", "US$ per oz", BLUE)], "US$/oz"), width="stretch")

# ───────────────────────── ore types ─────────────────────────
with tab_ore:
    if d.empty:
        empty_note()
    else:
        ore_tbl = by_ore_type(d)
        if ore_tbl["ore_type"].eq("Unspecified").all():
            st.info("No ore types recorded yet. Choose an ore type in Daily entry (for example Oxide, Transition, Fresh) "
                    "to see recovery by ore type and to forecast recovery.")
        else:
            st.subheader("Performance by ore type")
            cols = ["ore_type", "days", "tonnes_milled", "share_pct", "head_grade_gpt", "tails_grade_gpt", "recovery_grade_pct",
                    "nacn_kgpt", "power_kwhpt", "tph"]
            st.dataframe(pretty(ore_tbl[cols]), width="stretch", hide_index=True)
            left, right = st.columns(2)
            with left:
                plant_rec = overall(d)["recovery_grade_pct"]
                st.markdown(f"**Recovery vs the plant average of {plant_rec:.1f}% (percentage points)**")
                st.plotly_chart(bar_chart(ore_tbl.assign(diff=ore_tbl["recovery_grade_pct"] - plant_rec), "diff",
                                          "Points vs average", x="ore_type", fmt="%{y:+.2f} pts"), width="stretch")
            with right:
                st.markdown("**Weekly ore blend (tonnes)**")
                wk = d.assign(week=d["date"].dt.to_period("W").dt.start_time,
                              ore=d["ore_type"].fillna("Unspecified").astype(str)).groupby(["week", "ore"])["tonnes_milled"].sum().reset_index()
                fig = go.Figure()
                for i_, (o_, g_) in enumerate(wk.groupby("ore")):
                    fig.add_trace(go.Bar(x=g_["week"], y=g_["tonnes_milled"], name=o_, marker_color=PALETTE[i_ % len(PALETTE)]))
                fig.update_layout(barmode="stack", bargap=0.3)
                st.plotly_chart(style_fig(fig, 280, "Tonnes"), width="stretch")

            st.subheader("Recovery against head grade")
            sc = d[d["recovery_grade_pct"].notna() & d["head_grade_gpt"].notna()].assign(
                ore=lambda x: x["ore_type"].fillna("Unspecified").astype(str))
            fits = {}
            fig = go.Figure()
            for i_, (o_, g_) in enumerate(sorted(sc.groupby("ore"), key=lambda kv: kv[0])):
                colour = PALETTE[i_ % len(PALETTE)]
                fig.add_trace(go.Scatter(x=g_["head_grade_gpt"], y=g_["recovery_grade_pct"], mode="markers", name=o_,
                                         marker=dict(size=7, color=colour, line=dict(width=1, color="white")),
                                         hovertemplate="%{x:.2f} g/t → %{y:.1f}%"))
                if len(g_) >= 8 and g_["head_grade_gpt"].nunique() > 2:
                    b_, a_ = np.polyfit(g_["head_grade_gpt"], g_["recovery_grade_pct"], 1)
                    resid = g_["recovery_grade_pct"] - (a_ + b_ * g_["head_grade_gpt"])
                    fits[o_] = (a_, b_, float(resid.std(ddof=2)), len(g_))
                    xs = np.array([g_["head_grade_gpt"].min(), g_["head_grade_gpt"].max()])
                    fig.add_trace(go.Scatter(x=xs, y=a_ + b_ * xs, mode="lines", showlegend=False, line=dict(width=1.5, color=colour, dash="dot"),
                                             hoverinfo="skip"))
            fig.update_layout(height=340, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Head grade (g/t)", yaxis_title="Recovery (%)",
                              legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(fig, width="stretch")

            st.subheader("Recovery forecast")
            st.caption("A straight-line fit of recovery against head grade for the chosen ore type, from your own days. "
                       "Indicative only: it cannot see problems such as fouled carbon, and it needs at least 8 days per ore type.")
            fa, fb = st.columns(2)
            ore_choice = fa.selectbox("Ore type", sorted(sc["ore"].unique()), key="fc_ore")
            head_in = fb.number_input("Expected head grade (g/t)", 0.1, 60.0, float(round(d["head_grade_gpt"].mean(), 2)), 0.1)
            if ore_choice in fits:
                a_, b_, sd_, n_ = fits[ore_choice]
                pred = a_ + b_ * head_in
                st.metric(f"Predicted recovery for {ore_choice}", f"{pred:.1f}%",
                          f"likely range {pred - 1.96 * sd_:.1f}% to {pred + 1.96 * sd_:.1f}% · {n_} days · slope {b_:+.2f} pts per g/t")
            else:
                st.info("Not enough days with this ore type for a forecast (needs 8 and a spread of head grades).")

# ───────────────────────── lab QA/QC ─────────────────────────
with tab_qa:
    st.subheader("Add a QA/QC result")
    st.caption("Standards (certified reference material) check accuracy, blanks check contamination, duplicates check precision, "
               "and check assays compare your lab with an outside lab.")
    with st.form("qa_form", clear_on_submit=True):
        a, b, c, e = st.columns(4)
        q_date = a.date_input("Date", pd.Timestamp.today().date(), key="q_date")
        q_type = b.selectbox("Sample type", SAMPLE_TYPES)
        q_id = c.text_input("Sample ID")
        q_lab = e.text_input("Lab", "Site lab")
        a, b, c, e = st.columns(4)
        q_assay = a.number_input("Assay result (g/t)", 0.0, 1000.0, 0.0, 0.001, format="%.3f")
        q_exp = b.number_input("Standard: certified value (g/t)", 0.0, 1000.0, 0.0, 0.01)
        q_sd = c.number_input("Standard: certified SD (g/t)", 0.0, 100.0, 0.0, 0.001, format="%.3f")
        q_orig = e.number_input("Duplicate / check: original assay (g/t)", 0.0, 1000.0, 0.0, 0.001, format="%.3f")
        q_note = st.text_input("Note")
        if st.form_submit_button("Save result"):
            row = pd.DataFrame([dict(date=pd.Timestamp(q_date), sample_id=q_id or f"{q_type[:3].upper()}-{q_date:%Y%m%d}",
                                     sample_type=q_type, lab=q_lab, assay_gpt=q_assay,
                                     expected_gpt=(q_exp if q_exp > 0 else None), sd_gpt=(q_sd if q_sd > 0 else None),
                                     original_gpt=(q_orig if q_orig > 0 else None), note=q_note)])
            iss_q = validate_qaqc(row)
            if len(iss_q):
                for m_ in iss_q["message"]:
                    st.error(m_)
            else:
                storage.save_rows("qaqc", row)
                ev_new = qaqc.evaluate(row, qlim).iloc[0]
                st.session_state["qa_last"] = f"{ev_new['status']}: {ev_new['reason'] or 'within limits.'}"
                st.rerun()
    if st.session_state.get("qa_last"):
        st.info("Last result saved. " + st.session_state.pop("qa_last"))

    ev = in_range(ev_all, start, end) if not ev_all.empty else ev_all
    if ev is None or ev.empty:
        empty_note("No QA/QC results in this period.")
    else:
        sm = qaqc.summary(ev)
        st.subheader("Summary")
        sm_show = rnd(sm, 1).assign(mean_bias_pct=sm["mean_bias_pct"].map(lambda v: "–" if pd.isna(v) else f"{v:+.1f}"))
        st.dataframe(sm_show.rename(columns={"sample_type": "Sample type", "n": "Samples", "pass_pct": "Pass %",
                                             "warnings": "Warnings", "fails": "Fails", "mean_bias_pct": "Mean bias %"}),
                     width="stretch", hide_index=True)
        std = qaqc.standards_chart(ev)
        left, right = st.columns(2)
        with left:
            st.subheader("Standards control chart")
            if std.empty:
                empty_note("No standards with a certified value and SD.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=std["date"], y=std["z"], mode="lines+markers", name="z-score",
                                         line=dict(width=1.5, color=BLUE), marker=dict(size=5, color=BLUE),
                                         customdata=std[["sample_id", "rule"]], hovertemplate="z = %{y:.2f} · %{customdata[0]} %{customdata[1]}"))
                hit = std[std["rule"] != ""]
                if not hit.empty:
                    fig.add_trace(go.Scatter(x=hit["date"], y=hit["z"], mode="markers", name="Rule broken",
                                             marker=dict(size=10, color=RED, line=dict(width=2, color="white")),
                                             customdata=hit["rule"], hovertemplate="z = %{y:.2f} · %{customdata}"))
                for yv, lab_ in [(0, "Certified"), (2, "+2 SD"), (-2, "−2 SD"), (3, "+3 SD"), (-3, "−3 SD")]:
                    fig.add_hline(y=yv, line=dict(width=1, dash="solid" if yv == 0 else "dash", color=GREY),
                                  annotation_text=lab_, annotation_position="top left", annotation_font_size=9)
                st.plotly_chart(style_fig(fig, 320, "z-score (SD from certified value)"), width="stretch")
                st.caption("Rules: 1_3s one point beyond 3 SD · 2_2s two in a row beyond 2 SD · 4_1s four in a row beyond 1 SD (same side) · "
                           "8x eight in a row on one side (drift or bias).")
        with right:
            st.subheader("Duplicates: original vs repeat")
            dup = ev[(ev["sample_type"] == "Duplicate") & ev["original_gpt"].notna() & ev["assay_gpt"].notna()]
            if dup.empty:
                empty_note("No duplicates recorded.")
            else:
                mx = float(max(dup["original_gpt"].max(), dup["assay_gpt"].max())) * 1.05
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(width=1, dash="dash", color=GREY),
                                         name="1 : 1", hoverinfo="skip"))
                ok = dup[dup["status"] == "Pass"]
                bad_ = dup[dup["status"] != "Pass"]
                fig.add_trace(go.Scatter(x=ok["original_gpt"], y=ok["assay_gpt"], mode="markers", name="Pass",
                                         marker=dict(size=7, color=BLUE), hovertemplate="%{x:.2f} vs %{y:.2f}"))
                if not bad_.empty:
                    fig.add_trace(go.Scatter(x=bad_["original_gpt"], y=bad_["assay_gpt"], mode="markers", name="Warning / fail",
                                             marker=dict(size=9, color=RED, line=dict(width=2, color="white")),
                                             hovertemplate="%{x:.2f} vs %{y:.2f}"))
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Original (g/t)", yaxis_title="Duplicate (g/t)",
                                  legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(fig, width="stretch")
        cb = qaqc.check_bias(ev)
        if cb["n"]:
            ci = "" if pd.isna(cb["ci_low"]) else f" (95% interval {cb['ci_low']:+.1f}% to {cb['ci_high']:+.1f}%)"
            sig = "" if pd.isna(cb["ci_low"]) else (" The interval excludes zero, so the bias is real." if (cb["ci_low"] > 0 or cb["ci_high"] < 0)
                                                    else " The interval includes zero, so no bias is proven.")
            st.markdown(f"**Check assays against the site lab:** mean bias {cb['mean_bias_pct']:+.1f}%{ci} from {cb['n']} samples.{sig}")

        st.subheader("Results")
        show = ev.sort_values("date", ascending=False).head(300)
        tbl = show[["date", "sample_id", "sample_type", "lab", "assay_gpt", "expected_gpt", "sd_gpt", "original_gpt", "z",
                    "rd_pct", "status", "reason"]].copy()
        tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
        tbl["status"] = tbl["status"].map(QA_LABEL)
        st.dataframe(rnd(tbl, 3).rename(columns={"sample_type": "Type", "assay_gpt": "Assay", "expected_gpt": "Certified",
                                                 "sd_gpt": "Cert. SD", "original_gpt": "Original", "rd_pct": "Rel. diff %"}),
                     width="stretch", hide_index=True)
        with st.expander("Delete a result"):
            ids = show.set_index("id")
            del_q = st.selectbox("Result", ids.index.tolist(), key="del_qa",
                                 format_func=lambda i_: f"{i_}: {day_str(ids.loc[i_, 'date'])} {ids.loc[i_, 'sample_id']}")
            if st.button("Delete result"):
                storage.delete_ids("qaqc", [del_q])
                st.rerun()

# ───────────────────────── gold room ─────────────────────────
with tab_gold:
    st.subheader("Record a pour")
    st.caption("Fine gold = gross bar weight × gold fineness. The daily gold poured in Daily entry should be fine gold, "
               "so the bars here should add up to it over time.")
    with st.form("pour_form", clear_on_submit=True):
        a, b, c = st.columns(3)
        p_date = a.date_input("Date", pd.Timestamp.today().date(), key="p_date")
        p_bar = b.text_input("Bar number")
        p_gross = c.number_input("Gross bar weight (g)", 0.0, 1e7, 0.0, 10.0)
        a, b, c = st.columns(3)
        p_au = a.number_input("Gold fineness (%)", 0.0, 100.0, 0.0, 0.1)
        p_ag = b.number_input("Silver fineness (%)", 0.0, 100.0, 0.0, 0.1)
        p_slag = c.number_input("Gold in slag (g), optional", 0.0, 1e6, 0.0, 1.0)
        if st.form_submit_button("Save pour"):
            row = pd.DataFrame([dict(date=pd.Timestamp(p_date), bar_no=p_bar or "n/a", gross_g=p_gross, au_pct=p_au,
                                     ag_pct=(p_ag if p_ag > 0 else None), slag_au_g=(p_slag if p_slag > 0 else None))])
            iss_p = validate_pours(row)
            errs = iss_p[iss_p["severity"] == "error"]
            if len(errs):
                for m_ in errs["message"]:
                    st.error(m_)
            else:
                storage.save_rows("pours", row)
                for m_ in iss_p[iss_p["severity"] == "warning"]["message"]:
                    st.warning("Saved with a warning: " + m_)
                st.rerun()
    if pours_raw.empty:
        empty_note("No pours recorded yet.")
    else:
        pp = goldroom.prepare(pours_raw)
        pm = goldroom.monthly(pp)
        k = st.columns(4)
        tot_fine = pp["fine_au_g"].sum()
        k[0].metric("Bars poured", len(pp))
        k[1].metric("Fine gold", f"{tot_fine / 31.1035:,.0f} oz")
        k[2].metric("Overall fineness", f"{tot_fine / pp['gross_g'].sum() * 100:.1f}%")
        sm_ = pp[pp["smelt_recovery_pct"].notna()]
        k[3].metric("Smelt recovery", "–" if sm_.empty else f"{sm_['fine_au_g'].sum() / (sm_['fine_au_g'].sum() + sm_['slag_au_g'].sum()) * 100:.2f}%")
        left, right = st.columns(2)
        with left:
            st.subheader("Fine gold by month")
            st.plotly_chart(bar_chart(pm, "fine_au_oz", "Fine gold (oz)", x="period"), width="stretch")
        with right:
            st.subheader("Gold fineness by bar")
            st.plotly_chart(line_chart(pp.assign(label=pp["date"]), [("au_pct", "Au fineness (%)", BLUE)], "% Au"), width="stretch")
        st.subheader("Monthly summary")
        st.dataframe(rnd(pm, 2).rename(columns={"period": "Month", "bars": "Bars", "gross_kg": "Gross kg", "fine_au_g": "Fine Au g",
                                                "fine_au_oz": "Fine Au oz", "au_fineness_pct": "Au fineness %", "ag_to_au": "Ag : Au",
                                                "smelt_recovery_pct": "Smelt recovery %"}), width="stretch", hide_index=True)
        rc = goldroom.reconcile(pp, raw)
        if not rc.empty:
            st.subheader("Bars vs the daily gold poured")
            st.dataframe(rnd(rc, 1).rename(columns={"period": "Month", "bars_fine_g": "Fine gold in bars (g)",
                                                    "daily_poured_g": "Daily gold poured (g)", "difference_g": "Difference (g)",
                                                    "difference_pct": "Difference %", "cumulative_difference_g": "Running difference (g)"}),
                         width="stretch", hide_index=True)
            st.caption("A bar poured at the start of a month holds gold from the end of the last month, so month-end differences "
                       "reverse the next month. Judge the running difference: if it keeps growing, gold is being recorded in one place and not the other.")
        with st.expander("All bars"):
            show = pp.sort_values("date", ascending=False).assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
            st.dataframe(rnd(show.drop(columns=["fine_au_oz"]), 2), width="stretch", hide_index=True)
            del_p = st.selectbox("Delete a bar", show["id"].tolist(), key="del_pour",
                                 format_func=lambda i_: f"{i_}: {show.set_index('id').loc[i_, 'date']} {show.set_index('id').loc[i_, 'bar_no']}")
            if st.button("Delete bar"):
                storage.delete_ids("pours", [del_p])
                st.rerun()

# ───────────────────────── control charts ─────────────────────────
with tab_spc:
    if daily_all.empty:
        empty_note()
    else:
        st.caption("Control limits are worked out from your own data: a baseline of normal days gives a centre line and a normal range "
                   "(±3 sigma). A point outside the range, or 8 points in a row on one side, means the process has changed. "
                   "This is better than a fixed threshold because every plant and ore is different.")
        a, o, b = st.columns([2, 2, 1])
        metric = a.selectbox("Measure", list(spc.SPC_METRICS), format_func=lambda m: spc.SPC_METRICS[m][0], key="spc_metric")
        ore_opts = ["All ore types"] + sorted(daily_all["ore_type"].dropna().astype(str).unique())
        ore_f = o.selectbox("Ore type", ore_opts, key="spc_ore",
                            help="Compare like with like: limits worked out on one ore type only.")
        base_n = b.number_input("Baseline: first N days", 8, 365, baseline_days, 1, key="spc_base",
                                help="Choose a period when the plant ran normally.")
        spc_df = daily_all if ore_f == "All ore types" else daily_all[daily_all["ore_type"].astype(str) == ore_f]
        tbl, lim_ = spc.control_table(spc_df.reset_index(drop=True), metric, int(base_n))
        tbl_sel = in_range(tbl, start, end)
        if tbl_sel is None or tbl_sel.empty or not np.isfinite(lim_["sigma"]):
            empty_note("Not enough data for control limits (needs at least 8 days).")
        else:
            label = spc.SPC_METRICS[metric][0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tbl_sel["date"], y=tbl_sel["value"], mode="lines+markers", name=label,
                                     line=dict(width=1.6, color=BLUE), marker=dict(size=5, color=BLUE), hovertemplate="%{y:.3f}"))
            sig_pts = tbl_sel[tbl_sel["flag"] != ""]
            if not sig_pts.empty:
                fig.add_trace(go.Scatter(x=sig_pts["date"], y=sig_pts["value"], mode="markers", name="Signal",
                                         marker=dict(size=10, color=RED, line=dict(width=2, color="white")),
                                         customdata=sig_pts["flag"], hovertemplate="%{y:.3f} · %{customdata}"))
            c0, sg = lim_["center"], lim_["sigma"]
            for yv, lab_, dash in [(c0, "Centre", "solid"), (c0 + 3 * sg, "+3σ", "dash"), (c0 - 3 * sg, "−3σ", "dash"),
                                   (c0 + 2 * sg, "+2σ", "dot"), (c0 - 2 * sg, "−2σ", "dot")]:
                fig.add_hline(y=yv, line=dict(width=1, dash=dash, color=GREY), annotation_text=lab_,
                              annotation_position="top left", annotation_font_size=9)
            st.plotly_chart(style_fig(fig, 380, label), width="stretch")
            m = st.columns(5)
            m[0].metric("Centre", f"{c0:.3f}")
            m[1].metric("Lower limit (−3σ)", f"{lim_['lcl']:.3f}")
            m[2].metric("Upper limit (+3σ)", f"{lim_['ucl']:.3f}")
            m[3].metric("Sigma", f"{sg:.3f}")
            m[4].metric("Signals in view", len(sig_pts))
            if not sig_pts.empty:
                st.dataframe(rnd(sig_pts[["date", "value", "z", "flag"]].assign(date=sig_pts["date"].dt.strftime("%Y-%m-%d")), 3)
                             .rename(columns={"value": "Value", "z": "Sigmas from centre", "flag": "Signal"}),
                             width="stretch", hide_index=True)
            st.caption("If ore type changes, the centre line legitimately moves. Re-set the baseline to a period on the new ore, "
                       "or compare within one ore type.")

# ───────────────────────── alerts ─────────────────────────
with tab_alerts:
    if alerts is None or alerts.empty:
        st.success("No alerts in this period.")
    else:
        sev = st.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning"])
        types = sorted(alerts["type"].unique())
        typ = st.multiselect("Type", types, default=types)
        a_ = alerts[alerts["severity"].isin(sev) & alerts["type"].isin(typ)].sort_values("date", ascending=False)
        a_ = a_.assign(date=a_["date"].dt.strftime("%Y-%m-%d"), severity=a_["severity"].map(SEV_LABEL))
        st.caption(f"{len(a_)} alert(s)")
        st.dataframe(a_, width="stretch", hide_index=True)

# ───────────────────────── data quality ─────────────────────────
with tab_dq:
    sources = [("Daily records", issues)]
    if not tanks_raw.empty:
        sources.append(("Tank readings", validate_tanks(tanks_raw.reset_index(drop=True))))
    if not pours_raw.empty:
        sources.append(("Pours", validate_pours(pours_raw.reset_index(drop=True))))
    if not qa_raw.empty:
        sources.append(("QA/QC", validate_qaqc(qa_raw.reset_index(drop=True))))
    if not carbon_log.empty:
        sources.append(("Carbon log", validate_carbon(carbon_log.reset_index(drop=True))))
    all_iss = pd.concat([i_.assign(source=name) for name, i_ in sources if not i_.empty], ignore_index=True) \
        if any(not i_.empty for _, i_ in sources) else pd.DataFrame(columns=list(EMPTY_ISSUES.columns) + ["source"])
    n_err = int((all_iss["severity"] == "error").sum())
    n_warn = int((all_iss["severity"] == "warning").sum())
    k = st.columns(4)
    k[0].metric("Daily records", len(raw))
    k[1].metric("Errors", n_err)
    k[2].metric("Warnings", n_warn)
    k[3].metric("Inventory notes", len(inv_notes))
    if all_iss.empty and inv_notes.empty:
        st.success("No data issues found.")
    else:
        if exclude_errors and bad_rows:
            st.warning(f"{len(bad_rows)} daily record(s) with errors are excluded from all KPIs and charts.")
        v = all_iss.copy()
        if not v.empty:
            v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
            v["severity"] = v["severity"].map(SEV_LABEL)
            st.dataframe(v[["source", "date", "field", "severity", "message"]], width="stretch", hide_index=True)
        if not inv_notes.empty:
            st.subheader("Inventory notes")
            st.dataframe(inv_notes.assign(date=inv_notes["date"].dt.strftime("%Y-%m-%d"),
                                          severity=inv_notes["severity"].map(SEV_LABEL)), width="stretch", hide_index=True)

# ───────────────────────── reports & backup ─────────────────────────
with tab_rep:
    if daily_all.empty:
        empty_note("Add daily records to create reports.")
    else:
        st.subheader("Daily report (PDF)")
        dts = sorted(daily_all["date"].dt.date.unique(), reverse=True)
        rd = st.selectbox("Day", dts, key="rep_day")
        if st.button("Create daily report"):
            row = daily_all[daily_all["date"].dt.date == rd].iloc[0]
            al = alerts_all[alerts_all["date"].dt.date == rd] if not alerts_all.empty else alerts_all
            td = tcalc[tcalc["date"].dt.date == rd] if not tcalc.empty else None
            st.session_state["pdf_daily"] = (f"daily_report_{rd}.pdf", report.daily_report(
                plant_name, rd, row, daily_all[daily_all["date"].dt.date <= rd], al, targets, td))
        if "pdf_daily" in st.session_state:
            nm, data = st.session_state["pdf_daily"]
            st.download_button(f"⬇️ Download {nm}", data, nm, "application/pdf", key="dl_daily")

        st.subheader("Monthly report (PDF)")
        months = sorted(daily_all["date"].dt.to_period("M").astype(str).unique(), reverse=True)
        rm = st.selectbox("Month", months, key="rep_month")
        if st.button("Create monthly report"):
            md = daily_all[daily_all["date"].dt.to_period("M").astype(str) == rm].reset_index(drop=True)
            ms = overall(md)
            mal = alerts_all[alerts_all["date"].dt.to_period("M").astype(str) == rm] if not alerts_all.empty else alerts_all
            qa_m = qaqc.summary(ev_all[ev_all["date"].dt.to_period("M").astype(str) == rm]) if not ev_all.empty else None
            gm = None
            if not pours_raw.empty:
                g_ = goldroom.monthly(goldroom.prepare(pours_raw))
                g_ = g_[g_["period"] == rm]
                gm = g_.iloc[0] if len(g_) else None
            cl = carbon_log[carbon_log["date"].dt.to_period("M").astype(str) == rm] if not carbon_log.empty else carbon_log
            cs_m = circuit.carbon_stats(cl, float(md["tonnes_milled"].sum()), md["date"].max(), clim)
            title = pd.Period(rm).strftime("%B %Y")
            st.session_state["pdf_monthly"] = (f"monthly_report_{rm}.pdf", report.monthly_report(
                plant_name, title, md, ms, plan_vs_actual(ms, targets), mal, by_ore_type(md), qa_m, gm, cs_m, targets))
        if "pdf_monthly" in st.session_state:
            nm, data = st.session_state["pdf_monthly"]
            st.download_button(f"⬇️ Download {nm}", data, nm, "application/pdf", key="dl_monthly")

        st.subheader("Excel workbook")
        st.caption("Everything in the selected date range: daily results, monthly summary, ore types, tank readings, carbon log, "
                   "QA/QC, pours, alerts and data issues.")
        if st.button("Prepare Excel workbook"):
            sheets = {"Daily": rnd(d.drop(columns=["rec_7d_prev", "nacn_7d_prev"], errors="ignore")),
                      "Monthly": rnd(summarise(d, "M")), "Ore types": rnd(by_ore_type(d)),
                      "Targets": rnd(plan_vs_actual(overall(d), targets)),
                      "Alerts": alerts, "Data issues": all_iss.drop(columns="row", errors="ignore")}
            if not tcalc.empty:
                sheets["Tank readings"] = rnd(in_range(tcalc, start, end), 3)
            if not carbon_log.empty:
                sheets["Carbon log"] = in_range(carbon_log, start, end)
            if not ev_all.empty:
                sheets["QA-QC"] = rnd(in_range(ev_all, start, end), 3)
            if not pours_raw.empty:
                sheets["Pours"] = rnd(goldroom.prepare(pours_raw), 2)
            st.session_state["xlsx"] = to_excel(sheets)
        if "xlsx" in st.session_state:
            st.download_button("⬇️ Download plant_report.xlsx", st.session_state["xlsx"], "plant_report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
            st.download_button("⬇️ Daily results (.csv)", rnd(d).to_csv(index=False), "daily_results.csv", "text/csv", key="dl_csv")

    st.divider()
    st.subheader("Backup and restore")
    st.caption("Back up regularly. Some hosts wipe local files when the app restarts, so a backup you keep elsewhere is your safety net. "
               "The backup is one zip file with every record and your settings.")
    st.download_button("⬇️ Download full backup (.zip)", storage.backup_zip(), f"plant_backup_{pd.Timestamp.today():%Y%m%d}.zip",
                       "application/zip", key="dl_backup")
    rz = st.file_uploader("Restore from a backup zip", type=["zip"], key="restore_zip")
    ok_restore = st.checkbox("I understand this replaces all current records and settings", key="restore_ok")
    if rz is not None and st.button("Restore backup", disabled=not ok_restore):
        try:
            done = storage.restore_zip(rz.getvalue())
            st.success("Restored: " + ", ".join(f"{k_} ({v_})" for k_, v_ in done.items()))
            st.rerun()
        except Exception as ex:  # noqa: BLE001
            st.error(f"That file could not be restored: {ex}")

# ───────────────────────── ore scanner ─────────────────────────
@st.cache_data(show_spinner=False)
def _analyse_cached(data: bytes, tray_ref: bool):
    return scanner.analyse(data, tray_ref)


with tab_scan:
    SC = S["scanner"]
    scans = storage.load_table("scans")
    cutoff = float(SC["cutoff_gpt"])
    model = scanner.fit(scans, int(SC["min_scans"]))
    n_lab = len(scanner.labelled(scans))
    st.caption("Photograph crushed rock, and the app measures what a camera can see: dark sulphide grains, rust, "
               "quartz, colour, texture. Gold itself is invisible, so the grade comes only from the link your own mine's "
               "photos have with lab assays. Until that link is learned, the app shows measurements and no grade.")
    k = st.columns(4)
    k[0].metric("Scans saved", len(scans))
    k[1].metric("Scans with a lab assay", f"{n_lab} / {int(SC['min_scans'])} needed")
    if model:
        k[2].metric("Typical error", f"± {1.96 * model['loo_rmse']:.2f} g/t",
                    help="95% of leave-one-out test predictions fell within this of the lab assay.")
        k[3].metric("Calibration quality", model["quality"], f"R² {model['loo_r2']:.2f}", delta_color="off")
    else:
        k[2].metric("Typical error", "–")
        k[3].metric("Calibration quality", "Not calibrated")

    s_scan, s_cal, s_hist = st.tabs(["📷 Scan a sample", "Calibration", "History"])

    with s_scan:
        how = st.radio("Photo source", ["Take a photo", "Upload a photo", "Try a made-up photo"],
                       horizontal=True, key="scan_how")
        data, is_demo = None, False
        if how == "Take a photo":
            shot = st.camera_input("Photograph the rock on the tray", key="scan_cam")
            data = shot.getvalue() if shot else None
        elif how == "Upload a photo":
            up = st.file_uploader("Choose a photo (JPG or PNG)", type=["jpg", "jpeg", "png"], key="scan_up")
            data = up.getvalue() if up else None
        else:
            if st.button("Make a new practice photo", key="scan_mk"):
                rng_ = np.random.default_rng()
                st.session_state["scan_demo"] = scanner.synthetic_rock_image(
                    float(rng_.uniform(0.5, 5.5)), str(rng_.choice(["Oxide", "Transition", "Fresh"])), rng_,
                    float(rng_.uniform(0.85, 1.1)))
            data, is_demo = st.session_state.get("scan_demo"), True
            st.caption("A computer-drawn picture for practice. Its darkness-to-grade link is invented.")
        tray_ref = st.checkbox("Use the empty tray as the colour reference (recommended)", True, key="scan_ref")

        if data is None:
            st.info("Spread the crushed rock in one layer on a plain tray, leave empty tray showing at the edges, "
                    "use the same light and the same camera height every time, then take the photo.")
        else:
            try:
                res = _analyse_cached(data, tray_ref)
            except Exception:
                res = None
                st.error("That file could not be read as a photo. Try another one.")
            if res:
                f = res["features"]
                a, b = st.columns(2)
                a.image(res["image"], caption="Your photo", use_container_width=True)
                b.image(scanner.overlay(res["image"], res["mask"]), caption="What was measured (orange outline = rock)",
                        use_container_width=True)
                for w_ in res["warnings"]:
                    st.warning(w_)
                if model:
                    p = scanner.predict(model, f, cutoff)
                    r1, r2_, r3 = st.columns(3)
                    r1.metric("Estimated grade", f"{p['grade_gpt']:.2f} g/t")
                    r2_.metric("Likely range", f"{p['low']:.2f} to {p['high']:.2f} g/t")
                    r3.metric(f"Against cut-off ({cutoff:g} g/t)", p["call"])
                    if model["quality"] == "Poor":
                        st.error("The calibration is poor: photos on this mine do not predict the assay well yet. "
                                 "Do not use this number. Add more assayed scans and keep the photo method identical.")
                    elif model["quality"] == "Fair":
                        st.warning("The calibration is only fair. Use it to screen, and send anything near the cut-off to the lab.")
                    if p["unfamiliar"]:
                        st.warning("This photo looks unlike the calibration photos (different light, tray or rock). "
                                   "Treat the estimate with caution.")
                    guess = scanner.classify_ore_type(scans, f)
                    if guess:
                        st.caption(f"Colour and texture look most like **{guess[0]}** ore from your earlier scans.")
                else:
                    st.info(f"No grade yet: {max(0, int(SC['min_scans']) - n_lab)} more scans with a lab assay are needed "
                            "before the app can learn the link between photos and grade. Save this scan, and add its lab "
                            "assay later in History.")
                show = ["dark_frac", "light_frac", "rust_frac", "mean_v", "texture", "cover_frac", "n_particles", "d50_rel"]
                pct = {"dark_frac", "light_frac", "rust_frac", "cover_frac"}
                st.dataframe(pd.DataFrame({
                    "Measurement": [scanner.FEATURE_LABELS[c] for c in show],
                    "Value": [f"{f[c] * 100:.1f}%" if c in pct else f"{f[c]:.2f}" for c in show]}),
                    hide_index=True, use_container_width=True)
                with st.form("scan_form", clear_on_submit=True):
                    a, b, c_, e = st.columns(4)
                    sc_date = a.date_input("Date", pd.Timestamp.today().date(), key="sc_date")
                    sc_id = b.text_input("Sample ID", key="sc_id")
                    sc_ore = c_.selectbox("Ore type", ["", "Oxide", "Transition", "Fresh"], key="sc_ore")
                    sc_loc = e.text_input("Where from (stockpile, bench, belt)", key="sc_loc")
                    a, b = st.columns([1, 3])
                    sc_assay = a.number_input("Lab assay (g/t), if known", 0.0, 1000.0, 0.0, 0.001, format="%.3f",
                                              help="Leave at 0 if not assayed yet; add it later in History.", key="sc_assay")
                    sc_note = b.text_input("Note", key="sc_note")
                    if st.form_submit_button("Save this scan"):
                        row = pd.DataFrame([{"date": pd.Timestamp(sc_date),
                                             "sample_id": sc_id or f"SCAN-{pd.Timestamp.now():%Y%m%d-%H%M%S}",
                                             "ore_type": sc_ore or None, "location": sc_loc or None,
                                             "assay_gpt": sc_assay if sc_assay > 0 else None,
                                             "note": ("practice photo. " if is_demo else "") + sc_note, **f}])
                        storage.save_rows("scans", row)
                        st.success("Scan saved.")
                        st.rerun()

    with s_cal:
        if not model:
            st.info(f"Needs at least {int(SC['min_scans'])} scans with a lab assay (you have {n_lab}). "
                    "Aim for 40 or more, spread across low, medium and high grade and across ore types.")
        else:
            st.markdown(f"Calibrated on **{model['n']}** assayed scans. Each dot is a scan the model had **not seen** "
                        "when predicting it (leave-one-out test), so this is an honest picture of performance.")
            cal = pd.DataFrame({"assay": model["loo_actual"], "pred": model["loo_pred"]})
            top = float(max(cal.max()) * 1.05)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect",
                                     line=dict(width=1, dash="dash", color=GREY), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cal["assay"], y=cal["pred"], mode="markers", name="Scan",
                                     marker=dict(size=8, color=BLUE, opacity=0.8),
                                     hovertemplate="Lab %{x:.2f}, photo %{y:.2f} g/t"))
            fig = style_fig(fig, 380, "Estimated from photo (g/t)", legend=True)
            fig.update_xaxes(title="Lab assay (g/t)", range=[0, top])
            fig.update_yaxes(range=[0, top])
            st.plotly_chart(fig, use_container_width=True)
            wdf = pd.DataFrame({"feature": [scanner.FEATURE_LABELS[c] for c in model["features"]], "w": model["w"]})
            wdf = wdf.reindex(wdf["w"].abs().sort_values(ascending=False).index)
            st.markdown("**What the model leans on** (positive = more of this means higher grade)")
            st.plotly_chart(bar_chart(wdf, "w", "Weight", x="feature", height=300, fmt="%{y:+.2f}"),
                            use_container_width=True)
            if model["quality"] == "Poor":
                st.error("Poor calibration. Photos are not tracking the assays. Check the photo method is identical "
                         "each time (light, height, tray, dry rock), and whether grade on this deposit has any visible signature.")
        st.markdown("**Keeping photos comparable:** same tray colour, same light (a closed box is best), same camera "
                    "height, dry rock in a single layer, similar crush size, and a lab assay on the same sample you photographed. "
                    "Re-check against the lab every month, and again after any change of ore source or lighting.")

    with s_hist:
        if scans.empty:
            st.info("No scans saved yet.")
        else:
            h = scans.copy()
            h["photo_estimate_gpt"] = scanner.predict_all(model, h) if model else np.nan
            if model:
                loo = pd.Series(model["loo_pred"], index=scanner.labelled(scans).index)
                h.loc[loo.index, "photo_estimate_gpt"] = loo       # tested value for assayed scans
            cols = ["date", "sample_id", "ore_type", "location", "assay_gpt", "photo_estimate_gpt", "dark_frac",
                    "rust_frac", "note"]
            shown = h.sort_values("date", ascending=False)[["id", *cols]].head(200).copy()
            shown["date"] = shown["date"].dt.date
            st.dataframe(shown.rename(columns={"assay_gpt": "lab_assay_gpt"}), hide_index=True,
                         use_container_width=True, height=300)
            est = h.dropna(subset=["photo_estimate_gpt"])
            if len(est) > 1:
                st.markdown("**Photo estimate over time** (g/t)")
                st.plotly_chart(line_chart(est.sort_values("date"), [("photo_estimate_gpt", "Photo estimate", BLUE)],
                                           "g/t", height=260, hlines=[(cutoff, "Cut-off")]), use_container_width=True)
            a, b = st.columns(2)
            pend = h[h["assay_gpt"].isna()]
            with a:
                st.markdown("**Add a lab assay to a saved scan**")
                if pend.empty:
                    st.caption("Every saved scan already has an assay.")
                else:
                    opts = {f"{r.sample_id} ({r.date:%d %b %Y})": int(r.id) for r in pend.itertuples()}
                    pick = st.selectbox("Scan", list(opts), key="sc_pick")
                    val = st.number_input("Lab assay (g/t)", 0.0, 1000.0, 0.0, 0.001, format="%.3f", key="sc_val")
                    if st.button("Save assay", key="sc_saveassay") and val > 0:
                        storage.update_cells("scans", opts[pick], {"assay_gpt": val})
                        st.rerun()
            with b:
                st.markdown("**Delete a scan**")
                del_opts = {f"{r.sample_id} ({r.date:%d %b %Y})": int(r.id) for r in h.itertuples()}
                dpick = st.selectbox("Scan", list(del_opts), key="sc_delpick")
                if st.button("Delete this scan", key="sc_del"):
                    storage.delete_ids("scans", [del_opts[dpick]])
                    st.rerun()


# ───────────────────────── settings ─────────────────────────
with tab_set:
    st.caption("Settings are saved with your data. Set these to match your plant before relying on the alerts and reports.")
    new_name = st.text_input("Plant name", plant_name, key="set_name")
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Unit costs (US$)")
        n_costs = {"nacn_per_kg": numfield("NaCN per kg", costs.nacn_per_kg, 0, 50, 0.1, "set_c1"),
                   "lime_per_kg": numfield("Lime per kg", costs.lime_per_kg, 0, 10, 0.05, "set_c2"),
                   "power_per_kwh": numfield("Power per kWh", costs.power_per_kwh, 0, 2, 0.01, "set_c3")}
        st.subheader("Targets (for the Dashboard and reports)")
        tg = targets
        n_targets = {"recovery_pct": numfield("Recovery (%)", tg["recovery_pct"], 0, 100, 0.5, "set_t1"),
                     "head_gpt": numfield("Head grade (g/t)", tg["head_gpt"], 0, 60, 0.1, "set_t2"),
                     "tails_gpt": numfield("Tails grade (g/t)", tg["tails_gpt"], 0, 5, 0.01, "set_t3"),
                     "throughput_tph": numfield("Throughput (t/h)", tg["throughput_tph"], 0, 1500, 5, "set_t4"),
                     "availability_pct": numfield("Availability (%)", tg["availability_pct"], 0, 100, 1, "set_t5"),
                     "nacn_kgpt": numfield("NaCN (kg/t)", tg["nacn_kgpt"], 0, 5, 0.01, "set_t6"),
                     "power_kwhpt": numfield("Power (kWh/t)", tg["power_kwhpt"], 0, 100, 0.5, "set_t7"),
                     "unaccounted_pct": numfield("Unaccounted gold (% of feed, upper limit)", tg["unaccounted_pct"], 0, 20, 0.1, "set_t8")}
        st.subheader("Control charts")
        n_spc = {"baseline_days": int(numfield("Default baseline (first N days)", baseline_days, 8, 365, 1, "set_s1", "%d"))}
    with g2:
        st.subheader("Alert thresholds")
        n_th = {"recovery_drop_pts": numfield("Recovery drop vs 7-day mean (pts)", th.recovery_drop_pts, 0.5, 20, 0.5, "set_a1"),
                "unaccounted_warn_pct": numfield("Unaccounted gold warning (% of feed)", th.unaccounted_warn_pct, 0.5, 20, 0.5, "set_a2"),
                "unaccounted_crit_pct": numfield("Unaccounted gold critical (% of feed)", th.unaccounted_crit_pct, 0.5, 30, 0.5, "set_a3"),
                "head_gap_pct": numfield("Head grade mismatch (%)", th.head_gap_pct, 1, 30, 0.5, "set_a4"),
                "reagent_spike_factor": numfield("Cyanide spike factor (× 7-day mean)", th.reagent_spike_factor, 1.05, 3, 0.05, "set_a5"),
                "min_availability_pct": numfield("Minimum availability (%)", th.min_availability_pct, 50, 100, 1, "set_a6"),
                "solution_loss_pct": numfield("Dissolved gold loss (% of tails gold)", th.solution_loss_pct, 5, 90, 5, "set_a7")}
        st.subheader("Leach circuit limits")
        n_circ = {"min_ph": numfield("Minimum pH", clim.min_ph, 8, 12, 0.1, "set_k1"),
                  "min_do_ppm": numfield("Minimum dissolved oxygen (mg/L)", clim.min_do_ppm, 0, 20, 0.5, "set_k2"),
                  "min_free_cn_ppm": numfield("Minimum free cyanide (ppm)", clim.min_free_cn_ppm, 0, 1000, 10, "set_k3"),
                  "carbon_conc_min_gpl": numfield("Minimum carbon concentration (g/L)", clim.carbon_conc_min_gpl, 0, 40, 0.5, "set_k4"),
                  "carbon_conc_max_gpl": numfield("Maximum carbon concentration (g/L)", clim.carbon_conc_max_gpl, 1, 60, 0.5, "set_k5"),
                  "min_activity_pct": numfield("Minimum carbon activity (% of fresh)", clim.min_activity_pct, 10, 100, 5, "set_k6"),
                  "solids_sg": numfield("Ore solids specific gravity", clim.solids_sg, 2.0, 5.0, 0.05, "set_k7"),
                  "default_pct_solids": numfield("Default slurry % solids (if not measured)", clim.default_pct_solids, 20, 60, 1, "set_k8")}
        st.subheader("Lab QA/QC limits")
        n_q = {"z_warn": numfield("Standards warning (SD)", qlim.z_warn, 1, 4, 0.5, "set_q1"),
               "z_fail": numfield("Standards fail (SD)", qlim.z_fail, 2, 6, 0.5, "set_q2"),
               "blank_limit_gpt": numfield("Blank limit (g/t)", qlim.blank_limit_gpt, 0.001, 1, 0.005, "set_q3", "%.3f"),
               "dup_rd_pct": numfield("Duplicate relative difference limit (%)", qlim.dup_rd_pct, 1, 50, 1, "set_q4")}
    st.subheader("CIL tanks")
    nt = int(st.number_input("Number of CIL tanks", 1, 12, n_tanks, 1, key="set_ntanks"))
    vcols = st.columns(min(nt, 6))
    new_vols = [float(vcols[i % len(vcols)].number_input(f"Tank {i + 1} volume (m³)", 1.0, 20000.0,
                                                          float(tank_vols[i] if i < len(tank_vols) else tank_vols[-1]), 10.0,
                                                          key=f"set_vol_{i}")) for i in range(nt)]
    st.subheader("Ore scanner")
    sc1, sc2 = st.columns(2)
    n_scan = {"cutoff_gpt": numfield("Cut-off grade for scanner calls (g/t)", S["scanner"]["cutoff_gpt"], 0, 50, 0.1, "set_sc1"),
              "min_scans": int(numfield("Assayed scans needed before it gives a grade", S["scanner"]["min_scans"], 8, 200, 1, "set_sc2", "%d"))}
    if st.button("Save settings", type="primary"):
        storage.save_settings({"plant_name": new_name, "costs": n_costs, "targets": n_targets, "spc": n_spc,
                               "thresholds": n_th, "circuit": {**n_circ, "tank_volumes_m3": new_vols}, "qaqc": n_q, "scanner": n_scan})
        st.success("Settings saved.")
        st.rerun()
