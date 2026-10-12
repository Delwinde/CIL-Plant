"""PDF reports: a one-page daily metallurgical report and a monthly report (reportlab + matplotlib)."""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, MUTED, RULE = colors.HexColor("#1b1b1a"), colors.HexColor("#6b6a66"), colors.HexColor("#d9d8d3")
STATUS_COLOUR = {"On target": "#1a7f37", "Slightly behind": "#9a6700", "Behind": "#cf222e"}

FONT, FONT_BOLD = "Helvetica", "Helvetica-Bold"
try:  # DejaVu ships with matplotlib and covers σ, ≥, ² and similar characters
    _dir = matplotlib.get_data_path() + "/fonts/ttf/"
    pdfmetrics.registerFont(TTFont("DejaVu", _dir + "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", _dir + "DejaVuSans-Bold.ttf"))
    FONT, FONT_BOLD = "DejaVu", "DejaVu-Bold"
except Exception:  # pragma: no cover - fall back to the built-in font
    pass

S_TITLE = ParagraphStyle("title", fontName=FONT_BOLD, fontSize=15, leading=18, textColor=INK, spaceAfter=2)
S_SUB = ParagraphStyle("sub", fontName=FONT, fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=6)
S_H2 = ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=10, leading=13, textColor=INK, spaceBefore=7, spaceAfter=3)
S_BODY = ParagraphStyle("body", fontName=FONT, fontSize=8.3, leading=11, textColor=INK)
S_CELL = ParagraphStyle("cell", fontName=FONT, fontSize=7.8, leading=9.6, textColor=INK)
S_CELLB = ParagraphStyle("cellb", parent=S_CELL, fontName=FONT_BOLD)


def n(v, nd=1, suffix="", dash="–"):
    """Format a number; NaN or None becomes a dash."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return dash
    return f"{v:,.{nd}f}{suffix}"


def P(text, style=S_CELL):
    return Paragraph(escape(str(text)), style)


def _table(rows, widths, header=True, align_right_from=1, zebra=True):
    data = []
    for i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else P(c, S_CELLB if (header and i == 0) else S_CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeea")), ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK)]
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f8f6")))
    t.setStyle(TableStyle(style))
    return t


def _png(fig, width_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    scale = (width_mm * mm) / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return img


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bdbcb6")
    ax.spines["bottom"].set_color("#bdbcb6")
    ax.tick_params(labelsize=7, colors="#4a4944")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def _footer(plant_name):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(15 * mm, 9 * mm, f"{plant_name} · generated {datetime.now():%Y-%m-%d %H:%M}")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def _build(story, plant_name, title):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm,
                            bottomMargin=15 * mm, title=title, author=plant_name)
    doc.build(story, onFirstPage=_footer(plant_name), onLaterPages=_footer(plant_name))
    return buf.getvalue()


def _status_cell(status):
    return Paragraph(f'<font color="{STATUS_COLOUR.get(status, "#1b1b1a")}"><b>{escape(status)}</b></font>', S_CELL)


# ============================================================ daily report
def daily_report(plant_name: str, date, row: pd.Series, recent: pd.DataFrame, alerts: pd.DataFrame,
                 targets: dict, tanks_day: pd.DataFrame | None = None) -> bytes:
    """One page. row = the computed daily record, recent = computed rows up to and including that day."""
    date = pd.Timestamp(date)
    story = [P(f"Daily metallurgical report: {date:%A %d %B %Y}", S_TITLE),
             P(f"{plant_name} · ore type: {row.get('ore_type') or 'not recorded'}", S_SUB)]

    w = [58 * mm, 28 * mm, 28 * mm, 66 * mm]
    prod = [["Production", "Value", "Target", "Note"],
            ["Tonnes milled (t)", n(row["tonnes_milled"], 0), "", ""],
            ["Mill running hours", n(row["mill_hours"]), "", f"{n(row['availability_pct'], 0, '%')} of scheduled hours"],
            ["Throughput (t/h)", n(row["tph"]), n(targets.get("throughput_tph"), 0), ""],
            ["Head grade (g/t)", n(row["head_grade_gpt"], 2), n(targets.get("head_gpt"), 2), ""],
            ["Tails grade, total (g/t)", n(row["tails_grade_gpt"], 3), n(targets.get("tails_gpt"), 2),
             ("solution part " + n(row.get("solution_loss_gpt"), 3) + " g/t" if pd.notna(row.get("solution_loss_gpt")) else "")],
            ["Recovery from assays (%)", n(row["recovery_grade_pct"]), n(targets.get("recovery_pct"), 1), "(head − tails) ÷ head"],
            ["Recovery from gold produced (%)", n(row["recovery_accounted_pct"]), "", "recovered ÷ feed gold"]]
    story += [_table(prod, w)]

    src = row.get("inventory_source", "")
    bal = [["Gold balance (g)", "Value", "", "Note"],
           ["Feed gold", n(row["feed_gold_g"], 0), "", "tonnes × head grade"],
           ["Tails gold (lost)", n(row["tails_gold_g"], 0), "", "tonnes × tails grade"],
           ["Fine gold poured", n(row["gold_poured_g"], 0), "", ""],
           ["Change in in-circuit inventory", n(row.get("inventory_change_g"), 0), "",
            {"auto": "calculated from tank readings", "manual": "typed in"}.get(src, "none recorded (taken as 0)")],
           ["Gold recovered", n(row["gold_recovered_g"], 0), "", f"{n(row['gold_recovered_oz'], 1)} oz"],
           ["Unaccounted gold", n(row["unaccounted_g"], 0), "", f"{n(row['unaccounted_pct'], 2, '%')} of feed gold"],
           ["Calculated head grade (g/t)", n(row["calc_head_gpt"], 2), "", f"assay head differs by {n(row['head_gap_pct'], 1, '%')}"]]
    story += [P("Gold balance", S_H2), _table(bal, w)]

    reag = [["Reagents and energy", "Per tonne", "Target", "Cost"],
            ["Sodium cyanide (kg/t)", n(row["nacn_kgpt"], 3), n(targets.get("nacn_kgpt"), 2), ""],
            ["Lime (kg/t)", n(row["lime_kgpt"], 2), "", ""],
            ["Power (kWh/t)", n(row["power_kwhpt"]), n(targets.get("power_kwhpt"), 0), ""],
            ["Process cost", "", "", f"US$ {n(row['opex_per_t'], 2)} /t · US$ {n(row['process_cost_per_oz'], 0)} /oz"]]
    story += [P("Reagents and energy", S_H2), _table(reag, w)]

    if tanks_day is not None and not tanks_day.empty:
        t = [["Tank", "pH", "DO (mg/L)", "Free CN (ppm)", "Carbon (g/L)", "Loading (g/t)", "Solution Au (mg/L)"]]
        for _, r in tanks_day.sort_values("tank").iterrows():
            t.append([str(int(r["tank"])), n(r["ph"], 2), n(r["do_ppm"]), n(r["free_cn_ppm"], 0),
                      n(r["carbon_conc_gpl"]), n(r["carbon_loading_gpt"], 0), n(r["solution_au_mgl"], 3)])
        story += [P("Leach circuit readings", S_H2), _table(t, [14 * mm] + [28 * mm] * 6)]

    story.append(P("Alerts for the day", S_H2))
    if alerts is None or alerts.empty:
        story.append(P("No alerts.", S_BODY))
    else:
        for _, a in alerts.iterrows():
            story.append(Paragraph(f"<b>{escape(a['severity'].upper())}</b> · {escape(a['type'])}: {escape(a['message'])}", S_BODY))

    r = recent.tail(14)
    if len(r) >= 3:
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.1))
        axs[0].plot(r["date"], r["recovery_grade_pct"], color=BLUE, lw=1.6)
        if targets.get("recovery_pct"):
            axs[0].axhline(targets["recovery_pct"], color="#8a8983", lw=0.9, ls="--")
        axs[0].set_title("Recovery, last 14 days (%)", fontsize=8, loc="left")
        axs[1].plot(r["date"], r["head_grade_gpt"], color=BLUE, lw=1.6, label="Head")
        axs[1].plot(r["date"], r["tails_grade_gpt"], color=ORANGE, lw=1.6, label="Tails")
        axs[1].set_title("Head and tails grade (g/t)", fontsize=8, loc="left")
        axs[1].legend(fontsize=7, frameon=False)
        for ax in axs:
            _style_axes(ax)
            ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        story += [Spacer(1, 4), _png(fig, 180)]
    return _build(story, plant_name, f"Daily report {date:%Y-%m-%d}")


# ============================================================ monthly report
def monthly_report(plant_name: str, period: str, d: pd.DataFrame, summary: pd.Series, plan: pd.DataFrame,
                   alerts: pd.DataFrame, ore: pd.DataFrame, qa_summary: pd.DataFrame | None = None,
                   gold_month: pd.Series | None = None, carbon: dict | None = None,
                   targets: dict | None = None) -> bytes:
    """d = computed daily rows for the month; summary = overall(d)."""
    targets = targets or {}
    s = summary
    story = [P(f"Monthly metallurgical report: {period}", S_TITLE),
             P(f"{plant_name} · {int(s['days'])} days of data · tonnage-weighted figures", S_SUB)]

    k = [["Tonnes milled", "Head grade", "Recovery", "Gold recovered", "Unaccounted gold"],
         [n(s["tonnes_milled"], 0) + " t", n(s["head_grade_gpt"], 2) + " g/t", n(s["recovery_grade_pct"], 1, "%"),
          n(s["gold_recovered_oz"], 0) + " oz", f"{n(s['unaccounted_pct'], 2, '%')} ({n(s['unaccounted_g'], 0)} g)"]]
    story.append(_table(k, [36 * mm] * 5, zebra=False))

    story.append(P("Performance against targets", S_H2))
    if plan is None or plan.empty:
        story.append(P("No targets set. Add them in Settings.", S_BODY))
    else:
        rows = [["Measure", "Target", "Actual", "Variance", "Status"]]
        for _, r in plan.iterrows():
            rows.append([r["metric"], n(r["target"], 2), n(r["actual"], 2), f"{r['variance']:+.2f}", _status_cell(r["status"])])
        story.append(_table(rows, [58 * mm, 28 * mm, 28 * mm, 28 * mm, 38 * mm]))

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.2))
    axs[0].plot(d["date"], d["recovery_grade_pct"], color=BLUE, lw=1.4, label="Daily")
    axs[0].plot(d["date"], d["recovery_grade_pct"].rolling(7, min_periods=3).mean(), color=ORANGE, lw=1.6, label="7-day average")
    if targets.get("recovery_pct"):
        axs[0].axhline(targets["recovery_pct"], color="#8a8983", lw=0.9, ls="--")
    axs[0].set_title("Recovery (%)", fontsize=8, loc="left")
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    axs[1].bar(d["date"], d["unaccounted_g"], color=ORANGE, width=0.8)
    axs[1].axhline(0, color="#8a8983", lw=0.8)
    axs[1].set_title("Unaccounted gold per day (g)", fontsize=8, loc="left")
    for ax in axs:
        _style_axes(ax)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    story += [Spacer(1, 4), _png(fig, 180)]

    feed, tails, rec = d["feed_gold_g"].sum(), d["tails_gold_g"].sum(), d["gold_recovered_g"].sum()
    bal = [["Gold balance", "Grams", "% of feed"],
           ["Feed gold", n(feed, 0), "100.0"],
           ["Tails gold (lost)", n(tails, 0), n(tails / feed * 100 if feed else np.nan)],
           ["Gold recovered", n(rec, 0), n(rec / feed * 100 if feed else np.nan)],
           ["Unaccounted", n(feed - tails - rec, 0), n((feed - tails - rec) / feed * 100 if feed else np.nan, 2)]]
    sl = s.get("solution_loss_pct_of_tails")
    story += [P("Gold balance", S_H2), _table(bal, [70 * mm, 40 * mm, 40 * mm])]
    if pd.notna(sl):
        story.append(Spacer(1, 2))
        story.append(P(f"Dissolved gold is {sl:.0f}% of the gold lost in the tails.", S_BODY))

    ops = [["Operating figures", "Value"],
           ["Throughput (t/h)", n(s["tph"])], ["Availability (%)", n(s["availability_pct"])],
           ["Sodium cyanide (kg/t)", n(s["nacn_kgpt"], 3)], ["Lime (kg/t)", n(s["lime_kgpt"], 2)],
           ["Power (kWh/t)", n(s["power_kwhpt"])]]
    if pd.notna(s.get("mine_call_factor")):
        ops.append(["Mine call factor", n(s["mine_call_factor"], 3)])
    if "residence_time_h" in d and d["residence_time_h"].notna().any():
        ops.append(["Mean CIL residence time (h)", n(d["residence_time_h"].mean())])
    story += [P("Operating figures", S_H2), _table(ops, [70 * mm, 40 * mm])]

    if carbon and carbon.get("carbon_added_kg"):
        c = [["Carbon management", "Value"],
             ["Carbon added (kg)", n(carbon["carbon_added_kg"], 0)],
             ["Carbon consumption (g per t of ore)", n(carbon["carbon_consumption_gpt"], 1)],
             ["Elutions / regenerations", f"{carbon['elutions']} / {carbon['regenerations']}"],
             ["Latest activity test (% of fresh)", n(carbon["latest_activity_pct"], 0)]]
        story += [KeepTogether([P("Carbon management", S_H2), _table(c, [70 * mm, 40 * mm])])]

    if ore is not None and len(ore) > 1 or (ore is not None and len(ore) == 1 and ore.iloc[0]["ore_type"] != "Unspecified"):
        rows = [["Ore type", "Share of tonnes (%)", "Head (g/t)", "Tails (g/t)", "Recovery (%)"]]
        for _, r in ore.iterrows():
            rows.append([r["ore_type"], n(r["share_pct"]), n(r["head_grade_gpt"], 2), n(r["tails_grade_gpt"], 3),
                         n(r["recovery_grade_pct"])])
        story += [KeepTogether([P("Performance by ore type", S_H2), _table(rows, [40 * mm, 34 * mm, 28 * mm, 28 * mm, 28 * mm])])]

    if gold_month is not None:
        g = [["Gold room", "Value"], ["Bars poured", n(gold_month["bars"], 0)],
             ["Fine gold (oz)", n(gold_month["fine_au_oz"], 0)],
             ["Gold fineness (%)", n(gold_month["au_fineness_pct"])],
             ["Smelt recovery (%)", n(gold_month["smelt_recovery_pct"], 2)]]
        story += [KeepTogether([P("Gold room", S_H2), _table(g, [70 * mm, 40 * mm])])]

    if qa_summary is not None and not qa_summary.empty:
        rows = [["Sample type", "Samples", "Pass rate (%)", "Warnings", "Fails", "Mean bias (%)"]]
        for _, r in qa_summary.iterrows():
            rows.append([r["sample_type"], n(r["n"], 0), n(r["pass_pct"], 0), n(r["warnings"], 0), n(r["fails"], 0),
                         n(r["mean_bias_pct"], 1)])
        story += [KeepTogether([P("Laboratory QA/QC", S_H2), _table(rows, [40 * mm, 24 * mm, 28 * mm, 24 * mm, 20 * mm, 28 * mm])])]

    story.append(P("Alerts this month", S_H2))
    if alerts is None or alerts.empty:
        story.append(P("No alerts.", S_BODY))
    else:
        cnt = alerts.groupby(["type", "severity"]).size().reset_index(name="n").sort_values("n", ascending=False)
        rows = [["Alert type", "Severity", "Count"]] + [[r["type"], r["severity"], str(r["n"])] for _, r in cnt.head(12).iterrows()]
        story.append(_table(rows, [80 * mm, 34 * mm, 24 * mm]))
        crit = alerts[alerts["severity"] == "critical"].sort_values("date").head(8)
        if len(crit):
            story += [Spacer(1, 3), P("Critical alerts", S_H2)]
            for _, a in crit.iterrows():
                story.append(Paragraph(f"<b>{pd.Timestamp(a['date']):%d %b}</b> · {escape(a['type'])}: {escape(a['message'])}", S_BODY))
    return _build(story, plant_name, f"Monthly report {period}")
