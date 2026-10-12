"""Tests for the circuit, carbon, QA/QC, SPC, gold room, targets, storage and validation additions."""
import numpy as np
import pandas as pd
import pytest

import circuit
import goldroom
import qaqc
import spc
import storage
from calc import (by_ore_type, compute_daily, plan_vs_actual, resolve_tails, summarise, overall,
                  build_alerts, AlertThresholds)
from validation import (validate, validate_carbon, validate_pours, validate_qaqc, validate_tanks)


def day(**kw):
    base = dict(date="2026-08-01", tonnes_milled=4000.0, mill_hours=22.0, scheduled_hours=24.0,
                head_grade_gpt=2.5, tails_grade_gpt=0.2, gold_poured_g=9000.0,
                inventory_change_g=0.0, nacn_kg=2200.0, lime_kg=6400.0, power_kwh=88000.0)
    base.update(kw)
    return pd.DataFrame([base])


# ================= Phase 1: tails split =================
def test_solution_loss_hand_calc():
    # 40 % solids -> 1.5 m3 water per tonne of solids; 0.03 mg/L x 1.5 = 0.045 g/t
    d = resolve_tails(day(tails_grade_gpt=np.nan, tails_solid_gpt=0.20, tails_solution_mgl=0.03,
                          tails_pct_solids=40.0)).iloc[0]
    assert d["solution_loss_gpt"] == pytest.approx(0.045)
    assert d["tails_grade_gpt"] == pytest.approx(0.245)
    assert d["solution_loss_pct_of_tails"] == pytest.approx(0.045 / 0.245 * 100)


def test_components_take_priority_over_typed_total():
    d = resolve_tails(day(tails_grade_gpt=0.9, tails_solid_gpt=0.20, tails_solution_mgl=0.0,
                          tails_pct_solids=40.0)).iloc[0]
    assert d["tails_grade_gpt"] == pytest.approx(0.20)


def test_typed_total_used_when_no_components():
    d = resolve_tails(day()).iloc[0]
    assert d["tails_grade_gpt"] == pytest.approx(0.2)
    assert np.isnan(d["solution_loss_gpt"])


def test_tails_split_flows_into_recovery():
    r = compute_daily(day(tails_grade_gpt=np.nan, tails_solid_gpt=0.20, tails_solution_mgl=0.03,
                          tails_pct_solids=40.0)).iloc[0]
    assert r["tails_gold_g"] == pytest.approx(4000 * 0.245)
    assert r["recovery_grade_pct"] == pytest.approx((2.5 - 0.245) / 2.5 * 100)


def test_dissolved_gold_alert():
    d = compute_daily(day(tails_grade_gpt=np.nan, tails_solid_gpt=0.10, tails_solution_mgl=0.10,
                          tails_pct_solids=40.0))   # 0.15 g/t dissolved of 0.25 total = 60 %
    a = build_alerts(d, AlertThresholds())
    assert (a["type"] == "Dissolved gold loss").any()


def test_validation_accepts_split_tails_without_total_column():
    df = day().drop(columns=["tails_grade_gpt"])
    df["tails_solid_gpt"] = 0.2
    assert validate(df).query("severity=='error'").empty


def test_validation_flags_split_tails_above_head_and_missing_pct_solids():
    bad = day(tails_grade_gpt=np.nan, tails_solid_gpt=3.0, tails_solution_mgl=0.1, tails_pct_solids=40.0)
    assert ((validate(bad)["field"] == "tails_grade_gpt") & (validate(bad)["severity"] == "error")).any()
    warn = validate(day(tails_grade_gpt=np.nan, tails_solid_gpt=0.2, tails_solution_mgl=0.03))
    assert (warn["field"] == "tails_pct_solids").any()


def test_validation_still_reports_missing_tails_column():
    iss = validate(day().drop(columns=["tails_grade_gpt"]))
    assert "Missing required column" in iss.iloc[0]["message"]


# ================= circuit =================
def test_slurry_density_and_liquid_fraction():
    rho = float(circuit.slurry_density(40, 2.7))
    assert rho == pytest.approx(1 / (0.4 / 2.7 + 0.6))
    assert float(circuit.liquid_fraction(40, 2.7)) == pytest.approx(0.6 * rho)


def test_tank_gold_on_carbon_hand_calc():
    t = pd.DataFrame([dict(date="2026-08-01", tank=1, volume_m3=600, pct_solids=40, carbon_conc_gpl=15,
                           carbon_loading_gpt=800, solution_au_mgl=0.02)])
    r = circuit.tank_calcs(t).iloc[0]
    assert r["carbon_kg"] == pytest.approx(9000)          # 15 g/L x 600 m3
    assert r["gold_on_carbon_g"] == pytest.approx(7200)    # 9000 kg x 800 g/t / 1000
    liquid = 600 * 0.6 * float(circuit.slurry_density(40, 2.7))
    assert r["gold_in_solution_g"] == pytest.approx(0.02 * liquid)
    assert r["inventory_g"] == pytest.approx(7200 + 0.02 * liquid)


def test_adsorption_efficiency_between_tanks():
    t = pd.DataFrame([dict(date="2026-08-01", tank=1, solution_au_mgl=0.50),
                      dict(date="2026-08-01", tank=2, solution_au_mgl=0.20)])
    r = circuit.tank_calcs(t)
    assert np.isnan(r.loc[0, "adsorption_eff_pct"])
    assert r.loc[1, "adsorption_eff_pct"] == pytest.approx(60.0)


def _tank_rows(date, loadings):
    return [dict(date=date, tank=i + 1, volume_m3=100, carbon_conc_gpl=10, carbon_loading_gpt=L)
            for i, L in enumerate(loadings)]


def test_inventory_change_is_calculated_from_consecutive_days():
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"]),
                          "other_inventory_g": [100, 150, 150, 150],
                          "inventory_change_g": [np.nan, np.nan, 77.0, 55.0]})
    tanks = pd.DataFrame(_tank_rows("2026-08-01", [1000, 500]) + _tank_rows("2026-08-02", [1100, 600])
                         + _tank_rows("2026-08-03", [1100])          # incomplete (1 of 2 tanks)
                         + _tank_rows("2026-08-04", [1200, 700]))
    out, notes = circuit.apply_inventory(daily, tanks, n_tanks=2)
    # gold on carbon = conc x vol x loading / 1000 = loading (10 x 100 / 1000 = 1)
    assert out.loc[1, "inventory_change_g"] == pytest.approx((1100 + 600 + 150) - (1000 + 500 + 100))
    assert out.loc[1, "inventory_source"] == "auto"
    assert out.loc[0, "inventory_source"] == "none"          # no previous day
    assert out.loc[2, "inventory_change_g"] == 77.0 and out.loc[2, "inventory_source"] == "manual"
    assert out.loc[3, "inventory_change_g"] == 55.0          # previous day incomplete -> keeps manual
    assert len(notes) == 1 and "1 of 2" in notes.iloc[0]["message"]


def test_inventory_unchanged_without_tank_data():
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-08-01"]), "inventory_change_g": [120.0]})
    out, notes = circuit.apply_inventory(daily, pd.DataFrame(), n_tanks=6)
    assert out.loc[0, "inventory_change_g"] == 120.0 and notes.empty


def test_residence_time_hand_calc():
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-08-01"]), "tph": [165.0]})
    out = circuit.add_residence_time(daily, pd.DataFrame(), [800.0] * 6, circuit.CircuitLimits())
    rho = 1 / (0.4 / 2.7 + 0.6)
    flow = 165 / (0.4 * rho)
    assert out.loc[0, "residence_time_h"] == pytest.approx(4800 / flow)
    assert 15 < out.loc[0, "residence_time_h"] < 16


def test_circuit_alerts_group_tanks_per_day():
    t = circuit.tank_calcs(pd.DataFrame([
        dict(date="2026-08-01", tank=1, ph=9.2, do_ppm=4.0, free_cn_ppm=250, carbon_conc_gpl=15),
        dict(date="2026-08-01", tank=2, ph=10.6, do_ppm=4.5, free_cn_ppm=60, carbon_conc_gpl=15),
        dict(date="2026-08-02", tank=1, ph=10.6, do_ppm=8, free_cn_ppm=250, carbon_conc_gpl=15)]))
    a = circuit.circuit_alerts(t)
    assert set(a["type"]) == {"Low pH", "Low dissolved oxygen", "Low free cyanide"}
    do = a[a["type"] == "Low dissolved oxygen"].iloc[0]
    assert "tank 1" in do["message"] and "tank 2" in do["message"]
    assert a[a["type"] == "Low pH"].iloc[0]["severity"] == "critical"     # 9.2 is below 9.5
    assert (a["date"] == pd.Timestamp("2026-08-02")).sum() == 0


def test_carbon_stats_and_alerts():
    log = pd.DataFrame([
        dict(date="2026-08-01", event="Carbon addition", kg=400, activity_pct=np.nan),
        dict(date="2026-08-05", event="Carbon addition", kg=500, activity_pct=np.nan),
        dict(date="2026-08-02", event="Elution", kg=6000, activity_pct=np.nan),
        dict(date="2026-08-06", event="Regeneration", kg=3000, activity_pct=np.nan),
        dict(date="2026-08-03", event="Activity test", kg=np.nan, activity_pct=85),
        dict(date="2026-08-07", event="Activity test", kg=np.nan, activity_pct=50)])
    s = circuit.carbon_stats(log, 30000, "2026-08-10")
    assert s["carbon_added_kg"] == 900
    assert s["carbon_consumption_gpt"] == pytest.approx(900 * 1000 / 30000)     # 30 g/t
    assert s["days_since_elution"] == 8 and s["days_since_regen"] == 4
    assert s["latest_activity_pct"] == 50
    a = circuit.carbon_alerts(log)
    assert len(a) == 1 and a.iloc[0]["severity"] == "critical"                  # 50 < 70 - 15


# ================= QA/QC =================
def _qa(**kw):
    base = dict(date="2026-08-01", sample_id="S1", sample_type="Standard", lab="Site", assay_gpt=2.5,
                expected_gpt=2.5, sd_gpt=0.06, original_gpt=np.nan)
    base.update(kw)
    return base


def test_standard_z_and_status():
    ev = qaqc.evaluate(pd.DataFrame([_qa(assay_gpt=2.55), _qa(assay_gpt=2.63), _qa(assay_gpt=2.70)]))
    assert ev.loc[0, "z"] == pytest.approx(0.05 / 0.06)
    assert list(ev["status"]) == ["Pass", "Warning", "Fail"]
    assert ev.loc[2, "bias_pct"] == pytest.approx(8.0)


def test_blank_duplicate_and_incomplete():
    ev = qaqc.evaluate(pd.DataFrame([
        _qa(sample_type="Blank", assay_gpt=0.02), _qa(sample_type="Blank", assay_gpt=0.12),
        _qa(sample_type="Duplicate", assay_gpt=2.0, original_gpt=2.1),
        _qa(sample_type="Duplicate", assay_gpt=2.0, original_gpt=2.3),
        _qa(sample_type="Duplicate", assay_gpt=2.0, original_gpt=2.5),
        _qa(sample_type="Duplicate", assay_gpt=0.10, original_gpt=0.18),
        _qa(sample_type="Standard", assay_gpt=2.5, sd_gpt=np.nan)]))
    assert list(ev["status"]) == ["Pass", "Fail", "Pass", "Warning", "Fail", "Warning", "Incomplete"]
    assert ev.loc[3, "rd_pct"] == pytest.approx(0.3 / 2.15 * 100)


def test_westgard_rules():
    z = pd.Series([0.1, 2.5, 2.6, 0.0])
    assert list(qaqc.westgard(z)) == ["", "", "2_2s", ""]
    assert qaqc.westgard(pd.Series([3.5])).iloc[0] == "1_3s"
    assert qaqc.westgard(pd.Series([1.2, 1.3, 1.1, 1.4])).iloc[3] == "4_1s"
    eight = qaqc.westgard(pd.Series([0.3] * 8))
    assert eight.iloc[7] == "8x" and eight.iloc[6] == ""
    assert (qaqc.westgard(pd.Series([2.5, -2.6, 2.4])) == "").all()         # opposite sides do not count


def test_check_bias_and_summary():
    ev = qaqc.evaluate(pd.DataFrame([
        _qa(sample_type="Check assay", assay_gpt=2.04, original_gpt=2.0),
        _qa(sample_type="Check assay", assay_gpt=3.06, original_gpt=3.0),
        _qa(sample_type="Check assay", assay_gpt=1.02, original_gpt=1.0)]))
    b = qaqc.check_bias(ev)
    assert b["n"] == 3 and b["mean_bias_pct"] == pytest.approx(2.0)
    s = qaqc.summary(ev)
    assert s.loc[0, "sample_type"] == "Check assay" and s.loc[0, "pass_pct"] == 100


def test_qaqc_alerts():
    rows = [_qa(sample_id=f"S{i}", date=f"2026-08-{i + 1:02d}", assay_gpt=2.5 + 0.075) for i in range(9)]
    rows.append(_qa(sample_id="BAD", date="2026-08-20", assay_gpt=2.9))
    a = qaqc.qaqc_alerts(qaqc.evaluate(pd.DataFrame(rows)))
    assert (a["type"] == "QA/QC control rule").any()           # z=1.25 for 9 points: 4_1s and 8x
    assert (a["type"] == "QA/QC Standard fail").any()


# ================= SPC =================
BASE = [10, 10.2, 9.8, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 9.9]


def test_control_limits_hand_calc():
    x = pd.Series(BASE + [11.5])
    lim = spc.individuals_limits(x, baseline_n=10)
    mr_bar = np.mean([0.2, 0.4, 0.3, 0.2, 0.1, 0.2, 0.4, 0.3, 0.2])
    assert lim["center"] == pytest.approx(10.0)
    assert lim["sigma"] == pytest.approx(mr_bar / 1.128)
    assert lim["ucl"] == pytest.approx(10.0 + 3 * mr_bar / 1.128)


def test_control_table_flags_signal_and_shift():
    d = pd.DataFrame({"date": pd.date_range("2026-08-01", periods=11), "x": BASE + [11.5]})
    t, lim = spc.control_table(d, "x", baseline_n=10)
    assert t.loc[10, "flag"].startswith("beyond 3σ")
    assert (t.loc[:9, "flag"] == "").all()
    d2 = pd.DataFrame({"date": pd.date_range("2026-08-01", periods=18), "x": BASE + [10.3] * 8})
    t2, _ = spc.control_table(d2, "x", baseline_n=10)
    assert t2.loc[17, "flag"] == "8 in a row on one side" and t2.loc[16, "flag"] == ""


def test_spc_alerts_only_on_the_bad_side():
    d = pd.DataFrame({"date": pd.date_range("2026-08-01", periods=11),
                      "recovery_grade_pct": BASE + [12.0],      # far above: good for recovery
                      "tails_grade_gpt": BASE + [12.0]})        # far above: bad for tails
    a = spc.spc_alerts(d, baseline_n=10)
    assert len(a) == 1 and "Tails" in a.iloc[0]["message"]


# ================= gold room =================
def test_fine_gold_and_smelt_recovery():
    p = goldroom.prepare(pd.DataFrame([dict(date="2026-08-03", gross_g=1000, au_pct=90, ag_pct=8, slag_au_g=20)]))
    r = p.iloc[0]
    assert r["fine_au_g"] == 900 and r["fine_ag_g"] == 80
    assert r["smelt_recovery_pct"] == pytest.approx(900 / 920 * 100)
    assert r["ag_to_au"] == pytest.approx(8 / 90)


def test_monthly_fineness_is_weighted_and_reconciles():
    p = goldroom.prepare(pd.DataFrame([
        dict(date="2026-08-03", gross_g=1000, au_pct=90, ag_pct=8),
        dict(date="2026-08-20", gross_g=3000, au_pct=80, ag_pct=15)]))
    m = goldroom.monthly(p).iloc[0]
    assert m["fine_au_g"] == pytest.approx(900 + 2400)
    assert m["au_fineness_pct"] == pytest.approx(3300 / 4000 * 100)       # not the simple mean 85
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-08-01", "2026-08-02"]), "gold_poured_g": [1600.0, 1600.0]})
    rec = goldroom.reconcile(p, daily).iloc[0]
    assert rec["difference_g"] == pytest.approx(100) and rec["difference_pct"] == pytest.approx(100 / 3200 * 100)


# ================= ore types and targets =================
def test_ore_type_weighted_and_shares():
    df = pd.concat([day(date="2026-08-01", tonnes_milled=1000, head_grade_gpt=1.0, tails_grade_gpt=0.1,
                        gold_poured_g=900, ore_type="Oxide"),
                    day(date="2026-08-02", tonnes_milled=3000, head_grade_gpt=3.0, tails_grade_gpt=0.3,
                        gold_poured_g=8100, ore_type="Fresh")], ignore_index=True)
    o = by_ore_type(compute_daily(df)).set_index("ore_type")
    assert o.loc["Oxide", "share_pct"] == pytest.approx(25) and o.loc["Fresh", "recovery_grade_pct"] == pytest.approx(90)
    assert "Unspecified" in by_ore_type(compute_daily(day())).set_index("ore_type").index


def test_plan_vs_actual_status():
    actual = overall(compute_daily(day()))
    p = plan_vs_actual(actual, {"recovery_pct": 92.0, "tails_gpt": 0.15, "throughput_tph": 200.0,
                                "unaccounted_pct": 1.0}).set_index("metric")
    assert p.loc["Recovery (%)", "status"] == "On target"            # 92.0 >= 92.0
    assert p.loc["Tails grade (g/t)", "status"] == "Behind"          # 0.2 > 0.15 x 1.05
    assert p.loc["Throughput (t/h)", "status"] == "Behind"           # 181.8 < 0.98 x 200
    assert p.loc["Unaccounted gold (%)", "status"] == "Behind"       # 2.0 > 1.05
    assert p.loc["Recovery (%)", "variance"] == pytest.approx(0.0)


# ================= validation of new record types =================
def test_validate_tanks():
    t = pd.DataFrame([dict(date="2026-08-01", tank=1, volume_m3=800, ph=15, do_ppm=8),
                      dict(date="2026-08-01", tank=2, volume_m3=-5, ph=10.5, do_ppm=40),
                      dict(date="2026-08-01", tank=3, volume_m3=800, ph=10.5, carbon_conc_gpl=-1)])
    v = validate_tanks(t)
    errs = v[v["severity"] == "error"]
    assert set(errs["field"]) == {"ph", "volume_m3", "carbon_conc_gpl"}
    assert ((v["field"] == "do_ppm") & (v["severity"] == "warning")).any()


def test_validate_pours_qaqc_carbon():
    p = validate_pours(pd.DataFrame([dict(date="2026-08-01", gross_g=1000, au_pct=95, ag_pct=10),
                                     dict(date="2026-08-01", gross_g=0, au_pct=90, ag_pct=5)]))
    assert set(p["field"]) == {"ag_pct", "gross_g"}
    q = validate_qaqc(pd.DataFrame([dict(date="2026-08-01", sample_type="Oops", assay_gpt=1.0),
                                    dict(date="2026-08-01", sample_type="Blank", assay_gpt=-1)]))
    assert set(q["field"]) == {"sample_type", "assay_gpt"}
    c = validate_carbon(pd.DataFrame([dict(date="2026-08-01", event="Activity test", kg=np.nan, activity_pct=np.nan),
                                      dict(date="2026-08-01", event="Elution", kg=-4, activity_pct=np.nan)]))
    assert len(c) == 2


# ================= storage =================
def test_old_database_is_upgraded(tmp_path):
    import sqlite3
    p = tmp_path / "old.db"
    with sqlite3.connect(p) as c:
        c.execute("CREATE TABLE daily (date TEXT PRIMARY KEY, tonnes_milled REAL, mill_hours REAL, scheduled_hours REAL,"
                  " head_grade_gpt REAL, tails_grade_gpt REAL, gold_poured_g REAL, inventory_change_g REAL,"
                  " nacn_kg REAL, lime_kg REAL, power_kwh REAL, model_grade_gpt REAL)")
        c.execute("INSERT INTO daily VALUES ('2026-08-01',4000,22,24,2.5,0.2,9000,0,2200,6400,88000,NULL)")
    out = storage.load(p)
    assert len(out) == 1 and out.loc[0, "tonnes_milled"] == 4000
    assert {"tails_solid_gpt", "ore_type", "other_inventory_g"} <= set(out.columns)


def test_tanks_carbon_pours_roundtrip(tmp_path):
    p = tmp_path / "t.db"
    day1 = pd.DataFrame(_tank_rows("2026-08-01", [1000, 500]))
    storage.save_rows("tanks", day1, p)
    storage.replace_day("tanks", "2026-08-01", pd.DataFrame(_tank_rows("2026-08-01", [900])), p)   # edit the day
    t = storage.load_table("tanks", p)
    assert len(t) == 1 and t.loc[0, "carbon_loading_gpt"] == 900 and t.loc[0, "tank"] == 1
    storage.save_rows("carbon_log", pd.DataFrame([dict(date="2026-08-01", event="Elution", kg=6000)]), p)
    log = storage.load_table("carbon_log", p)
    assert len(log) == 1
    storage.delete_ids("carbon_log", log["id"], p)
    assert storage.load_table("carbon_log", p).empty
    storage.save_rows("pours", pd.DataFrame([dict(date="2026-08-02", bar_no="B1", gross_g=1000, au_pct=90)]), p)
    assert storage.load_table("pours", p).loc[0, "bar_no"] == "B1"


def test_delete_day_also_removes_its_tank_readings(tmp_path):
    p = tmp_path / "t.db"
    storage.upsert(day(), p)
    storage.save_rows("tanks", pd.DataFrame(_tank_rows("2026-08-01", [1000])), p)
    storage.delete("2026-08-01", p)
    assert storage.load(p).empty and storage.load_table("tanks", p).empty


def test_settings_merge_with_defaults(tmp_path):
    p = tmp_path / "t.db"
    s = storage.get_settings(p)
    assert s["targets"]["recovery_pct"] == 92.0
    s["targets"]["recovery_pct"] = 93.5
    storage.save_settings({"targets": s["targets"]}, p)
    s2 = storage.get_settings(p)
    assert s2["targets"]["recovery_pct"] == 93.5 and s2["targets"]["head_gpt"] == 2.5   # untouched keys survive
    assert s2["circuit"]["tank_volumes_m3"] == [800.0] * 6


def test_backup_and_restore(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    storage.upsert(day(ore_type="Oxide"), a)
    storage.save_rows("tanks", pd.DataFrame(_tank_rows("2026-08-01", [1000, 500])), a)
    storage.save_rows("qaqc", pd.DataFrame([dict(date="2026-08-01", sample_id="S1", sample_type="Blank", assay_gpt=0.01)]), a)
    storage.save_settings({"plant_name": "Plant A"}, a)
    out = storage.restore_zip(storage.backup_zip(a), b)
    assert out["daily"] == 1 and out["tanks"] == 2 and out["qaqc"] == 1
    assert storage.load(b).loc[0, "ore_type"] == "Oxide"
    assert storage.get_settings(b)["plant_name"] == "Plant A"
