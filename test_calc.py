import numpy as np
import pandas as pd
import pytest

from calc import (G_PER_OZ, AlertThresholds, Costs, build_alerts, compute_daily,
                  mine_call_factor, summarise, two_product_recovery)
from sample_data import generate
from validation import error_rows, validate
import storage


def day(**kw):
    base = dict(date="2026-08-01", tonnes_milled=4000.0, mill_hours=22.0, scheduled_hours=24.0,
                head_grade_gpt=2.5, tails_grade_gpt=0.2, gold_poured_g=9000.0,
                inventory_change_g=0.0, nacn_kg=2200.0, lime_kg=6400.0, power_kwh=88000.0)
    base.update(kw)
    return pd.DataFrame([base])


# ---------- calculations ----------
def test_two_product_formula_hand_calc():
    # f=2, c=50, t=0.2 -> 50*(1.8)/(2*49.8)*100 = 90.36...
    assert two_product_recovery(2, 50, 0.2) == pytest.approx(90.3614, abs=1e-3)


def test_two_product_undefined_when_c_equals_t():
    assert np.isnan(two_product_recovery(2, 0.2, 0.2))


def test_basic_daily_numbers():
    d = compute_daily(day()).iloc[0]
    assert d["feed_gold_g"] == pytest.approx(10000)
    assert d["tails_gold_g"] == pytest.approx(800)
    assert d["recovery_grade_pct"] == pytest.approx(92.0)
    assert d["recovery_accounted_pct"] == pytest.approx(90.0)
    assert d["unaccounted_g"] == pytest.approx(200)          # 10000 - 800 - 9000
    assert d["unaccounted_pct"] == pytest.approx(2.0)
    assert d["calc_head_gpt"] == pytest.approx(2.45)
    assert d["head_gap_pct"] == pytest.approx(2.0)
    assert d["tph"] == pytest.approx(4000 / 22)
    assert d["availability_pct"] == pytest.approx(22 / 24 * 100)
    assert d["nacn_kgpt"] == pytest.approx(0.55)
    assert d["gold_recovered_oz"] == pytest.approx(9000 / G_PER_OZ)


def test_mass_balance_identity_on_sample():
    d = compute_daily(generate())
    ok = d["feed_gold_g"].notna()
    resid = d["feed_gold_g"] - d["tails_gold_g"] - d["gold_recovered_g"] - d["unaccounted_g"]
    assert resid[ok].abs().max() < 1e-6


def test_inventory_change_counts_as_recovered():
    d = compute_daily(day(gold_poured_g=8500.0, inventory_change_g=500.0)).iloc[0]
    assert d["gold_recovered_g"] == pytest.approx(9000)


def test_monthly_summary_is_tonnage_weighted_not_averaged():
    df = pd.concat([day(date="2026-08-01", tonnes_milled=1000, head_grade_gpt=1.0, tails_grade_gpt=0.1,
                        gold_poured_g=900),
                    day(date="2026-08-02", tonnes_milled=9000, head_grade_gpt=3.0, tails_grade_gpt=0.3,
                        gold_poured_g=24300)])
    s = summarise(compute_daily(df)).iloc[0]
    # weighted head = (1000*1 + 9000*3)/10000 = 2.8, NOT the simple mean 2.0
    assert s["head_grade_gpt"] == pytest.approx(2.8)
    assert s["recovery_grade_pct"] == pytest.approx(90.0)


def test_mine_call_factor():
    assert mine_call_factor(9500, 10000) == pytest.approx(0.95)
    assert np.isnan(mine_call_factor(1, 0))


def test_zero_tonnes_does_not_crash():
    d = compute_daily(day(tonnes_milled=0, mill_hours=0, gold_poured_g=0, nacn_kg=0, lime_kg=0, power_kwh=0))
    assert d.loc[0, "recovery_accounted_pct"] != d.loc[0, "recovery_accounted_pct"]  # NaN, no exception


# ---------- validation ----------
def test_valid_day_has_no_errors():
    assert validate(day()).query("severity=='error'").empty


def test_tails_above_head_is_error():
    iss = validate(day(tails_grade_gpt=24.0))
    assert ((iss["field"] == "tails_grade_gpt") & (iss["severity"] == "error")).any()


def test_hours_over_24_and_over_scheduled_are_errors():
    iss = validate(day(mill_hours=26.0))
    assert (iss["field"] == "mill_hours").sum() >= 1
    assert error_rows(iss) == {0}


def test_impossible_pour_is_error():
    iss = validate(day(gold_poured_g=50000.0))
    assert ((iss["field"] == "gold_poured_g") & (iss["severity"] == "error")).any()


def test_duplicate_and_missing_dates():
    df = pd.concat([day(date="2026-08-01"), day(date="2026-08-01"), day(date="2026-08-04")], ignore_index=True)
    iss = validate(df)
    assert (iss["message"].str.contains("Duplicate")).sum() == 2
    assert (iss["message"].str.contains("gap")).sum() == 2  # Aug 2 and Aug 3


def test_negative_and_missing_values():
    iss = validate(day(nacn_kg=-5.0))
    assert ((iss["field"] == "nacn_kg") & (iss["severity"] == "error")).any()
    d = day()
    d.loc[0, "head_grade_gpt"] = np.nan
    assert ((validate(d)["field"] == "head_grade_gpt")).any()


def test_missing_required_column():
    iss = validate(day().drop(columns=["tails_grade_gpt"]))
    assert "Missing required column" in iss.iloc[0]["message"]


def test_sample_data_issues_are_caught():
    raw = generate()
    iss = validate(raw)
    assert ((iss["field"] == "tails_grade_gpt") & (iss["severity"] == "error")).any()
    assert iss["message"].str.contains("gap").any()


# ---------- alerts ----------
def test_recovery_drop_alert():
    rows = [day(date=f"2026-08-{i:02d}", tails_grade_gpt=0.2) for i in range(1, 9)]
    rows.append(day(date="2026-08-09", tails_grade_gpt=0.45))  # 92% -> 82%
    alerts = build_alerts(compute_daily(pd.concat(rows, ignore_index=True)))
    assert (alerts["type"] == "Recovery drop").any()
    assert alerts[alerts["type"] == "Recovery drop"].iloc[0]["severity"] == "critical"


def test_no_alert_on_steady_plant():
    rows = [day(date=f"2026-08-{i:02d}", gold_poured_g=9100.0) for i in range(1, 12)]
    alerts = build_alerts(compute_daily(pd.concat(rows, ignore_index=True)))
    assert alerts.empty


def test_unaccounted_gold_alert():
    d = compute_daily(day(gold_poured_g=8300.0))  # 9% unaccounted
    assert (build_alerts(d)["type"] == "Unaccounted gold").any()


def test_sample_dataset_triggers_expected_alerts():
    raw = generate()
    iss = validate(raw)
    d = compute_daily(raw)
    types = set(build_alerts(d, AlertThresholds())["type"])
    assert "Recovery drop" in types
    assert "Cyanide spike" in types


# ---------- storage ----------
def test_storage_roundtrip(tmp_path):
    p = tmp_path / "t.db"
    storage.upsert(day(), p)
    storage.upsert(day(head_grade_gpt=3.0), p)      # same date -> overwrite
    out = storage.load(p)
    assert len(out) == 1 and out.loc[0, "head_grade_gpt"] == 3.0
    storage.delete("2026-08-01", p)
    assert storage.load(p).empty
