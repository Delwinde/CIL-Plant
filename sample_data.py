"""Generate a realistic synthetic gold CIL plant dataset for demos and tests.

generate_all() returns every table: daily, tanks, carbon_log, qaqc, pours.
generate() returns just the daily table.

The data is internally consistent: the daily gold poured is calculated from the gold recovered minus the
change in the in-circuit inventory that the tank readings imply, so the mass balance closes the way it
does in a well-run plant (about 1 % noise). It contains deliberate, documented problems so validation,
alerts, QA/QC rules and control charts can be shown:

  * a 6-day recovery dip in late July: carbon fouling (low activity tests), low dissolved oxygen in the
    first two tanks and a high dissolved-gold loss in the tails solution
  * a typo where solid tails > head grade
  * one cyanide dosing spike
  * a missing day
  * a period of unaccounted gold (pour not matching the books) in September
  * a lab drift in the standards (positive bias) in September, one failed standard,
    one contaminated blank and two duplicate pairs that do not agree
Ore type changes in campaigns of 5 to 15 days (oxide, transition, fresh), each with its own recovery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import circuit
from calc import G_PER_OZ

ORE_HEAD = {"Oxide": 2.2, "Transition": 2.6, "Fresh": 3.0}
ORE_REC = {"Oxide": 93.4, "Transition": 92.3, "Fresh": 90.9}
N_TANKS = 6
TANK_VOL = 800.0


def generate_all(start="2026-06-01", end="2026-09-30", seed=7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # ---- ore campaigns -------------------------------------------------
    ore = np.empty(n, dtype=object)
    i = 0
    while i < n:
        length = int(rng.integers(5, 16))
        ore[i:i + length] = rng.choice(list(ORE_HEAD), p=[0.3, 0.4, 0.3])
        i += length

    # ---- plant running ------------------------------------------------
    sched = np.full(n, 24.0)
    hours = np.clip(rng.normal(21.8, 1.4, n), 12, 24)
    hours[rng.random(n) < 0.04] = 0           # full-day stoppages
    tph = rng.normal(165, 8, n)
    tonnes = hours * tph

    mean_head = np.array([ORE_HEAD[o] for o in ore])
    head = np.clip(rng.lognormal(mean=np.log(mean_head), sigma=0.16), 0.8, 8)
    model = head * rng.normal(1.04, 0.06, n)

    base_rec = np.array([ORE_REC[o] for o in ore]) + 0.35 * (head - mean_head) + rng.normal(0, 0.7, n)
    base_rec = np.clip(base_rec / 100, 0.6, 0.97)
    dip = slice(55, 61)
    base_rec[dip] -= 0.045                    # carbon fouling / low oxygen

    # ---- tails split ---------------------------------------------------
    pct_solids = np.clip(rng.normal(40, 1.2, n), 34, 46)
    ratio = (100 - pct_solids) / pct_solids

    # ---- circuit: tank readings --------------------------------------------
    base_load = np.array([1800, 1200, 800, 500, 300, 200.0])
    f = np.ones(n)
    for t in range(1, n):
        f[t] = np.clip(f[t - 1] + 0.15 * (1 - f[t - 1]) + rng.normal(0, 0.012), 0.85, 1.15)
    cf = np.ones(n)
    for t in range(1, n):
        cf[t] = np.clip(cf[t - 1] + 0.2 * (1 - cf[t - 1]) + rng.normal(0, 0.004), 0.9, 1.1)
    loading = base_load[None, :] * f[:, None] * (1 + rng.normal(0, 0.001, (n, N_TANKS)))
    conc = 15.0 * cf[:, None] * (1 + rng.normal(0, 0.001, (n, N_TANKS)))
    sol = np.tile([0.60, 0.30, 0.14, 0.07, 0.035, 0.016], (n, 1))
    sol[dip] = [0.60, 0.42, 0.30, 0.22, 0.17, 0.12]
    sol = sol * (1 + rng.normal(0, 0.05, (n, N_TANKS)))
    cn = np.linspace(250, 150, N_TANKS)[None, :] + rng.normal(0, 12, (n, N_TANKS))
    ph = rng.normal(10.6, 0.12, (n, N_TANKS))
    do = rng.normal(8.2, 0.5, (n, N_TANKS))
    do[dip, :2] = rng.normal(4.6, 0.4, (6, 2))
    ps_tank = np.clip(pct_solids[:, None] + rng.normal(0, 0.4, (n, N_TANKS)), 30, 50)

    rows = []
    for t, dt in enumerate(dates):
        for k in range(N_TANKS):
            rows.append(dict(date=dt, tank=k + 1, volume_m3=TANK_VOL, pct_solids=round(ps_tank[t, k], 1),
                             free_cn_ppm=round(cn[t, k]), ph=round(ph[t, k], 2), do_ppm=round(do[t, k], 1),
                             carbon_conc_gpl=round(conc[t, k], 2), carbon_loading_gpt=round(loading[t, k]),
                             solution_au_mgl=round(sol[t, k], 3)))
    tanks = pd.DataFrame(rows)

    # gold held outside the tanks (elution column, cells, kiln carbon)
    other = np.empty(n)
    other[0] = 4000
    for t in range(1, n):
        other[t] = np.clip(other[t - 1] + rng.normal(0, 40), 2500, 6000)
    other = other.round()

    # ---- inventory and gold poured -----------------------------------------
    inv = circuit.daily_inventory(circuit.tank_calcs(tanks), N_TANKS).set_index("date")["tank_inventory_g"]
    total_inv = inv.reindex(dates).to_numpy() + other
    delta = np.concatenate([[0.0], np.diff(total_inv)])

    feed_gold = tonnes * head
    recovered_true = feed_gold * base_rec
    poured = recovered_true * rng.normal(1.0, 0.015, n) - delta
    poured[100:108] *= 0.93                   # unaccounted gold in September
    poured = np.clip(poured, 0, None)

    nacn = tonnes * np.clip(rng.normal(0.55, 0.04, n), 0.3, 1)
    lime = tonnes * np.clip(rng.normal(1.6, 0.15, n), 0.8, 3)
    power = tonnes * np.clip(rng.normal(22, 1.5, n), 12, 40)

    # tails: total from the head and recovery, solution part measured in the last tank
    tails_total = head * (1 - base_rec)
    sol_mgl = sol[:, -1]
    solution_loss = sol_mgl * ratio
    tails_solid = np.maximum(tails_total - solution_loss, 0.01)

    daily = pd.DataFrame({
        "date": dates,
        "tonnes_milled": tonnes.round(0),
        "mill_hours": hours.round(1),
        "scheduled_hours": sched,
        "head_grade_gpt": head.round(2),
        "tails_grade_gpt": np.nan,              # worked out from the parts below
        "gold_poured_g": poured.round(0),
        "inventory_change_g": np.nan,           # calculated from the tank readings
        "nacn_kg": nacn.round(0),
        "lime_kg": lime.round(0),
        "power_kwh": power.round(0),
        "model_grade_gpt": model.round(2),
        "tails_solid_gpt": tails_solid.round(3),
        "tails_solution_mgl": sol_mgl.round(3),
        "tails_pct_solids": pct_solids.round(1),
        "other_inventory_g": other,
        "ore_type": ore,
    })

    z = daily["mill_hours"] == 0
    daily.loc[z, ["tonnes_milled", "nacn_kg", "lime_kg", "power_kwh"]] = 0

    # ---- injected problems --------------------------------------------------
    daily.loc[30, ["head_grade_gpt", "tails_solid_gpt"]] = [2.40, 24.0]    # typo: 0.24 -> 24.0
    daily.loc[75, "nacn_kg"] = (daily.loc[75, "nacn_kg"] * 1.6).round(0)    # cyanide spike
    # the day after the missing day cannot use the automatic inventory change: give it a typed value
    daily.loc[46, "inventory_change_g"] = round(delta[46])
    missing = dates[45]
    daily = daily[daily["date"] != missing].reset_index(drop=True)
    tanks = tanks[tanks["date"] != missing].reset_index(drop=True)

    # ---- carbon log -----------------------------------------------------------
    log = []
    for t in range(2, n, 3):
        log.append(dict(date=dates[t], event="Elution", kg=6000.0, activity_pct=np.nan, note="Batch elution"))
    for t in range(1, n, 5):
        log.append(dict(date=dates[t], event="Carbon addition", kg=float(round(rng.normal(470, 40))),
                        activity_pct=np.nan, note="Fresh carbon"))
    for t in range(4, n, 7):
        log.append(dict(date=dates[t], event="Regeneration", kg=3000.0, activity_pct=np.nan, note="Kiln"))
    for t in range(3, n, 7):
        act = rng.normal(60, 3) if 50 <= t <= 62 else rng.normal(88, 3)
        log.append(dict(date=dates[t], event="Activity test", kg=np.nan, activity_pct=round(act),
                        note="Fouled carbon" if 50 <= t <= 62 else ""))
    carbon_log = pd.DataFrame(log).sort_values("date").reset_index(drop=True)
    carbon_log = carbon_log[carbon_log["date"] != missing].reset_index(drop=True)

    # ---- lab QA/QC ----------------------------------------------------------
    qa = []
    for t, dt in enumerate(dates):
        if dt == missing:
            continue
        bias = 0.09 if 90 <= t < 100 else 0.0
        std = 2.50 + bias + rng.normal(0, 0.04)
        if t == 20:
            std = 2.50 + 0.21                                               # failed standard
        qa.append(dict(date=dt, sample_id=f"STD-{t:03d}", sample_type="Standard", lab="Site lab",
                       assay_gpt=round(std, 3), expected_gpt=2.50, sd_gpt=0.06, original_gpt=np.nan,
                       note="CRM 2.50 g/t"))
        if t % 2 == 0:
            blank = 0.12 if t == 40 else max(0.0, rng.normal(0.01, 0.004))
            qa.append(dict(date=dt, sample_id=f"BLK-{t:03d}", sample_type="Blank", lab="Site lab",
                           assay_gpt=round(blank, 3), expected_gpt=np.nan, sd_gpt=np.nan, original_gpt=np.nan,
                           note="Barren quartz"))
        orig = float(daily.loc[daily["date"] == dt, "head_grade_gpt"].iloc[0]) if dt != dates[30] else float(head[30])
        dup = orig * (1.28 if t in (60, 61) else rng.normal(1.0, 0.03))
        qa.append(dict(date=dt, sample_id=f"DUP-{t:03d}", sample_type="Duplicate", lab="Site lab",
                       assay_gpt=round(dup, 3), expected_gpt=np.nan, sd_gpt=np.nan, original_gpt=round(orig, 3),
                       note="Head sample duplicate"))
        if t % 7 == 5:
            chk = orig * rng.normal(1.02, 0.015)
            qa.append(dict(date=dt, sample_id=f"CHK-{t:03d}", sample_type="Check assay", lab="Umpire lab",
                           assay_gpt=round(chk, 3), expected_gpt=np.nan, sd_gpt=np.nan, original_gpt=round(orig, 3),
                           note="Umpire check"))
    qaqc = pd.DataFrame(qa)

    # ---- pours: a bar every 3 days from the daily fine gold ----------------------
    d2 = daily[["date", "gold_poured_g"]].reset_index(drop=True)
    pours, last, k = [], 0, 1
    for idx in range(len(d2)):
        if idx % 3 == 2 or idx == len(d2) - 1:
            fine = float(d2.loc[last:idx, "gold_poured_g"].sum())
            if fine > 0:
                au = rng.normal(88.5, 1.2)
                pours.append(dict(date=d2.loc[idx, "date"], bar_no=f"B{k:03d}", gross_g=round(fine / (au / 100)),
                                  au_pct=round(au, 1), ag_pct=round(rng.normal(8.5, 1.0), 1),
                                  slag_au_g=round(fine * rng.normal(0.012, 0.003))))
                k += 1
            last = idx + 1
    pours = pd.DataFrame(pours)

    return {"daily": daily, "tanks": tanks, "carbon_log": carbon_log, "qaqc": qaqc, "pours": pours}


def generate(start="2026-06-01", end="2026-09-30", seed=7) -> pd.DataFrame:
    """Just the daily production table (kept for simple demos and tests)."""
    return generate_all(start, end, seed)["daily"]


if __name__ == "__main__":
    out = generate()
    out.to_csv("sample_plant_data.csv", index=False)
    print(f"wrote {len(out)} rows to sample_plant_data.csv")
