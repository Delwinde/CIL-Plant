"""Generate a realistic synthetic gold CIL plant dataset for demos and tests.

Contains deliberate, documented problems so validation and alerts can be shown:
  * a 6-day recovery dip (poor oxygen / carbon activity) in late July
  * a typo where tails grade > head grade
  * one cyanide dosing spike
  * a missing day
  * a period of unaccounted gold (pour not matching the books) in September
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calc import G_PER_OZ


def generate(start="2026-06-01", end="2026-09-30", seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    sched = np.full(n, 24.0)
    hours = np.clip(rng.normal(21.8, 1.4, n), 12, 24)
    hours[rng.random(n) < 0.04] = 0  # full-day stoppages are rare; set below to 0 tonnes
    tph = rng.normal(165, 8, n)
    tonnes = hours * tph

    head = np.clip(rng.lognormal(mean=np.log(2.6), sigma=0.18, size=n), 0.8, 8)
    model = head * rng.normal(1.04, 0.06, n)  # model slightly optimistic

    base_rec = rng.normal(92.5, 0.8, n) / 100
    base_rec[55:61] -= 0.045  # recovery dip
    tails = head * (1 - base_rec)

    inv_change = rng.normal(0, 150, n)
    metal_in = tonnes * head
    recovered = metal_in * base_rec  # what the books should show
    # real-world noise: pour timing, sampling error ~1.5%
    poured = recovered * rng.normal(1.0, 0.015, n) - inv_change
    # unaccounted gold period in September
    poured[100:108] *= 0.93

    nacn = tonnes * np.clip(rng.normal(0.55, 0.04, n), 0.3, 1)
    lime = tonnes * np.clip(rng.normal(1.6, 0.15, n), 0.8, 3)
    power = tonnes * np.clip(rng.normal(22, 1.5, n), 12, 40)

    df = pd.DataFrame({
        "date": dates,
        "tonnes_milled": tonnes.round(0),
        "mill_hours": hours.round(1),
        "scheduled_hours": sched,
        "head_grade_gpt": head.round(2),
        "tails_grade_gpt": tails.round(3),
        "gold_poured_g": np.clip(poured, 0, None).round(0),
        "inventory_change_g": inv_change.round(0),
        "nacn_kg": nacn.round(0),
        "lime_kg": lime.round(0),
        "power_kwh": power.round(0),
        "model_grade_gpt": model.round(2),
    })

    # zero-hour days: nothing produced
    z = df["mill_hours"] == 0
    df.loc[z, ["tonnes_milled", "gold_poured_g", "nacn_kg", "lime_kg", "power_kwh"]] = 0
    df.loc[z, "inventory_change_g"] = 0

    # Injected problems
    df.loc[30, ["head_grade_gpt", "tails_grade_gpt"]] = [2.40, 24.0]   # typo: 0.24 -> 24.0
    df.loc[75, "nacn_kg"] = (df.loc[75, "nacn_kg"] * 1.6).round(0)      # cyanide spike
    df = df.drop(index=45).reset_index(drop=True)                       # missing day
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("sample_plant_data.csv", index=False)
    print(f"wrote {len(out)} rows to sample_plant_data.csv")
