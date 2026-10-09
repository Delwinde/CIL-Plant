# Gold CIL Plant: Metallurgical Accounting & Performance Dashboard

Turns daily plant figures into recovery, gold reconciliation, reagent/energy cost, alerts
and management reports. Built around how a plant metallurgist actually checks the books.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **Load demo data** in the sidebar to see it working (4 months of synthetic data with
deliberate faults), or import your own CSV/Excel. Use **Download blank CSV template** for
the exact column names. Data is stored in `plant.db` (SQLite) next to the app; no internet needed.

## Daily inputs (one row per day)

| Column | Unit | Notes |
|---|---|---|
| date | | one record per day |
| tonnes_milled | dry t | |
| mill_hours / scheduled_hours | h | each ≤ 24 |
| head_grade_gpt | g/t | assayed feed grade |
| tails_grade_gpt | g/t | final residue |
| gold_poured_g | g | dore / gold produced |
| inventory_change_g | g | change in carbon/tank gold; + = built up. Blank = 0 |
| nacn_kg, lime_kg, power_kwh | | daily usage |
| model_grade_gpt | g/t | optional; grade-control/model grade for mine call factor |

## What it calculates

- **Recovery two independent ways**: from assays `(head - tails) / head`, and from gold actually
  produced `(poured + inventory change) / feed gold`. Disagreement = a sampling/accounting problem.
- **Gold reconciliation**: feed = tails + recovered + **unaccounted**. Calculated head grade vs
  assayed head grade.
- **Monthly/weekly summaries are tonnage-weighted** from contained metal. Grades and recoveries are
  never simply averaged.
- Throughput (t/h), utilisation, availability, NaCN / lime / power per tonne, process cost per
  tonne and per ounce, mine call factor.
- `two_product_recovery(f, c, t)` is available in `calc.py` for the classic concentrator formula.

## Data validation (the "never trust the data" layer)

Errors (excluded from KPIs by default): tails > head, mill hours > 24 or > scheduled, tonnes with
zero hours, negative values, gold recovered greater than gold fed, duplicate dates, missing or
non-numeric fields. Warnings: values outside plausible ranges (editable in `validation.py`),
missing days, zero-tonnage days. The daily entry form blocks saving if there are errors.

## Alerts (thresholds editable in the sidebar)

Recovery more than 2 points below the previous 7-day mean · unaccounted gold ≥ 3% (warning) / 5%
(critical) of feed · assay vs calculated head mismatch > 5% · cyanide > 1.25 × 7-day mean · low
availability.

## Tests

```bash
python -m pytest -q
```

21 tests cover the formulas against hand calculations, the mass-balance identity, weighted
summaries, every validation rule, the alert logic, and the database.

## Before using on a real plant, calibrate these

1. **Plausible ranges** in `validation.py` (`LIMITS`) are generic CIL values. Set them to your plant.
2. **Tails grade** is treated as one figure. If you sample solids and solution tails separately,
   add the solution loss to the tails figure first.
3. **Unit costs** in the sidebar are placeholders.
4. **Inventory change** must come from your own carbon/in-circuit inventory method. If it's left at
   0, normal inventory swings will show up as unaccounted gold.
5. Check the formulas with your own senior metallurgist against one month of real data before
   trusting the alerts.

## Files

`app.py` dashboard · `calc.py` calculations & alerts · `validation.py` checks · `storage.py` SQLite ·
`sample_data.py` demo generator · `test_calc.py` tests
