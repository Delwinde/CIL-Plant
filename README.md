# Gold CIL Plant: Metallurgical Accounting & Performance Dashboard

Daily plant figures in; recovery, gold reconciliation, circuit control, lab quality, gold room and
PDF reports out.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Load demo data** in the sidebar to see everything working (122 days of synthetic data with
deliberate faults), or import your own CSV/Excel (**Download blank template** gives the columns).
Tests: `python -m pytest -q`.

## What is inside
**Core:** daily entry with validation, recovery two ways (assay and gold produced), reconciliation,
tonnage-weighted summaries, reagents and cost per t/oz, alerts, data-quality checks, Excel export.

**Phase 1 - Tails and inventory.** Total tails = solid tails + solution loss
(solution mg/L x (100 - %solids)/%solids). In-circuit gold inventory is calculated from tank and carbon
data; the day-to-day change replaces the typed inventory figure when both days are complete.

**Phase 2 - Circuit.** Per-tank leach profile (Au in solution, DO, pH, cyanide), residence time,
carbon management (g/L, loading, activity, elutions, regenerations, carbon consumption).

**Phase 3 - Ore and lab.** Ore type tracking and recovery by ore type; lab QA/QC (standards, blanks,
duplicates, check assays, Westgard rules); SPC control charts (individuals charts, 3-sigma, run rules).

**Phase 4 - Gold room and reports.** Dore bars, fine gold, fineness, smelt recovery, reconciliation of
bars against daily poured gold; targets and plan vs actual; daily (1 page) and monthly (2 page) PDFs.

**Ore grade scanner (new tab).** Photograph crushed rock (camera or upload). The app outlines the rock and
measures dark sulphide area, rust, quartz, colour, texture and fragment size, using the empty tray as the
exposure/colour reference. Gold is invisible in a photo, so the grade comes only from a ridge-regression
calibration learned from YOUR scans that have lab assays (needs 15+, aim for 40+), tested leave-one-out.
Until calibrated it shows measurements only; it reports typical error, flags poor calibration and photos unlike
the calibration set, and calls Above / Below / Near the cut-off (near = send to lab). Demo scans use computer-drawn
photos with an invented grade link: they show how it works, not how accurate it will be on your ore.
Photo method must stay identical: same tray, light, camera height, dry single layer.

## Calibrate before real use
All limits (cyanide, DO, pH, carbon loading/activity, solution loss, QA/QC tolerances, targets) are
generic defaults. Set them in **Settings** to your plant's values. Tank volumes must be entered.
Known simplification: gold in ore solids inside the tanks is not counted in inventory.

## Storage, backup and privacy
Data lives in SQLite (`plant.db`). Set `PLANT_DB_PATH` to put it on a persistent disk.
On streamlit.app local files can be wiped on restart and the app is public unless you restrict it:
use **Settings > Backup** regularly and prefer private hosting for real data. There is no login yet.
