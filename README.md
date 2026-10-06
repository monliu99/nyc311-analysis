# NYC 311 lab (MGT 634, Lab Day 2)

The brief is `brief/Lab_Day_2_Challenge.pdf`: 11 questions on NYC 311 service requests,
answered from 2,000 JSON batch files (about 23 GB, about 20M rows) that only exist on the
cluster. Everything runs with one command and reproduces every number.

```
python run_all.py --data "/path/to/311_batch_*.json" --work /persistent/path/work
```

`lab.ipynb` runs the same stages one cell at a time.

## Status

| Stage | What it does | State |
|---|---|---|
| 0 env | Python, package versions, CPUs, RAM, free disk | done |
| 1 ingest | Checks every file, recovers records from cut-off files, converts JSON to all-text Parquet (only new files) | done |
| 2 discover | Describes the data's structure (reports `02a`-`02f`) before any question is answered | done; **waiting on cluster run #1** |
| 3 clean | Typed, flagged Parquet + shared aggregate tables; rules settled from run #1 | not started |
| Q1-Q11 | One module per question | not started |

## Running on the cluster

**The cluster (checked 2026-10-06):** one Jupyter kernel, Python 3.9.24 with pip 21.3.1 (the shared Python isn't
writable, so packages go to `~/.local` with `--user`); the job has 4 CPUs (`SLURM_CPUS_PER_TASK=4`) and 32 GB
(`SLURM_MEM_PER_NODE=32768`) on a 128-CPU node; workers start with `fork`. The data is
`/gpfs/project/esai_2026/data/311/311_batch_*.json`: 2,824 files, 33.3 GB, 28.24M rows, batch numbers 0001-2843
(about 20 numbers missing; 02a lists them). Treat it as read-only. The code works on Python 3.9-3.14 with
duckdb >= 1.1, pandas >= 2.0 and pyarrow >= 15. The full test suite passes on the cluster's exact stack (duckdb 1.4.5,
pandas 2.3.3, numpy 2.0.2, pyarrow 21.0.0) and on the current one (Python 3.14, duckdb 1.5, pandas 3.0).

**Steps** (every run)

1. **Build the upload zip** here: `python3 tools/make_upload_zip.py` (standard library only). It writes
   `../nyc311_lab_upload.zip`, next to this folder (this folder is only read): `nyc311/`, `run_all.py`, `lab.ipynb`,
   `requirements.txt`, `README.md`, `data/external/`, `report/prose/` and `VERSION.txt` (the commit and the question
   stages included). It stops if any of that code has uncommitted changes, so commit first; your prose in
   `report/prose/*.md` goes in as it is now, committed or not.
2. **Upload** `nyc311_lab_upload.zip` with the JupyterLab file browser's Upload button (the up arrow). Don't
   double-click the zip; if it opens anyway, the "not UTF-8" error is harmless: close that tab.
3. **Unzip** it with a Python cell in the same folder (File > New > Notebook):
   ```python
   import os, time, zipfile
   if os.path.exists("nyc311_lab"):  # keep the previous upload and its outputs aside instead of mixing files
       os.rename("nyc311_lab", time.strftime("nyc311_lab_old_%Y%m%d_%H%M"))
   zipfile.ZipFile("nyc311_lab_upload.zip").extractall()
   print(open("nyc311_lab/VERSION.txt").read())
   ```
4. **Open `nyc311_lab/lab.ipynb`.** In Settings (cell 3) set
   `DATA = "/gpfs/project/esai_2026/data/311/311_batch_*.json"` and `WORK = "/gpfs/home/ml3338/esai_311_work"`,
   and leave `THREADS` and `MEMORY_LIMIT` at `None` (4 threads, DuckDB memory limit 22GB). Reuse that WORK: it
   already holds the 28.24M-row Parquet copy, so ingest only checks for new files.
5. **Run > Run All Cells.** Check the `00_env` block: Python 3.9.24, 4 CPUs, 2,824 files (33.3 GB), no WARNING
   lines. If Settings says a package is not installed or `00_env` shows one as MISSING, cell 2 has just installed
   it: **Kernel > Restart**, then Run All again.
6. **Download:** in the file browser, right-click `nyc311_lab/outputs_bundle.zip` > Download, and the same for
   `nyc311_lab/outputs/report.html`. Unzip the bundle into `cluster_outputs/runN/` here.

Without Jupyter (inside the same SLURM job, not on a login node), steps 4-5 are, from `nyc311_lab/`:
`python3 run_all.py --data "/gpfs/project/esai_2026/data/311/311_batch_*.json" --work /gpfs/home/ml3338/esai_311_work`.

**Time and memory.** Ingest took 171 s for the 2,824 files on the cluster (4 CPUs). Discovery on a 27.6M-row
synthetic copy took 75 s on a laptop with 4 threads and a peak of 1.7 GB, so plan for 3-5 minutes on the cluster.
Both scale linearly: 20M+ rows show no superlinear step (`tools/make_synthetic_batches.py` builds the test data).
Ingest peaks at about 1.1 GB (main process plus 4 workers of ~150 MB each), and that doesn't grow with the number
of files, because it works 100 files at a time. Discovery stays under the DuckDB memory limit. Either way, the
32 GB job has plenty of room.

**WORK size.** The raw Parquet copy is 4-12% of the JSON (1.5-4 GB for 33 GB). Cleaning adds a copy of similar
size, and DuckDB spills to `WORK/duckdb_tmp` when a query exceeds the memory limit. Files that need a clean copy
(cut off, invalid UTF-8) add their own size under `WORK/recovered`. Allow about 20 GB; the `00_env` block warns
when WORK has less than half the JSON size free. Keep WORK on GPFS home, which has plenty of space, not in `/tmp`
(cleared between jobs) or inside the code folder.

**Threads and memory.** Keep the defaults. To change them, set `THREADS` or `MEMORY_LIMIT` (`--threads`,
`--memory-limit`). Never set more threads than the job has CPUs. If DuckDB stops with "Out of Memory", set
`MEMORY_LIMIT = "16GB"`; DuckDB then spills more to WORK. If the kernel dies during ingest, set `THREADS = 2`.

**If it dies halfway, rerun the same cell (or command) with the same WORK.** Ingest records a batch of 100 files in
`WORK/manifest.parquet` only after their Parquet part is completely written (under a temporary name, then
renamed), so the manifest never lists a file whose rows aren't in Parquet. On a rerun it deletes any part that the
manifest doesn't list, then converts only the files it doesn't have yet. A file that couldn't be read (for example,
permission denied) is retried on the next run. Discovery keeps no state, so it simply runs again. A WORK folder
written by the first version of ingest (such as `/gpfs/home/ml3338/esai_311_work`) is upgraded in place on the next
ingest, in seconds and without rescanning; discovery reads it either way. Only if a data file changes size does
ingest stop with "delete WORK and rerun".

If something fails, paste the error and the `00_env` block.

## Run locally (the 20 sample files)

From this folder, with [uv](https://docs.astral.sh/uv/):

```
uv run --no-project --with-requirements requirements.txt python run_all.py   # defaults to data/sample/
uv run --no-project --with-requirements requirements.txt pytest              # the full test suite, on the sample
```

## Folder layout

```
brief/            the assignment and the NY State Comptroller's NYC311 report (benchmark numbers)
data/sample/      20 batch files (Jan 1 - Feb 3, 2010); local dev and tests only, not in git
data/external/    weather, community district boundaries, district population/income (being fetched)
nyc311/           the package: config, report, env, ingest, discover, rules, bundle (+ nytplot for charts)
run_all.py        the single command
lab.ipynb         the same stages, cell by cell, for the cluster
tests/            pytest; expected numbers come from the sample
work/             large intermediates (manifest, Parquet, DuckDB spill); not in git
outputs/          small results: reports/*.txt, agg/*.csv, later figs/ and the dashboard; not in git
cluster_outputs/  unzipped bundles downloaded from each cluster run
```

## Why DuckDB

The JSON is read once into Parquet (about 1-2 GB). Every later stage queries the Parquet with DuckDB, which runs
in parallel on one node and spills to disk instead of running out of memory. Dask was not used: its JSON reading
is slower, every partition carries pandas' memory cost for 41 text columns, and its scheduler pays off only
across several machines. pandas handles the small aggregates that come out of DuckDB.

## What the 20 sample files already show

These are test cases in `tests/`. Run #1 checks whether they hold across all 2,000 files.

- **One cut-off file.** `311_batch_0010.json` ends mid-record. 6,270 of its records are recoverable, giving
  196,270 rows in total, with no duplicate `unique_key`.
- **Rows outside their file's dates.** Batch 0001 starts with 42 DOHMH rows from 2015-10-01 and 1 DOB row from
  2013. Jan 17, 2010 has only 337 requests (other days have about 6,600), probably because of the cut-off file.
- **Dates that can't be right.** 367 close dates are 1900-01-01 and 356 due dates are 1900-01-02, both
  placeholders. Leaving those aside, 2,921 requests close before they were opened (2,677 of them DOT, more than a
  day early), 8,600 close at the exact second they opened, and 2,983 have a real close date but a status other
  than Closed. 43% of opening times are exactly midnight: HPD records dates only.
- **"Missing" written as text.** `borough` and `community_board` say `Unspecified` for 45% of rows, almost all
  HPD, yet zip code and coordinates are usually present. `open_data_channel_type` is `UNKNOWN` for 64%.
  Zip codes include `N/A`, `NA`, `0000` and 9-digit out-of-state codes.
- **Columns blank by design.** The taxi, bridge/highway, vehicle and landmark columns apply only to one or two
  complaint types. `intersection_*` applies only to intersection addresses.
- **Inconsistent labels.** There are 174 complaint types, with variants such as `PLUMBING`/`Plumbing` and
  `Derelict Vehicle`/`Derelict Vehicles`. Requests for information (`Literature Request`, `SCRIE`) sit alongside
  complaints. There are 18 agency codes but 110 agency names; DOF alone has 17 unit names.

## Cleaning decisions (settled 2026-10-06 from the 20 local files)

The 20 local files are treated as the whole dataset (the cluster queue was blocked). Every rule lives in
`nyc311/rules.py`. **Nothing is deleted:** a rule either turns a value into NULL or adds a flag, and each question
states which flagged rows it leaves out. Counts are for the 196,270 local rows.

| # | Issue | Rows | Decision |
|---|---|---|---|
| 1 | Rows created >30 days from their file's median date | 43 local; 422,505 on the cluster | Flag `out_of_file_order`, descriptive only (not excluded). On the full data most are older records (mainly DOB, 2010-2019) backfilled into batches 1930-2088, so they are real requests |
| 2 | Days partly in files we don't have: after the last date-ordered file's median, the late days of a cut-off file, and the days between the files around a missing batch number (cluster: batch 0010 and 18 batches in 2787-2841) | 24 days on the cluster (Jan 15-18 2010; 23 days Sept-Nov 2021) | Flag `possibly_incomplete_day`; left out of daily trends, shown as caveats. Only date-ordered files (>=50% of rows within 30 days of the median) are used |
| 3 | `closed_date` = 1900-01-01 (DOHMH) | 367 | Set to NULL, flag `closed_placeholder` |
| 4 | Closed before opened (2,677 DOT closed more than a day early) | 2,921 | Flag `closed_before_created`; no duration |
| 5 | Close date present (not a placeholder) but status is not Closed | 2,983 | Flag `status_conflict`; durations only for status = Closed |
| 5b | Closed at the exact second it was opened (DOT, DSNY, DOB mostly) | 8,600 | Flag `closed_at_creation`; Q4 reports medians with and without them |
| 6 | Test records (`Closed - Testing`, `Draft`) | 6 | Flag `is_test`; left out everywhere |
| 7 | Any status other than Closed, Cancelled, test, or `Unspecified` (Open, Pending, Assigned, Started, In Progress, Email Sent, Unassigned, ...) | 6,435 local | Flag `unresolved` (Q9) |
| 8 | Opening time exactly midnight (HPD records dates only) | 84,883 | Flag `date_only_created`; durations for these are precise to the day only |
| 9 | Text that means missing: `Unspecified` (borough, park fields), `UNKNOWN` (channel), `N/A` (facility, location type, descriptor, city) | see `02c` | Set to NULL |
| 10 | ZIP codes: ZIP+4, `N/A`, `0000`, out-of-state | 157 | Keep NYC 5-digit only (`zip`); original in `zip_raw` |
| 11 | Borough `Unspecified` (almost all HPD) | 89,164 | Recover in order: reported borough, community board text, coordinates inside a district polygon, the ZIP's usual borough. `borough_source` says which |
| 12 | Community district missing (`0 Unspecified`, `Unspecified QUEENS`) | see `03_clean` | Parse `NN BOROUGH` (districts 1-12/18/14/3); higher numbers are parks/airports (`special_area_cd`); otherwise use the polygon containing the coordinates (`cd_source`) |
| 13 | 174 complaint-type spellings | 174 | Map to `canonical_type`, `category`, `origin` via `nyc311/complaint_type_map.csv` (Q3) |

## Clean table (`work/clean.parquet`), for every question module

One row per raw row. Read with `from nyc311.clean import clean_glob` then `con.sql(f"select ... from {clean_glob(cfg)}")`.

- **Keys and times:** `unique_key`, `_file`, `created`, `closed`, `due`, `resolution_updated` (timestamps; placeholders are NULL), `duration_h` (hours from opening to closing; only for status Closed, closed after opened).
- **What:** `agency`, `agency_name`, `complaint_type` (raw), `descriptor`, `canonical_type`, `category`, `origin` (`city`/`private`/`natural`/`other`), `is_info_request`, `type_needs_review`, `status`, `channel`, `resolution_description`.
- **Where:** `borough` + `borough_source`, `boro_cd` (standard district, e.g. 101 = Manhattan 1) + `cd_source`, `special_area_cd` (parks/airports), `zip`, `zip_raw`, `latitude`, `longitude`, plus the raw address, location and highway/taxi/park text columns.
- **Flags (boolean):** `dup_key`, `out_of_file_order`, `possibly_incomplete_day`, `is_test`, `unresolved`, `status_conflict`, `closed_placeholder`, `closed_before_created`, `closed_at_creation`, `closed_after_1yr`, `date_only_created`, `date_only_closed`.

Default analysis set: `not is_test and not dup_key`. Daily or weekly trends also drop `possibly_incomplete_day`.

## Question modules (contract for every `nyc311/qNN_*.py`)

- **Names** are fixed in `nyc311/config.py` (`STAGE_MODULES`): `q01_missing`, `q02_timing`, `q03_complaint_types`,
  `q04_response_times`, `q05_top_types`, `q06_weather`, `q07_geography`, `q08_dashboard`, `q09_equity`,
  `q10_features`, `q11_summary`. Each exposes `run(cfg)`; `run_all.py` and `lab.ipynb` call it.
- **Inputs:** the clean table via `from nyc311.clean import clean_glob` (`con = cfg.connect()`), and `data/external/`
  for weather, district polygons and demographics. Raw text columns, if needed, via
  `read_parquet('<raw glob>', union_by_name=true)`.
- **Scale:** the cluster data is 28.24M rows (Jan 1 2010 - Nov 20 2021), 4 threads, 22 GB for DuckDB. Aggregate in
  DuckDB and pull only small results into pandas. Regressions run on a fixed-seed sample
  (`using sample reservoir(1000000 rows) repeatable (42)`).
- **Outputs:** one report block named `qNN_<slug>` (via `nyc311.report.Report`), charts in `outputs/figs/qNN_*.png`
  (nyt-charts style: `from nyc311 import nytplot as nyt`), small tables in `outputs/agg/qNN_*.csv`, bullet findings
  in `outputs/answers/qNN.md`. Every number in text is computed at run time.
- **Facts about the full data to design for:** 12 years with a 2020 surge (June-August above 300k/month); late
  2021 incomplete (24 flagged days); HPD opening times are date-only before about 2015; about 1.04M requests close
  at the second they open; 478 raw complaint types (Q3 maps them); a backfilled block of older DOB records.
  Full-run aggregates for reference while building (never read at run time): `cluster_outputs/full/outputs/agg/`.
- **Tests:** `tests/test_qNN*.py` on the local sample (fixtures in `tests/conftest.py`). Code must run on Python 3.9
  with the cluster stack: duckdb 1.4.5, pandas 2.3.3, numpy 2.0.2, pyarrow 21.0.0, matplotlib 3.9.4,
  statsmodels 0.14.6, folium 0.20.0.
