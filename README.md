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

## Cluster run #1 (discovery)

1. Upload this folder **without `data/sample/`** (that's 222 MB of local test data). Zipping it is easiest:
   from the parent folder, `zip -r nyc311_lab.zip nyc311_lab -x 'nyc311_lab/data/sample/*' 'nyc311_lab/.git/*' 'nyc311_lab/.superpowers/*'`.
2. Open `lab.ipynb` from inside the uploaded folder. In the Settings cell, set `DATA` to the path pattern for the
   2,000 files, and `WORK` to persistent storage with about 5 GB free. Keep `WORK` between runs: later runs then skip
   files already converted.
3. Run every cell. The first installs `requirements.txt`. Stage 1 should take a few minutes.
4. Download the zip that the last cell prints (`outputs_bundle.zip`, well under 50 MB). Unzip it into
   `cluster_outputs/run1/` here.

If something fails, paste the error and the `00_env` block.

## Run locally (the 20 sample files)

From this folder, with [uv](https://docs.astral.sh/uv/):

```
uv run --no-project --with-requirements requirements.txt python run_all.py   # defaults to data/sample/
uv run --no-project --with-requirements requirements.txt pytest              # 33 tests on the sample
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
