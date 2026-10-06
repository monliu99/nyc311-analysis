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
- **Dates that can't be right.** 3,288 requests close before they were opened. 367 close dates are 1900-01-01
  and 356 due dates are 1900-01-02, both placeholders. 3,350 requests have a close date but a status other
  than Closed. 43% of opening times are exactly midnight: HPD records dates only.
- **"Missing" written as text.** `borough` and `community_board` say `Unspecified` for 45% of rows, almost all
  HPD, yet zip code and coordinates are usually present. `open_data_channel_type` is `UNKNOWN` for 64%.
  Zip codes include `N/A`, `NA`, `0000` and 9-digit out-of-state codes.
- **Columns blank by design.** The taxi, bridge/highway, vehicle and landmark columns apply only to one or two
  complaint types. `intersection_*` applies only to intersection addresses.
- **Inconsistent labels.** There are 174 complaint types, with variants such as `PLUMBING`/`Plumbing` and
  `Derelict Vehicle`/`Derelict Vehicles`. Requests for information (`Literature Request`, `SCRIE`) sit alongside
  complaints. There are 18 agency codes but 110 agency names; DOF alone has 17 unit names.

## Cleaning decisions

Not settled yet. `nyc311/rules.py` holds the provisional list of placeholder values from the sample. After run #1,
each decision is recorded here with its row counts, the same way as in session 9.
