# External data sources

All three files are produced by `nyc311/fetch_external.py` (stdlib only), run from the repo root:

```
uv run --no-project --with-requirements requirements.txt python nyc311/fetch_external.py          # skips existing files
uv run --no-project --with-requirements requirements.txt python nyc311/fetch_external.py --force  # rebuild all
```

Retrieved **2026-10-06**. Two `--force` runs on that date gave byte-identical outputs. The script prints each output's SHA-256 and reports whether it matches the hash below. A mismatch means the upstream source was revised after 2026-10-06. Raw-download hashes are listed so you can tell which input changed.

All CD-level files use the key **`boro_cd`** (int): borough code × 100 + district number (Manhattan 1, Bronx 2, Brooklyn 3, Queens 4, Staten Island 5). In this scheme a 311 `community_board` of `"01 MANHATTAN"` becomes 101 and `"12 QUEENS"` becomes 412.

---

## 1. `weather_central_park_daily.csv`

| | |
|---|---|
| Rows | 6,116 (one per day, 2010-01-01 to 2026-09-29, continuous) |
| SHA-256 | `e0fdc101647472156945247d26e962900dc8c94106d1f2071f55335f426e9444` |
| Source org | NOAA National Centers for Environmental Information (NCEI), GHCN-Daily |
| Station | USW00094728, "NY CITY CENTRAL PARK, NY US" (40.77898, -73.96925) |
| URL used | https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv |
| Raw download | 17,831,810 bytes, sha256 `3945b3d0183e26dc1f3ca41c542f2bd9b40838736c9f9572c490fbc8d17a5ccc` (full station history 1869 to 2026-09-29) |
| Unit reference | https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt |

**Columns and units:** `date` (YYYY-MM-DD), `tmax_f` and `tmin_f` (°F, whole degrees), `prcp_in` (inches, 2 decimals), `snow_in` (snowfall, inches, 1 decimal), `snwd_in` (snow depth, inches, 1 decimal).

**Conversion:** the source uses GHCN native units: PRCP is in tenths of mm, SNOW and SNWD are in mm, and TMAX and TMIN are in tenths of °C. Conversions:
- °F = (v/10)·9/5 + 32, rounded to whole °F.
- in = v/254 for PRCP and v/25.4 for SNOW and SNWD.

I checked the converted values against NCEI's own `units=standard` output (Access Data Service) for 1,576 days in 2010-2014. Every value matched.

**Caveats**
- **End date is pinned** at 2026-09-29 (`WEATHER_END` in the script). That was the last day in the file on the retrieval date. Raise it to extend the series.
- **Missing values** are empty cells, which pandas reads as NaN. Nothing is filled. Only 4 cells are missing: `snwd_in` on 2025-09-17, 2026-05-31, 2026-06-01 and 2026-06-09. Values with a non-blank GHCN quality flag would also be set to missing, but there were none in 2010-2026.
- **Trace amounts** (GHCN measurement flag `T`) are stored as 0.00.
- **Source flags change over time.** From May 2025 onward, values come from the NWS CF6 preliminary daily climate summaries (source flag `1`). Earlier values come from ISD/ASOS summaries (flags `W` and `X`) and, for 2010, Cooperative Summary of the Day (flag `0`). NOAA may revise recent values, which would change the SHA-256.
- **One station only.** Central Park represents city-wide weather. Rain and snow totals vary across the five boroughs.

---

## 2. `community_districts.geojson`

| | |
|---|---|
| Features | 71: the 59 standard community districts plus 12 joint interest areas (JIAs) |
| SHA-256 | `ca0ae3e46300dae9cc4b6998af03eae13663c712c288ed206f14f01a4ed9b9bf` |
| File size | 1.88 MB |
| Source org | NYC Department of City Planning (DCP), published on NYC Open Data |
| Dataset | "Community Districts" (`5crt-au7u`), clipped to shoreline. Metadata says "Current version: 26b" (data updated 2026-05-26) |
| URL used | https://data.cityofnewyork.us/resource/5crt-au7u.geojson?$limit=1000&$order=boro_cd |
| Raw download | 3,793,109 bytes, sha256 `899af69ebb3e35b5f5974fbf4fdea691e07354aaffd8a85191fd126f2db71e1b` |
| Landing page | https://data.cityofnewyork.us/City-Government/Community-Districts/5crt-au7u |

**Properties:**
- `boro_cd` (int)
- `borough` (str)
- `cd_number` (int)
- `is_jia` (bool, true when district number > 18)

**Geometry:** every feature is a MultiPolygon in WGS84 lon/lat (EPSG:4326). Coordinates are rounded to 5 decimals (about 1 m). Consecutive duplicate vertices left by rounding were removed, which took the vertex count from 96,822 to 89,855. No polygon parts or rings were lost. Features are sorted by `boro_cd`.

**JIAs:** 164 Central Park; 226, 227 and 228 (Bronx parks: Van Cortlandt, Bronx Park, Pelham Bay); 355 and 356 (Prospect Park and the Brooklyn Gateway area); 480 through 484 (Queens: LaGuardia, Flushing Meadows, Forest Park, JFK, Gateway/Rockaway); 595 (Staten Island Gateway). The 311 `community_board` field uses these same JIA numbers, for example "64 MANHATTAN" (164), "83 QUEENS" (483) and "95 STATEN ISLAND" (595).

**Caveats**
- **The source changes quarterly.** NYC Open Data serves only the current release, so a later re-run may get version 26c or newer and produce a different hash. DCP's own ArcGIS service (`services5.arcgis.com/GfwWNkhOj9bNBqoJ/.../NYC_Community_Districts/FeatureServer/0`) had already been edited on 2026-08-21 and differs slightly in vertex detail. Older vintages are on DCP's BYTES of the BIG APPLE archive, but only as State Plane shapefiles, which this script cannot reproject without extra libraries.
- **Boundaries are current, not historical.** The analysis covers 2010 onward, and CD boundaries have changed little since then. The 311 `community_board` value was assigned by the city geocoder when each request was made.

---

## 3. `cd_demographics.csv`

| | |
|---|---|
| Rows | 59, one per standard CD. JIAs are excluded because their populations are tiny. |
| SHA-256 | `b22a79b8d0e3b6f7c079600afe61ddc6c3cecf2c47d2e07c12383cb938f291b9` |

**Columns:**
- `boro_cd`, `borough`, `cd_number`, `cd_name`
- `population_2010`, `population_2020`
- `puma_2010`: 5-digit 2010 PUMA code. Read it with `dtype=str` to keep the leading zero.
- `puma_name`
- `puma_shared_with`: the other `boro_cd` in a combined PUMA, otherwise empty
- `median_hh_income`, `median_hh_income_moe`: USD, 90% margin of error
- `income_vintage`, `notes`

### population_2010
- **Source:** NYC DCP Population Division, NYC Open Data "New York City Population By Community Districts" (`xi7c-iiu2`). Data is from Census 2010 SF1.
- **URL used:** https://data.cityofnewyork.us/resource/xi7c-iiu2.csv?$limit=1000 (raw sha256 `f17b2f1e1289cf61f9a2a8193279af827733184cf5d2c2ea08278351b8b91920`)
- **Total:** 8,171,680. The official NYC 2010 count is 8,175,133. The 3,453 difference is people living in JIAs.

### population_2020
- **Source:** NYC DCP Population Division, "Decennial Census Data for New York City's 59 Community Districts, 2020 and 2010" (September 2021). Data is from the 2010 and 2020 Census Redistricting Data (P.L. 94-171).
- **URL used:** https://www.nyc.gov/assets/planning/download/pdf/planning-level/nyc-population/census2020/nyc-community-districs-profiles-2020-2010.pdf (the "districs" typo is in the real URL). Raw sha256 `2316f5e746e7da35c352b7ddc7dcbb03d7f6f9dde624d475f44f5a83aa2fda51`.
- **Total:** 8,803,220. The official NYC 2020 count is 8,804,190; the difference is the JIA population.
- **How it was extracted:** the PDF is encrypted and uses CID fonts, so the stdlib script cannot parse it. I extracted the "Total population" row for each CD once with `pdftotext -layout` and embedded the values in the script (`POP_PDF_2010_2020`). On every run the script re-downloads the PDF and refuses to proceed if its SHA-256 differs.
- **Transcription checks:**
  - The PDF's 2010 column equals `xi7c-iiu2` for all 59 CDs. The script asserts this on every run.
  - For all 59 CDs, 2020 minus 2010 equals the PDF's own change column.
- **I found no machine-readable official 2020 CD table.** DCP's 2020 products are tabulated by CDTA, which only approximates CDs. A DCP redistricting story-map layer (`CDCCDIntersect_23_*`) has CD populations, but its 2010 figures do not match the official SF1 counts, so I did not use it.
- **Caveat:** 2020 counts are subject to the Census Bureau's differential-privacy disclosure avoidance, which adds noise at small geographies. CD totals are large enough that the noise is minor.

### median_hh_income
- **Source:** U.S. Census Bureau, **ACS 2010-2014 5-year** Summary File, table B19013 (median household income in the past 12 months, **2014 inflation-adjusted dollars**), by 2010 PUMA.
- **Why this vintage:** its 2010-2014 window matches the early-2010s 311 data. It also uses the 2010 PUMAs, which map cleanly onto CDs. ACS releases from 2022 onward use 2020 PUMAs with different codes.
- **URLs used:**
  - Geography file: https://www2.census.gov/programs-surveys/acs/summary_file/2014/data/5_year_seq_by_state/NewYork/All_Geographies_Not_Tracts_Block_Groups/g20145ny.csv (sha256 `19cb4de7c18d4d4a3476c7e9da61674c95f4040b8b032ba94864cb7e3bfc3940`)
  - Sequence 0058 (estimates and MOEs): https://www2.census.gov/programs-surveys/acs/summary_file/2014/data/5_year_seq_by_state/NewYork/All_Geographies_Not_Tracts_Block_Groups/20145ny0058000.zip (sha256 `59119bdd404044630c4006c784f0130c2e67660e2952fb30b2e196cd859f58f6`)
  - Table lookup confirming B19013 is in sequence 0058 at field position 177: https://www2.census.gov/programs-surveys/acs/summary_file/2014/documentation/user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt (sha256 `e56b9463facb38c73de8b565922e6e6ed5c33baf56da560f236f0d9e8c7cfb18`)
- **Why not the Census API:** `api.census.gov` now requires a key. A keyless request returned HTTP 302 to `missing_key.html` with header `X-DataWebAPI-KeyError: 1`. The Summary File on the Census Bureau's file server holds the same published estimates and needs no key.
- **CD↔PUMA crosswalk:** taken from the official 2010 PUMA names in the Census geography file, for example "NYC-Bronx Community District 1 & 2--Hunts Point, Longwood & Melrose PUMA". NYC DCP delineated these PUMAs from CDs. The script asserts that the 55 PUMAs map one-to-one onto the 59 CDs. I also checked the mapping against DCP's Community District Profiles table (`community_district_profiles_v202402` on planninglabs.carto.com, fields `puma` and `shared_puma`): all 59 CDs agree.
- **CDs that share a PUMA** (4 PUMAs cover 8 CDs, so 59 CDs map to 55 PUMAs). Both CDs in each pair carry the PUMA-wide median:

  | PUMA | CDs |
  |---|---|
  | 03810 | Manhattan 1 & 2 (101, 102) |
  | 03807 | Manhattan 4 & 5 (104, 105) |
  | 03710 | Bronx 1 & 2 (201, 202) |
  | 03705 | Bronx 3 & 6 (203, 206) |

- **Caveats**
  - PUMAs are built from census tracts and only approximate CD boundaries.
  - Medians of two combined CDs cannot be split apart. For CD-level equity comparisons, either treat each pair as one unit or note the shared value.
  - Dollar amounts are nominal 2014 dollars. Do not compare them directly with incomes from other ACS vintages.
