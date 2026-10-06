"""Fetch the external datasets that the NYC 311 analysis joins to.

Writes three files under data/external/ (see data/external/SOURCES.md):

  weather_central_park_daily.csv   NOAA GHCN-Daily, station USW00094728
  community_districts.geojson      NYC DCP community district polygons (WGS84)
  cd_demographics.csv              population 2010/2020 + ACS median household income

Usage (from the repo root):
    uv run --no-project --with-requirements requirements.txt python nyc311/fetch_external.py [--force] [--only NAME ...]

Idempotent: an output that already exists is skipped unless --force is given.
After running, prints each file's SHA-256 and row/feature count, and whether the
hash matches the one recorded in SOURCES.md (EXPECTED_SHA256 below).

Stdlib only. Downloaded files are parsed as data (csv/json/zip members read in
memory); nothing from them is executed or extracted to disk.
"""
from __future__ import annotations

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "external"

WEATHER_FILE = "weather_central_park_daily.csv"
CD_FILE = "community_districts.geojson"
DEMO_FILE = "cd_demographics.csv"

# SHA-256 of the files as written on 2026-10-06 (also recorded in SOURCES.md).
# A mismatch on re-run means the upstream source was revised.
EXPECTED_SHA256 = {
    WEATHER_FILE: "e0fdc101647472156945247d26e962900dc8c94106d1f2071f55335f426e9444",
    CD_FILE: "ca0ae3e46300dae9cc4b6998af03eae13663c712c288ed206f14f01a4ed9b9bf",
    DEMO_FILE: "b22a79b8d0e3b6f7c079600afe61ddc6c3cecf2c47d2e07c12383cb938f291b9",
}

USER_AGENT = "nyc311-lab-fetch/1.0 (Yale SOM MGT634 coursework)"

BOROUGHS = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
BORO_CODE = {name: code for code, name in BOROUGHS.items()}
# Standard community districts per borough (59 total).
CD_COUNTS = {1: 12, 2: 12, 3: 18, 4: 14, 5: 3}
STANDARD_CDS = [b * 100 + n for b, k in CD_COUNTS.items() for n in range(1, k + 1)]

# ---------------------------------------------------------------- weather
WEATHER_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/"
    "access/USW00094728.csv"
)
WEATHER_START = dt.date(2010, 1, 1)
# Last date present in the NOAA file when retrieved on 2026-10-06. Pinned so a
# re-run reproduces the same file; raise it to extend the series.
WEATHER_END = dt.date(2026, 9, 29)

# ---------------------------------------------------------------- CD polygons
# NYC Open Data "Community Districts" (DCP), version 26b at retrieval.
CD_URL = "https://data.cityofnewyork.us/resource/5crt-au7u.geojson?$limit=1000&$order=boro_cd"
COORD_DECIMALS = 5  # ~1.1 m at NYC latitude

# ---------------------------------------------------------------- demographics
POP_2010_URL = "https://data.cityofnewyork.us/resource/xi7c-iiu2.csv?$limit=1000"

POP_2020_PDF_URL = (
    "https://www.nyc.gov/assets/planning/download/pdf/planning-level/nyc-population/"
    "census2020/nyc-community-districs-profiles-2020-2010.pdf"
)
POP_2020_PDF_SHA256 = "2316f5e746e7da35c352b7ddc7dcbb03d7f6f9dde624d475f44f5a83aa2fda51"
# "Total population" row (2010, 2020) for each CD, transcribed from the PDF
# above (encrypted PDF with CID fonts, so it is not parsed here; the download is
# instead checked against POP_2020_PDF_SHA256). Extracted with `pdftotext
# -layout` on 2026-10-06. Checks: the 2010 column equals NYC Open Data
# xi7c-iiu2 for all 59 CDs (asserted below), and 2020-2010 equals the PDF's own
# change column for all 59 CDs.
POP_PDF_2010_2020 = {
    101: (60978, 78390), 102: (90016, 92445), 103: (163277, 163141),
    104: (103245, 131351), 105: (51673, 63600), 106: (142745, 155614),
    107: (209084, 222129), 108: (219920, 231983), 109: (110193, 110458),
    110: (115723, 130440), 111: (120511, 125771), 112: (190020, 180206),
    201: (91497, 100929), 202: (52246, 54454), 203: (79762, 93755),
    204: (146441, 150436), 205: (128200, 132584), 206: (83268, 89216),
    207: (139286, 146813), 208: (101731, 106924), 209: (172298, 188249),
    210: (120392, 130763), 211: (113232, 119666), 212: (152344, 163489),
    301: (173083, 204125), 302: (99617, 130021), 303: (152985, 174960),
    304: (112634, 120747), 305: (182896, 200293), 306: (104709, 113933),
    307: (126230, 133230), 308: (96317, 108259), 309: (98429, 102000),
    310: (124491, 136071), 311: (181981, 198870), 312: (191382, 205377),
    313: (104278, 108905), 314: (160664, 164568), 315: (159650, 169632),
    316: (86468, 98506), 317: (155252, 162446), 318: (193543, 204095),
    401: (191105, 196803), 402: (113200, 137981), 403: (171576, 179134),
    404: (172598, 181025), 405: (169190, 178178), 406: (113257, 120955),
    407: (247354, 269587), 408: (151107, 161402), 409: (143317, 152839),
    410: (122396, 129411), 411: (116431, 122211), 412: (225919, 256278),
    413: (188593, 199218), 414: (114978, 124173), 501: (175756, 189663),
    502: (132003, 140795), 503: (160209, 164723),
}

# ACS 2010-2014 5-year Summary File (Census Bureau FTP; the Census API now
# requires a key). Table B19013 = median household income.
ACS_BASE = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2014/data/"
    "5_year_seq_by_state/NewYork/All_Geographies_Not_Tracts_Block_Groups/"
)
ACS_GEO_URL = ACS_BASE + "g20145ny.csv"
ACS_SEQ_URL = ACS_BASE + "20145ny0058000.zip"
ACS_LOOKUP_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2014/documentation/"
    "user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt"
)
ACS_TABLE, ACS_SEQ, ACS_START_POS = "B19013", "0058", 177  # 1-based field position
INCOME_VINTAGE = "ACS 2010-2014 5-year; 2014 inflation-adjusted dollars; 2010 PUMAs"


# ====================================================================== helpers
def download(url: str, label: str, retries: int = 3, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            print(f"  downloaded {label}: {len(data):,} bytes, sha256 {sha256_bytes(data)}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry any network error
            if attempt == retries:
                raise RuntimeError(f"download failed for {url}: {exc}") from exc
            print(f"  retry {attempt} for {label}: {exc}")
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def fmt(value: float | None, decimals: int) -> str:
    """Deterministic number formatting; None -> empty cell (read as NaN)."""
    if value is None:
        return ""
    if decimals == 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


# ====================================================================== weather
def build_weather(path: Path) -> None:
    raw = download(WEATHER_URL, "GHCN-Daily USW00094728")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))

    # GHCN-Daily native units: PRCP tenths of mm; SNOW, SNWD mm; TMAX, TMIN tenths of degC.
    elements = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
    by_date: dict[str, dict[str, float | None]] = {}
    for row in reader:
        if row["STATION"] != "USW00094728":
            raise ValueError(f"unexpected station {row['STATION']}")
        d = row["DATE"]
        if not (WEATHER_START.isoformat() <= d <= WEATHER_END.isoformat()):
            continue
        vals: dict[str, float | None] = {}
        for e in elements:
            v = (row.get(e) or "").strip()
            attrs = (row.get(f"{e}_ATTRIBUTES") or "").split(",")
            qflag = attrs[1].strip() if len(attrs) > 1 else ""
            # Values that failed a GHCN quality check (non-blank QFLAG) -> missing.
            vals[e] = float(v) if v and not qflag else None
        by_date[d] = vals

    def conv_temp(x):  # tenths degC -> degF, rounded to whole degF (NWS reporting unit)
        return None if x is None else x / 10 * 9 / 5 + 32

    rows = []
    day = WEATHER_START
    while day <= WEATHER_END:
        v = by_date.get(day.isoformat(), dict.fromkeys(elements))
        rows.append([
            day.isoformat(),
            fmt(conv_temp(v["TMAX"]), 0),
            fmt(conv_temp(v["TMIN"]), 0),
            fmt(None if v["PRCP"] is None else v["PRCP"] / 254.0, 2),  # tenths mm -> in
            fmt(None if v["SNOW"] is None else v["SNOW"] / 25.4, 1),   # mm -> in
            fmt(None if v["SNWD"] is None else v["SNWD"] / 25.4, 1),   # mm -> in
        ])
        day += dt.timedelta(days=1)
    if WEATHER_END.isoformat() not in by_date:
        raise ValueError(f"source has no data for WEATHER_END={WEATHER_END}")
    write_csv(path, ["date", "tmax_f", "tmin_f", "prcp_in", "snow_in", "snwd_in"], rows)


# ====================================================================== CD polygons
def _clean_ring(ring: list) -> list | None:
    out: list[list[float]] = []
    for x, y, *_ in ring:
        pt = [round(x, COORD_DECIMALS), round(y, COORD_DECIMALS)]
        if not out or pt != out[-1]:
            out.append(pt)
    if out[0] != out[-1]:
        out.append(out[0])
    return out if len(out) >= 4 else None


def _clean_geometry(geom: dict) -> dict:
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        raise ValueError(f"unexpected geometry type {geom['type']}")
    new_polys = []
    for poly in polys:
        rings = [_clean_ring(r) for r in poly]
        if rings[0] is None:  # exterior collapsed by rounding: drop the part
            continue
        new_polys.append([r for r in rings if r is not None])
    return {"type": "MultiPolygon", "coordinates": new_polys}


def build_cd_geojson(path: Path) -> None:
    raw = download(CD_URL, "NYC Open Data 5crt-au7u (Community Districts)")
    src = json.loads(raw.decode("utf-8"))
    features = []
    for f in src["features"]:
        boro_cd = int(float(f["properties"]["boro_cd"]))
        boro, num = divmod(boro_cd, 100)
        geom = _clean_geometry(f["geometry"])
        for poly in geom["coordinates"]:
            for ring in poly:
                for lon, lat in ring:
                    if not (-74.3 < lon < -73.6 and 40.4 < lat < 41.0):
                        raise ValueError(f"coordinate outside NYC lon/lat box: {lon}, {lat}")
        features.append({
            "type": "Feature",
            "properties": {
                "boro_cd": boro_cd,
                "borough": BOROUGHS[boro],
                "cd_number": num,
                "is_jia": num > 18,
            },
            "geometry": geom,
        })
    features.sort(key=lambda f: f["properties"]["boro_cd"])
    ids = [f["properties"]["boro_cd"] for f in features]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate boro_cd in source")
    missing = set(STANDARD_CDS) - set(ids)
    if missing:
        raise ValueError(f"source is missing standard CDs: {sorted(missing)}")
    out = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


# ====================================================================== demographics
def _pop_2010() -> dict[int, tuple[int, str]]:
    raw = download(POP_2010_URL, "NYC Open Data xi7c-iiu2 (population by CD)")
    out = {}
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        boro_cd = BORO_CODE[r["borough"].strip()] * 100 + int(r["cd_number"])
        out[boro_cd] = (int(r["_2010_population"]), r["cd_name"].strip())
    return out


def _check_pop_2020_pdf(pop2010: dict[int, tuple[int, str]]) -> None:
    pdf = download(POP_2020_PDF_URL, "DCP 2020/2010 CD census profiles PDF")
    got = sha256_bytes(pdf)
    if got != POP_2020_PDF_SHA256:
        raise RuntimeError(
            "DCP 2020 census PDF changed since transcription "
            f"(sha256 {got} != {POP_2020_PDF_SHA256}); re-verify POP_PDF_2010_2020."
        )
    bad = [cd for cd, (p10, _) in POP_PDF_2010_2020.items() if pop2010[cd][0] != p10]
    if bad or set(POP_PDF_2010_2020) != set(STANDARD_CDS):
        raise RuntimeError(f"PDF 2010 column disagrees with xi7c-iiu2 for CDs {bad}")


def _acs_income_by_puma() -> dict[str, tuple[str, int | None, int | None]]:
    """Return {puma5: (puma_name, median_hh_income, moe)} for NYC PUMAs."""
    lookup = download(ACS_LOOKUP_URL, "ACS 2014 5-yr sequence/table lookup").decode("latin-1")
    hits = [r for r in csv.reader(io.StringIO(lookup))
            if len(r) > 4 and r[1] == ACS_TABLE and r[4].strip()]
    if not hits or hits[0][2] != ACS_SEQ or int(hits[0][4]) != ACS_START_POS:
        raise RuntimeError(f"{ACS_TABLE} not at sequence {ACS_SEQ} position {ACS_START_POS}")

    geo_raw = download(ACS_GEO_URL, "ACS 2014 5-yr NY geography file").decode("latin-1")
    pumas = {}  # logrecno -> (puma5, name)
    for r in csv.reader(io.StringIO(geo_raw)):
        # r[2]=SUMLEVEL, r[3]=COMPONENT, r[4]=LOGRECNO, r[48]=GEOID, r[49]=NAME
        if r[2] == "795" and r[3] == "00" and r[49].startswith("NYC-"):
            pumas[r[4]] = (r[48][-5:], r[49])

    seq_zip = zipfile.ZipFile(io.BytesIO(download(ACS_SEQ_URL, "ACS 2014 5-yr NY sequence 0058")))
    col = ACS_START_POS - 1

    def read(member: str) -> dict[str, str]:
        text = seq_zip.read(member).decode("latin-1")
        return {r[5]: r[col] for r in csv.reader(io.StringIO(text)) if r[5] in pumas}

    est = read(f"e20145ny{ACS_SEQ}000.txt")
    moe = read(f"m20145ny{ACS_SEQ}000.txt")

    def num(s: str) -> int | None:
        s = s.strip()
        return int(s) if s and s != "." else None

    return {p: (name, num(est[lr]), num(moe[lr])) for lr, (p, name) in pumas.items()}


def _puma_to_cds(name: str) -> list[int]:
    """Parse the official 2010 PUMA name, e.g.
    'NYC-Bronx Community District 1 & 2--Hunts Point, Longwood & Melrose PUMA; New York'."""
    m = re.match(r"NYC-(Bronx|Brooklyn|Manhattan|Queens|Staten Island) Community District "
                 r"(\d+)(?: & (\d+))?--", name)
    if not m:
        raise ValueError(f"cannot parse PUMA name: {name}")
    boro = BORO_CODE[m.group(1)]
    return [boro * 100 + int(g) for g in m.groups()[1:] if g]


def build_demographics(path: Path) -> None:
    pop2010 = _pop_2010()
    if set(pop2010) != set(STANDARD_CDS):
        raise RuntimeError("xi7c-iiu2 does not contain exactly the 59 standard CDs")
    _check_pop_2020_pdf(pop2010)
    income = _acs_income_by_puma()

    cd_to_puma: dict[int, str] = {}
    puma_cds: dict[str, list[int]] = {}
    for puma, (name, _, _) in income.items():
        cds = _puma_to_cds(name)
        puma_cds[puma] = cds
        for cd in cds:
            if cd in cd_to_puma:
                raise RuntimeError(f"CD {cd} appears in two PUMAs")
            cd_to_puma[cd] = puma
    if set(cd_to_puma) != set(STANDARD_CDS) or len(income) != 55:
        raise RuntimeError("PUMA names do not map one-to-one onto the 59 CDs / 55 PUMAs")

    rows = []
    for cd in STANDARD_CDS:
        boro, num = divmod(cd, 100)
        puma = cd_to_puma[cd]
        name, inc, moe = income[puma]
        if inc is None:
            raise RuntimeError(f"no income estimate for PUMA {puma}")
        partners = [c for c in puma_cds[puma] if c != cd]
        note = (f"Income is for PUMA {puma}, which combines CDs "
                f"{' & '.join(str(c) for c in puma_cds[puma])}; same value on both rows."
                if partners else "")
        rows.append([
            cd, BOROUGHS[boro], num, pop2010[cd][1],
            pop2010[cd][0], POP_PDF_2010_2020[cd][1],
            puma, name.removesuffix("; New York").removesuffix(", New York"),
            ";".join(str(c) for c in partners),
            inc, moe, INCOME_VINTAGE, note,
        ])
    header = ["boro_cd", "borough", "cd_number", "cd_name",
              "population_2010", "population_2020",
              "puma_2010", "puma_name", "puma_shared_with",
              "median_hh_income", "median_hh_income_moe", "income_vintage", "notes"]
    write_csv(path, header, rows)


# ====================================================================== main
def count(path: Path) -> str:
    if path.suffix == ".geojson":
        return f"{len(json.loads(path.read_text())['features'])} features"
    with path.open() as f:
        return f"{sum(1 for _ in f) - 1} rows"


BUILDERS = {"weather": (WEATHER_FILE, build_weather),
            "cd": (CD_FILE, build_cd_geojson),
            "demographics": (DEMO_FILE, build_demographics)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    ap.add_argument("--only", nargs="+", choices=sorted(BUILDERS), help="build only these")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.only or list(BUILDERS):
        fname, builder = BUILDERS[key]
        path = OUT_DIR / fname
        if path.exists() and not args.force:
            print(f"[skip] {fname} exists (use --force to rebuild)")
        else:
            print(f"[build] {fname}")
            tmp = path.with_suffix(path.suffix + ".tmp")
            builder(tmp)
            tmp.replace(path)

    print("\nOutputs:")
    for fname, _ in BUILDERS.values():
        path = OUT_DIR / fname
        if not path.exists():
            print(f"  {fname}: (missing)")
            continue
        digest = sha256_file(path)
        expected = EXPECTED_SHA256.get(fname)
        status = ("matches SOURCES.md" if digest == expected
                  else "DIFFERS from SOURCES.md" if expected else "no recorded hash")
        print(f"  {fname}: {count(path)}, sha256 {digest} ({status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
