"""Q8: a web dashboard (folium / Leaflet) for zooming in on an area and reading the details of its complaints.

Layers (outputs/dashboard/q08_dashboard.html):
- The 59 community districts shaded by complaints per 1,000 residents a year (quintile classes). One shading at a
  time, chosen with radio buttons: all complaints, or one of the N_THEMES largest complaint categories. Hover shows
  the district and its rate; a click shows complaints, rate and rank, the top complaint types, median time to close
  and the complaint-mix-adjusted gap (Q4's `mix_adjusted`), share unresolved, and complaints per year.
- Parks and airports (joint interest areas) in gray: few residents, so no rate.
- Individual complaints from the WINDOW_DAYS most recent complete days, clustered: a fixed-seed sample of at most
  POINT_CAP (the requests with the smallest seeded hash of unique_key), each with a popup.

Complaint set: rules.COMPLAINTS_SQL (not is_test, not dup_key, not an information request, not "Invalid entries"),
opened on a complete day (a date with in-order rows and no possibly_incomplete_day flag), so a rate per
resident-year divides complaints by the days they cover. Population: 2010 Census for days before 2015 and 2020
Census from 2015 on (the same rule as Q7). Totals labeled "all N districts" cover the mapped districts only (no
parks, airports or unlocated complaints). Commuter districts (Q7's COMMUTER_CDS: Manhattan 5, Midtown) are shaded at
their rate but not ranked, as in Q7: ranks run over the other districts. Every number on the map, in the report and
in outputs/answers/q08.md is computed here; the same code runs on one month or twelve years.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nyc311 import geo, rules
from nyc311.clean import clean_glob
from nyc311.q07_geography import COMMUTER_CDS, COMMUTER_LABEL   # one definition: Q7's commuter districts
from nyc311.report import Report

STAGE = "q08_dashboard"
DASHBOARD = Path("dashboard") / "q08_dashboard.html"     # relative to cfg.out_dir; build_report embeds it
DEMO_CSV = geo.DISTRICTS_GEOJSON.parent / "cd_demographics.csv"
COMPLAINTS = rules.COMPLAINTS_SQL
INSTANT = rules.INSTANT_SQL   # as in Q4: closed at the second of opening, with a time of day
POP_2020_FROM_YEAR = 2015   # days before 2015 use population_2010, days from 2015 population_2020 (as in Q7)
DAYS_PER_YEAR = 365.25
MIN_N = 100                 # complaints for a stable rate; closed complaints for a median and an adjusted gap (Q4)
N_CLASSES = 5               # quintiles of the districts
N_THEMES = 4                # extra shadings: the largest complaint categories
NOT_THEMES = ("Other", "Invalid entries", "Information requests")
TOP_TYPES = 5
WINDOW_DAYS = 30            # point layer: the most recent complete days
POINT_CAP = 20_000
SEED = 42
SIMPLIFY_DEG = 0.0001       # Douglas-Peucker tolerance for district outlines (about 10 m)
MAX_HTML_MB = 15
# Base map: Esri's light gray canvas with its labels drawn above the district shading, plus OpenStreetMap's street
# map as an optional overlay. Both load without an API key (checked 2026-10-06; CARTO's tiles now need one).
_ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_ATTR = ("Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; "
             '<a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')
TILES = (_ESRI.format("World_Light_Gray_Base"), ESRI_ATTR)
LABEL_TILES = (_ESRI.format("World_Light_Gray_Reference"), ESRI_ATTR)
ESRI_MAX_NATIVE_ZOOM = 16
MAX_ZOOM = 18
STREET_TILES = ("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')
# District fill fades as staff zoom in, so streets and individual complaints stay readable: (from zoom, opacity).
FADE = [(13, 0.4), (15, 0.12)]
FILL_OPACITY = 0.78
WIDE_FRAME = (1000, 700)   # smaller map frames start with the notes and the layer list folded
POINT_FIELDS = ["lat", "lon", "type", "descriptor", "agency", "status", "opened", "closed", "address", "zip",
                "unique_key", "open"]
EN = "\N{EN DASH}"
MINUS = "\N{MINUS SIGN}"
SANS = '"Helvetica Neue", Helvetica, Arial, sans-serif'
SERIF = 'Georgia, "Times New Roman", serif'
GRAY = "#ececec"           # parks and airports: light fill, dashed gray outline
PARK_LINE = "#9a9a9a"
TEXT = "#333333"
NOTE_GRAY = "#767676"


# ------------------------------------------------------------------ formatting
def _finite(x):
    try:
        return x is not None and bool(np.isfinite(x))
    except TypeError:
        return False


def fmt_rate(x):
    """Rates per 1,000: whole numbers from 100 up, one decimal below."""
    if not _finite(x):
        return "n/a"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}"


def fmt_days(d):
    """Durations given in days: hours below two days, days above (as Q4)."""
    if not _finite(d):
        return "n/a"
    return f"{d * 24:,.1f} hours" if d < 2 else f"{d:,.1f} days"


def _pct1(x):
    return f"{x:.1f}%" if _finite(x) else "n/a"


def fmt_date(d):
    """A date as 'Sep 22, 2021'."""
    d = pd.Timestamp(d)
    return f"{d:%b} {d.day}, {d.year}"


def fmt_pct(x):
    """Signed whole percent with a true minus sign: +12%, \N{MINUS SIGN}8%."""
    return "n/a" if not _finite(x) else f"{x:+,.0f}%".replace("-", MINUS)


def js_escape(s):
    """A string as it appears inside a JSON/JS string literal in folium's output (json.dumps, then <>&' escaped)."""
    return _js(str(s))[1:-1]


def _js(obj):
    return (json.dumps(obj).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("'", "\\u0027"))


def _e(s):
    return html.escape(str(s))


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def theme_col(i):
    return f"theme{i + 1}"


# ------------------------------------------------------------------ geometry
def _rdp(pts, tol):
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, seg = pts[a], pts[a + 1:b]
        d = pts[b] - p
        length = float(np.hypot(*d))
        if length == 0:
            dist = np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / length
        i = int(np.argmax(dist))
        if dist[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def simplify_ring(coords, tol):
    """Douglas-Peucker on one closed ring; rings that would collapse below 4 points are kept as they are."""
    pts = np.asarray(coords, dtype=float)
    if len(pts) <= 4:
        return pts.tolist()
    out = _rdp(pts, tol)
    return (out if len(out) >= 4 else pts).tolist()


def simplify_geojson(gj, tol):
    """Copy of a Polygon/MultiPolygon FeatureCollection with simplified rings rounded to 5 decimals (about 1 m).
    Returns (geojson, vertices before, vertices after)."""
    before = after = 0
    feats = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        new = []
        for rings in polys:
            out = []
            for r in rings:
                s = simplify_ring(r, tol)
                before, after = before + len(r), after + len(s)
                out.append([[round(x, 5), round(y, 5)] for x, y in s])
            new.append(out)
        feats.append({"type": "Feature", "properties": dict(f["properties"]),
                      "geometry": {"type": "MultiPolygon", "coordinates": new}})
    return {"type": "FeatureCollection", "features": feats}, before, after


def load_districts(path=geo.DISTRICTS_GEOJSON):
    return json.loads(Path(path).read_text())


def _bounds(gj):
    xy = np.array([pt for f in gj["features"] for poly in f["geometry"]["coordinates"] for ring in poly
                   for pt in ring])
    return [[float(xy[:, 1].min()), float(xy[:, 0].min())], [float(xy[:, 1].max()), float(xy[:, 0].max())]]


# ------------------------------------------------------------------ classes and colors
def classify(rates, k=N_CLASSES):
    """Quantile classes 0..k-1 (ties broken by order, so every class is filled) and the k+1 class edges."""
    rates = pd.Series(rates, dtype=float)
    k = max(1, min(k, len(rates)))
    cls = pd.Series(pd.qcut(rates.rank(method="first"), k, labels=False), index=rates.index).astype(int)
    edges = [float(rates[cls == c].min()) for c in range(k)] + [float(rates.max())]
    return cls, edges


def rank_districts(boro_cd, rate):
    """Rank 1 = highest rate among districts that are not commuter districts (q07_geography.COMMUTER_CDS); commuter
    districts get no rank (<NA>): their per-resident rate is not comparable with residential districts (as in Q7)."""
    commuter = pd.Series(boro_cd).astype(int).isin(COMMUTER_CDS)
    rate = pd.Series(rate, dtype=float)
    return rate.where(~commuter.to_numpy()).rank(ascending=False, method="first").astype("Int64")


def rank_text(row, rank):
    """'rank 3 of 58', or 'commuter district, not ranked'."""
    if row.get("commuter", False):
        return f"{COMMUTER_LABEL}, not ranked"
    return f"rank {int(rank)} of {int(row.get('n_ranked', row['n_districts']))}"


def ramp(line, k=N_CLASSES):
    """k tints of one hue, light to dark (Q7's ramp, with a darker lightest class so it stands out from the gray
    base map and the gray parks)."""
    import matplotlib.colors as mc

    darkest = np.array(mc.to_rgb(line)) * 0.72
    return [mc.to_hex(1 - a * (1 - darkest)) for a in np.linspace(0.22, 1.0, k)]


def _hues(n):
    from nyc311 import nytplot as nyt
    return [nyt.HUES[i % (len(nyt.HUES) - 1)].line for i in range(n)]  # the last hue (gray) is kept for context


# ------------------------------------------------------------------ data
def load_demographics(path=DEMO_CSV):
    return pd.read_csv(path, dtype={"puma_2010": str})


def person_years(demo, days_by_year):
    """Resident-years per (boro_cd, year): population_2010 for years before POP_2020_FROM_YEAR, else 2020."""
    parts = []
    for y, days in pd.Series(days_by_year).items():
        pop = demo.population_2010 if int(y) < POP_2020_FROM_YEAR else demo.population_2020
        idx = pd.MultiIndex.from_arrays([demo.boro_cd.astype(int), np.full(len(demo), int(y))],
                                        names=["boro_cd", "year"])
        parts.append(pd.Series(pop.to_numpy(float) * float(days) / DAYS_PER_YEAR, index=idx))
    return pd.concat(parts).sort_index()


def complete_days(con, src):
    """Dates with at least one in-order row and no possibly_incomplete_day flag, oldest first."""
    d = con.sql(f"""select created::date as d from {src} where {rules.BASE_SQL} and created is not null
                    group by 1 having bool_or(not out_of_file_order)
                    and not bool_or(coalesce(possibly_incomplete_day, false)) order by 1""").df()
    return [pd.Timestamp(x).date() for x in d.d]


def load_base(con, src, days):
    """Temp table q08 (complaints opened on complete days) and the exclusion waterfall from all rows."""
    con.register("q08_days_df", pd.DataFrame({"d": pd.to_datetime(pd.Series(days, dtype=object))}))
    con.execute("create or replace temp table q08_days as select distinct cast(d as date) as d from q08_days_df")
    con.unregister("q08_days_df")
    con.execute(f"""create or replace temp table q08 as
        select created, year(created) as yr, canonical_type, category, boro_cd, special_area_cd, duration_h,
               closed_at_creation, date_only_created, coalesce(unresolved, false) as unresolved
        from {src} where {COMPLAINTS} and created::date in (select d from q08_days)""")
    w = con.sql(f"""
        select count(*) n_rows, count_if(is_test) n_test, count_if(not is_test and dup_key) n_dup,
               count_if({rules.BASE_SQL} and coalesce(is_info_request, false)) n_info,
               count_if({rules.BASE_SQL} and not coalesce(is_info_request, false) and not ({COMPLAINTS})) n_invalid,
               count_if({COMPLAINTS} and created is null) n_nodate,
               count_if({COMPLAINTS} and created is not null and coalesce(possibly_incomplete_day, false)) n_inc,
               count_if({COMPLAINTS} and created is not null and not coalesce(possibly_incomplete_day, false)
                        and k.d is null) n_outside
        from {src} r left join q08_days k on r.created::date = k.d""").fetchone()
    q = con.sql("""select count(*), count_if(boro_cd is null and special_area_cd is not null),
                          count_if(boro_cd is null and special_area_cd is null), count_if(boro_cd is not null)
                   from q08""").fetchone()
    steps = [("all rows", w[0]), ("- is_test", w[1]), ("- dup_key", w[2]), ("- is_info_request", w[3]),
             (f"- invalid entries (category '{rules.INVALID_CATEGORY}')", w[4]),
             ("- no created date", w[5]), ("- possibly_incomplete_day", w[6]),
             ("- dated outside the files' period (only out_of_file_order rows that day)", w[7]),
             ("complaints on complete days", q[0]), ("- parks or airports (special_area_cd)", q[1]),
             ("- no community district", q[2]), ("mapped to a community district", q[3])]
    df = pd.DataFrame(steps, columns=["step", "rows"])
    df["rows"] = df.rows.astype(int)
    df["pct"] = 100 * df.rows / max(int(w[0]), 1)
    return df


def choose_themes(con, k=N_THEMES):
    """The k largest complaint categories among mapped complaints (catch-all categories left out)."""
    c = con.sql("""select category, count(*) n from q08 where boro_cd is not null and category is not null
                   group by 1 order by n desc, category""").df()
    return c[~c.category.isin(NOT_THEMES)].category.head(k).tolist()


def district_tables(con, demo, days_by_year, themes):
    """One row per standard district (all 59), the per-year table, the top types and the city figures."""
    from nyc311.q04_response_times import mix_adjusted

    py = person_years(demo, days_by_year)
    d = demo[["boro_cd", "borough", "cd_number", "cd_name", "population_2010", "population_2020"]].copy()
    d["boro_cd"] = d.boro_cd.astype(int)
    d["label"] = [f"{r.borough} {r.cd_number} ({r.cd_name})" for r in d.itertuples()]  # as Q4's district_label
    d["person_years"] = d.boro_cd.map(py.groupby(level=0).sum())
    base = con.sql("""select boro_cd, count(*) n, count_if(unresolved) n_unresolved from q08
                      where boro_cd is not null group by 1""").df().set_index("boro_cd")
    d["n"] = d.boro_cd.map(base.n).fillna(0).astype(int)
    d["n_unresolved"] = d.boro_cd.map(base.n_unresolved).fillna(0).astype(int)
    d["rate"] = 1000 * d.n / d.person_years
    d["commuter"] = d.boro_cd.isin(COMMUTER_CDS)
    d["rank"] = rank_districts(d.boro_cd, d.rate)
    d["n_districts"] = len(d)
    d["n_ranked"] = int((~d.commuter).sum())
    d["low_n"] = d.n < MIN_N
    d["pct_unresolved"] = 100 * d.n_unresolved / d.n.where(d.n > 0)
    d["rate_class"] = classify(d.rate)[0]

    con.execute(f"""create or replace temp view q08r as select boro_cd, canonical_type, created, duration_h
                    from q08 where duration_h is not null and not ({INSTANT})""")
    mix = mix_adjusted(con, "q08r", "boro_cd", min_n=1).rename(columns={"grp": "boro_cd", "n": "n_closed"})
    mix = mix.assign(boro_cd=mix.boro_cd.astype(int)).set_index("boro_cd")
    d["n_closed"] = d.boro_cd.map(mix.n_closed).fillna(0).astype(int)
    d["median_days"] = d.boro_cd.map(mix.median_h) / 24
    d["adj_pct"] = d.boro_cd.map(mix.adj_pct)

    cats = con.sql("""select boro_cd, category, count(*) n from q08 where boro_cd is not null and category is not null
                      group by all""").df()
    for i, t in enumerate(themes):
        col = theme_col(i)
        n = cats[cats.category == t].set_index("boro_cd").n
        d[f"{col}_n"] = d.boro_cd.map(n).fillna(0).astype(int)
        d[f"{col}_rate"] = 1000 * d[f"{col}_n"] / d.person_years
        d[f"{col}_rank"] = rank_districts(d.boro_cd, d[f"{col}_rate"])
        d[f"{col}_class"] = classify(d[f"{col}_rate"])[0]

    top = con.sql(f"""select boro_cd, canonical_type, n from (
                          select boro_cd, canonical_type, count(*) n from q08
                          where boro_cd is not null and canonical_type is not null group by all) t
                      qualify row_number() over (partition by boro_cd order by n desc, canonical_type) <= {TOP_TYPES}
                      order by boro_cd, n desc, canonical_type""").df()
    top["boro_cd"] = top.boro_cd.astype(int)
    top["n"] = top.n.astype(int)
    top["pct"] = 100 * top.n / top.boro_cd.map(d.set_index("boro_cd").n)
    listed = {}
    for r in top.itertuples():
        listed.setdefault(r.boro_cd, []).append(f"{r.canonical_type} ({r.n:,})")
    d["top_types"] = ["; ".join(listed.get(cd, [])) for cd in d.boro_cd]

    yc = con.sql("select boro_cd, yr as year, count(*) n from q08 where boro_cd is not null group by all").df()
    yc = yc.astype({"boro_cd": int, "year": int}).set_index(["boro_cd", "year"]).n
    years = py.rename("person_years").to_frame()
    years["n"] = yc.reindex(years.index).fillna(0).astype(int)
    years = years.reset_index()
    years["complete_days"] = years.year.map(pd.Series(days_by_year)).astype(int)
    years["rate"] = 1000 * years.n / years.person_years
    years = years[["boro_cd", "year", "complete_days", "person_years", "n", "rate"]]

    city_med = con.sql("select quantile_cont(duration_h, 0.5) / 24 from q08r where boro_cd is not null").fetchone()[0]
    city = {"n": int(d.n.sum()), "rate": 1000 * d.n.sum() / d.person_years.sum(),
            "median_days": float(city_med) if city_med is not None else float("nan"),
            "pct_unresolved": 100 * d.n_unresolved.sum() / max(int(d.n.sum()), 1),
            "n_closed": int(d.n_closed.sum())}
    for i in range(len(themes)):
        city[f"{theme_col(i)}_rate"] = 1000 * d[f"{theme_col(i)}_n"].sum() / d.person_years.sum()
    return d, years, top, city


def recent_unresolved(con, window_days):
    """Share unresolved among mapped complaints opened in the point window (the most recent complete days) and on
    the complete days before it, so any claim about recent complaints is computed."""
    w0 = min(window_days)
    n_r, u_r, n_b, u_b = con.sql(f"""
        select count_if(created::date >= DATE '{w0}'), count_if(created::date >= DATE '{w0}' and unresolved),
               count_if(created::date < DATE '{w0}'), count_if(created::date < DATE '{w0}' and unresolved)
        from q08 where boro_cd is not null""").fetchone()
    return {"days": len(window_days), "n_recent": int(n_r), "n_before": int(n_b),
            "pct_recent": 100 * u_r / n_r if n_r else float("nan"),
            "pct_before": 100 * u_b / n_b if n_b else float("nan")}


def unresolved_caveat(ru):
    """One clause on whether recent complaints inflate the unresolved shares, from recent_unresolved()."""
    if not ru["n_recent"] or not ru["n_before"]:
        return ("the period is too short to compare the share unresolved for recent and earlier complaints, so part "
                "of it may be complaints that have not had time to close")
    recent = (f"{ru['pct_recent']:.1f}% of complaints from the {ru['days']} most recent complete days are unresolved "
              f"vs. {ru['pct_before']:.1f}% for earlier days")
    if ru["pct_recent"] > ru["pct_before"]:
        return (f"{recent}, so part of a district's unresolved share is recent complaints that have had less time "
                "to close")
    return f"the share unresolved is not higher for recent complaints ({recent}), so they do not inflate it"


def data_end(con, src, last_complete):
    """Last opening date among in-order rows, and how many such dates fall after the last complete day."""
    end, n_after = con.sql(f"""select max(created::date), count(distinct created::date)
                                    filter (where created::date > DATE '{last_complete}')
                               from {src} where {rules.BASE_SQL} and not out_of_file_order""").fetchone()
    return pd.Timestamp(end).date(), int(n_after)


def special_counts(con):
    s = con.sql("""select special_area_cd, count(*) n from q08 where boro_cd is null and special_area_cd is not null
                   group by 1""").df()
    return dict(zip(s.special_area_cd.astype(int), s.n.astype(int)))


# ------------------------------------------------------------------ points
def _window_where(window_days):
    days = ", ".join(f"DATE '{d}'" for d in window_days)
    return f"{COMPLAINTS} and created::date in ({days})"


def window_counts(con, src, window_days):
    n, eligible = con.sql(f"""select count(*), count_if(latitude is not null and longitude is not null)
                             from {src} where {_window_where(window_days)}""").fetchone()
    return int(n), int(eligible)


def sample_points(con, src, window_days, cap=POINT_CAP, seed=SEED):
    """At most `cap` complaints with coordinates from the window: those with the smallest seeded hash of unique_key
    (a uniform random sample that does not depend on row order or thread count), oldest first."""
    return con.sql(f"""
        select unique_key, created, closed, latitude, longitude, canonical_type, descriptor, agency, status,
               coalesce(unresolved, false) as unresolved, date_only_created, date_only_closed,
               coalesce(incident_address,
                        case when intersection_street_1 is not null and intersection_street_2 is not null
                             then intersection_street_1 || ' & ' || intersection_street_2 end,
                        street_name) as address, zip
        from {src} where {_window_where(window_days)} and latitude is not null and longitude is not null
        order by hash(unique_key || ':{int(seed)}'), unique_key limit {int(cap)}""").df() \
        .sort_values(["created", "unique_key"]).reset_index(drop=True)


def _stamp(ts, date_only):
    if ts is None or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    return f"{ts:%Y-%m-%d}" if bool(date_only) else f"{ts:%Y-%m-%d %H:%M}"


def point_rows(p):
    """Compact rows for the cluster layer (repeated text replaced by indexes into lookup lists)."""
    luts = {}
    codes = {}
    for col in ["canonical_type", "descriptor", "agency", "status"]:
        vals = sorted({str(v) for v in p[col] if isinstance(v, str)})
        luts[col] = vals
        pos = {v: i for i, v in enumerate(vals)}
        codes[col] = [pos.get(v) if isinstance(v, str) else None for v in p[col]]
    rows = []
    for i, r in enumerate(p.itertuples(index=False)):
        rows.append([round(float(r.latitude), 5), round(float(r.longitude), 5), codes["canonical_type"][i],
                     codes["descriptor"][i], codes["agency"][i], codes["status"][i],
                     _stamp(r.created, r.date_only_created), _stamp(r.closed, r.date_only_closed),
                     r.address if isinstance(r.address, str) else None, r.zip if isinstance(r.zip, str) else None,
                     str(r.unique_key), int(bool(r.unresolved))])
    return rows, luts


def point_callback(luts):
    """JS for FastMarkerCluster: one small dot per complaint (coral if still open) with a lazily built popup."""
    f = {k: POINT_FIELDS.index(k) for k in POINT_FIELDS}
    return f"""(function () {{
    var T = {_js(luts["canonical_type"])}, D = {_js(luts["descriptor"])}, A = {_js(luts["agency"])},
        S = {_js(luts["status"])};
    var icons = [L.divIcon({{className: "q08-pt q08-pt-closed", iconSize: [10, 10]}}),
                 L.divIcon({{className: "q08-pt q08-pt-open", iconSize: [10, 10]}})];
    function esc(s) {{
        return String(s).replace(/[&<>"]/g, function (c) {{
            return {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}}[c]; }});
    }}
    function pick(a, i) {{ return i === null || i === undefined ? null : a[i]; }}
    function line(k, v) {{
        return v === null || v === undefined || v === "" ? "" : "<tr><td>" + k + "</td><td>" + esc(v) + "</td></tr>";
    }}
    return function (row) {{
        var m = L.marker([row[0], row[1]], {{icon: icons[row[{f["open"]}]]}});
        m.bindPopup(function () {{
            return '<div class="q08-pop"><div class="q08-h">' + esc(pick(T, row[{f["type"]}]) || "(no type)") +
                '</div><table class="q08-t">' + line("Descriptor", pick(D, row[{f["descriptor"]}])) +
                line("Agency", pick(A, row[{f["agency"]}])) + line("Opened", row[{f["opened"]}]) +
                line("Closed", row[{f["closed"]}]) + line("Status", pick(S, row[{f["status"]}])) +
                line("Address", row[{f["address"]}]) + line("ZIP", row[{f["zip"]}]) +
                line("Request #", row[{f["unique_key"]}]) + '</table></div>';
        }}, {{maxWidth: 320}});
        return m;
    }};
}})()"""


CLUSTER_ICON = """function (cluster) {
    var n = cluster.getChildCount();
    var s = n < 10 ? 26 : n < 100 ? 32 : n < 1000 ? 38 : 46;
    return L.divIcon({html: "<div><span>" + n.toLocaleString() + "</span></div>", className: "q08-cluster",
                      iconSize: L.point(s, s)});
}"""


# ------------------------------------------------------------------ popups, tooltips, legends
def sparkline_svg(years, width=220, height=44):
    """Inline SVG bars of the rate per year (one bar per year, first and last year labeled); '' for one year."""
    years = years.sort_values("year")
    if len(years) < 2:
        return ""
    from nyc311 import nytplot as nyt

    vmax = float(years.rate.max()) or 1.0
    bar_h, gap = height - 14, 2
    w = (width - gap * (len(years) - 1)) / len(years)
    parts = [f'<svg class="q08-spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'xmlns="http://www.w3.org/2000/svg" role="img">']
    for i, r in enumerate(years.itertuples()):
        h = max(bar_h * float(r.rate) / vmax, 0.5)
        x = i * (w + gap)
        parts.append(f'<rect x="{x:.1f}" y="{bar_h - h:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{nyt.SAGE.band}">'
                     f'<title>{int(r.year)}: {int(r.n):,} complaints, {fmt_rate(r.rate)} per 1,000 residents a year'
                     f'</title></rect>')
    parts.append(f'<line x1="0" y1="{bar_h}" x2="{width}" y2="{bar_h}" stroke="{TEXT}" stroke-width="1"/>')
    first, last = years.iloc[0], years.iloc[-1]
    parts.append(f'<text x="0" y="{height - 2}" font-size="10" fill="{NOTE_GRAY}">{int(first.year)}</text>')
    parts.append(f'<text x="{width}" y="{height - 2}" font-size="10" fill="{NOTE_GRAY}" text-anchor="end">'
                 f'{int(last.year)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _row(k, v):
    return f"<tr><td>{k}</td><td>{v}</td></tr>"


def district_popup(row, top, years, city):
    """Click popup for one district: every value comes from the district table, its top types and years. The
    comparison figures (`city`) are totals over all mapped districts, and are labeled that way."""
    n = int(row["n"])
    area = f"all {int(row['n_districts'])} districts"
    rate_line = f"<b>{fmt_rate(row['rate'])}</b> ({area}: {fmt_rate(city['rate'])}; {rank_text(row, row['rank'])})"
    rows = [_row("Complaints", f"<b>{n:,}</b>"), _row("Per 1,000 residents a year", rate_line)]
    if row.get("commuter", False):
        rows.append(_row("", f"<i>{COMMUTER_LABEL}: many workers and visitors, few residents, so its rate per "
                             "resident is not comparable with residential districts</i>"))
    if row.get("low_n"):
        rows.append(_row("", f"<i>fewer than {MIN_N} complaints: rate unstable</i>"))
    if int(row["n_closed"]) >= MIN_N:
        gap = row["adj_pct"]
        word = "slower" if gap > 0 else "faster"
        rows.append(_row("Median time to close", f"<b>{fmt_days(row['median_days'])}</b> "
                                                 f"({area}: {fmt_days(city['median_days'])})"))
        rows.append(_row("vs. same complaint types citywide", f"<b>{fmt_pct(gap)}</b> ({word} than typical)"))
    else:
        rows.append(_row("Median time to close", f"<i>fewer than {MIN_N} closed complaints "
                                                 f"({int(row['n_closed']):,}): not shown</i>"))
    if n:
        rows.append(_row("Still unresolved", f"<b>{row['pct_unresolved']:.1f}%</b> "
                                             f"({area}: {city['pct_unresolved']:.1f}%)"))
    parts = [f'<div class="q08-pop"><div class="q08-h">{_e(row["label"])}</div>']
    if row.get("pop_text"):
        parts.append(f'<div class="q08-sub">{_e(row["pop_text"])}</div>')
    parts.append(f'<table class="q08-t">{"".join(rows)}</table>')
    if len(top):
        parts.append(f'<div class="q08-s">Top {len(top)} complaint types</div><table class="q08-t q08-num">')
        for i, r in enumerate(top.itertuples(), 1):
            share = f"{100 * r.n / n:.0f}%" if n else ""
            parts.append(f"<tr><td>{i}. {_e(r.canonical_type)}</td><td>{int(r.n):,}</td><td>{share}</td></tr>")
        parts.append("</table>")
    if len(years):
        y = years.sort_values("year")
        spark = sparkline_svg(y)
        if spark:
            parts.append(f'<div class="q08-s">Complaints per 1,000 residents a year, {int(y.year.iloc[0])}{EN}'
                         f'{int(y.year.iloc[-1])}</div>{spark}')
        parts.append(f'<details><summary>{"Table by year" if spark else "By year"}</summary>'
                     '<table class="q08-t q08-num"><tr><th>Year</th>'
                     '<th>Days</th><th>Complaints</th><th>Per 1,000/yr</th></tr>')
        for r in y.itertuples():
            parts.append(f"<tr><td>{int(r.year)}</td><td>{int(r.complete_days)}</td><td>{int(r.n):,}</td>"
                         f"<td>{fmt_rate(r.rate)}</td></tr>")
        parts.append("</table></details>")
    parts.append(f'<div class="q08-note">Time to close: closed complaints without instant (0-second) closures, '
                 f'n = {int(row["n_closed"]):,}; "vs. same complaint types" compares each closure with the citywide '
                 f'typical time for its complaint type and year. Days = complete days of data that year.</div></div>')
    return "".join(parts)


def _pop_text(r, years_used):
    early, late = [y for y in years_used if y < POP_2020_FROM_YEAR], [y for y in years_used if y >= POP_2020_FROM_YEAR]
    if early and late:
        return (f"Residents: {int(r.population_2010):,} (2010 Census, used to {POP_2020_FROM_YEAR - 1}), "
                f"{int(r.population_2020):,} (2020 Census, used from {POP_2020_FROM_YEAR})")
    if late:
        return f"Residents: {int(r.population_2020):,} (2020 Census)"
    return f"Residents: {int(r.population_2010):,} (2010 Census)"


def tooltip(row, what, rate, rank):
    tip = (f"<b>{_e(row['label'])}</b><br>{_e(what)}: {fmt_rate(rate)} per 1,000 residents a year "
           f"({rank_text(row, rank)})")
    if row.get("low_n"):
        tip += f"<br><i>fewer than {MIN_N} complaints: rate unstable</i>"
    return tip + "<br><span class='q08-hint'>Click for details</span>"


def legend_html(title, ranges, colors, n_special):
    """Legend block: one swatch per class with the class's lowest and highest district rate."""
    rows = [f'<div class="q08-lt">{_e(title)}</div>']
    for (lo, hi, k), c in zip(ranges, colors):
        rows.append(f'<div class="q08-lr"><span class="q08-sw" style="background:{c}"></span>'
                    f'{fmt_rate(lo)}{EN}{fmt_rate(hi)} <span class="q08-hint">({k} districts)</span></div>')
    rows.append(f'<div class="q08-lr"><span class="q08-sw" style="background:{GRAY};outline:1px dashed '
                f'{PARK_LINE};border:none"></span>'
                f'Park or airport ({n_special:,} complaints, no rate)</div>')
    return "".join(rows)


def class_ranges(rates, cls):
    """(lowest, highest, number of districts) for each class, lowest class first."""
    rates, cls = pd.Series(rates, dtype=float), pd.Series(cls)
    return [(float(rates[cls == c].min()), float(rates[cls == c].max()), int((cls == c).sum()))
            for c in sorted(cls.unique())]


# ------------------------------------------------------------------ the map
CSS = f"""<style>
#q08-panel {{position: fixed; top: 10px; left: 56px; z-index: 1000; max-width: 270px; max-height: calc(100% - 30px);
  overflow-y: auto; background: rgba(255,255,255,.95); padding: 9px 12px; border-radius: 4px; color: {TEXT};
  box-shadow: 0 1px 5px rgba(0,0,0,.25); font: 12px/1.4 {SANS};}}
.q08-title {{font-weight: bold; font-size: 14.5px; line-height: 1.25; margin-bottom: 3px;}}
#q08-more {{margin-top: 6px;}} #q08-more summary {{cursor: pointer; font-weight: bold; color: {NOTE_GRAY};}}
.q08-meta {{color: {TEXT}; margin-bottom: 6px;}}
.q08-lt, .q08-s {{font-weight: bold; margin: 7px 0 3px;}}
.q08-lr {{display: flex; align-items: center; gap: 6px; line-height: 1.5;}}
.q08-sw {{display: inline-block; width: 16px; height: 11px; border: 1px solid #fff; outline: 1px solid #e5e5e5;}}
.q08-dot {{display: inline-block; width: 9px; height: 9px; border-radius: 50%; vertical-align: -1px; margin: 0 2px 0 4px;}}
.q08-note {{font: 11px/1.4 {SERIF}; color: {NOTE_GRAY}; margin-top: 7px;}}
.q08-hint {{color: {NOTE_GRAY}; font-size: 11px;}}
.q08-pop {{font: 12px/1.4 {SANS}; color: {TEXT}; min-width: 250px;}}
.q08-h {{font-weight: bold; font-size: 14px; margin-bottom: 2px;}}
.q08-sub {{color: {NOTE_GRAY}; margin-bottom: 4px;}}
.q08-t {{border-collapse: collapse; width: 100%;}}
.q08-t td, .q08-t th {{padding: 1px 6px 1px 0; vertical-align: top; text-align: left;}}
.q08-t th {{font-weight: normal; color: {NOTE_GRAY};}}
.q08-num td:nth-child(n+2), .q08-num th:nth-child(n+2) {{text-align: right;}}
.q08-pop details {{margin-top: 4px;}} .q08-pop summary {{cursor: pointer; color: {NOTE_GRAY};}}
.q08-pt {{border-radius: 50%; border: 1.5px solid #fff; box-shadow: 0 0 1px rgba(0,0,0,.7);}}
.q08-pt-closed {{background: #6b817a;}} .q08-pt-open {{background: #cf715c;}}
.q08-cluster div {{width: 100%; height: 100%; border-radius: 50%; background: rgba(51,51,51,.62); color: #fff;
  border: 2px solid rgba(255,255,255,.9); box-sizing: border-box; display: flex; align-items: center;
  justify-content: center; font: bold 11px {SANS};}}
.leaflet-control-layers {{font: 12px/1.5 {SANS}; color: {TEXT};}}
""" + "".join(f".q08-fade{i} .leaflet-overlay-pane path {{fill-opacity: {op} !important;}}\n"
              for i, (_, op) in enumerate(FADE)) + "</style>"


def _panel_element(body, legends, control):
    from branca.element import MacroElement
    from jinja2 import Template

    class Panel(MacroElement):
        _template = Template("""
            {% macro html(this, kwargs) %}{{ this.body }}{% endmacro %}
            {% macro script(this, kwargs) %}
            var q08_legends = {{ this.legends|tojson }};
            var q08_fade = {{ this.fade|tojson }};
            {{ this._parent.get_name() }}.on("baselayerchange", function (e) {
                var el = document.getElementById("q08-legend");
                if (el && q08_legends[e.name] !== undefined) { el.innerHTML = q08_legends[e.name]; }
            });
            function q08_zoom() {
                var z = {{ this._parent.get_name() }}.getZoom(), box = {{ this._parent.get_name() }}.getContainer();
                q08_fade.forEach(function (f, i) {
                    var next = q08_fade[i + 1];
                    box.classList.toggle("q08-fade" + i, z >= f[0] && (!next || z < next[0]));
                });
            }
            {{ this._parent.get_name() }}.on("zoomend", q08_zoom);
            q08_zoom();
            (function () {  // small frames (e.g. embedded in the report): fold the notes and the layer list
                var box = {{ this._parent.get_name() }}.getContainer(), lc = {{ this.control.get_name() }};
                if (box.clientWidth >= {{ this.wide[0] }} && box.clientHeight >= {{ this.wide[1] }}) { return; }
                var more = document.getElementById("q08-more");
                if (more) { more.open = false; }
                lc.collapse();
                L.DomEvent.on(lc.getContainer(), {mouseenter: lc.expand, mouseleave: lc.collapse}, lc);
            })();
            {% endmacro %}""")

        def __init__(self):
            super().__init__()
            self._name = "Q08Panel"
            self.body, self.legends, self.fade, self.control, self.wide = body, legends, FADE, control, WIDE_FRAME

    return Panel()


def _fix_ids(root):
    """Number every element in tree order so the HTML is byte-identical between runs (folium ids are random)."""
    n = 0

    def walk(el):
        nonlocal n
        el._id = f"{n:08x}"
        n += 1
        for c in list(el._children.values()):
            walk(c)

    walk(root)


def build_map(simple, d, themes, popups, special_n, rows, luts, texts):
    import folium
    from branca.element import Element
    from folium import plugins

    std = {"type": "FeatureCollection",
           "features": [f for f in simple["features"] if not f["properties"].get("is_jia")]}
    jia = {"type": "FeatureCollection", "features": [f for f in simple["features"] if f["properties"].get("is_jia")]}
    bounds = _bounds(std)
    m = folium.Map(location=[(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2], zoom_start=10,
                   tiles=None, control_scale=True, max_zoom=MAX_ZOOM)
    m.get_root().header.add_child(Element(f"<title>{_e(texts['title'])}</title>"), name="q08_title")
    m.get_root().header.add_child(Element(CSS), name="q08_css")
    folium.TileLayer(TILES[0], attr=TILES[1], max_native_zoom=ESRI_MAX_NATIVE_ZOOM, max_zoom=MAX_ZOOM,
                     overlay=True, control=False, name="Base map").add_to(m)
    folium.TileLayer(STREET_TILES[0], attr=STREET_TILES[1], max_zoom=MAX_ZOOM, overlay=True, show=False,
                     name=texts["layer_names"]["streets"]).add_to(m)
    folium.map.CustomPane("q08labels", z_index=450, pointer_events=False).add_to(m)  # above the district fills
    folium.TileLayer(LABEL_TILES[0], attr=LABEL_TILES[1], max_native_zoom=ESRI_MAX_NATIVE_ZOOM, max_zoom=MAX_ZOOM,
                     pane="q08labels", overlay=True, control=False, name="Place names").add_to(m)

    by_cd = d.set_index("boro_cd")
    names, legends = texts["layer_names"], {}
    shadings = [(names["all"], "rate", "rank", "rate_class", "All complaints", None)]
    shadings += [(names["themes"][i], f"{theme_col(i)}_rate", f"{theme_col(i)}_rank", f"{theme_col(i)}_class", t, i)
                 for i, t in enumerate(themes)]
    for j, (name, rate_col, rank_col, cls_col, what, ti) in enumerate(shadings):
        colors = ramp(_hues(len(shadings))[j])
        legends[name] = legend_html(f"{what}: complaints per 1,000 residents a year",
                                    class_ranges(d[rate_col], d[cls_col]), colors, sum(special_n.values()))
        feats = []
        for f in std["features"]:
            cd = int(f["properties"]["boro_cd"])
            r = by_cd.loc[cd]
            head = ""
            if ti is not None:
                head = (f'<div class="q08-theme"><b>{_e(what)}</b>: {fmt_rate(r[rate_col])} per 1,000 residents a '
                        f'year ({rank_text(r, r[rank_col])}; '
                        f'{int(r[f"{theme_col(ti)}_n"]):,} complaints)</div>')
            feats.append({"type": "Feature", "geometry": f["geometry"],
                          "properties": {"boro_cd": cd, "fill": colors[int(r[cls_col])],
                                         "tip": tooltip(r, what, r[rate_col], r[rank_col]),
                                         "popup": head + popups[cd]}})
        folium.GeoJson(
            {"type": "FeatureCollection", "features": feats}, name=name, overlay=False, show=(j == 0),
            style_function=lambda f: {"fillColor": f["properties"]["fill"], "color": "#ffffff", "weight": 0.8,
                                      "fillOpacity": FILL_OPACITY},
            highlight_function=lambda f: {"weight": 2.5, "color": TEXT, "fillOpacity": 0.9},
            tooltip=folium.GeoJsonTooltip(["tip"], labels=False, sticky=True),
            popup=folium.GeoJsonPopup(["popup"], labels=False, max_width=360, max_height=430),
        ).add_to(m)

    jfeats = []
    for f in jia["features"]:
        cd = int(f["properties"]["boro_cd"])
        jfeats.append({"type": "Feature", "geometry": f["geometry"],
                       "properties": {"boro_cd": cd,
                                      "tip": f"<b>Park or airport area {cd}</b> (joint interest area)<br>"
                                             f"{special_n.get(cd, 0):,} complaints; few residents, so no rate"}})
    if jfeats:
        folium.GeoJson({"type": "FeatureCollection", "features": jfeats}, name=names["parks"], overlay=True,
                       show=True, style_function=lambda f: {"fillColor": GRAY, "color": PARK_LINE, "weight": 0.8,
                                                            "dashArray": "2 2", "fillOpacity": 0.85},
                       tooltip=folium.GeoJsonTooltip(["tip"], labels=False, sticky=True)).add_to(m)

    plugins.FastMarkerCluster(rows, callback=point_callback(luts), name=names["points"], overlay=True,
                              icon_create_function=CLUSTER_ICON, show_coverage_on_hover=False,
                              max_cluster_radius=50, chunked_loading=True).add_to(m)
    lc = folium.LayerControl(position="bottomright", collapsed=False)
    lc.add_to(m)
    m.add_child(_panel_element(texts["panel"].replace("{legend}", legends[names["all"]]), legends, lc))
    m.fit_bounds(bounds)
    _fix_ids(m.get_root())
    return m


# ------------------------------------------------------------------ run
def run(cfg):
    from nyc311.q05_top_types import period_label, period_long

    con = cfg.connect()
    src = clean_glob(cfg)
    days = complete_days(con, src)
    if not days:
        raise RuntimeError("Q8: no complete days in the clean table")
    waterfall = load_base(con, src, days)
    demo = load_demographics()
    days_by_year = pd.Series([d.year for d in days]).value_counts().sort_index()
    themes = choose_themes(con)
    d, years, top, city = district_tables(con, demo, days_by_year, themes)
    d["pop_text"] = [_pop_text(r, list(days_by_year.index)) for r in d.itertuples()]
    special_n = special_counts(con)

    window = days[-WINDOW_DAYS:]
    n_window, eligible = window_counts(con, src, window)
    pts = sample_points(con, src, window)
    rows, luts = point_rows(pts)
    point_info = {"window_days": window, "n_window": n_window, "eligible": eligible, "no_coords": n_window - eligible,
                  "sampled": len(pts), "cap": POINT_CAP, "seed": SEED, "n_open": int(pts.unresolved.sum()),
                  "point_bytes": len(_js(rows)) + len(point_callback(luts)),
                  "n_skipped": (window[-1] - window[0]).days + 1 - len(window)}
    point_info["data_end"], point_info["n_days_after"] = data_end(con, src, days[-1])

    gj = load_districts()
    simple, v0, v1 = simplify_geojson(gj, SIMPLIFY_DEG)
    popups = {}
    for r in d.itertuples():
        row = pd.Series(r._asdict())
        popups[int(r.boro_cd)] = district_popup(row, top[top.boro_cd == r.boro_cd],
                                                years[years.boro_cd == r.boro_cd], city)

    d0, d1 = days[0], days[-1]
    res = {"waterfall": waterfall, "districts": d, "years": years, "top_types": top, "city": city, "themes": themes,
           "period": (d0, d1), "period_label": period_label(d0, d1), "period_long": period_long(d0, d1),
           "n_days": len(days), "days_by_year": days_by_year, "points": pts, "point_info": point_info,
           "window_label": period_label(window[0], window[-1]), "popups": popups, "special_n": special_n,
           "vertices": (v0, v1), "n_inc_days": _n_incomplete_days(con, src),
           "recent_unresolved": recent_unresolved(con, window)}
    res["layer_names"] = {"all": "All complaints", "themes": list(themes), "parks": "Parks and airports",
                          "points": "Individual complaints (sample)", "streets": "Street map"}
    texts = {"title": f"NYC 311 complaints dashboard, {res['period_label']}", "layer_names": res["layer_names"],
             "panel": _panel_html(res)}
    m = build_map(simple, d, themes, popups, special_n, rows, luts, texts)
    path = cfg.out_dir / DASHBOARD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(m.get_root().render(), encoding="utf-8")
    res["html_bytes"] = path.stat().st_size

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    _summary_csv(d, themes).to_csv(agg / "q08_district_summary.csv", index=False)
    years.round(4).to_csv(agg / "q08_district_years.csv", index=False)
    _report(cfg, res)
    _write_answer(cfg, res)
    return res


def _n_incomplete_days(con, src):
    return int(con.sql(f"""select count(distinct created::date) from {src} where {rules.BASE_SQL}
                           and coalesce(possibly_incomplete_day, false)""").fetchone()[0])


def _summary_csv(d, themes):
    cols = ["boro_cd", "borough", "cd_number", "cd_name", "population_2010", "population_2020", "person_years", "n",
            "rate", "rank", "commuter", "rate_class", "low_n", "n_unresolved", "pct_unresolved", "n_closed", "median_days",
            "adj_pct", "top_types"]
    out = d[cols].rename(columns={"rate": "rate_per_1000_yr"})
    for i, t in enumerate(themes):
        s = _slug(t)
        out[f"n_{s}"] = d[f"{theme_col(i)}_n"]
        out[f"rate_{s}"] = d[f"{theme_col(i)}_rate"]
    return out.round(4)


def _excluded(res):
    w = dict(zip(res["waterfall"].step, res["waterfall"].rows))
    return {"info": w["- is_info_request"], "test": w["- is_test"], "dup": w["- dup_key"],
            "invalid": w[f"- invalid entries (category '{rules.INVALID_CATEGORY}')"],
            "nodate": w["- no created date"], "inc": w["- possibly_incomplete_day"],
            "outside": w["- dated outside the files' period (only out_of_file_order rows that day)"],
            "special": w["- parks or airports (special_area_cd)"], "nocd": w["- no community district"],
            "complete": w["complaints on complete days"], "mapped": w["mapped to a community district"],
            "rows": w["all rows"]}


def _commuter_note(d):
    """' Manhattan 5 (Midtown Business District) is a commuter district: shaded at its rate, not ranked (ranks run
    over the other 58).' (empty without commuter districts)."""
    comm = d[d.commuter]
    if comm.empty:
        return ""
    names = ", ".join(_e(x) for x in comm.label)
    verb = "is a" if len(comm) == 1 else "are"
    return (f" {names} {verb} {COMMUTER_LABEL}{'' if len(comm) == 1 else 's'} (many workers and visitors, few "
            f"residents): shaded at {'its' if len(comm) == 1 else 'their'} rate but not ranked; ranks run over the "
            f"other {int((~d.commuter).sum())}.")


def _panel_html(res):
    x, p, d = _excluded(res), res["point_info"], res["districts"]
    yrs = list(res["days_by_year"].index)
    pop = ("2010 Census population" if max(yrs) < POP_2020_FROM_YEAR else "2020 Census population"
           if min(yrs) >= POP_2020_FROM_YEAR else
           f"2010 Census population before {POP_2020_FROM_YEAR}, 2020 Census from {POP_2020_FROM_YEAR}")
    outside = f", {x['outside']:,} dated outside the files' period" if x["outside"] else ""
    skipped = f"; {p['n_skipped']} incomplete days in between skipped" if p["n_skipped"] else ""
    later = (f" The {p['n_days_after']} later days, to {_e(fmt_date(p['data_end']))}, are partly missing from the "
             "files and not shown." if p["n_days_after"] else "")
    share = 100 * p["sampled"] / p["eligible"] if p["eligible"] else 0
    return (
        '<div id="q08-panel"><div class="q08-title">NYC 311 complaints by community district</div>'
        f'<div class="q08-meta">{_e(res["period_long"])}: {x["mapped"]:,} complaints in {len(d)} districts, '
        f'{res["n_days"]:,} complete days. All {len(d)} districts: {fmt_rate(res["city"]["rate"])} per 1,000 '
        f'residents a year.{_commuter_note(d)}</div>'
        '<div id="q08-legend">{legend}</div>'
        '<div class="q08-hint">Hover or click a district for details; layers at bottom right.</div>'
        '<details id="q08-more" open><summary>About this map</summary>'
        '<div class="q08-s">Individual complaints</div>'
        f'<div>The {len(res["point_info"]["window_days"])} most recent complete days '
        f'({_e(res["window_label"])}{skipped}): a random sample of {p["sampled"]:,} of {p["eligible"]:,} complaints '
        f'with a location ({share:.0f}%).{later} Circles count sampled complaints; zoom in or click one to split it.'
        '<br><span class="q08-dot q08-pt-open"></span>still open<span class="q08-dot q08-pt-closed"></span>'
        'closed or other</div>'
        '<div class="q08-s">How to use</div><div>Hover over a district for its rate; click it for details. Switch '
        'the shading or hide layers at bottom right. Zoom in and the shading fades so streets show.</div>'
        f'<div class="q08-note">Complaints only, opened on complete days. Left out: {x["info"]:,} information '
        f'requests, {x["invalid"]:,} invalid entries, {x["test"]:,} test records, {x["dup"]:,} duplicate keys, '
        f'{x["inc"]:,} complaints on '
        f'{res["n_inc_days"]} days partly missing from the files{outside}. Not shaded: {x["special"]:,} in parks '
        f'and airports, {x["nocd"]:,} with no district. Rates use the {pop}. Sources: NYC 311 Service Requests; '
        f'district outlines NYC Department of City Planning.</div></details></div>')


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    d, p, x, c = res["districts"], res["point_info"], _excluded(res), res["city"]
    v0, v1 = res["vertices"]
    rep.add(f"Q8. Neighborhood dashboard: {DASHBOARD.as_posix()} ({res['html_bytes'] / 1e6:.2f} MB, limit "
            f"{MAX_HTML_MB} MB)",
            f"period {res['period_long']}: {res['n_days']:,} complete days ({res['n_inc_days']} possibly incomplete "
            f"days left out); population rule: 2010 Census before {POP_2020_FROM_YEAR}, 2020 Census from "
            f"{POP_2020_FROM_YEAR}")
    rep.table(res["waterfall"], title="Rows from the clean table to the district layer")
    ru = res["recent_unresolved"]
    rep.add("", f"all {len(d)} districts: {c['n']:,} complaints, {fmt_rate(c['rate'])} per 1,000 residents a year; "
                f"median time to close {fmt_days(c['median_days'])} (n {c['n_closed']:,}, instant closures excluded); "
                f"unresolved {c['pct_unresolved']:.1f}%",
            f"unresolved, {ru['days']} most recent complete days: {_pct1(ru['pct_recent'])} of {ru['n_recent']:,}; "
            f"earlier complete days: {_pct1(ru['pct_before'])} of {ru['n_before']:,}",
            f"districts with < {MIN_N} complaints (rate flagged): {int(d.low_n.sum())}; with < {MIN_N} closed "
            f"(median hidden): {int((d.n_closed < MIN_N).sum())}")
    show = d.sort_values("rate", ascending=False)[["rank", "label", "n", "rate", "n_closed", "median_days",
                                                   "adj_pct", "pct_unresolved"]]
    comm = d[d.commuter]
    if len(comm):
        rep.add(f"{COMMUTER_LABEL}s (shaded, not ranked; rank <NA> below): " + "; ".join(
            f"{r.label} {fmt_rate(r.rate)}" for r in comm.itertuples()) + f"; ranks run 1-{int(d.n_ranked.iloc[0])}")
    rep.table(show, max_rows=10, title="Highest rates (popup values)")
    rep.table(show.tail(5), title="Lowest rates")
    ranked = d[~d.commuter]
    th = pd.DataFrame([{"layer": t, "complaints": int(d[f"{theme_col(i)}_n"].sum()),
                        "all_districts_rate": c[f"{theme_col(i)}_rate"],
                        "highest_ranked": ranked.loc[ranked[f"{theme_col(i)}_rate"].idxmax(), "label"],
                        "highest_rate": ranked[f"{theme_col(i)}_rate"].max()} for i, t in enumerate(res["themes"])])
    rep.table(th, title=f"Category shadings (the {len(res['themes'])} largest categories)")
    rep.add("", f"points: {len(p['window_days'])} most recent complete days ({res['window_label']}, "
                f"{p['n_skipped']} incomplete days skipped; data end {fmt_date(p['data_end'])}, "
                f"{p['n_days_after']} later days not complete); {p['n_window']:,} complaints, {p['no_coords']:,} "
                f"without coordinates; sampled {p['sampled']:,} of {p['eligible']:,} (cap {p['cap']:,}, seeded hash "
                f"{p['seed']}); still open {p['n_open']:,}",
            f"point data {p['point_bytes'] / 1e6:.2f} MB ({p['point_bytes'] / max(p['sampled'], 1):.0f} bytes per "
            f"point); outlines simplified at {SIMPLIFY_DEG} degrees: {v0:,} -> {v1:,} vertices; "
            f"{1 + len(res['themes'])} shadings + parks layer")
    rep.emit()


def _write_answer(cfg, res):
    d, p, x, c = res["districts"], res["point_info"], _excluded(res), res["city"]
    themes = res["themes"]
    ranked, comm = d[~d.commuter].sort_values("rank"), d[d.commuter].sort_values("rate", ascending=False)
    n_ranked = len(ranked)
    top, low = ranked.iloc[0], ranked.iloc[-1]
    tt = res["top_types"][res["top_types"].boro_cd == top.boro_cd]
    yrs = list(res["days_by_year"].index)
    area = f"all {len(d)} districts"
    hi_themes = "; ".join(f"{t}: {ranked.loc[ranked[f'{theme_col(i)}_rate'].idxmax(), 'label']} "
                          f"({fmt_rate(ranked[f'{theme_col(i)}_rate'].max())} per 1,000 a year; {area}: "
                          f"{fmt_rate(c[f'{theme_col(i)}_rate'])})" for i, t in enumerate(themes))
    one = len(comm) == 1
    commuters = (" " + " and ".join(f"{r.label}, at {fmt_rate(r.rate)}," for r in comm.itertuples())
                 + f" {'is a' if one else 'are'} {COMMUTER_LABEL}{'' if one else 's'} (many workers and visitors, few "
                 f"residents): shaded at {'its' if one else 'their'} rate but not ranked, as in Q7, so ranks run "
                 f"from 1 to {n_ranked}." if len(comm) else "")
    adj_txt = (f"; median time to close {fmt_days(top.median_days)} ({area}: {fmt_days(c['median_days'])}), "
               f"{fmt_pct(top.adj_pct)} vs. the citywide typical time for the same complaint types and years"
               if top.n_closed >= MIN_N else f"; fewer than {MIN_N} closed complaints, so no median is shown")
    if max(yrs) < POP_2020_FROM_YEAR:
        applies = "all days here are before it, so only the 2010 count is used"
    elif min(yrs) >= POP_2020_FROM_YEAR:
        applies = "all days here are after it, so only the 2020 count is used"
    else:
        applies = "both counts are used here"
    pop = (f"the 2010 Census population for days before {POP_2020_FROM_YEAR} and the 2020 Census population from "
           f"{POP_2020_FROM_YEAR} on (the nearer census, and the same rule as Q7; {applies})")
    outside = (f", {x['outside']:,} dated outside the files' period (`out_of_file_order` rows on dates the files do "
               "not cover)" if x["outside"] else "")
    share = 100 * p["sampled"] / p["eligible"] if p["eligible"] else 0
    later = (f" The data run to {fmt_date(p['data_end'])}, but the {p['n_days_after']} days after "
             f"{fmt_date(max(p['window_days']))} are partly missing from the files (`possibly_incomplete_day`), so the "
             "window ends there." if p["n_days_after"] else "")
    table = ["| Rank | District | Complaints | Per 1,000 residents a year | Median time to close | vs. typical | "
             "Unresolved |", "|---|---|---|---|---|---|---|"]
    head = ranked.head(5)
    shown = pd.concat([head, comm[comm.rate >= head.rate.min()]]).sort_values("rate", ascending=False)
    for r in shown.itertuples():
        med = fmt_days(r.median_days) if r.n_closed >= MIN_N else "n/a"
        adj = fmt_pct(r.adj_pct) if r.n_closed >= MIN_N else "n/a"
        rank = f"not ranked ({COMMUTER_LABEL})" if r.commuter else str(int(r.rank))
        table.append(f"| {rank} | {r.label} | {r.n:,} | {fmt_rate(r.rate)} | {med} | {adj} | "
                     f"{r.pct_unresolved:.1f}% |")
    lines = [
        "# Q8. Neighborhood dashboard",
        "",
        f"*Computed by `nyc311/q08_dashboard.py` from {x['mapped']:,} complaints, {res['period_long']}.*",
        "",
        f"- **What it is.** An interactive web map, `outputs/dashboard/{DASHBOARD.name}` "
        f"({res['html_bytes'] / 1e6:.1f} MB, one self-contained file that opens in any browser; the base map tiles "
        "and the Leaflet library load from the internet). It is embedded live below.",
        f"- **District layer.** The {len(d)} community districts shaded by complaints per 1,000 residents a year, in "
        f"{N_CLASSES} quantile classes. The rate divides the {x['mapped']:,} complaints opened on {res['n_days']:,} "
        f"complete days by resident-years, using {pop}. Rate across {area}: {fmt_rate(c['rate'])}. "
        f"Highest{' ranked' if len(comm) else ''}: {top.label} ({fmt_rate(top.rate)}); lowest: {low.label} "
        f"({fmt_rate(low.rate)}).{commuters}",
        f"- **Hover and click.** Hovering shows a district's name, rate and rank"
        + (f" (for a {COMMUTER_LABEL}: \"{COMMUTER_LABEL}, not ranked\")" if len(comm) else "")
        + ". Clicking opens its details: "
        f"complaints, rate and rank, the top {TOP_TYPES} complaint types with counts and shares, median time to close "
        "with the gap to the citywide typical time for the same complaint types and years (Q4's mix adjustment), the "
        "share still unresolved, and a bar for each year's rate with a by-year table.",
        f"- **Example popup: {top.label}.** {int(top.n):,} complaints, {fmt_rate(top.rate)} per 1,000 residents a "
        f"year (rank 1 of {n_ranked})" + adj_txt + f"; {top.pct_unresolved:.1f}% still unresolved ({area}: "
        f"{c['pct_unresolved']:.1f}%). Top types: " + "; ".join(
            f"{r.canonical_type} {r.n:,} ({r.pct:.0f}%)" for r in tt.itertuples()) + ".",
        f"- **Other shadings.** Radio buttons switch the shading to the {len(themes)} largest complaint categories "
        f"({', '.join(themes)}); the legend follows the selected layer. Highest district for each: {hi_themes}.",
        f"- **Zoom in for individual complaints.** A clustered layer shows complaints from the "
        f"{len(p['window_days'])} most recent complete days ({res['window_label']}"
        + (f"; {p['n_skipped']} incomplete days in between are skipped" if p["n_skipped"] else "")
        + f"): a fixed-seed random sample of {p['sampled']:,} of the {p['eligible']:,} complaints with a location "
        f"({share:.0f}%; cap {POINT_CAP:,}).{later} Each dot opens the complaint type, descriptor, agency, opened "
        f"and closed dates, status, address, ZIP and request number; coral dots ({p['n_open']:,}) are still open. Cluster "
        f"numbers count sampled complaints, not all complaints. From zoom level {FADE[0][0]} the district shading "
        "fades so streets and dots stay readable.",
        f"- **Left out (by flag).** {x['test']:,} test records (`is_test`), {x['dup']:,} duplicate keys "
        f"(`dup_key`), {x['info']:,} information requests (`is_info_request`), {x['invalid']:,} invalid entries "
        f"(category `{rules.INVALID_CATEGORY}`), {x['nodate']:,} with no opening date "
        f"and {x['inc']:,} complaints on {res['n_inc_days']} days partly missing from the files "
        f"(`possibly_incomplete_day`){outside}. Of the {x['complete']:,} complaints left, {x['special']:,} in parks "
        f"and airports (`special_area_cd`, shown in gray without a rate) and {x['nocd']:,} with no community "
        f"district are not in the shading; {p['no_coords']:,} complaints in the point window have no coordinates.",
        f"- **Reading it carefully.** Rates are per resident, so districts with many workers or visitors and few "
        f"residents look high"
        + (f" ({', '.join(comm.label)} {'is' if one else 'are'} marked as {'a ' if one else ''}{COMMUTER_LABEL}"
           f"{'' if one else 's'} and left out of the ranks; others with many visitors are not)" if len(comm) else "")
        + f"; {unresolved_caveat(res['recent_unresolved'])}; the totals labeled \"{area}\" "
        "leave out parks, airports and unlocated complaints; the comparisons describe where complaints are filed, "
        "not why. District outlines are DCP's current "
        f"boundaries, simplified to about 10 m ({res['vertices'][0]:,} to {res['vertices'][1]:,} vertices) to keep "
        "the file small.",
    ]
    if res["n_days"] < 365:
        lines.append(f"- **Short period.** The data cover only {res['n_days']} complete days, less than one year, so "
                     "the rates reflect one season (heat complaints peak in winter, noise in summer) and the by-year "
                     "view has a single bar; rerun on the full data for annual patterns.")
    lines += [
        "",
        "## How staff use it",
        "",
        "1. Open the file in a browser; the shading starts on all complaints.",
        "1. Hover over a district to compare its rate with the city; click it for the details listed above.",
        "1. Use the radio buttons at bottom right to shade by a complaint category; untick layers to hide them.",
        "1. Zoom in (scroll or the + button); clusters split until single complaints appear; click a dot for its record.",
        "",
        f"Highest rates on the map ({res['period_label']}):",
        "",
        *table,
        "",
    ]
    path = cfg.out_dir / "answers" / "q08.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
