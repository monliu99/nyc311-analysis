"""Q7: where complaints concentrate. Complaints per 1,000 residents a year by community district, the category
mix behind the gaps, and the link to median household income.

Complaints are rules.COMPLAINTS_SQL rows (default analysis set minus information requests and "Invalid entries")
opened on complete days (a date with in-order rows and no possibly_incomplete_day flag): the same set as Q8 and
Q9. Rates divide complaints by resident-years: 2010 Census population for days in 2010-2014 and 2020 Census
population for days from 2015 on, summed over the complete days in the data. Parks/airports (special_area_cd)
and complaints with no district are excluded and reported. Commuter districts (COMMUTER_CDS) are mapped and
listed, but a per-resident rate is not comparable there, so rankings, top/bottom groups and income comparisons
leave them out, and the spread is given with and without them. The choropleth is drawn with matplotlib patches
from the DCP GeoJSON (no geopandas). The three insights in the figure and in outputs/answers/q07.md are f-strings
over computed values; the same code runs on one month or twelve years.
"""
from __future__ import annotations

import re
import textwrap

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text

from nyc311 import geo, rules
from nyc311 import nytplot as nyt
from nyc311.clean import clean_glob
from nyc311.report import Report

STAGE = "q07_geography"
DEMO_CSV = geo.DISTRICTS_GEOJSON.parent / "cd_demographics.csv"
HOUSING = ("Heat & hot water", "Housing conditions")
POP_2020_FROM_YEAR = 2015   # days before 2015 use population_2010; days from 2015 use population_2020
DAYS_PER_YEAR = 365.25
TOP_K = 10                  # "top" and "bottom" district groups in the insights
N_CLASSES = 5               # map classes: quintiles of the 59 districts
MIN_N = 100                 # districts with fewer complaints are hatched on the map and flagged in tables
INCOME_GROUPS = 5           # income fifths of PUMAs
TOP_TYPES = 3               # complaint types named for a commuter district
HATCH_GRAY = "#a6a6a6"
MAP_LAT = 40.7              # aspect correction for lon/lat
# Districts whose complaints come largely from people who work or visit there: Manhattan 5 (Midtown Business
# District) has about 60,000 residents (2010 and 2020 Census) but is the city's main office district, so
# complaints per resident are not comparable with residential districts. A design choice from outside the 311
# data (like the population counts), not a result; its numbers are still computed and shown.
COMMUTER_CDS = (105,)
COMMUTER_LABEL = "commuter district"
# Exclusion steps after the default analysis set, in order: (label, column of the grouped table in district_counts)
WATERFALL = [
    ("information requests (is_info_request)", "info"),
    (f"category '{rules.INVALID_CATEGORY}'", "invalid"),
    ("no valid created date", "nodate"),
    ("dates with only out_of_file_order rows (outside the files' period)", "outside"),
    ("possibly_incomplete_day", "inc"),
    ("park or airport (special_area_cd)", "special"),
    ("no community district", "no_cd"),
]
MAPPED = "mapped to a community district"


# ------------------------------------------------------------------ pure functions
def load_demographics(path=DEMO_CSV):
    return pd.read_csv(path, dtype={"puma_2010": str, "puma_shared_with": str})


def person_years(demo, days_2010, days_2020):
    """Resident-years per district: population_2010 for days before POP_2020_FROM_YEAR, population_2020 after."""
    d = demo.set_index("boro_cd")
    return (d.population_2010 * days_2010 + d.population_2020 * days_2020) / DAYS_PER_YEAR


def district_table(counts, demo, py, years=None):
    """One row per district: complaints, housing complaints, rates per 1,000 residents a year, top category.
    years (length of the period in years) gives `residents`, the average population over the period."""
    tot = counts.groupby("boro_cd").n.sum()
    hous = counts[counts.category.isin(HOUSING)].groupby("boro_cd").n.sum()
    top = counts.sort_values("n", ascending=False).drop_duplicates("boro_cd").set_index("boro_cd")
    keep = ["borough", "cd_number", "cd_name", "median_hh_income", "puma_2010"]
    d = demo.set_index("boro_cd")[[c for c in keep if c in demo.columns]].copy()
    d["n"] = tot.reindex(d.index).fillna(0).astype(int)
    d["n_housing"] = hous.reindex(d.index).fillna(0).astype(int)
    d["person_years"] = py.reindex(d.index)
    d["residents"] = d.person_years / years if years else np.nan
    d["rate"] = 1000 * d.n / d.person_years
    d["housing_rate"] = 1000 * d.n_housing / d.person_years
    d["other_rate"] = d.rate - d.housing_rate
    d["housing_share_pct"] = 100 * d.n_housing / d.n.where(d.n > 0)
    d["top_category"] = top.category.reindex(d.index)
    d["top_category_pct"] = 100 * top.n.reindex(d.index) / d.n.where(d.n > 0)
    d["low_n"] = d.n < MIN_N
    d["commuter"] = d.index.isin(COMMUTER_CDS)
    return d.reset_index()


def gap_share(top, bottom):
    """Percent of the total-rate gap between two groups that housing complaints account for."""
    return 100 * (top["housing_rate"] - bottom["housing_rate"]) / (top["rate"] - bottom["rate"])


def classify(rates, k=N_CLASSES):
    """Quantile classes 0..k-1 (ties broken by order, so every class is filled) and the k+1 class edges."""
    rates = pd.Series(rates)
    k = min(k, len(rates))
    cls = pd.Series(pd.qcut(rates.rank(method="first"), k, labels=False), index=rates.index).astype(int)
    edges = [float(rates[cls == c].min()) for c in range(k)] + [float(rates.max())]
    return cls, edges


def puma_units(d):
    """Pool districts that share one PUMA (and so one income value): 59 districts -> 55 income areas."""
    p = d.groupby("puma_2010").agg(median_hh_income=("median_hh_income", "first"), n=("n", "sum"),
                                   n_housing=("n_housing", "sum"), person_years=("person_years", "sum"),
                                   districts=("boro_cd", lambda s: ", ".join(str(x) for x in sorted(s))))
    p["rate"] = 1000 * p.n / p.person_years
    p["housing_rate"] = 1000 * p.n_housing / p.person_years
    p["other_rate"] = p.rate - p.housing_rate
    return p.reset_index()


def _group_rates(g):
    py = g.person_years.sum()
    return {"rate": 1000 * g.n.sum() / py, "housing_rate": 1000 * g.n_housing.sum() / py,
            "housing_share": 100 * g.n_housing.sum() / g.n.sum()}


def _boro(b):
    return "the Bronx" if b == "Bronx" else b


def _name(r):
    return f"{r.borough} {r.cd_number} ({r.cd_name})"


def _residential(d):
    return d[~d.commuter] if "commuter" in d else d


def insight_facts(d, p, k=TOP_K, counts=None, commuter_types=None):
    """Every number the three insights quote. Rankings and top/bottom groups use districts that are not commuter
    districts; ratio_all is the spread over every district. p: PUMA table (already without commuter districts)."""
    r = _residential(d)
    top, bottom = r.loc[r.rate.idxmax()], r.loc[r.rate.idxmin()]
    topk, botk = r.nlargest(k, "rate"), r.nsmallest(k, "rate")
    gt, gb = _group_rates(topk), _group_rates(botk)
    boro = topk.borough.value_counts()
    commuters = []
    for c in d[d.commuter].itertuples() if "commuter" in d else []:
        commuters.append({"boro_cd": int(c.boro_cd), "name": _name(c), "short": f"{c.borough} {c.cd_number}",
                          "rate": float(c.rate), "residents": float(getattr(c, "residents", np.nan)),
                          "housing_share": float(c.housing_share_pct),
                          "rank": int((d.rate > c.rate).sum()) + 1,
                          "types": list((commuter_types or {}).get(int(c.boro_cd), []))})
    f = {"k": k, "n_districts": len(d), "n_rated": len(r), "top_name": _name(top), "top_rate": top.rate,
         "top_short": f"{top.borough} {top.cd_number}", "top_housing": top.housing_share_pct,
         "bottom_name": _name(bottom), "bottom_rate": bottom.rate, "ratio": top.rate / bottom.rate,
         "ratio_all": d.rate.max() / d.rate.min(), "commuters": commuters,
         "top_pop_share": 100 * topk.person_years.sum() / r.person_years.sum(),
         "top_n_share": 100 * topk.n.sum() / r.n.sum(), "top_boro": boro.index[0], "top_boro_count": int(boro.iloc[0]),
         "rate_top": gt["rate"], "rate_bot": gb["rate"], "h_top": gt["housing_share"], "h_bot": gb["housing_share"],
         "other_top": gt["rate"] - gt["housing_rate"], "other_bot": gb["rate"] - gb["housing_rate"],
         "gap_housing_pct": gap_share(gt, gb), "gap_cats": []}
    if counts is not None:
        c = counts.pivot_table(index="boro_cd", columns="category", values="n", aggfunc="sum", fill_value=0)
        py = d.set_index("boro_cd").person_years
        rate_t = 1000 * c.reindex(topk.boro_cd).sum() / py.reindex(topk.boro_cd).sum()
        rate_b = 1000 * c.reindex(botk.boro_cd).sum() / py.reindex(botk.boro_cd).sum()
        contrib = (100 * (rate_t - rate_b) / (gt["rate"] - gb["rate"])).sort_values(ascending=False)
        f["gap_cats"] = list(contrib.head(3).items())
    p = p.sort_values("median_hh_income").reset_index(drop=True)
    q = pd.qcut(p.median_hh_income.rank(method="first"), INCOME_GROUPS, labels=False)
    lo, hi = p[q == 0], p[q == INCOME_GROUPS - 1]
    f.update({"n_units": len(p), "inc_low_max": lo.median_hh_income.max(), "inc_high_min": hi.median_hh_income.min(),
              "inc_low_housing": 1000 * lo.n_housing.sum() / lo.person_years.sum(),
              "inc_high_housing": 1000 * hi.n_housing.sum() / hi.person_years.sum(),
              "inc_low_other": 1000 * (lo.n - lo.n_housing).sum() / lo.person_years.sum(),
              "inc_high_other": 1000 * (hi.n - hi.n_housing).sum() / hi.person_years.sum(),
              "rho_housing": p.median_hh_income.corr(p.housing_rate, method="spearman"),
              "rho_other": p.median_hh_income.corr(p.other_rate, method="spearman")})
    return f


def _types_text(types):
    return ", ".join(f"{t} {pct:.0f}%" for t, pct in types[:TOP_TYPES])


def commuter_sentence(c, f):
    """One sentence on a commuter district: its rate, residents, top complaint types, and the spread with it."""
    types = _types_text(c["types"])
    who = (f"{c['residents'] / 1000:,.1f}k residents" if np.isfinite(c["residents"]) else "few residents") + (
        f"; top complaints {types}" if types else "")
    extreme = abs(f["ratio_all"] - f["ratio"]) > 1e-9
    return (f"{c['name']} is a {COMMUTER_LABEL} ({who}): its {c['rate']:,.0f} complaints per 1,000 residents "
            + (f"would make the spread {f['ratio_all']:.1f}-fold, but are" if extreme else "are")
            + " not comparable with where people live, so it is left out of the rankings.")


def insights(f):
    """The three insights for the Mayor, as text built only from insight_facts()."""
    one = (f"{f['top_name']} files {f['top_rate']:,.0f} complaints per 1,000 residents a year, {f['ratio']:.1f}x "
           f"{f['bottom_name']}, the lowest at {f['bottom_rate']:,.0f}. ")
    one += "".join(commuter_sentence(c, f) + " " for c in f["commuters"])
    one += (f"The {f['k']} highest-rate districts hold {f['top_pop_share']:.0f}% of residents but "
            f"{f['top_n_share']:.0f}% of complaints; {f['top_boro_count']} of the {f['k']} are in "
            f"{_boro(f['top_boro'])}.")
    two = (f"Housing accounts for {f['gap_housing_pct']:.0f}% of the gap in rates between the {f['k']} highest- and "
           f"{f['k']} lowest-rate districts ({f['rate_top']:,.0f} vs {f['rate_bot']:,.0f} per 1,000 a year): heat, hot "
           f"water and housing-condition complaints are {f['h_top']:.0f}% of complaints in the top {f['k']} vs "
           f"{f['h_bot']:.0f}% in the bottom {f['k']}. All other complaints: {f['other_top']:,.0f} vs "
           f"{f['other_bot']:,.0f} per 1,000"
           + ("; other complaints narrow the gap." if f["gap_housing_pct"] > 100 else "."))
    if f["gap_cats"]:
        two += " Shares of the gap by category: " + ", ".join(f"{c} {v:.0f}%" for c, v in f["gap_cats"]) + "."
    three = (f"Income. In the poorest fifth of the {f['n_units']} income areas (median household income up to "
             f"${f['inc_low_max'] / 1000:,.0f}k), residents file {f['inc_low_housing']:,.0f} housing complaints per "
             f"1,000 a year vs {f['inc_high_housing']:,.0f} in the richest fifth (${f['inc_high_min'] / 1000:,.0f}k "
             f"and up). Other complaints: {f['inc_low_other']:,.0f} vs {f['inc_high_other']:,.0f}. Rank correlation "
             f"with income: housing {f['rho_housing']:+.2f}, other {f['rho_other']:+.2f}.")
    return [one, two, three]


def map_title(f):
    extreme = [c for c in f["commuters"] if abs(f["ratio_all"] - f["ratio"]) > 1e-9]
    if extreme:
        return (f"311 complaints per resident vary {f['ratio']:.1f}-fold across {f['n_rated']} community districts "
                f"({f['ratio_all']:.1f}-fold counting {COMMUTER_LABEL} {extreme[0]['short']})")
    return f"311 complaints per resident vary {f['ratio_all']:.1f}-fold across {f['n_districts']} community districts"


# ------------------------------------------------------------------ data
def analysis_days(con, src):
    """Complete days: dates with at least one in-order row, minus possibly_incomplete_day."""
    days = con.sql(f"""select created::date as d, bool_or(not out_of_file_order) as covered,
                       bool_or(possibly_incomplete_day) as inc
                       from {src} where {rules.BASE_SQL} and created is not null group by 1""").df()
    days["d"] = pd.to_datetime(days.d)
    return days[days.covered & ~days.inc.fillna(False).astype(bool)].d.sort_values().reset_index(drop=True)


def days_table(con, days, name):
    """Temp table `name` (one column d, DATE) holding the analysis days."""
    con.register(f"{name}_df", pd.DataFrame({"d": pd.to_datetime(pd.Series(days))}))
    con.execute(f"create or replace temp table {name} as select distinct cast(d as date) as d from {name}_df")
    con.unregister(f"{name}_df")
    return name


def district_counts(con, src):
    """Complaints by district and category on complete days, the exclusion waterfall (WATERFALL) from the default
    analysis set, and the complaints with no district (by reported borough)."""
    raw = con.sql(f"""
        with b as (select created::date as d, coalesce(is_info_request, false) as info,
                          coalesce(category, '') = '{rules.INVALID_CATEGORY}' as invalid,
                          coalesce(possibly_incomplete_day, false) as inc, boro_cd,
                          special_area_cd is not null as special, borough, coalesce(category, '(unmapped)') as category
                   from {src} where {rules.BASE_SQL}),
        cov as (select distinct created::date as cd from {src}
                where {rules.BASE_SQL} and not out_of_file_order and created is not null)
        select b.d is null as nodate, cov.cd is null as outside, info, invalid, inc, special, boro_cd is null as no_cd,
               boro_cd, borough, category, count(*)::bigint as n
        from b left join cov on b.d = cov.cd group by all""").df()
    default = int(raw.n.sum())
    steps, keep, no_district = [("default analysis set", default)], raw, None
    for label, col in WATERFALL:
        mask = keep[col].astype(bool)
        if col == "no_cd":
            no_district = keep[mask]
        steps.append((label, int(keep.n[mask].sum())))
        keep = keep[~mask]
    steps.append((MAPPED, int(keep.n.sum())))
    ex = pd.DataFrame(steps, columns=["step", "rows"])
    ex["pct"] = 100 * ex.rows / max(default, 1)
    counts = keep.assign(boro_cd=keep.boro_cd.astype(int)).groupby(["boro_cd", "category"], as_index=False).n.sum()
    return counts, ex, no_district


def commuter_types(con, src, days_rel, cds=COMMUTER_CDS, top=TOP_TYPES):
    """{boro_cd: [(canonical type, % of the district's complaints), ...]} for the commuter districts."""
    if not cds:
        return {}
    t = con.sql(f"""select boro_cd, coalesce(canonical_type, '(unmapped)') as t, count(*)::bigint as n from {src}
                    where {rules.COMPLAINTS_SQL} and boro_cd in ({', '.join(str(int(c)) for c in cds)})
                    and created::date in (select d from {days_rel}) group by all""").df()
    out = {}
    for cd, g in t.groupby("boro_cd"):
        g = g.sort_values(["n", "t"], ascending=[False, True])
        out[int(cd)] = [(r.t, 100 * r.n / g.n.sum()) for r in g.head(top).itertuples()]
    return out


def unknown_by_agency(con, src, days_rel):
    return con.sql(f"""select agency, count(*)::bigint as n from {src}
                       where {rules.COMPLAINTS_SQL} and created::date in (select d from {days_rel})
                       and boro_cd is null and special_area_cd is null
                       group by 1 order by n desc""").df()


def borough_table(d, no_district, demo):
    names = demo.assign(code=demo.boro_cd // 100).groupby("code").borough.first()
    up = {k: names[v] for k, v in rules.BORO_CODE.items() if v in names.index}
    b = d.groupby("borough").agg(districts=("boro_cd", "size"), n=("n", "sum"), n_housing=("n_housing", "sum"),
                                 person_years=("person_years", "sum"))
    b["rate"] = 1000 * b.n / b.person_years
    b["housing_rate"] = 1000 * b.n_housing / b.person_years
    b["housing_share_pct"] = 100 * b.n_housing / b.n
    nd = no_district.assign(b=no_district.borough.map(up)).groupby("b").n.sum()
    b["n_with_borough_no_district"] = nd.reindex(b.index).fillna(0).astype(int)
    b["rate_incl_no_district"] = 1000 * (b.n + b.n_with_borough_no_district) / b.person_years
    r = _residential(d)
    b["highest_district"] = r.loc[r.groupby("borough").rate.idxmax()].set_index("borough").cd_name
    return b.sort_values("rate", ascending=False).reset_index()


# ------------------------------------------------------------------ labels
LEADER_COLOR = "#222222"   # leader lines and anchor dots: dark enough to read at the report's column width
LEADER_LW = 1.0
LABEL_RADII = (14, 22, 32, 44, 58, 74, 92, 112, 134)   # candidate distances from the anchor, in points
LABEL_ANGLES = np.radians(np.arange(0, 360, 15))
LAND_COST = 6.0            # a label fully over drawn districts costs this much more than one in open space...
LENGTH_COST = 1 / 60       # ...and each point of leader line this much, so labels move to open space if it is near
CLASH_COST = 200           # overlapping a label or legend, or covering another anchor
CROSS_COST = 100           # leaders crossing each other or another label, or passing next to another anchor


def interior_point(outer, holes=(), n=24):
    """A point deep inside a polygon (close to the one farthest from its edges, measured on the ground), to anchor a
    label: the vertex mean of a hooked or concave district can fall outside it or on its edge."""
    v = np.asarray(outer.vertices, dtype=float)
    k = np.array([np.cos(np.radians(MAP_LAT)), 1.0])   # degrees of longitude are shorter on the ground
    rings = [v] + [np.asarray(h.vertices, dtype=float) for h in holes]
    a = np.concatenate([r[:-1] for r in rings]) * k
    ab = np.concatenate([r[1:] for r in rings]) * k - a
    ll = np.maximum((ab ** 2).sum(1), 1e-30)

    def clearance(p):   # distance to the nearest edge (ground units), -inf outside the polygon
        inside = outer.contains_points(p)
        for h in holes:
            inside &= ~h.contains_points(p)
        out = np.full(len(p), -np.inf)
        idx = np.flatnonzero(inside)
        for s in range(0, len(idx), 128):
            q = p[idx[s:s + 128]] * k
            t = np.clip(((q[:, None] - a[None]) * ab[None]).sum(2) / ll[None], 0, 1)
            out[idx[s:s + 128]] = np.sqrt(((q[:, None] - a[None] - t[..., None] * ab[None]) ** 2).sum(2)).min(1)
        return out

    lo, hi = v.min(0), v.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n + 1), np.linspace(lo[1], hi[1], n + 1))
    pts, step = np.column_stack([gx.ravel(), gy.ravel()]), (hi - lo) / n
    best, best_c = None, -np.inf
    for _ in range(3):   # a coarse grid, then two finer ones around the best point so far
        c = clearance(pts)
        if c.max() > best_c:
            best, best_c = pts[int(np.argmax(c))], c.max()
        if best is None:
            break
        gx, gy = np.meshgrid(np.linspace(best[0] - step[0], best[0] + step[0], 9),
                             np.linspace(best[1] - step[1], best[1] + step[1], 9))
        pts, step = np.column_stack([gx.ravel(), gy.ravel()]), step / 4
    if best is None:
        best = v[:-1].mean(axis=0)
    return float(best[0]), float(best[1])


def _cross(p1, p2, q1, q2):
    """Whether segments p1-p2 and q1-q2 cross."""
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return orient(p1, p2, q1) * orient(p1, p2, q2) < 0 and orient(q1, q2, p1) * orient(q1, q2, p2) < 0


def _hits_box(p, q, box):
    """Whether segment p-q touches the box."""
    if box.contains(*p) or box.contains(*q):
        return True
    c = [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]
    return any(_cross(p, q, c[i], c[(i + 1) % 4]) for i in range(4))


def _near_segment(p, q, r, tol):
    p, q, r = (np.asarray(z, dtype=float) for z in (p, q, r))
    d = q - p
    t = np.clip(np.dot(r - p, d) / max(np.dot(d, d), 1e-12), 0, 1)
    return np.hypot(*(p + t * d - r)) < tol


def place_labels(fig, ax, items, avoid=(), fontsize=8.5, **kw):
    """Map labels with leader lines. items: (x, y, text) in data coordinates, in priority order. Every anchor gets a
    dot, and its label sits off to the side, joined to the dot by a dark leader line. Candidate spots ring each anchor
    (LABEL_RADII x LABEL_ANGLES); the chosen ones avoid each other, the `avoid` artists, every anchor dot, crossing
    leaders and the axes edge, and prefer open (undrawn) space to drawn districts, then a short leader. Chosen
    greedily in priority order, then revised a few times given the others. Returns the labels' display boxes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.transforms import Bbox

    canvas = fig.canvas if hasattr(fig.canvas, "get_renderer") else FigureCanvasAgg(fig)
    canvas.draw()   # resolves the constrained layout, so display coordinates are final
    renderer = canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    pt = fig.dpi / 72
    parts = []   # an inset axes (a color key) counts as its box, tick labels and title, not their joint bounding box
    for a in avoid:
        parts += [a, a.xaxis, a.yaxis, a.title] if isinstance(a, plt.Axes) else [a]
    taken = [a.get_window_extent(renderer) if isinstance(a, plt.Axes) else a.get_tightbbox(renderer) for a in parts]
    taken = [b.padded(2 * pt) for b in taken if b is not None and b.width > 0 and b.height > 0]
    rgba = np.asarray(canvas.buffer_rgba())
    drawn = np.pad((rgba[..., :3] < 235).any(axis=2), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h_px, w_px = rgba.shape[:2]

    def land(b):   # share of the box's pixels already drawn on (districts, hatching)
        x0, x1 = int(np.clip(b.x0, 0, w_px)), int(np.clip(b.x1, 0, w_px))
        y0, y1 = int(np.clip(h_px - b.y1, 0, h_px)), int(np.clip(h_px - b.y0, 0, h_px))
        area = max((x1 - x0) * (y1 - y0), 1)
        return (drawn[y1, x1] - drawn[y0, x1] - drawn[y1, x0] + drawn[y0, x0]) / area

    style = {"fontsize": fontsize, "fontfamily": nyt.SANS, "fontweight": "bold", "color": "#111111",
             "path_effects": [pe.withStroke(linewidth=2.5, foreground="white")], **kw}
    anchors = [tuple(ax.transData.transform((x, y))) for x, y, _ in items]
    cands = []
    for i, (x, y, text) in enumerate(items):
        t = ax.text(x, y, text, **style)
        bb = t.get_window_extent(renderer)
        t.remove()
        w, h = bb.width, bb.height
        others = [a for j, a in enumerate(anchors) if j != i]
        lst = []
        for r in LABEL_RADII:
            for th in LABEL_ANGLES:
                c, s = np.cos(th), np.sin(th)
                px, py = anchors[i][0] + r * pt * c, anchors[i][1] + r * pt * s
                ha = "left" if c > 0.38 else "right" if c < -0.38 else "center"
                va = "bottom" if s > 0.38 else "top" if s < -0.38 else "center"
                x0 = px if ha == "left" else px - w if ha == "right" else px - w / 2
                y0 = py if va == "bottom" else py - h if va == "top" else py - h / 2
                box = Bbox.from_bounds(x0, y0, w, h)
                pad = box.padded(4 * pt)
                center = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
                inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
                cost = (1000 * (not inside) + CLASH_COST * sum(box.padded(2 * pt).overlaps(b) for b in taken)
                        + CLASH_COST * sum(pad.contains(*a) for a in others)
                        + CROSS_COST * sum(_near_segment(center, anchors[i], a, 4 * pt) for a in others)
                        + LAND_COST * land(box) + LENGTH_COST * r)
                lst.append((cost, box, center, r * c, r * s, ha, va))
        cands.append(sorted(lst, key=lambda z: z[0]))

    def pair_cost(i, cand, j, other):
        box, center = cand[1], cand[2]
        box2, center2 = other[1], other[2]
        return (CLASH_COST * box.padded(2 * pt).overlaps(box2)
                + CROSS_COST * (_cross(center, anchors[i], center2, anchors[j])
                                + _hits_box(center, anchors[i], box2) + _hits_box(center2, anchors[j], box)))

    def best_for(i, chosen):
        return min(cands[i], key=lambda cd: cd[0] + sum(pair_cost(i, cd, j, o) for j, o in chosen.items() if j != i))

    chosen = {}
    for i in range(len(items)):
        chosen[i] = best_for(i, chosen)
    for _ in range(3):
        for i in range(len(items)):
            chosen[i] = best_for(i, chosen)

    boxes = []
    for i, (x, y, text) in enumerate(items):
        _, _, _, dx, dy, ha, va = chosen[i]
        t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, zorder=6,
                        arrowprops={"arrowstyle": "-", "color": LEADER_COLOR, "lw": LEADER_LW, "shrinkA": 0,
                                    "shrinkB": 0}, **style)
        ax.plot([x], [y], ls="none", marker="o", ms=3.6, mfc=LEADER_COLOR, mec="white", mew=0.7, zorder=7)
        t.get_window_extent(renderer)              # places the text at its offset
        boxes.append(Text.get_window_extent(t, renderer))   # the text only (an Annotation's extent includes its leader)
    return boxes


# ------------------------------------------------------------------ figures
def ramp(k, hue=nyt.SAGE):
    rgb = np.array(plt.matplotlib.colors.to_rgb(hue.line))
    darkest = rgb * 0.72
    shades = np.linspace(0.12, 1.0, k)
    return [plt.matplotlib.colors.to_hex(1 - a * (1 - darkest)) for a in shades]


def _area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def polygon_path(outer, holes):
    """One compound path per polygon part; holes are wound opposite to the outer ring so they stay empty."""
    rings = [outer.vertices]
    for h in holes:
        v = h.vertices
        rings.append(v[::-1] if np.sign(_area(v)) == np.sign(_area(outer.vertices)) else v)
    verts = np.concatenate(rings)
    codes = np.concatenate([[MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 1) for r in rings])
    return MplPath(verts, codes)


def _fig_text(s):
    return s.replace("$", r"\$")


def _label_picks(d):
    """Districts labeled on the map: the three highest rates, the highest non-commuter one, the lowest, and every
    commuter district."""
    ranked = d.sort_values("rate", ascending=False)
    picks = list(ranked.boro_cd.head(3))
    r = _residential(d)
    for b in [r.loc[r.rate.idxmax()].boro_cd, ranked.boro_cd.iloc[-1], *d.boro_cd[d.commuter]]:
        if b not in picks:
            picks.append(b)
    return picks


def fig_map(d, texts, path, note, title):
    nyt.use()
    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax, tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    cls, edges = classify(d.rate)
    colors = ramp(len(edges) - 1)
    fill = dict(zip(d.boro_cd, [colors[c] for c in cls]))
    low = set(d.boro_cd[d.low_n])
    commuter = set(d.boro_cd[d.commuter])
    parts = {}
    for code, outer, holes, lo, hi in geo.load_polygons():
        shape = polygon_path(outer, holes)
        ax.add_patch(PathPatch(shape, facecolor=fill.get(code, "white"), edgecolor="white", lw=0.5))
        hatch = "...." if code not in fill else "////" if code in low else None
        if hatch:  # overlay so the hatch can be gray while district borders stay white
            ax.add_patch(PathPatch(shape, facecolor="none", edgecolor=HATCH_GRAY, lw=0, hatch=hatch))
        if code in commuter:
            ax.add_patch(PathPatch(shape, facecolor="none", edgecolor="#111111", lw=1.0, zorder=3))
        area = abs(_area(outer.vertices))
        if area > parts.get(code, (0, None, None))[0]:
            parts[code] = (area, outer, holes)   # the largest part carries the label
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(MAP_LAT)))
    ax.axis("off")
    key = ax.inset_axes([0.02, 0.875, 0.31, 0.028])   # narrow, so labels for the north can use the open space
    for i, c in enumerate(colors):
        key.add_patch(Rectangle((i, 0), 1, 1, color=c))
    key.set_xlim(0, len(colors))
    key.set_ylim(0, 1)
    key.set_xticks(range(len(edges)))
    key.set_xticklabels([f"{e:,.0f}" for e in edges], fontsize=8.5)
    key.set_yticks([])
    for s in key.spines.values():
        s.set_visible(False)
    key.grid(False)
    key.set_title("Complaints per 1,000 residents a year\n(district fifths)", loc="left", fontsize=9.5,
                  fontfamily=nyt.SANS)
    swatch = ax.add_patch(Rectangle((0.80, 0.06), 0.035, 0.03, transform=ax.transAxes, facecolor="white",
                                    edgecolor=HATCH_GRAY, lw=0.5, hatch="...."))
    parks = ax.text(0.845, 0.075, "Parks & airports\n(not rated)", transform=ax.transAxes, va="center",
                    fontsize=8.5, color=nyt.NOTE_GRAY)
    tx.axis("off")
    tx.text(0, 1, "Three insights for the Mayor", fontfamily=nyt.SANS, fontweight="bold", fontsize=13, va="top")
    y = 0.93
    for i, t in enumerate(texts, 1):
        tx.text(0, y, str(i), fontfamily=nyt.SANS, fontweight="bold", fontsize=15, color=nyt.CORAL.line, va="top")
        body = textwrap.fill(_fig_text(t), 56)
        tx.text(0.07, y, body, fontsize=10, va="top", linespacing=1.3)
        y -= 0.036 * (body.count("\n") + 1) + 0.03
    nyt.title(fig, title)
    nyt.note(fig, textwrap.fill(note, 175))
    rows = d.set_index("boro_cd")
    items = []
    for b in _label_picks(d):
        r = rows.loc[b]
        tag = f"\n{COMMUTER_LABEL}" if r.commuter else ""
        items.append((*interior_point(*parts[b][1:]), f"{r.borough} {r.cd_number}\n{r.rate:,.0f}{tag}"))
    place_labels(fig, ax, items, avoid=[key, swatch, parks], fontsize=9)
    return nyt.save(fig, path)


def fig_district_rates(d, city_rate, path, period, f):
    nyt.use()
    order = d.groupby("borough").rate.apply(lambda s: 1000 * d.loc[s.index, "n"].sum() /
                                            d.loc[s.index, "person_years"].sum()).sort_values(ascending=False)
    sizes = d.borough.value_counts().reindex(order.index)
    fig, axes = plt.subplots(len(order), 1, figsize=(10, 0.24 * len(d) + 2.2), sharex=True,
                             gridspec_kw={"height_ratios": sizes.to_numpy() + 1.2})
    xmax = d.rate.max() * 1.02
    for ax, (boro, brate) in zip(np.atleast_1d(axes), order.items()):
        s = d[d.borough == boro].sort_values("rate")
        y = np.arange(len(s))
        ax.barh(y, s.housing_rate, color=nyt.CORAL.line, height=0.72)
        ax.barh(y, s.other_rate, left=s.housing_rate, color=nyt.SAGE.band, height=0.72)
        for yi, r in zip(y, s.itertuples()):
            if r.commuter:
                ax.text(r.housing_rate + 0.01 * xmax, yi, f"{COMMUTER_LABEL}: per-resident rate not comparable",
                        va="center", fontsize=8.5, fontfamily=nyt.SANS, color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.cd_number} {textwrap.shorten(r.cd_name, 30, placeholder='...')}"
                            + (" *" if r.commuter else "") for r in s.itertuples()], fontsize=9)
        ax.axvline(city_rate, color=nyt.NOTE_GRAY, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(s) - 0.4)
        nyt.panel_title(ax, f"{boro}: {brate:,.0f} per 1,000", fontsize=10.5)
        nyt.format_x(ax)
    first = np.atleast_1d(axes)[0]
    first.text(0.99, 0.30, "Housing (heat, hot water, conditions)", transform=first.transAxes, ha="right",
               color=nyt.CORAL.line, fontfamily=nyt.SANS, fontweight="bold", fontsize=10)
    first.text(0.99, 0.12, "All other complaints", transform=first.transAxes, ha="right", color=nyt.SAGE.line,
               fontfamily=nyt.SANS, fontweight="bold", fontsize=10)
    first.text(city_rate, 1.0, f" all {len(d)} districts: {city_rate:,.0f}", transform=first.get_xaxis_transform(),
               va="bottom",
               fontsize=8.5, color=nyt.NOTE_GRAY)
    title = (f"Complaints per 1,000 residents a year: {f['top_short']} leads at {f['top_rate']:,.0f}, "
             f"{f['top_housing']:.0f}% of it housing")
    if f["commuters"]:
        title += " (" + ", ".join(f"{COMMUTER_LABEL} {c['short']}, {c['rate']:,.0f}, not ranked"
                                  for c in f["commuters"]) + ")"
    nyt.title(fig, textwrap.fill(title, 100))
    star = (f" * {COMMUTER_LABEL}: many workers and visitors, few residents." if f["commuters"] else "")
    nyt.note(fig, f"Note: {period}; boroughs ordered by rate, districts within each by rate. Dashed line: rate "
                  f"across the {len(d)} districts (complaints with no district left out).{star}")
    return nyt.save(fig, path)


def fig_income(p, f, path, period, excluded):
    nyt.use()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    x = p.median_hh_income / 1000
    for ax, col, h, label, rho in [(axes[0], "housing_rate", nyt.CORAL, "Housing complaints", f["rho_housing"]),
                                   (axes[1], "other_rate", nyt.SAGE, "All other complaints", f["rho_other"])]:
        ax.scatter(x, p[col], color=h.line, s=22, alpha=0.85, linewidths=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.03)
        nyt.format_y(ax, nbins=4)
        nyt.panel_title(ax, f"{label} per 1,000 residents a year (rank correlation with income {rho:+.2f})",
                        fontsize=10.5)
    nyt.format_x(axes[1], prefix="$", suffix="k", nbins=6)
    nyt.title(fig, f"Poorest fifth of areas: {f['inc_low_housing']:,.0f} housing complaints per 1,000 vs "
                   f"{f['inc_high_housing']:,.0f} in the richest")
    left_out = (f" {excluded} ({COMMUTER_LABEL}) left out; its PUMA partner stands alone." if excluded else "")
    nyt.note(fig, textwrap.fill(
        f"Note: each dot is one of {len(p)} PUMAs (districts sharing a PUMA are pooled).{left_out} Median "
        f"household income: ACS 2010-2014, 2014 dollars. Complaints: {period}.", 105))
    return nyt.save(fig, path)


# ------------------------------------------------------------------ run
def run(cfg):
    con = cfg.connect()
    src = clean_glob(cfg)
    demo = load_demographics()
    days = analysis_days(con, src)
    days_rel = days_table(con, days, "q07_days")
    n_2020 = int((days.dt.year >= POP_2020_FROM_YEAR).sum())
    n_2010 = len(days) - n_2020
    py = person_years(demo, n_2010, n_2020)
    counts, ex, no_district = district_counts(con, src)
    d = district_table(counts, demo, py, years=len(days) / DAYS_PER_YEAR)
    boroughs = borough_table(d, no_district, demo)
    pumas = puma_units(_residential(d))
    ctypes = commuter_types(con, src, days_rel)
    facts = insight_facts(d, pumas, counts=counts, commuter_types=ctypes)
    texts = insights(facts)
    unknown = unknown_by_agency(con, src, days_rel)
    period = f"{days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}"
    pops = ", ".join(x for x in [f"2010 Census for {n_2010:,} days before {POP_2020_FROM_YEAR}" if n_2010 else "",
                                 f"2020 Census for {n_2020:,} days from {POP_2020_FROM_YEAR}" if n_2020 else ""] if x)
    exi = ex.set_index("step")
    commuters = ", ".join(c["short"] for c in facts["commuters"])
    note = (f"Note: {int(d.n.sum()):,} complaints with a community district, {period} ({len(days):,} complete days); "
            f"population: {pops}. Not rated: parks and airports "
            f"{exi.loc['park or airport (special_area_cd)', 'pct']:.1f}% and no district "
            f"{exi.loc['no community district', 'pct']:.1f}% of rows; information requests excluded."
            + (f" Outlined: {commuters}, a {COMMUTER_LABEL} (not ranked)." if commuters else "")
            + " Sources: NYC 311; NYC DCP districts and Census counts; ACS 2010-2014 income.")
    city_rate = 1000 * d.n.sum() / d.person_years.sum()

    agg, figs = cfg.out_dir / "agg", cfg.out_dir / "figs"
    agg.mkdir(parents=True, exist_ok=True)
    d.sort_values("rate", ascending=False).to_csv(agg / "q07_districts.csv", index=False)
    boroughs.to_csv(agg / "q07_boroughs.csv", index=False)
    ex.to_csv(agg / "q07_exclusions.csv", index=False)
    pumas.to_csv(agg / "q07_pumas.csv", index=False)
    unknown.to_csv(agg / "q07_no_district_by_agency.csv", index=False)
    fig_map(d, texts, figs / "q07_district_map.png", note, map_title(facts))
    fig_district_rates(d, city_rate, figs / "q07_district_rates.png", period, facts)
    fig_income(pumas, facts, figs / "q07_income.png", period, commuters)

    v = dict(d=d, boroughs=boroughs, ex=ex, pumas=pumas, facts=facts, texts=texts, unknown=unknown, period=period,
             pops=pops, days=days, city_rate=city_rate)
    _report(cfg.out_dir, v)
    _write_answer(cfg.out_dir, v)
    return {"districts": d, "boroughs": boroughs, "exclusions": ex, "pumas": pumas, "facts": facts,
            "insights": texts, "no_district_by_agency": unknown, "days": days, "period": period}


def _report(out_dir, v):
    d, f = v["d"], v["facts"]
    rep = Report(STAGE, out_dir)
    rep.add(f"Q7. Complaints per 1,000 residents a year by community district, {v['period']} "
            f"({len(v['days']):,} complete days)", f"population: {v['pops']}",
            f"rate across the {len(d)} districts: {v['city_rate']:,.1f}",
            f"spread: {f['ratio']:.2f}-fold across {f['n_rated']} districts; {f['ratio_all']:.2f}-fold across all "
            f"{f['n_districts']} (with {COMMUTER_LABEL}s)")
    for c in f["commuters"]:
        rep.add(f"{COMMUTER_LABEL}: {c['name']} rate {c['rate']:,.1f} (would rank {c['rank']} of {len(d)}; not "
                f"ranked), residents {c['residents']:,.0f}, housing {c['housing_share']:.1f}%, top types: "
                f"{_types_text(c['types'])}")
    rep.table(v["ex"].round(2), title="Rows (default set -> mapped to a district)")
    rep.table(v["unknown"], max_rows=6, title="No community district, by agency")
    rep.table(v["boroughs"].drop(columns=["person_years"]).round(1), title="Boroughs")
    cols = ["boro_cd", "borough", "cd_name", "n", "rate", "housing_share_pct", "top_category", "top_category_pct",
            "median_hh_income", "commuter"]
    ranked = d.sort_values("rate", ascending=False)[cols].round(1)
    rep.table(ranked.head(TOP_K), title=f"Top {TOP_K} districts by rate (all districts)")
    rep.table(ranked.tail(5), title="Bottom 5 districts by rate")
    if d.low_n.any():
        rep.add(f"districts with fewer than {MIN_N} complaints (hatched): " + ", ".join(map(str, d.boro_cd[d.low_n])))
    rep.add("", "Three insights (in the map figure and answers/q07.md):")
    for i, t in enumerate(v["texts"], 1):
        rep.add(*textwrap.wrap(f"{i}. {t}", 110, subsequent_indent="   "))
    rep.add("", f"gap contributions (top {TOP_K} vs bottom {TOP_K}, {COMMUTER_LABEL}s left out): " + "; ".join(
        f"{c} {x:.1f}%" for c, x in f["gap_cats"]))
    rep.emit()


def _pipe_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return rows


def _md(text):
    """Column and flag names (snake_case) as `code`, so markdown renderers don't read underscores as italics."""
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: f"`{m.group(0)}`", text)


def _write_answer(out_dir, v):
    d, f, b, exi = v["d"], v["facts"], v["boroughs"], v["ex"].set_index("step")
    ranked = _residential(d).sort_values("rate", ascending=False)
    bt = pd.DataFrame({
        "Borough": b.borough, "Districts": b.districts, "Complaints": [f"{x:,}" for x in b.n],
        "Per 1,000 a year": [f"{x:,.0f}" for x in b.rate], "Housing share": [f"{x:.0f}%" for x in b.housing_share_pct],
        "Borough known, no district": [f"{x:,}" for x in b.n_with_borough_no_district],
        "Highest district": b.highest_district})
    unk = v["unknown"]
    n_all = int(d.n.sum())
    comm = f["commuters"]
    housing_comm = ", ".join(f"{c['housing_share']:.0f}%" for c in comm)
    no_cd = int(exi.loc["no community district", "rows"])
    rate_unlocated = 1000 * (n_all + no_cd) / d.person_years.sum()   # same resident-years, unlocated complaints added
    lines = [
        "# Q7. Complaints by community district",
        "",
        f"*Computed by `nyc311/q07_geography.py` from {n_all:,} complaints, {v['period']}.*",
        "",
        "## Data and method",
        "",
        f"- **Rate** = complaints per 1,000 residents a year: complaints divided by resident-years, using the "
        f"{v['pops']} (rule: 2010 Census for days before {POP_2020_FROM_YEAR}, 2020 Census from "
        f"{POP_2020_FROM_YEAR}). Only complete days count ({len(v['days']):,} days), so partial days don't dilute rates. "
        f"Across the {len(d)} districts: {v['city_rate']:,.0f} per 1,000 a year ({rate_unlocated:,.0f} counting the "
        f"{no_cd:,} complaints with no district).",
        f"- **Rows.** Of the default set's {int(exi.rows.iloc[0]):,} rows, {n_all:,} complaints "
        f"({exi.loc[MAPPED, 'pct']:.1f}%) map to one of the {len(d)} districts. Left out: " +
        "; ".join(f"{_md(k)} {int(r.rows):,} ({r.pct:.1f}%)" for k, r in exi.iloc[1:-1].iterrows()) + ".",
        f"- **No district** is mostly " + ", ".join(f"{r.agency} {r.n:,}" for r in unk.head(3).itertuples()) +
        " (requests with no usable address or coordinates).",
        *([f"- **Commuter district.** " + " ".join(commuter_sentence(c, f) for c in comm) +
           f" Housing (heat, hot water, conditions) is {housing_comm} of its complaints. "
           f"Spread: {f['ratio']:.1f}-fold across the other {f['n_rated']} districts, {f['ratio_all']:.1f}-fold "
           f"across all {f['n_districts']}. It is mapped and counted in its borough and the rate across the "
           f"{f['n_districts']} districts, but left "
           "out of rankings, top/bottom groups and income comparisons."] if comm else []),
        "",
        "![Complaints per 1,000 residents by community district, with three insights](figs/q07_district_map.png)",
        "",
        "## Three insights for the Mayor",
        "",
        *[f"{i}. {t}" for i, t in enumerate(v["texts"], 1)],
        "",
        "## Disparities by borough and district",
        "",
        *_pipe_table(bt),
        "",
        f"- **Highest rates** ({COMMUTER_LABEL}s left out): " + "; ".join(
            f"{r.borough} {r.cd_number} ({r.cd_name}) {r.rate:,.0f}, top category {r.top_category} "
            f"{r.top_category_pct:.0f}%" for r in ranked.head(5).itertuples()) + ".",
        f"- **Lowest rates:** " + "; ".join(
            f"{r.borough} {r.cd_number} ({r.cd_name}) {r.rate:,.0f}, top category {r.top_category} "
            f"{r.top_category_pct:.0f}%" for r in ranked.tail(3).itertuples()) + ".",
        "",
        "![District rates split into housing and other complaints](figs/q07_district_rates.png)",
        "",
        f"- **Income** ({f['n_units']} PUMAs; districts that share one ACS value are pooled"
        + (f"; {', '.join(c['short'] for c in comm)} left out" if comm else "") + "): rank correlation of median "
        f"household income with the housing-complaint rate {f['rho_housing']:+.2f}, with all other complaints "
        f"{f['rho_other']:+.2f}.",
        "",
        "![Complaint rates against median household income](figs/q07_income.png)",
        "",
        "## Limits",
        "",
        "- **311 measures reporting, not need.** Rates mix how often problems occur with how often residents call. "
        "Housing complaints are tenants reporting landlords to HPD, so districts with more renters generate more "
        "of them; a low rate is not proof of good conditions.",
        "- **Denominators.** Census counts (2020 counts carry small differential-privacy noise); ACS income is from "
        "PUMAs that only approximate districts, in 2014 dollars. Boundaries are current, not historical.",
        "- **Workers and visitors.** Any district with many commuters, shoppers or visitors (not only the one "
        "flagged) has a per-resident rate pushed up by people who don't live there.",
        f"- **Coverage.** Districts with fewer than {MIN_N} complaints would be hatched on the map "
        f"({int(d.low_n.sum())} here). Correlations across areas say nothing about individual households.",
        "",
    ]
    path = out_dir / "answers" / "q07.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
