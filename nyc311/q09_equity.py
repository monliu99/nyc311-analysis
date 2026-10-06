"""Q9: equity in city services, by community district.

Four parts, all on complaints (default analysis set minus information requests and "Invalid entries") that map to
one of the 59 standard community districts:
  1. Unresolved complaints: count, per 1,000 residents and share, each compared with the citywide share for the
     same year (age) and for the same complaint type and year (age and mix), by indirect standardization.
  2. Concentration: location quotients (district share of a category or type / citywide share) and a Hoover
     index per category (% of its complaints that would have to move to match the overall map of complaints).
  3. Origin: the city / private / natural / other mix by district (Q3's taxonomy).
  4. Inequity: rank correlations (n, 95% CI) of median household income with the mix-adjusted time to close
     (Q4's method, imported from q04_response_times) and the mix-adjusted unresolved share, by district and by
     PUMA (4 district pairs share one income value), plus income fifths and within-borough slopes.
Population: linear interpolation between the 2010 and 2020 Census counts by calendar year, averaged with weights
equal to the days of data in each year. Every number in text is computed here; the same code runs on one month
or twelve years.
"""
from __future__ import annotations

import datetime as dt
import re
import textwrap

import numpy as np
import pandas as pd

from nyc311 import geo
from nyc311 import q04_response_times as q4
from nyc311.clean import clean_glob
from nyc311.report import Report
from nyc311.taxonomy import INVALID, ORIGINS

STAGE = "q09_equity"
DEMO_CSV = geo.DISTRICTS_GEOJSON.parent / "cd_demographics.csv"
DAYS_PER_YEAR = 365.25
CENSUS_YEARS = (2010, 2020)
MIN_N = 100                 # complaints a district needs to be ranked and correlated (fewer: low_n, reported)
MIN_N_CLOSED = 100          # closed complaints for the mix-adjusted time to close (as in Q4)
MIN_CELL = 30               # complaints of a category/type in a district before the district is named for it
LQ_HIGH = 2.0               # "concentrated": at least twice the citywide share
MAIN_CATEGORY_SHARE = 2.0   # % of complaints: categories at least this big are the "main" ones
TOP_TYPES = 15              # canonical types (by volume) checked for concentration
FINAL_DAYS = 90             # "final months": complaints opened in the last 90 days of the data
TOP_K = 5
INCOME_GROUPS = 5           # income fifths of PUMAs
ALPHA = 0.05
Z95 = 1.96
BASE = "not is_test and not dup_key"
COMPLAINTS = f"{BASE} and not coalesce(is_info_request, false) and coalesce(category, '') <> '{INVALID}'"
ORIGIN_LABEL = {"city": "city (infrastructure and services)", "private": "private (people and businesses)",
                "natural": "natural (weather, trees, animals)", "other": "other or mixed"}
METRICS = {
    "adj_pct": "time to close vs typical for the same type and year (%)",
    "excess_pts_mix": "unresolved share minus expected for type and year (points)",
    "pct_unresolved": "unresolved share, raw (%)",
    "unresolved_per_1k": "unresolved per 1,000 residents",
    "rate_per_1k_yr": "complaints per 1,000 residents a year",
    "pct_city": "city-origin share (%)",
    "pct_private": "private-origin share (%)",
}
EQUITY_METRICS = ["adj_pct", "excess_pts_mix"]
COUNT_COLS = ["n", "n_unresolved", "expected_year", "expected_mix", "var_mix", "n_unresolved_final", "n_closed",
              "avg_pop", "resident_years", *[f"n_{o}" for o in ORIGINS]]


# ---------- pure functions ----------

def load_demographics(path=DEMO_CSV):
    return pd.read_csv(path, dtype={"puma_2010": str, "puma_shared_with": str})


def interpolated_population(demo, days_by_year):
    """Average population per district over the data: Census 2010 and 2020 counts joined by a straight line by
    calendar year (years outside 2010-2020 extrapolated on the same line), weighted by days of data in each year.
    resident_years = sum over years of days x population / 365.25."""
    y0, y1 = CENSUS_YEARS
    days = pd.Series(days_by_year, dtype=float)
    days = days[days > 0]
    frac = float(((days.index.to_numpy(dtype=float) - y0) / (y1 - y0) * days).sum() / days.sum())
    p0, p1 = demo.population_2010.to_numpy(float), demo.population_2020.to_numpy(float)
    avg = p0 + (p1 - p0) * frac
    return pd.DataFrame({"boro_cd": demo.boro_cd.astype(int).to_numpy(), "avg_pop": avg,
                         "resident_years": avg * days.sum() / DAYS_PER_YEAR})


def location_quotients(counts):
    """counts: boro_cd, key, n. Full district x key grid: share of the district's complaints, citywide share, LQ."""
    g = counts.groupby(["boro_cd", "key"]).n.sum().unstack(fill_value=0)
    c = g.stack().rename("n").reset_index()
    c["n_district"] = c.boro_cd.map(g.sum(axis=1))
    tot = float(c.n.sum())
    c["share_pct"] = 100 * c.n / c.n_district
    c["city_share_pct"] = 100 * c.key.map(g.sum(axis=0)) / tot
    c["lq"] = c.share_pct / c.city_share_pct
    c["n"] = c.n.astype(int)
    return c


def concentration(lq, min_cell=MIN_CELL, min_share=0.0, lq_high=LQ_HIGH):
    """Per key: Hoover index (% of its complaints that would have to move district to match all complaints), the
    districts with LQ >= lq_high and at least min_cell complaints of it, and the top such district by LQ."""
    tot = lq.drop_duplicates("boro_cd").set_index("boro_cd").n_district
    rows = []
    for key, g in lq.groupby("key", sort=False):
        nk = g.n.sum()
        if nk == 0 or g.city_share_pct.iloc[0] < min_share:
            continue
        s_k = g.set_index("boro_cd").n / nk
        hoover = 50 * float((s_k - tot.reindex(s_k.index) / tot.sum()).abs().sum())
        ok = g[g.n >= min_cell].sort_values("lq", ascending=False)
        high = ok[ok.lq >= lq_high]
        top = ok.iloc[0] if len(ok) else None
        rows.append({"key": key, "n": int(nk), "city_share_pct": float(g.city_share_pct.iloc[0]),
                     "hoover_pct": hoover, "n_districts_high": len(high),
                     "pct_of_key_in_high": 100 * high.n.sum() / nk,
                     "pct_of_all_in_high": 100 * high.n_district.sum() / tot.sum(),
                     "top_boro_cd": int(top.boro_cd) if top is not None else np.nan,
                     "top_lq": float(top.lq) if top is not None else np.nan,
                     "top_n": int(top.n) if top is not None else np.nan,
                     "top_share_pct": float(top.share_pct) if top is not None else np.nan,
                     "high_districts": list(high.boro_cd.astype(int))})
    cols = ["key", "n", "city_share_pct", "hoover_pct", "n_districts_high", "pct_of_key_in_high",
            "pct_of_all_in_high", "top_boro_cd", "top_lq", "top_n", "top_share_pct", "high_districts"]
    return pd.DataFrame(rows, columns=cols).sort_values("hoover_pct", ascending=False).reset_index(drop=True)


def spearman(x, y):
    """Rank correlation with n, a two-sided p-value (t-test on ranks, as scipy's) and a 95% CI (Fisher z with the
    Fieller-Hartley-Pearson variance 1.06 / (n - 3)). NaN when fewer than 4 complete pairs or no variation."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    out = {"n": n, "rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n < 4:
        return out
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return out
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1 - 1e-12:
        return {**out, "rho": rho, "p": 0.0, "ci_low": rho, "ci_high": rho}
    import statsmodels.api as sm
    p = float(sm.OLS(ry, sm.add_constant(rx)).fit().pvalues[1])
    z, se = np.arctanh(rho), np.sqrt(1.06 / (n - 3)) if n > 3 else np.inf
    return {"n": n, "rho": rho, "p": p, "ci_low": float(np.tanh(z - Z95 * se)), "ci_high": float(np.tanh(z + Z95 * se))}


def _rates(t):
    """Derived rates from pooled counts (districts or PUMAs)."""
    t["pct_unresolved"] = 100 * t.n_unresolved / t.n
    t["excess_pts_mix"] = 100 * (t.n_unresolved - t.expected_mix) / t.n
    if "avg_pop" in t:
        t["unresolved_per_1k"] = 1000 * t.n_unresolved / t.avg_pop
    if "resident_years" in t:
        t["rate_per_1k_yr"] = 1000 * t.n / t.resident_years
    for o in ORIGINS:
        if f"n_{o}" in t:
            t[f"pct_{o}"] = 100 * t[f"n_{o}"] / t.n
    return t


def puma_units(d):
    """Pool districts that share a PUMA (and so one income value): counts add up; the mix-adjusted time to close is
    pooled on the log scale, weighted by closed complaints."""
    d = d.copy()
    d["_lr"] = np.log1p(d.adj_pct / 100) * d.n_closed
    d["_w"] = d.n_closed.where(d.adj_pct.notna(), 0)
    agg = {c: (c, "sum") for c in COUNT_COLS if c in d}
    p = d.groupby("puma_2010").agg(median_hh_income=("median_hh_income", "first"), _lr=("_lr", "sum"),
                                   _w=("_w", "sum"), n_districts=("boro_cd", "size"),
                                   districts=("boro_cd", lambda s: ", ".join(str(int(x)) for x in sorted(s))), **agg)
    p["adj_pct"] = 100 * np.expm1(p._lr / p._w.where(p._w > 0))
    return _rates(p.drop(columns=["_lr", "_w"]).reset_index())


def income_fifths(p, k=INCOME_GROUPS):
    """PUMAs in k income groups (ties broken by order); pooled metrics per group."""
    p = p.dropna(subset=["median_hh_income"]).sort_values("median_hh_income").reset_index(drop=True)
    k = min(k, len(p))
    p["fifth"] = pd.qcut(p.median_hh_income.rank(method="first"), k, labels=False) + 1
    rows = []
    for f, g in p.groupby("fifth"):
        w = g.n_closed.where(g.adj_pct.notna(), 0)
        lr = (np.log1p(g.adj_pct.fillna(0) / 100) * w).sum() / w.sum() if w.sum() else np.nan
        rows.append({"fifth": int(f), "income_min": g.median_hh_income.min(), "income_max": g.median_hh_income.max(),
                     "pumas": len(g), "districts": int(g.n_districts.sum()), "n": int(g.n.sum()),
                     "adj_pct": 100 * np.expm1(lr), "excess_pts_mix": 100 * (g.n_unresolved.sum() - g.expected_mix.sum())
                     / g.n.sum(), "pct_unresolved": 100 * g.n_unresolved.sum() / g.n.sum(),
                     "unresolved_per_1k": 1000 * g.n_unresolved.sum() / g.avg_pop.sum(),
                     "pct_city": 100 * g.n_city.sum() / g.n.sum() if "n_city" in g else np.nan})
    return pd.DataFrame(rows)


def income_slopes(d, metrics=EQUITY_METRICS):
    """OLS of each metric on log2(income), without and with borough fixed effects (HC1 SEs): change per doubling."""
    import statsmodels.api as sm
    rows = []
    for m in metrics:
        s = d.dropna(subset=[m, "median_hh_income"])
        if len(s) < 8:
            continue
        x = np.log2(s.median_hh_income.to_numpy(float))
        for label, X in [("all districts", pd.DataFrame({"log2_income": x}, index=s.index)),
                         ("within borough", pd.concat([pd.DataFrame({"log2_income": x}, index=s.index),
                                                       pd.get_dummies(s.borough, drop_first=True, dtype=float)],
                                                      axis=1))]:
            fit = sm.OLS(s[m].to_numpy(float), sm.add_constant(X, has_constant="add")).fit(cov_type="HC1")
            ci = fit.conf_int(ALPHA).loc["log2_income"]
            rows.append({"metric": m, "model": label, "n": len(s), "per_doubling": float(fit.params["log2_income"]),
                         "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
                         "p": float(fit.pvalues["log2_income"])})
    return pd.DataFrame(rows, columns=["metric", "model", "n", "per_doubling", "ci_low", "ci_high", "p"])


# ---------- data ----------

def _load(con, src):
    """Exclusion steps and the analysis set as temp table q9all (district may be NULL) and view q9 (districts)."""
    raw = con.sql(f"""
        with b as (select created::date as d, coalesce(is_info_request, false) as info,
                          coalesce(category, '') = '{INVALID}' as invalid, special_area_cd is not null as special,
                          boro_cd is null as nocd
                   from {src} where {BASE}),
        cov as (select distinct created::date as cd from {src}
                where {BASE} and not out_of_file_order and created is not null)
        select info, invalid, b.d is null as nodate, cov.cd is not null as covered, special, nocd, count(*)::bigint n
        from b left join cov on b.d = cov.cd group by all""").df()
    steps, keep = [("default analysis set", int(raw.n.sum()))], raw

    def drop(label, mask):
        nonlocal keep
        steps.append((label, int(keep.n[mask].sum())))
        keep = keep[~mask]

    drop("information requests (is_info_request)", keep["info"])
    drop(f"category '{INVALID}'", keep.invalid)
    drop("no valid created date", keep.nodate)
    drop("dates with only out_of_file_order rows (outside the files' period)", ~keep.covered)
    n_benchmark = int(keep.n.sum())
    drop("park or airport (special_area_cd)", keep.special)
    drop("no community district", keep.nocd)
    steps.append(("analyzed: one of the 59 community districts", int(keep.n.sum())))
    ex = pd.DataFrame(steps, columns=["step", "rows"])
    ex["pct"] = 100 * ex.rows / max(int(ex.rows.iloc[0]), 1)
    con.execute(f"""
        create or replace temp table q9all as
        select created, coalesce(year(created), -1) as yr, boro_cd,
               canonical_type, coalesce(canonical_type, '(none)') as ct, coalesce(category, '(none)') as category,
               coalesce(origin, 'other') as origin, unresolved, duration_h, closed_at_creation,
               date_only_created, possibly_incomplete_day
        from {src} where {COMPLAINTS}
          and created::date in (select distinct created::date from {src}
                                where {BASE} and not out_of_file_order and created is not null)""")
    con.execute("create or replace temp view q9 as select * from q9all where boro_cd is not null")
    con.execute(f"create or replace temp view q9r as select * from q9all where duration_h is not null "
                f"and not ({q4.INSTANT})")
    return ex, n_benchmark


def days_by_year(con, src):
    d = con.sql(f"""select year(cd) y, count(*) n_days from (select distinct created::date cd from {src}
                    where {BASE} and not out_of_file_order and created is not null) group by 1 order by 1""").df()
    return dict(zip(d.y.astype(int), d.n_days.astype(int)))


def unresolved_by(con, rel, group_col, d_final):
    """Observed vs expected unresolved per group (NULL group kept). expected_year = sum over rows of the citywide
    unresolved share for the row's year; expected_mix = the same for its canonical type and year."""
    return con.sql(f"""
        with b as (select {group_col} as grp, ct, yr, unresolved::int as u, created::date >= '{d_final}' as late
                   from {rel}),
        cy as (select yr, avg(u) p_y from b group by 1),
        cty as (select ct, yr, avg(u) p_ty from b group by all)
        select grp, count(*)::bigint as n, sum(u)::bigint as n_unresolved, sum(p_y) as expected_year,
               sum(p_ty) as expected_mix, sum(p_ty * (1 - p_ty)) as var_mix,
               count_if(u = 1 and late)::bigint as n_unresolved_final
        from b join cy using (yr) join cty using (ct, yr) group by grp""").df()


def by_year(con):
    return con.sql("""select yr as year, count(*)::bigint n, count_if(unresolved)::bigint n_unresolved,
                      100 * avg(unresolved::int) pct_unresolved from q9all group by 1 order by 1""").df()


def district_table(con, demo, pop, d_final, names):
    u = unresolved_by(con, "q9all", "boro_cd", d_final)
    u_all = u.copy()
    u = u.dropna(subset=["grp"]).rename(columns={"grp": "boro_cd"})
    u["boro_cd"] = u.boro_cd.astype(int)
    o = con.sql("select boro_cd, origin, count(*)::bigint n from q9 group by all").df()
    o = o.pivot_table(index="boro_cd", columns="origin", values="n", aggfunc="sum", fill_value=0)
    o = o.reindex(columns=list(ORIGINS), fill_value=0).add_prefix("n_").reset_index()
    o["boro_cd"] = o.boro_cd.astype(int)
    t = q4.mix_adjusted(con, "q9r", "boro_cd", min_n=MIN_N_CLOSED).rename(
        columns={"grp": "boro_cd", "n": "n_closed"})[["boro_cd", "n_closed", "median_h", "adj_pct"]]
    t["boro_cd"] = t.boro_cd.astype(int)
    closed = con.sql("select boro_cd, count(*)::bigint n_closed_all from q9r where boro_cd is not null "
                     "group by 1").df().astype(int)
    keep = ["boro_cd", "borough", "cd_number", "cd_name", "puma_2010", "puma_shared_with", "median_hh_income"]
    d = (demo[keep].merge(pop, on="boro_cd").merge(u, on="boro_cd").merge(o, on="boro_cd", how="left")
         .merge(closed, on="boro_cd", how="left").merge(t.drop(columns="n_closed"), on="boro_cd", how="left"))
    d["n_closed"] = d.pop("n_closed_all").fillna(0).astype(int)
    d = _rates(d)
    d["expected_pct_year"] = 100 * d.expected_year / d.n
    d["excess_pts_year"] = d.pct_unresolved - d.expected_pct_year
    d["expected_pct_mix"] = 100 * d.expected_mix / d.n
    d["z_mix"] = (d.n_unresolved - d.expected_mix) / np.sqrt(d.var_mix.where(d.var_mix > 0))
    d["dominant_origin"] = d[[f"pct_{o}" for o in ORIGINS]].idxmax(axis=1).str.replace("pct_", "", regex=False)
    d["low_n"] = d.n < MIN_N
    d["label"] = [q4.district_label(c, names) for c in d.boro_cd]
    d["short"] = [_short(r) for r in d.itertuples()]
    return d.sort_values("n_unresolved", ascending=False).reset_index(drop=True), u_all


def _short(r):
    first = re.split(r",| and ", str(r.cd_name))[0].strip()
    return f"{first} ({q4.BORO_ABBR.get(r.boro_cd // 100, '?')} {r.boro_cd % 100})"


def lq_tables(con, d_ok):
    cats = con.sql("select boro_cd, category as key, count(*)::bigint n from q9 group by all").df()
    cats = cats[cats.boro_cd.isin(d_ok)]
    lq_c = location_quotients(cats)
    types = con.sql("select boro_cd, ct as key, count(*)::bigint n from q9 group by all").df()
    types = types[types.boro_cd.isin(d_ok)]
    top = types.groupby("key").n.sum().sort_values(ascending=False).head(TOP_TYPES).index
    lq_all_types = location_quotients(types)
    lq_t = lq_all_types[lq_all_types.key.isin(top)].reset_index(drop=True)
    conc_c = concentration(lq_c, min_share=MAIN_CATEGORY_SHARE)
    conc_t = concentration(lq_t)
    return lq_c, lq_t, conc_c, conc_t


def origin_city(con):
    o = con.sql("""select origin, count(*)::bigint n, count_if(unresolved)::bigint n_unresolved
                   from q9 group by 1""").df()
    o = o.set_index("origin").reindex(list(ORIGINS), fill_value=0).reset_index()
    o["pct"] = 100 * o.n / o.n.sum()
    o["pct_unresolved"] = 100 * o.n_unresolved / o.n.where(o.n > 0)
    top = con.sql("""select origin, category, count(*)::bigint n from q9 group by all""").df()
    o["main_categories"] = [", ".join(f"{r.category} {100 * r.n / max(int(o.n[o.origin == k].iloc[0]), 1):.0f}%"
                                      for r in top[top.origin == k].nlargest(3, "n").itertuples())
                            for k in o.origin]
    return o


def correlations(d, p):
    rows = []
    for unit, t in [("district", d), ("PUMA", p)]:
        for m in METRICS:
            if m in t:
                s = spearman(t.median_hh_income, t[m])
                rows.append({"unit": unit, "metric": m, "label": METRICS[m], **s})
    return pd.DataFrame(rows)


# ---------- chart ----------

def _nice(v):
    if v <= 0 or not np.isfinite(v):
        return 0.0
    e = 10 ** np.floor(np.log10(v))
    return float(min([1, 2, 2.5, 5, 10], key=lambda m: abs(m * e - v)) * e)


def _fmt_num(v):
    return f"{v:,.0f}" if v >= 10 else f"{v:,.1f}".rstrip("0").rstrip(".") if v >= 1 else f"{v:.2g}"


def _place_labels(fig, ax, items, dots, fontsize=8.5):
    """Greedy label placement: for each (x, y, text, radius_pt), try offsets around the dot and keep the first that
    overlaps no placed label, no dot and stays inside the axes (else the least-overlapping one)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.transforms import Bbox

    from nyc311 import nytplot as nyt

    canvas = FigureCanvasAgg(fig)
    canvas.draw()   # resolves the constrained layout, so display coordinates are final
    renderer = canvas.get_renderer()
    px = fig.dpi / 72
    boxes = []
    for x, y, r in dots:
        cx, cy = ax.transData.transform((x, y))
        boxes.append(Bbox([[cx - r * px, cy - r * px], [cx + r * px, cy + r * px]]))
    frame = ax.get_window_extent(renderer)
    placed = []
    for x, y, text, r in items:
        g = r + 3
        cands = [(g, 0, "left", "center"), (-g, 0, "right", "center"), (0, g + 2, "center", "bottom"),
                 (0, -g - 2, "center", "top"), (g, g + 6, "left", "bottom"), (-g, g + 6, "right", "bottom"),
                 (g, -g - 6, "left", "top"), (-g, -g - 6, "right", "top"), (g + 14, 0, "left", "center"),
                 (-g - 14, 0, "right", "center"), (0, g + 16, "center", "bottom"), (0, -g - 16, "center", "top")]
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, fontfamily=nyt.SANS)
            bb = t.get_window_extent(renderer).expanded(1.04, 1.1)
            t.remove()
            inside = (bb.x0 >= frame.x0 and bb.x1 <= frame.x1 and bb.y0 >= frame.y0 and bb.y1 <= frame.y1)
            score = 10 * sum(bb.overlaps(b) for b in placed) + sum(bb.overlaps(b) for b in boxes) + 100 * (not inside)
            if best is None or score < best[0]:
                best = (score, dx, dy, ha, va, bb)
            if score == 0:
                break
        _, dx, dy, ha, va, bb = best
        far = (dx != 0 and dy != 0) or abs(dx) > r + 5 or abs(dy) > r + 8   # diagonal or distant: draw a leader
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                    fontfamily=nyt.SANS, color="#333333", zorder=5,
                    arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.6, "shrinkA": 1, "shrinkB": r}
                    if far else None)
        placed.append(bb)


def chart(d, path, title, note, rho_text):
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    s = d[~d.low_n & d.adj_pct.notna() & d.median_hh_income.notna()].copy()
    fig = Figure(figsize=(9.5, 7.4))
    key, ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 7]})
    x, y = (s.median_hh_income / 1000).to_numpy(), s.adj_pct.to_numpy()
    vmax = max(float(s.unresolved_per_1k.max()), 1e-9)
    area = lambda v: 25 + 475 * np.asarray(v, float) / vmax   # noqa: E731  marker area (pt^2) grows with the value
    sizes = area(s.unresolved_per_1k)
    over = (s.excess_pts_mix > 0).to_numpy()
    for mask, hue in [(over, nyt.CORAL), (~over, nyt.SAGE)]:
        ax.scatter(x[mask], y[mask], s=sizes[mask], color=hue.line, alpha=0.72, linewidths=0.7, edgecolors="white",
                   zorder=3)
    ax.axhline(0, color="#333333", linewidth=1.0, zorder=2)
    ax.spines["bottom"].set_color("#d9d9d9")
    span = float(np.ptp(y)) or 1.0
    xspan = float(np.ptp(x)) or 1.0
    ax.set_ylim(min(y.min(), 0) - 0.10 * span, max(y.max(), 0) + 0.10 * span)
    ax.set_xlim(x.min() - 0.06 * xspan, x.max() + 0.06 * xspan)
    nyt.format_y(ax, suffix="%")
    nyt.format_x(ax, prefix="$", suffix="k", nbins=6)
    ax.text(0.005, 0.99, "Slower than typical for the same complaints", transform=ax.transAxes, va="top",
            fontsize=9, color=nyt.NOTE_GRAY, fontfamily=nyt.SANS)
    ax.text(0.005, 0.01, "Faster than typical", transform=ax.transAxes, va="bottom", fontsize=9,
            color=nyt.NOTE_GRAY, fontfamily=nyt.SANS)

    # key strip above the plot: colors named directly, the correlation, and a dot-size key (no legend box)
    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.text(0, 0.78, "More complaints still open than expected for their mix", fontsize=10.5, fontweight="bold",
             color=nyt.CORAL.line, fontfamily=nyt.SANS, va="center")
    key.text(0, 0.42, "Fewer still open than expected", fontsize=10.5, fontweight="bold", color=nyt.SAGE.line,
             fontfamily=nyt.SANS, va="center")
    key.text(0, 0.06, rho_text, fontsize=9, color="#333333", fontfamily=nyt.SANS, va="center")
    vals = sorted({_nice(v) for v in [vmax * 0.2, vmax * 0.5, vmax] if _nice(v) > 0})
    key.text(1, 0.95, "Dot size: unresolved per 1,000 residents", ha="right", va="top", fontsize=8.5,
             color=nyt.NOTE_GRAY, fontfamily=nyt.SANS)
    for i, v in enumerate(vals[::-1]):
        xk = 0.975 - 0.07 * i
        key.scatter([xk], [0.38], s=area(v), color="none", edgecolors=nyt.NOTE_GRAY, linewidths=0.8, clip_on=False)
        key.text(xk, 0.0, _fmt_num(v), ha="center", va="center", fontsize=8, color=nyt.NOTE_GRAY,
                 fontfamily=nyt.SANS)

    # label the extremes: slowest, fastest, largest backlog per resident, largest excess still open
    picks = []
    for col, asc, k in [("adj_pct", False, 3), ("adj_pct", True, 2), ("unresolved_per_1k", False, 1),
                        ("excess_pts_mix", False, 1)]:
        for b in s.sort_values(col, ascending=asc).boro_cd.head(k):
            if b not in picks:
                picks.append(b)
    radius = np.sqrt(sizes / np.pi)
    pos = {b: i for i, b in enumerate(s.boro_cd)}
    items = [(x[pos[b]], y[pos[b]], s.short.iloc[pos[b]], radius[pos[b]]) for b in picks]
    nyt.title(fig, title)
    nyt.note(fig, note)
    _place_labels(fig, ax, items, list(zip(x, y, radius)))
    return nyt.save(fig, path)


# ---------- run ----------

def run(cfg):
    from nyc311.q05_top_types import data_period, period_label, period_long

    con = cfg.connect()
    src = clean_glob(cfg)
    demo = load_demographics()
    names = dict(zip(demo.boro_cd.astype(int), demo.cd_name))
    ex, n_benchmark = _load(con, src)
    d0, d1 = data_period(con, src)
    n_days = (d1 - d0).days + 1
    d_final = d1 - dt.timedelta(days=FINAL_DAYS - 1)
    dby = days_by_year(con, src)
    pop = interpolated_population(demo, dby)
    d, u_all = district_table(con, demo, pop, d_final, names)
    ok = d[~d.low_n]
    lq_c, lq_t, conc_c, conc_t = lq_tables(con, set(ok.boro_cd))
    p = puma_units(ok)
    res = {
        "exclusions": ex, "n_benchmark": n_benchmark, "districts": d, "unresolved_all_groups": u_all,
        "by_year": by_year(con), "lq_categories": lq_c, "lq_types": lq_t, "concentration_categories": conc_c,
        "concentration_types": conc_t, "origin": d[["boro_cd", "label", *[f"pct_{o}" for o in ORIGINS],
                                                     "dominant_origin"]],
        "origin_city": origin_city(con), "pumas": p, "correlations": correlations(ok, p),
        "fifths": income_fifths(p), "slopes": income_slopes(ok), "period": period_label(d0, d1),
        "period_long": period_long(d0, d1), "n_days": n_days, "d_final": d_final, "days_by_year": dby,
        "n_shared_pairs": int((demo.puma_shared_with.notna() & (demo.boro_cd < pd.to_numeric(
            demo.puma_shared_with, errors="coerce"))).sum()),
        "n_unresolved_incomplete_day": int(con.sql(
            "select count(*) from q9 where unresolved and possibly_incomplete_day").fetchone()[0]),
    }
    res["facts"] = _facts(res)

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["short"]).to_csv(agg / "q09_districts.csv", index=False)
    lq_c.to_csv(agg / "q09_lq_categories.csv", index=False)
    lq_t.to_csv(agg / "q09_lq_types.csv", index=False)
    pd.concat([conc_c.assign(level="category"), conc_t.assign(level="canonical type")]).to_csv(
        agg / "q09_concentration.csv", index=False)
    res["origin_city"].to_csv(agg / "q09_origin.csv", index=False)
    res["correlations"].to_csv(agg / "q09_correlations.csv", index=False)
    p.to_csv(agg / "q09_pumas.csv", index=False)
    res["fifths"].to_csv(agg / "q09_income_fifths.csv", index=False)
    res["slopes"].to_csv(agg / "q09_income_slopes.csv", index=False)
    ex.to_csv(agg / "q09_exclusions.csv", index=False)

    figs = cfg.out_dir / "figs"
    for old in figs.glob("q09_*.png"):
        if old.name != "q09_equity.png":
            old.unlink()
    f = res["facts"]
    chart(d, figs / "q09_equity.png", f["title"], f["note"], f["rho_text"])
    _report(cfg, res)
    _write_answer(cfg, res)
    return res


# ---------- text ----------

def _pct(x):
    return q4._pct(x)


def _pts(x):
    return f"{x:+.1f}".replace("-", "\N{MINUS SIGN}") + " points"


def _rho(s):
    if not np.isfinite(s["rho"]):
        return f"not estimable (n {s['n']})"
    return (f"\N{GREEK SMALL LETTER RHO} {s['rho']:+.2f} (95% CI {s['ci_low']:+.2f} to {s['ci_high']:+.2f}, "
            f"n {s['n']}, p {_p(s['p'])})").replace("-", "\N{MINUS SIGN}")


def _signed(v):
    return f"{v:+.2f}".replace("-", "\N{MINUS SIGN}") if np.isfinite(v) else "n/a"


def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def _corr(res, unit, metric):
    c = res["correlations"]
    r = c[(c.unit == unit) & (c.metric == metric)]
    return r.iloc[0].to_dict() if len(r) else {"n": 0, "rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}


def _direction(s, low_text, high_text, none_text):
    if np.isfinite(s["rho"]) and s["p"] < ALPHA:
        return low_text if s["rho"] < 0 else high_text
    return none_text


def _facts(res):
    d = res["districts"]
    ok = d[~d.low_n]
    t = _corr(res, "district", "adj_pct")
    u = _corr(res, "district", "excess_pts_mix")
    n_ok = int(ok.adj_pct.notna().sum())
    title = _direction(t, "In lower-income districts, the same complaints take longer to close",
                       "In higher-income districts, the same complaints take longer to close",
                       f"Across {n_ok} districts, time to close for the same complaints shows no clear link to income")
    rho_text = (f"Rank correlation with income: time to close {t['rho']:+.2f}, still open {u['rho']:+.2f} "
                f"(n {t['n']})").replace("-", "\N{MINUS SIGN}") if np.isfinite(t["rho"]) else ""
    note = (f"Note: Each dot is a community district, {res['period']}. Vertical: time to close vs. citywide typical "
            f"for the same complaint type and year; color: share still open vs. expected for that mix. Income: ACS "
            f"2010{chr(8211)}2014 median household income, 2014 dollars ({res['n_shared_pairs']} pairs share a value).")
    return {"title": title, "rho_text": rho_text, "note": textwrap.fill(note, 150), "t": t, "u": u}


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    d, ex = res["districts"], res["exclusions"]
    ok = d[~d.low_n]
    rep.add(f"Q9. Equity in city services by community district, {res['period_long']} ({res['n_days']:,} days)",
            "population: Census 2010 -> 2020 linear by year, weighted by days of data per year: "
            + ", ".join(f"{y} {n:,}" for y, n in res["days_by_year"].items()),
            f"districts ranked (>= {MIN_N} complaints): {len(ok)} of {len(d)}; unresolved opened in the final "
            f"{FINAL_DAYS} days (from {res['d_final']:%Y-%m-%d}): {int(d.n_unresolved_final.sum()):,} of "
            f"{int(d.n_unresolved.sum()):,}; on possibly_incomplete_day: {res['n_unresolved_incomplete_day']:,}")
    rep.table(ex.round(2), title="Rows (default set -> analyzed)")
    rep.table(res["by_year"].round(2), max_rows=14, title="Unresolved by year opened (all analyzed complaints)")
    cols = ["boro_cd", "short", "n", "n_unresolved", "pct_unresolved", "unresolved_per_1k", "expected_pct_year",
            "expected_pct_mix", "excess_pts_mix", "z_mix", "adj_pct", "median_hh_income"]
    rep.table(ok.sort_values("n_unresolved", ascending=False)[cols].round(2), max_rows=10,
              title="Most unresolved complaints (expected = citywide share for same year / same type and year)")
    rep.table(ok.sort_values("excess_pts_mix", ascending=False)[cols].round(2), max_rows=TOP_K,
              title="Most still open beyond their complaint mix (excess_pts_mix)")
    cc = res["concentration_categories"].copy()
    cc["top"] = [q4.district_short(int(b), {}) if np.isfinite(b) else "" for b in cc.top_boro_cd]
    rep.table(cc[["key", "n", "city_share_pct", "hoover_pct", "n_districts_high", "pct_of_key_in_high", "top",
                  "top_lq", "top_n"]].round(2), title=f"Main categories (>= {MAIN_CATEGORY_SHARE}% of complaints): "
              f"Hoover index; districts with LQ >= {LQ_HIGH} and >= {MIN_CELL} complaints")
    ct = res["concentration_types"].copy()
    ct["top"] = [q4.district_short(int(b), {}) if np.isfinite(b) else "" for b in ct.top_boro_cd]
    rep.table(ct[["key", "n", "hoover_pct", "n_districts_high", "top", "top_lq", "top_n"]].round(2), max_rows=TOP_TYPES,
              title=f"Top {TOP_TYPES} canonical types: concentration")
    rep.table(res["origin_city"].round(2), title="Origin, citywide (59 districts)")
    oc = [f"pct_{o}" for o in ORIGINS]
    rep.table(ok[oc].describe().loc[["min", "50%", "max"]].round(1).reset_index(), title="Origin shares across districts")
    rep.table(res["correlations"].drop(columns="label").round(3), title="Spearman with median household income")
    rep.table(res["fifths"].round(2), title="Income fifths of PUMAs (pooled)")
    rep.table(res["slopes"].round(3), title="OLS on log2(income), HC1: change per doubling of income")
    rep.emit()


def _md(text):
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: f"`{m.group(0)}`", text)


def _dl(rows, fmt):
    return "; ".join(fmt(r) for r in rows.itertuples())


def _write_answer(cfg, res):
    d, ex, f = res["districts"], res["exclusions"].set_index("step"), res["facts"]
    ok = d[~d.low_n]
    n_an = int(d.n.sum())
    tot_unres = int(d.n_unresolved.sum())
    city_pct = 100 * tot_unres / n_an
    by_count = ok.sort_values("n_unresolved", ascending=False).head(TOP_K)
    by_rate = ok.sort_values("unresolved_per_1k", ascending=False).head(TOP_K)
    by_share = ok.sort_values("pct_unresolved", ascending=False).head(3)
    by_excess = ok.sort_values("excess_pts_mix", ascending=False)
    sig_over = by_excess[by_excess.z_mix > Z95]
    sig_under = by_excess[by_excess.z_mix < -Z95]
    yr = res["by_year"]
    n_final = int(d.n_unresolved_final.sum())
    conc_c, conc_t, oc = res["concentration_categories"], res["concentration_types"], res["origin_city"].set_index(
        "origin")
    t, u = f["t"], f["u"]
    raw = _corr(res, "district", "pct_unresolved")
    t_p, u_p = _corr(res, "PUMA", "adj_pct"), _corr(res, "PUMA", "excess_pts_mix")
    city_c, priv_c = _corr(res, "district", "pct_city"), _corr(res, "district", "pct_private")
    rate_c = _corr(res, "district", "rate_per_1k_yr")
    both = spearman(ok.adj_pct, ok.excess_pts_mix)
    changed = (np.isfinite(raw["rho"]) and np.isfinite(u["rho"])
               and (np.sign(raw["rho"]) != np.sign(u["rho"]) or abs(raw["rho"] - u["rho"]) >= 0.2))
    like_line = (f"- **Like with like matters.** The raw unresolved share correlates with income at {_rho(raw)}; "
                 f"compared with the expected share for the same complaint types and years it is "
                 f"{_signed(u['rho'])}. "
                 + ("Much of the raw pattern comes from which kinds of complaints (and so which agencies) a district "
                    "files, not from how its complaints are handled." if changed else
                    "Adjusting for complaint type and year changes the picture little."))
    both_line = "- **The two service measures together:** across districts, time to close vs the excess still open " \
                f"{_rho(both)}. " + (
        "Districts with more complaints still open than expected tend to close the rest faster: closed-only times can flatter "
        "districts that leave slow complaints open, so read the two measures together."
        if np.isfinite(both["rho"]) and both["p"] < ALPHA and both["rho"] < 0 else
        "Districts that close slowly also leave more complaints open, so both measures point the same way."
        if np.isfinite(both["rho"]) and both["p"] < ALPHA else
        "The two measures are not clearly related, so each adds information.")
    fif = res["fifths"]
    sl = res["slopes"]
    lab = dict(zip(d.boro_cd, d.label))
    excl = "; ".join(f"{_md(k)} {int(r.rows):,} ({r.pct:.1f}%)" for k, r in ex.iloc[1:-1].iterrows())
    short = res["n_days"] < 365

    def dname(b):
        return lab.get(int(b), str(b))

    def conc_line(r):
        hi = ", ".join(q4.district_short(b, dict(zip(d.boro_cd, d.cd_name))) for b in r.high_districts[:4])
        more = f" and {len(r.high_districts) - 4} more" if len(r.high_districts) > 4 else ""
        top = (f"; strongest in {dname(r.top_boro_cd)} ({r.top_share_pct:.1f}% of its complaints, LQ {r.top_lq:.1f}, "
               f"n {int(r.top_n):,})" if np.isfinite(r.top_boro_cd) else "")
        return (f"{r.key} (Hoover {r.hoover_pct:.0f}%; {r.n_districts_high} districts at LQ {LQ_HIGH:g}+"
                + (f": {hi}{more}" if hi else "") + f"{top})")

    if np.isfinite(t["rho"]):
        q_lo, q_hi = fif.iloc[0], fif.iloc[-1]
        fifth_line = (f"- **Income fifths** ({len(res['pumas'])} PUMAs in {len(fif)} groups): the poorest fifth (median "
                      f"household income ${q_lo.income_min / 1000:,.0f}k-${q_lo.income_max / 1000:,.0f}k) closes the "
                      f"same complaints {_pct(q_lo.adj_pct)} vs typical and has {_pts(q_lo.excess_pts_mix)} still open "
                      f"beyond its mix; the richest fifth (${q_hi.income_min / 1000:,.0f}k-"
                      f"${q_hi.income_max / 1000:,.0f}k) {_pct(q_hi.adj_pct)} and {_pts(q_hi.excess_pts_mix)}. "
                      f"Unresolved per 1,000 residents: {q_lo.unresolved_per_1k:,.2f} vs {q_hi.unresolved_per_1k:,.2f}.")
    else:
        fifth_line = "- **Income fifths** could not be compared (too few districts with enough complaints)."

    def slope_line(m, unit):
        s = sl[sl.metric == m].set_index("model")
        if s.empty:
            return "not estimable"
        a, w = s.loc["all districts"], s.loc["within borough"]
        return (f"{a.per_doubling:+.1f} {unit} per doubling of income (95% CI {a.ci_low:+.1f} to {a.ci_high:+.1f}); "
                f"within boroughs {w.per_doubling:+.1f} ({w.ci_low:+.1f} to {w.ci_high:+.1f})").replace(
            "-", "\N{MINUS SIGN}")

    years_line = ""
    if len(yr) >= 2:
        old, new = yr.iloc[0], yr.iloc[-1]
        years_line = (f" By year opened, {int(old.year)} is {old.pct_unresolved:.1f}% unresolved and {int(new.year)} "
                      f"{new.pct_unresolved:.1f}%.")
    final_line = (f"- **Recent complaints.** {n_final:,} of the {tot_unres:,} unresolved complaints "
                  f"({100 * n_final / max(tot_unres, 1):.0f}%) were opened in the final {FINAL_DAYS} days of the data "
                  f"(from {res['d_final']:%b %d, %Y}); {res['n_unresolved_incomplete_day']:,} on possibly incomplete "
                  "days. Those had the least time to close, so every comparison below is against the citywide share "
                  "for the same year." + years_line)
    if short:
        final_line += (f" **The data cover only {res['n_days']} days, so all of it counts as 'final months' and "
                       "the age adjustment cannot separate old from new backlog here.** Status is as of the data "
                       "extract, which the data do not record; rerun on the full data.")

    lines = [
        "# Q9. Equity in city services",
        "",
        f"*Drafted by `nyc311/q09_equity.py` from {n_an:,} complaints in {len(d)} community districts, "
        f"{res['period_long']}. Every number is computed; write the prose from these bullets.*",
        "",
        "## What is counted",
        "",
        f"- **Unit:** the 59 standard community districts (`boro_cd`). Of {int(res['exclusions'].rows.iloc[0]):,} rows in "
        f"the default analysis set (not `is_test`, not `dup_key`), {n_an:,} ({ex.pct.iloc[-1]:.1f}%) are analyzed. "
        f"Left out: {excl}. The rows without a district still count in the citywide benchmarks "
        f"({res['n_benchmark']:,} complaints).",
        f"- **Unresolved** = a real status other than Closed or Cancelled (`unresolved`: Open, Pending, Assigned, ...). "
        f"**Expected** = the citywide unresolved share for the same year (age), or for the same complaint type and "
        f"year (age and mix); excess = observed minus expected, in percentage points.",
        f"- **Per 1,000 residents** uses each district's average population over the data: the 2010 and 2020 Census "
        f"counts joined by a straight line by year, weighted by the days of data each year ("
        + ", ".join(f"{y}: {n:,} days" for y, n in res["days_by_year"].items()) + ").",
        f"- **Time to close** is Q4's mix-adjusted measure: each closed complaint vs the citywide geometric mean for "
        f"the same complaint type and year, averaged per district (instant closures left out; districts need "
        f"{MIN_N_CLOSED} closures).",
        "",
        "## Where the unresolved complaints are",
        "",
        f"- **Citywide** {tot_unres:,} of {n_an:,} complaints ({city_pct:.1f}%) are unresolved.",
        f"- **Most unresolved complaints:** " + _dl(by_count, lambda r: f"{r.label} {int(r.n_unresolved):,} "
                                                     f"({r.pct_unresolved:.1f}% of {int(r.n):,})") + ".",
        f"- **Per 1,000 residents:** " + _dl(by_rate, lambda r: f"{r.label} {r.unresolved_per_1k:,.2f}") + ".",
        f"- **Highest share still open:** " + _dl(by_share, lambda r: f"{r.label} {r.pct_unresolved:.1f}% (expected "
                                                  f"{r.expected_pct_year:.1f}% for the year, {r.expected_pct_mix:.1f}% "
                                                  f"for its complaint mix)") + ".",
        f"- **Beyond their complaint mix** (observed minus expected for the same type and year), the most open: "
        + _dl(by_excess.head(3), lambda r: f"{r.label} {_pts(r.excess_pts_mix)} ({int(r.n_unresolved):,} open vs "
                                         f"{r.expected_mix:,.0f} expected)")
        + f"; the fewest: " + _dl(by_excess.tail(2).iloc[::-1], lambda r: f"{r.label} {_pts(r.excess_pts_mix)}")
        + f". {len(sig_over)} districts are above expectation and {len(sig_under)} below by more than 1.96 binomial "
        f"standard errors (which ignore clustering, so treat these as screens, not tests).",
        final_line,
        "",
        "## Are complaint types concentrated in specific areas?",
        "",
        f"- **Method.** Location quotient (LQ) = a district's share of a category / the citywide share; LQ {LQ_HIGH:g} "
        f"means twice as common there. A district is named only with {MIN_CELL}+ complaints of that kind. The Hoover "
        f"index is the % of a category's complaints that would have to move to another district to match the map of "
        f"all complaints (0 = spread like everything else).",
        f"- **Most concentrated main categories** ({MAIN_CATEGORY_SHARE:g}%+ of complaints): "
        + "; ".join(conc_line(r) for r in conc_c.head(4).itertuples()) + ".",
        f"- **Least concentrated:** " + "; ".join(f"{r.key} (Hoover {r.hoover_pct:.0f}%)"
                                              for r in conc_c.tail(2).iloc[::-1].itertuples()) + ".",
        f"- **Among the {len(conc_t)} biggest complaint types,** the most concentrated are "
        + "; ".join(conc_line(r) for r in conc_t.head(3).itertuples()) + ".",
        "",
        "## Where complaints originate",
        "",
        "- **Citywide mix** (Q3's taxonomy): " + "; ".join(
            f"{ORIGIN_LABEL[k]} {oc.loc[k, 'pct']:.1f}% ({int(oc.loc[k, 'n']):,}; {oc.loc[k, 'pct_unresolved']:.1f}% "
            f"unresolved; mainly {oc.loc[k, 'main_categories']})" for k in ORIGINS if oc.loc[k, "n"] > 0) + ".",
        f"- **By district,** private origin is the largest share in {int((ok.dominant_origin == 'private').sum())} of "
        f"{len(ok)} districts (range {ok.pct_private.min():.0f}%-{ok.pct_private.max():.0f}%); city origin ranges "
        f"{ok.pct_city.min():.0f}%-{ok.pct_city.max():.0f}% (highest: "
        + _dl(ok.nlargest(3, "pct_city"), lambda r: f"{r.label} {r.pct_city:.0f}%") + "; lowest: "
        + _dl(ok.nsmallest(2, "pct_city"), lambda r: f"{r.label} {r.pct_city:.0f}%") + ").",
        f"- **With income:** city-origin share {_rho(city_c)}; private-origin share {_rho(priv_c)}.",
        "",
        "## Patterns that suggest inequitable service",
        "",
        f"- **Time to close vs income:** {_rho(t)} across districts; {_rho(t_p)} across PUMAs (the {res['n_shared_pairs']} "
        f"district pairs that share one income value pooled). Linear fit: {slope_line('adj_pct', 'points')}.",
        f"- **Still open beyond the complaint mix vs income:** {_rho(u)} across districts; {_rho(u_p)} across "
        f"PUMAs; {slope_line('excess_pts_mix', 'points')}.",
        like_line,
        both_line,
        fifth_line,
        f"- **Volume:** complaints per 1,000 residents a year vs income {_rho(rate_c)}.",
        "",
        f"![{f['title']}](figs/q09_equity.png)",
        "",
        "- **Reading the chart.** Each dot is a district: left-right is median household income, up-down is how much "
        "longer (or shorter) the same kinds of complaints take to close than the citywide typical, color says whether "
        "more or fewer complaints are still open than expected for its mix, and size is unresolved complaints per "
        "1,000 residents. The labeled dots are the extremes.",
        "",
        "## Limits and what it would take to test causes",
        "",
        "- **Correlation, not causation.** These are associations across districts. Income moves with housing stock, "
        "renters, language, density and borough operations; a district-level link says nothing about individual "
        "households (ecological fallacy).",
        "- **Measurement.** Closing a ticket is not fixing the problem; agencies close differently (referrals, "
        "date-only HPD closures). 311 measures reporting, not need: lower-reporting areas can look well served. "
        f"ACS income is for PUMAs ({res['n_shared_pairs']} pairs share a value), in 2014 dollars; boundaries are "
        "current.",
        "- **To test causes:** ticket-level models with descriptor, building and agency-unit controls; independent "
        "measures of need (HPD violations, inspections, street-condition ratings) to separate reporting from "
        "conditions; crew and inspector deployment by district; and changes over time, such as a new staffing rule or "
        "service-level target rolled out in some districts first (difference-in-differences), or resident follow-up "
        "on whether the problem was fixed.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q09.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
