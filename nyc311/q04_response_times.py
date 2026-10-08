"""Q4: time to closure as the response-time KPI.

a) summary statistics and outliers; b) community districts with longer than typical times, raw and after adjusting
for complaint type and year; c) whether agencies differ once complaint type is accounted for (statsmodels on a
fixed-seed sample). Every number in the report, the charts and outputs/answers/q04.md is computed here.

Complaint set: rules.COMPLAINTS_SQL (not is_test, not dup_key, not an information request, not "Invalid entries").
Durations (duration_h) exist only for status Closed and closed >= opened. "Excluding instant closures" always means
rules.INSTANT_SQL: closed at the second they were opened, with a time of day (likely administrative). Same-day
closures recorded as dates only (both timestamps midnight, mainly HPD) stay in. That set is the response set used
in b) and c), and its median is the one quoted as "excluding instant closures" everywhere (Q8, Q11 use the same).
"""
from __future__ import annotations

import calendar
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from nyc311 import clean, rules
from nyc311.clean import clean_glob
from nyc311.report import Report

STAGE = "q04_response_times"
MIN_N = 100            # closed complaints needed to report an agency, district, borough, year or complaint type
MIN_N_SHARED = 30      # closed complaints an agency needs in a complaint type to be compared with another agency
TOP_K = 10             # slowest/fastest districts shown and compared across the two rankings
SAMPLE_ROWS = 1_000_000
SEED = 42
TUKEY_K = 1.5
Z95 = 1.96
AGENCY_ONLY = "agency alone"
NESTED_MODELS = ["complaint type", "type + year", "type + year + borough", "type + year + borough + agency"]
MODEL_LABELS = {AGENCY_ONLY: "Agency alone", "complaint type": "Complaint type alone", "type + year": "+ year opened",
                "type + year + borough": "+ borough", "type + year + borough + agency": "+ agency"}
DEMOGRAPHICS = Path(__file__).resolve().parent.parent / "data" / "external" / "cd_demographics.csv"
BORO_TITLE = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
BORO_ABBR = {1: "MN", 2: "BX", 3: "BK", 4: "QN", 5: "SI"}
COMPLAINTS = rules.COMPLAINTS_SQL
INSTANT = rules.INSTANT_SQL     # kept as a name: Q9 and Q10 import it
MIN_AGENCY_SHARE = 0.5          # % of the response set an agency needs to be charted and ranked by its median
DRIVER_K = 3                    # most extreme districts at each end whose gaps are broken down by complaint type
DRIVER_TYPES = 5                # complaint types listed per district in q04_district_drivers.csv
DESC_USED_PCT = 5               # a descriptor is part of an agency's work in a type if >= 5% of its closures there
LIKE_FOR_LIKE_PCT = 50          # a shared type is like-for-like if every agency has >= 50% of its closures in
                                # descriptors another agency in the type also uses (NULL descriptors never match)
ELEVATED_X = 3                  # a year (or month) stands out when its share still open is at least 3 times the
                                # median yearly share of the other years
BLOCK_MAX_MONTHS = 6            # a run of consecutive standout months is reported as a block if it is this short...
BLOCK_MIN_PCT = 50              # ...and holds at least this % of the peak year's unresolved complaints
UNRESOLVED_COVER_PCT = 75       # agencies named for the peak year: largest first until they hold this % of it...
UNRESOLVED_MAX_AGENCIES = 4     # ...but no more than this many
SIDE_LABEL_GAP = 0.95           # district chart: series names sit this many rows outside their group of bars
RESPONSE_SET = "excluding instant closures"
STAT_SETS = [
    ("all closed complaints", "true"),
    (RESPONSE_SET, f"not ({INSTANT})"),
    ("exact timestamps, excluding instant", f"not ({INSTANT}) and not date_only_created and not date_only_closed"),
]
QUANTILES = {"p10_h": 0.10, "p25_h": 0.25, "median_h": 0.50, "p75_h": 0.75, "p90_h": 0.90, "p99_h": 0.99}


def fmt_h(h):
    """Hours as text: seconds below a minute, minutes below an hour, hours below two days, days above."""
    if h is None or not np.isfinite(h):
        return "n/a"
    if h < 1 / 60:
        s = round(h * 3600)
        return f"{s:,.0f} second" + ("" if s == 1 else "s")
    if h < 1:
        m = round(h * 60)
        return f"{m:,.0f} minute" + ("" if m == 1 else "s")
    return f"{h:,.1f} hours" if h < 48 else f"{h / 24:,.1f} days"


def _days(h):
    return f"{h / 24:,.1f} days"


def _years(r):
    return str(r.first_year) if r.first_year == r.last_year else f"{r.first_year}-{r.last_year}"


def _pct(x):
    s = f"{x:+,.0f}%"
    return s.replace("-", "\N{MINUS SIGN}")


# ---------- data ----------

def _load(con, cfg):
    src = clean_glob(cfg)
    excl = con.sql(f"""select count(*) n_rows, count_if(is_test) n_test, count_if(dup_key and not is_test) n_dup,
                       count_if({rules.BASE_SQL} and coalesce(is_info_request, false)) n_info,
                       count_if({rules.BASE_SQL} and not coalesce(is_info_request, false)
                                and not ({COMPLAINTS})) n_invalid,
                       count_if({COMPLAINTS}) n_complaints from {src}""").df().iloc[0].astype(int)
    con.execute(f"""create or replace temp table q4 as
        select created, year(created) as year, agency, canonical_type, borough, boro_cd, status,
               closed is null as no_close_date, duration_h, unresolved, status_conflict, closed_before_created,
               closed_at_creation, closed_after_1yr, date_only_created, date_only_closed
        from {src} where {COMPLAINTS}""")
    con.execute(f"create or replace temp view q4r as select * from q4 where duration_h is not null and not ({INSTANT})")
    return src, excl


def district_names(path=DEMOGRAPHICS):
    if not Path(path).exists():
        return {}
    d = pd.read_csv(path, usecols=["boro_cd", "cd_name"])
    return dict(zip(d.boro_cd.astype(int), d.cd_name))


def district_label(cd, names):
    base = f"{BORO_TITLE.get(cd // 100, '?')} {cd % 100}"
    return f"{base} ({names[cd]})" if cd in names else base


def district_short(cd, names):
    code = f"{BORO_ABBR.get(cd // 100, '?')} {cd % 100}"
    return f"{names[cd]} ({code})" if cd in names else district_label(cd, {})


# ---------- a) summary statistics ----------

def _stats_select():
    # One list-quantile aggregate per set: separate quantile_cont calls each keep their own copy of the values.
    qs = ", ".join(repr(q) for q in QUANTILES.values())
    return (f"count(*) as n, avg(duration_h) as mean_h, stddev_samp(duration_h) as sd_h, "
            f"quantile_cont(duration_h, [{qs}]) as qs")


def summary_stats(con):
    parts = [f"select '{name}' as set, {_stats_select()} from q4 where duration_h is not null and {w}"
             for name, w in STAT_SETS]
    df = con.sql(" union all ".join(parts)).df()
    df["order"] = df.set.map({name: i for i, (name, _) in enumerate(STAT_SETS)})
    df = df[df.n > 1].sort_values("order").drop(columns="order").reset_index(drop=True)
    for i, c in enumerate(QUANTILES):
        df[c] = [float(v[i]) for v in df.qs]
    df = df.drop(columns="qs")
    for c in ["mean", "median", "sd"]:
        df[f"{c}_d"] = df[f"{c}_h"] / 24
    return df


def fences(con):
    q1, q3 = con.sql("""select quantile_cont(ln(duration_h), 0.25), quantile_cont(ln(duration_h), 0.75)
                        from q4 where duration_h > 0""").fetchone()
    iqr = q3 - q1
    return {"q1_h": float(np.exp(q1)), "q3_h": float(np.exp(q3)),
            "lower_h": float(np.exp(q1 - TUKEY_K * iqr)), "upper_h": float(np.exp(q3 + TUKEY_K * iqr))}


def type_outliers(con):
    """Tukey fences on log hours computed within each complaint type (types with >= MIN_N positive durations)."""
    df = con.sql(f"""
        with f as (select canonical_type, count(*) n, quantile_cont(ln(duration_h), 0.25) q1,
                          quantile_cont(ln(duration_h), 0.75) q3
                   from q4 where duration_h > 0 and canonical_type is not null group by 1),
        g as (select canonical_type, n, exp(q1 - {TUKEY_K} * (q3 - q1)) lower_h, exp(q3 + {TUKEY_K} * (q3 - q1)) upper_h
              from f where n >= {MIN_N} and q3 > q1)
        select g.canonical_type, mode(q.agency) agency, any_value(g.n) n, any_value(g.lower_h) lower_h,
               quantile_cont(q.duration_h, 0.5) median_h, any_value(g.upper_h) upper_h,
               count_if(q.duration_h < g.lower_h) n_below, count_if(q.duration_h > g.upper_h) n_above
        from q4 q join g using (canonical_type) where q.duration_h > 0 group by 1""").df()
    df["pct_outside"] = 100 * (df.n_below + df.n_above) / df.n
    return df.sort_values(["n_above", "n"], ascending=False).reset_index(drop=True)


def outliers(con, f, n_dur, tout):
    lo, hi = f["lower_h"], f["upper_h"]
    c = con.sql(f"""select count_if(duration_h = 0), count_if(closed_at_creation and date_only_created),
                    count_if({INSTANT}),
                    count_if(duration_h > 0 and duration_h < {lo!r}), count_if(duration_h > {hi!r}),
                    count_if(closed_after_1yr)
                    from q4 where duration_h is not null""").fetchone()
    rows = [
        ("closed at creation (0 hours)", c[0], "closed == opened"),
        ("  0 hours, dates only (closed the same day)", c[1], "both timestamps midnight"),
        ("  0 hours, exact second with a time of day", c[2], "instant closures (rules.INSTANT_SQL)"),
        ("below citywide Tukey fence (log scale)", c[3], f"> 0 and < {fmt_h(lo)}"),
        ("above citywide Tukey fence (log scale)", c[4], f"> {fmt_h(hi)}"),
        ("outside own complaint type's fences (log scale)", int(tout.n_below.sum() + tout.n_above.sum()),
         f"{len(tout)} types with >= {MIN_N} positive durations"),
        ("closed more than a year after opening", c[5], "closed_after_1yr"),
    ]
    df = pd.DataFrame(rows, columns=["kind", "n", "rule"])
    df["n"] = df.n.astype(int)
    df["pct"] = 100 * df.n / n_dur
    return df[["kind", "n", "pct", "rule"]]


def no_duration_reasons(con):
    df = con.sql("""
        select case when duration_h is not null then 'closed, with a duration'
                    when unresolved then 'still open (unresolved)'
                    when status_conflict then 'open status but a close date (status_conflict)'
                    when closed_before_created then 'closed before it was opened'
                    when trim(status) = 'Closed' and no_close_date then 'status Closed, no close date'
                    when status is null then 'no status'
                    else 'other status: ' || status end as reason, count(*) n
        from q4 group by 1 order by n desc""").df()
    df["pct"] = 100 * df.n / df.n.sum()
    return df


def by_agency(con):
    """Agencies with >= MIN_N closures. share_pct = the agency's share of the response set (all agencies)."""
    return con.sql(f"""
        select agency, count(*) n_complaints, count(duration_h) n,
               count(duration_h) filter (where not ({INSTANT})) n_excl_instant,
               100 * count(duration_h) filter (where not ({INSTANT})) / (select count(*) from q4r) share_pct,
               100 * avg(unresolved::int) pct_unresolved,
               avg(duration_h) / 24 mean_d, quantile_cont(duration_h, 0.5) / 24 median_d,
               quantile_cont(duration_h, 0.5) filter (where not ({INSTANT})) / 24 median_d_excl_instant,
               stddev_samp(duration_h) / 24 sd_d, quantile_cont(duration_h, 0.9) / 24 p90_d,
               100 * count_if(duration_h is not null and {INSTANT}) / count(duration_h) pct_instant,
               100 * count_if(duration_h is not null and (date_only_created or date_only_closed))
                   / count(duration_h) pct_day_precision
        from q4 group by agency having count(duration_h) >= {MIN_N} order by n desc""").df()


def compared_agencies(a):
    """The agencies charted and ranked by median: at least MIN_AGENCY_SHARE % of the response set (all of `a` if
    fewer than two qualify), fastest first by the median excluding instant closures."""
    shown = a[a.share_pct >= MIN_AGENCY_SHARE]
    if len(shown) < 2:
        shown = a
    return shown.sort_values(["median_d_excl_instant", "agency"]).reset_index(drop=True)


def by_year(con, full_years=()):
    y = con.sql(f"""
        select year, count(*) n_complaints, count(duration_h) n, 100 * avg(unresolved::int) pct_unresolved,
               quantile_cont(duration_h, 0.5) median_h,
               quantile_cont(duration_h, 0.5) filter (where not ({INSTANT})) median_h_excl_instant,
               avg(duration_h) mean_h, quantile_cont(duration_h, 0.9) p90_h
        from q4 group by year order by year""").df()
    y["full_year"] = y.year.isin(list(full_years))
    return y


def unresolved_by_year_agency(con):
    """Complaints and unresolved complaints by year opened x agency. pct_unresolved: of that agency's complaints that
    year; pct_of_year_unresolved: of the year's unresolved; pct_of_all_unresolved: of every unresolved complaint."""
    t = con.sql("""select year, coalesce(agency, '(none)') as agency, count(*) as n_complaints,
                          count_if(unresolved) as n_unresolved
                   from q4 where year is not null group by all""").df()
    for c in ["year", "n_complaints", "n_unresolved"]:
        t[c] = t[c].astype(int)
    t["pct_unresolved"] = 100 * t.n_unresolved / t.n_complaints
    year_total = t.groupby("year").n_unresolved.transform("sum")
    t["pct_of_year_unresolved"] = (100 * t.n_unresolved / year_total.where(year_total > 0)).fillna(0.0)
    total = t.n_unresolved.sum()
    t["pct_of_all_unresolved"] = 100 * t.n_unresolved / total if total else 0.0
    return t.sort_values(["year", "n_unresolved", "agency"], ascending=[True, False, True]).reset_index(drop=True)


def unresolved_by_month(con):
    """Complaints and unresolved complaints by month opened (year, month)."""
    m = con.sql("""select year, month(created) as month, count(*) as n_complaints, count_if(unresolved) as n_unresolved
                   from q4 where year is not null group by all order by 1, 2""").df()
    for c in ["year", "month", "n_complaints", "n_unresolved"]:
        m[c] = m[c].astype(int)
    m["pct_unresolved"] = 100 * m.n_unresolved / m.n_complaints
    return m


def _month_label(start, end):
    """'Feb–Jun 2011', 'Mar 2011' or 'Nov 2010–Feb 2011' for a run of months given as (year, month)."""
    from nyc311.q05_top_types import EN

    (y0, m0), (y1, m1) = start, end
    a, b = calendar.month_abbr[m0], calendar.month_abbr[m1]
    if (y0, m0) == (y1, m1):
        return f"{a} {y0}"
    return f"{a}{EN}{b} {y1}" if y0 == y1 else f"{a} {y0}{EN}{b} {y1}"


def _month_block(months, peak, typical, min_n):
    """The run of consecutive months around the peak year's month with the most unresolved complaints in which every
    month (>= min_n complaints) has at least ELEVATED_X times the typical share still open; None if that month is not
    elevated, or if the run is longer than BLOCK_MAX_MONTHS or holds less than BLOCK_MIN_PCT % of the year's."""
    m = months.copy()
    m["idx"] = m.year * 12 + m.month - 1
    m["pct_unresolved"] = 100 * m.n_unresolved / m.n_complaints
    in_year = m[m.year == peak]
    if in_year.empty or in_year.n_unresolved.sum() == 0:
        return None
    hi = m[(m.n_complaints >= min_n) & (m.pct_unresolved > ELEVATED_X * typical)].set_index("idx").sort_index()
    top = int(in_year.sort_values(["n_unresolved", "idx"], ascending=[False, True]).idx.iloc[0])
    if top not in hi.index:
        return None
    lo_i = hi_i = top
    while lo_i - 1 in hi.index:
        lo_i -= 1
    while hi_i + 1 in hi.index:
        hi_i += 1
    run = hi.loc[lo_i:hi_i]
    n_year = int(run[run.year == peak].n_unresolved.sum())
    pct_year = 100 * n_year / in_year.n_unresolved.sum()
    if len(run) > BLOCK_MAX_MONTHS or pct_year < BLOCK_MIN_PCT:
        return None
    start, end = divmod(lo_i, 12), divmod(hi_i, 12)
    start, end = (start[0], start[1] + 1), (end[0], end[1] + 1)
    return {"start": start, "end": end, "months": len(run), "n": int(run.n_unresolved.sum()), "n_in_year": n_year,
            "pct_of_year": pct_year, "label": _month_label(start, end),
            "rate_lo": float(run.pct_unresolved.min()), "rate_hi": float(run.pct_unresolved.max())}


def unresolved_peak(ya, months, min_n=MIN_N):
    """The year opened with the highest share of complaints still open (among years with >= min_n complaints), set
    against the other such years: its count, its share of all unresolved complaints, the agencies that hold most of
    it and, if the year stands out (ELEVATED_X), a short block of months that holds most of it. None without
    complaints."""
    y = ya.groupby("year")[["n_complaints", "n_unresolved"]].sum()
    y = y[y.n_complaints >= min_n]
    if y.empty:
        return None
    y["pct"] = 100 * y.n_unresolved / y.n_complaints
    total = int(ya.n_unresolved.sum())
    peak = int(y.sort_values(["pct", "n_unresolved"], ascending=False).index[0])
    others = y.drop(index=peak)
    typical = float(others.pct.median()) if len(others) else float("nan")
    p = y.loc[peak]
    stands_out = bool(len(others)) and p.pct > ELEVATED_X * typical
    a = ya[(ya.year == peak) & (ya.n_unresolved > 0)].sort_values(["n_unresolved", "agency"], ascending=[False, True])
    a = a.assign(pct_unresolved=100 * a.n_unresolved / a.n_complaints,
                 pct_of_year=100 * a.n_unresolved / max(int(p.n_unresolved), 1))
    k = int((a.pct_of_year.cumsum().shift(fill_value=0) < UNRESOLVED_COVER_PCT).sum())
    agencies = a.head(min(max(k, 1), UNRESOLVED_MAX_AGENCIES))
    return {"years": y, "total": total, "peak": peak, "n": int(p.n_unresolved), "n_complaints": int(p.n_complaints),
            "pct": float(p.pct), "share_all": 100 * p.n_unresolved / total if total else 0.0,
            "others": len(others), "typical": typical,
            "lo": float(others.pct.min()) if len(others) else float("nan"),
            "hi": float(others.pct.max()) if len(others) else float("nan"),
            "is_last": peak == int(y.index.max()) and len(others) > 0, "stands_out": stands_out,
            "agencies": agencies[["agency", "n_complaints", "n_unresolved", "pct_unresolved", "pct_of_year"]],
            "block": _month_block(months, peak, typical, min_n) if stands_out else None}


# ---------- b) neighborhoods ----------

def mix_adjusted(con, rel, group_col, min_n=MIN_N):
    """Per group: raw median hours and the mix-adjusted difference from typical.

    Typical = the citywide mean of ln(hours + 1) for the same complaint type and year (all rows of `rel`, so
    rows without a group still set the benchmark). adj_pct = 100 * (exp(mean(ln(h + 1) - typical)) - 1): how much
    longer (or shorter) the group's requests take than requests of the same type and year citywide, as a
    geometric mean. Equivalent to indirect standardization with type x year fixed effects.
    """
    return con.sql(f"""
        with r as (select {group_col} as grp, coalesce(canonical_type, '') as ct,
                          coalesce(year(created), -1) as yr, duration_h, ln(duration_h + 1) as y
                   from {rel} where duration_h is not null),
        c as (select ct, yr, avg(y) as cell_y from r group by all),
        j as (select r.grp, r.duration_h, r.y, c.cell_y from r join c using (ct, yr))
        select grp, count(*) n, quantile_cont(duration_h, 0.5) median_h, avg(duration_h) mean_h,
               exp(avg(y)) - 1 as typical_h, exp(avg(cell_y)) - 1 as expected_h,
               100 * (exp(avg(y - cell_y)) - 1) as adj_pct
        from j where grp is not null group by grp having count(*) >= {int(min_n)} order by grp""").df()


def _unresolved_by(con, col):
    return con.sql(f"""select {col}, count(*) n_complaints, 100 * avg(unresolved::int) pct_unresolved
                       from q4 where {col} is not null group by 1""").df()


def district_table(con, names):
    d = mix_adjusted(con, "q4r", "boro_cd").rename(columns={"grp": "boro_cd"})
    d["boro_cd"] = d.boro_cd.astype(int)
    u = _unresolved_by(con, "boro_cd")
    u["boro_cd"] = u.boro_cd.astype(int)
    d = d.merge(u, on="boro_cd", how="left")
    d["district"] = [district_label(c, names) for c in d.boro_cd]
    d["short"] = [district_short(c, names) for c in d.boro_cd]
    d = d.sort_values(["median_h", "adj_pct"], ascending=False)
    d["rank_raw"] = np.arange(1, len(d) + 1)
    d = d.sort_values(["adj_pct", "median_h"], ascending=False)
    d["rank_adjusted"] = np.arange(1, len(d) + 1)
    d["rank_change"] = d.rank_raw - d.rank_adjusted
    return d.reset_index(drop=True)


def rank_compare(d):
    k = min(TOP_K, len(d))
    raw_top = list(d.nsmallest(k, "rank_raw").boro_cd)
    adj_top = list(d.nsmallest(k, "rank_adjusted").boro_cd)
    rho = float(np.corrcoef(d.rank_raw, d.rank_adjusted)[0, 1]) if len(d) > 2 else float("nan")
    movers = d.reindex(d.rank_change.abs().sort_values(ascending=False).index).head(3)
    return {"k": k, "spearman": rho, "only_raw": [c for c in raw_top if c not in adj_top],
            "only_adjusted": [c for c in adj_top if c not in raw_top], "movers": movers}


def borough_table(con):
    b = mix_adjusted(con, "q4r", "borough").rename(columns={"grp": "borough"})
    return b.merge(_unresolved_by(con, "borough"), on="borough", how="left").sort_values(
        "adj_pct", ascending=False).reset_index(drop=True)


def gap_drivers(con, rel, group_col, groups):
    """Each complaint type's contribution to a group's mix-adjusted gap (same benchmark as mix_adjusted).

    A group's log gap G is the mean of e = ln(h + 1) - typical over its closures, so G = sum over types of
    (n_t / N) * g_t, where g_t is the group's mean gap within type t. contrib_log = n_t * g_t / N (sums to G);
    share_of_gap_pct = 100 * contrib_log / G; type_adj_pct = 100 * (exp(g_t) - 1); adj_pct_without = the group's
    gap with type t left out. One row per (group, type), every type, ordered so the types that push the gap in the
    group's own direction come first.
    """
    con.register("q4_driver_groups", pd.DataFrame({"grp": list(groups)}))
    try:
        df = con.sql(f"""
            with r as (select {group_col} as grp, coalesce(canonical_type, '') as ct,
                              coalesce(year(created), -1) as yr, duration_h, ln(duration_h + 1) as y
                       from {rel} where duration_h is not null),
            c as (select ct, yr, avg(y) as cell_y from r group by all),
            j as (select r.grp, r.ct, r.duration_h, r.y - c.cell_y as e from r join c using (ct, yr)
                  where r.grp in (select grp from q4_driver_groups)),
            g as (select grp, count(*) as group_n, sum(e) as group_sum from j group by grp),
            t as (select grp, ct, count(*) as n, sum(e) as type_sum, quantile_cont(duration_h, 0.5) as median_h
                  from j group by all),
            m as (select ct, quantile_cont(duration_h, 0.5) as city_type_median_h from r
                  where ct in (select ct from t) group by ct)
            select t.grp, t.ct as canonical_type, t.n, g.group_n, t.median_h, m.city_type_median_h,
                   t.type_sum / t.n as type_gap, t.type_sum / g.group_n as contrib_log,
                   g.group_sum / g.group_n as group_gap, g.group_sum, t.type_sum
            from t join g using (grp) join m using (ct)""").df()
    finally:
        con.unregister("q4_driver_groups")
    df["canonical_type"] = df.canonical_type.replace("", "(no type)")
    df["pct_of_closures"] = 100 * df.n / df.group_n
    df["type_adj_pct"] = 100 * np.expm1(df.type_gap)
    df["group_adj_pct"] = 100 * np.expm1(df.group_gap)
    df["share_of_gap_pct"] = 100 * df.contrib_log / df.group_gap.where(df.group_gap != 0)
    rest_n = (df.group_n - df.n).where(df.group_n > df.n)
    df["adj_pct_without"] = 100 * np.expm1((df.group_sum - df.type_sum) / rest_n)
    df["_order"] = -np.sign(df.group_gap) * df.contrib_log
    df = df.sort_values(["grp", "_order", "canonical_type"]).reset_index(drop=True)
    return df[["grp", "canonical_type", "n", "group_n", "pct_of_closures", "median_h", "city_type_median_h",
               "type_adj_pct", "contrib_log", "share_of_gap_pct", "group_adj_pct", "adj_pct_without"]]


def district_drivers(con, d):
    """The DRIVER_TYPES complaint types that contribute most to the gap of the DRIVER_K slowest and fastest
    districts after adjustment (low_n: fewer than MIN_N closures of that type in the district)."""
    k = min(DRIVER_K, len(d) // 2)
    cols = ["boro_cd", "district", "side", "rank_adjusted", "canonical_type", "n", "group_n", "low_n",
            "pct_of_closures", "median_h", "city_type_median_h", "type_adj_pct", "contrib_log", "share_of_gap_pct",
            "group_adj_pct", "adj_pct_without"]
    if k == 0:
        return pd.DataFrame(columns=cols)
    order = d.sort_values(["adj_pct", "median_h"], ascending=False)
    side = {**{int(c): "slowest" for c in order.head(k).boro_cd}, **{int(c): "fastest" for c in order.tail(k).boro_cd}}
    dr = gap_drivers(con, "q4r", "boro_cd", list(side))
    dr["boro_cd"] = dr.grp.astype(int)
    dr = dr.groupby("boro_cd", sort=False).head(DRIVER_TYPES)
    dr["side"] = dr.boro_cd.map(side)
    lab = d.set_index("boro_cd")
    dr["district"] = dr.boro_cd.map(lab.district)
    dr["rank_adjusted"] = dr.boro_cd.map(lab.rank_adjusted)
    dr["low_n"] = dr.n < MIN_N
    dr["_s"] = -np.sign(dr.group_adj_pct) * dr.contrib_log  # biggest push in the district's own direction first
    return dr.sort_values(["rank_adjusted", "_s", "canonical_type"])[cols].reset_index(drop=True)


# ---------- c) agencies ----------

def regression_sample(con, cfg):
    """The response set (as q4r), reservoir-sampled with a fixed seed. Single-threaded so the sample repeats."""
    src = clean_glob(cfg)
    con.execute("set threads = 1")
    try:
        return con.sql(f"""
            select ln(duration_h + 1) as y, coalesce(canonical_type, '(none)') as canonical_type,
                   coalesce(cast(year(created) as varchar), '(none)') as year,
                   coalesce(borough, '(unknown)') as borough, coalesce(agency, '(none)') as agency
            from (select * from {src} where {COMPLAINTS} and duration_h is not null and not ({INSTANT}))
            using sample reservoir({SAMPLE_ROWS} rows) repeatable ({SEED})""").df()
    finally:
        con.execute(f"set threads = {int(cfg.threads)}")


def _dummies(df, cols):
    if not cols:
        return np.empty((len(df), 0)), []
    X = pd.get_dummies(df[cols].astype(str), drop_first=True, dtype=float)
    return X.to_numpy(), list(X.columns)


def variance_explained(df):
    """R^2 of ln(hours + 1) for a ladder of models. Complaint-type fixed effects are absorbed by demeaning within
    type (Frisch-Waugh-Lovell); year, borough and agency dummies are then fit with statsmodels OLS."""
    import statsmodels.api as sm

    y = df.y.to_numpy(float)
    sst = float(((y - y.mean()) ** 2).sum())
    codes = pd.factorize(df.canonical_type)[0]
    counts = np.bincount(codes)

    def demean(a):
        a = np.asarray(a, float)
        if a.ndim == 1:
            return a - (np.bincount(codes, weights=a) / counts)[codes]
        out = np.empty_like(a)
        for j in range(a.shape[1]):
            out[:, j] = a[:, j] - (np.bincount(codes, weights=a[:, j]) / counts)[codes]
        return out

    Xa, _ = _dummies(df, ["agency"])
    rows = [(AGENCY_ONLY, sm.OLS(y, sm.add_constant(Xa, has_constant="add")).fit().rsquared)]
    y_dm = demean(y)
    rows.append(("complaint type", 1 - float((y_dm ** 2).sum()) / sst))
    fits = {}
    for name, cols in [("type + year", ["year"]), ("type + year + borough", ["year", "borough"]),
                       ("type + year + borough + agency", ["year", "borough", "agency"])]:
        X, names = _dummies(df, cols)
        if X.shape[1]:
            with warnings.catch_warnings():  # agencies nested in their types make the design rank-deficient by design
                warnings.filterwarnings("ignore", message=".*rank-deficient.*")
                fit = sm.OLS(y_dm, demean(X)).fit()
            resid, params = np.asarray(fit.resid), pd.Series(np.asarray(fit.params), index=names)
        else:
            resid, params = y_dm, pd.Series(dtype=float)
        rows.append((name, 1 - float((resid ** 2).sum()) / sst))
        fits[name] = (resid, params)
    v = pd.DataFrame(rows, columns=["model", "r2"])
    v["r2_pct"] = 100 * v.r2
    v["n"] = len(df)
    v["n_types"] = len(counts)
    return v, fits


def year_effects(params, years):
    """% difference from the first year, same complaint type and borough (from the type + year + borough model)."""
    out = {}
    for y in years:
        key = f"year_{y}"
        out[str(y)] = 100 * (np.exp(params[key]) - 1) if key in params.index else 0.0
    return out


def agency_effects(df, resid, types=None):
    """For complaint types that two or more agencies each handle >= MIN_N_SHARED times in the sample (and, if `types`
    is given, only those types: the like-for-like ones): each agency's mean residual (ln scale) from the type + year
    + borough model, as % faster/slower than typical for the same type."""
    d = pd.DataFrame({"canonical_type": df.canonical_type.to_numpy(), "agency": df.agency.to_numpy(), "e": resid})
    if types is not None:
        d = d[d.canonical_type.isin(list(types))]
    cnt = d.groupby(["canonical_type", "agency"]).size().rename("n").reset_index()
    ok = cnt[cnt.n >= MIN_N_SHARED]
    shared = ok.groupby("canonical_type").agency.nunique()
    pairs = ok[ok.canonical_type.isin(shared[shared >= 2].index)][["canonical_type", "agency"]]
    s = d.merge(pairs, on=["canonical_type", "agency"])
    if s.empty:
        return pd.DataFrame(columns=["agency", "n_shared", "pct_vs_typical", "ci_low", "ci_high", "shared_types"])
    g = s.groupby("agency").e.agg(["size", "mean", "std"]).reset_index()
    se = g["std"] / np.sqrt(g["size"])
    types = pairs.groupby("agency").canonical_type.apply(lambda t: ", ".join(sorted(t)))
    out = pd.DataFrame({"agency": g.agency, "n_shared": g["size"],
                        "pct_vs_typical": 100 * (np.exp(g["mean"]) - 1),
                        "ci_low": 100 * (np.exp(g["mean"] - Z95 * se) - 1),
                        "ci_high": 100 * (np.exp(g["mean"] + Z95 * se) - 1)})
    out["shared_types"] = out.agency.map(types)
    return out.sort_values("pct_vs_typical").reset_index(drop=True)


def agency_coverage(con):
    """Share of each agency's closures (full response set) in complaint types another agency also handles."""
    return con.sql(f"""
        with a as (select coalesce(canonical_type, '(none)') ct, coalesce(agency, '(none)') agency, count(*) n
                   from q4r group by all),
        s as (select ct from a where n >= {MIN_N_SHARED} group by 1 having count(*) >= 2)
        select agency, sum(n)::bigint n, count(*) n_types,
               100 * sum(case when n >= {MIN_N_SHARED} and ct in (select ct from s) then n else 0 end) / sum(n)
                   as pct_volume_in_shared_types
        from a group by agency order by n desc""").df()


def descriptor_overlap(con, rel):
    """Is a complaint type that two or more agencies handle (>= MIN_N_SHARED closures each) the same work?

    `rel` has canonical_type, agency, descriptor (one row per closure). A descriptor is a regular part of an
    agency's work in a type if >= DESC_USED_PCT % of its closures there carry it. pct_desc_shared = share of the
    agency's closures whose descriptor is a regular descriptor of another of those agencies (NULL never matches).
    like_for_like: every agency in the type reaches LIKE_FOR_LIKE_PCT. top_descriptor: the agency's commonest one.
    """
    df = con.sql(f"""
        with r as (select coalesce(canonical_type, '(none)') ct, coalesce(agency, '(none)') agency, descriptor
                   from {rel}),
        a as (select ct, agency, count(*) n from r group by all having count(*) >= {MIN_N_SHARED}),
        s as (select ct from a group by ct having count(*) >= 2),
        d as (select r.ct, r.agency, r.descriptor, count(*) nd, any_value(a.n) n from r join a using (ct, agency)
              where r.ct in (select ct from s) group by all),
        u as (select ct, agency, descriptor from d where descriptor is not null and 100.0 * nd / n >= {DESC_USED_PCT}),
        m as (select distinct d.ct, d.agency, d.descriptor from d join u
              on u.ct = d.ct and u.descriptor = d.descriptor and u.agency <> d.agency)
        select d.ct as canonical_type, d.agency, any_value(d.n) n,
               100.0 * coalesce(sum(d.nd) filter (where m.ct is not null), 0) / any_value(d.n) pct_desc_shared,
               first(d.descriptor order by d.nd desc, d.descriptor nulls last) as top_descriptor,
               100.0 * max(d.nd) / any_value(d.n) as pct_top_descriptor
        from d left join m on m.ct = d.ct and m.agency = d.agency and m.descriptor is not distinct from d.descriptor
        group by d.ct, d.agency order by d.ct, n desc""").df()
    df["like_for_like"] = df.canonical_type.map(df.groupby("canonical_type").pct_desc_shared.min()
                                                >= LIKE_FOR_LIKE_PCT).astype(bool)
    return df


def within_type(con):
    """Direct comparison: agencies with >= MIN_N_SHARED closures in a complaint type another agency also handles."""
    return con.sql(f"""
        with r as (select coalesce(canonical_type, '(none)') ct, coalesce(agency, '(none)') agency, duration_h, year
                   from q4r),
        a as (select ct, agency, count(*) n, min(year) first_year, max(year) last_year,
                     quantile_cont(duration_h, 0.5) median_h, avg(ln(duration_h + 1)) ly
              from r group by all having count(*) >= {MIN_N_SHARED}),
        s as (select ct from a group by 1 having count(*) >= 2),
        t as (select ct, count(*) type_n, quantile_cont(duration_h, 0.5) type_median_h, avg(ln(duration_h + 1)) type_ly
              from r where ct in (select ct from s) group by 1)
        select a.ct as canonical_type, a.agency, a.n, a.first_year, a.last_year, a.median_h, t.type_median_h,
               a.median_h / nullif(t.type_median_h, 0) as ratio_to_type_median,
               100 * (exp(a.ly - t.type_ly) - 1) as pct_vs_type
        from a join t using (ct) order by t.type_n desc, a.n desc""").df()


# ---------- charts ----------

def _text(ax, x, y, s, color, ha="left", va="center", **kw):
    from nyc311 import nytplot as nyt
    return ax.text(x, y, s, color=color, ha=ha, va=va, fontfamily=nyt.SANS, **kw)


def _hbar_axes(ax, left=False):
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.tick_params(axis="y", labelcolor="#333333")


def side_labels(ypos, vals, span):
    """Where the series names go, as (key, x, y, ha): 'slower' right of zero in a row just above the slower bars,
    'faster' left of zero in a row just below the faster bars (the y axis is inverted), so each name sits on its
    own side of the zero line without crossing a bar or a value label."""
    ypos, vals = np.asarray(ypos, float), np.asarray(vals, float)
    out = []
    pos, neg = ypos[vals > 0], ypos[vals < 0]
    if len(pos):
        out.append(("slower", 0.02 * span, pos.min() - SIDE_LABEL_GAP, "left"))
    if len(neg):
        out.append(("faster", -0.02 * span, neg.max() + SIDE_LABEL_GAP, "right"))
    return out


def chart_districts(d, path, note):
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    k = min(TOP_K, len(d) // 2)
    ordered = d.sort_values("adj_pct", ascending=False)
    if len(d) > 2 * k:
        show = pd.concat([ordered.head(k), ordered.tail(k)])
        ypos = np.r_[np.arange(k), np.arange(k) + k + 1.2]
        hidden = len(d) - 2 * k
    else:
        show, ypos, hidden = ordered, np.arange(len(ordered)), 0
    fig = Figure(figsize=(8, 0.32 * (len(ypos) + 2) + 1.9))
    ax = fig.subplots()
    vals = show.adj_pct.to_numpy()
    colors = [nyt.CORAL.band if v > 0 else nyt.SAGE.band for v in vals]
    ax.barh(ypos, vals, color=colors, height=0.72, zorder=2)
    span = max(np.abs(vals).max(), 1)
    for y, v in zip(ypos, vals):
        hue = nyt.CORAL if v > 0 else nyt.SAGE
        _text(ax, v + (0.015 if v >= 0 else -0.015) * span, y, _pct(v), hue.line, ha="left" if v >= 0 else "right",
              fontsize=9, fontweight="bold")
    ax.set_yticks(ypos, show.short)
    ax.axvline(0, color="#333333", linewidth=1.0, zorder=3)
    ax.set_xlim(-span * 1.25 if vals.min() < 0 else 0, span * 1.25 if vals.max() > 0 else 0)
    _hbar_axes(ax)
    nyt.format_x(ax, suffix="%")
    if hidden:
        _text(ax, 0, k + 0.1, f"{hidden} districts in between", nyt.NOTE_GRAY, ha="center", fontsize=9,
              fontstyle="italic", zorder=4, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    # series names on their own side of the zero line, in a row of their own next to their group
    labels = side_labels(ypos, vals, span)
    names = {"slower": ("Slower than typical", nyt.CORAL), "faster": ("Faster than typical", nyt.SAGE)}
    for key, x, y, ha in labels:
        text, hue = names[key]
        _text(ax, x, y, text, hue.line, ha=ha, fontweight="bold", fontsize=12)
    keys = {key for key, *_ in labels}
    top = ypos.min() - (SIDE_LABEL_GAP + 0.55 if "slower" in keys else 0.6)
    bottom = ypos.max() + (SIDE_LABEL_GAP + 0.55 if "faster" in keys else 0.6)
    ax.set_ylim(bottom, top)  # inverted: the slowest district at the top
    nyt.title(fig, "Where 311 complaints take longer than typical to close")
    nyt.note(fig, note)
    return nyt.save(fig, path)


def chart_agencies(a, v, path, title, note, col="median_d_excl_instant"):
    """Top: median days to close for the agencies in `a` (as given: compared_agencies), slowest at the top.
    Bottom: the variance-explained ladder."""
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    a = a.sort_values([col, "agency"])
    fig = Figure(figsize=(7.5, 0.3 * (len(a) + len(v)) + 2.6))
    top, bot = fig.subplots(2, 1, gridspec_kw={"height_ratios": [len(a), len(v) + 0.5]})
    top.barh(np.arange(len(a)), a[col], color=nyt.SAGE.band, height=0.72, zorder=2)
    for i, d in enumerate(a[col]):
        _text(top, d + 0.01 * a[col].max(), i, fmt_h(d * 24), nyt.SAGE.line, fontsize=9, fontweight="bold")
    top.set_yticks(np.arange(len(a)), a.agency)
    top.set_xlim(0, a[col].max() * 1.22)
    _hbar_axes(top, left=True)
    nyt.format_x(top, suffix=" days")
    nyt.panel_title(top, "Median time to close, by agency (excluding instant closures)")

    order = [AGENCY_ONLY] + NESTED_MODELS
    v = v.set_index("model").loc[[m for m in order if m in set(v.model)]].reset_index()
    pct = 100 * v.r2.to_numpy()
    colors = [nyt.CORAL.band if m == AGENCY_ONLY else nyt.SAGE.band for m in v.model]
    bot.barh(np.arange(len(v)), pct, color=colors, height=0.72, zorder=2)
    for i, (m, p) in enumerate(zip(v.model, pct)):
        hue = nyt.CORAL if m == AGENCY_ONLY else nyt.SAGE
        _text(bot, p + 1, i, f"{p:.0f}%", hue.line, fontsize=9, fontweight="bold")
    bot.set_yticks(np.arange(len(v)), [MODEL_LABELS.get(m, m) for m in v.model])
    bot.invert_yaxis()
    bot.set_xlim(0, 100)
    _hbar_axes(bot, left=True)
    nyt.format_x(bot, suffix="%", step=25)
    nyt.panel_title(bot, "Share of the variation in time to close explained")
    nyt.title(fig, title)
    nyt.note(fig, note)
    return nyt.save(fig, path)


# ---------- run ----------

def run(cfg):
    from nyc311.q05_top_types import data_period, period_label, period_long

    con = cfg.connect()
    src, excl = _load(con, cfg)
    d0, d1 = data_period(con, src)
    period, period_l = period_label(d0, d1), period_long(d0, d1)
    names = district_names()

    summary = summary_stats(con)
    n_dur = int(summary.set_index("set").loc["all closed complaints", "n"])
    f = fences(con)
    tout = type_outliers(con)
    outl = outliers(con, f, n_dur, tout)
    nod = no_duration_reasons(con)
    agencies = by_agency(con)
    shown = compared_agencies(agencies)
    full = clean.full_years(con, src)
    years = by_year(con, full)
    unres_ya = unresolved_by_year_agency(con)
    unres_months = unresolved_by_month(con)
    unres_peak = unresolved_peak(unres_ya, unres_months)
    precision = con.sql("""select 100 * avg((date_only_created or date_only_closed)::int),
                           100 * avg((date_only_created and date_only_closed)::int)
                           from q4 where duration_h is not null""").fetchone()
    n_instant = con.sql(f"select count(*) from q4 where duration_h is not null and {INSTANT}").fetchone()[0]
    n_resp = con.sql("select count(*) from q4r").fetchone()[0]
    instant_by_agency = con.sql(f"""select agency, count(*) n from q4 where duration_h is not null and {INSTANT}
                                    group by 1 order by n desc limit 4""").df()

    districts = district_table(con, names)
    drivers = district_drivers(con, districts)
    cmp = rank_compare(districts)
    boroughs = borough_table(con)
    n_no_cd = con.sql("select count(*) from q4r where boro_cd is null").fetchone()[0]

    sample = regression_sample(con, cfg)
    variance, fits = variance_explained(sample)
    resid, params = fits["type + year + borough"]
    yeff = year_effects(params, sorted(sample.year.unique()))
    sample_year_n = sample.year.value_counts()
    years["adj_pct_vs_first"] = [yeff.get(str(y)) if sample_year_n.get(str(y), 0) >= MIN_N else np.nan
                                 for y in years.year]
    overlap = descriptor_overlap(con, f"(select canonical_type, agency, descriptor from {src} where {COMPLAINTS} "
                                      f"and duration_h is not null and not ({INSTANT}))")
    effects = agency_effects(sample, resid, types=set(overlap.canonical_type[overlap.like_for_like]))
    coverage = agency_coverage(con)
    wt = within_type(con).merge(overlap[["canonical_type", "agency", "pct_desc_shared", "top_descriptor",
                                         "pct_top_descriptor", "like_for_like"]],
                                on=["canonical_type", "agency"], how="left")

    res = {"summary": summary, "fences": f, "outliers": outl, "type_outliers": tout, "no_duration": nod,
           "by_agency": agencies, "by_year": years, "districts": districts, "rank_compare": cmp,
           "boroughs": boroughs, "variance": variance, "agency_effects": effects, "agency_coverage": coverage,
           "within_type": wt, "n_response_set": int(n_resp), "excl": excl, "period": period,
           "period_long": period_l, "precision": precision, "n_instant": int(n_instant),
           "instant_by_agency": instant_by_agency, "n_no_cd": int(n_no_cd), "agency_chart": shown,
           "district_drivers": drivers, "full_years": full, "unresolved_by_year_agency": unres_ya,
           "unresolved_by_month": unres_months, "unresolved_peak": unres_peak}

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    for name in ["summary", "outliers", "type_outliers", "no_duration", "by_agency", "by_year", "boroughs",
                 "variance", "agency_effects", "agency_coverage", "within_type", "district_drivers",
                 "unresolved_by_year_agency", "unresolved_by_month"]:
        res[name].to_csv(agg / f"q04_{name}.csv", index=False)
    districts.drop(columns="short").to_csv(agg / "q04_districts.csv", index=False)

    figs = cfg.out_dir / "figs"
    chart_districts(districts, figs / "q04_district_adjusted.png", _district_note(res))
    title, note = _agency_chart_text(res)
    chart_agencies(shown, variance, figs / "q04_agency_variance.png", title, note)
    _report(cfg, res)
    _write_answer(cfg, res)
    return res


def _r2(res, model):
    return float(res["variance"].set_index("model").loc[model, "r2"])


def _district_note(res):
    d = res["districts"]
    k = min(TOP_K, len(d) // 2)
    shown = f"The {k} slowest and {k} fastest" if len(d) > 2 * k else "All"
    return (f"Note: Difference from the citywide typical time to close (geometric mean) for the same complaint type and "
            f"year. {shown} of {len(d)} community\ndistricts with at least {MIN_N} closed complaints, {res['period']}. "
            f"Complaints only; excludes {res['n_instant']:,} instant (0-second) closures.")


def _agency_chart_text(res):
    r2a, r2t = _r2(res, AGENCY_ONLY), _r2(res, "complaint type")
    gain = _r2(res, NESTED_MODELS[-1]) - _r2(res, NESTED_MODELS[-2])
    if r2t >= r2a:
        title = f"Complaint type explains more of the gap than agency does: {r2t:.0%} vs. {r2a:.0%}"
    else:
        title = f"Agency explains more of the variation than complaint type: {r2a:.0%} vs. {r2t:.0%}"
    n = int(res["variance"].n.iloc[0])
    shown, a = res["agency_chart"], res["by_agency"]
    if len(shown) < len(a):
        who = (f"the {len(shown)} agencies with at least {MIN_AGENCY_SHARE:g}% of closures each "
               f"({len(a) - len(shown)} smaller ones not shown)")
    else:
        who = f"agencies with at least {MIN_N} closures"
    note = (f"Note: Closed complaints, {res['period']}, without {res['n_instant']:,} instant (0-second) closures; "
            f"{who}.\nVariation = R\N{SUPERSCRIPT TWO} of log(hours + 1) on a fixed-seed sample of {n:,}; adding "
            f"agency to complaint type, year and borough adds {100 * gain:.1f} points.")
    return title, note


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    e, s = res["excl"], res["summary"]
    rep.add(f"Q4. Time to closure on {e.n_complaints:,} complaints ({res['period_long']})",
            f"rows {e.n_rows:,}; left out: is_test {e.n_test:,}, dup_key {e.n_dup:,}, is_info_request {e.n_info:,}, "
            f"category '{rules.INVALID_CATEGORY}' {e.n_invalid:,}",
            f"instant closures (rules.INSTANT_SQL: 0 s, with a time of day) {res['n_instant']:,}; response set "
            f"(closed, excluding instant closures; used in b) and c)) {res['n_response_set']:,}")
    rep.table(s[["set", "n", "mean_h", "sd_h", "p10_h", "p25_h", "median_h", "p75_h", "p90_h", "p99_h", "median_d",
                 "mean_d"]], title="a) Time to closure, hours (days at right)")
    f = res["fences"]
    rep.add(f"citywide Tukey fences on ln(hours): Q1 {f['q1_h']:.1f} h, Q3 {f['q3_h']:.1f} h -> "
            f"{f['lower_h']:.2f} h to {f['upper_h']:.0f} h; durations with a midnight timestamp (day precision): "
            f"{res['precision'][0]:.1f}%")
    rep.table(res["outliers"], title="Outliers and suspicious values (pct of closed complaints)")
    rep.table(res["type_outliers"][["canonical_type", "agency", "n", "lower_h", "median_h", "upper_h", "n_below",
                                    "n_above", "pct_outside"]], max_rows=5,
              title="Within-type outliers (own Tukey fences), most above-fence first")
    rep.table(res["no_duration"], max_rows=8, title="Complaints without a duration (closed-only stats are biased)")
    rep.table(res["by_agency"][["agency", "n", "share_pct", "pct_unresolved", "mean_d", "median_d",
                                "median_d_excl_instant", "p90_d", "pct_instant", "pct_day_precision"]], max_rows=12,
              title=f"By agency (>= {MIN_N} closed), days; charted: share_pct >= {MIN_AGENCY_SHARE:g}% of the "
                    f"response set ({len(res['agency_chart'])} agencies)")
    rep.table(res["by_year"], max_rows=13, title="By year opened (adj_pct_vs_first: same type and borough, sample)")
    u = res["unresolved_peak"]
    if u is not None:
        b = u["block"]
        rep.add(f"unresolved by year opened: highest {u['peak']} {u['pct']:.2f}% ({u['n']:,}, {u['share_all']:.1f}% of "
                f"all {u['total']:,} unresolved); " + (f"other {u['others']} years {u['lo']:.2f}-{u['hi']:.2f}% "
                                                       f"(median {u['typical']:.2f}%)" if u["others"] else
                                                       "no other year to compare")
                + (f"; block {b['label']}: {b['n']:,} ({b['pct_of_year']:.1f}% of the year's)" if b else ""))
        rep.table(u["agencies"].head(UNRESOLVED_MAX_AGENCIES),
                  title=f"Unresolved complaints opened in {u['peak']}, by agency (q04_unresolved_by_year_agency.csv)")

    rep.table(res["boroughs"][["borough", "n", "median_h", "adj_pct", "pct_unresolved"]],
              title="b) Boroughs: raw median hours and % vs typical for the same type and year")
    d, cmp = res["districts"], res["rank_compare"]
    rep.add(f"{len(d)} districts with >= {MIN_N} closed complaints; {res['n_no_cd']:,} response-set rows have no "
            f"standard district (special areas or unlocated)")
    rep.table(d.sort_values("rank_adjusted")[["district", "n", "median_h", "rank_raw", "adj_pct", "rank_adjusted",
                                              "pct_unresolved"]], max_rows=TOP_K,
              title=f"Slowest {TOP_K} districts after adjustment (rank_raw = rank by raw median)")
    lab = dict(zip(d.boro_cd, d.district))
    dr = res["district_drivers"]
    if len(dr):
        rep.table(dr.groupby("boro_cd", sort=False).head(2)[
            ["district", "group_adj_pct", "canonical_type", "n", "pct_of_closures", "type_adj_pct", "share_of_gap_pct",
             "adj_pct_without"]], max_rows=4 * DRIVER_K,
            title=f"What drives the {DRIVER_K} slowest and fastest districts' gaps (top 2 types each)")
    rep.add(f"rank correlation raw vs adjusted: {cmp['spearman']:.2f}",
            f"in raw top {cmp['k']} only: " + ("; ".join(lab[c] for c in cmp["only_raw"]) or "none"),
            f"in adjusted top {cmp['k']} only: " + ("; ".join(lab[c] for c in cmp["only_adjusted"]) or "none"))

    rep.table(res["variance"], title="c) Variance in ln(hours + 1) explained (statsmodels, fixed-seed sample)")
    cov = res["coverage_summary"] = _coverage_summary(res)
    rep.add(f"agencies with a complaint type another agency also handles: {cov['n_shared']} of {cov['n_agencies']} "
            f"({cov['pct_closures']:.1f}% of closures are in shared types)")
    rep.table(res["agency_effects"], max_rows=10,
              title="Agency vs typical for the same complaint type, year and borough (like-for-like types only, "
                    "sample)")
    rep.table(res["within_type"][["canonical_type", "agency", "n", "median_h", "pct_vs_type", "top_descriptor",
                                  "pct_desc_shared", "like_for_like"]], max_rows=10,
              title=f"Same complaint type, different agencies (like_for_like: each >= {LIKE_FOR_LIKE_PCT}% of "
                    f"closures in descriptors the other also uses)")
    rep.emit()


def _coverage_summary(res):
    cov = res["agency_coverage"]
    shared_n = (cov.n * cov.pct_volume_in_shared_types / 100).sum()
    return {"n_agencies": len(cov), "n_shared": int((cov.pct_volume_in_shared_types > 0).sum()),
            "pct_closures": 100 * shared_n / cov.n.sum() if cov.n.sum() else 0.0}


def _agency_who(shown):
    if (shown.share_pct >= MIN_AGENCY_SHARE).all():
        return (f"the {len(shown)} agencies that each handle at least {MIN_AGENCY_SHARE:g}% of closures (together "
                f"{shown.share_pct.sum():.1f}%)")
    return f"the {len(shown)} agencies with at least {MIN_N} closures"


def _agency_line(res):
    a, shown, col = res["by_agency"], res["agency_chart"], "median_d_excl_instant"
    fast, slow = shown.iloc[0], shown.iloc[-1]
    lo, hi = shown.sort_values("pct_unresolved").iloc[0], shown.sort_values("pct_unresolved").iloc[-1]
    small = a[~a.agency.isin(shown.agency)].sort_values([col, "agency"])
    line = (f"- **By agency** (median excluding instant closures; chart in Q4c). Among {_agency_who(shown)}, the "
            f"fastest is {fast.agency} at {fmt_h(24 * fast[col])} (n {int(fast.n_excl_instant):,}) and the slowest "
            f"{slow.agency} at {fmt_h(24 * slow[col])} (n {int(slow.n_excl_instant):,}). Among the same agencies, "
            f"the share still open ranges from {lo.pct_unresolved:.1f}% ({lo.agency}) to {hi.pct_unresolved:.1f}% "
            f"({hi.agency}).")
    if len(small):
        meds = (f"its median is {fmt_h(24 * small[col].iloc[0])}" if len(small) == 1 else
                f"their medians run from {fmt_h(24 * small[col].iloc[0])} ({small.agency.iloc[0]}) to "
                f"{fmt_h(24 * small[col].iloc[-1])} ({small.agency.iloc[-1]})")
        line += (f" The other {len(small)} agencies with at least {MIN_N} closures ("
                 f"{int(small.n_excl_instant.sum()):,} closures, {small.share_pct.sum():.1f}%) are too small to rank "
                 f"with them and are not charted; {meds}.")
    return line


def _agency_counts(a):
    """'X 948 (94%), Y 40 (4%) and Z 20 (2%)' from the agencies table of unresolved_peak."""
    parts = [f"{r.agency} {int(r.n_unresolved):,} ({r.pct_of_year:.0f}%)" for r in a.itertuples()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _pct_range(lo, hi):
    """'1.1% to 2.5%', or '1.0%' when both ends print the same."""
    a, b = f"{lo:.1f}%", f"{hi:.1f}%"
    return a if a == b else f"{a} to {b}"


def _unresolved_line(u):
    """The findings bullet on unresolved complaints by year opened (from unresolved_peak)."""
    head = "- **Unresolved by year opened.** "
    a = u["agencies"]
    held = (f" By agency (share of {u['peak']}'s unresolved): {_agency_counts(a)}"
            + (f", together {a.pct_of_year.sum():.0f}%" if len(a) > 1 else "") + ".") if len(a) else ""
    ref = " Counts by year and agency: `q04_unresolved_by_year_agency.csv`; by month: `q04_unresolved_by_month.csv`."
    peak = u["peak"]
    if not u["others"]:
        return (head + f"Only {peak} has at least {MIN_N} complaints, so years cannot be compared; {u['n']:,} of its "
                f"complaints ({u['pct']:.1f}%) are still open." + held + ref)
    if not u["stands_out"]:
        return (head + f"The share still open is highest for {peak}, {u['pct']:.1f}% ({u['n']:,} complaints, "
                f"{u['share_all']:.0f}% of all {u['total']:,} unresolved), against {_pct_range(u['lo'], u['hi'])} "
                f"in the other {u['others']} years; no year is {ELEVATED_X} times the median of the others "
                f"({u['typical']:.1f}%)." + held + ref)
    txt = (head + f"{peak} stands out: {u['pct']:.1f}% of its complaints are still open ({u['n']:,}), against "
           f"{_pct_range(u['lo'], u['hi'])} in the other {u['others']} years with at least {MIN_N} complaints "
           f"(median {u['typical']:.1f}%), so {peak} alone holds {u['share_all']:.0f}% of all {u['total']:,} "
           "unresolved complaints." + held)
    b = u["block"]
    if b:
        txt += (f" Most were opened in {b['label']}: {b['n']:,} unresolved complaints"
                + (f" ({b['pct_of_year']:.0f}% of {peak}'s)" if b["n"] == b["n_in_year"] else
                   f" ({b['n_in_year']:,} of them in {peak}, {b['pct_of_year']:.0f}% of its total)")
                + f", with {_pct_range(b['rate_lo'], b['rate_hi'])} of each month's complaints still open.")
    if u["is_last"]:
        txt += (f" {peak} is the most recent year, so some of these may still be in progress; the data cannot show "
                "why they are open.")
    else:
        txt += (" These are likely records that were never closed out rather than requests still being worked on, "
                "but the data cannot show why; they are missing from that year's closed-only statistics.")
    return txt + ref


def _year_line(res):
    y = res["by_year"]
    big = y[y.n >= MIN_N]
    full = big[big.full_year]
    ends = full if len(full) >= 2 else big
    if len(ends) < 2:
        only = f" ({int(big.year.iloc[0])})" if len(big) else ""
        return (f"- **By year.** Only {len(big)} year{only} has at least {MIN_N} closed complaints, so the data "
                "cannot show a trend.")

    def rel(r1, r0):
        a1, a0 = r1.adj_pct_vs_first, r0.adj_pct_vs_first
        return 100 * ((1 + a1 / 100) / (1 + a0 / 100) - 1) if np.isfinite(a1) and np.isfinite(a0) else np.nan

    y0, y1 = ends.iloc[0], ends.iloc[-1]
    kind = "full calendar years, " if ends is full else ""
    line = (f"- **By year** ({kind}median excluding instant closures). It went from {fmt_h(y0.median_h_excl_instant)} "
            f"({int(y0.year)}) to {fmt_h(y1.median_h_excl_instant)} ({int(y1.year)})")
    r = rel(y1, y0)
    if np.isfinite(r):
        line += f"; holding complaint type and borough fixed, {int(y1.year)} closures took {_pct(r)} vs. {int(y0.year)}"
    pre = ends[ends.year == rules.PROFILE_LAST_YEAR]
    if len(pre) and int(y0.year) < rules.PROFILE_LAST_YEAR < int(y1.year):
        p = pre.iloc[0]
        rp = rel(p, y0)
        line += (f". To {rules.PROFILE_LAST_YEAR}, the last year before the pandemic, the median went to "
                 f"{fmt_h(p.median_h_excl_instant)}" + (f" ({_pct(rp)} for the same type and borough)"
                                                         if np.isfinite(rp) else ""))
    line += "."
    last, earlier = big.iloc[-1], big.iloc[:-1]
    m = earlier.pct_unresolved.median()
    part = "" if last.full_year else " (a partial year in these data)"
    if last.pct_unresolved > m:
        line += (f" {int(last.year)}{part} is {last.pct_unresolved:.1f}% unresolved vs. a median of {m:.1f}% for "
                 "earlier years: its requests have had less time to close, so its closed-only median is likely "
                 "biased down.")
    else:
        line += (f" {int(last.year)}{part} is {last.pct_unresolved:.1f}% unresolved, no more than earlier years "
                 f"(median {m:.1f}%), so recent requests are not disproportionately missing from its closed-only "
                 "median.")
    return line


def _driver_line(res):
    dr, d = res["district_drivers"], res["districts"].set_index("boro_cd")
    if dr.empty:
        return None
    city = res["summary"].set_index("set").loc[RESPONSE_SET, "median_h"]
    parts = []
    for side, pick in [("slowest", 0), ("fastest", -1)]:
        cds = dr[dr.side == side].sort_values("rank_adjusted").boro_cd.unique()
        if not len(cds):
            continue
        cd = cds[pick]
        r = d.loc[cd]
        rel = "below" if r.median_h < city else "above"
        head = (f"{r.district}, {_pct(r.adj_pct)} (raw median {fmt_h(r.median_h)}, {rel} the median of all closures, "
                f"{fmt_h(city)})")
        g = dr[(dr.boro_cd == cd) & ~dr.low_n]
        if g.empty:
            parts.append(f"{head}: no complaint type has {MIN_N} closures there")
            continue
        t1 = g.iloc[0]
        txt = (f"{head}: {t1.canonical_type} accounts for {t1.share_of_gap_pct:.0f}% of the gap ({int(t1.n):,} "
               f"closures, {t1.pct_of_closures:.0f}% of the district's, {_pct(t1.type_adj_pct)} vs. typical for the "
               f"same type and year); without it the gap would be {_pct(t1.adj_pct_without)}")
        if len(g) > 1:
            t2 = g.iloc[1]
            txt += f"; next is {t2.canonical_type} ({t2.share_of_gap_pct:.0f}%)"
        parts.append(txt)
    return ("- **What drives the extremes.** " + ". ".join(parts) + ". A type's share is its part of the district's "
            "mix-adjusted gap on the log scale (shares above 100% mean other types pull the other way). The "
            f"{DRIVER_TYPES} largest contributors for each of the {min(DRIVER_K, len(d) // 2)} slowest and fastest "
            "districts are in `q04_district_drivers.csv`.")


def _peer_lines(res):
    """Bullets on complaint types two agencies share: a speed comparison only where the work is like-for-like."""
    wt = res["within_type"]
    like_mask = wt.like_for_like.fillna(False).astype(bool) if len(wt) else pd.Series(dtype=bool)
    like, unlike = wt[like_mask], wt[~like_mask]

    def who(r):
        desc = f"'{r.top_descriptor}'" if isinstance(r.top_descriptor, str) else "(no descriptor)"
        return f"{r.agency} {desc}"

    def types(rows, k=4):
        groups = list(rows.groupby("canonical_type", sort=False))
        out = "; ".join(f"{t}: " + ", ".join(f"{who(r)} {fmt_h(r.median_h)} (n {r.n:,}, {_years(r)}; "
                                              f"{r.pct_desc_shared:.0f}% under shared descriptors)"
                                              for r in g.itertuples()) for t, g in groups[:k])
        rest = [t for t, _ in groups[k:]]
        return out + (f"; also {', '.join(rest)}" if rest else "")

    lines = []
    if len(like):
        lines.append(f"- **Direct comparison, like-for-like types:** {types(like)}. Check the years: an agency that "
                     "handled a type only in some years (renamed agencies, transferred programs) is partly a time "
                     "comparison.")
        if len(unlike):
            lines.append(f"- **Same name, different work (not a benchmark):** {types(unlike)}.")
    elif len(wt):
        k = unlike.canonical_type.nunique()
        where = ("In the one complaint type that two agencies handle" if k == 1 else
                 f"In each of the {k} complaint types that two agencies handle")
        lines.append(f"- **No like-for-like agency comparison.** {where}, at least one agency logs mostly descriptors "
                     "the other does not use, so they look like different services under one name: "
                     f"{types(unlike)}. The data "
                     "therefore offer no fair agency-vs-agency speed benchmark; the regression above (what agency "
                     "adds once complaint type is known) is the best available test of whether agencies differ "
                     "beyond their complaint mix.")
    else:
        lines.append(f"- **No like-for-like agency comparison.** No complaint type is handled by two agencies with at "
                     f"least {MIN_N_SHARED} closures each.")
    return {"n_types": int(wt.canonical_type.nunique()), "n_like": int(like.canonical_type.nunique()),
            "lines": lines}


def _write_answer(cfg, res):
    e, s = res["excl"], res["summary"].set_index("set")
    al = s.loc["all closed complaints"]
    rs = s.loc[RESPONSE_SET] if RESPONSE_SET in s.index else al
    exact = s.loc[STAT_SETS[2][0]] if STAT_SETS[2][0] in s.index else None
    o = res["outliers"].set_index("kind")
    f = res["fences"]
    a, shown = res["by_agency"], res["agency_chart"]
    d, cmp, b = res["districts"], res["rank_compare"], res["boroughs"]
    v = res["variance"].set_index("model").r2
    eff, wt = res["agency_effects"], res["within_type"]
    cov = res.get("coverage_summary") or _coverage_summary(res)
    nod = res["no_duration"].set_index("reason")
    n_open = int(nod.n.get("still open (unresolved)", 0))
    lab = dict(zip(d.boro_cd, d.district))
    tout = res["type_outliers"]
    top_out = tout.sort_values("pct_outside", ascending=False).head(3)
    inst = res["instant_by_agency"]
    col = "median_d_excl_instant"
    fast, slow = shown.iloc[0], shown.iloc[-1]
    hpd = a.set_index("agency").loc["HPD"] if "HPD" in set(a.agency) else None

    raw_slow = d.sort_values("rank_raw").head(3)
    adj_slow = d.sort_values("rank_adjusted").head(3)
    adj_fast = d.sort_values("rank_adjusted").tail(3).iloc[::-1]
    sig_fast = eff[eff.ci_high < 0]
    sig_slow = eff[eff.ci_low > 0]
    gain = v[NESTED_MODELS[-1]] - v[NESTED_MODELS[-2]]

    def dlist(rows, how):
        return "; ".join(f"{r.district} {_days(r.median_h) if how == 'raw' else _pct(r.adj_pct)}"
                         for r in rows.itertuples())

    def elist(rows):
        return "; ".join(f"{r.agency} {_pct(r.pct_vs_typical)} (95% CI {_pct(r.ci_low)} to {_pct(r.ci_high)}, "
                         f"n {r.n_shared:,}; {r.shared_types})" for r in rows.itertuples()) or "none"

    drivers = _driver_line(res)
    peer = _peer_lines(res)
    lines = [
        "# Q4. Response times",
        "",
        f"*Computed by `nyc311/q04_response_times.py` from {e.n_complaints:,} complaints, {res['period_long']}.*",
        "",
        "## Q4a. Summary statistics and outliers",
        "",
        f"- **What is measured.** The data have no first-response time, so the KPI is time to closure (closed minus "
        f"opened), defined only for status Closed with a close date on or after the opening. {int(al.n):,} of "
        f"{e.n_complaints:,} complaints ({100 * al.n / e.n_complaints:.1f}%) have one. Left out: "
        f"{e.n_info:,} information requests (`is_info_request`), {e.n_invalid:,} invalid entries (category "
        f"`{rules.INVALID_CATEGORY}`), {e.n_test:,} test records (`is_test`), {e.n_dup:,} duplicate keys (`dup_key`).",
        f"- **Summary.** Mean {fmt_h(al.mean_h)}, median {fmt_h(al.median_h)}, standard deviation {fmt_h(al.sd_h)}; "
        f"10th percentile {fmt_h(al.p10_h)}, 25th {fmt_h(al.p25_h)}, 75th {fmt_h(al.p75_h)}, 90th {fmt_h(al.p90_h)}, "
        f"99th {fmt_h(al.p99_h)}. The mean is {al.mean_h / al.median_h:.1f} times the median: durations are heavily "
        "right-skewed, so the dashboard should report medians and percentiles, not averages.",
        f"- **Zero-hour closures.** {int(o.n.iloc[0]):,} closures ({o.pct.iloc[0]:.1f}%) are at the second of "
        f"opening: {int(o.n.iloc[2]):,} with a time of day (instant closures, likely administrative; mostly "
        + ", ".join(f"{r.agency} {r.n:,}" for r in inst.itertuples()) +
        f") and {int(o.n.iloc[1]):,} same-day closures recorded as dates only, which stay in because they may be "
        f"real same-day closures. Without the instant closures the median is {fmt_h(rs.median_h)} (mean "
        f"{fmt_h(rs.mean_h)}, n {int(rs.n):,}); every figure below marked \"excluding instant closures\", and all "
        "of Q4b and Q4c, use this set.",
        f"- **Outliers.** Tukey fences on log hours (1.5 x IQR) flag {int(o.n.iloc[3]):,} closures under "
        f"{fmt_h(f['lower_h'])} and {int(o.n.iloc[4]):,} over {fmt_h(f['upper_h'])} "
        f"({o.pct.iloc[3] + o.pct.iloc[4]:.1f}% together); {int(o.n.iloc[6]):,} took more than a year. Judged "
        f"against each complaint type's own fences, {int(o.n.iloc[5]):,} ({o.pct.iloc[5]:.1f}%) are outliers; the "
        "highest shares are in " + "; ".join(f"{r.canonical_type} ({r.agency}, {r.pct_outside:.1f}% of {r.n:,})"
                                             for r in top_out.itertuples()) + ".",
        f"- **Precision.** {res['precision'][0]:.1f}% of durations have an opening or closing time of exactly "
        f"midnight, so they are precise only to the day"
        + (f"; for HPD it is {hpd.pct_day_precision:.1f}%, so HPD's median of {hpd.median_d:.1f} days is a whole "
           "number of days" if hpd is not None else "")
        + (f". Using only durations with exact timestamps, excluding instant closures ({int(exact.n):,}), the median "
           f"is {fmt_h(exact.median_h)}." if exact is not None else "."),
        f"- **Unresolved requests bias closed-only numbers.** {n_open:,} complaints "
        f"({100 * n_open / e.n_complaints:.1f}%) are still open (`unresolved`) and "
        f"{e.n_complaints - int(al.n) - n_open:,} more have no usable close date (an open status with a close date, "
        "closed before opened, no date, other status). They are missing from every duration statistic; a request "
        "still open will take at least as long as it has already waited, so closed-only medians likely understate "
        "the true time to closure.",
        *([_unresolved_line(res["unresolved_peak"])] if res.get("unresolved_peak") is not None else []),
        _agency_line(res),
        _year_line(res),
        "",
        "## Q4b. Neighborhoods with longer than typical times",
        "",
        f"- **Method.** {len(d)} community districts with at least {MIN_N} closed complaints (special areas such as "
        f"parks and airports, and unlocated requests: {res['n_no_cd']:,} closures, are left out). Raw ranking: median "
        "time to close. Adjusted ranking: each closure is compared with the citywide typical time (geometric mean of "
        "hours + 1) for the same complaint type and year, and the district's average gap is reported as % longer or "
        "shorter. Instant closures are left out.",
        f"- **Raw medians** put {dlist(raw_slow, 'raw')} slowest.",
        f"- **After adjusting for complaint mix,** the slowest are {dlist(adj_slow, 'adj')}; the fastest are "
        f"{dlist(adj_fast, 'adj')}. Districts' adjusted gaps range from {_pct(d.adj_pct.min())} to "
        f"{_pct(d.adj_pct.max())}.",
        *([drivers] if drivers else []),
        f"- **Where the rankings disagree.** Rank correlation {cmp['spearman']:.2f}. In the raw top {cmp['k']} only: "
        + ("; ".join(lab[c] for c in cmp["only_raw"]) or "none") + f". In the adjusted top {cmp['k']} only: "
        + ("; ".join(lab[c] for c in cmp["only_adjusted"]) or "none") + ". Districts that are slow only on raw "
        "medians receive slower kinds of complaints; districts that are slow only after adjustment wait longer than "
        "others for the same kind of complaint, which is where a promise of improvement should focus.",
        "- **Boroughs** (adjusted gap; raw median): " + "; ".join(
            f"{r.borough.title()} {_pct(r.adj_pct)} ({_days(r.median_h)}, {r.pct_unresolved:.1f}% unresolved)"
            for r in b.itertuples()) + ".",
        "",
        "![Community districts where complaints take longer or shorter than typical for the same complaint type and "
        "year](figs/q04_district_adjusted.png)",
        "",
        "## Q4c. Agencies: faster, or easier complaints?",
        "",
        f"- **Raw gap.** Among {_agency_who(shown)}, median time to close (excluding instant closures) ranges from "
        f"{fmt_h(24 * fast[col])} ({fast.agency}) to {fmt_h(24 * slow[col])} ({slow.agency}).",
        f"- **Complaint type explains {'more' if v['complaint type'] >= v[AGENCY_ONLY] else 'less'} than agency.** "
        f"On a fixed-seed sample of {int(res['variance'].n.iloc[0]):,} closures, a regression of log(hours + 1) on "
        f"agency alone explains {v[AGENCY_ONLY]:.1%} of the variation; complaint type alone explains "
        f"{v['complaint type']:.1%}; type + year + borough {v[NESTED_MODELS[2]]:.1%}; adding agency raises it to "
        f"{v[NESTED_MODELS[3]]:.1%} (+{100 * gain:.1f} points). Most of what looks like agency speed is what each "
        "agency is asked to handle.",
        "",
        "![Median time to close by agency, and how much of the variation complaint type and agency "
        "explain](figs/q04_agency_variance.png)",
        "",
        f"- **Most agencies cannot be benchmarked against another agency.** Only {cov['n_shared']} of "
        f"{cov['n_agencies']} agencies handle a complaint type that another agency also handles (at least "
        f"{MIN_N_SHARED} closures each); those shared types are {cov['pct_closures']:.1f}% of closures, and the "
        f"agencies log the same kinds of request in {peer['n_like']} of {peer['n_types']} shared "
        f"type{'' if peer['n_types'] == 1 else 's'} (each has at least {LIKE_FOR_LIKE_PCT}% of its closures under "
        "descriptors the other also uses regularly). For every "
        "other agency, 'exemplary' can only mean fast relative to its own complaint mix (its trend over time or its "
        "own targets), not faster than a peer.",
        *([("- **Where two agencies do the same work** (like-for-like types; vs typical for the same type, year and "
            "borough; sample): faster: " + elist(sig_fast) + "; slower: " + elist(sig_slow) + ".")]
          if peer["n_like"] else []),
        *peer["lines"],
        "- **Caveats.** Closure time measures when the ticket was closed, not whether the problem was fixed; "
        "agencies differ in when they close tickets (instant and date-only closures above). The comparisons are "
        "associations, not causes; the confidence intervals ignore clustering, so treat small gaps as noise.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q04.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
