"""Q2: is data collection consistent over time? Daily openings (a), closing times (b), and a board-ready
chart of openings and closings by day of week and week of year (c).

Rows: the README's default set (rules.BASE_SQL: not is_test and not dup_key). The analysis window runs from the
first to the last opening date of rows inside their file's dates (not out_of_file_order); rows opened outside it are
counted and left out of daily series. Daily and weekly averages also leave out possibly_incomplete_day. The typical
day-of-week and week-of-year profiles average the full calendar years (clean.full_years) up to
rules.PROFILE_LAST_YEAR; later years (the pandemic) are drawn and described separately. Aggregation runs in DuckDB;
only daily or monthly tables come into pandas. Every number in text and charts is computed.
"""
from __future__ import annotations

import math
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.tseries.holiday import USFederalHolidayCalendar  # noqa: E402

from nyc311 import clean, discover, nytplot as nyt, rules  # noqa: E402
from nyc311.clean import clean_glob  # noqa: E402
from nyc311.q01_missing import period_sql, period_unit  # noqa: E402
from nyc311.q05_top_types import profile_years, years_label  # noqa: E402
from nyc311.report import Report  # noqa: E402

STAGE = "q02_timing"
BASE = rules.BASE_SQL
BASELINE_WEEKS = 4        # same weekday, up to 4 weeks before and after
MIN_BASELINE = 3          # at least 3 such days to score a day
ROBUST_Z = 3.5            # |robust z| of log(day / same-weekday median) that counts as unusual
MIN_SPREAD = 0.05         # floor on the robust spread of that log ratio (a perfectly regular series flags nothing)
TOP_SURGE_DAYS = 5
SURGE_MIN_GROWTH = 0.20   # a month at least 20% above the same month a year earlier
TOP_EPISODES = 3
TOP_DRIVERS = 3
EDGE_DAYS = 30            # an agency first seen more than 30 days after the window starts "starts late"
DATE_ONLY_MIN_PCT = 5.0   # agencies shown in date-only tables reach this share in some period...
MIN_CELL_ROWS = 100       # ...with at least this many rows
BULK_RATIO = 5            # an agency closing >= 5x its usual daily closures...
BULK_MIN_ROWS = 200       # ...and at least this many in one day
USUAL_WINDOW_DAYS = 29    # "usual" = centered rolling median of the agency's daily closures
MAX_MARKS = 15            # dots on the daily chart: the most unusual days and largest bulk closures
CLOSE_WARMUP_QUANTILE = 0.75  # closings in the first p75-duration days miss requests opened before the data
UNEXPLAINED = "flagged by the cleaning stage; reason not reconstructed"
GRAY, LIGHT_GRAY = "#8c8c8c", "#e6e6e6"
LATER_HUES = nyt.HUES[2:5]  # years after rules.PROFILE_LAST_YEAR, each in its own color
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _d(x):
    """'Jan 17, 2010' for a date-like value."""
    t = pd.Timestamp(x)
    return f"{t:%b} {t.day}, {t.year}"


def _lit_date(x):
    return f"DATE '{pd.Timestamp(x):%Y-%m-%d}'"


def _period_pretty(unit):
    return {"week": "week (starting Monday)", "month": "month", "year": "year"}[unit]


# ---------------------------------------------------------------- (a) daily openings

def analysis_window(con, src):
    start, end = con.sql(f"""select min(created)::date, max(created)::date from {src}
                             where {BASE} and not out_of_file_order""").fetchone()
    n_out = con.sql(f"""select count(*) from {src} where {BASE}
                        and (created is null or created::date not between {_lit_date(start)} and {_lit_date(end)})
                     """).fetchone()[0]
    return pd.Timestamp(start), pd.Timestamp(end), int(n_out)


def outside_sentence(n):
    """The sentence about requests opened outside the window; empty when there are none."""
    return (f"{n:,} requests opened outside the window (stray backfilled rows) are left out of the daily series."
            if n else "")


def daily_table(con, src, start, end):
    """Every day in the window: openings, closings, date-only shares, and whether the day is possibly incomplete."""
    w = f"between {_lit_date(start)} and {_lit_date(end)}"
    o = con.sql(f"""select created::date as d, count(*) as n_open, bool_or(possibly_incomplete_day) as incomplete,
                           count_if(date_only_created) as n_open_date_only
                    from {src} where {BASE} and created::date {w} group by 1""").df()
    c = con.sql(f"""select closed::date as d, count(*) as n_close, count_if(date_only_closed) as n_close_date_only
                    from {src} where {BASE} and closed::date {w} group by 1""").df()
    spine = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    for t in (o, c):
        t["d"] = pd.to_datetime(t.d)
    out = spine.merge(o, left_on="date", right_on="d", how="left").drop(columns="d")
    out = out.merge(c, left_on="date", right_on="d", how="left").drop(columns="d")
    for col in ["n_open", "n_open_date_only", "n_close", "n_close_date_only"]:
        out[col] = out[col].fillna(0).astype("int64")
    out["incomplete"] = out.incomplete.astype("boolean").fillna(False).astype(bool)
    out["weekday"] = out.date.dt.dayofweek
    return out


def missing_batch_ranges(files):
    return discover.batch_sequence(files)["missing_ranges"]


def _file_windows(con, src):
    w = rules.OUT_OF_WINDOW_DAYS
    return con.sql(f"""
        with r as (select _file, created as ts from {src} where created is not null),
        m as (select _file, quantile_disc(ts, 0.5) as med from r group by 1),
        j as (select _file, ts, med, abs(date_diff('day', med, ts)) <= {w} as inwin from r join m using (_file))
        select _file as file, cast(any_value(med) as date) as p50, max(case when inwin then cast(ts as date) end) as mx,
               avg(inwin::int) as ordered_share
        from j group by 1 order by 1""").df()


def _num_ranges(nums):
    """[10, 2820, 2821, 2822] -> '10, 2820-2822'."""
    nums, out = sorted(nums), []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def sparse_tail(numbered, present, last_day):
    """The cleaning stage's sparse-tail rule (clean.incomplete_days, rules.SPARSE_TAIL_*): from the earliest missing
    batch number g with enough of [g, last] missing, every day after the median date of the last date-ordered file
    before g is possibly incomplete. Returns (first day, last day, reason) or None."""
    if not present:
        return None
    last_num = max(present)
    missing = sorted(set(range(min(present), last_num + 1)) - set(present))
    for i, g in enumerate(missing):
        k, span = len(missing) - i, last_num - g + 1
        if k >= rules.SPARSE_TAIL_MIN_MISSING and k / span >= rules.SPARSE_TAIL_MISSING_SHARE:
            before = numbered[numbered.num < g]
            if not len(before):
                return None
            lo = before.p50.iloc[-1] + pd.Timedelta(days=1)
            return lo, last_day, (f"end of the pull: {k} of the {span} batch numbers {g}-{last_num} are missing "
                                  f"({_num_ranges(missing[i:])}); files there overlap in time, so every day after "
                                  f"the median date of {before.file.iloc[-1]} is flagged")
    return None


def incomplete_reasons(con, src, manifest, daily):
    """Why each possibly_incomplete_day is flagged, reconstructed with the cleaning stage's rules: the late days of
    a cut-off file, the days between the files around missing batch numbers, days after the last file, and every
    day from the start of a run of missing numbers at the end of the pull (the sparse tail, one reason for all)."""
    days = daily[daily.incomplete]
    reasons = {d: [] for d in days.date}
    if not reasons:
        return pd.DataFrame(columns=["date", "n_open", "reason"])
    fw = _file_windows(con, src)
    fw = fw[fw.ordered_share >= rules.ORDERED_FILE_SHARE].copy()
    for c in ["p50", "mx"]:
        fw[c] = pd.to_datetime(fw[c])

    def add(lo, hi, text):
        for d in pd.date_range(lo, hi, freq="D"):
            if d in reasons:
                reasons[d].append(text)

    if len(fw):
        last = fw.iloc[-1]
        add(last.p50 + pd.Timedelta(days=1), last.mx,
            f"after the median date of the last file ({last.file}); later rows would be in files not in this pull")
        usual = int(manifest.n_records.mode().iloc[0]) if len(manifest) else 0
        bad = manifest[~manifest.parse_ok.astype(bool)].set_index("file")
        from nyc311.ingest import problem_kind
        for r in fw[fw.file.isin(bad.index)].itertuples(index=False):
            m = bad.loc[r.file]
            add(r.p50, r.mx, f"{r.file} is {problem_kind(m.error).split(' (')[0]}: {int(m.n_records):,} rows kept, "
                             f"files usually hold {usual:,}")
        fw["num"] = fw.file.str.extract(r"_(\d+)\.json$", expand=False).astype(float)
        numbered = fw.dropna(subset=["num"]).sort_values("num")
        present = set(manifest.file.str.extract(r"_(\d+)\.json$", expand=False).dropna().astype(int))
        for prev, nxt in zip(numbered.itertuples(), numbered.iloc[1:].itertuples()):
            between = set(range(int(prev.num) + 1, int(nxt.num)))
            if not between or (between & present):
                continue
            lo, hi = min(between), max(between)
            label = f"{lo}" if lo == hi else f"{lo}-{hi}"
            add(prev.p50 + pd.Timedelta(days=1), nxt.p50 - pd.Timedelta(days=1),
                f"batch {label} missing (between {prev.file} and {nxt.file})")
        tail = sparse_tail(numbered, present, fw.mx.max())
        if tail:
            for d in pd.date_range(tail[0], tail[1], freq="D"):
                if d in reasons:
                    reasons[d] = [tail[2]]  # one reason for the whole tail (it covers the gaps inside it)
    out = days[["date", "n_open"]].copy()
    out["reason"] = [("; ".join(reasons[d]) if reasons[d] else UNEXPLAINED) for d in out.date]
    return out.reset_index(drop=True)


def runs_of_days(df, key):
    """Consecutive days with the same key value, as (first, last, days, rows, key)."""
    if df.empty:
        return pd.DataFrame(columns=["from", "to", "days", "rows", key])
    d = df.sort_values("date").reset_index(drop=True)
    brk = (d.date.diff() != pd.Timedelta(days=1)) | (d[key] != d[key].shift())
    g = d.groupby(brk.cumsum())
    return pd.DataFrame({"from": g.date.min().dt.date, "to": g.date.max().dt.date, "days": g.size(),
                         "rows": g.n_open.sum(), key: g[key].first()}).reset_index(drop=True)


def _holidays(start, end):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = USFederalHolidayCalendar().holidays(start, end, return_name=True)
    return h


def score_days(n, excluded, weeks=BASELINE_WEEKS, min_base=MIN_BASELINE, z_cut=ROBUST_Z):
    """Robust score of each day against the median of the same weekday up to `weeks` weeks either side.

    n and excluded are Series on a consecutive daily index. Excluded days and days with no rows at all (gaps,
    reported as runs instead) are neither scored nor used as baselines.
    z = (log ratio - median log ratio) / (1.4826 * MAD of log ratios), with the spread floored at MIN_SPREAD.
    kind is 'high', 'low', '' (usual), 'no baseline', 'empty' or 'excluded'."""
    excluded = excluded.astype(bool)
    empty = (n == 0) & ~excluded
    s = n.astype(float).where(~excluded & ~empty)
    cols = [s.shift(7 * k).to_numpy() for k in range(1, weeks + 1)]
    cols += [s.shift(-7 * k).to_numpy() for k in range(1, weeks + 1)]
    m = np.column_stack(cols) if cols else np.full((len(s), 1), np.nan)
    cnt = (~np.isnan(m)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.where(cnt >= min_base, np.nanmedian(m, axis=1), np.nan)
    lr = np.log((s.to_numpy() + 1) / (base + 1))
    ok = ~np.isnan(lr)
    center = np.median(lr[ok]) if ok.any() else 0.0
    spread = max(1.4826 * np.median(np.abs(lr[ok] - center)), MIN_SPREAD) if ok.any() else MIN_SPREAD
    z = (lr - center) / spread
    kind = np.where(excluded.to_numpy(), "excluded", np.where(empty.to_numpy(), "empty",
                    np.where(~ok, "no baseline", np.where(z >= z_cut, "high", np.where(z <= -z_cut, "low", "")))))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(base > 0, n.to_numpy() / base, np.nan)
    out = pd.DataFrame({"date": n.index, "weekday": n.index.dayofweek, "n": n.to_numpy(), "baseline": base,
                        "ratio": ratio, "z": z, "kind": kind})
    hol = _holidays(n.index.min(), n.index.max()) if len(n) else pd.Series(dtype=object)
    out["holiday"] = out.date.map(hol)
    return out


def agency_spans(con, src, start, end):
    df = con.sql(f"""select agency, min(created)::date as first_day, max(created)::date as last_day, count(*) as n
                     from {src} where {BASE} and created::date between {_lit_date(start)} and {_lit_date(end)}
                     group by 1 order by n desc""").df()
    late = pd.to_datetime(df.first_day) > start + pd.Timedelta(days=EDGE_DAYS)
    early = pd.to_datetime(df.last_day) < end - pd.Timedelta(days=EDGE_DAYS)
    df["change"] = np.where(late & early, "starts late, stops early",
                            np.where(late, "starts late", np.where(early, "stops early", "")))
    return df


def share_by_agency_period(con, src, flag, ts, unit, start, end, among="true"):
    """Long table agency x period: rows and % with `flag` (period of `ts`), plus an '(all)' row per period."""
    p = period_sql(unit, ts)
    w = f"{ts}::date between {_lit_date(start)} and {_lit_date(end)}"
    return con.sql(f"""
        select coalesce(agency, '(none)') as agency, {p} as period, count(*) as n,
               100.0 * count_if({flag}) / count(*) as pct
        from {src} where {BASE} and {w} and ({among}) group by all
        union all
        select '(all)', {p}, count(*), 100.0 * count_if({flag}) / count(*)
        from {src} where {BASE} and {w} and ({among}) group by all
        order by 1, 2""").df()


def notable_pivot(long, min_pct=DATE_ONLY_MIN_PCT, min_rows=MIN_CELL_ROWS):
    """agency x period % for '(all)' and agencies reaching min_pct in a period with min_rows rows."""
    big = long[(long.n >= min_rows) & (long.pct >= min_pct) & (long.agency != "(all)")]
    keep = ["(all)"] + list(long[long.agency.isin(big.agency)].groupby("agency").n.sum()
                            .sort_values(ascending=False).index)
    t = long[long.agency.isin(keep)].pivot(index="agency", columns="period", values="pct").reindex(keep)
    return t.round(1)


def lasting_change(series, below):
    """First label from which every later value is below `below` (None if never)."""
    s = series.dropna()
    b = (s < below).to_numpy()
    for i in range(len(s)):
        if b[i:].all():
            return s.index[i]
    return None


def channel_by_period(con, src, unit, start, end):
    p = period_sql(unit)
    df = con.sql(f"""select {p} as period, coalesce(channel, 'missing (UNKNOWN or NULL)') as channel, count(*) as n
                     from {src} where {BASE} and created::date between {_lit_date(start)} and {_lit_date(end)}
                     group by all""").df()
    t = df.pivot(index="period", columns="channel", values="n").fillna(0)
    order = t.sum().sort_values(ascending=False).index
    pct = (100 * t[order].div(t.sum(axis=1), axis=0)).round(1)
    pct.insert(0, "n", t.sum(axis=1).astype("int64"))
    return pct.reset_index()


def backfilled(con, src):
    return con.sql(f"""select agency, year(created) as "year", count(*) as n,
                              min(_file) as first_file, max(_file) as last_file
                       from {src} where {BASE} and out_of_file_order group by all order by n desc""").df()


def _type_expr():
    return "coalesce(canonical_type, complaint_type, '(none)')"


def surge_days(con, src, scored, k=TOP_SURGE_DAYS, weeks=BASELINE_WEEKS):
    """The k highest-scoring days and the complaint types that drove their excess over the same weekday.

    baseline is the MEAN of the usable same-weekday days up to `weeks` weeks either side (the scoring uses their
    median); per-type means over the same days add up to it, so each type's share of the excess (n - baseline) is
    coherent. Shares can pass 100% when other types run below their usual level that day."""
    hi = scored[scored.kind == "high"].sort_values("z", ascending=False).head(k)
    if hi.empty:
        return pd.DataFrame(columns=["date", "n", "baseline", "excess", "n_baseline_days", "holiday", "drivers"])
    usable = set(scored[~scored.kind.isin(["excluded", "empty"])].date)
    neigh = {}
    for d in hi.date:
        cand = [d + pd.Timedelta(days=7 * j) for j in list(range(-weeks, 0)) + list(range(1, weeks + 1))]
        neigh[d] = [c for c in cand if c in usable]
    dates = sorted(set(hi.date) | {c for v in neigh.values() for c in v})
    lst = ", ".join(_lit_date(x) for x in dates)
    t = con.sql(f"""select created::date as d, {_type_expr()} as t, count(*) as n from {src}
                    where {BASE} and created::date in ({lst}) group by all""").df()
    t["d"] = pd.to_datetime(t.d)
    wide = t.pivot(index="t", columns="d", values="n").fillna(0)
    rows = []
    for r in hi.itertuples(index=False):
        # per-type means over the same baseline days add up to the mean total, so shares of excess are coherent
        base = wide.reindex(columns=neigh[r.date], fill_value=0).mean(axis=1)
        day = wide[r.date] if r.date in wide.columns else pd.Series(0, index=wide.index)
        ex = (day - base).sort_values(ascending=False)
        total = float(day.sum() - base.sum())
        rows.append({"date": r.date, "n": int(r.n), "baseline": float(base.sum()), "excess": total,
                     "n_baseline_days": len(neigh[r.date]), "holiday": r.holiday,
                     "drivers": "; ".join(f"{t} +{v:,.0f} ({100 * v / total:.0f}% of excess)"
                                          for t, v in ex.head(TOP_DRIVERS).items() if v > 0)})
    return pd.DataFrame(rows)


def surge_episodes(con, src, daily, start, end):
    """Runs of complete months at least SURGE_MIN_GROWTH above the same month a year earlier, with the
    complaint types behind the extra requests. Empty when the data has no complete month a year apart."""
    cols = ["months", "n", "n_year_before", "excess", "growth_pct", "drivers"]
    d = daily.assign(month=daily.date.dt.to_period("M"))
    full = pd.period_range(start, end, freq="M")
    days_in = d.groupby("month").size()
    bad = d[d.incomplete].month.unique()
    complete = [m for m in full if days_in.get(m, 0) == m.days_in_month and m not in bad]
    pairs = [m for m in complete if (m - 12) in complete]
    if not pairs:
        return pd.DataFrame(columns=cols)
    t = con.sql(f"""select date_trunc('month', created)::date as m, {_type_expr()} as t, count(*) as n from {src}
                    where {BASE} and created::date between {_lit_date(start)} and {_lit_date(end)}
                    group by all""").df()
    t["m"] = pd.to_datetime(t.m).dt.to_period("M")
    wide = t.pivot(index="t", columns="m", values="n").fillna(0)
    tot = wide.sum()
    growth = pd.Series({m: tot.get(m, 0) / tot[m - 12] - 1 for m in pairs if tot.get(m - 12, 0) > 0}, dtype=float)
    surge = sorted(growth[growth >= SURGE_MIN_GROWTH].index)
    episodes, cur = [], []
    for m in surge:
        if cur and (m - cur[-1]).n != 1:
            episodes.append(cur)
            cur = []
        cur.append(m)
    if cur:
        episodes.append(cur)
    rows = []
    for ep in episodes:
        now = wide.reindex(columns=ep, fill_value=0).sum(axis=1)
        before = wide.reindex(columns=[m - 12 for m in ep], fill_value=0).sum(axis=1)
        ex = (now - before).sort_values(ascending=False)
        excess = float(now.sum() - before.sum())
        rows.append({"months": f"{ep[0]}" if len(ep) == 1 else f"{ep[0]} to {ep[-1]}",
                     "n": int(now.sum()), "n_year_before": int(before.sum()), "excess": excess,
                     "growth_pct": 100 * excess / before.sum(),
                     "drivers": "; ".join(f"{k} +{v:,.0f} ({100 * v / excess:.0f}%)"
                                          for k, v in ex.head(TOP_DRIVERS + 1).items() if v > 0)})
    return pd.DataFrame(rows, columns=cols).sort_values("excess", ascending=False).head(TOP_EPISODES)


# ---------------------------------------------------------------- (b) closings

def closing_checks(con, src, start, end):
    r = con.sql(f"""
        select count(*) as n, count_if(closed is not null) as n_closed,
               count_if(closed is null and not closed_placeholder) as n_no_close,
               count_if(closed_placeholder) as n_placeholder,
               count_if(closed_before_created) as n_before,
               count_if(closed_before_created and closed < created - interval 1 day) as n_before_1d,
               count_if(closed_at_creation) as n_at_creation,
               count_if(date_only_closed) as n_date_only,
               count_if(closed_after_1yr) as n_after_1yr,
               count_if(closed::date > {_lit_date(end)}) as n_after_last,
               count_if(closed > now()::timestamp) as n_future,
               count_if(closed::date < {_lit_date(start)}) as n_before_start,
               min(closed) as min_closed, max(closed) as max_closed
        from {src} where {BASE}""").df().iloc[0]
    n, nc = int(r.n), max(int(r.n_closed), 1)
    rows = [("requests (default set)", r.n, n, ""),
            ("with a close date", r.n_closed, n, "1900 placeholders are NULL after cleaning"),
            ("no close date", r.n_no_close, n, "open, pending, or never closed"),
            ("1900 placeholder close date", r.n_placeholder, n, "set to NULL by cleaning"),
            ("closed before opened", r.n_before, nc, "no duration"),
            ("... more than a day before", r.n_before_1d, nc, ""),
            ("closed at the second opened", r.n_at_creation, nc, "closed_at_creation"),
            ("close time exactly midnight", r.n_date_only, nc, "date recorded without time"),
            ("closed more than a year after opening", r.n_after_1yr, nc, "status Closed only"),
            ("closed after the last opening date", r.n_after_last, nc, f"after {_d(end)}"),
            ("closed before the first opening date", r.n_before_start, nc, f"before {_d(start)}"),
            ("close date later than today", r.n_future, nc, "impossible dates")]
    out = pd.DataFrame(rows, columns=["check", "rows", "denom", "note"])
    out["rows"] = out.rows.astype("int64")
    out["pct"] = (100 * out.rows / out.denom).round(2)
    return out.drop(columns="denom")[["check", "rows", "pct", "note"]], r.min_closed, r.max_closed


def closing_by_agency(con, src):
    return con.sql(f"""
        select agency, count(*) as n, count(closed) as n_closed,
               round(100 * count_if(closed_before_created) / nullif(count(closed), 0), 2) as pct_before_open,
               round(median(case when closed_before_created then date_diff('second', closed, created) / 86400.0
                                 end), 1) as median_days_early,
               round(100 * count_if(closed_at_creation) / nullif(count(closed), 0), 2) as pct_at_creation,
               round(100 * count_if(date_only_closed) / nullif(count(closed), 0), 1) as pct_midnight,
               count_if(closed_placeholder) as n_1900,
               round(100 * count_if(closed_after_1yr) / nullif(count(closed), 0), 2) as pct_over_1yr
        from {src} where {BASE} group by 1 order by n desc""").df()


def placeholder_status(con, src):
    return con.sql(f"""select coalesce(status, '(none)') as status, agency, count(*) as n from {src}
                       where {BASE} and closed_placeholder group by all order by n desc""").df()


def closes_outside(con, src, start, end):
    return con.sql(f"""select case when closed::date > {_lit_date(end)} then 'after last opening'
                                   else 'before first opening' end as side, year(closed) as "year", count(*) as n
                       from {src} where {BASE} and closed is not null
                         and closed::date not between {_lit_date(start)} and {_lit_date(end)}
                       group by all order by 1, 2""").df()


def agency_close_days(con, src, start, end):
    """agency x close day: closures, distinct timestamps, and the busiest single timestamp."""
    df = con.sql(f"""
        with t as (select agency, closed, count(*) as n from {src}
                   where {BASE} and closed::date between {_lit_date(start)} and {_lit_date(end)} group by all)
        select coalesce(agency, '(none)') as agency, closed::date as d, sum(n)::bigint as n,
               max(n)::bigint as top_ts_n, arg_max(closed, n) as top_ts, count(*)::bigint as n_ts
        from t group by all""").df()
    df["d"] = pd.to_datetime(df.d)
    return df


def bulk_closures(ad, start, end):
    """Agency-days closing >= BULK_RATIO x the agency's usual daily closures (centered rolling median over
    USUAL_WINDOW_DAYS, zeros included) and >= BULK_MIN_ROWS rows; largest excess first."""
    days = pd.date_range(start, end, freq="D")
    found = []
    for agency, g in ad.groupby("agency"):
        s = g.set_index("d").n.reindex(days, fill_value=0).astype(float)
        usual = s.rolling(USUAL_WINDOW_DAYS, center=True, min_periods=min(7, len(s))).median()
        hit = (s >= BULK_RATIO * usual.clip(lower=1)) & (s >= BULK_MIN_ROWS)
        if hit.any():
            info = g.set_index("d").reindex(s.index[hit])
            found.append(pd.DataFrame({"date": s.index[hit], "agency": agency, "n": s[hit].astype("int64").values,
                                       "usual": usual[hit].values, "top_ts": info.top_ts.values,
                                       "top_ts_n": info.top_ts_n.values, "n_ts": info.n_ts.values}))
    cols = ["date", "agency", "n", "usual", "ratio", "top_ts", "top_ts_n", "top_share", "n_ts", "excess"]
    if not found:
        return pd.DataFrame(columns=cols)
    out = pd.concat(found, ignore_index=True)
    out["ratio"] = out.n / out.usual.clip(lower=1)
    out["top_share"] = out.top_ts_n / out.n
    out["excess"] = out.n - out.usual
    return out[cols].sort_values("excess", ascending=False).reset_index(drop=True)


def warmup_days(con, src):
    q = con.sql(f"""select quantile_cont(duration_h, {CLOSE_WARMUP_QUANTILE}) from {src}
                    where {BASE} and duration_h is not null""").fetchone()[0]
    return int(math.ceil((q or 0) / 24))


# ---------------------------------------------------------------- (c) day-of-week and week-of-year patterns

def week_patterns(daily, start, end, warmup, full_years):
    """Average openings/closings per day by weekday and by week of the year (week 1 = Jan 1-7; week 53 dropped).
    Usable days are not flagged possibly incomplete and have at least one opening (a day with no rows at all is a
    gap in the files, not a quiet day); closings also skip the first `warmup` days. A year-week counts only if all 7
    days are usable. The typical profile averages the full calendar years (clean.full_years)
    up to rules.PROFILE_LAST_YEAR; later years are kept apart (later_years); see profile_years for fallbacks."""
    d = daily.copy()
    d["open_ok"] = ~d.incomplete & (d.n_open > 0)
    d["close_ok"] = d.open_ok & (d.date >= start + pd.Timedelta(days=warmup))
    d["year"] = d.date.dt.year
    d["week"] = (d.date.dt.dayofyear - 1) // 7 + 1
    years = sorted(int(y) for y in d.year.unique())
    avg_years, later = profile_years(full_years, years)
    a = d[d.year.isin(avg_years)]
    dow = pd.DataFrame({"weekday": range(7), "name": WEEKDAYS})
    for kind, ok in [("open", "open_ok"), ("close", "close_ok")]:
        g = a[a[ok]].groupby("weekday")[f"n_{kind}"]
        dow[kind] = dow.weekday.map(g.mean())
        dow[f"n_days_{kind}"] = dow.weekday.map(g.size()).fillna(0).astype(int)
    dy = []
    for kind, ok in [("open", "open_ok"), ("close", "close_ok")]:
        g = d[d[ok]].groupby(["year", "weekday"])[f"n_{kind}"].mean().rename(kind)
        dy.append(g)
    dow_years = pd.concat(dy, axis=1).reset_index()
    w = d[d.week <= 52]
    parts = []
    for kind, ok in [("open", "open_ok"), ("close", "close_ok")]:
        g = w.groupby(["year", "week"]).agg(m=(f"n_{kind}", "mean"), k=(ok, "sum"))
        parts.append(g.m.where(g.k == 7).rename(kind))
    woy = pd.concat(parts, axis=1).reset_index()
    woy = woy[woy[["open", "close"]].notna().any(axis=1)].reset_index(drop=True)
    wa = woy[woy.year.isin(avg_years)].groupby("week").agg(open=("open", "mean"), close=("close", "mean"),
                                                          n_years_open=("open", "count"),
                                                          n_years_close=("close", "count")).reset_index()
    return {"dow": dow, "dow_years": dow_years, "woy": woy, "woy_avg": wa, "full_years": list(full_years),
            "avg_years": avg_years, "later_years": later,
            "other_years": [y for y in years if y not in avg_years and y not in later], "warmup": warmup,
            "n_empty": int(((d.n_open == 0) & ~d.incomplete).sum())}


def later_year_summary(pat):
    """Per later year: its busiest weekday and its peak week (openings), against the typical profile."""
    dy, woy, wa = pat["dow_years"], pat["woy"], pat["woy_avg"].set_index("week")
    rows = []
    for y in pat["later_years"]:
        w = woy[(woy.year == y)].dropna(subset=["open"])
        dd = dy[dy.year == y].dropna(subset=["open"])
        r = {"year": y, "n_weeks": len(w), "peak_week": None, "peak_open": np.nan, "typical_open": np.nan,
             "first_week": None, "last_week": None, "top_day": None, "top_day_open": np.nan}
        if len(w):
            p = w.loc[w.open.idxmax()]
            r.update(peak_week=int(p.week), peak_open=float(p.open), typical_open=float(wa.open.get(p.week, np.nan)),
                     first_week=int(w.week.min()), last_week=int(w.week.max()))
        if len(dd):
            t = dd.loc[dd.open.idxmax()]
            r.update(top_day=WEEKDAYS[int(t.weekday)], top_day_open=float(t.open))
        rows.append(r)
    return pd.DataFrame(rows, columns=["year", "n_weeks", "peak_week", "peak_open", "typical_open", "first_week",
                                       "last_week", "top_day", "top_day_open"])


# ---------------------------------------------------------------- charts

def _date_axis(ax, start, end):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    span = (end - start).days
    if span > 3 * 365:
        fmt = lambda t: f"{t:%Y}"
    elif span > 120:
        fmt = lambda t: f"{t:%b} {t.year}"
    else:
        fmt = lambda t: f"{t:%b} {t.day}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos=None: fmt(mdates.num2date(v))))
    ax.set_xlim(start, end + pd.Timedelta(days=1))


def _label(ax, text, x, y, frac):
    """Small gray sans label above a marked point, on a white backing; flipped left near the right edge."""
    right = frac > 0.7
    return ax.annotate(text, (x, y), xytext=(-6 if right else 6, 6), textcoords="offset points", fontsize=8,
                       color=GRAY, fontfamily=nyt.SANS, ha="right" if right else "left", va="bottom", zorder=4,
                       bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))


def plot_daily(daily, scored, bulk, inc_runs, path, start, end):
    """Daily openings (sage) and closings (coral). Gray bands = possibly incomplete days; dots = the most unusual
    days (openings) and the largest bulk-closure days (closings)."""
    nyt.use()
    fig, (top, bot) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.2))
    x = daily.date + pd.Timedelta(hours=12)
    lw = 1.5 if len(daily) <= 120 else 0.5
    span = (end - start).days + 1
    frac = lambda d: (pd.Timestamp(d) - start).days / span
    for ax, col, h, label in [(top, "n_open", nyt.SAGE, "Opened"), (bot, "n_close", nyt.CORAL, "Closed")]:
        for r in inc_runs.itertuples(index=False):
            ax.axvspan(pd.Timestamp(r[0]), pd.Timestamp(r[1]) + pd.Timedelta(days=1), color=LIGHT_GRAY, lw=0,
                       zorder=0.5)
        line = nyt.area(ax, x, daily[col], h)
        line.set_linewidth(lw)
        nyt.panel_label(ax, label, h.line, xy=(0.012, 0.88))
        nyt.format_y(ax)
        ax.set_ylim(0, daily[col].max() * 1.15)
    odd = scored[scored.kind.isin(["low", "high"])]
    marks = odd.reindex(odd.z.abs().sort_values(ascending=False).index).head(MAX_MARKS)
    top.scatter(marks.date + pd.Timedelta(hours=12), marks.n, s=12, color=nyt.SAGE.line, zorder=3, linewidths=0)
    for kind in ["high", "low"]:  # label the most extreme high and low day
        t = marks[marks.kind == kind]
        if len(t):
            r = t.loc[t.z.abs().idxmax()]
            _label(top, f"{_d(r.date)}: {r.n:,.0f} ({kind})", r.date + pd.Timedelta(hours=12), r.n, frac(r.date))
    bmarks = bulk.head(MAX_MARKS)
    if len(bmarks):
        closes = daily.set_index("date").n_close
        b = bmarks.groupby("date").n.sum()
        bot.scatter(b.index + pd.Timedelta(hours=12), closes.reindex(b.index), s=12, color=nyt.CORAL.line,
                    zorder=3, linewidths=0)
        r = bmarks.iloc[0]
        _label(bot, f"{_d(r.date)}: {r.agency} closed {r.n:,}", r.date + pd.Timedelta(hours=12),
               closes.get(r.date, r.n), frac(r.date))
    _date_axis(bot, start, end)
    nyt.title(fig, f"311 requests opened and closed each day, {_d(start)} to {_d(end)}")
    n_inc = int(daily.incomplete.sum())
    bulk_txt = (f"the {len(bmarks)} largest bulk-closure days (bottom)" if len(bmarks) else
                "no bulk-closure days were found (bottom)")
    nyt.note(fig, f"Note: gray bands mark {n_inc} days flagged as possibly incomplete (batch files missing or cut "
                  f"off). Dots mark the {len(marks)} days furthest from the\nsame weekday's usual level (top; "
                  f"{len(odd)} unusual days in all); {bulk_txt}. Closings start low\nbecause requests opened "
                  f"before {_d(start)} are not in the data.")
    return nyt.save(fig, path)


def _week_label(w):
    first = pd.Timestamp("2001-01-01") + pd.Timedelta(days=7 * (int(w) - 1))  # a non-leap year
    return f"{first:%b} {first.day}"


def _week_ticks(ax, lo, hi):
    if hi - lo >= 12:
        firsts = pd.date_range("2001-01-01", periods=12, freq="MS")
        ax.set_xticks([(f.dayofyear - 1) / 7 + 1 for f in firsts], [f"{f:%b}" for f in firsts])
        ax.set_xlim(1, 52)
    else:
        ticks = list(range(lo, hi + 1))
        step = max(1, len(ticks) // 6)
        ax.set_xticks(ticks[::step], [_week_label(t) for t in ticks[::step]])
        ax.set_xlim(lo - 0.5, hi + 0.5)


KEY_HEADROOM = 1.32  # y limit = this x the highest value, so the series key above the data never covers it


def _key_row(ax, items, xy=(0.012, 0.97), fontsize=10.5):
    """Series names in one row at the top of the panel (in the headroom above the data), bold in each series'
    color, instead of a legend."""
    prev = None
    for text, color in items:
        kw = dict(color=color, fontfamily=nyt.SANS, fontweight="bold", fontsize=fontsize, ha="left", va="top",
                  zorder=5)
        if prev is None:
            prev = ax.text(*xy, text, transform=ax.transAxes, **kw)
        else:
            prev = ax.annotate(text, xy=(1, 1), xycoords=prev, xytext=(10, 0), textcoords="offset points", **kw)
    return prev


def _avg_key(years):
    """Key text for the average: the years when they run on, else how many."""
    lab = years_label(years)
    return f"Average, {lab}" if "," not in lab else f"Average of {len(years)} years"


def _avg_note(pat, start, end):
    yrs = years_label(pat["avg_years"])
    if pat["full_years"]:
        what = f"the full calendar years {yrs}"
    else:
        what = f"{yrs} (no full calendar year in {_d(start)} to {_d(end)})"
    later = pat["later_years"]
    drawn = [y for y in later if (pat["woy"].year == y).any()]
    blank = [y for y in later if y not in drawn]
    if drawn:
        what += f"; {years_label(drawn)} {'is' if len(drawn) == 1 else 'are'} drawn separately and not averaged"
    if blank:
        what += f"; {years_label(blank)} {'has' if len(blank) == 1 else 'have'} no complete week"
    return what


def plot_week_patterns(pat, daily, path, start, end):
    """Board-ready: typical openings and closings per day by weekday (bars) and week of year (coral line), from the
    full years up to rules.PROFILE_LAST_YEAR; later years as their own colored lines and dots; other years gray."""
    nyt.use()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharey=True, gridspec_kw={"width_ratios": [1, 2.4]})
    dow, dy, woy, wa = pat["dow"], pat["dow_years"], pat["woy"], pat["woy_avg"]
    later = pat["later_years"]
    hue_of = {y: LATER_HUES[i % len(LATER_HUES)] for i, y in enumerate(later)}
    drawn_later = [y for y in later if (woy.year == y).any()]
    gray_years = sorted(set(woy.year) - set(later))
    weeks = woy.week if len(woy) else pd.Series([1, 52])
    lo, hi = int(weeks.min()), int(weeks.max())
    few = len(wa) < 10
    top = np.nanmax([dow.open.max(), dow.close.max(), woy.open.max(), woy.close.max(),
                     dy.open.max() if len(dy) else 0, dy.close.max() if len(dy) else 0])
    for row, (kind, label) in enumerate([("open", "Opened"), ("close", "Closed")]):
        ax_d, ax_w = axes[row]
        vals = dow[kind].to_numpy(dtype=float)
        peak = int(np.nanargmax(vals)) if np.isfinite(vals).any() else 0
        colors = [nyt.CORAL.line if i == peak else nyt.CORAL.band for i in range(7)]
        ax_d.bar(range(7), np.nan_to_num(vals), color=colors, width=0.7, zorder=2)
        if dy.year.nunique() > 1:
            g = dy[~dy.year.isin(later)]
            ax_d.scatter(g.weekday, g[kind], s=8, color=GRAY, alpha=0.7, zorder=3, linewidths=0)
            for y in later:
                g = dy[dy.year == y]
                ax_d.scatter(g.weekday, g[kind], s=14, color=hue_of[y].line, zorder=4, linewidths=0)
        ax_d.set_xticks(range(7), [n[:3] for n in WEEKDAYS])
        ax_d.grid(axis="x", visible=False)
        nyt.panel_title(ax_d, f"{label}, by day of week")
        if np.isfinite(vals[peak]):
            ax_d.annotate(f"{vals[peak]:,.0f}", (peak, vals[peak]), xytext=(0, 3), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8.5, color=nyt.CORAL.line, fontfamily=nyt.SANS,
                          fontweight="bold", zorder=5)
        for y in gray_years:
            g = woy[woy.year == y].set_index("week")[kind].reindex(range(1, 53))
            ax_w.plot(g.index, g.values, color="#b3b3b3", lw=0.7, zorder=1, marker="o" if few else None, ms=3)
        for y in drawn_later:
            g = woy[woy.year == y].set_index("week")[kind].reindex(range(1, 53))
            ax_w.plot(g.index, g.values, color=hue_of[y].line, lw=1.3, zorder=2, marker="o" if few else None, ms=3)
        avg = wa.set_index("week")[kind].reindex(range(1, 53)) if len(wa) else pd.Series(dtype=float)
        ax_w.plot(avg.index, avg.values, color=nyt.CORAL.line, lw=2.2, zorder=3, marker="o" if few else None, ms=4)
        _week_ticks(ax_w, lo, hi)
        ax_w.grid(axis="x", visible=False)
        nyt.panel_title(ax_w, f"{label}, by week of the year")
        items = [(_avg_key(pat["avg_years"]), nyt.CORAL.line)] + [(str(y), hue_of[y].line) for y in drawn_later]
        if len(gray_years) > 1 or (gray_years and drawn_later):
            items.append(("Each year" if not later else f"Each year, {years_label(gray_years)}", GRAY))
        _key_row(ax_w, items)
        nyt.format_y(ax_d)
    axes[0, 0].set_ylim(0, top * KEY_HEADROOM if np.isfinite(top) and top > 0 else 1)
    po, pc = dow.loc[dow.open.idxmax(), "name"], dow.loc[dow.close.idxmax(), "name"]
    title = (f"311 requests are opened and closed most on {po}s" if po == pc else
             f"311 openings peak on {po}s, closings on {pc}s")
    nyt.title(fig, title)
    n_inc = int(daily.incomplete.sum())
    dot_later = [y for y in later if dy[dy.year == y].open.notna().any()]
    dots = (f" Dots on the bars: single years ({years_label(dot_later)} in "
            f"{'its' if len(dot_later) == 1 else 'their'} line colors)." if dot_later and dy.year.nunique() > 1 else
            " Dots on the bars: single years." if dy.year.nunique() > 1 else "")
    empty = f" and {pat['n_empty']:,} with no rows at all" if pat["n_empty"] else ""
    nyt.note(fig, textwrap.fill(
        f"Note: average requests per day over {_avg_note(pat, start, end)}.{dots} Leaves out {n_inc} days flagged as "
        f"possibly incomplete{empty}; closings also skip the first {pat['warmup']} days, when requests opened before "
        "the data starts are missing. A week counts only if all 7 days are usable; week 1 = Jan 1-7.", 135))
    return nyt.save(fig, path)


# ---------------------------------------------------------------- run

def run(cfg):
    con = cfg.connect()
    src = clean_glob(cfg)
    manifest = pd.read_parquet(cfg.work_dir / "manifest.parquet")
    n_all, n_test, n_dup = con.sql(f"select count(*), count_if(is_test), count_if(dup_key) from {src}").fetchone()
    start, end, n_outside = analysis_window(con, src)
    unit = period_unit(con, src)
    daily = daily_table(con, src, start, end)
    inc = incomplete_reasons(con, src, manifest, daily)
    inc_runs = runs_of_days(inc, "reason")
    empty_runs = runs_of_days(daily[(daily.n_open == 0) & ~daily.incomplete].assign(reason="empty"), "reason")
    missing = missing_batch_ranges(manifest.file)
    scored = score_days(daily.set_index("date").n_open, daily.set_index("date").incomplete)
    daily = daily.merge(scored[["date", "baseline", "z", "kind", "holiday"]], on="date", how="left")
    spans = agency_spans(con, src, start, end)
    open_mid = share_by_agency_period(con, src, "date_only_created", "created", unit, start, end)
    chan = channel_by_period(con, src, unit, start, end)
    back = backfilled(con, src)
    sdays = surge_days(con, src, scored)
    eps = surge_episodes(con, src, daily, start, end)

    checks, min_closed, max_closed = closing_checks(con, src, start, end)
    by_ag = closing_by_agency(con, src)
    ph_status = placeholder_status(con, src)
    outside = closes_outside(con, src, start, end)
    close_mid = share_by_agency_period(con, src, "date_only_closed", "closed", unit, start, end)
    bulk = bulk_closures(agency_close_days(con, src, start, end), start, end)
    warm = warmup_days(con, src)
    pat = week_patterns(daily, start, end, warm, clean.full_years(con, src))
    later = later_year_summary(pat)

    figs = cfg.out_dir / "figs"
    f_daily = plot_daily(daily, scored, bulk, inc_runs, figs / "q02_daily_volume.png", start, end)
    f_week = plot_week_patterns(pat, daily, figs / "q02_week_patterns.png", start, end)

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    daily.to_csv(agg / "q02_daily.csv", index=False)
    inc.to_csv(agg / "q02_incomplete_days.csv", index=False)
    scored[scored.kind.isin(["low", "high"])].to_csv(agg / "q02_unusual_days.csv", index=False)
    spans.to_csv(agg / "q02_agency_spans.csv", index=False)
    open_mid.to_csv(agg / "q02_date_only_opened.csv", index=False)
    chan.to_csv(agg / "q02_channel.csv", index=False)
    back.to_csv(agg / "q02_backfilled.csv", index=False)
    sdays.to_csv(agg / "q02_surge_days.csv", index=False)
    eps.to_csv(agg / "q02_surge_episodes.csv", index=False)
    checks.to_csv(agg / "q02_closing_checks.csv", index=False)
    by_ag.to_csv(agg / "q02_closing_by_agency.csv", index=False)
    close_mid.to_csv(agg / "q02_date_only_closed.csv", index=False)
    outside.to_csv(agg / "q02_closed_outside_window.csv", index=False)
    bulk.to_csv(agg / "q02_bulk_closures.csv", index=False)
    pd.concat([pat["dow"].assign(table="day_of_week"),
               pat["woy_avg"].assign(table="week_of_year_average"),
               pat["woy"].assign(table="week_of_year_by_year"),
               pat["dow_years"].assign(table="day_of_week_by_year")], ignore_index=True).to_csv(
        agg / "q02_week_patterns.csv", index=False)

    v = dict(n_all=n_all, n_test=n_test, n_dup=n_dup, start=start, end=end, n_outside=n_outside, unit=unit,
             daily=daily, inc=inc, inc_runs=inc_runs, empty_runs=empty_runs, missing=missing, scored=scored, spans=spans,
             open_mid=open_mid, chan=chan, back=back, sdays=sdays, eps=eps, checks=checks, by_ag=by_ag,
             ph_status=ph_status, outside=outside, close_mid=close_mid, bulk=bulk, pat=pat, later=later,
             min_closed=min_closed, max_closed=max_closed, manifest=manifest, f_daily=f_daily, f_week=f_week)
    _report(cfg, v)
    _write_answer(cfg, v)
    return {"daily": daily, "incomplete": inc, "scored": scored, "missing_batches": missing,
            "closing_checks": checks, "bulk": bulk, "patterns": pat, "unit": unit, "n_outside": n_outside,
            "surge_days": sdays, "surge_episodes": eps, "later_years": later}


def _pivot_for_report(long):
    t = notable_pivot(long)
    t.columns = [str(c) for c in t.columns]
    return t.reset_index()


def _report(cfg, v):
    rep = Report(STAGE, cfg.out_dir)
    d, start, end, unit = v["daily"], v["start"], v["end"], v["unit"]
    n_base = int(v["checks"].set_index("check").rows["requests (default set)"])
    rep.add(f"Q2. Opening and closing times | window {start.date()} to {end.date()} ({len(d):,} days) | "
            f"{n_base:,} requests (left out: is_test {v['n_test']:,}, dup_key {v['n_dup']:,})",
            (f"opened outside the window (stray backfilled rows), left out of daily counts: {v['n_outside']:,} | "
             if v["n_outside"] else "") + f"days with no openings: {int((d.n_open == 0).sum()):,}")
    er = v["empty_runs"]
    if len(er):
        rep.add("empty stretches (no openings), longest first: " + "; ".join(
            f"{r[0]} to {r[1]} ({r[2]} days)" for r in er.sort_values("days", ascending=False).head(6)
            .itertuples(index=False)) + (f"; ... {len(er)} in all" if len(er) > 6 else ""))
    m = v["manifest"]
    bad = m[~m.parse_ok.astype(bool)]
    rep.add("", f"## (a) Possibly incomplete days: {int(d.incomplete.sum())} days, {int(d[d.incomplete].n_open.sum()):,}"
                f" rows | missing batch numbers: {', '.join(v['missing']) or 'none'} | problem files: "
                f"{', '.join(bad.file) or 'none'}")
    rep.table(v["inc_runs"], max_rows=10)
    sc = v["scored"]
    odd = sc[sc.kind.isin(["low", "high"])].copy()
    odd["date"] = odd.date.dt.date
    rep.table(odd.sort_values("z")[["date", "kind", "n", "baseline", "ratio", "z", "holiday"]], max_rows=12,
              title=f"(a) Unusual days vs the same weekday +-{BASELINE_WEEKS} weeks (|robust z| >= {ROBUST_Z}): "
                    f"{int((sc.kind == 'low').sum())} low, {int((sc.kind == 'high').sum())} high; lowest first")
    rep.table(v["sdays"].assign(date=lambda x: pd.to_datetime(x.date).dt.date)[["date", "n", "baseline", "drivers"]],
              max_rows=TOP_SURGE_DAYS, title="(a) Highest days: complaint types above their same-weekday level "
                                             "(baseline = mean of the same weekday +-4 weeks; shares are of n - baseline)")
    if len(v["eps"]):
        rep.table(v["eps"], max_rows=TOP_EPISODES, title=f"(a) Surge months (>= {SURGE_MIN_GROWTH:.0%} above the "
                                                         "same month a year earlier) and what drove them")
    else:
        rep.add("", "(a) Surge months: none, or no complete month a year apart to compare")
    sp = v["spans"]
    rep.table(sp[sp.change != ""], max_rows=10,
              title=f"(a) Agencies starting or stopping > {EDGE_DAYS} days inside the window "
                    f"({int((sp.change == '').sum())} of {len(sp)} present throughout)")
    rep.table(_pivot_for_report(v["open_mid"]), max_rows=8,
              title=f"(a) % of openings at exactly midnight (date only), by agency and {_period_pretty(unit)}")
    rep.table(v["chan"], max_rows=14, title=f"(a) Channel (% of openings) by {_period_pretty(unit)}")
    b = v["back"]
    rep.table(b, max_rows=8, title=f"(a) Backfilled rows (out_of_file_order): {int(b.n.sum()):,} by agency and "
                                   "year opened")
    rep.table(v["checks"], max_rows=15, title=f"(b) Closing checks (pct of requests for the first rows, "
                                              f"else of those with a close date); close dates span "
                                              f"{pd.Timestamp(v['min_closed']).date()} to "
                                              f"{pd.Timestamp(v['max_closed']).date()}")
    ag = v["by_ag"]
    rep.table(ag[ag.n >= MIN_CELL_ROWS], max_rows=10, title="(b) By agency (most requests first)")
    rep.table(_pivot_for_report(v["close_mid"]), max_rows=8,
              title=f"(b) % of closings at exactly midnight, by agency and {_period_pretty(unit)} of closing")
    bk = v["bulk"].copy()
    bk["date"] = pd.to_datetime(bk.date).dt.date
    rep.table(bk[["date", "agency", "n", "usual", "ratio", "top_ts", "top_share", "n_ts"]], max_rows=8,
              title=f"(b) Bulk-closure days: {len(bk)} agency-days >= {BULK_RATIO}x usual and >= {BULK_MIN_ROWS} rows")
    o = v["outside"]
    rep.add("", "(b) Close dates outside the window by year: " + "; ".join(
        f"{r.side} {r.year}: {r.n:,}" for r in o.itertuples(index=False)))
    pat = v["pat"]
    dow = pat["dow"]
    rep.table(dow[["name", "open", "n_days_open", "close", "n_days_close"]].round(0), max_rows=7,
              title=f"(c) Average per day by weekday ({years_label(pat['avg_years'])}; closings skip the first "
                    f"{pat['warmup']} days)")
    wa = pat["woy_avg"].dropna(subset=["open"])
    if len(wa):
        rep.add(f"(c) week of year (openings, {years_label(pat['avg_years'])}): highest week "
                f"{int(wa.loc[wa.open.idxmax(), 'week'])} ({wa.open.max():,.0f}/day), lowest week "
                f"{int(wa.loc[wa.open.idxmin(), 'week'])} ({wa.open.min():,.0f}/day); {len(wa)} weeks with complete "
                f"data | full years: {years_label(pat['full_years'])}; profile up to {rules.PROFILE_LAST_YEAR}")
    if len(v["later"]):
        rep.table(v["later"].round(0), title=f"(c) Years after {rules.PROFILE_LAST_YEAR}, shown apart (openings)")
    rep.emit()


def _pct(x, tot):
    return f"{100 * x / tot:.1f}%" if tot else "n/a"


def _later_lines(pat, later):
    """One bullet on the years after rules.PROFILE_LAST_YEAR, which the typical profile leaves out."""
    if later.empty:
        return []
    parts = []
    for r in later.itertuples(index=False):
        if r.n_weeks:
            vs = (f", {100 * (r.peak_open / r.typical_open - 1):+.0f}% vs the typical week {r.peak_week}"
                  if r.typical_open > 0 else "")
            cover = (f"; complete weeks {r.first_week}-{r.last_week} only" if (r.first_week, r.last_week) != (1, 52)
                     else "")
            text = f"{r.year}: openings peaked in week {r.peak_week} ({r.peak_open:,.0f} a day{vs}){cover}"
        else:
            text = f"{r.year}: no week with all 7 days usable"
        if isinstance(r.top_day, str):
            text += f"; busiest weekday {r.top_day} ({r.top_day_open:,.0f} a day)"
        parts.append(text)
    return [f"- **{years_label(pat['later_years'])}, shown apart** (after {rules.PROFILE_LAST_YEAR}; drawn as their "
            "own lines, not in the average): " + "; ".join(parts) + "."]


def _averages_sentence(pat):
    yrs = years_label(pat["avg_years"])
    if not pat["full_years"]:
        return f"No full calendar year is in the data, so averages use {yrs} as it is."
    text = f"Averages use the full calendar years {yrs}"
    if pat["later_years"]:
        text += f" (up to {rules.PROFILE_LAST_YEAR}); {years_label(pat['later_years'])} are drawn in color"
    other = pat["other_years"]
    if other:
        text += f"; {years_label(other)} ({'not a full year' if len(other) == 1 else 'not full years'}) gray"
    return text + "."


def _write_answer(cfg, v):
    d, start, end, unit, pat = v["daily"], v["start"], v["end"], v["unit"], v["pat"]
    chk = v["checks"].set_index("check")
    n_base = int(chk.rows["requests (default set)"])
    n_closed = int(chk.rows["with a close date"])
    sc = v["scored"]
    low, high = sc[sc.kind == "low"], sc[sc.kind == "high"]
    hol_low = low[low.holiday.notna()]
    inc_runs = v["inc_runs"]
    usual_open = d[~d.incomplete].n_open.median()
    sp = v["spans"]
    changed = sp[sp.change != ""]
    om = notable_pivot(v["open_mid"])
    cm = notable_pivot(v["close_mid"])
    chan = v["chan"].set_index("period")
    miss_col = "missing (UNKNOWN or NULL)"
    back = v["back"]
    ag = v["by_ag"]
    ag_big = ag[ag.n >= MIN_CELL_ROWS]
    bk = v["bulk"]
    out = v["outside"]
    ph = v["ph_status"]
    dow = pat["dow"].set_index("name")
    wa = pat["woy_avg"]
    yrs = years_label(pat["avg_years"])

    def per(p):
        return f"the week of {p}" if unit == "week" else str(p)

    def mid_line(t, below=DATE_ONLY_MIN_PCT):
        parts = []
        for agency, r in t.drop(index="(all)", errors="ignore").iterrows():
            r = r.dropna()
            if r.empty:
                continue
            ch = lasting_change(r, below)
            tail = f", below {below:.0f}% from {per(ch)} on" if ch is not None and ch != r.index[0] else ""
            parts.append(f"{agency} {r.iloc[0]:.1f}% in {per(r.index[0])} -> {r.iloc[-1]:.1f}% in "
                         f"{per(r.index[-1])}{tail}")
        return "; ".join(parts) or "none"

    def top_ag(col, k=3, fmt="{:.1f}%"):
        t = ag_big.sort_values(col, ascending=False).head(k)
        return "; ".join(f"{r.agency} {fmt.format(getattr(r, col))} of {int(r.n_closed):,} closed"
                         for r in t.itertuples(index=False) if getattr(r, col) and getattr(r, col) > 0)

    def odd_list(t, k=5):
        return "; ".join(f"{_d(r.date)} {int(r.n):,} vs {r.baseline:,.0f} usual"
                         + (f" ({r.holiday})" if isinstance(r.holiday, str) else "")
                         for r in t.itertuples(index=False)) if len(t) else "none"

    after = out[out.side == "after last opening"]
    before = out[out.side == "before first opening"]
    is_year = unit == "year"
    peak_w = wa.dropna(subset=["open"])
    week_line = (f"Openings peak in week {int(peak_w.loc[peak_w.open.idxmax(), 'week'])} "
                 f"({peak_w.open.max():,.0f} a day) and bottom out in week {int(peak_w.loc[peak_w.open.idxmin(), 'week'])}"
                 f" ({peak_w.open.min():,.0f} a day), averaged over {_avg_note(pat, start, end)}."
                 if len(peak_w) >= 12 else
                 f"Only {len(peak_w)} complete week(s) are available ({_d(start)} to {_d(end)}), so the "
                 "week-of-year panel shows coverage, not seasonality; the same code draws 52 weeks per year on the "
                 "full data.")
    lines = [
        "# Q2. Consistency of data collection over time",
        "",
        f"*Computed by `nyc311/q02_timing.py` from {n_base:,} requests (not `is_test` and not `dup_key`), "
        f"opened {_d(start)} to {_d(end)} ({len(d):,} days).*",
        "",
        "## (a) Daily volume of requests",
        "",
        f"![Daily 311 requests opened and closed, with flagged days marked](figs/{v['f_daily'].name})",
        "",
        f"- **Typical day.** A median of {usual_open:,.0f} requests a day are opened (days not flagged)."
        + (" " + outside_sentence(v["n_outside"]) if v["n_outside"] else ""),
        f"- **Gaps.** {int((d.n_open == 0).sum())} days in the window have no openings at all"
        + (": " + "; ".join(f"{_d(r[0])} to {_d(r[1])} ({r[2]} days)" for r in v["empty_runs"].sort_values(
            "days", ascending=False).head(5).itertuples(index=False)) if len(v["empty_runs"]) else "") + ".",
        f"- **Days the data can't fully show.** {int(d.incomplete.sum())} days ({int(d[d.incomplete].n_open.sum()):,} "
        "requests) are flagged `possibly_incomplete_day` and left out of daily and weekly averages:",
        *[f"  - {_d(r[0])}" + (f" to {_d(r[1])}" if r[1] != r[0] else "") + f" ({r[2]} days, {r[3]:,} rows): {r[4]}"
          for r in inc_runs.itertuples(index=False)],
        f"- **Missing files.** Batch numbers missing from the sequence: {', '.join(v['missing']) or 'none'}. "
        "Files cut off or unreadable: " + (", ".join(
            f"{r.file} ({int(r.n_records):,} rows kept)" for r in v["manifest"][~v["manifest"].parse_ok.astype(bool)]
            .itertuples(index=False)) or "none") + ".",
        f"- **Unusually low days** (vs. the median of the same weekday {BASELINE_WEEKS} weeks either side, robust z "
        f"<= -{ROBUST_Z}): "
        f"{len(low)} days, {len(hol_low)} of them federal holidays. Lowest: "
        + odd_list(low.sort_values("z").head(5)) + ".",
        f"- **Unusually high days** (robust z >= {ROBUST_Z}): {len(high)} days. Highest, with the complaint types "
        f"above their usual level (usual = average of the same weekday up to {BASELINE_WEEKS} weeks either side; "
        "shares are of the day's excess over it and pass 100% when other types run below their usual level): "
        + ("; ".join(
            f"{_d(r.date)} {r.n:,} vs {r.baseline:,.0f} usual ({r.drivers})" for r in v["sdays"].itertuples(index=False))
            or "none") + ".",
        "- **Surges by month** (at least "
        f"{SURGE_MIN_GROWTH:.0%} above the same month a year earlier): " + ("; ".join(
            f"{r.months}: {r.n:,} vs {r.n_year_before:,} (+{r.growth_pct:.0f}%), driven by {r.drivers}"
            for r in v["eps"].itertuples(index=False)) if len(v["eps"]) else
            "none found, or the data has no complete month a year apart to compare") + ".",
        f"- **Agencies entering or leaving** (first or last request more than {EDGE_DAYS} days inside the window): "
        + ("; ".join(f"{r.agency} {r.change} ({r.first_day} to {r.last_day}, {r.n:,} requests)"
                     for r in changed.head(10).itertuples(index=False)) or "none") + ".",
        f"- **Opening times recorded as dates only** (exactly midnight), by {unit}: " + mid_line(om) + ".",
        f"- **Channel not recorded** (`UNKNOWN` or NULL), by {unit}: "
        + (f"{chan[miss_col].iloc[0]:.1f}% in {per(chan.index[0])} -> {chan[miss_col].iloc[-1]:.1f}% in "
           f"{per(chan.index[-1])} (peak {chan[miss_col].max():.1f}% in {per(chan[miss_col].idxmax())})"
           if miss_col in chan.columns and len(chan) else "none") + ".",
        f"- **Backfilled records.** {int(back.n.sum()):,} requests sit in files whose other rows are more than "
        f"{rules.OUT_OF_WINDOW_DAYS} days away (`out_of_file_order`); largest groups: " + ("; ".join(
            f"{r.agency} opened in {r.year}: {r.n:,} (" + (r.first_file if r.first_file == r.last_file else
                                                            f"{r.first_file} to {r.last_file}") + ")"
            for r in back.head(5).itertuples(index=False))
            or "none") + ".",
        "",
        "## (b) Closing times",
        "",
        f"- **Coverage.** {n_closed:,} of {n_base:,} requests ({_pct(n_closed, n_base)}) have a close date; "
        f"{int(chk.rows['no close date']):,} have none.",
        f"- **Placeholders.** {int(chk.rows['1900 placeholder close date']):,} close dates are 1900-01-01; by status "
        "and agency: " + ("; ".join(f"status {r.status}, {r.agency}: {r.n:,}" for r in ph.head(4).itertuples(index=False))
                          or "none") + f". {int(ph[ph.status != 'Closed'].n.sum()):,} of them are on requests whose "
        "status is not Closed, so the placeholder stands in for a close that has not happened.",
        f"- **Closed before opened.** {int(chk.rows['closed before opened']):,} "
        f"({chk.pct['closed before opened']:.2f}% of closed); {int(chk.rows['... more than a day before']):,} by more "
        "than a day. Highest rates: " + (top_ag("pct_before_open", fmt="{:.2f}%") or "none") + ".",
        f"- **Closed at the second opened.** {int(chk.rows['closed at the second opened']):,} "
        f"({chk.pct['closed at the second opened']:.2f}% of closed). Highest rates: "
        + (top_ag("pct_at_creation", fmt="{:.2f}%") or "none") + ".",
        f"- **Close time recorded as a date only.** {int(chk.rows['close time exactly midnight']):,} closings "
        f"({chk.pct['close time exactly midnight']:.1f}%) are at exactly midnight; by {unit}: " + mid_line(cm) + ".",
        f"- **Closed long after opening.** {int(chk.rows['closed more than a year after opening']):,} requests "
        f"({chk.pct['closed more than a year after opening']:.2f}% of closed) closed more than a year after opening. "
        "Highest rates: " + (top_ag("pct_over_1yr", fmt="{:.2f}%") or "none") + ".",
        f"- **Close dates outside the window.** {int(chk.rows['closed after the last opening date']):,} closed after "
        f"the last opening date ({_d(end)}), i.e. they were still open when the data ends. By year of closing: " + ("; ".join(f"{r.year} {r.n:,}" for r in after.itertuples(index=False))
                                                  or "none")
        + f"; latest {_d(v['max_closed'])}. {int(chk.rows['close date later than today']):,} close dates are later "
        f"than today. {int(chk.rows['closed before the first opening date']):,} close dates fall before the first "
        "opening date" + (", by year: " + "; ".join(f"{r.year} {r.n:,}" for r in before.itertuples(index=False))
                          if len(before) else "") + ".",
        f"- **Bulk closures.** {len(bk)} agency-days closed at least {BULK_RATIO}x the agency's usual daily volume "
        f"(and {BULK_MIN_ROWS}+ rows). Largest: " + ("; ".join(
            f"{_d(r.date)} {r.agency} closed {r.n:,} vs {r.usual:,.0f} usual, {100 * r.top_share:.0f}% at one "
            f"timestamp ({pd.Timestamp(r.top_ts):%H:%M:%S})" for r in bk.head(5).itertuples(index=False))
            or "none found") + ".",
        "",
        "## (c) Openings and closings by day of week and week of year",
        "",
        f"![Average 311 openings and closings per day by weekday and week of the year](figs/{v['f_week'].name})",
        "",
        f"- **Day of week** ({yrs}): openings average {dow.open.max():,.0f} a day on {dow.open.idxmax()}s and "
        f"{dow.open.min():,.0f} on {dow.open.idxmin()}s ({100 * (dow.open.max() / dow.open.min() - 1):.0f}% more); "
        f"closings average {dow.close.max():,.0f} on {dow.close.idxmax()}s and {dow.close.min():,.0f} on "
        f"{dow.close.idxmin()}s ({100 * (dow.close.max() / dow.close.min() - 1):.0f}% more).",
        f"- **Week of year.** {week_line}",
        *_later_lines(pat, v["later"]),
        f"- **What the chart leaves out.** {int(d.incomplete.sum())} possibly incomplete days"
        + (f" and {pat['n_empty']:,} days with no rows at all" if pat["n_empty"] else "") + "; closings in the first "
        f"{pat['warmup']} days (the {CLOSE_WARMUP_QUANTILE:.0%} quantile of time to close), when requests opened "
        "before the data starts are missing; weeks with any unusable day. " + _averages_sentence(pat),
        "" if is_year else f"- **Too short for seasonality.** The data covers {len(d)} days; week-of-year effects "
        "need several years.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q02.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
