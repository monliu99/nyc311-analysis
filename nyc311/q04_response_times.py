"""Q4: time to closure as the response-time KPI.

a) summary statistics and outliers; b) community districts with longer than typical times, raw and after adjusting
for complaint type and year; c) whether agencies differ once complaint type is accounted for (statsmodels on a
fixed-seed sample). Every number in the report, the charts and outputs/answers/q04.md is computed here.

Complaint set: the default analysis set (not is_test, not dup_key) minus information requests (is_info_request).
Durations (duration_h) exist only for status Closed and closed >= opened. Parts b) and c) also leave out
"instant" closures: closed at the second they were opened, with a time of day (likely administrative). Same-day
closures recorded as dates only (both timestamps midnight, mainly HPD) stay in.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
COMPLAINTS = "not is_test and not dup_key and not coalesce(is_info_request, false)"
INSTANT = "closed_at_creation and not date_only_created"
STAT_SETS = [
    ("all closed complaints", "true"),
    ("excluding closed at creation", "not closed_at_creation"),
    ("exact timestamps only", "not date_only_created and not date_only_closed"),
    ("used in b) and c)", f"not ({INSTANT})"),
]
QUANTILES = {"p10_h": 0.10, "p25_h": 0.25, "median_h": 0.50, "p75_h": 0.75, "p90_h": 0.90, "p99_h": 0.99}


def fmt_h(h):
    """Hours as text: hours below two days, days above."""
    if h is None or not np.isfinite(h):
        return "n/a"
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
                       count_if(not is_test and not dup_key and coalesce(is_info_request, false)) n_info,
                       count_if({COMPLAINTS}) n_complaints from {src}""").df().iloc[0].astype(int)
    con.execute(f"""create or replace temp table q4 as
        select created, year(created) as year, agency, canonical_type, borough, boro_cd, status,
               closed is null as no_close_date, duration_h, unresolved, closed_before_created, closed_at_creation,
               closed_after_1yr, date_only_created, date_only_closed
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
    qs = ", ".join(f"quantile_cont(duration_h, {q}) as {c}" for c, q in QUANTILES.items())
    return f"count(*) as n, avg(duration_h) as mean_h, stddev_samp(duration_h) as sd_h, {qs}"


def summary_stats(con):
    parts = [f"select '{name}' as set, {_stats_select()} from q4 where duration_h is not null and {w}"
             for name, w in STAT_SETS]
    df = con.sql(" union all ".join(parts)).df()
    df["order"] = df.set.map({name: i for i, (name, _) in enumerate(STAT_SETS)})
    df = df[df.n > 1].sort_values("order").drop(columns="order").reset_index(drop=True)
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
    c = con.sql(f"""select count_if(duration_h = 0), count_if(duration_h = 0 and date_only_created),
                    count_if(duration_h = 0 and not date_only_created),
                    count_if(duration_h > 0 and duration_h < {lo!r}), count_if(duration_h > {hi!r}),
                    count_if(closed_after_1yr)
                    from q4 where duration_h is not null""").fetchone()
    rows = [
        ("closed at creation (0 hours)", c[0], "closed == opened"),
        ("  0 hours, dates only (closed the same day)", c[1], "both timestamps midnight"),
        ("  0 hours, exact second with a time of day", c[2], "likely administrative closures"),
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
                    when closed_before_created then 'closed before it was opened'
                    when trim(status) = 'Closed' and no_close_date then 'status Closed, no close date'
                    when status is null then 'no status'
                    else 'other status: ' || status end as reason, count(*) n
        from q4 group by 1 order by n desc""").df()
    df["pct"] = 100 * df.n / df.n.sum()
    return df


def by_agency(con):
    return con.sql(f"""
        select agency, count(*) n_complaints, count(duration_h) n, 100 * avg(unresolved::int) pct_unresolved,
               avg(duration_h) / 24 mean_d, quantile_cont(duration_h, 0.5) / 24 median_d,
               quantile_cont(duration_h, 0.5) filter (where duration_h > 0) / 24 median_d_excl_zero,
               stddev_samp(duration_h) / 24 sd_d, quantile_cont(duration_h, 0.9) / 24 p90_d,
               100 * count_if(duration_h = 0) / count(duration_h) pct_zero_hours,
               100 * count_if(duration_h is not null and (date_only_created or date_only_closed))
                   / count(duration_h) pct_day_precision
        from q4 group by agency having count(duration_h) >= {MIN_N} order by n desc""").df()


def by_year(con):
    return con.sql("""
        select year, count(*) n_complaints, count(duration_h) n, 100 * avg(unresolved::int) pct_unresolved,
               quantile_cont(duration_h, 0.5) median_h,
               quantile_cont(duration_h, 0.5) filter (where not closed_at_creation) median_h_excl_zero,
               avg(duration_h) mean_h, quantile_cont(duration_h, 0.9) p90_h
        from q4 group by year order by year""").df()


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


def agency_effects(df, resid):
    """For complaint types that two or more agencies each handle >= MIN_N_SHARED times in the sample: each agency's
    mean residual (ln scale) from the type + year + borough model, as % faster/slower than typical for the same type."""
    d = pd.DataFrame({"canonical_type": df.canonical_type.to_numpy(), "agency": df.agency.to_numpy(), "e": resid})
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
    fig = Figure(figsize=(8, 0.32 * len(ypos) + 1.9))
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
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=1.0, zorder=3)
    ax.set_xlim(-span * 1.25 if vals.min() < 0 else 0, span * 1.25 if vals.max() > 0 else 0)
    _hbar_axes(ax)
    nyt.format_x(ax, suffix="%")
    if hidden:
        _text(ax, 0, k + 0.1, f"{hidden} districts in between", nyt.NOTE_GRAY, ha="center", fontsize=9,
              fontstyle="italic", zorder=4, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    # series names sit on the empty side of the zero line, next to their group
    pos, neg = ypos[vals > 0], ypos[vals < 0]
    if len(pos):
        _text(ax, -0.03 * span, pos.mean(), "Slower than typical", nyt.CORAL.line, ha="right", fontweight="bold",
              fontsize=12)
    if len(neg):
        _text(ax, 0.03 * span, neg.mean(), "Faster than typical", nyt.SAGE.line, ha="left", fontweight="bold",
              fontsize=12)
    nyt.title(fig, "Where 311 complaints take longer than typical to close")
    nyt.note(fig, note)
    return nyt.save(fig, path)


def chart_agencies(a, v, path, title, note, max_agencies=12):
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    a = a.head(max_agencies).sort_values("median_d")
    fig = Figure(figsize=(7.5, 0.3 * (len(a) + len(v)) + 2.6))
    top, bot = fig.subplots(2, 1, gridspec_kw={"height_ratios": [len(a), len(v) + 0.5]})
    top.barh(np.arange(len(a)), a.median_d, color=nyt.SAGE.band, height=0.72, zorder=2)
    for i, (d, h) in enumerate(zip(a.median_d, a.median_d * 24)):
        _text(top, d + 0.01 * a.median_d.max(), i, fmt_h(h), nyt.SAGE.line, fontsize=9, fontweight="bold")
    top.set_yticks(np.arange(len(a)), a.agency)
    top.set_xlim(0, a.median_d.max() * 1.22)
    _hbar_axes(top, left=True)
    nyt.format_x(top, suffix=" days")
    nyt.panel_title(top, "Median time to close, by agency")

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
    years = by_year(con)
    precision = con.sql("""select 100 * avg((date_only_created or date_only_closed)::int),
                           100 * avg((date_only_created and date_only_closed)::int)
                           from q4 where duration_h is not null""").fetchone()
    n_instant = con.sql(f"select count(*) from q4 where duration_h is not null and {INSTANT}").fetchone()[0]
    n_resp = con.sql("select count(*) from q4r").fetchone()[0]
    instant_by_agency = con.sql(f"""select agency, count(*) n from q4 where duration_h is not null and {INSTANT}
                                    group by 1 order by n desc limit 4""").df()

    districts = district_table(con, names)
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
    effects = agency_effects(sample, resid)
    coverage = agency_coverage(con)
    wt = within_type(con)

    res = {"summary": summary, "fences": f, "outliers": outl, "type_outliers": tout, "no_duration": nod,
           "by_agency": agencies, "by_year": years, "districts": districts, "rank_compare": cmp,
           "boroughs": boroughs, "variance": variance, "agency_effects": effects, "agency_coverage": coverage,
           "within_type": wt, "n_response_set": int(n_resp), "excl": excl, "period": period,
           "period_long": period_l, "precision": precision, "n_instant": int(n_instant),
           "instant_by_agency": instant_by_agency, "n_no_cd": int(n_no_cd)}

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    for name in ["summary", "outliers", "type_outliers", "no_duration", "by_agency", "by_year", "boroughs",
                 "variance", "agency_effects", "agency_coverage", "within_type"]:
        res[name].to_csv(agg / f"q04_{name}.csv", index=False)
    districts.drop(columns="short").to_csv(agg / "q04_districts.csv", index=False)

    figs = cfg.out_dir / "figs"
    chart_districts(districts, figs / "q04_district_adjusted.png", _district_note(res))
    title, note = _agency_chart_text(res)
    chart_agencies(agencies, variance, figs / "q04_agency_variance.png", title, note)
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
            f"Excludes information requests and {res['n_instant']:,} instant (0-second) closures.")


def _agency_chart_text(res):
    r2a, r2t = _r2(res, AGENCY_ONLY), _r2(res, "complaint type")
    gain = _r2(res, NESTED_MODELS[-1]) - _r2(res, NESTED_MODELS[-2])
    if r2t >= r2a:
        title = f"Complaint type explains more of the gap than agency does: {r2t:.0%} vs. {r2a:.0%}"
    else:
        title = f"Agency explains more of the variation than complaint type: {r2a:.0%} vs. {r2t:.0%}"
    n = int(res["variance"].n.iloc[0])
    note = (f"Note: Closed complaints, {res['period']}; agencies with at least {MIN_N}. Variation = R\N{SUPERSCRIPT TWO} "
            f"of log(hours + 1) on a fixed-seed sample of {n:,};\nadding agency to complaint type, year and borough "
            f"adds {100 * gain:.1f} points.")
    return title, note


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    e, s = res["excl"], res["summary"]
    rep.add(f"Q4. Time to closure on {e.n_complaints:,} complaints ({res['period_long']})",
            f"rows {e.n_rows:,}; left out: is_test {e.n_test:,}, dup_key {e.n_dup:,}, is_info_request {e.n_info:,}",
            f"b) and c) also leave out {res['n_instant']:,} instant closures (0 s, with a time of day); "
            f"response set {res['n_response_set']:,}")
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
    rep.table(res["by_agency"], max_rows=12, title=f"By agency (>= {MIN_N} closed), days")
    rep.table(res["by_year"], max_rows=13, title="By year opened (adj_pct_vs_first: same type and borough, sample)")

    rep.table(res["boroughs"][["borough", "n", "median_h", "adj_pct", "pct_unresolved"]],
              title="b) Boroughs: raw median hours and % vs typical for the same type and year")
    d, cmp = res["districts"], res["rank_compare"]
    rep.add(f"{len(d)} districts with >= {MIN_N} closed complaints; {res['n_no_cd']:,} response-set rows have no "
            f"standard district (special areas or unlocated)")
    rep.table(d.sort_values("rank_adjusted")[["district", "n", "median_h", "rank_raw", "adj_pct", "rank_adjusted",
                                              "pct_unresolved"]], max_rows=TOP_K,
              title=f"Slowest {TOP_K} districts after adjustment (rank_raw = rank by raw median)")
    lab = dict(zip(d.boro_cd, d.district))
    rep.add(f"rank correlation raw vs adjusted: {cmp['spearman']:.2f}",
            f"in raw top {cmp['k']} only: " + ("; ".join(lab[c] for c in cmp["only_raw"]) or "none"),
            f"in adjusted top {cmp['k']} only: " + ("; ".join(lab[c] for c in cmp["only_adjusted"]) or "none"))

    rep.table(res["variance"], title="c) Variance in ln(hours + 1) explained (statsmodels, fixed-seed sample)")
    cov = res["coverage_summary"] = _coverage_summary(res)
    rep.add(f"agencies with a complaint type another agency also handles: {cov['n_shared']} of {cov['n_agencies']} "
            f"({cov['pct_closures']:.1f}% of closures are in shared types)")
    rep.table(res["agency_effects"], max_rows=10,
              title="Agency vs typical for the same complaint type, year and borough (shared types only, sample)")
    rep.table(res["within_type"][["canonical_type", "agency", "n", "first_year", "last_year", "median_h",
                                  "type_median_h", "pct_vs_type"]], max_rows=10,
              title="Same complaint type, different agencies (full response set)")
    rep.emit()


def _coverage_summary(res):
    cov = res["agency_coverage"]
    shared_n = (cov.n * cov.pct_volume_in_shared_types / 100).sum()
    return {"n_agencies": len(cov), "n_shared": int((cov.pct_volume_in_shared_types > 0).sum()),
            "pct_closures": 100 * shared_n / cov.n.sum() if cov.n.sum() else 0.0}


def _write_answer(cfg, res):
    e, s = res["excl"], res["summary"].set_index("set")
    al = s.loc["all closed complaints"]
    o = res["outliers"].set_index("kind")
    f = res["fences"]
    a = res["by_agency"]
    y = res["by_year"]
    d, cmp, b = res["districts"], res["rank_compare"], res["boroughs"]
    v = res["variance"].set_index("model").r2
    eff, wt = res["agency_effects"], res["within_type"]
    cov = res.get("coverage_summary") or _coverage_summary(res)
    nod = res["no_duration"].set_index("reason")
    n_open = int(nod.n.get("still open (unresolved)", 0))
    ex0 = s.loc["excluding closed at creation"] if "excluding closed at creation" in s.index else al
    exact = s.loc["exact timestamps only"] if "exact timestamps only" in s.index else None
    lab = dict(zip(d.boro_cd, d.district))
    tout = res["type_outliers"]
    top_out = tout.sort_values("pct_outside", ascending=False).head(3)
    inst = res["instant_by_agency"]

    fast, slow = a.sort_values("median_d").iloc[0], a.sort_values("median_d").iloc[-1]
    hpd = a.set_index("agency").loc["HPD"] if "HPD" in set(a.agency) else None
    big_years = y[y.n >= MIN_N]
    if len(big_years) >= 2:
        y0, y1 = big_years.iloc[0], big_years.iloc[-1]
        adj = (f"; holding complaint type and borough fixed, {int(y1.year)} closures took "
               f"{_pct(y1.adj_pct_vs_first)} vs. {int(y0.year)}" if np.isfinite(y1.adj_pct_vs_first) else "")
        others = big_years.iloc[:-1].pct_unresolved.median()
        year_line = (f"- **By year.** Median time to close went from {fmt_h(y0.median_h)} ({int(y0.year)}) to "
                     f"{fmt_h(y1.median_h)} ({int(y1.year)}){adj}. {int(y1.year)} is still {y1.pct_unresolved:.1f}% "
                     f"unresolved (median of earlier years {others:.1f}%): requests opened late in the data have had "
                     "less time to close, so the latest year's closed-only median is biased down.")
    else:
        only = f" ({int(big_years.year.iloc[0])})" if len(big_years) else ""
        year_line = (f"- **By year.** Only {len(big_years)} year{only} has at least {MIN_N} closed complaints, so "
                     "the data cannot show a trend; rerun on the full data.")

    raw_slow = d.sort_values("rank_raw").head(3)
    adj_slow = d.sort_values("rank_adjusted").head(3)
    adj_fast = d.sort_values("rank_adjusted").tail(3).iloc[::-1]
    sig_fast = eff[eff.ci_high < 0]
    sig_slow = eff[eff.ci_low > 0]
    gain = v[NESTED_MODELS[-1]] - v[NESTED_MODELS[-2]]

    def dlist(rows, col):
        return "; ".join(f"{r.district} {_days(r.median_h) if col == 'raw' else _pct(r.adj_pct)}"
                         for r in rows.itertuples())

    def elist(rows):
        return "; ".join(f"{r.agency} {_pct(r.pct_vs_typical)} (95% CI {_pct(r.ci_low)} to {_pct(r.ci_high)}, "
                         f"n {r.n_shared:,}; {r.shared_types})" for r in rows.itertuples()) or "none"

    wt_lines = []
    for t, g in list(wt.groupby("canonical_type", sort=False))[:4]:
        wt_lines.append(f"{t}: " + ", ".join(f"{r.agency} {fmt_h(r.median_h)} (n {r.n:,}, {_years(r)})"
                                             for r in g.itertuples()))
    lines = [
        "# Q4. Response times",
        "",
        f"*Drafted by `nyc311/q04_response_times.py` from {e.n_complaints:,} complaints, {res['period_long']}. "
        "Every number is computed; write the prose from these bullets.*",
        "",
        "## Q4a. Summary statistics and outliers",
        "",
        f"- **What is measured.** The data have no first-response time, so the KPI is time to closure (closed minus "
        f"opened), defined only for status Closed with a close date on or after the opening. {int(al.n):,} of "
        f"{e.n_complaints:,} complaints ({100 * al.n / e.n_complaints:.1f}%) have one. Left out: "
        f"{e.n_info:,} information requests (`is_info_request`), {e.n_test:,} test records (`is_test`), "
        f"{e.n_dup:,} duplicate keys (`dup_key`).",
        f"- **Summary.** Mean {fmt_h(al.mean_h)}, median {fmt_h(al.median_h)}, standard deviation {fmt_h(al.sd_h)}; "
        f"10th percentile {fmt_h(al.p10_h)}, 25th {fmt_h(al.p25_h)}, 75th {fmt_h(al.p75_h)}, 90th {fmt_h(al.p90_h)}, "
        f"99th {fmt_h(al.p99_h)}. The mean is {al.mean_h / al.median_h:.1f} times the median: durations are heavily "
        "right-skewed, so the dashboard should report medians and percentiles, not averages.",
        f"- **Zero-hour closures.** {int(o.n.iloc[0]):,} closures ({o.pct.iloc[0]:.1f}%) are at the second of "
        f"opening: {int(o.n.iloc[2]):,} with a time of day (likely administrative; mostly "
        + ", ".join(f"{r.agency} {r.n:,}" for r in inst.itertuples()) +
        f") and {int(o.n.iloc[1]):,} same-day closures recorded as dates only. Without all of them the median is "
        f"{fmt_h(ex0.median_h)} (mean {fmt_h(ex0.mean_h)}).",
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
        + (f". Using only durations with exact timestamps ({int(exact.n):,}), the median is {fmt_h(exact.median_h)}."
           if exact is not None else "."),
        f"- **Unresolved requests bias closed-only numbers.** {n_open:,} complaints "
        f"({100 * n_open / e.n_complaints:.1f}%) are still open and "
        f"{e.n_complaints - int(al.n) - n_open:,} more have no usable close date (closed before opened, no date, "
        "other status). They are missing from every duration statistic, and they are likely the slow ones, so "
        "closed-only medians understate the true time to closure.",
        f"- **By agency** (at least {MIN_N} closures): fastest median {fast.agency} {fmt_h(fast.median_d * 24)}, "
        f"slowest {slow.agency} {fmt_h(slow.median_d * 24)} ({slow.median_d / fast.median_d:.0f} times longer). "
        f"Share still open ranges from {a.pct_unresolved.min():.1f}% to {a.pct_unresolved.max():.1f}% "
        f"({a.sort_values('pct_unresolved').iloc[-1].agency} highest).",
        year_line,
        "",
        "## Q4b. Neighborhoods with longer than typical times",
        "",
        f"- **Method.** {len(d)} community districts with at least {MIN_N} closed complaints (special areas such as "
        f"parks and airports, and unlocated requests: {res['n_no_cd']:,} closures) are left out. Raw ranking: median time to "
        "close. Adjusted ranking: each closure is compared with the citywide typical time (geometric mean of hours "
        "+ 1) for the same complaint type and year, and the district's average gap is reported as % longer or "
        "shorter. Instant closures are left out.",
        f"- **Raw medians** put {dlist(raw_slow, 'raw')} slowest.",
        f"- **After adjusting for complaint mix,** the slowest are {dlist(adj_slow, 'adj')}; the fastest are "
        f"{dlist(adj_fast, 'adj')}. Districts' adjusted gaps range from {_pct(d.adj_pct.min())} to "
        f"{_pct(d.adj_pct.max())}.",
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
        f"- **Raw gap.** Agency medians range from {fmt_h(fast.median_d * 24)} ({fast.agency}) to "
        f"{fmt_h(slow.median_d * 24)} ({slow.agency}).",
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
        f"{MIN_N_SHARED} closures each); those shared types are {cov['pct_closures']:.1f}% of closures. For every "
        "other agency, 'exemplary' can only mean fast relative to its own complaint mix (its trend over time or its "
        "own targets), not faster than a peer.",
        "- **Where two agencies do the same work** (vs typical for the same type, year and borough; sample): "
        "faster: " + elist(sig_fast) + "; slower: " + elist(sig_slow) + ". A shared type can still hide "
        "different work (one agency may close by referral, or a merged type may cover two programs), so check the "
        "descriptors before calling an agency exemplary.",
        "- **Direct comparison, same complaint type:** " + "; ".join(wt_lines) + ". Check the years: an agency "
        "that handled a type only in some years (renamed agencies, transferred programs) is partly a time "
        "comparison.",
        "- **Caveats.** Closure time measures when the ticket was closed, not whether the problem was fixed; "
        "agencies differ in when they close tickets (instant and date-only closures above). The comparisons are "
        "associations, not causes; the confidence intervals ignore clustering, so treat small gaps as noise.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q04.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
