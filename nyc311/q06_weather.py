"""Q6: how complaint volume and content relate to the weather (Central Park daily weather, NOAA GHCN-Daily).

Daily complaint counts (total, by category, and for weather-plausible groups of canonical types) are merged with
that day's weather. Weather is separated from seasonality by comparing like with like: each series' daily
log(count + 1) is regressed on overnight-low bins and on rain/snow indicators (same day and day before), with
day-of-week, federal-holiday and year-month fixed effects, so a cold night is compared with other days of the same
month of the same year and the same weekday mix. Effects are % changes with 95% CIs (HAC standard errors).
Rows: complaints (rules.COMPLAINTS_SQL), on days not flagged possibly_incomplete_day. The seasonal profile for
staffing averages the full calendar years (clean.full_years) up to rules.PROFILE_LAST_YEAR; later years (the
pandemic) are shown separately so they don't move the typical peak.
Every number in text is computed at run time; the same code runs on one month or twelve years.
"""
from __future__ import annotations

import re
import textwrap
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.tseries.holiday import USFederalHolidayCalendar

from nyc311 import clean, geo, rules
from nyc311 import nytplot as nyt
from nyc311.clean import clean_glob
from nyc311.q05_top_types import profile_years, years_label
from nyc311.report import Report

STAGE = "q06_weather"
WEATHER_CSV = geo.DISTRICTS_GEOJSON.parent / "weather_central_park_daily.csv"

TMIN_EDGES = (20, 30, 40, 50, 60, 70)  # overnight-low bins (°F): <20, 20-29, ..., 70+
REF_TMIN = 55          # reference bin: the one holding 55°F (50-59°F on a full year), if it has MIN_BIN_DAYS days
MIN_BIN_DAYS = 5       # an edge bin with fewer days merges inward
MIN_EVENT_DAYS = 5     # a rain/snow indicator is estimated only if at least this many days are on (and off)
LIGHT_RAIN_IN, HEAVY_RAIN_IN = 0.1, 0.5
NO_SNOW_IN = 0.1       # a rain day has less snowfall than this (melted snow also counts as precipitation)
SNOW_IN = 1.0
HAC_LAGS = 7
MIN_SERIES_MEAN = 5    # categories averaging fewer complaints a day are not modeled
TOP_CATEGORIES = 8
MIN_MONTHS_FOR_SEASONS = 3
OUTLIER_RATIO = 4      # a scatter panel whose days above this x its 99.5th percentile ...
OUTLIER_Q = 0.995      # ... are drawn at a capped y axis (1.12 x the highest other day) and labeled
HEADROOM = 1.25        # y limit = this x the highest plotted value: room for the key and labels above the data
GRAY = nyt.HUES[-1]
LATER_HUES = nyt.HUES[2:5]  # years after rules.PROFILE_LAST_YEAR in the seasonal chart
INVALID_STEP = f"not service requests (category '{rules.INVALID_CATEGORY}')"

EVENTS = {
    "rain_light": f"Light rain ({LIGHT_RAIN_IN}-{HEAVY_RAIN_IN} in)",
    "rain_heavy": f"Heavy rain ({HEAVY_RAIN_IN}+ in)",
    "rain_heavy_lag1": "Heavy rain the day before",
    "snow": f"Snow ({SNOW_IN:g}+ in)",
    "snow_lag1": "Snow the day before",
}
DRIVERS = {"tmin_f": ("overnight low", "°F"), "prcp_in": ("precipitation", " in"),
           "snow_2day_in": ("snowfall, that day and the day before", " in")}
DRIVER_BINS = {"prcp_in": [0, LIGHT_RAIN_IN, HEAVY_RAIN_IN, np.inf], "snow_2day_in": [0, SNOW_IN, 6, np.inf]}


class Group(NamedTuple):
    key: str
    label: str
    where: str   # SQL predicate on the clean table's canonical_type / category
    driver: str  # weather column for the raw scatter


GROUPS = [
    Group("heat", "Heat & hot water", "category = 'Heat & hot water'", "tmin_f"),
    Group("plumbing", "Housing plumbing & leaks",
          "canonical_type ilike 'Housing - %' and (canonical_type ilike '%plumb%' or canonical_type ilike '%leak%')",
          "tmin_f"),
    Group("water", "Water mains & hydrants", "canonical_type = 'Water System'", "tmin_f"),
    Group("noise", "Noise", "category = 'Noise'", "tmin_f"),
    Group("sewer", "Sewer & flooding", "canonical_type = 'Sewer' or canonical_type ilike '%flood%'", "prcp_in"),
    Group("street", "Street condition (potholes)", "canonical_type in ('Street Condition', 'Highway Condition')",
          "prcp_in"),
    Group("trees", "Damaged & dead trees",
          "regexp_matches(canonical_type, '(?i)\\btrees?\\b') and canonical_type not ilike '%request%' "
          "and canonical_type not ilike '%illegal%'", "prcp_in"),
    Group("snowice", "Snow & ice", "canonical_type ilike '%snow%'", "snow_2day_in"),
]
LABELS = {"total": "All complaints", **{g.key: g.label for g in GROUPS}}


# ------------------------------------------------------------------ weather
def load_weather(path=WEATHER_CSV):
    return weather_features(pd.read_csv(path, parse_dates=["date"]))


def weather_features(w):
    """Add rain/snow indicators and day-before lags (taken from the calendar day before, not the previous row)."""
    f = w.sort_values("date").reset_index(drop=True).copy()
    no_snow = f.snow_in.fillna(0) < NO_SNOW_IN
    f["rain_light"] = no_snow & (f.prcp_in >= LIGHT_RAIN_IN) & (f.prcp_in < HEAVY_RAIN_IN)
    f["rain_heavy"] = no_snow & (f.prcp_in >= HEAVY_RAIN_IN)
    f["snow"] = f.snow_in >= SNOW_IN
    prev = f.set_index("date")[["rain_heavy", "snow", "snow_in"]].shift(1, freq="D")
    f["rain_heavy_lag1"] = f.date.map(prev.rain_heavy).eq(True)
    f["snow_lag1"] = f.date.map(prev.snow).eq(True)
    f["snow_2day_in"] = f.snow_in + f.date.map(prev.snow_in).fillna(0)
    return f


def _bin_label(lo, hi):
    if lo == -np.inf:
        return f"<{hi}°F"
    if hi == np.inf:
        return f"{lo}°F+"
    return f"{lo}-{hi - 1}°F"


def tmin_bins(tmin, edges=TMIN_EDGES, min_days=MIN_BIN_DAYS, ref_temp=REF_TMIN):
    """Ordered 10°F bins of the overnight low; edge bins with fewer than min_days days merge inward.

    Returns (labels per day as an ordered categorical Series, reference label)."""
    tmin = pd.Series(tmin).reset_index(drop=True)
    edges = list(edges)
    while edges:
        counts = pd.cut(tmin, [-np.inf, *edges, np.inf], right=False).value_counts(sort=False).to_numpy()
        if counts[0] < min_days:
            edges.pop(0)
        elif counts[-1] < min_days:
            edges.pop()
        else:
            break
    cuts = [-np.inf, *edges, np.inf]
    labels = [_bin_label(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])] if edges else ["all days"]
    cat = pd.cut(tmin, cuts, right=False, labels=labels, ordered=True)
    counts = cat.value_counts()
    ref = labels[int(np.searchsorted(edges, ref_temp, side="right"))]
    if counts[ref] < min_days:
        ref = counts.idxmax()
    return cat, ref


def short_bin(label):
    """'20-29°F' -> '20s', '<20°F' -> '<20', '70°F+' -> '70+' (axis ticks)."""
    s = label.replace("°F", "")
    return f"{s.split('-')[0]}s" if "-" in s else s


# ------------------------------------------------------------------ data
def group_membership(con, src):
    flags = ", ".join(f"coalesce(({g.where}), false) as {g.key}" for g in GROUPS)
    t = con.sql(f"""select coalesce(category, '(unmapped)') as category, canonical_type, count(*)::bigint as n, {flags}
                    from {src} where {rules.COMPLAINTS_SQL}
                    group by all""").df()
    rows = [t[t[g.key]].assign(group=g.key, label=g.label) for g in GROUPS]
    m = pd.concat(rows)[["group", "label", "category", "canonical_type", "n"]]
    return m.sort_values(["group", "n"], ascending=[True, False]).reset_index(drop=True)


def daily_counts(con, src, weather):
    """Daily complaint counts merged with weather, the category counts per day, and the exclusion steps."""
    flags = ", ".join(f"coalesce(({g.where}), false)::int as g_{g.key}" for g in GROUPS)
    sums = ", ".join(f"sum(g_{g.key})::bigint as {g.key}" for g in GROUPS)
    raw = con.sql(f"""
        with b as (select created::date as d, coalesce(is_info_request, false) as info,
                          ({rules.COMPLAINTS_SQL}) as complaint, possibly_incomplete_day as inc,
                          out_of_file_order as ooo, coalesce(category, '(unmapped)') as category, {flags}
                   from {src} where {rules.BASE_SQL}),
        cov as (select distinct d as cd from b where not ooo and d is not null)
        select b.d, b.info, b.complaint, b.inc, cov.cd is not null as covered, b.category, count(*)::bigint as n,
               {sums}
        from b left join cov on b.d = cov.cd group by all""").df()
    raw["d"] = pd.to_datetime(raw.d)
    steps = [("default analysis set", int(raw.n.sum()))]
    keep = raw

    def drop(label, mask):
        nonlocal keep
        steps.append((label, int(keep.n[mask].sum())))
        keep = keep[~mask]

    drop("information requests (is_info_request)", keep["info"])
    drop(INVALID_STEP, ~keep["complaint"].astype(bool))
    drop("no valid created date", keep.d.isna())
    drop("dates with only out_of_file_order rows (outside the files' period)", ~keep.covered)
    drop("possibly_incomplete_day", keep.inc)
    drop("no weather record for the date", ~keep.d.isin(weather.date))
    steps.append(("analyzed", int(keep.n.sum())))
    exclusions = pd.DataFrame(steps, columns=["step", "rows"])

    keys = [g.key for g in GROUPS]
    days = keep.groupby("d")[["n", *keys]].sum().rename(columns={"n": "total"}).reset_index()
    days = days.rename(columns={"d": "date"}).merge(weather, on="date", how="left").sort_values("date")
    cat = keep.groupby(["d", "category"], as_index=False).n.sum().rename(columns={"d": "date"})
    return days.reset_index(drop=True), cat, exclusions


# ------------------------------------------------------------------ model
def _controls(dates):
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    dow = pd.get_dummies(dates.dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    hol = USFederalHolidayCalendar().holidays(dates.min(), dates.max())
    holiday = dates.isin(hol).astype(float).rename("holiday")
    ym = pd.get_dummies(dates.dt.strftime("ym%Y_%m"), drop_first=True, dtype=float)
    return pd.concat([dow, holiday, ym], axis=1)


EFFECT_COLUMNS = ["series", "term", "kind", "days_with", "n_days", "mean_daily", "effect_pct", "ci_low_pct",
                  "ci_high_pct", "p_value"]


def fit_effects(days, ycols, bin_col, ref, events, min_event_days=MIN_EVENT_DAYS, hac_lags=HAC_LAGS):
    """OLS of log(1 + count) on temperature-bin dummies (vs ref) and event indicators, with weekday, holiday and
    year-month fixed effects; HAC (Newey-West) 95% CIs. Returns (effects, {skipped event: days on})."""
    d = days.sort_values("date").reset_index(drop=True)
    n = len(d)
    kept, skipped = [], {}
    for e in events:
        on = int(d[e].sum())
        if on >= min_event_days and n - on >= min_event_days:
            kept.append(e)
        else:
            skipped[e] = on
    present = d[bin_col].value_counts()
    bins = [b for b in d[bin_col].cat.categories if b != ref and present.get(b, 0) > 0]
    X = pd.DataFrame({b: (d[bin_col] == b).astype(float) for b in bins}, index=d.index)
    for e in kept:
        X[e] = d[e].astype(float)
    X = pd.concat([X, _controls(d.date)], axis=1)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "const", 1.0)
    rank = np.linalg.matrix_rank(X.to_numpy())
    terms = [t for t in [*bins, *kept] if t in X and np.linalg.matrix_rank(X.drop(columns=t).to_numpy()) < rank]
    for e in kept:
        if e not in terms:
            skipped[e] = int(d[e].sum())
    rows = []
    if n <= X.shape[1] + 2 or not terms:
        return pd.DataFrame(rows, columns=EFFECT_COLUMNS), skipped
    lags = min(hac_lags, max(1, n // 4))
    for y in ycols:
        res = sm.OLS(np.log1p(d[y].astype(float)), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=True)
        ci = res.conf_int(0.05)
        for t in terms:
            rows.append({"series": y, "term": t, "kind": "temperature" if t in bins else "event",
                         "days_with": int(X[t].sum()), "n_days": n, "mean_daily": float(d[y].mean()),
                         "effect_pct": 100 * np.expm1(res.params[t]), "ci_low_pct": 100 * np.expm1(ci.loc[t, 0]),
                         "ci_high_pct": 100 * np.expm1(ci.loc[t, 1]), "p_value": float(res.pvalues[t])})
    return pd.DataFrame(rows, columns=EFFECT_COLUMNS), skipped


def extreme_day(days, col, series, threshold):
    """The day with the most `col` (if >= threshold) and, per series, its count vs the median of days in the same
    year-month and day type (weekday vs weekend/holiday) with no heavy rain or snow that day or the day before."""
    if days[col].max() < threshold:
        return None
    d = days.reset_index(drop=True)
    i = d[col].idxmax()
    hol = USFederalHolidayCalendar().holidays(d.date.min(), d.date.max())
    off = (d.date.dt.dayofweek >= 5) | d.date.isin(hol)
    calm = ~(d.rain_heavy | d.snow | d.rain_heavy_lag1 | d.snow_lag1)
    same = (d.date.dt.to_period("M") == d.date[i].to_period("M")) & (off == off[i]) & calm & (d.index != i)
    rows = [(s, int(d.loc[i, s]), float(d.loc[same, s].median())) for s in series if same.any()]
    return {"date": d.date[i], "value": float(d.loc[i, col]), "n_baseline_days": int(same.sum()), "rows": rows}


def category_shares(days, cat, bin_order):
    m = cat.merge(days[["date", "tmin_bin"]], on="date")
    s = m.groupby(["tmin_bin", "category"], observed=True).n.sum().reset_index()
    s["share_pct"] = 100 * s.n / s.groupby("tmin_bin", observed=True).n.transform("sum")
    s["days"] = s.tmin_bin.map(days.tmin_bin.value_counts()).astype(int)
    s["tmin_bin"] = pd.Categorical(s.tmin_bin.astype(str), categories=bin_order, ordered=True)
    return s.sort_values(["tmin_bin", "n"], ascending=[True, False]).reset_index(drop=True)


# ------------------------------------------------------------------ text helpers
def _fmt_eff(r):
    return f"{r.effect_pct:+.0f}% ({r.ci_low_pct:+.0f}% to {r.ci_high_pct:+.0f}%)"


def _term_label(t):
    return EVENTS.get(t, f"{t} night")


def _sig(r):
    return r.ci_low_pct > 0 or r.ci_high_pct < 0


def _days(n):
    return f"{n:,} day" if n == 1 else f"{n:,} days"


def _d(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ figures
def _hide_baseline(ax):
    ax.spines["bottom"].set_color("#d9d9d9")


def axis_cap(values, ratio=OUTLIER_RATIO, q=OUTLIER_Q):
    """y-axis cap for a scatter panel, or None. Days above `ratio` x the q-quantile are off-scale outliers (a storm
    day can be 10x the next); the cap is 1.12 x the highest remaining day, so they no longer flatten the rest."""
    v = pd.Series(values, dtype=float).dropna()
    ref = v.quantile(q) if len(v) else 0
    if ref <= 0 or v.max() <= ratio * ref:
        return None
    return 1.12 * float(v[v <= ratio * ref].max())


def _key_row(ax, items, fontsize=10):
    """Series names in one row at the top right of the panel (in the headroom above the data), no legend box."""
    prev = None
    for text, color in reversed(items):
        kw = dict(color=color, fontfamily=nyt.SANS, fontweight="bold", fontsize=fontsize, ha="right", va="top",
                  zorder=5)
        if prev is None:
            prev = ax.text(1.0, 0.99, text, transform=ax.transAxes, **kw)
        else:
            prev = ax.annotate(text, xy=(0, 1), xycoords=prev, xytext=(-10, 0), textcoords="offset points", **kw)


def volume_figure(days, bin_order):
    """Daily counts against the day's weather, one panel per group (weekday dots sage, weekend/holiday gray, coral
    weekday mean by weather bin). Returns (figure, {group: (cap, off-scale days)} for panels with a capped axis)."""
    nyt.use()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    hol = USFederalHolidayCalendar().holidays(days.date.min(), days.date.max())
    off = (days.date.dt.dayofweek >= 5) | days.date.isin(hol)
    small = len(days) < 200
    capped = {}
    for ax, g in zip(axes.flat, GROUPS):
        what, unit = DRIVERS[g.driver]
        cap = axis_cap(days[g.key])
        y = days[g.key] if cap is None else days[g.key].clip(upper=cap)
        over = pd.Series(False, index=days.index) if cap is None else days[g.key] > cap
        for mask, color, z, a_small, a_big in [(off, GRAY.band, 2, 0.9, 0.4), (~off, nyt.SAGE.line, 3, 0.85, 0.35)]:
            m = mask & ~over
            ax.scatter(days.loc[m, g.driver], y[m], color=color, s=18 if small else 6, alpha=a_small if small else a_big,
                       linewidths=0, zorder=z)
        wk = days[~off]
        groups = wk.groupby("tmin_bin", observed=True) if g.driver == "tmin_f" else \
            wk.groupby(pd.cut(wk[g.driver], DRIVER_BINS[g.driver], right=False), observed=True)
        means = groups.agg(x=(g.driver, "mean"), y=(g.key, "mean"), k=(g.key, "size"))
        means = means[means.k >= MIN_BIN_DAYS]
        if len(means) >= 2:
            ax.plot(means.x, means.y, color=nyt.CORAL.line, lw=2, zorder=4)
        ax.margins(x=0.04)
        top = max(float(y.max()) if len(y) else 0.0, float(means.y.max()) if len(means) >= 2 else 0.0) or 1.0
        ax.set_ylim(0, top * HEADROOM)
        if cap is not None:
            out = days[over].sort_values(g.key, ascending=False)
            capped[g.key] = (cap, out)
            ax.scatter(out[g.driver], [cap] * len(out), marker="^", s=34, color=nyt.SAGE.line, linewidths=0, zorder=5,
                       clip_on=False)
            r = out.iloc[0]
            more = f" (+{len(out) - 1} more off scale)" if len(out) > 1 else " (off scale)"
            x0, x1 = ax.get_xlim()
            right = (r[g.driver] - x0) / (x1 - x0) > 0.5
            ax.annotate(f"{_d(r.date)}: {int(r[g.key]):,}{more}", (r[g.driver], cap), xytext=(-4 if right else 4, 7),
                        textcoords="offset points", ha="right" if right else "left", va="bottom", fontsize=8,
                        color=nyt.NOTE_GRAY, fontfamily=nyt.SANS, zorder=6)
        nyt.format_y(ax, nbins=4)
        nyt.format_x(ax, suffix=unit, nbins=4)
        nyt.panel_title(ax, f"{g.label}\nvs {what}", fontsize=10.5)
    key_ax = next((ax for ax, g in zip(axes.flat, GROUPS) if g.key not in capped), axes.flat[0])
    _key_row(key_ax, [("Weekdays", nyt.SAGE.line), ("Weekends & holidays", GRAY.line)])
    nyt.title(fig, f"Daily complaints against the day's weather, {_d(days.date.min())} to {_d(days.date.max())}")
    cap_note = (f" Triangles: days off the scale (more than {OUTLIER_RATIO}x the {100 * OUTLIER_Q:g}th percentile), "
                "plotted at the top." if capped else "")
    nyt.note(fig, textwrap.fill(
        f"Note: each dot is one day ({len(days):,} days). Coral line: weekday mean by weather bin (bins with at least "
        f"{MIN_BIN_DAYS} days).{cap_note} Raw counts also carry season and weekday; see q06_temperature_effects for "
        "like-with-like comparisons. Weather: NOAA GHCN-Daily, Central Park.", 160))
    return fig, capped


def fig_volume_vs_weather(days, bin_order, path):
    fig, capped = volume_figure(days, bin_order)
    nyt.save(fig, path)
    return capped


def _effect_panel(ax, xs, sub, ref_x=None):
    ax.axhline(0, color="#333333", lw=1, zorder=1)
    _hide_baseline(ax)
    sig = sub.apply(_sig, axis=1) if len(sub) else pd.Series(dtype=bool)
    for x, r, s in zip(xs, sub.itertuples(), sig):
        h = nyt.CORAL if s else nyt.SAGE
        ax.plot([x, x], [r.ci_low_pct, r.ci_high_pct], color=h.band if not s else h.line, lw=2.2, zorder=2)
        ax.scatter([x], [r.effect_pct], color=h.line, s=22, zorder=3, linewidths=0)
    if ref_x is not None:
        ax.scatter([ref_x], [0], facecolor="white", edgecolor=nyt.SAGE.line, s=22, zorder=3, linewidths=1.2)


def fig_temperature_effects(eff, bin_order, ref, series, path, n_days, period):
    nyt.use()
    cols = 3
    rows = int(np.ceil(len(series) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.5 * rows + 1), sharex=True)
    pos = {b: i for i, b in enumerate(bin_order)}
    for ax, s in zip(axes.flat, series):
        sub = eff[(eff.series == s) & (eff.kind == "temperature")]
        _effect_panel(ax, [pos[t] for t in sub.term], sub, ref_x=pos[ref])
        nyt.format_y(ax, suffix="%", nbins=4)
        nyt.panel_title(ax, LABELS.get(s, s), fontsize=10.5)
    for ax in axes.flat[len(series):]:
        ax.set_visible(False)
    for ax in axes.flat:
        ax.set_xticks(range(len(bin_order)))
        ax.set_xticklabels([short_bin(b) for b in bin_order])
        ax.set_xlim(-0.5, len(bin_order) - 0.5)
    heat = eff[(eff.series == "heat") & (eff.term == bin_order[0])]
    if len(heat) and bin_order[0] != ref:
        r = heat.iloc[0]
        head = (f"Same month, same weekday: heat complaints run {r.effect_pct:+.0f}% on {bin_order[0]} nights "
                f"vs {ref} nights")
    else:
        head = f"Complaints by overnight low, vs {ref} nights, same month and weekday"
    nyt.title(fig, head)
    nyt.note(fig, textwrap.fill(
        f"Note: % difference in daily complaints vs days with an overnight low of {ref} (hollow dot), with 95% "
        f"confidence intervals; coral where the interval excludes zero. Regression of log daily count on low-"
        f"temperature bins, rain and snow, with weekday, federal-holiday and year-month controls (HAC, {HAC_LAGS} "
        f"lags). {n_days:,} days, {period}. Weather: Central Park (NOAA).", 140))
    return nyt.save(fig, path)


def fig_event_effects(eff, terms, series, path, n_days, period):
    nyt.use()
    fig, axes = plt.subplots(1, len(terms), figsize=(3.1 * len(terms) + 2.2, 0.42 * len(series) + 1.6),
                             sharey=True, squeeze=False)
    fig.get_layout_engine().set(wspace=0.12)
    ys = np.arange(len(series))[::-1]
    for ax, t in zip(axes.flat, terms):
        sub = eff[(eff.term == t)].set_index("series").reindex(series)
        ax.axvline(0, color="#333333", lw=1, zorder=1)
        for y, s in zip(ys, series):
            if s not in sub.index or pd.isna(sub.loc[s, "effect_pct"]):
                continue
            r = sub.loc[s]
            h = nyt.CORAL if (r.ci_low_pct > 0 or r.ci_high_pct < 0) else nyt.SAGE
            ax.plot([r.ci_low_pct, r.ci_high_pct], [y, y], color=h.line if h is nyt.CORAL else h.band, lw=2.2)
            ax.scatter([r.effect_pct], [y], color=h.line, s=22, zorder=3, linewidths=0)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.spines["bottom"].set_visible(False)
        ax.margins(x=0.12)
        nyt.format_x(ax, suffix="%", nbins=4)
        nyt.panel_title(ax, f"{EVENTS[t]}\n({int(sub.days_with.max())} days)", fontsize=10.5)
    axes.flat[0].set_yticks(ys)
    axes.flat[0].set_yticklabels([LABELS.get(s, s) for s in series])
    sig = eff[eff.term.isin(terms) & eff.series.isin(series)]
    sig = sig[sig.apply(_sig, axis=1)]
    if len(sig):
        r = sig.loc[sig.effect_pct.abs().idxmax()]
        head = f"{LABELS.get(r.series, r.series)}: {r.effect_pct:+.0f}% on days with {EVENTS[r.term].lower()}"
    else:
        head = "Rain and snow: no effect whose 95% interval excludes zero"
    nyt.title(fig, head)
    nyt.note(fig, textwrap.fill(
        f"Note: % difference in daily complaints vs days without the event, same month and weekday, holding the "
        f"overnight low fixed; 95% confidence intervals, coral where they exclude zero. {n_days:,} days, {period}.",
        120))
    return nyt.save(fig, path)


def fig_content(shares, cats, bin_order, path, period):
    nyt.use()
    rows = int(np.ceil(len(cats) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(13, 2.6 * rows + 1.2), sharex=True)
    x = np.arange(len(bin_order))
    piv = shares.pivot_table(index="category", columns="tmin_bin", values="share_pct", observed=False).reindex(
        columns=bin_order).fillna(0)
    days = shares.drop_duplicates("tmin_bin").set_index("tmin_bin").days.reindex(bin_order)
    for ax, c in zip(axes.flat, cats):
        h = nyt.CORAL if c == "Heat & hot water" else nyt.SAGE
        ax.bar(x, piv.loc[c], color=h.band, width=0.7)
        for xi, val in zip(x, piv.loc[c]):
            ax.text(xi, val, f"{val:.0f}%" if val >= 10 else f"{val:.1f}%", ha="center", va="bottom",
                    color=h.line, fontsize=9.5)
        ax.set_ylim(0, piv.loc[c].max() * 1.2)
        nyt.format_y(ax, suffix="%", nbins=4)
        nyt.panel_title(ax, c, fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels([short_bin(b) for b in bin_order])
    for ax in axes.flat[len(cats):]:
        ax.set_visible(False)
    swing = (piv.loc[cats, bin_order[-1]] - piv.loc[cats, bin_order[0]])
    c = swing.abs().idxmax()
    head = (f"{c} is {piv.loc[c, bin_order[0]]:.0f}% of complaints on {bin_order[0]} nights and "
            f"{piv.loc[c, bin_order[-1]]:.0f}% on {bin_order[-1]} nights")
    nyt.title(fig, head)
    nyt.note(fig, textwrap.fill(
        "Note: share of each day's complaints by category, pooled over days in each overnight-low band (°F; days per "
        "band: " + ", ".join(f"{short_bin(b)} {int(n)}" for b, n in days.items()) + f"). {period}. Raw shares: a "
        "band can mix seasons and weekdays; the like-with-like comparison is in q06_temperature_effects.", 150))
    return nyt.save(fig, path)


SERIES = ["total", *[g.key for g in GROUPS]]


def seasonal_profile(days, years):
    """Mean complaints per day by calendar month over `years` (months with at least MIN_BIN_DAYS days)."""
    m = days[days.date.dt.year.isin(years)].assign(month=lambda x: x.date.dt.month)
    prof = m.groupby("month")[SERIES].mean()
    prof["days"] = m.groupby("month").size()
    return prof[prof.days >= MIN_BIN_DAYS]


def later_profiles(days, years):
    """The same by year for each later year (shown next to the typical profile, not in it)."""
    m = days[days.date.dt.year.isin(years)].assign(year=lambda x: x.date.dt.year, month=lambda x: x.date.dt.month)
    out = m.groupby(["year", "month"])[SERIES].mean()
    out["days"] = m.groupby(["year", "month"]).size()
    return out[out.days >= MIN_BIN_DAYS].reset_index()


def _full_years_from_days(days):
    """Years with every calendar day in `days` (used when the caller has no clean table, e.g. tests)."""
    y = days.date.dt.year
    return [int(k) for k, n in y.value_counts().items() if n == (366 if pd.Timestamp(int(k), 12, 31).dayofyear == 366
                                                                 else 365)]


def _month(m, fmt="%B"):
    return pd.Timestamp(2000, int(m), 1).strftime(fmt)


def fig_seasons(prof, later, path, avg_years, other_years):
    """Bars: typical mean per day by month (full years up to rules.PROFILE_LAST_YEAR); dots: each later year."""
    nyt.use()
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.3), sharex=True)
    years = sorted(later.year.unique()) if len(later) else []
    hue_of = {y: LATER_HUES[i % len(LATER_HUES)] for i, y in enumerate(years)}
    for ax, s in zip(axes.flat, SERIES):
        h = nyt.CORAL if s == "heat" else nyt.SAGE
        ax.bar(prof.index, prof[s], color=h.band, width=0.7, zorder=2)
        top = prof[s].idxmax()
        ax.bar([top], [prof.loc[top, s]], color=h.line, width=0.7, zorder=2)
        hi = prof[s].max()
        for y in years:
            g = later[later.year == y]
            ax.scatter(g.month, g[s], s=16, color=hue_of[y].line, zorder=3, linewidths=0)
            hi = max(hi, g[s].max())
        ax.set_ylim(0, (hi if hi > 0 else 1) * HEADROOM)
        nyt.format_y(ax, nbins=4)
        nyt.panel_title(ax, LABELS[s], fontsize=10.5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([_month(mo, "%b")[0] for mo in range(1, 13)])
        ax.set_xlim(0.4, 12.6)
    yrs = years_label(avg_years)
    items = [(f"Bars: average, {yrs}" if "," not in yrs else f"Bars: average of {len(avg_years)} years",
              nyt.SAGE.line)] + [(str(y), hue_of[y].line) for y in years]
    prev = None
    for text, color in items:  # the key, in the headroom of the first panel
        kw = dict(color=color, fontfamily=nyt.SANS, fontweight="bold", fontsize=9.5, ha="left", va="top", zorder=5)
        prev = (axes.flat[0].text(0.01, 0.99, text, transform=axes.flat[0].transAxes, **kw) if prev is None else
                axes.flat[0].annotate(text, xy=(1, 1), xycoords=prev, xytext=(8, 0), textcoords="offset points", **kw))
    nyt.title(fig, f"Average complaints per day by calendar month, {yrs}")
    dots = (f" Dots: {years_label(years)}, shown apart and not in the average (pandemic years)." if years else "")
    other = (f" {years_label(other_years)} left out (not a full year)." if other_years else "")
    nyt.note(fig, textwrap.fill(
        f"Note: bars are mean daily complaints by calendar month over the full calendar years {yrs}; the darkest bar "
        f"is the peak month.{dots}{other} Months with fewer than {MIN_BIN_DAYS} usable days are left out.", 150))
    return nyt.save(fig, path)


# ------------------------------------------------------------------ run
def run(cfg):
    con = cfg.connect()
    src = clean_glob(cfg)
    weather = load_weather()
    groups = group_membership(con, src)
    days, cat, exclusions = daily_counts(con, src, weather)
    return build(cfg.out_dir, days, cat, groups, exclusions, full_years=clean.full_years(con, src))


def build(out_dir, days, cat, groups, exclusions, full_years=None):
    """Model, figures, tables, report and answers from the daily tables (separate from run() so tests can feed
    synthetic multi-year data). full_years: clean.full_years of the clean table (default: years with every day in
    `days`); the seasonal profile averages those up to rules.PROFILE_LAST_YEAR and shows later years apart."""
    days = days.copy()
    days["tmin_bin"], ref = tmin_bins(days.tmin_f)
    bin_order = [b for b in days.tmin_bin.cat.categories if (days.tmin_bin == b).any()]

    cat_wide = cat.pivot_table(index="date", columns="category", values="n", aggfunc="sum", fill_value=0)
    cat_wide = cat_wide.reindex(days.date).fillna(0)
    model_cats = [c for c in cat_wide.columns if cat_wide[c].mean() >= MIN_SERIES_MEAN]
    frame = pd.concat([days.set_index("date"), cat_wide[model_cats]], axis=1).reset_index()
    series = ["total", *[g.key for g in GROUPS]]
    eff, skipped = fit_effects(frame, series + model_cats, "tmin_bin", ref, list(EVENTS))
    eff.insert(1, "label", eff.series.map(lambda s: LABELS.get(s, s)))
    shares = category_shares(days, cat, bin_order)
    full = sorted(full_years) if full_years is not None else sorted(_full_years_from_days(days))
    avg_years, later_years = profile_years(full, days.date.dt.year.unique())
    other_years = sorted(int(y) for y in days.date.dt.year.unique() if y not in avg_years and y not in later_years)
    prof = seasonal_profile(days, avg_years)
    later = later_profiles(days, later_years)

    agg, figs = out_dir / "agg", out_dir / "figs"
    agg.mkdir(parents=True, exist_ok=True)
    days.assign(tmin_bin=days.tmin_bin.astype(str)).to_csv(agg / "q06_daily.csv", index=False)
    eff.to_csv(agg / "q06_effects.csv", index=False)
    shares.to_csv(agg / "q06_shares_by_temperature.csv", index=False)
    groups.to_csv(agg / "q06_groups.csv", index=False)
    exclusions.to_csv(agg / "q06_exclusions.csv", index=False)

    period = f"{_d(days.date.min())} to {_d(days.date.max())}"
    n_days = len(days)
    capped = fig_volume_vs_weather(days, bin_order, figs / "q06_volume_vs_weather.png")
    fig_temperature_effects(eff, bin_order, ref, series, figs / "q06_temperature_effects.png", n_days, period)
    event_terms = [t for t in EVENTS if t in set(eff.term)]
    event_path = figs / "q06_rain_snow_effects.png"
    if event_terms:
        fig_event_effects(eff, event_terms, series, event_path, n_days, period)
    elif event_path.exists():
        event_path.unlink()  # stale figure from an earlier run on other data
    top_cats = cat.groupby("category").n.sum().sort_values(ascending=False).index[:TOP_CATEGORIES].tolist()
    fig_content(shares, top_cats, bin_order, figs / "q06_content_by_temperature.png", period)
    season_path = figs / "q06_seasonal_profile.png"
    if len(prof) >= MIN_MONTHS_FOR_SEASONS:
        fig_seasons(prof, later, season_path, avg_years, other_years)
        pd.concat([prof.reset_index().assign(year=f"typical {years_label(avg_years)}"),
                   later.assign(year=later.year.astype(str))], ignore_index=True).to_csv(
            agg / "q06_seasonal_profile.csv", index=False)
    elif season_path.exists():
        season_path.unlink()

    extremes = {"rain": extreme_day(days, "prcp_in", ["sewer", "trees", "street", "total"], HEAVY_RAIN_IN),
                "snow": extreme_day(days, "snow_in", ["snowice", "street", "heat", "total"], SNOW_IN)}
    written = {p.name for p in figs.glob("q06_*.png")}
    v = dict(days=days, eff=eff, skipped=skipped, ref=ref, bin_order=bin_order, shares=shares, groups=groups,
             exclusions=exclusions, prof=prof, later=later, avg_years=avg_years, later_years=later_years,
             other_years=other_years, full_years=full, period=period, top_cats=top_cats, model_cats=model_cats,
             event_terms=event_terms, extremes=extremes, written=written, capped=capped)
    _report(out_dir, v)
    _write_answer(out_dir, v)
    return {"days": days, "category_daily": cat, "effects": eff, "skipped_events": skipped, "ref": ref,
            "bin_order": bin_order, "shares": shares, "groups": groups, "exclusions": exclusions,
            "seasonal": prof, "seasonal_later": later, "profile_years": avg_years, "capped": capped}


def _key_terms(v):
    bins = [b for b in v["bin_order"] if b != v["ref"]]
    temps = [bins[0]] + ([bins[-1]] if len(bins) > 1 and bins[-1] != bins[0] else []) if bins else []
    return temps + v["event_terms"]


def _effects_table(eff, series, terms):
    rows = []
    for s in series:
        row = {"series": LABELS.get(s, s)}
        sub = eff[eff.series == s].set_index("term")
        for t in terms:
            row[short_bin(t) if t not in EVENTS else t] = _fmt_eff(sub.loc[t]) if t in sub.index else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def _report(out_dir, v):
    days, eff, ref = v["days"], v["eff"], v["ref"]
    rep = Report(STAGE, out_dir)
    rep.add(f"Q6. Complaints vs Central Park weather, {v['period']}: {days.total.sum():,} complaints on "
            f"{len(days):,} days")
    rep.table(v["exclusions"], title="Rows (default set -> analyzed)")
    rep.add("", f"Weather on analyzed days: overnight low {days.tmin_f.min():.0f} to {days.tmin_f.max():.0f}°F; "
            f"precipitation {days.prcp_in.max():.2f} in max; " + "; ".join(
                f"{EVENTS[e].lower()}: {_days(int(days[e].sum()))}" for e in EVENTS))
    bins = days.groupby("tmin_bin", observed=True).agg(days=("total", "size"), mean_total=("total", "mean"),
                                                        mean_heat=("heat", "mean")).reset_index()
    rep.table(bins, title=f"Overnight-low bins (reference: {ref}; edge bins with < {MIN_BIN_DAYS} days merged)")
    m = v["groups"]
    g = pd.DataFrame([{"group": x.label, "rows": int(m.n[m.group == x.key].sum()),
                       "largest canonical types": ", ".join(f"{r.canonical_type} {r.n:,}" for r in
                                                            m[m.group == x.key].head(4).itertuples())}
                      for x in GROUPS])
    rep.table(g, title="Weather-plausible groups (from canonical_type / category)")
    terms = _key_terms(v)
    rep.table(_effects_table(eff, ["total", *[x.key for x in GROUPS]], terms),
              title=f"Effect on daily complaints, % vs {ref} night / no event (95% CI); weekday, holiday, "
                    "year-month controls")
    if v["skipped"]:
        rep.add(f"not estimated (< {MIN_EVENT_DAYS} days on, or not identified): " + ", ".join(
            f"{e} ({_days(n)})" for e, n in v["skipped"].items()))
    if terms:
        cats = eff[(eff.series.isin(v["model_cats"])) & (eff.term == terms[0])].sort_values("effect_pct")
        rep.table(cats[["series", "mean_daily", "effect_pct", "ci_low_pct", "ci_high_pct"]], max_rows=20,
                  title=f"Categories (>= {MIN_SERIES_MEAN}/day): {terms[0]} night vs {ref}")
    piv = v["shares"].pivot_table(index="category", columns="tmin_bin", values="share_pct", observed=True)
    rep.table(piv.loc[v["top_cats"]].round(1).reset_index(), title="Category share of complaints (%) by overnight low")
    if v["capped"]:
        rep.add("", "volume chart, y-axes capped (days off scale): " + "; ".join(
            f"{LABELS[k]} at {cap:,.0f}: " + ", ".join(f"{_d(r.date)} {int(r[k]):,}" for _, r in out.head(3).iterrows())
            + (f" (+{len(out) - 3} more)" if len(out) > 3 else "") for k, (cap, out) in v["capped"].items()))
    prof = v["prof"]
    if len(prof) >= MIN_MONTHS_FOR_SEASONS:
        rows = []
        for s in SERIES:
            r = {"series": LABELS[s], "peak": _month(prof[s].idxmax(), "%b"), "peak_day": prof[s].max(),
                 "low": _month(prof[s].idxmin(), "%b"), "low_day": prof[s].min()}
            for y in v["later_years"]:
                g = v["later"][v["later"].year == y].set_index("month")[s]
                r[f"peak_{y}"] = f"{_month(g.idxmax(), '%b')} {g.max():,.0f}" if len(g) else "-"
            rows.append(r)
        rep.table(pd.DataFrame(rows).round(0), title=f"Seasonal profile, mean per day: full years "
                  f"{years_label(v['avg_years'])}" + (f"; shown apart: {years_label(v['later_years'])}"
                                                       if v["later_years"] else "")
                  + (f"; left out (not full): {years_label(v['other_years'])}" if v["other_years"] else ""))
    rep.emit()


def _fig(name, caption, v):
    return [f"![{caption}](figs/{name})", ""] if name in v["written"] else []


def _extreme_line(x, what):
    if x is None:
        return None
    parts = "; ".join(f"{LABELS[s]} {n:,} vs a median of {b:,.0f}" for s, n, b in x["rows"])
    return (f"- **{what}: {_d(x['date'])} ({x['value']:.2f} in).** Complaints that day vs the median of "
            f"{_days(x['n_baseline_days'])} in the same month and day type without heavy rain or snow: {parts}. "
            "One day is an example, not an estimate.")


def _capped_lines(v):
    if not v["capped"]:
        return []
    parts = [f"{LABELS[k]} {_d(out.date.iloc[0])} ({int(out[k].iloc[0]):,})"
             + (f" and {len(out) - 1} more day{'s' if len(out) > 2 else ''}" if len(out) > 1 else "")
             for k, (cap, out) in v["capped"].items()]
    return [f"- **Off-scale days.** In the chart below, a panel's y-axis is capped when a few days exceed "
            f"{OUTLIER_RATIO}x its {100 * OUTLIER_Q:g}th percentile; those days are drawn as triangles at the top: "
            + "; ".join(parts) + "."]


def _season_lines(v):
    """Typical peak and low month per series (full years up to rules.PROFILE_LAST_YEAR), then each later year's
    peak month against the typical level of that month."""
    prof, later = v["prof"], v["later"]
    head = f"- **Typical year** = the average of the full calendar years {years_label(v['avg_years'])}"
    if v["later_years"]:
        head += (f"; {years_label(v['later_years'])} (pandemic years) are shown apart and not averaged, so they don't "
                 "move the typical peak")
    if v["other_years"]:
        head += f"; {years_label(v['other_years'])} left out (not a full year)"
    lines = [head + "."]
    for s in SERIES:
        line = (f"- **{LABELS[s]}:** peaks in {_month(prof[s].idxmax())} ({prof[s].max():,.0f} a day), lowest in "
                f"{_month(prof[s].idxmin())} ({prof[s].min():,.0f} a day)")
        parts = []
        for y in v["later_years"]:
            g = later[later.year == y].set_index("month")[s]
            if g.empty:
                continue
            pm, typ = g.idxmax(), prof[s].get(g.idxmax(), np.nan)
            vs = f", {100 * (g.max() / typ - 1):+.0f}% vs a typical {_month(pm)}" if typ > 0 else ""
            span = (f"; {_month(g.index.min(), '%b')}-{_month(g.index.max(), '%b')} only" if len(g) < 12 else "")
            parts.append(f"{y} peaked in {_month(pm)} ({g.max():,.0f} a day{vs}{span})")
        lines.append(line + ("; " + "; ".join(parts) if parts else "") + ".")
    return lines


def _md(text):
    """Column and flag names (snake_case) as `code`, so markdown renderers don't read underscores as italics."""
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: f"`{m.group(0)}`", text)


def _write_answer(out_dir, v):
    days, eff, ref, bins, ex = v["days"], v["eff"], v["ref"], v["bin_order"], v["exclusions"].set_index("step")
    n_months = days.date.dt.to_period("M").nunique()
    total = int(days.total.sum())
    coldest, warmest = bins[0], bins[-1]
    removed = ex.iloc[1:-1]
    series = ["total", *[g.key for g in GROUPS]]
    lines = [
        "# Q6. Complaints and the weather",
        "",
        f"*Computed by `nyc311/q06_weather.py` from {total:,} complaints, {v['period']}.*",
        "",
        "## Data",
        "",
        f"- **Merged** {total:,} complaints on {_days(len(days))} with NOAA daily weather for Central Park "
        f"(overnight low, precipitation, snowfall). Removed from the default set's {int(ex.rows.iloc[0]):,} rows: " +
        "; ".join(f"{_md(k)} {int(r):,}" for k, r in removed.rows.items()) + ".",
        f"- **Weather in these days:** overnight lows {days.tmin_f.min():.0f}°F to {days.tmin_f.max():.0f}°F; " +
        "; ".join(f"{EVENTS[e].lower()} on {_days(int(days[e].sum()))}" for e in EVENTS) + ".",
        "- **Weather-plausible groups** of canonical types, complaints on these days (full list of types in "
        "`outputs/agg/q06_groups.csv`): " + "; ".join(
            f"{g.label} {int(days[g.key].sum()):,}" for g in GROUPS) + ".",
        *_capped_lines(v),
        "",
        *_fig("q06_volume_vs_weather.png", "Daily complaints against the day's weather", v),
        "## Weather vs volume, like with like",
        "",
        f"- **Method.** For each series, log(1 + daily complaints) is regressed on overnight-low bins (reference "
        f"{ref}), rain and snow on the day and the day before, with day-of-week, federal-holiday and year-month "
        f"fixed effects; 95% CIs use HAC standard errors. A rain or snow indicator is estimated only with at least "
        f"{MIN_EVENT_DAYS} days on. Effects are % differences in daily complaints.",
    ]
    for s in series:
        sub = eff[eff.series == s]
        if sub.empty:
            continue
        sig = sub[sub.apply(_sig, axis=1)]
        top = sig.loc[sig.effect_pct.abs().idxmax()] if len(sig) else None
        parts = [f"{_term_label(r.term)} {_fmt_eff(r)}" for r in sub.itertuples()
                 if r.term in (coldest, warmest) or r.kind == "event"]
        if top is None:
            lead = "no weather term's 95% CI excludes zero. "
        elif len(parts) > 1:
            lead = f"largest effect whose CI excludes zero: {_term_label(top.term)} {_fmt_eff(top)}. Key terms: "
        else:
            lead = ""
        lines.append(f"- **{LABELS[s]}** (mean {sub.mean_daily.iloc[0]:,.1f} a day): {lead}" + "; ".join(parts) + ".")
    if v["skipped"]:
        lines.append(f"- **Not estimated** (fewer than {MIN_EVENT_DAYS} days on, or not separable from the "
                     "controls): " + "; ".join(f"{EVENTS[e].lower()} ({_days(n)})" for e, n in v["skipped"].items())
                     + ".")
    for key, what in [("rain", "Wettest day"), ("snow", "Snowiest day")]:
        line = _extreme_line(v["extremes"][key], what)
        if line:
            lines.append(line)
    lines += ["", *_fig("q06_temperature_effects.png", "Effect of the overnight low, same month and weekday", v)]
    lines += _fig("q06_rain_snow_effects.png", "Effect of rain and snow, same month and weekday", v)
    cat_eff = eff[eff.series.isin(v["model_cats"]) & (eff.term == coldest)] if coldest != ref else eff.iloc[:0]
    piv = v["shares"].pivot_table(index="category", columns="tmin_bin", values="share_pct", observed=True)
    swing = (piv[warmest] - piv[coldest]).loc[v["top_cats"]].sort_values()
    lines += [
        "## Content: what people complain about",
        "",
        f"- **Shares by overnight low** ({coldest} vs {warmest} nights): " + "; ".join(
            f"{c} {piv.loc[c, coldest]:.1f}% vs {piv.loc[c, warmest]:.1f}%" for c in swing.index) + ". These "
        "bands mix seasons when the data span several months, so they describe content rather than weather effects.",
    ]
    if len(cat_eff):
        up = cat_eff[cat_eff.ci_low_pct > 0].sort_values("effect_pct", ascending=False)
        down = cat_eff[cat_eff.ci_high_pct < 0].sort_values("effect_pct")
        lines.append(f"- **Within the same month** ({coldest} vs {ref} nights, categories averaging at least "
                     f"{MIN_SERIES_MEAN} a day), CI above zero: " +
                     ("; ".join(f"{r.series} {_fmt_eff(r)}" for r in up.itertuples()) or "none") +
                     ". CI below zero: " +
                     ("; ".join(f"{r.series} {_fmt_eff(r)}" for r in down.itertuples()) or "none") + ".")
    lines += ["", *_fig("q06_content_by_temperature.png", "Category shares by overnight low", v)]
    if len(v["prof"]) >= MIN_MONTHS_FOR_SEASONS:
        lines += ["## Seasonal pattern, for staffing", "", *_season_lines(v), "",
                  *_fig("q06_seasonal_profile.png", "Average complaints per day by calendar month", v)]
    lines += [
        "## Limits",
        "",
        (f"- **Coverage.** This run has {_days(len(days))} in {n_months} calendar "
         f"month{'s' if n_months != 1 else ''}. It cannot show seasonal patterns: every effect comes from day-to-day "
         "swings within the month, rare events (heavy rain, snow) have too few days to estimate, and CIs are wide. "
         f"Cold days can also coincide with calendar effects: the {coldest} days are "
         + ", ".join(_d(x) for x in days.date[days.tmin_bin == coldest]) + ". The same code on a multi-year pull "
         "estimates every bin and event."
         if n_months < 12 else
         f"- **Coverage.** {_days(len(days))} across {days.date.dt.year.nunique()} years. Year-month fixed effects "
         "absorb seasonality, the 2020 surge and changes in how 311 was used, so effects come from weather swings "
         "within a month."),
        "- **Association, not causation.** Weather can change reporting as well as conditions (people stay in "
        "during storms and go out on warm nights). One station (Central Park) stands in for all five boroughs. The "
        "heat and plumbing groups depend on how HPD codes complaint types.",
        "",
    ]
    path = out_dir / "answers" / "q06.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
