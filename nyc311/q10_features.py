"""Q10: which features would predict daily complaint volume, and do the same features move with response times?

Target: daily complaints (`rules.COMPLAINTS_SQL`) on complete days (possibly_incomplete_day and dates outside the files'
period left out), merged with Central Park weather.

Forecast, one week ahead. The baseline is the mean count of the last BASE_WEEKS same weekdays (7, 14, 21 and 28 days
earlier; at least MIN_BASE_WEEKS of them complete). Models predict log(count / baseline) with no intercept, so with no
features the forecast is the baseline itself and the level always comes from recent weeks (no long-run trend). Feature
groups: holidays (that day, and the share of the baseline days that were holidays), month (12 indicators: the typical
drift away from the baseline weeks), weather (high and low temperature relative to the baseline days; precipitation and
snowfall that day, the day before and on the baseline days). Each test year is predicted by models fit on the
TRAIN_YEARS calendar years before it, because the relations drift (the holiday dip and the weekday pattern shrink over
the years). The last N_TYPICAL test years up to `rules.PROFILE_LAST_YEAR` are pooled as the typical case; later years
(the pandemic) are reported one by one as a structural break. Benchmarks next to every model: last week's same-weekday
count, the mean of the last 28 days, the training-window average, and a calendar-and-weather model of log(count)
without recent counts.

A group's value is the drop in mean absolute percentage error (MAPE) when it is added to a forecast that lacks it:
recent volume = calendar model -> full model; day of week = 28-day mean -> same-weekday baseline; holidays, month and
weather = baseline -> baseline + group (and, as a check, the rise in error when each is dropped from the full model).
When the data have no later year to test on (the 5-week local sample) the module says why and reports what needs no
training (the naive forecasts) and in-sample associations only.

Response times: the daily median time to close and the daily mix-adjusted time to close (Q4's method) against a
benchmark that does not absorb the feature tested: the complaint type and year for day-level features (weekday,
holiday, season, weather, volume) and the complaint type alone for the year-level trend and 2020 regime (full calendar
years only). Every feature is classified by one test, OLS with Newey-West (HAC) standard errors, with a window of up to
HAC_LAGS days for day-level features and YEAR_HAC_LAGS days for year-level ones; Spearman's rho shows the size of each
link. Every number in text is computed at run time.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
import textwrap

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from nyc311 import geo, rules
from nyc311.clean import clean_glob
from nyc311.clean import full_years as clean_full_years
from nyc311.report import Report

STAGE = "q10_features"
WEATHER_CSV = geo.DISTRICTS_GEOJSON.parent / "weather_central_park_daily.csv"
REGIME_START = dt.date(2020, 3, 22)   # New York State on PAUSE (executive order 202.8) took effect
BASE_WEEKS = 4              # baseline: mean of the counts 7, 14, 21 and 28 days earlier (same weekday)
MIN_BASE_WEEKS = 3          # ... of which at least this many must be complete days
LOOKBACK_DAYS = 28          # weekday-blind benchmark: mean of the last 28 days
MIN_LOOKBACK_DAYS = 21
TRAIN_YEARS = 3             # models are fit on the calendar years just before the test year (rolling window)
MIN_TRAIN_DAYS = 365        # usable training days before a test year
MIN_TEST_DAYS = 90          # usable days in a test year
N_TYPICAL = 3               # the last test years up to rules.PROFILE_LAST_YEAR, pooled as the typical case
MIN_GAIN = 0.5              # MAPE points: a smaller drop in error is not called a gain
MIN_EVENT_DAYS = 3          # holidays (or weekend/regime days) needed to estimate an indicator
MIN_WEATHER_DAYS = 30
MIN_REGIME_DAYS = 60        # regime: days needed on each side of REGIME_START
MIN_TREND_SPAN_DAYS = 730   # trend: two years, or it is confounded with the season
MIN_FULL_YEARS = 2          # year-level features are compared across full calendar years
MIN_WET_DAYS = 5            # an amount (rain, snow) is tested only with this many days at or above its threshold
WET_THRESHOLD = {"prcp_in": 0.1, "prcp_lag1": 0.1, "snow_in": 1.0, "snow_lag1": 1.0}   # inches, as in Q6
MIN_DAYS_PER_PARAM = 10     # a joint (in-sample) model needs this many days per parameter
MIN_CLOSED_DAY = 30         # closed complaints a day needs for its response-time values
HAC_LAGS = 7                # day-level features: cap; the lag length follows Newey-West's rule floor(4 (n/100)^(2/9))
YEAR_HAC_LAGS = 365         # year-level features: their residuals persist for months, so a year-long window
MIN_RELIABLE_DAYS = 90      # fewer days: tests are flagged as descriptive (little power, HAC SEs biased down)
MIN_NAIVE_DAYS = 5          # days needed to report the naive forecasts' error in short mode
FINAL_DAYS = 60             # complaints opened in the final days had the least time to close
ALPHA = 0.05
WEATHER_COLS = ["tmax_f", "tmin_f", "prcp_in", "snow_in", "prcp_lag1", "snow_lag1"]
WEATHER_REL = ["tmax_rel", "tmin_rel", "prcp_in", "snow_in", "prcp_lag1", "snow_lag1", "prcp_base", "snow_base"]
REQUIRED = ["log_n", "base", "lag7", "mean28", *WEATHER_COLS, "tmax_rel", "tmin_rel", "prcp_base", "snow_base"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_COLS = [f"dow_{i}" for i in range(1, 7)]

# Feature groups fitted on log(count / baseline), no intercept.
GROUPS = {
    "holiday": ("Holidays", ["holiday", "hol_base"]),
    "season": ("Month", [f"m_{i}" for i in range(1, 13)]),
    "weather": ("Weather", WEATHER_REL),
}
# The benchmark without recent counts: log(count) on weekday, holiday, month and weather, with an intercept.
CAL_COLS = [*DOW_COLS, "holiday", *(f"m_{i}" for i in range(2, 13)), *WEATHER_COLS]
VALUE_GROUPS = {
    "volume": "Recent volume",
    "dow": "Day of week",
    "holiday": "Holidays",
    "season": "Month",
    "weather": "Weather",
}
HOW = {
    "volume": "calendar-and-weather model without recent counts -> full model",
    "dow": f"mean of the last {LOOKBACK_DAYS} days -> mean of the last {BASE_WEEKS} same weekdays",
    "holiday": "baseline -> baseline + holidays",
    "season": "baseline -> baseline + month",
    "weather": "baseline -> baseline + weather",
}
HOW_TEXT = {
    "volume": "a calendar-and-weather model without recent counts vs the full model",
    "dow": f"the mean of the last {LOOKBACK_DAYS} days vs the mean of the last {BASE_WEEKS} same weekdays",
    "holiday": "added to the baseline",
    "season": "added to the baseline",
    "weather": "added to the baseline",
}
BREAK_RATIO = 1.5           # a later year whose calendar-model error is 1.5x the typical one is called a break
DRIFT_POINTS = 10           # a change of 10+ percentage points (weekend share, holiday dip) is called a drift
MODELS = {   # the first eight are the chart's rows, in this order
    "calendar": "Calendar and weather, no recent counts",
    "mean28": f"Mean of the last {LOOKBACK_DAYS} days",
    "lag7": "Same weekday last week",
    "base": f"Mean of the last {BASE_WEEKS} same weekdays (baseline)",
    "base+holiday": "Baseline + holidays",
    "base+season": "Baseline + month",
    "base+weather": "Baseline + weather",
    "full": "Baseline + holidays, month, weather",
    "full-holiday": "Full model without holidays",
    "full-season": "Full model without month",
    "full-weather": "Full model without weather",
    "constant": "Training-window average",
    "full_all": "Full model fit on every earlier year",
}
CHART_MODELS = list(MODELS)[:8]
NAIVE = ["mean28", "lag7", "base"]
WHY = {
    "volume": "recent weeks carry the current level (growth, campaigns, outbreaks, backlogs, a pandemic) that a "
              "calendar cannot know, and they are known a week ahead",
    "dow": "agencies, inspectors and many callers work weekdays, so filing follows the work week; matching the weekday "
           "puts that pattern into the baseline",
    "holiday": "offices close and routines break, so a holiday (or a holiday among the baseline days) pulls the count "
               "away from the recent same-weekday level",
    "season": "volume drifts with the season (heat complaints with the heating season, noise in summer), so the count "
              "moves away from the level of the previous weeks",
    "weather": "cold snaps drive heat and hot-water calls and storms drive flooding, tree and street calls (Q6); a real "
               "forecast would have to use forecast weather",
}
# (column, label, scale, unit, level) for the response-time tests; scale converts a coefficient to "per unit".
RT_FEATURES = [
    ("weekend", "Weekend (Sat-Sun)", 1.0, "vs weekdays", "day"),
    ("holiday", "Federal holiday", 1.0, "vs other days", "day"),
    ("heat_season", "Heating season (Oct-May)", 1.0, "vs Jun-Sep", "day"),
    ("tmax_f", "High temperature", 10.0, "per 10\N{DEGREE SIGN}F", "day"),
    ("tmin_f", "Low temperature", 10.0, "per 10\N{DEGREE SIGN}F", "day"),
    ("prcp_in", "Precipitation", 1.0, "per inch", "day"),
    ("snow_in", "Snowfall", 1.0, "per inch", "day"),
    ("prcp_lag1", "Precipitation the day before", 1.0, "per inch", "day"),
    ("snow_lag1", "Snowfall the day before", 1.0, "per inch", "day"),
    ("log_n", "Complaints that day", float(np.log(2)), "per doubling", "day"),
    ("base_log", "Recent same-weekday volume", float(np.log(2)), "per doubling", "day"),
    ("trend_years", "Long-run trend", 1.0, "per year", "year"),
    ("regime", "2020 regime", 1.0, f"after vs before {REGIME_START:%b %d, %Y}", "year"),
]
BENCHMARK = {"day": ("adj_log", "type and year"), "year": ("adj_type_log", "type only")}
BINARY = {"weekend", "holiday", "heat_season", "regime"}
# Joint response-time model on the type-and-year benchmark: the day-level groups only (that benchmark absorbs years).
RT_GROUPS = {
    "dow": ("Day of week", DOW_COLS),
    "holiday": ("Federal holidays", ["holiday"]),
    "season": ("Month (season)", [f"m_{i}" for i in range(2, 13)]),
    "weather": ("Weather", WEATHER_COLS),
    "volume": ("Complaints that day", ["log_n"]),
}
FEATURE_GROUP = {"weekend": "dow", "holiday": "holiday", "heat_season": "season", "log_n": "volume",
                 **{c: "weather" for c in WEATHER_COLS}}
TERM = {   # (text, scale, unit) for coefficients quoted in effect texts
    "tmax_f": ("high temperature", 10.0, "per 10\N{DEGREE SIGN}F"),
    "tmin_f": ("low temperature", 10.0, "per 10\N{DEGREE SIGN}F"),
    "tmax_rel": ("high temperature vs the baseline days", 10.0, "per 10\N{DEGREE SIGN}F"),
    "tmin_rel": ("low temperature vs the baseline days", 10.0, "per 10\N{DEGREE SIGN}F"),
    "prcp_in": ("precipitation", 1.0, "per inch"), "snow_in": ("snowfall", 1.0, "per inch"),
    "prcp_lag1": ("precipitation the day before", 1.0, "per inch"),
    "snow_lag1": ("snowfall the day before", 1.0, "per inch"),
    "prcp_base": ("precipitation on the baseline days", 1.0, "per inch"),
    "snow_base": ("snowfall on the baseline days", 1.0, "per inch"),
    "log_n": ("complaints that day", float(np.log(2)), "per doubling"),
}
OBSOLETE = ["q10_importance.csv", "q10_volume_effects.csv", "q10_response_spearman.csv", "q10_response_hac.csv"]


# ---------- features ----------

def load_weather(path=WEATHER_CSV):
    return pd.read_csv(path, parse_dates=["date"])


def weather_lags(w):
    """Precipitation and snowfall of the calendar day before (not the previous row)."""
    w = w.copy()
    w["date"] = pd.to_datetime(w.date)
    prev = w.set_index("date")[["prcp_in", "snow_in"]].shift(1, freq="D")
    w["prcp_lag1"] = w.date.map(prev.prcp_in)
    w["snow_lag1"] = w.date.map(prev.snow_in)
    return w


def make_features(daily, weather):
    """One row per day: the count and its log; the baseline (mean count of the last BASE_WEEKS same weekdays, at least
    MIN_BASE_WEEKS of them complete), log(count / baseline), the count exactly 7 days earlier and the mean of the last
    28 days (at least MIN_LOOKBACK_DAYS complete); calendar dummies (Monday as the weekday base, all 12 months);
    holidays that day and their share among the baseline days; weather that day, the day before, and relative to the
    baseline days; years since the first day; the 2020 regime indicator; response-time outcomes as % when present."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values("date").reset_index(drop=True)
    w = weather_lags(weather)
    d = d.drop(columns=[c for c in WEATHER_COLS if c in d]).merge(w[["date", *WEATHER_COLS]], on="date", how="left")
    dow = d.date.dt.dayofweek
    for i in range(1, 7):
        d[f"dow_{i}"] = (dow == i).astype(float)
    d["weekend"] = (dow >= 5).astype(float)
    start = d.date.min() - pd.Timedelta(days=LOOKBACK_DAYS)
    hol = USFederalHolidayCalendar().holidays(start, d.date.max())
    d["holiday"] = d.date.isin(hol).astype(float)
    for m in range(1, 13):
        d[f"m_{m}"] = (d.date.dt.month == m).astype(float)
    d["heat_season"] = (~d.date.dt.month.between(6, 9)).astype(float)
    d["log_n"] = np.log(d.n.where(d.n > 0))

    idx = pd.date_range(start, d.date.max(), freq="D")
    counts = pd.Series(d.n.where(d.n > 0).to_numpy(float), index=d.date).reindex(idx)
    wx = w.set_index("date")[["tmax_f", "tmin_f", "prcp_in", "snow_in"]].reindex(idx)
    hx = pd.Series(idx.isin(hol).astype(float), index=idx)

    def back(s, k):
        return s.shift(k).reindex(d.date).to_numpy(float)

    d["lag7"] = back(counts, 7)
    d["lag7_log"] = np.log(d.lag7)
    parts = np.column_stack([back(counts, 7 * k) for k in range(1, BASE_WEEKS + 1)])
    ok = np.isfinite(parts)
    k_ok = ok.sum(axis=1)

    def on_base_days(a):
        s = np.where(ok, a, 0.0).sum(axis=1)
        return np.where(k_ok >= MIN_BASE_WEEKS, s / np.maximum(k_ok, 1), np.nan)

    d["base"] = on_base_days(parts)
    d["base_log"] = np.log(d.base)
    d["ratio_log"] = d.log_n - d.base_log
    d["hol_base"] = on_base_days(np.column_stack([back(hx, 7 * k) for k in range(1, BASE_WEEKS + 1)]))
    for c in ["tmax_f", "tmin_f", "prcp_in", "snow_in"]:
        on_base = on_base_days(np.column_stack([back(wx[c], 7 * k) for k in range(1, BASE_WEEKS + 1)]))
        if c.startswith("t"):
            d[c.replace("_f", "_rel")] = d[c] - on_base
        else:
            d[c.replace("_in", "_base")] = on_base
    last = np.column_stack([back(counts, k) for k in range(1, LOOKBACK_DAYS + 1)])
    n_last = np.isfinite(last).sum(axis=1)
    d["mean28"] = np.where(n_last >= MIN_LOOKBACK_DAYS, np.nansum(last, axis=1) / np.maximum(n_last, 1), np.nan)
    d["trend_years"] = (d.date - d.date.min()).dt.days / 365.25
    d["regime"] = (d.date >= pd.Timestamp(REGIME_START)).astype(float)
    for c in ["adj_log", "adj_type_log"]:
        if c in d:
            d[c.replace("_log", "_pct")] = 100 * np.expm1(d[c])
    return d


def full_years_in(f):
    """Calendar years with every day present in f (used when the caller does not pass clean.full_years)."""
    c = f.groupby(f.date.dt.year).date.nunique()
    return [int(y) for y, n in c.items() if n == (366 if calendar.isleap(int(y)) else 365)]


# ---------- which groups the data can support ----------

def group_availability(train, test):
    """For each value group: can the data estimate it (and use it on `test`)? With the reason."""
    rows = []
    n_base = int(train.base.notna().sum()) if "base" in train else 0
    rows.append(("volume", n_base >= MIN_NAIVE_DAYS,
                 f"{n_base:,} days have at least {MIN_BASE_WEEKS} of the {BASE_WEEKS} earlier same weekdays "
                 f"(needs {MIN_NAIVE_DAYS})"))
    wk = train.date.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    rows.append(("dow", wk.min() >= 2, f"each weekday appears at least {int(wk.min())} times in {len(train):,} days "
                                       "(needs 2)"))
    h_tr = int(train.holiday.sum())
    rows.append(("holiday", h_tr >= MIN_EVENT_DAYS, f"{h_tr} holidays to learn from (needs {MIN_EVENT_DAYS})"))
    m_tr, m_te = set(train.date.dt.month), set(test.date.dt.month)
    rows.append(("season", len(m_tr) == 12 and m_te <= m_tr,
                 f"training covers {len(m_tr)} of 12 calendar months (a month effect needs a full year to learn "
                 "from and a later period to test)"))
    rows.append(("weather", len(train) >= MIN_WEATHER_DAYS,
                 f"{len(train):,} training days (needs {MIN_WEATHER_DAYS} to separate weather from noise)"))
    out = pd.DataFrame(rows, columns=["group", "estimable", "reason"])
    out["estimable"] = out.estimable.astype(bool)
    out.insert(1, "label", out.group.map(VALUE_GROUPS))
    return out


def plan_folds(u):
    """Test years: every calendar year with MIN_TEST_DAYS usable days and MIN_TRAIN_DAYS usable days in the TRAIN_YEARS
    calendar years before it. Kept: the last N_TYPICAL of them up to rules.PROFILE_LAST_YEAR (flagged typical) and
    every later one. With no test year up to that year, every test year is typical. `train_all` (every earlier usable
    day) is for the robustness check of the rolling window."""
    cand = []
    yr = u.date.dt.year
    for y in sorted(int(v) for v in yr.unique()):
        train, test = u[(yr >= y - TRAIN_YEARS) & (yr < y)], u[yr == y]
        if len(train) >= MIN_TRAIN_DAYS and len(test) >= MIN_TEST_DAYS:
            cand.append({"test_year": y, "train": train, "test": test, "train_all": u[yr < y]})
    early = [p for p in cand if p["test_year"] <= rules.PROFILE_LAST_YEAR][-N_TYPICAL:]
    later = [p for p in cand if p["test_year"] > rules.PROFILE_LAST_YEAR]
    if not early:
        early, later = later, []
    return [{**p, "typical": True} for p in early] + [{**p, "typical": False} for p in later]


# ---------- models ----------

def _cols(groups):
    return [c for g in groups for c in GROUPS[g][1]]


def fit_ratio(train, test, groups):
    """Baseline x exp(fitted adjustment): OLS of log(count / baseline) on the groups' columns, no intercept, so with no
    groups the forecast is the baseline. A column that never varies in training gets no weight (minimum-norm fit)."""
    base = test.base.to_numpy(float)
    cols = _cols(groups)
    if not cols:
        return base
    beta = np.linalg.lstsq(train[cols].to_numpy(float), train.ratio_log.to_numpy(float), rcond=None)[0]
    return base * np.exp(test[cols].to_numpy(float) @ beta)


def fit_calendar(train, test):
    """The benchmark without recent counts: OLS of log(count) on weekday, holiday, month and weather, with an
    intercept, so it carries the training years' level."""
    cols = [c for c in CAL_COLS if train[c].std() > 0]
    X = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(float) for c in cols])
    beta = np.linalg.lstsq(X, train.log_n.to_numpy(float), rcond=None)[0]
    Xt = np.column_stack([np.ones(len(test))] + [test[c].to_numpy(float) for c in cols])
    return np.exp(Xt @ beta)


def _years_label(years):
    ys = sorted(int(y) for y in years)
    if not ys:
        return ""
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}-{ys[-1]}" if ys == list(range(ys[0], ys[-1] + 1)) \
        else ", ".join(map(str, ys))


def naive_scores(f):
    """MAPE of the forecasts that need no training, on the days where all of them exist."""
    d = f.dropna(subset=["log_n", *NAIVE])
    if len(d) < MIN_NAIVE_DAYS:
        return pd.DataFrame(columns=["model", "label", "mape", "n_days"])
    a = d.n.to_numpy(float)
    return pd.DataFrame([{"model": m, "label": MODELS[m], "n_days": len(d),
                          "mape": float(np.mean(100 * np.abs(a - d[m].to_numpy(float)) / a))} for m in NAIVE])


def evaluate(f):
    """Full mode: per test year, every model in MODELS; pooled MAPE over the typical years and per later year; each
    group's value. Short mode: why not, plus the naive forecasts' error."""
    days = f.dropna(subset=["log_n"])
    u = f.dropna(subset=REQUIRED)
    plans = plan_folds(u)
    n_years = days.date.dt.year.nunique()
    out = {"availability": group_availability(days, days), "n_usable": len(u), "n_years": int(n_years),
           "first": days.date.min(), "last": days.date.max(), "plans": [p["test_year"] for p in plans]}
    if not plans:
        out.update(mode="short", naive=naive_scores(f), typical_years=[], later_years=[], typical_label="",
                   scores=pd.DataFrame(), summary=pd.DataFrame(), per_year=pd.DataFrame(), values=pd.DataFrame(),
                   fold_groups=pd.DataFrame(), effects=None, effects_reason="no test year", periods=[],
                   reason=(f"{len(u):,} days with a complete forecast history ({_d(days.date.min())} to "
                           f"{_d(days.date.max())}, {n_years} calendar year{'s' if n_years != 1 else ''}): an "
                           f"out-of-sample test needs {MIN_TRAIN_DAYS} such days in the {TRAIN_YEARS} calendar years "
                           f"before a test year with at least {MIN_TEST_DAYS}"))
        return out
    rows, fold_groups = [], []
    for p in plans:
        tr, te, y = p["train"], p["test"], p["test_year"]
        ga = group_availability(tr, te)
        fold_groups.append(ga.assign(test_year=y))
        est = set(ga.group[ga.estimable])
        groups = [g for g in GROUPS if g in est]
        preds = {"calendar": fit_calendar(tr, te), "constant": np.full(len(te), np.exp(tr.log_n.mean())),
                 "mean28": te.mean28.to_numpy(float), "lag7": te.lag7.to_numpy(float), "base": te.base.to_numpy(float),
                 "full": fit_ratio(tr, te, groups), "full_all": fit_ratio(p["train_all"], te, groups)}
        for g in groups:
            preds[f"base+{g}"] = fit_ratio(tr, te, [g])
            preds[f"full-{g}"] = fit_ratio(tr, te, [h for h in groups if h != g])
        a = te.n.to_numpy(float)
        for m, label in MODELS.items():
            ape = 100 * np.abs(a - preds[m]) / a if m in preds else None
            rows.append({"test_year": y, "typical": p["typical"], "model": m, "label": label,
                         "mape": float(ape.mean()) if ape is not None else np.nan,
                         "sum_ape": float(ape.sum()) if ape is not None else np.nan, "n_test": len(te),
                         "n_train": len(tr), "train_from": int(tr.date.dt.year.min()),
                         "train_to": int(tr.date.dt.year.max()), "test_from": te.date.min(), "test_to": te.date.max()})
    s = pd.DataFrame(rows)
    typical = [p["test_year"] for p in plans if p["typical"]]
    later = [p["test_year"] for p in plans if not p["typical"]]
    label = _years_label(typical)
    periods = [(label, typical)] + [(str(y), [y]) for y in later]
    summ = pd.DataFrame({"model": list(MODELS), "label": list(MODELS.values())})
    for lab, ys in periods:
        t = s[s.test_year.isin(ys)]
        g = t.groupby("model").agg(sum_ape=("sum_ape", lambda v: v.sum(min_count=len(ys))), n=("n_test", "sum"))
        summ[lab] = summ.model.map(g.sum_ape / g.n)
    per_year = s.pivot_table(index="model", columns="test_year", values="mape").reindex(list(MODELS))
    i_last = max(i for i, p in enumerate(plans) if p["typical"])
    last_typ = plans[i_last]
    eff_groups = {g: GROUPS[g] for g in GROUPS if g in set(fold_groups[i_last].query("estimable").group)}
    effects, why = joint_model(last_typ["train"], "ratio_log", eff_groups, intercept=False)
    out.update(mode="full", scores=s, summary=summ, per_year=per_year, typical_years=typical, later_years=later,
               typical_label=label, periods=periods, effects=effects, effects_reason=why,
               effects_window=(int(last_typ["train"].date.dt.year.min()), int(last_typ["train"].date.dt.year.max())),
               fold_groups=pd.concat(fold_groups, ignore_index=True)[["test_year", "group", "label", "estimable",
                                                                      "reason"]],
               values=value_table(summ, per_year, periods, typical))
    return out


def _value(g, tab, col):
    def at(m):
        return tab.loc[m, col] if m in tab.index else np.nan
    if g == "volume":
        return at("calendar") - at("full")
    if g == "dow":
        return at("mean28") - at("base")
    return at("base") - at(f"base+{g}")


def value_table(summary, per_year, periods, typical):
    """Per group: drop in MAPE when added (value_*), rise when dropped from the full model (dropped_*, fitted groups
    only), per period and per typical year; n_better = typical years in which adding it lowered the error."""
    S = summary.set_index("model")
    rows = []
    for g, label in VALUE_GROUPS.items():
        r = {"group": g, "label": label, "how": HOW[g]}
        for lab, _ in periods:
            r[f"value_{lab}"] = float(_value(g, S, lab))
            r[f"dropped_{lab}"] = (float(S.loc[f"full-{g}", lab] - S.loc["full", lab]) if g in GROUPS else np.nan)
        per = [float(_value(g, per_year, y)) for y in typical]
        for y, v in zip(typical, per):
            r[f"value_{y}"] = v
        fin = [v for v in per if np.isfinite(v)]
        r["n_years"], r["n_better"] = len(fin), sum(v > 0 for v in fin)
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values(f"value_{periods[0][0]}", ascending=False, na_position="last").reset_index(drop=True)


def drift_table(f):
    """Per calendar year: days, mean count, weekend vs weekday volume (non-holiday days), and the mean holiday count
    relative to its same-weekday baseline (holidays with a baseline)."""
    d = f.dropna(subset=["log_n"])
    rows = []
    for y, g in d.groupby(d.date.dt.year):
        wd, we = g[(g.weekend == 0) & (g.holiday == 0)].n, g[(g.weekend == 1) & (g.holiday == 0)].n
        h = g[g.holiday == 1].ratio_log.dropna()
        rows.append({"year": int(y), "n_days": len(g), "mean_n": float(g.n.mean()),
                     "weekend_ratio": float(we.mean() / wd.mean()) if len(we) and len(wd) else np.nan,
                     "n_holidays": int(g.holiday.sum()), "n_holidays_base": len(h),
                     "holiday_pct": float(100 * np.expm1(h.mean())) if len(h) else np.nan})
    return pd.DataFrame(rows)


# ---------- in-sample models and tests ----------

def _hac_lags(n):
    return int(min(HAC_LAGS, max(1, np.floor(4 * (n / 100) ** (2 / 9)))))


def _year_hac_lags(n):
    return int(min(YEAR_HAC_LAGS, max(1, n // 4)))


def _plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def feature_ok(d, c, n_years=None):
    """Can a single feature be tested on these days? (ok, reason, short reason)."""
    s = d[c].dropna()
    if n_years is not None and n_years < MIN_FULL_YEARS:
        return False, (f"{_plural(n_years, 'full calendar year')} (a year-level feature is compared across at least "
                       f"{MIN_FULL_YEARS})"), f"under {MIN_FULL_YEARS} full years"
    if not len(s):
        return False, "no days with this outcome", "no data"
    if c == "trend_years":
        span = (d.date.max() - d.date.min()).days + 1 if len(d) else 0
        if span < MIN_TREND_SPAN_DAYS:
            return False, (f"the days span {_plural(span, 'day')} (a trend needs {MIN_TREND_SPAN_DAYS:,}, or it is "
                           "confounded with the season)"), "under two years of data"
    if c == "regime":
        on, off = int((s == 1).sum()), int((s == 0).sum())
        if min(on, off) < MIN_REGIME_DAYS:
            return False, (f"{_plural(on, 'day')} on or after {REGIME_START:%b %d, %Y} and {off:,} before (needs "
                           f"{MIN_REGIME_DAYS} each)"), f"too few days on one side of {REGIME_START:%b %Y}"
    if c in BINARY:
        on, off = int((s == 1).sum()), int((s == 0).sum())
        if min(on, off) < MIN_EVENT_DAYS:
            short = f"only {_plural(on, 'day')} on" if on < MIN_EVENT_DAYS else f"only {_plural(off, 'day')} off"
            return False, f"{on} of {_plural(len(s), 'day')} on (needs {MIN_EVENT_DAYS} on and off)", short
    if c in WET_THRESHOLD:
        wet = int((s >= WET_THRESHOLD[c]).sum())
        if wet < MIN_WET_DAYS:
            return False, (f"{_plural(wet, 'day')} with {WET_THRESHOLD[c]:g} in or more (needs {MIN_WET_DAYS}; "
                           "fewer events would let one storm decide the slope)"), (f"only {_plural(wet, 'day')} "
                                                                                    f"with {WET_THRESHOLD[c]:g}+ in")
    if len(s) < 10:
        return False, f"{_plural(len(s), 'day')} (needs 10)", "too few days"
    if s.std() == 0:
        return False, "no variation", "no variation"
    return True, "", ""


def _all_cols(groups):
    return [c for g in groups.values() for c in g[1]]


def joint_model(df, ycol, groups, intercept=True):
    """OLS of ycol on every group's varying columns with HAC SEs, if there are MIN_DAYS_PER_PARAM days per parameter.
    Returns (table of groups with a Wald p-value and an effect text, note on groups left out) or (None, reason)."""
    import statsmodels.api as sm

    d = df.dropna(subset=[ycol, *_all_cols(groups)])
    cols = [c for c in _all_cols(groups) if len(d) > 1 and d[c].std() > 0]
    p = len(cols) + int(intercept)
    if len(d) < MIN_DAYS_PER_PARAM * p or not cols:
        return None, (f"{_plural(len(d), 'day')} for {_plural(p, 'parameter')} (needs {MIN_DAYS_PER_PARAM} days "
                      "per parameter)")
    X = d[cols].astype(float)
    if intercept:
        X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(d[ycol].to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": _hac_lags(len(d))},
                                                 use_t=True)
    rows, skipped = [], []
    for g, (label, gcols) in groups.items():
        gc = [c for c in gcols if c in cols]
        if not gc:
            skipped.append(label)
            continue
        R = np.zeros((len(gc), len(X.columns)))
        for i, c in enumerate(gc):
            R[i, list(X.columns).index(c)] = 1
        pval = float(np.asarray(fit.wald_test(R, scalar=True, use_f=True).pvalue))
        rows.append({"group": g, "label": label, "n_params": len(gc), "p": pval,
                     "effect": _effect_text(g, fit.params, fit.pvalues, gc)})
    return pd.DataFrame(rows).assign(n_days=len(d)), (("no variation here: " + ", ".join(skipped)) if skipped else "")


def _effect_text(g, params, pvals, cols):
    pct = lambda b: _pct1(100 * np.expm1(b))   # noqa: E731
    if g == "dow":
        eff = {DAYS[int(c.split("_")[1])]: params[c] for c in cols}
        lo, hi = min(eff, key=eff.get), max(eff, key=eff.get)
        return f"vs Monday: {lo} {pct(eff[lo])}, {hi} {pct(eff[hi])}"
    if g == "season":
        eff = {MONTHS[int(c.split("_")[1]) - 1]: params[c] for c in cols}
        lo, hi = min(eff, key=eff.get), max(eff, key=eff.get)
        ref = "vs the baseline weeks" if "m_1" in cols else "vs January"
        return f"{ref}: lowest {lo} {pct(eff[lo])}, highest {hi} {pct(eff[hi])}"
    if g == "holiday" and "hol_base" in cols:
        return (f"a holiday {pct(params['holiday'])} vs the baseline; one holiday among the {BASE_WEEKS} baseline "
                f"days {pct(params['hol_base'] / BASE_WEEKS)}")
    if g == "weather":
        sig = [c for c in cols if pvals[c] < ALPHA]
        if not sig:
            return "no single weather term significant at 5%"
        return "; ".join(f"{TERM[c][0]} {pct(params[c] * TERM[c][1])} {TERM[c][2]}" for c in sig)
    c = cols[0]
    if c in TERM:
        return f"{pct(params[c] * TERM[c][1])} {TERM[c][2]}"
    if c == "holiday":
        return f"{pct(params[c])} vs other days"
    return f"{params[c]:+.3f}"


VOLUME_FEATURES = [r for r in RT_FEATURES if r[0] not in ("log_n",)] + [
    ("lag7_log", "Complaints 7 days earlier", float(np.log(2)), "per doubling", "day")]


def associations(f, ycol="log_n", features=VOLUME_FEATURES):
    """Spearman of ycol with each feature (pairwise complete days); descriptive, for the short mode."""
    from nyc311.q09_equity import spearman
    rows = []
    for c, label, *_ in features:
        if c == ycol or c not in f:
            continue
        ok, why, short = feature_ok(f.dropna(subset=[ycol]), c)
        s = spearman(f[c], f[ycol]) if ok else {"n": int(f[[c, ycol]].dropna().shape[0]), "rho": np.nan,
                                                  "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        rows.append({"feature": c, "label": label, **s, "reason": why, "short": short})
    return pd.DataFrame(rows)


def response_tests(f, full_years=None):
    """Response times vs each feature, one test for all: OLS of the daily mix-adjusted log time to close on the feature
    with HAC (Newey-West) SEs; day-level features against the type-and-year benchmark (HAC window up to HAC_LAGS days),
    year-level features against the type-only benchmark on full calendar years (window YEAR_HAC_LAGS days).
    `significant` is p < ALPHA in that test. Spearman rho (same outcome) and rho with the raw median time are context.
    Also a joint model of the day-level groups on the type-and-year benchmark."""
    import statsmodels.api as sm

    from nyc311.q09_equity import spearman

    fy = full_years_in(f) if full_years is None else sorted(int(y) for y in full_years)
    d = f[f.n_closed >= MIN_CLOSED_DAY] if "n_closed" in f else f
    n_few = len(f) - len(d)
    rows = []
    for c, label, scale, unit, level in RT_FEATURES:
        ycol, bench = BENCHMARK[level]
        s = d if level == "day" else d[d.date.dt.year.isin(fy)]
        s = s.dropna(subset=[c, ycol]) if ycol in s else s.iloc[:0]
        ny = int(s.date.dt.year.nunique())
        ok, why, short = feature_ok(s, c, n_years=len(fy) if level == "year" else None)
        if short == "no data" and n_few:
            why, short = (f"no day has {MIN_CLOSED_DAY}+ closed complaints ({_plural(n_few, 'day')} have fewer)",
                          f"no day with {MIN_CLOSED_DAY}+ closures")
        r = {"feature": c, "label": label, "unit": unit, "level": level, "benchmark": bench, "n": len(s),
             "n_years": ny, "estimable": ok, "significant": False, "rho": np.nan, "p_spearman": np.nan,
             "rho_median": np.nan, "effect_pct": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
             "hac_lags": np.nan, "reason": why, "short": short}
        if ok:
            sp = spearman(s[c], s[ycol])
            lags = _hac_lags(len(s)) if level == "day" else _year_hac_lags(len(s))
            fit = sm.OLS(s[ycol].to_numpy(float), sm.add_constant(s[c].to_numpy(float))).fit(
                cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=True)
            b, (lo, hi) = fit.params[1], fit.conf_int(ALPHA)[1]
            pv = float(fit.pvalues[1])
            r.update(rho=sp["rho"], p_spearman=sp["p"], effect_pct=100 * np.expm1(b * scale),
                     ci_low=100 * np.expm1(lo * scale), ci_high=100 * np.expm1(hi * scale), p=pv, hac_lags=lags,
                     significant=bool(pv < ALPHA),
                     rho_median=spearman(s[c], s.median_h)["rho"] if "median_h" in s else np.nan)
        rows.append(r)
    tests = pd.DataFrame(rows)
    joint, why = (joint_model(d, "adj_log", RT_GROUPS) if "adj_log" in d
                  else (None, "no response-time data"))
    return {"tests": tests, "joint": joint, "joint_reason": why, "n_days": int(d.adj_log.notna().sum())
            if "adj_log" in d else 0, "full_years": fy, "n_few": n_few}


# ---------- data ----------

def daily_table(con, src, weather):
    raw = con.sql(f"""
        with b as (select created::date as d, coalesce(is_info_request, false) as info,
                          coalesce(category, '') = '{rules.INVALID_CATEGORY}' as invalid,
                          possibly_incomplete_day as inc, unresolved
                   from {src} where {rules.BASE_SQL}),
        cov as (select distinct created::date as cd from {src}
                where {rules.BASE_SQL} and not out_of_file_order and created is not null)
        select b.d, info, invalid, b.d is null as nodate, cov.cd is not null as covered, inc, count(*)::bigint as n,
               count_if(unresolved)::bigint as n_unresolved
        from b left join cov on b.d = cov.cd group by all""").df()
    raw["d"] = pd.to_datetime(raw.d)
    steps, keep = [("default analysis set", int(raw.n.sum()))], raw

    def drop(label, mask):
        nonlocal keep
        steps.append((label, int(keep.n[mask].sum())))
        keep = keep[~mask]

    drop("information requests (is_info_request)", keep["info"])
    drop(f"category '{rules.INVALID_CATEGORY}'", keep.invalid)
    drop("no valid created date", keep.nodate)
    drop("dates with only out_of_file_order rows (outside the files' period)", ~keep.covered)
    drop("possibly_incomplete_day", keep.inc)
    drop("no weather record for the date", ~keep.d.isin(pd.to_datetime(weather.date)))
    steps.append(("analyzed (complete days)", int(keep.n.sum())))
    ex = pd.DataFrame(steps, columns=["step", "rows"])
    ex["pct"] = 100 * ex.rows / max(int(ex.rows.iloc[0]), 1)
    daily = keep.groupby("d", as_index=False)[["n", "n_unresolved"]].sum().rename(columns={"d": "date"})
    # Mix-adjusted time to close (log scale): each closure vs the mean log time of its complaint type in its year
    # (adj_log) and of its complaint type over all years (adj_type_log, for year-level features).
    rt = con.sql(f"""
        with c as (select created::date as d, coalesce(canonical_type, '') as ct, coalesce(year(created), -1) as yr,
                          duration_h, ln(duration_h + 1) as y
                   from {src} where {rules.COMPLAINTS_SQL} and duration_h is not null and not ({rules.INSTANT_SQL})
                     and created::date in (select distinct created::date from {src}
                                           where {rules.BASE_SQL} and not out_of_file_order and created is not null)),
        cell as (select ct, yr, avg(y) as cy from c group by all),
        tcell as (select ct, avg(y) as ty from c group by all)
        select d as date, count(*)::bigint as n_closed, quantile_cont(duration_h, 0.5) as median_h,
               avg(y - cy) as adj_log, avg(y - ty) as adj_type_log
        from c join cell using (ct, yr) join tcell using (ct) group by 1""").df()
    rt["date"] = pd.to_datetime(rt.date)
    daily = daily.merge(rt, on="date", how="left")
    daily["n_closed"] = daily.n_closed.fillna(0).astype(int)
    return daily.sort_values("date").reset_index(drop=True), ex


# ---------- charts ----------

def _hbar_axes(ax):
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelcolor="#333333", labelsize=10.5)


def _nice_step(x):
    e = 10 ** np.floor(np.log10(x))
    return float(min([1, 2, 2.5, 5, 10], key=lambda m: abs(m * e - x)) * e)


def _panel_title(ev, lab, years):
    if lab == ev["typical_label"]:
        pre = max(years) <= rules.PROFILE_LAST_YEAR
        return f"{lab}{', pre-pandemic' if pre else ''}"
    s = ev["scores"]
    t = s[s.test_year == years[0]].iloc[0]
    a, b = pd.Timestamp(t.test_from), pd.Timestamp(t.test_to)
    if (a.month, a.day) == (1, 1) and (b.month, b.day) == (12, 31):
        return lab
    return f"{lab} ({a:%b} {a.day}-{b:%b} {b.day})"


def forecast_figure(ev, title, note):
    """Small multiples, one panel per period (the pooled typical years, then each later year), all on one x scale:
    MAPE of each forecast; the same-weekday baseline in dark sage, naive and calendar benchmarks in gray, the fitted
    models in light sage."""
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    summ = ev["summary"].set_index("model")
    labels = [lab for lab, _ in ev["periods"]]
    rows = [m for m in CHART_MODELS if summ.loc[m, labels].notna().any()]
    vals = summ.loc[rows, labels].to_numpy(float)
    vmax = float(np.nanmax(vals))
    step = _nice_step(vmax / 3.2)
    xmax = max(vmax * 1.2, step * 2.05)
    gray = nyt.HUES[-1]
    fig = Figure(figsize=(2.6 + 3.1 * len(labels), 0.4 * len(rows) + 2.4))
    axes = np.atleast_1d(fig.subplots(1, len(labels), sharex=True, sharey=True))
    y = np.arange(len(rows))
    for ax, (lab, years) in zip(axes, ev["periods"]):
        v = summ.loc[rows, lab].to_numpy(float)
        colors = [nyt.SAGE.line if m == "base" else gray.band if m in ("calendar", "mean28", "lag7") else nyt.SAGE.band
                  for m in rows]
        ax.barh(y, np.nan_to_num(v), color=colors, height=0.72, zorder=2)
        for i, val in enumerate(v):
            if not np.isfinite(val):
                ax.text(0.02 * xmax, i, "not estimable", va="center", ha="left", color=nyt.NOTE_GRAY,
                        fontfamily=nyt.SANS, fontsize=8.5, fontstyle="italic")
                continue
            ax.text(val + 0.02 * xmax, i, f"{val:.1f}%", va="center", ha="left", fontfamily=nyt.SANS, fontsize=9,
                    color=nyt.SAGE.line if rows[i] not in ("calendar", "mean28", "lag7") else nyt.NOTE_GRAY,
                    fontweight="bold" if rows[i] in ("base", "full") else "normal")
        ax.set_xlim(0, xmax)
        ax.axvline(0, color="#333333", linewidth=1.0, zorder=3)
        _hbar_axes(ax)
        nyt.format_x(ax, suffix="%", step=step)
        nyt.panel_title(ax, _panel_title(ev, lab, years), fontsize=10.5)
    axes[0].set_yticks(y, [MODELS[m] for m in rows])
    axes[0].invert_yaxis()
    width = int(9.2 * fig.get_figwidth())
    nyt.title(fig, textwrap.fill(title, width))
    nyt.note(fig, textwrap.fill(note, int(15 * fig.get_figwidth())))
    return fig


def chart_associations(assoc, path, title, note):
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    a = assoc.copy()
    a["order"] = a.rho.abs().fillna(-1)
    a = a.sort_values("order", ascending=False).reset_index(drop=True)
    vals = a.rho.fillna(0).to_numpy()
    fig = Figure(figsize=(8, 0.42 * len(a) + 1.9))
    ax = fig.subplots()
    y = np.arange(len(a))
    sig = (a.p < ALPHA).to_numpy()
    colors = [(nyt.SAGE if v > 0 else nyt.CORAL).line if s else (nyt.SAGE if v > 0 else nyt.CORAL).band
              for v, s in zip(vals, sig)]
    ax.barh(y, vals, color=colors, height=0.72, zorder=2)
    for i, r in a.iterrows():
        if not np.isfinite(r.rho):
            ax.text(0.02, i, f"not testable: {r.short}", va="center", ha="left", color=nyt.NOTE_GRAY,
                    fontfamily=nyt.SANS, fontsize=9, fontstyle="italic")
            continue
        v = vals[i]
        hue = nyt.SAGE if v > 0 else nyt.CORAL
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}".replace("-", "\N{MINUS SIGN}"), va="center",
                ha="left" if v >= 0 else "right", color=hue.line, fontfamily=nyt.SANS, fontsize=9.5,
                fontweight="bold" if sig[i] else "normal")
    ax.set_yticks(y, a.label)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=1.0, zorder=3)
    ax.set_xlim(-1.15, 1.15)
    _hbar_axes(ax)
    nyt.format_x(ax, step=0.5)
    nyt.title(fig, title)
    nyt.note(fig, note)
    return nyt.save(fig, path)


# ---------- run ----------

def run(cfg):
    con = cfg.connect()
    src = clean_glob(cfg)
    weather = load_weather()
    daily, ex = daily_table(con, src, weather)
    res = analyze(make_features(daily, weather), ex, full_years=clean_full_years(con, src))
    write_outputs(cfg, res)
    return res


def analyze(f, ex, full_years=None):
    """Everything computed from the daily feature table `f`, the exclusion steps `ex`, and the full calendar years
    (clean.full_years; inferred from f when not given)."""
    from nyc311.q05_top_types import period_long

    fy = full_years_in(f) if full_years is None else sorted(int(y) for y in full_years)
    d0, d1 = f.date.min(), f.date.max()
    final = f[f.date > d1 - pd.Timedelta(days=FINAL_DAYS)]
    early = f[f.date <= d1 - pd.Timedelta(days=FINAL_DAYS)]
    return {"daily": f, "exclusions": ex, "evaluation": evaluate(f), "drift": drift_table(f), "full_years": fy,
            "associations": associations(f.dropna(subset=["log_n"])), "response": response_tests(f, fy),
            "period_long": period_long(d0.date(), d1.date()), "n_days": len(f), "volume": _volume_facts(f),
            "censor": {"n_days_final": len(final),
                       "pct_unres_final": 100 * final.n_unresolved.sum() / max(final.n.sum(), 1),
                       "pct_unres_early": (100 * early.n_unresolved.sum() / early.n.sum()
                                           if early.n.sum() else np.nan)}}


def write_outputs(cfg, res):
    f, ex, ev, rt = res["daily"], res["exclusions"], res["evaluation"], res["response"]
    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    keep_cols = ["date", "n", "n_unresolved", "n_closed", "median_h", "adj_pct", "adj_type_pct", *WEATHER_COLS,
                 "holiday", "lag7", "base", "mean28", "hol_base", "regime"]
    f[[c for c in keep_cols if c in f]].to_csv(agg / "q10_daily.csv", index=False)
    for name in OBSOLETE:
        (agg / name).unlink(missing_ok=True)
    tables = [("q10_exclusions", ex), ("q10_availability", ev["availability"]), ("q10_drift", res["drift"]),
              ("q10_volume_associations", res["associations"]), ("q10_response_tests", rt["tests"]),
              ("q10_response_joint", rt["joint"]), ("q10_scores", ev["scores"]), ("q10_summary", ev["summary"]),
              ("q10_values", ev["values"]), ("q10_fold_groups", ev["fold_groups"]),
              ("q10_ratio_effects", ev["effects"]), ("q10_naive", ev.get("naive"))]
    for name, tab in tables:
        path = agg / f"{name}.csv"
        if tab is not None and len(tab):
            tab.to_csv(path, index=False)
        else:
            path.unlink(missing_ok=True)

    figs = cfg.out_dir / "figs"
    fig_name = "q10_feature_importance.png" if ev["mode"] == "full" else "q10_feature_correlations.png"
    if figs.is_dir():
        for old in figs.glob("q10_*.png"):
            if old.name != fig_name:
                old.unlink()
    title, note = _chart_text(res)
    if ev["mode"] == "full":
        from nyc311 import nytplot as nyt
        nyt.save(forecast_figure(ev, title, note), figs / fig_name)
    else:
        chart_associations(res["associations"], figs / fig_name, title, note)
    res["figure"], res["figure_title"] = fig_name, title
    _report(cfg, res)
    _write_answer(cfg, res)


def _volume_facts(f):
    wd, we = f[f.weekend == 0].n, f[f.weekend == 1].n
    hol = f[(f.holiday == 1) & (f.weekend == 0)].n
    return {"weekday_mean": float(wd.mean()) if len(wd) else np.nan, "weekend_mean": float(we.mean()) if len(we)
            else np.nan, "n_weekday": len(wd), "n_weekend": len(we), "holiday_mean": float(hol.mean()) if len(hol)
            else np.nan, "n_holiday": len(hol), "mean": float(f.n.mean()), "min": int(f.n.min()), "max": int(f.n.max())}


# ---------- text ----------

def _d(x):
    x = pd.Timestamp(x)
    return f"{x:%b} {x.day}, {x.year}"


def _pct1(x):
    return f"{x:+.1f}%".replace("-", "\N{MINUS SIGN}") if np.isfinite(x) else "n/a"


def _num(x, fmt="+.2f"):
    return format(x, fmt).replace("-", "\N{MINUS SIGN}") if np.isfinite(x) else "n/a"


def _p(p):
    return "n/a" if not np.isfinite(p) else "< 0.001" if p < 0.001 else f"{p:.3g}" if p < 0.1 else f"{p:.2f}"


def _rules():
    return (f"binary features need {MIN_EVENT_DAYS} days on and off; rain and snow need {MIN_WET_DAYS} days at "
            f"{WET_THRESHOLD['prcp_in']:g} in and {WET_THRESHOLD['snow_in']:g} in or more, so one storm cannot decide "
            f"a slope; a trend needs {MIN_TREND_SPAN_DAYS:,} days and the year-level features {MIN_FULL_YEARS} full "
            f"calendar years; the 2020 regime {MIN_REGIME_DAYS} days on each side")


def _size(rho):
    a = abs(rho)
    return "large" if a >= 0.5 else "moderate" if a >= 0.3 else "small" if a >= 0.1 else "negligible"


def _md(text):
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: f"`{m.group(0)}`", text)


def _and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _adds(ev):
    """Fitted groups whose value in the typical period reaches MIN_GAIN."""
    v = ev["values"]
    return v[v.group.isin(list(GROUPS)) & (v[f"value_{ev['typical_label']}"] >= MIN_GAIN)]


def _rel(x):
    return f"{abs(x):.0f}% {'below' if x < 0 else 'above'}"


def _typ(ev):
    lab = ev["typical_label"]
    return lab, ev["summary"].set_index("model")[lab]


def _typical_desc(ev):
    ys = ev["typical_years"]
    pre = max(ys) <= rules.PROFILE_LAST_YEAR
    kind = "pre-pandemic test year" if pre else "test year"
    return f"{ev['typical_label']}, {_plural(len(ys), kind)}{' pooled' if len(ys) > 1 else ''}"


def _chart_text(res):
    ev = res["evaluation"]
    if ev["mode"] == "full":
        lab, t = _typ(ev)
        base, full, cal = t["base"], t["full"], t["calendar"]
        gain = base - full
        adds = _adds(ev)
        if cal > base:
            head = (f"Averaging the last {BASE_WEEKS} same weekdays misses daily complaints by {base:.1f}% a week "
                    f"ahead ({lab})")
        else:
            head = (f"A calendar model without recent counts beats the last {BASE_WEEKS} same weekdays a week ahead "
                    f"({cal:.1f}% vs {base:.1f}% error, {lab})")
        if abs(gain) < MIN_GAIN:
            tail = f"; holidays, month and weather barely change that ({full:.1f}%)"
        elif gain < 0:
            tail = f"; adding holidays, month and weather makes it worse ({full:.1f}%)"
        elif len(adds) == 1:
            g = adds.iloc[0]
            tail = f"; {g.label.lower()} cut that to {t['base+' + g.group]:.1f}%, all features to {full:.1f}%"
        else:
            tail = f"; holidays, month and weather cut that to {full:.1f}%"
        n_typ = int(ev["scores"].query("model == 'base' and typical").n_test.sum())
        note = (f"Note: Mean absolute percentage error of one-week-ahead forecasts of daily complaints. Baseline: mean "
                f"of the counts 7, 14, 21 and 28 days earlier. Fitted models use the {TRAIN_YEARS} calendar years "
                f"before each test year; {lab} {'pools' if len(ev['typical_years']) > 1 else 'covers'} {n_typ:,} test "
                "days. Light sage: the baseline adjusted for "
                "holidays, month or weather; gray: benchmarks.")
        return head + tail, note
    n = int(res["associations"].n.max()) if len(res["associations"]) else 0
    title = f"Only {res['n_days']} complete days: links with daily complaint counts, not a forecast test"
    note = (f"Note: Spearman correlation of daily complaints with each feature, {n} complete days "
            f"({res['period_long']}); dark bars p < {ALPHA}. In-sample only: no later period to test a forecast on.")
    return title, textwrap.fill(note, 135)


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    ev, rt = res["evaluation"], res["response"]
    v = res["volume"]
    rep.add(f"Q10. Features for daily complaint volume, {res['period_long']}: {res['n_days']:,} complete days "
            f"(mean {v['mean']:,.0f} a day, range {v['min']:,}-{v['max']:,})",
            f"mode: {ev['mode']}" + (f" (typical {ev['typical_label']}, later {ev['later_years']}; models fit on the "
                                     f"{TRAIN_YEARS} calendar years before each test year)" if ev["mode"] == "full"
                                     else f": {ev['reason']}"))
    rep.table(res["exclusions"].round(2), title="Rows (default set -> complete days)")
    if ev["mode"] == "full":
        s = ev["summary"].set_index("model").drop(columns="label")
        py = ev["per_year"][[y for y in ev["typical_years"]]] if len(ev["typical_years"]) > 1 else None
        tab = s if py is None else s.join(py.rename(columns=str), rsuffix=" ")
        rep.table(tab.round(2).reset_index(), title="MAPE (%) by model: pooled typical years, then each year")
        vcols = ["group", *[c for c in ev["values"].columns if c.startswith(("value_", "dropped_"))], "n_better",
                 "n_years"]
        rep.table(ev["values"][vcols].round(2), title="Value (MAPE points): drop when added; rise when dropped "
                                                       "from the full model")
        fg = ev["fold_groups"]
        if (~fg.estimable).any():
            rep.table(fg[~fg.estimable][["test_year", "group", "reason"]], title="Groups not estimable in a fold")
        if ev["effects"] is not None:
            w0, w1 = ev["effects_window"]
            rep.table(ev["effects"].round(4), title=f"In-sample effects on log(count / baseline), {w0}-{w1}, HAC")
    else:
        rep.table(ev["availability"], title="Feature groups the data can support (in-sample)")
        if len(ev["naive"]):
            rep.table(ev["naive"].round(2), title="Naive forecasts (no training needed)")
        rep.table(res["associations"][["feature", "n", "rho", "p", "short"]].round(3),
                  title="Spearman: daily log(count) vs feature (in-sample)")
    rep.table(res["drift"].round(2), title="Drift by year (weekend/weekday volume; holiday vs its baseline, %)")
    t = rt["tests"]
    rep.table(t[["feature", "benchmark", "n", "n_years", "rho", "rho_median", "effect_pct", "ci_low", "ci_high", "p",
                 "hac_lags", "significant"]].round(3),
              title="Response times: one test per feature (OLS on mix-adjusted log time, HAC)")
    if rt["joint"] is not None:
        rep.table(rt["joint"].round(4), title="Joint model, type-and-year benchmark, HAC (Wald p)")
    else:
        rep.add("", f"joint response-time model skipped: {rt['joint_reason']}")
    rep.add("", _censor_line(res["censor"])[2:].replace("**", ""))
    rep.emit()


def _write_answer(cfg, res):
    ev, v, ex = res["evaluation"], res["volume"], res["exclusions"]
    n_rows = int(ex.rows.iloc[-1])
    excl = "; ".join(f"{_md(r.step)} {int(r.rows):,} ({r.pct:.1f}%)" for r in ex.iloc[1:-1].itertuples())
    lines = [
        "# Q10. Features for predicting complaint volume",
        "",
        f"*Computed by `nyc311/q10_features.py` from {n_rows:,} complaints on {res['n_days']:,} complete days, "
        f"{res['period_long']}.*",
        "",
        "## What is modeled",
        "",
        f"- **Target:** complaints per day (mean {v['mean']:,.0f}, range {v['min']:,}-{v['max']:,}). Left out of "
        f"{int(ex.rows.iloc[0]):,} default-set rows: {excl or 'none'}.",
        f"- **Forecast, one week ahead.** The baseline is the mean count of the last {BASE_WEEKS} same weekdays "
        f"(7 to {7 * BASE_WEEKS} days earlier, at least {MIN_BASE_WEEKS} of them complete). Fitted models adjust it "
        "by modeling log(count / baseline) on holidays (that day, and holidays among the baseline days), month, and "
        "weather (high and low temperature relative to the baseline days; precipitation and snowfall that day, the day "
        "before and on the baseline days). There is no long-run trend: the level comes from the recent weeks. "
        f"Models are fit on the {TRAIN_YEARS} calendar years before each test year and scored on that year by the "
        "mean absolute percentage error (MAPE).",
        "- **Value of a feature group:** the drop in MAPE when it is added to a forecast that lacks it. Recent volume: "
        "a calendar-and-weather model without recent counts vs the full model. Day of week: the mean of the last "
        f"{LOOKBACK_DAYS} days vs the mean of the last {BASE_WEEKS} same weekdays. Holidays, month and weather: the "
        "baseline vs the baseline plus that group; as a check, the rise in error when each is dropped from the full "
        "model.",
        "",
        "## Which features are most valuable, and why",
        "",
    ]
    lines += _full_lines(res) if ev["mode"] == "full" else _short_lines(res)
    lines += ["", f"![{res['figure_title']}](figs/{res['figure']})", ""]
    lines += _response_lines(res)
    n_rt = res["response"]["n_days"]
    lines += [
        "",
        "## Caveats",
        "",
        "- **Correlation, not causation.** Calendar, weather and volume all move together (cold months are heat "
        "season and also have fewer noise complaints), so an association with response times does not show that the "
        "feature slows closures.",
        f"- **Forecast use.** Observed weather overstates what a forecast can use; the baseline needs the counts 7 to "
        f"{7 * BASE_WEEKS} days back, so it serves horizons up to a week. Standard errors are HAC (Newey-West, "
        f"{_plural(_hac_lags(n_rt), 'lag')} for {_plural(n_rt, 'day')}) for day-to-day autocorrelation; one-feature "
        "tests do not control for the others.",
        *_year_level_caveat(res),
        _censor_line(res["censor"]),
        "",
    ]
    path = cfg.out_dir / "answers" / "q10.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _year_level_caveat(res):
    t, fy = res["response"]["tests"], res["response"]["full_years"]
    if not t[(t.level == "year") & t.estimable].shape[0]:
        return []
    return [(f"- **Year-level features.** The trend and the 2020 regime are compared across "
            f"{_plural(len(fy), 'full calendar year')}" + (f" ({_years_label(fy)})" if fy else "")
            + f" with a {YEAR_HAC_LAGS}-day HAC window; the regime is a single event, so a significant shift says "
              "that it stands out from normal variation, not what caused it.")]


def _censor_line(c):
    if np.isfinite(c["pct_unres_early"]):
        return (f"- **Censoring.** Complaints opened in the final {FINAL_DAYS} days ({c['n_days_final']} complete) are "
                f"{c['pct_unres_final']:.1f}% unresolved vs {c['pct_unres_early']:.1f}% earlier; closed-only daily "
                "times there lean toward fast closures.")
    return (f"- **Censoring.** All {c['n_days_final']} days fall within the final {FINAL_DAYS} days of the data, so "
            f"whether recent complaints look faster only because slow ones are still open ({c['pct_unres_final']:.1f}% "
            "unresolved) cannot be checked here; status is as of the data extract.")


def _by_year(r, years):
    return ", ".join(f"{y} {_num(r[f'value_{y}'], '+.1f')}" for y in years if np.isfinite(r.get(f"value_{y}", np.nan)))


def _full_lines(res):
    ev, dr = res["evaluation"], res["drift"].set_index("year")
    lab, t = _typ(ev)
    vals, typ_years, later = ev["values"], ev["typical_years"], ev["later_years"]
    v = vals.set_index("group")
    n_typ = int(ev["scores"].query("model == 'base' and typical").n_test.sum())
    gain = t["base"] - t["full"]
    eff = {} if ev["effects"] is None else dict(zip(ev["effects"].group, zip(ev["effects"].effect, ev["effects"].p)))
    fitted = vals[vals.group.isin(list(GROUPS))]
    adds = _adds(ev)
    mostly = len(adds) and gain > 0 and adds[f"value_{lab}"].sum() >= gain / 2
    core = t["base"] < min(t["calendar"], t["constant"], t["lag7"], t["mean28"])
    dow_v, vol_v = v.loc["dow", f"value_{lab}"], v.loc["volume", f"value_{lab}"]
    if abs(gain) < MIN_GAIN:
        verdict = ("Recent same-weekday volume is nearly all a one-week forecast needs." if core else
                   f"Holidays, month and weather change the error by less than {MIN_GAIN} points.")
    elif gain < 0:
        verdict = "The fitted features make the forecast worse out of sample."
    else:
        verdict = ((f"Recent same-weekday volume does most of the work: matching the weekday is worth {dow_v:.1f} "
                    f"points and recent counts {vol_v:.1f} (ranking below), all fitted features together {gain:.1f}. "
                    if core and max(dow_v, vol_v) > gain else "")
                   + (f"Fitted groups worth {MIN_GAIN}+ points on their own: {_and(adds.label.str.lower())}."
                      if len(adds) else f"No fitted group adds {MIN_GAIN}+ points on its own."))
    out = [f"- **Headline: {_typical_desc(ev)} ({n_typ:,} test days).** The mean of the last {BASE_WEEKS} same weekdays "
           f"missed daily complaints by {t['base']:.1f}% on average, against {t['lag7']:.1f}% for last week's "
           f"same-weekday count, {t['mean28']:.1f}% for the mean of the last {LOOKBACK_DAYS} days, {t['calendar']:.1f}% "
           f"for a calendar-and-weather model without recent counts and {t['constant']:.1f}% for the training-window "
           f"average. Adding holidays, month and weather brings it to {t['full']:.1f}% ({_num(-gain, '+.1f')} points"
           + (f", mostly through {_and(adds.label.str.lower())}" if mostly else "") + f"). {verdict}"]
    out.append(f"- **Ranking** by the drop in error when a group is added ({lab}"
               + (", with each year" if len(typ_years) > 1 else "") + (" and the later years" if later else "")
               + "):")
    for i, r in enumerate(vals.to_dict("records"), 1):
        val = r[f"value_{lab}"]
        if not np.isfinite(val):
            continue
        per = (f"; by year {_by_year(r, typ_years)} (lower error in {r['n_better']} of {r['n_years']})"
               if len(typ_years) > 1 else "")
        lat = ", ".join(f"{y} {_num(r[f'value_{y}'], '+.1f')}" for y in later
                        if np.isfinite(r.get(f"value_{y}", np.nan)))
        dv = r.get(f"dropped_{lab}", np.nan)
        drop = (f" Dropped from the full model, the error {'rises' if dv >= 0 else 'falls'} {abs(dv):.2f} points."
                if np.isfinite(dv) else "")
        e = eff.get(r["group"])
        size = f" In-sample ({ev['effects_window'][0]}-{ev['effects_window'][1]}): {e[0]} (p {_p(e[1])})." if e else ""
        out.append(f"{i}. **{r['label']}**: {_num(val, '+.2f')} points ({HOW_TEXT[r['group']]}){per}"
                   + (f"; later {lat}" if lat else "") + "." + drop + f" Why: {WHY[r['group']]}.{size}")
    weak = fitted[(fitted[f"value_{lab}"] < MIN_GAIN) | (fitted.n_better < fitted.n_years)]
    if len(weak):
        small = bool((weak[[f"value_{lab}", f"dropped_{lab}"]].max(axis=1) < MIN_GAIN).all())
        out.append(f"- **No reliable gain from {_and(weak.label.str.lower())}** in {lab}: "
                   + "; ".join(f"{r['label'].lower()} {_num(r[f'value_{lab}'], '+.2f')} points when added to the "
                               "baseline" + (f" (lower error in {r['n_better']} of {r['n_years']} years)"
                                             if len(typ_years) > 1 else "")
                               + f" and {_num(r[f'dropped_{lab}'], '+.2f')} when dropped from the full model"
                               for r in weak.to_dict("records"))
                   + (f". Neither measure reaches {MIN_GAIN} points" if small else
                      ". The gains are not consistent across years")
                   + ", so a production model can leave " + ("them" if len(weak) > 1 else "it")
                   + " out or needs a better form.")
    out += _drift_lines(ev, dr)
    return out


def _drift_lines(ev, dr):
    lab, t = _typ(ev)
    out = []
    last_typ = max(ev["typical_years"])
    later = [y for y in ev["later_years"] if y in dr.index]
    wk = dr.weekend_ratio.dropna()
    hol = dr[dr.n_holidays_base >= MIN_EVENT_DAYS].holiday_pct.dropna()
    if len(wk) >= 2 and len(hol) >= 2:
        wy = [int(wk.index[0]), *(y for y in [last_typ, *later] if y in wk.index and y != int(wk.index[0]))]
        hy = [int(hol.index[0]), *(y for y in [last_typ, *later] if y in hol.index and y != int(hol.index[0]))]
        w_span = 100 * (wk[wy].max() - wk[wy].min())
        h_span = hol[hy].max() - hol[hy].min()
        drift = max(w_span, h_span) >= DRIFT_POINTS
        flatter = 100 * (abs(1 - wk[wy[-1]]) - abs(1 - wk[wy[0]])) <= -DRIFT_POINTS
        better = drift and t["full"] < t["full_all"]
        out.append(f"- **{'The patterns drift' if drift else 'Weekly and holiday patterns'}.** Weekend days averaged "
                   + _and([f"{100 * wk[y]:.0f}% of weekday volume in {y}" if i == 0 else f"{100 * wk[y]:.0f}% in {y}"
                           for i, y in enumerate(wy)])
                   + "; holidays ran " + _and([f"{_rel(hol[y])} their same-weekday baseline in {y}" if i == 0 else
                                               f"{_rel(hol[y])} in {y}" for i, y in enumerate(hy)]) + "."
                   + (" With patterns this mobile, recent years are the better guide:" if better else "")
                   + (f" {'f' if better else 'F'}it on the {TRAIN_YEARS} years before each test year, the full model "
                      f"misses by {t['full']:.1f}% in {lab}, vs {t['full_all']:.1f}% when fit on every earlier year."
                      if np.isfinite(t["full_all"]) else "")
                   + (" Matching the weekday also matters less as weekends catch up with weekdays." if flatter else ""))
    typ_cal = t["calendar"]
    for y in later:
        s = ev["summary"].set_index("model")[str(y)]
        sc = ev["scores"][(ev["scores"].test_year == y) & (ev["scores"].model == "base")].iloc[0]
        prev = y - 1 if (y - 1) in dr.index else last_typ
        brk = s["calendar"] >= BREAK_RATIO * typ_cal
        out.append(f"- **{y}{', a structural break' if brk else ''}** (test days {_d(sc.test_from)} to "
                   f"{_d(sc.test_to)}). The mean of the last {BASE_WEEKS} same weekdays missed by {s['base']:.1f}% "
                   f"(vs {t['base']:.1f}% in {lab}); the full model {s['full']:.1f}%, last week's count "
                   f"{s['lag7']:.1f}%, the {LOOKBACK_DAYS}-day mean {s['mean28']:.1f}%. The calendar-and-weather model, "
                   f"trained on {int(sc.train_from)}-{int(sc.train_to)}, missed by {s['calendar']:.1f}%"
                   + (f" (vs {typ_cal:.1f}% in {lab})" if brk else "")
                   + f": complaints averaged {dr.loc[y, 'mean_n']:,.0f} a day vs {dr.loc[prev, 'mean_n']:,.0f} in "
                     f"{prev}" + (", and a model without recent counts keeps its training years' level."
                                  if s["calendar"] > s["base"] else ".")
                   + (f" The {LOOKBACK_DAYS}-day mean beat the same-weekday mean: the weekday pattern had faded."
                      if s["mean28"] < s["base"] else ""))
    return out


def _short_lines(res):
    ev, a, v = res["evaluation"], res["associations"], res["volume"]
    av = ev["availability"]
    out = [f"- **Not estimable out of sample here.** {ev['reason']}. So no feature can be ranked by forecast value "
           "on these data; a run on several years of data runs the test automatically.",
           "- **What these days can and cannot support** (in-sample):"]
    for r in av.itertuples():
        out.append(f"  - {r.label}: {'estimable' if r.estimable else 'not estimable'}; {r.reason}.")
    nv = ev["naive"]
    if len(nv):
        out.append(f"- **Forecasts that need no training** (on the {_plural(int(nv.n_days.iloc[0]), 'day')} where all "
                   "exist): " + "; ".join(f"{r.label.lower()} {r.mape:.1f}% MAPE" for r in nv.itertuples()) + ".")
    else:
        out.append(f"- **Forecasts that need no training** cannot be scored either: fewer than {MIN_NAIVE_DAYS} days "
                   f"have {MIN_BASE_WEEKS} earlier same weekdays and {MIN_LOOKBACK_DAYS} of the last "
                   f"{LOOKBACK_DAYS} days.")
    sig = a[(a.p < ALPHA)].sort_values("rho", key=np.abs, ascending=False)
    untestable = a[a.rho.isna()]
    out.append(f"- **In-sample links with daily volume** (Spearman, {int(a.n.max()) if len(a) else 0} days at most; "
               "descriptive only): " + ("; ".join(f"{r.label} {_num(r.rho)} (n {r.n}, p {_p(r.p)})"
                                                  for r in sig.itertuples()) or "none at p < 0.05") + "."
               + (" Not testable on these days: " + "; ".join(f"{r.label} ({r.short})" for r in untestable.itertuples())
                  + f". Rules: {_rules()}." if len(untestable) else ""))
    if np.isfinite(v["weekend_mean"]) and v["n_weekend"]:
        out.append(f"- **Weekly pattern:** weekdays average {v['weekday_mean']:,.0f} complaints ({v['n_weekday']} days) "
                   f"vs {v['weekend_mean']:,.0f} on weekends ({v['n_weekend']} days), "
                   f"{v['weekday_mean'] / v['weekend_mean']:.2f} times as many"
                   + (f"; the {v['n_holiday']} weekday holiday{'s' if v['n_holiday'] != 1 else ''} averaged "
                      f"{v['holiday_mean']:,.0f}" if v["n_holiday"] else "") + ".")
    out.append("- **Expected value of each group, and why** (to be confirmed on the full data): " + "; ".join(
        f"{VALUE_GROUPS[g].lower()}: {WHY[g]}" for g in VALUE_GROUPS) + ".")
    return out


def _response_lines(res):
    rt = res["response"]
    t = rt["tests"]
    fy = rt["full_years"]
    est = t[t.estimable].copy()
    est["order"] = est.rho.abs()

    def item(r):
        yrs = f", {_plural(int(r.n_years), 'year')}" if r.level == "year" else ""
        return (f"{r.label}: {_pct1(r.effect_pct)} {r.unit} (95% CI {_pct1(r.ci_low)} to {_pct1(r.ci_high)}, p "
                f"{_p(r.p)}; \N{GREEK SMALL LETTER RHO} {_num(r.rho)}, {_size(r.rho)}; n {int(r.n):,} days{yrs}; raw "
                f"median time \N{GREEK SMALL LETTER RHO} {_num(r.rho_median)})")

    if not rt["n_days"]:
        return ["## Do the same features correlate with response times?", "",
                f"- **Not testable here:** no complete day has at least {MIN_CLOSED_DAY} closed complaints (the "
                f"minimum for a daily response-time value; {_plural(rt['n_few'], 'day')} have fewer)."]
    yes = est[est.significant].sort_values("order", ascending=False)
    no = est[~est.significant].sort_values("order", ascending=False)
    na = t[~t.estimable]
    windows = [f"{'/'.join(str(x) for x in sorted({int(v) for v in est[est.level == lv].hac_lags}))}-day window "
               f"for {lv}-level features" for lv in ("day", "year") if (est.level == lv).any()]
    out = [
        "## Do the same features correlate with response times?",
        "",
        f"- **Outcomes:** for complaints opened each day ({_plural(rt['n_days'], 'day')} with at least "
        f"{MIN_CLOSED_DAY} closures; instant closures left out), the mix-adjusted time to close (Q4's method, as %): "
        "each closure against the citywide geometric mean of its complaint type in its year for day-level features "
        "(weekday, holiday, season, weather, volume), and of its complaint type over all years for the year-level "
        "trend and 2020 regime, on full calendar years only"
        + (f" ({_years_label(fy)})" if fy else "") + ". A by-year benchmark would absorb a year-level feature.",
        "- **One test for every feature:** OLS of the daily mix-adjusted log time on the feature with Newey-West (HAC) "
        "standard errors" + (f" ({'; '.join(windows)})" if windows else "")
        + f"; a feature counts as correlated when p < {ALPHA} in that test. Spearman's \N{GREEK SMALL LETTER RHO} "
        "gives the size of the link against the same benchmark; the raw median time's \N{GREEK SMALL LETTER RHO} is "
        "context.",
        "- **Correlated (strongest first):** " + ("; ".join(item(r) for r in yes.itertuples()) or "none") + ".",
        "- **Not correlated:** " + ("; ".join(item(r) for r in no.itertuples()) or "none") + ".",
    ]
    if len(na):
        out.append("- **Not testable on these days:** " + "; ".join(f"{r.label} ({r.short})" for r in na.itertuples())
                   + f". Rules: {_rules()}.")
    if rt["n_days"] < MIN_RELIABLE_DAYS:
        out.append(f"- **Few days.** With {_plural(rt['n_days'], 'day')} these tests have little power and HAC "
                   "standard errors are biased down in short samples (t-based p-values are used, which helps only "
                   "partly); treat every p-value here as descriptive and rerun on the full data.")
    if rt["joint"] is not None:
        j = rt["joint"]
        out.append("- **Joint model** (day-level groups together on the type-and-year benchmark, HAC; Wald test per "
                   f"group, {int(j.n_days.iloc[0]):,} days): " + "; ".join(
                       f"{r.label} p {_p(r.p)} ({r.effect})" for r in j.itertuples()) + "."
                   + (f" {rt['joint_reason'][0].upper()}{rt['joint_reason'][1:]}." if rt["joint_reason"] else ""))
        jp = dict(zip(j.group, j.p))
        lost = [r.label for r in yes.itertuples() if FEATURE_GROUP.get(r.feature) in jp
                and jp[FEATURE_GROUP[r.feature]] >= ALPHA]
        kept = [r.label for r in j.itertuples() if r.p < ALPHA]
        out.append(f"- **Overlap.** Held together, the groups still associated with response time (p < {ALPHA}) are: "
                   + (", ".join(kept) or "none") + "."
                   + (f" One-feature links that fade once the others are held fixed: {', '.join(lost)}; they mostly "
                      "travel with the features that remain." if lost else ""))
    else:
        out.append(f"- **Joint model skipped:** {rt['joint_reason']}; each feature is tested alone, so the "
                   "associations above may overlap.")
    return out
