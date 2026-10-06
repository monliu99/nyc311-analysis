"""Q10: which features would predict daily complaint volume, and do the same features move with response times?

Target: daily complaints (default analysis set minus information requests and "Invalid entries") on complete days
(possibly_incomplete_day and dates outside the files' period left out), merged with Central Park weather.
Feature groups: day of week, US federal holidays, month (season), weather (high/low temperature, precipitation,
snowfall, and precipitation/snowfall the day before), the count 7 days earlier, a long-run trend, and a 2020
regime indicator (from NY State's PAUSE order, Mar 22, 2020).

Value for prediction is measured out of sample by rolling origin: each of the last N_FOLDS calendar years is a test
year, predicted by an OLS model of log(count) trained on every earlier day. A group's value is the rise in MAPE
(and fall in R^2) when it is dropped from the full model, averaged over the test years in which it can be estimated.
Each fold checks which groups its training data can support (a month effect needs every test month in training, a
trend two years, the regime 60 days on each side); when the data have no later year to test on (the 5-week local
sample) the module says which groups cannot be estimated and why, and reports in-sample associations only.

Response times: the daily median time to close and the daily mix-adjusted time to close (Q4's method: each closed
complaint vs the citywide geometric mean for its complaint type and year) are tested against each feature with
Spearman correlations (n) and OLS with HAC (Newey-West) standard errors, one feature at a time and jointly.
Every number in text is computed at run time.
"""
from __future__ import annotations

import datetime as dt
import re
import textwrap

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from nyc311 import geo
from nyc311 import q04_response_times as q4
from nyc311.clean import clean_glob
from nyc311.report import Report
from nyc311.taxonomy import INVALID

STAGE = "q10_features"
WEATHER_CSV = geo.DISTRICTS_GEOJSON.parent / "weather_central_park_daily.csv"
REGIME_START = dt.date(2020, 3, 22)   # New York State on PAUSE (executive order 202.8) took effect
N_FOLDS = 3                 # test years (the last ones with enough data)
MIN_TRAIN_DAYS = 365        # training days before a test year
MIN_TEST_DAYS = 90          # usable days in a test year
MIN_EVENT_DAYS = 3          # holidays (or weekend/regime days) needed to estimate an indicator
MIN_REGIME_DAYS = 60        # regime: days needed on each side of REGIME_START
MIN_TREND_SPAN_DAYS = 730   # trend: training must span two years, or it is confounded with the season
MIN_WEATHER_DAYS = 30
MIN_WET_DAYS = 5            # an amount (rain, snow) is tested only with this many days at or above its threshold
WET_THRESHOLD = {"prcp_in": 0.1, "prcp_lag1": 0.1, "snow_in": 1.0, "snow_lag1": 1.0}   # inches, as in Q6
MIN_LAG_DAYS = 14
MIN_DAYS_PER_PARAM = 10     # a joint (in-sample) model needs this many days per parameter
MIN_CLOSED_DAY = 30         # closed complaints a day needs for its response-time values
HAC_LAGS = 7                # cap; the lag length follows Newey-West's rule floor(4 (n/100)^(2/9))
MIN_RELIABLE_DAYS = 90      # fewer days: tests are flagged as descriptive (little power, HAC SEs biased down)
FINAL_DAYS = 60             # complaints opened in the final days had the least time to close
BREAK_RATIO = 2.0           # a test year whose full-model MAPE is 2x the best year's, and worse than last
                            # week's count, is a level shift the training years had not seen
ALPHA = 0.05
MIN_GAIN = 0.5              # MAPE points: a smaller rise when a group is dropped is not called an addition
BASE = "not is_test and not dup_key"
COMPLAINTS = f"{BASE} and not coalesce(is_info_request, false) and coalesce(category, '') <> '{INVALID}'"
WEATHER_COLS = ["tmax_f", "tmin_f", "prcp_in", "snow_in", "prcp_lag1", "snow_lag1"]
REQUIRED = [*WEATHER_COLS, "lag7_log", "log_n"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

GROUPS = {
    "dow": ("Day of week", [f"dow_{i}" for i in range(1, 7)]),
    "holiday": ("Federal holidays", ["holiday"]),
    "season": ("Month (season)", [f"m_{i}" for i in range(2, 13)]),
    "weather": ("Weather", WEATHER_COLS),
    "lag7": ("Count 7 days earlier", ["lag7_log"]),
    "trend": ("Long-run trend", ["trend_years"]),
    "regime": ("2020 regime", ["regime"]),
}
WHY = {
    "dow": "agencies, inspectors and many callers work weekdays, so filing follows the work week",
    "holiday": "holidays behave like extra weekend days (offices closed, routines broken)",
    "season": "heat complaints follow the heating season and noise peaks in summer, so volume and mix swing by month",
    "weather": "cold snaps drive heat and hot-water calls and storms drive flooding, tree and street calls (Q6); "
               "a real forecast would have to use forecast weather",
    "lag7": "last week's count carries the current level (campaigns, outbreaks, backlogs, shifts) that calendar "
            "features miss, and it is known a week ahead",
    "trend": "311 use grew as the app and web channels spread",
    "regime": "the pandemic changed what people reported and how agencies worked",
}
# (column, label, scale, unit) for one-feature tests; scale converts a coefficient to "per unit"
FEATURES = [
    ("weekend", "Weekend (Sat-Sun)", 1.0, "vs weekdays"),
    ("holiday", "Federal holiday", 1.0, "vs other days"),
    ("heat_season", "Heating season (Oct-May)", 1.0, "vs Jun-Sep"),
    ("tmax_f", "High temperature", 10.0, "per 10\N{DEGREE SIGN}F"),
    ("tmin_f", "Low temperature", 10.0, "per 10\N{DEGREE SIGN}F"),
    ("prcp_in", "Precipitation", 1.0, "per inch"),
    ("snow_in", "Snowfall", 1.0, "per inch"),
    ("prcp_lag1", "Precipitation the day before", 1.0, "per inch"),
    ("snow_lag1", "Snowfall the day before", 1.0, "per inch"),
    ("log_n", "Complaints that day", float(np.log(2)), "per doubling"),
    ("lag7_log", "Complaints 7 days earlier", float(np.log(2)), "per doubling"),
    ("trend_years", "Long-run trend", 1.0, "per year"),
    ("regime", "2020 regime", 1.0, "after vs before Mar 22, 2020"),
]
FEATURE = {c: (label, scale, unit) for c, label, scale, unit in FEATURES}
BINARY = {"weekend", "holiday", "heat_season", "regime"}
OUTCOMES = {"median_h": "median time to close", "adj_pct": "mix-adjusted time to close"}
FEATURE_GROUP = {"weekend": "dow", "holiday": "holiday", "heat_season": "season", "tmax_f": "weather",
                 "tmin_f": "weather", "prcp_in": "weather", "snow_in": "weather", "prcp_lag1": "weather",
                 "snow_lag1": "weather", "log_n": "log_n", "trend_years": "trend", "regime": "regime"}
RT_GROUPS = {"dow": GROUPS["dow"], "holiday": GROUPS["holiday"], "season": GROUPS["season"],
             "weather": GROUPS["weather"], "log_n": ("Complaints that day", ["log_n"]),
             "trend": GROUPS["trend"], "regime": GROUPS["regime"]}


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
    """One row per day: the count, its log, calendar dummies (Monday and January as base), holiday, heating season,
    weather with day-before lags, log count exactly 7 calendar days earlier (NaN if that day is missing), years since
    the first day, and the 2020 regime indicator."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values("date").reset_index(drop=True)
    w = weather_lags(weather)[["date", "tmax_f", "tmin_f", "prcp_in", "snow_in", "prcp_lag1", "snow_lag1"]]
    d = d.drop(columns=[c for c in w.columns if c != "date" and c in d]).merge(w, on="date", how="left")
    dow = d.date.dt.dayofweek
    for i in range(1, 7):
        d[f"dow_{i}"] = (dow == i).astype(float)
    d["weekend"] = (dow >= 5).astype(float)
    hol = USFederalHolidayCalendar().holidays(d.date.min(), d.date.max())
    d["holiday"] = d.date.isin(hol).astype(float)
    for m in range(2, 13):
        d[f"m_{m}"] = (d.date.dt.month == m).astype(float)
    d["heat_season"] = (~d.date.dt.month.between(6, 9)).astype(float)
    d["log_n"] = np.log(d.n.where(d.n > 0))
    lag = d.set_index("date").log_n.shift(7, freq="D")
    d["lag7_log"] = d.date.map(lag)
    d["trend_years"] = (d.date - d.date.min()).dt.days / 365.25
    d["regime"] = (d.date >= pd.Timestamp(REGIME_START)).astype(float)
    if "adj_log" in d:
        d["adj_pct"] = 100 * np.expm1(d.adj_log)
    return d


# ---------- which groups the data can support ----------

def group_availability(train, test):
    """For each feature group: can a model trained on `train` estimate it and use it on `test`? With the reason."""
    rows = []
    span = (train.date.max() - train.date.min()).days + 1 if len(train) else 0
    wk = train.date.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    rows.append(("dow", wk.min() >= 2, f"each weekday appears at least {int(wk.min())} times in {len(train):,} "
                                       f"training days (needs 2)"))
    h_tr, h_te = int(train.holiday.sum()), int(test.holiday.sum())
    rows.append(("holiday", h_tr >= MIN_EVENT_DAYS and h_te >= 1,
                 f"{h_tr} holidays to learn from (needs {MIN_EVENT_DAYS}) and {h_te} to predict"))
    m_tr, m_te = set(train.date.dt.month), set(test.date.dt.month)
    rows.append(("season", len(m_tr) == 12 and m_te <= m_tr,
                 f"training covers {len(m_tr)} of 12 calendar months (a month effect needs a full year to learn "
                 "from and a later period to test)"))
    rows.append(("weather", len(train) >= MIN_WEATHER_DAYS,
                 f"{len(train):,} training days (needs {MIN_WEATHER_DAYS} to separate weather from noise)"))
    n_lag = int(train.lag7_log.notna().sum())
    rows.append(("lag7", n_lag >= MIN_LAG_DAYS, f"{n_lag:,} days with a complete day 7 days earlier "
                                                f"(needs {MIN_LAG_DAYS})"))
    rows.append(("trend", span >= MIN_TREND_SPAN_DAYS,
                 f"training spans {span:,} days (a trend needs {MIN_TREND_SPAN_DAYS:,}, or it is confounded with "
                 "the season)"))
    r_on, r_off, r_te = int(train.regime.sum()), int((train.regime == 0).sum()), int(test.regime.sum())
    rows.append(("regime", r_on >= MIN_REGIME_DAYS and r_off >= MIN_REGIME_DAYS and r_te >= 1,
                 f"{r_on:,} training days on or after {REGIME_START:%b %d, %Y} and {r_off:,} before (needs "
                 f"{MIN_REGIME_DAYS} each); {r_te:,} test days after it"))
    out = pd.DataFrame(rows, columns=["group", "estimable", "reason"])
    out["estimable"] = out.estimable.astype(bool)
    out.insert(1, "label", out.group.map(lambda g: GROUPS[g][0]))
    return out


def plan_folds(f, n_folds=N_FOLDS):
    """Rolling origin by calendar year: each candidate test year with MIN_TEST_DAYS usable days, trained on every
    usable day before its January 1 (at least MIN_TRAIN_DAYS). The last n_folds such years."""
    u = f.dropna(subset=REQUIRED)
    plans = []
    for y in sorted(u.date.dt.year.unique())[1:]:
        start = pd.Timestamp(f"{int(y)}-01-01")
        train, test = u[u.date < start], u[u.date.dt.year == y]
        if len(train) >= MIN_TRAIN_DAYS and len(test) >= MIN_TEST_DAYS:
            plans.append({"test_year": int(y), "train": train, "test": test})
    return plans[-n_folds:]


# ---------- out-of-sample evaluation ----------

def _cols(groups):
    return [c for g in groups for c in GROUPS[g][1]]


def _fit_predict(train, test, cols):
    cols = [c for c in cols if train[c].std() > 0]
    Xtr = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(float) for c in cols])
    beta, *_ = np.linalg.lstsq(Xtr, train.log_n.to_numpy(float), rcond=None)
    Xte = np.column_stack([np.ones(len(test))] + [test[c].to_numpy(float) for c in cols])
    return np.exp(Xte @ beta)


def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    sst = float(((a - a.mean()) ** 2).sum())
    return {"mape": float(100 * np.mean(np.abs(a - p) / a)),
            "r2": 1 - float(((a - p) ** 2).sum()) / sst if sst > 0 else np.nan}


def importance(scores):
    """Per group: rise in MAPE (and fall in R^2) when dropped, per test year and as the median over the years in which
    it is estimable (so one unusual year does not decide the ranking); its stand-alone MAPE."""
    s = scores
    full = s[s.model == "full"].set_index("test_year")
    rows = []
    for g, (label, _) in GROUPS.items():
        w = s[(s.model == "without") & (s.group == g)].set_index("test_year")
        a = s[(s.model == "alone") & (s.group == g)]
        d = w.mape - full.mape.reindex(w.index)
        r = {"group": g, "label": label, "n_folds": len(w),
             "d_mape": float(d.median()) if len(w) else np.nan, "d_mape_mean": float(d.mean()) if len(w) else np.nan,
             "d_r2": float((full.r2.reindex(w.index) - w.r2).median()) if len(w) else np.nan,
             "mape_alone": float(a.mape.median()) if len(a) else np.nan}
        for y in full.index:
            r[f"d_mape_{y}"] = float(d.get(y, np.nan))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("d_mape", ascending=False, na_position="last").reset_index(drop=True)


def evaluate(f):
    """Full mode: per fold, the full model, intercept only, last week's count as the forecast, each group dropped
    and each group alone; importance = mean rise in MAPE when a group is dropped. Short mode: why not."""
    usable = f.dropna(subset=REQUIRED)
    avail = group_availability(usable, usable)
    plans = plan_folds(f)
    n_years = usable.date.dt.year.nunique()
    out = {"availability": avail, "n_usable": len(usable), "n_years": int(n_years),
           "first": usable.date.min(), "last": usable.date.max(), "plans": [p["test_year"] for p in plans]}
    if not plans:
        out["mode"] = "short"
        out["reason"] = (f"{len(usable):,} usable days ({_d(usable.date.min())} to {_d(usable.date.max())}) in "
                         f"{n_years} calendar year{'s' if n_years != 1 else ''}: an out-of-sample test needs at least "
                         f"{MIN_TRAIN_DAYS} training days before a later calendar year with at least {MIN_TEST_DAYS} "
                         "usable days")
        out["scores"] = pd.DataFrame(columns=["test_year", "model", "group", "mape", "r2", "n_train", "n_test"])
        out["fold_groups"] = pd.DataFrame(columns=["test_year", "group", "label", "estimable", "reason"])
        out["importance"] = pd.DataFrame(columns=["group", "label", "n_folds", "d_mape", "d_r2", "mape_alone"])
        out["breaks"] = []
        return out
    scores, fold_groups = [], []
    for p in plans:
        tr, te = p["train"], p["test"]
        ga = group_availability(tr, te)
        fold_groups.append(ga.assign(test_year=p["test_year"]))
        groups = list(ga.group[ga.estimable])
        base = {"test_year": p["test_year"], "n_train": len(tr), "n_test": len(te)}

        def add(model, group, pred):
            scores.append({**base, "model": model, "group": group, **metrics(te.n, pred)})

        add("full", None, _fit_predict(tr, te, _cols(groups)))
        add("intercept only", None, _fit_predict(tr, te, []))
        add("same weekday last week", None, np.exp(te.lag7_log.to_numpy(float)))
        for g in groups:
            add("without", g, _fit_predict(tr, te, _cols([h for h in groups if h != g])))
            add("alone", g, _fit_predict(tr, te, _cols([g])))
    s = pd.DataFrame(scores)
    fg = pd.concat(fold_groups, ignore_index=True)[["test_year", "group", "label", "estimable", "reason"]]
    imp = importance(s)
    full = s[s.model == "full"].set_index("test_year")
    naive = s[s.model == "same weekday last week"].set_index("test_year")
    breaks = [int(y) for y in full.index
              if naive.loc[y, "mape"] < full.loc[y, "mape"] and full.loc[y, "mape"] > BREAK_RATIO * full.mape.min()]
    out["breaks"] = breaks
    return {**out, "mode": "full", "scores": s, "fold_groups": fg, "importance": imp}


# ---------- in-sample models and tests ----------

def _hac_lags(n):
    return int(min(HAC_LAGS, max(1, np.floor(4 * (n / 100) ** (2 / 9)))))


def _plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def feature_ok(d, c):
    """Can a single feature be tested on these days? (ok, reason, short reason). Same rules as the groups."""
    s = d[c].dropna()
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
                           "fewer events would let one storm decide the slope)"), f"only {_plural(wet, 'day')} "\
                f"with {WET_THRESHOLD[c]:g}+ in"
    if len(s) < 10:
        return False, f"{_plural(len(s), 'day')} (needs 10)", "too few days"
    if s.std() == 0:
        return False, "no variation", "no variation"
    return True, "", ""


def joint_model(df, ycol, groups):
    """OLS of ycol on every estimable group (in-sample) with HAC SEs, if there are MIN_DAYS_PER_PARAM days per
    parameter. Returns (table of groups with a Wald p-value and an effect text, None) or (None, reason)."""
    import statsmodels.api as sm

    d = df.dropna(subset=[ycol, *_all_cols(groups)])
    ga = group_availability(d, d).set_index("group") if len(d) else None
    keep = {g: v for g, v in groups.items() if g not in GROUPS or (ga is not None and ga.loc[g, "estimable"])}
    cols = [c for g in keep for c in keep[g][1] if d[c].std() > 0]
    p = len(cols) + 1
    if len(d) < MIN_DAYS_PER_PARAM * p or not cols:
        return None, (f"{_plural(len(d), 'day')} for {_plural(p, 'parameter')} (needs {MIN_DAYS_PER_PARAM} days "
                      "per parameter)")
    X = sm.add_constant(d[cols].astype(float), has_constant="add")
    fit = sm.OLS(d[ycol].to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": _hac_lags(len(d))},
                                                 use_t=True)
    rows = []
    for g, (label, gcols) in keep.items():
        gc = [c for c in gcols if c in cols]
        if not gc:
            continue
        R = np.zeros((len(gc), len(X.columns)))
        for i, c in enumerate(gc):
            R[i, list(X.columns).index(c)] = 1
        pval = float(np.asarray(fit.wald_test(R, scalar=True, use_f=True).pvalue))
        rows.append({"group": g, "label": label, "n_params": len(gc), "p": pval,
                     "effect": _effect_text(g, fit.params, fit.pvalues, gc)})
    skipped = [g for g in groups if g not in keep]
    return pd.DataFrame(rows).assign(n_days=len(d)), (
        ("not estimable here: " + ", ".join(groups[g][0] for g in skipped)) if skipped else "")


def _all_cols(groups):
    return [c for g in groups.values() for c in g[1]]


def _effect_text(g, params, pvals, cols):
    pct = lambda b: _pct1(100 * np.expm1(b))   # noqa: E731
    if g == "dow":
        eff = {DAYS[int(c.split("_")[1])]: params[c] for c in cols}
        lo, hi = min(eff, key=eff.get), max(eff, key=eff.get)
        return f"vs Monday: {lo} {pct(eff[lo])}, {hi} {pct(eff[hi])}"
    if g == "season":
        eff = {MONTHS[int(c.split("_")[1]) - 1]: params[c] for c in cols}
        lo, hi = min(eff, key=eff.get), max(eff, key=eff.get)
        return f"vs January: lowest {lo} {pct(eff[lo])}, highest {hi} {pct(eff[hi])}"
    if g == "weather":
        sig = [c for c in cols if pvals[c] < ALPHA]
        if not sig:
            return "no single weather term significant at 5%"
        return "; ".join(f"{FEATURE[c][0].lower()} {pct(params[c] * FEATURE[c][1])} {FEATURE[c][2]}" for c in sig)
    c = cols[0]
    if c in FEATURE:
        _, scale, unit = FEATURE[c]
        return f"{pct(params[c] * scale)} {unit}"
    return f"{params[c]:+.3f}"


def associations(f, ycol="log_n", features=FEATURES):
    """Spearman of ycol with each feature (pairwise complete days)."""
    from nyc311.q09_equity import spearman
    rows = []
    for c, label, _, _ in features:
        if c == ycol or c not in f:
            continue
        ok, why, short = feature_ok(f.dropna(subset=[ycol]), c)
        s = spearman(f[c], f[ycol]) if ok else {"n": int(f[[c, ycol]].dropna().shape[0]), "rho": np.nan,
                                                  "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        rows.append({"feature": c, "label": label, **s, "reason": why, "short": short})
    return pd.DataFrame(rows)


def response_tests(f):
    """Response times vs features: Spearman (median and mix-adjusted, n), one-feature OLS with HAC SEs on the
    mix-adjusted log time to close, and a joint model when there are enough days."""
    import statsmodels.api as sm

    from nyc311.q09_equity import spearman

    d = f[f.adj_log.notna()]
    if "n_closed" in d:
        d = d[d.n_closed >= MIN_CLOSED_DAY]
    sp, uni = [], []
    for c, label, scale, unit in FEATURES:
        s = d.dropna(subset=[c, "adj_log"])
        ok, why, short = feature_ok(s, c)
        for o in OUTCOMES:
            sp.append({"feature": c, "label": label, "outcome": o,
                       **(spearman(d[c], d[o]) if ok else {"n": len(s), "rho": np.nan, "p": np.nan,
                                                           "ci_low": np.nan, "ci_high": np.nan})})
        if not ok:
            uni.append({"feature": c, "label": label, "unit": unit, "n": len(s), "estimable": False, "reason": why,
                        "short": short})
            continue
        fit = sm.OLS(s.adj_log.to_numpy(float), sm.add_constant(s[c].to_numpy(float))).fit(
            cov_type="HAC", cov_kwds={"maxlags": _hac_lags(len(s))}, use_t=True)
        b, (lo, hi) = fit.params[1], fit.conf_int(ALPHA)[1]
        uni.append({"feature": c, "label": label, "unit": unit, "n": len(s), "estimable": True, "reason": "",
                    "effect_pct": 100 * np.expm1(b * scale), "ci_low": 100 * np.expm1(lo * scale),
                    "ci_high": 100 * np.expm1(hi * scale), "p": float(fit.pvalues[1])})
    joint, why = joint_model(d, "adj_log", RT_GROUPS)
    return {"spearman": pd.DataFrame(sp), "univariate": pd.DataFrame(uni), "joint": joint, "joint_reason": why,
            "n_days": len(d)}


# ---------- data ----------

def daily_table(con, src, weather):
    raw = con.sql(f"""
        with b as (select created::date as d, coalesce(is_info_request, false) as info,
                          coalesce(category, '') = '{INVALID}' as invalid, possibly_incomplete_day as inc, unresolved
                   from {src} where {BASE}),
        cov as (select distinct created::date as cd from {src}
                where {BASE} and not out_of_file_order and created is not null)
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
    drop(f"category '{INVALID}'", keep.invalid)
    drop("no valid created date", keep.nodate)
    drop("dates with only out_of_file_order rows (outside the files' period)", ~keep.covered)
    drop("possibly_incomplete_day", keep.inc)
    drop("no weather record for the date", ~keep.d.isin(pd.to_datetime(weather.date)))
    steps.append(("analyzed (complete days)", int(keep.n.sum())))
    ex = pd.DataFrame(steps, columns=["step", "rows"])
    ex["pct"] = 100 * ex.rows / max(int(ex.rows.iloc[0]), 1)
    daily = keep.groupby("d", as_index=False)[["n", "n_unresolved"]].sum().rename(columns={"d": "date"})
    rt = con.sql(f"""
        with c as (select created::date as d, coalesce(canonical_type, '') as ct, coalesce(year(created), -1) as yr,
                          duration_h, ln(duration_h + 1) as y
                   from {src} where {COMPLAINTS} and duration_h is not null and not ({q4.INSTANT})
                     and created::date in (select distinct created::date from {src}
                                           where {BASE} and not out_of_file_order and created is not null)),
        cell as (select ct, yr, avg(y) as cy from c group by all)
        select d as date, count(*)::bigint as n_closed, quantile_cont(duration_h, 0.5) as median_h,
               avg(y - cy) as adj_log
        from c join cell using (ct, yr) group by 1""").df()
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


def chart_importance(imp, scores, path, title, note, emph=None):
    """Small multiples: one panel per test year, rise in MAPE when each group is dropped; rows ordered by the median."""
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    years = sorted(int(y) for y in scores.test_year.unique())
    full = scores[scores.model == "full"].set_index("test_year")
    gray = nyt.HUES[-1]
    fig = Figure(figsize=(2.9 * len(years) + 2.0, 0.42 * len(imp) + 2.2))
    axes = np.atleast_1d(fig.subplots(1, len(years), sharey=True))
    y = np.arange(len(imp))
    for ax, yr in zip(axes, years):
        vals = imp[f"d_mape_{yr}"].to_numpy(float)
        ok = np.isfinite(vals)
        v0 = np.where(ok, vals, 0.0)
        colors = [(nyt.SAGE.line if g == emph else nyt.SAGE.band) if v > 0 else gray.band
                  for g, v in zip(imp.group, v0)]
        ax.barh(y, v0, color=colors, height=0.72, zorder=2)
        span = max(np.abs(v0).max(), 0.5)
        for i in y:
            if not ok[i]:
                ax.text(0.03 * span, i, "not estimable", va="center", ha="left", color=nyt.NOTE_GRAY,
                        fontfamily=nyt.SANS, fontsize=8.5, fontstyle="italic")
                continue
            v = v0[i]
            ax.text(v + (0.04 if v >= 0 else -0.04) * span, i, f"{v:+.1f}".replace("-", "\N{MINUS SIGN}"),
                    va="center", ha="left" if v >= 0 else "right", fontfamily=nyt.SANS, fontsize=9,
                    color=nyt.SAGE.line if v > 0 else nyt.NOTE_GRAY,
                    fontweight="bold" if imp.group.iloc[i] == emph else "normal")
        lo, hi = min(v0.min(), 0.0), max(v0.max(), 0.0)
        step = _nice_step(span / 2.5)
        ax.set_xlim(lo - (0.45 * span if lo < 0 else 0.0), max(hi + 0.6 * span, 1.05 * step))
        ax.axvline(0, color="#333333", linewidth=1.0, zorder=3)
        _hbar_axes(ax)
        nyt.format_x(ax, suffix=" pts", step=step)   # at least one positive tick, so the unit never sits on 0
        nyt.panel_title(ax, f"{yr}: full model {full.loc[yr, 'mape']:.1f}% error", fontsize=10.5)
    axes[0].set_yticks(y, imp.label)
    axes[0].invert_yaxis()
    nyt.title(fig, title)
    nyt.note(fig, textwrap.fill(note, int(15 * fig.get_figwidth())))
    return nyt.save(fig, path)


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
    res = analyze(make_features(daily, weather), ex)
    write_outputs(cfg, res)
    return res


def analyze(f, ex):
    """Everything computed from the daily feature table `f` (and the exclusion steps `ex`)."""
    from nyc311.q05_top_types import period_long

    ev = evaluate(f)
    usable = f.dropna(subset=REQUIRED)
    effects, effects_reason = (joint_model(usable, "log_n", {g: GROUPS[g] for g in GROUPS})
                               if len(usable) else (None, "no usable days"))
    d0, d1 = f.date.min(), f.date.max()
    final = f[f.date > d1 - pd.Timedelta(days=FINAL_DAYS)]
    early = f[f.date <= d1 - pd.Timedelta(days=FINAL_DAYS)]
    return {"daily": f, "exclusions": ex, "evaluation": ev, "associations": associations(f.dropna(subset=["log_n"])),
            "response": response_tests(f), "effects": effects, "effects_reason": effects_reason,
            "period_long": period_long(d0.date(), d1.date()), "n_days": len(f), "volume": _volume_facts(f),
            "censor": {"n_days_final": len(final),
                       "pct_unres_final": 100 * final.n_unresolved.sum() / max(final.n.sum(), 1),
                       "pct_unres_early": (100 * early.n_unresolved.sum() / early.n.sum()
                                           if early.n.sum() else np.nan)}}


def write_outputs(cfg, res):
    f, ex, ev, rt = res["daily"], res["exclusions"], res["evaluation"], res["response"]
    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    keep_cols = ["date", "n", "n_unresolved", "n_closed", "median_h", "adj_pct", *WEATHER_COLS, "holiday",
                 "lag7_log", "regime"]
    f[[c for c in keep_cols if c in f]].to_csv(agg / "q10_daily.csv", index=False)
    ex.to_csv(agg / "q10_exclusions.csv", index=False)
    ev["availability"].to_csv(agg / "q10_availability.csv", index=False)
    res["associations"].to_csv(agg / "q10_volume_associations.csv", index=False)
    rt["spearman"].to_csv(agg / "q10_response_spearman.csv", index=False)
    rt["univariate"].to_csv(agg / "q10_response_hac.csv", index=False)
    for name, tab in [("q10_scores", ev["scores"]), ("q10_importance", ev["importance"]),
                      ("q10_fold_groups", ev["fold_groups"]), ("q10_volume_effects", res["effects"]),
                      ("q10_response_joint", rt["joint"])]:
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
        emph = _typical_top(ev["importance"].dropna(subset=["d_mape"])).group
        chart_importance(ev["importance"], ev["scores"], figs / fig_name, title, note, emph=emph)
    else:
        chart_associations(res["associations"], figs / fig_name, title, note)
    res["figure"], res["figure_title"] = fig_name, title
    _report(cfg, res)
    _write_answer(cfg, res)


def _volume_facts(f):
    wd, we = f[f.weekend == 0].n, f[f.weekend == 1].n
    hol, other = f[(f.holiday == 1) & (f.weekend == 0)].n, f[(f.holiday == 0) & (f.weekend == 0)].n
    return {"weekday_mean": float(wd.mean()) if len(wd) else np.nan, "weekend_mean": float(we.mean()) if len(we)
            else np.nan, "n_weekday": len(wd), "n_weekend": len(we), "holiday_mean": float(hol.mean()) if len(hol)
            else np.nan, "n_holiday": len(hol), "other_weekday_mean": float(other.mean()) if len(other) else np.nan,
            "mean": float(f.n.mean()), "min": int(f.n.min()), "max": int(f.n.max())}


# ---------- text ----------

def _d(x):
    return f"{pd.Timestamp(x):%b %d, %Y}"


def _pct1(x):
    return f"{x:+.1f}%".replace("-", "\N{MINUS SIGN}") if np.isfinite(x) else "n/a"


def _num(x, fmt="+.2f"):
    return format(x, fmt).replace("-", "\N{MINUS SIGN}") if np.isfinite(x) else "n/a"


def _p(p):
    return "n/a" if not np.isfinite(p) else "< 0.001" if p < 0.001 else f"{p:.3g}" if p < 0.1 else f"{p:.2f}"


def _rules():
    return (f"binary features need {MIN_EVENT_DAYS} days on and off; rain and snow need {MIN_WET_DAYS} days at "
            f"{WET_THRESHOLD['prcp_in']:g} in and {WET_THRESHOLD['snow_in']:g} in or more, so one storm cannot decide "
            f"a slope; a trend needs {MIN_TREND_SPAN_DAYS:,} days; the 2020 regime {MIN_REGIME_DAYS} days on each side")


def _size(rho):
    a = abs(rho)
    return "large" if a >= 0.5 else "moderate" if a >= 0.3 else "small" if a >= 0.1 else "negligible"


def _typical_top(imp):
    """The top group among those estimable in the most test years (a group seen in one year is not 'typical')."""
    return imp[imp.n_folds == imp.n_folds.max()].sort_values("d_mape", ascending=False).iloc[0]


def _change(v, what, up="raises", down="lowers", fmt=".2f", unit=" points"):
    return f"{up if v >= 0 else down} {what} by {abs(v):{fmt}}{unit}"


def _chart_text(res):
    ev = res["evaluation"]
    if ev["mode"] == "full":
        imp = ev["importance"].dropna(subset=["d_mape"])
        top = _typical_top(imp)
        s = ev["scores"]
        naive = s[s.model == "same weekday last week"]
        years = ", ".join(str(y) for y in ev["plans"])
        title = (f"{top.label} adds the most to a daily volume forecast: without it, error rises "
                 f"{top.d_mape:.1f} points in a typical test year" if top.d_mape >= MIN_GAIN else
                 f"No feature group adds {MIN_GAIN:g}+ points of accuracy to a daily volume forecast in a typical "
                 "test year")
        note = (f"Note: Rise in mean absolute percentage error (MAPE, points) of daily complaint counts when each "
                f"feature group is dropped from the full model; each panel is a test year predicted from all earlier "
                f"days; rows ordered by the median over {years}. Last week's same-weekday count as the forecast: "
                + ", ".join(f"{y} {naive.set_index('test_year').loc[y, 'mape']:.1f}%" for y in ev["plans"]) + ".")
        return title, note
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
            f"mode: {ev['mode']}" + (f" (test years {ev['plans']})" if ev["mode"] == "full" else f": {ev['reason']}"))
    rep.table(res["exclusions"].round(2), title="Rows (default set -> complete days)")
    rep.table(ev["availability"], title="Feature groups the whole data can support (in-sample)")
    if ev["mode"] == "full":
        s = ev["scores"]
        piv = s[s.model.isin(["full", "intercept only", "same weekday last week"])].pivot_table(
            index="model", columns="test_year", values="mape")
        rep.table(piv.round(2).reset_index(), title="MAPE (%) by test year")
        rep.table(ev["importance"].round(3), title="Importance: mean rise in MAPE / fall in R2 when dropped")
        fg = ev["fold_groups"]
        rep.table(fg[~fg.estimable][["test_year", "group", "reason"]], title="Groups not estimable in a fold")
    if res["effects"] is not None:
        rep.table(res["effects"].round(4), title="In-sample effects on log(count), all usable days, HAC")
    else:
        rep.add("", f"in-sample joint model skipped: {res['effects_reason']}")
    rep.table(res["associations"].round(3), title="Spearman: daily log(count) vs feature (in-sample)")
    sp = rt["spearman"].pivot_table(index="feature", columns="outcome", values="rho").round(3).reset_index()
    rep.table(sp, title=f"Spearman: response times vs feature ({rt['n_days']:,} days with >= {MIN_CLOSED_DAY} "
                        "closures)")
    rep.table(rt["univariate"].round(3), title="One-feature OLS on mix-adjusted log time to close, HAC")
    if rt["joint"] is not None:
        rep.table(rt["joint"].round(4), title="Joint model on mix-adjusted log time to close, HAC (Wald p)")
    else:
        rep.add("", f"joint response-time model skipped: {rt['joint_reason']}")
    c = res["censor"]
    rep.add("", _censor_line(c)[2:].replace("**", ""))
    rep.emit()


def _write_answer(cfg, res):
    ev, rt, v, ex = res["evaluation"], res["response"], res["volume"], res["exclusions"]
    n_rows = int(ex.rows.iloc[-1])
    excl = "; ".join(f"{_md(r.step)} {int(r.rows):,} ({r.pct:.1f}%)" for r in ex.iloc[1:-1].itertuples())
    lines = [
        "# Q10. Features for predicting complaint volume",
        "",
        f"*Drafted by `nyc311/q10_features.py` from {n_rows:,} complaints on {res['n_days']:,} complete days, "
        f"{res['period_long']}. Every number is computed; write the prose from these bullets.*",
        "",
        "## What is modeled",
        "",
        f"- **Target:** complaints per day (mean {v['mean']:,.0f}, range {v['min']:,}-{v['max']:,}). Left out of "
        f"{int(ex.rows.iloc[0]):,} default-set rows: {excl}.",
        "- **Candidate features:** day of week; US federal holidays; month (season); Central Park weather (high and "
        "low temperature, precipitation, snowfall, and precipitation and snowfall the day before); the count 7 days "
        f"earlier; a long-run trend; a 2020 regime indicator from {REGIME_START:%B %d, %Y} (New York State on PAUSE).",
        "- **Test:** an OLS model of log(daily count) is trained on earlier days and scored on a later calendar year "
        f"it has not seen (rolling origin, up to {N_FOLDS} test years). A group's value is how much the mean absolute "
        "percentage error (MAPE) rises when it is dropped; its stand-alone MAPE shows what it carries by itself.",
        "",
        "## Which features are most valuable, and why",
        "",
    ]
    if ev["mode"] == "full":
        lines += _full_lines(res)
    else:
        lines += _short_lines(res)
    lines += ["", f"![{res['figure_title']}](figs/{res['figure']})", ""]
    lines += _response_lines(res)
    lines += [
        "",
        "## Caveats",
        "",
        "- **Correlation, not causation.** Calendar, weather and volume all move together (cold months are heat "
        "season and also have fewer noise complaints), so an association with response times does not show that the "
        "feature slows closures.",
        "- **Forecast use.** Observed weather overstates what a forecast can use; the count 7 days earlier is "
        "available only for horizons up to a week. Standard errors are HAC (Newey-West, "
        f"{_plural(_hac_lags(res['response']['n_days']), 'lag')} for {_plural(res['response']['n_days'], 'day')}) "
        "for day-to-day "
        "autocorrelation; one-feature tests do not control for the others.",
        _censor_line(res["censor"]),
        "",
    ]
    path = cfg.out_dir / "answers" / "q10.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _censor_line(c):
    if np.isfinite(c["pct_unres_early"]):
        return (f"- **Censoring.** Complaints opened in the final {FINAL_DAYS} days ({c['n_days_final']} complete) are "
                f"{c['pct_unres_final']:.1f}% unresolved vs {c['pct_unres_early']:.1f}% earlier; closed-only daily "
                "times there lean toward fast closures.")
    return (f"- **Censoring.** All {c['n_days_final']} days fall within the final {FINAL_DAYS} days of the data, so "
            f"whether recent complaints look faster only because slow ones are still open ({c['pct_unres_final']:.1f}% "
            "unresolved) cannot be checked here; status is as of the data extract.")


def _md(text):
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: f"`{m.group(0)}`", text)


def _effects_by_group(res):
    e = res["effects"]
    return {} if e is None else dict(zip(e.group, zip(e.effect, e.p)))


def _full_lines(res):
    ev = res["evaluation"]
    s, imp, fg = ev["scores"], ev["importance"], ev["fold_groups"]
    full = s[s.model == "full"].set_index("test_year")
    naive = s[s.model == "same weekday last week"].set_index("test_year")
    base = s[s.model == "intercept only"].set_index("test_year")
    eff = _effects_by_group(res)
    out = [f"- **Accuracy.** Full model MAPE by test year: " + "; ".join(
        f"{y} {full.loc[y, 'mape']:.1f}% (R\N{SUPERSCRIPT TWO} {_num(full.loc[y, 'r2'])}, trained on "
        f"{int(full.loc[y, 'n_train']):,} days, tested on {int(full.loc[y, 'n_test']):,})" for y in full.index)
        + ". Benchmarks: last week's same-weekday count "
        + "; ".join(f"{y} {naive.loc[y, 'mape']:.1f}%" for y in naive.index) + "; a constant "
        + "; ".join(f"{y} {base.loc[y, 'mape']:.1f}%" for y in base.index) + "."]
    for y in ev.get("breaks", []):
        helped = imp[imp[f"d_mape_{y}"] < 0].label.tolist()
        out.append(f"- **Level shift in {y}.** The full model missed by {full.loc[y, 'mape']:.1f}% while last week's "
                   f"count missed by {naive.loc[y, 'mape']:.1f}%: the level moved beyond anything in the training "
                   "years, and calendar effects learned earlier cannot follow it."
                   + (f" That year, dropping {', '.join(helped)} lowers the error, because the count 7 days earlier "
                      "then carries more weight." if helped else "")
                   + " A production model should be re-estimated often or weight the recent level.")
    years = list(full.index)
    ranked = imp.dropna(subset=["d_mape"])
    out.append(f"- **Ranking** (median over the test years in which a group can be estimated, so one unusual year "
               "does not decide it):")
    for i, r in enumerate(ranked.itertuples(), 1):
        e = eff.get(r.group)
        size = f" In-sample effect: {e[0]} (p {_p(e[1])})." if e else ""
        per = ", ".join(f"{y} {_num(getattr(r, f'd_mape_{y}'), '+.1f')}" for y in years
                        if np.isfinite(getattr(r, f"d_mape_{y}")))
        some = (f", estimable in only {r.n_folds} of {len(years)} test years" if r.n_folds < len(years) else "")
        out.append(f"{i}. **{r.label}**: dropping it {_change(r.d_mape, 'MAPE')} (by year: {per}{some}) and "
                   f"{_change(r.d_r2, 'R' + chr(178), 'lowers', 'raises', '.3f', '')}; alone it reaches "
                   f"{r.mape_alone:.1f}% MAPE. Why: {WHY[r.group]}.{size}")
    never = imp[imp.d_mape.isna()]
    for r in never.itertuples():
        why = fg[(fg.group == r.group)].reason.iloc[-1]
        out.append(f"- **{r.label}** could not be estimated in any test year: {why}.")
    hurt = ranked[ranked.d_mape <= 0]
    if len(hurt):
        it = "it" if len(hurt) == 1 else "them"
        out.append("- **No out-of-sample value in a typical year:** " + ", ".join(hurt.label) + f" (dropping {it} does "
                   f"not raise the median error), so a production model can leave {it} out or needs a better form.")
    return out


def _short_lines(res):
    ev, a, v = res["evaluation"], res["associations"], res["volume"]
    av = ev["availability"]
    out = [f"- **Not estimable out of sample here.** {ev['reason']}. So no feature can be ranked by forecast value "
           "on these data; the full data (2010-2021) run the test automatically.",
           "- **What these days can and cannot support** (in-sample):"]
    for r in av.itertuples():
        out.append(f"  - {r.label}: {'estimable' if r.estimable else 'not estimable'}; {r.reason}.")
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
        f"{GROUPS[g][0].lower()}: {WHY[g]}" for g in GROUPS) + ".")
    if res["effects"] is None:
        out.append(f"- **In-sample joint model skipped:** {res['effects_reason']}.")
    return out


def _response_lines(res):
    rt = res["response"]
    sp, uni = rt["spearman"], rt["univariate"]
    adj = sp[sp.outcome == "adj_pct"].set_index("feature")
    med = sp[sp.outcome == "median_h"].set_index("feature")
    u = uni.set_index("feature")
    yes, no, na = [], [], []
    for c, label, _, unit in FEATURES:
        r = adj.loc[c] if c in adj.index else None
        if r is None or not np.isfinite(r.rho):
            na.append(label + (f" ({u.loc[c, 'short']})" if c in u.index and not u.loc[c, "estimable"] else ""))
            continue
        hac = (f"; HAC regression {_pct1(u.loc[c, 'effect_pct'])} {unit} (95% CI {_pct1(u.loc[c, 'ci_low'])} to "
               f"{_pct1(u.loc[c, 'ci_high'])}, p {_p(u.loc[c, 'p'])})" if c in u.index and u.loc[c, "estimable"]
               else "")
        txt = (f"{label}: \N{GREEK SMALL LETTER RHO} {_num(r.rho)} ({_size(r.rho)}, n {int(r.n)}, p {_p(r.p)}; "
               f"median time {_num(med.loc[c, 'rho'])}){hac}")
        (yes if r.p < ALPHA else no).append((abs(r.rho), txt))
    out = [
        "## Do the same features correlate with response times?",
        "",
        f"- **Outcomes:** for complaints opened each day, the median hours to close and the mix-adjusted time to "
        f"close (Q4's method: each closure vs the citywide geometric mean for its complaint type and year, as %), "
        f"on {_plural(rt['n_days'], 'day')} with at least {MIN_CLOSED_DAY} closures (instant closures left out).",
        "- **Correlated (p < 0.05, mix-adjusted; strongest first):** "
        + ("; ".join(t for _, t in sorted(yes, key=lambda x: -x[0])) or "none") + ".",
        "- **Not correlated:** " + ("; ".join(t for _, t in sorted(no, key=lambda x: -x[0])) or "none") + ".",
    ]
    if na:
        out.append("- **Not testable on these days:** " + "; ".join(na) + f". Rules: {_rules()}.")
    if rt["n_days"] < MIN_RELIABLE_DAYS:
        out.append(f"- **Few days.** With {_plural(rt['n_days'], 'day')} these tests have little power and HAC standard errors are "
                   "biased down in short samples (t-based p-values are used, which helps only partly); treat every "
                   "p-value here as descriptive and rerun on the full data.")
    if rt["joint"] is not None:
        j = rt["joint"]
        out.append("- **Joint model** (all groups together, HAC; Wald test per group, "
                   f"{int(j.n_days.iloc[0]):,} days): " + "; ".join(
                       f"{r.label} p {_p(r.p)} ({r.effect})" for r in j.itertuples()) + "."
                   + (f" {rt['joint_reason'][0].upper()}{rt['joint_reason'][1:]}." if rt["joint_reason"] else ""))
        jp = dict(zip(j.group, j.p))
        one = [c for c, *_ in FEATURES if c in adj.index and adj.loc[c, "p"] < ALPHA]   # NaN p compares False
        lost = [FEATURE[c][0] for c in one if FEATURE_GROUP.get(c) in jp and jp[FEATURE_GROUP[c]] >= ALPHA]
        kept = [r.label for r in j.itertuples() if r.p < ALPHA]
        out.append(f"- **Overlap.** Held together, the groups still associated with response time (p < {ALPHA}) are: "
                   + (", ".join(kept) or "none") + "."
                   + (f" One-feature links that fade once the others are held fixed: {', '.join(lost)}; they mostly "
                      "travel with the features that remain." if lost else ""))
    else:
        out.append(f"- **Joint model skipped:** {rt['joint_reason']}; each feature is tested alone, so the "
                   "associations above may overlap.")
    return out
