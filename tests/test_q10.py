"""Q10 features for predicting complaint volume.

Full path on a synthetic multi-year daily table with known structure (a drifting level, a weekday pattern, a holiday
dip, no weather effect, a 2020 level shift); end to end on a multi-year synthetic pull (tools/make_synthetic_batches.py);
the short path (and the whole module) on the 20-file sample (fixture `cleaned_cfg`)."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyc311 import clean, ingest, q10_features as q10, rules
from nyc311.clean import clean_glob
from nyc311.config import Config

REPO = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^(#{1,3} |- |  - |\d+\. |!\[|\*[^ ]|\||$)")


# ---------- synthetic data ----------

def synthetic_daily(start="2014-01-01", end="2021-09-22", seed=1, regime_effect=0.3, walk_sd=0.005):
    """Daily counts: a slowly drifting level (random walk), a weekday pattern, a seasonal cycle and a holiday dip; no
    weather effect; from REGIME_START a level shift and a much flatter weekday pattern. Response times: 5% faster each
    year (seen only against a type-only benchmark), slower on weekends, slower on days busier than the same weekday
    that year."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    doy = d.date.dt.dayofyear.to_numpy()
    w = pd.DataFrame({"date": pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=40), end, freq="D")})
    wd = w.date.dt.dayofyear.to_numpy()
    w["tmin_f"] = (45 + 18 * np.sin(2 * np.pi * (wd - 110) / 365.25) + rng.normal(0, 6, len(w))).round()
    w["tmax_f"] = (w.tmin_f + 14 + rng.normal(0, 3, len(w))).round()
    w["prcp_in"] = np.where(rng.random(len(w)) < 0.3, rng.exponential(0.4, len(w)), 0.0).round(2)
    w["snow_in"] = np.where((w.tmax_f < 36) & (w.prcp_in > 0), 10 * w.prcp_in, 0.0).round(1)
    after = (d.date >= pd.Timestamp(q10.REGIME_START)).to_numpy()
    dow_eff = np.array([0.15, 0.12, 0.10, 0.08, 0.05, -0.30, -0.35])[d.date.dt.dayofweek]
    dow_eff = np.where(after, 0.2 * dow_eff, dow_eff)
    hol = d.date.isin(pd.tseries.holiday.USFederalHolidayCalendar().holidays(start, end)).to_numpy()
    level = np.cumsum(rng.normal(0, walk_sd, len(d)))
    log_n = (8 + level + dow_eff + 0.12 * np.sin(2 * np.pi * doy / 365.25) - 0.4 * hol + regime_effect * after
             + rng.normal(0, 0.05, len(d)))
    d["n"] = np.round(np.exp(log_n)).astype(int)
    d["n_unresolved"] = (0.02 * d.n).round().astype(int)
    d["n_closed"] = (0.8 * d.n).round().astype(int)
    year = d.date.dt.year
    ln = np.log(d.n)
    busy = ln - ln.groupby([year, d.date.dt.dayofweek]).transform("mean")   # vs the same weekday that year
    trend = (d.date - d.date.min()).dt.days / 365.25
    weekend = (d.date.dt.dayofweek >= 5).astype(float)
    d["adj_type_log"] = (np.log(0.95) * trend + 0.3 * busy + 0.08 * weekend + rng.normal(0, 0.03, len(d)))
    d["adj_log"] = d.adj_type_log - d.adj_type_log.groupby(year).transform("mean")   # type-and-year benchmark
    d["median_h"] = 24 * np.exp(d.adj_type_log)
    return d, w


@pytest.fixture(scope="module")
def synth():
    d, w = synthetic_daily()
    return q10.make_features(d, w)


@pytest.fixture(scope="module")
def ev(synth):
    return q10.evaluate(synth)


# ---------- features ----------

def test_baseline_is_the_mean_of_the_last_four_same_weekdays():
    d, w = synthetic_daily("2019-01-01", "2019-04-30")
    d = d[d.date != "2019-02-01"]                     # a missing (incomplete) day
    f = q10.make_features(d, w).set_index("date")
    n = d.set_index("date").n
    day = pd.Timestamp("2019-03-01")                  # a Friday; Feb 22, 15, 8 and 1 before it, Feb 1 missing
    assert f.loc[day, "base"] == pytest.approx(np.mean([n[day - pd.Timedelta(days=7 * k)] for k in (1, 2, 3)]))
    full = pd.Timestamp("2019-03-14")
    assert f.loc[full, "base"] == pytest.approx(np.mean([n[full - pd.Timedelta(days=7 * k)] for k in range(1, 5)]))
    assert f.loc[full, "ratio_log"] == pytest.approx(np.log(n[full] / f.loc[full, "base"]))
    assert np.isnan(f.loc["2019-01-20", "base"])     # only 2 earlier same weekdays: no baseline
    # the count exactly 7 calendar days earlier (not the previous row), and the 28-day mean known a week ahead:
    # the 28 days from 34 to 7 days earlier, never the six days just before the target
    assert f.loc["2019-02-10", "lag7"] == n["2019-02-03"]
    assert np.isnan(f.loc["2019-02-08", "lag7"])
    m28 = n[(n.index >= full - pd.Timedelta(days=34)) & (n.index <= full - pd.Timedelta(days=7))]
    assert len(m28) == 28 and f.loc[full, "mean28"] == pytest.approx(m28.mean())


def _perturbed(d, days, factor=3.0):
    d = d.copy()
    d.loc[d.date.isin(days), "n"] = (factor * d.n[d.date.isin(days)]).round().astype(int)
    return d


def test_no_week_ahead_input_uses_counts_from_the_last_seven_days():
    """Blow up the counts of the target day and the six days before it: no forecast input on that day may move.
    (The original 28-day mean averaged days t-1..t-28 and failed this.)"""
    d, w = synthetic_daily("2018-01-01", "2018-12-31")
    f0 = q10.make_features(d, w).set_index("date")
    inputs = [c for c in q10.FORECAST_INPUTS if c in f0]
    assert {"lag7", "base", "mean28", "hol_base"} <= set(inputs) and len(inputs) == len(q10.FORECAST_INPUTS)
    for t in pd.to_datetime(["2018-03-01", "2018-07-04", "2018-11-15"]):
        recent = pd.date_range(t - pd.Timedelta(days=q10.HORIZON - 1), t)
        f1 = q10.make_features(_perturbed(d, recent), w).set_index("date")
        assert f1.loc[t, "n"] != f0.loc[t, "n"]                       # the perturbation took
        pd.testing.assert_series_equal(f1.loc[t, inputs], f0.loc[t, inputs], check_names=False)
        # ... while a count exactly a week back does move them (the test can see a change)
        f2 = q10.make_features(_perturbed(d, [t - pd.Timedelta(days=q10.HORIZON)]), w).set_index("date")
        assert f2.loc[t, "lag7"] != f0.loc[t, "lag7"] and f2.loc[t, "mean28"] != f0.loc[t, "mean28"]


def test_only_the_declared_weather_inputs_use_the_last_week_and_they_use_no_counts():
    """Weather on the target day and the day before stands in for a weather forecast (said in the findings); weather
    two to six days back is not used at all."""
    d, w = synthetic_daily("2018-01-01", "2018-12-31")
    t = pd.Timestamp("2018-07-18")
    f0 = q10.make_features(d, w).set_index("date")
    inputs = list(q10.FORECAST_INPUTS)

    def shifted(days):
        w1 = w.copy()
        m = w1.date.isin(days)
        w1.loc[m, ["tmax_f", "tmin_f"]] += 15
        w1.loc[m, ["prcp_in", "snow_in"]] += 1.0
        return q10.make_features(d, w1).set_index("date").loc[t, inputs]

    mid = pd.date_range(t - pd.Timedelta(days=q10.HORIZON - 1), t - pd.Timedelta(days=2))
    pd.testing.assert_series_equal(shifted(mid), f0.loc[t, inputs], check_names=False)
    moved = shifted([t - pd.Timedelta(days=1), t]) != f0.loc[t, inputs]
    assert set(moved[moved].index) <= set(q10.OBSERVED_WEATHER) and moved.any()
    assert not set(q10.OBSERVED_WEATHER) & {"lag7", "base", "mean28", "hol_base", "prcp_base", "snow_base"}


def test_fitted_models_train_only_on_days_a_week_before_the_test_year():
    """A forecast for Jan 1 is made by Dec 25, so the models scored on a test year are fit on days up to a week
    before its first day; perturbing the last week before a test day leaves every model's forecast for it unchanged."""
    d, w = synthetic_daily("2013-06-01", "2017-06-30")
    gap = pd.Timedelta(days=q10.HORIZON)

    def first_fold(dd):
        u = q10.make_features(dd, w).dropna(subset=q10.REQUIRED)
        p = next(p for p in q10.plan_folds(u) if p["test_year"] == 2017)
        groups = [g for g in q10.GROUPS if q10.group_availability(p["train"], p["test"]).set_index("group")
                  .loc[g, "estimable"]]
        return p, groups

    p0, groups = first_fold(d)
    t = p0["test"].date.min()
    assert p0["train"].date.max() <= t - gap and p0["train_all"].date.max() <= t - gap
    p1, _ = first_fold(_perturbed(d, pd.date_range(t - gap + pd.Timedelta(days=1), t)))
    pr0, pr1 = q10.fold_predictions(p0, groups), q10.fold_predictions(p1, groups)
    assert set(pr0) >= set(q10.MODELS) - {f"{k}{g}" for k in ("base+", "full-") for g in q10.GROUPS if g not in groups}
    for m in pr0:
        assert pr1[m][0] == pytest.approx(pr0[m][0], rel=1e-12), m


def test_holidays_and_weather_on_the_baseline_days():
    d, w = synthetic_daily("2019-06-01", "2019-08-31")
    f = q10.make_features(d, w).set_index("date")
    wi = w.set_index("date")
    day = pd.Timestamp("2019-07-18")                  # Thursday: July 11, 4 (a holiday), June 27 and 20 before it
    assert f.loc[day, "hol_base"] == pytest.approx(0.25)
    assert f.loc["2019-07-04", "holiday"] == 1 and f.loc["2019-07-05", "holiday"] == 0
    tb = np.mean([wi.loc[day - pd.Timedelta(days=7 * k), "tmax_f"] for k in range(1, 5)])
    assert f.loc[day, "tmax_rel"] == pytest.approx(wi.loc[day, "tmax_f"] - tb)
    pb = np.mean([wi.loc[day - pd.Timedelta(days=7 * k), "prcp_in"] for k in range(1, 5)])
    assert f.loc[day, "prcp_base"] == pytest.approx(pb)
    assert f.loc["2019-07-10", "prcp_lag1"] == pytest.approx(wi.loc["2019-07-09", "prcp_in"])


def test_calendar_and_regime_features():
    d, w = synthetic_daily("2019-06-25", "2020-04-10")
    f = q10.make_features(d, w).set_index("date")
    assert f.loc["2019-07-06", "dow_5"] == 1 and f.loc["2019-07-06", "weekend"] == 1   # a Saturday
    assert f.loc["2019-07-08", [f"dow_{i}" for i in range(1, 7)]].sum() == 0           # Monday base
    assert f.loc["2019-12-15", "m_12"] == 1 and f.loc["2019-12-15", [f"m_{i}" for i in range(1, 13)]].sum() == 1
    assert f.loc["2020-03-21", "regime"] == 0 and f.loc["2020-03-22", "regime"] == 1
    assert f.loc["2019-12-15", "heat_season"] == 1 and f.loc["2019-07-10", "heat_season"] == 0


# ---------- forecasting: full path (synthetic, multi-year) ----------

def test_folds_pool_recent_prepandemic_years_and_train_on_a_rolling_window(synth):
    plans = q10.plan_folds(synth.dropna(subset=q10.REQUIRED))
    years = [p["test_year"] for p in plans]
    typical = [p["test_year"] for p in plans if p["typical"]]
    assert typical == [2017, 2018, 2019] and years == [2017, 2018, 2019, 2020, 2021]
    assert max(typical) <= rules.PROFILE_LAST_YEAR
    for p in plans:
        y = p["test_year"]
        assert p["train"].date.max() < pd.Timestamp(f"{y}-01-01") <= p["test"].date.min()
        assert p["train"].date.min() >= pd.Timestamp(f"{y - q10.TRAIN_YEARS}-01-01")
        assert len(p["train"]) >= q10.MIN_TRAIN_DAYS and len(p["test"]) >= q10.MIN_TEST_DAYS
        # the robustness fit uses every earlier usable day, the window's days included
        assert p["train_all"].date.max() < pd.Timestamp(f"{y}-01-01") and len(p["train_all"]) >= len(p["train"])


def test_ratio_model_without_features_is_the_baseline_and_carries_no_trend():
    d, w = synthetic_daily("2014-01-01", "2019-12-31", regime_effect=0.0, walk_sd=0.0)
    d.loc[d.date >= "2017-01-01", "n"] = (0.7 * d.n[d.date >= "2017-01-01"]).round().astype(int)  # a level drop
    f = q10.make_features(d, w).dropna(subset=q10.REQUIRED)
    tr, te = f[f.date.dt.year.between(2014, 2016)], f[f.date >= "2017-03-01"]
    assert np.allclose(q10.fit_ratio(tr, te, []), te.base)
    pred = q10.fit_ratio(tr, te, ["holiday", "season", "weather"])
    assert abs(np.median(pred / te.n) - 1) < 0.03          # the level comes from recent weeks, not the training years
    cal = q10.fit_calendar(tr, te)
    assert np.median(cal / te.n) > 1.3                     # a calendar model without recent counts keeps the old level


def test_evaluation_values_the_planted_signals(ev, synth):
    assert ev["mode"] == "full"
    s = ev["scores"]
    assert s.mape.dropna().between(0, 100).all()
    per = s.pivot_table(index="model", columns="test_year", values="mape")
    typ = ev["summary"].set_index("model")[ev["typical_label"]]
    # pooled MAPE = all pooled days' errors / days, i.e. the day-weighted mean of the yearly MAPEs
    t = s[s.test_year.isin(ev["typical_years"])]
    b = t[t.model == "base"]
    assert typ["base"] == pytest.approx((b.mape * b.n_test).sum() / b.n_test.sum())
    assert (per.loc["base", ev["typical_years"]] < per.loc["lag7", ev["typical_years"]]).all()  # 4 weeks beat 1
    assert typ["full"] < typ["base"] < typ["calendar"]
    assert np.isfinite(typ["full_all"])                     # the same model fit on every earlier year, for comparison
    v = ev["values"].set_index("group")
    col = f"value_{ev['typical_label']}"
    assert v.loc["holiday", col] > q10.MIN_GAIN                             # the planted holiday dip
    assert v.loc["holiday", "n_better"] == len(ev["typical_years"])          # in every pre-pandemic test year
    assert v.loc["dow", col] > 1 and v.loc["volume", col] > 1                # weekday pattern and drifting level
    assert abs(v.loc["weather", col]) < q10.MIN_GAIN                         # no weather effect was planted
    assert v.loc["dow", col] == pytest.approx(typ["mean28"] - typ["base"])
    assert v.loc["holiday", col] == pytest.approx(typ["base"] - typ["base+holiday"])
    assert v.loc["holiday", f"dropped_{ev['typical_label']}"] == pytest.approx(typ["full-holiday"] - typ["full"])
    # 2020: the level shift breaks a calendar model; recent volume follows it, and the flatter week erodes the
    # value of matching the weekday
    assert per.loc["calendar", 2020] > per.loc["base", 2020] + 5
    assert v.loc["dow", "value_2020"] < v.loc["dow", col]


def test_drift_table_tracks_the_weekday_pattern_and_holiday_dip(synth):
    dr = q10.drift_table(synth).set_index("year")
    assert dr.loc[2016, "weekend_ratio"] < 0.8 < dr.loc[2021, "weekend_ratio"]   # the week flattened after the shift
    assert dr.loc[2016, "holiday_pct"] < -20                                     # planted dip: exp(-0.4) - 1 = -33%
    assert (dr.n_holidays <= 12).all() and dr.n_days.sum() == len(synth)


# ---------- response times ----------

def test_year_level_features_use_a_benchmark_that_does_not_absorb_them(synth):
    rt = q10.response_tests(synth)
    t = rt["tests"].set_index("feature")
    assert t.loc["trend_years", "benchmark"] == "type only" and t.loc["regime", "benchmark"] == "type only"
    assert t.loc["weekend", "benchmark"] == "type and year"
    assert -7 < t.loc["trend_years", "effect_pct"] < -3 and t.loc["trend_years", "significant"]
    assert t.loc["trend_years", "hac_lags"] == q10.YEAR_HAC_LAGS > t.loc["weekend", "hac_lags"]
    # year-level tests use full calendar years only (2021 is partial here)
    assert t.loc["trend_years", "n_years"] == 7 and t.loc["trend_years", "n"] == (synth.date.dt.year <= 2020).sum()
    assert t.loc["weekend", "effect_pct"] > 0 and t.loc["weekend", "significant"]
    assert t.rho.dropna().between(-1, 1).all()
    j = rt["joint"].set_index("group")                 # held with weekday and month, busier days close slower
    assert j.loc["volume", "p"] < q10.ALPHA and j.loc["volume", "effect"].startswith("+")


def test_significance_is_classified_by_the_hac_test_only(synth):
    t = q10.response_tests(synth)["tests"]
    est = t[t.estimable]
    assert (est.significant == (est.p < q10.ALPHA)).all()
    assert not t[~t.estimable].significant.any()


# ---------- outputs (full path) ----------

@pytest.fixture(scope="module")
def full_out(synth, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("q10full")
    ex = pd.DataFrame({"step": ["default analysis set", "analyzed (complete days)"],
                       "rows": [int(synth.n.sum())] * 2, "pct": [100.0, 100.0]})
    res = q10.analyze(synth, ex)
    (tmp / "figs").mkdir()
    (tmp / "figs" / "q10_feature_correlations.png").write_bytes(b"stale")   # from an earlier short run
    q10.write_outputs(SimpleNamespace(out_dir=tmp), res)
    return res, tmp


def test_full_path_writes_the_chart_and_findings(full_out):
    res, tmp = full_out
    assert (tmp / "figs" / "q10_feature_importance.png").stat().st_size > 10_000
    assert not (tmp / "figs" / "q10_feature_correlations.png").exists()
    md = (tmp / "answers" / "q10.md").read_text()
    lines = md.splitlines()
    assert lines[0] == "# Q10. Features for predicting complaint volume"
    assert lines[2].startswith("*Computed by `nyc311/q10_features.py` from ") and lines[2].endswith(".*")
    assert "write the prose" not in md and "nan" not in md.lower()
    assert "figs/q10_feature_importance.png" in md
    ev = res["evaluation"]
    typ = ev["summary"].set_index("model")[ev["typical_label"]]
    assert f"{typ['base']:.1f}%" in res["figure_title"] and f"{typ['full']:.1f}%" in res["figure_title"]
    assert ev["typical_label"] in md and "2020" in md
    for line in lines:
        assert LINE.match(line), line
    rep = (tmp / "reports" / "q10_features.txt").read_text().splitlines()
    assert len(rep) <= 152


def test_findings_follow_the_week_ahead_comparison(full_out):
    """Every sentence about the weekday's value matches the corrected (week-ahead) numbers."""
    res, tmp = full_out
    lines = (tmp / "answers" / "q10.md").read_text().splitlines()
    ev = res["evaluation"]
    summ, v, lab = ev["summary"].set_index("model"), ev["values"].set_index("group"), ev["typical_label"]
    assert ev["later_years"]
    for y in ev["later_years"]:
        bullet = next(l for l in lines if l.startswith(f"- **{y}"))
        beat = summ.loc["mean28", str(y)] < summ.loc["base", str(y)]
        assert ("beat the same-weekday mean" in bullet) == beat, y
        assert "faded" not in bullet
    rank = next(l for l in lines if "**Day of week**" in l)
    assert f"{q10._num(v.loc['dow', f'value_{lab}'], '+.2f')} points" in rank and "7 to 34 days earlier" in rank
    drift = next(l for l in lines if l.startswith(("- **The patterns drift", "- **Weekly and holiday")))
    later = [v.loc["dow", f"value_{y}"] for y in ev["later_years"]]
    if "worth less" in drift:
        assert max(later) < v.loc["dow", f"value_{lab}"]
    assert "7-34 days earlier" in q10.MODELS["mean28"]


def test_findings_put_the_response_tests_in_multiple_testing_context(full_out):
    res, tmp = full_out
    md = (tmp / "answers" / "q10.md").read_text()
    est = res["response"]["tests"].query("estimable")
    n, k = len(est), int(est.significant.sum())
    line = next(l for l in md.splitlines() if l.startswith("- **Multiple tests.**"))
    assert f"{n} one-feature response-time tests" in line and f"about {n * q10.ALPHA:.1f} would pass" in line
    assert f"{k} passed" in line
    if k:
        assert f"{int((est.p < q10.ALPHA / n).sum())} of them also below the Bonferroni threshold" in line


def test_findings_list_features_by_the_same_test_they_show(full_out):
    res, tmp = full_out
    md = (tmp / "answers" / "q10.md").read_text()
    yes = next(l for l in md.splitlines() if l.startswith("- **Correlated"))
    no = next(l for l in md.splitlines() if l.startswith("- **Not correlated"))
    for r in res["response"]["tests"].itertuples():
        if r.estimable:
            assert f"{r.label}:" in (yes if r.significant else no), r.label


def test_chart_panels_share_one_readable_scale(full_out):
    res, _ = full_out
    fig = q10.forecast_figure(res["evaluation"], "title", "note")
    axes = [a for a in fig.axes if a.get_visible()]
    assert len(axes) == 1 + len(res["evaluation"]["later_years"])
    lims = {tuple(np.round(a.get_xlim(), 6)) for a in axes}
    assert len(lims) == 1
    lo, hi = lims.pop()
    ticks = [t for t in axes[0].get_xticks() if lo <= t <= hi]
    assert 3 <= len(ticks) <= 6


# ---------- short path ----------

def test_short_data_is_detected_and_explained():
    d, w = synthetic_daily("2010-01-01", "2010-02-15")
    f = q10.make_features(d, w)
    ev = q10.evaluate(f)
    assert ev["mode"] == "short" and ev["reason"]
    av = ev["availability"].set_index("group")
    assert not av.loc["season", "estimable"] and av.loc["dow", "estimable"]
    assert av.reason.str.len().gt(0).all()
    nv = ev["naive"].set_index("model")
    assert nv.n_days.nunique() == 1 and nv.loc["base", "n_days"] > 0 and nv.mape.between(0, 100).all()
    rt = q10.response_tests(f)                       # still runs; the joint model is skipped for lack of days
    assert rt["joint"] is None and rt["joint_reason"]
    assert not rt["tests"].set_index("feature").loc["trend_years", "estimable"]


# ---------- the sample ----------

@pytest.fixture(scope="module")
def res(cleaned_cfg):
    return q10.run(cleaned_cfg)


def test_sample_daily_counts_reconcile_and_skip_incomplete_days(res, cleaned_cfg):
    con, src = cleaned_cfg.connect(), clean_glob(cleaned_cfg)
    daily = res["daily"]
    inc = set(pd.to_datetime(con.sql(f"select distinct created::date d from {src} where possibly_incomplete_day")
                             .df().d))
    assert not (set(daily.date) & inc)
    days = ", ".join(f"'{d:%Y-%m-%d}'" for d in daily.date)
    n = con.sql(f"select count(*) from {src} where {rules.COMPLAINTS_SQL} and created::date in ({days})").fetchone()[0]
    assert daily.n.sum() == n
    ex = res["exclusions"]
    assert ex.rows.iloc[-1] == daily.n.sum()
    assert ex.rows.iloc[-1] == ex.rows.iloc[0] - ex.rows.iloc[1:-1].sum()
    # both response-time benchmarks are present; the type-only one averages to about zero over all closures
    assert daily.adj_log.notna().any() and daily.adj_type_log.notna().any()


def test_sample_runs_in_short_mode_and_says_so(res, cleaned_cfg):
    assert res["evaluation"]["mode"] == "short"
    out = cleaned_cfg.out_dir
    md = (out / "answers" / "q10.md").read_text()
    assert md.splitlines()[2].startswith("*Computed by `nyc311/q10_features.py` from ")
    assert "not estimable" in md.lower() or "cannot" in md.lower()
    assert "nan" not in md.lower().replace("nanc", "")
    assert (out / "figs" / "q10_feature_correlations.png").stat().st_size > 10_000
    assert not (out / "figs" / "q10_feature_importance.png").exists()
    assert "figs/q10_feature_correlations.png" in md
    for line in md.splitlines():
        assert LINE.match(line), line
    t = res["response"]["tests"]
    assert (t.n <= len(res["daily"])).all() and t.rho.dropna().between(-1, 1).all()
    rep = (out / "reports" / "q10_features.txt").read_text().splitlines()
    assert rep[0] == "===== BEGIN q10_features =====" and rep[-1] == "===== END q10_features ====="


# ---------- several years (synthetic pull, end to end) ----------

@pytest.fixture(scope="module")
def pull(tmp_path_factory):
    sys.path.insert(0, str(REPO / "tools"))
    import make_synthetic_batches as synth
    root = tmp_path_factory.mktemp("q10years")
    synth.main(["--files", "70", "--rows-per-file", "300", "--start", "2010-01-01", "--end", "2013-12-31",
                "--out", str(root / "data")])
    cfg = Config.from_args(["--data", str(root / "data" / "311_batch_*.json"), "--out", str(root / "out"),
                            "--work", str(root / "work"), "--threads", "2", "--memory-limit", "2GB"])
    ingest.run(cfg)
    clean.run(cfg)
    return cfg, q10.run(cfg)


def test_several_years_run_the_forecast_test_end_to_end(pull):
    cfg, out = pull
    ev = out["evaluation"]
    assert ev["mode"] == "full" and ev["typical_years"] and not ev["later_years"]
    con, src = cfg.connect(), clean_glob(cfg)
    assert out["full_years"] == clean.full_years(con, src)
    assert ev["scores"].mape.dropna().between(0, 1000).all()
    md = (cfg.out_dir / "answers" / "q10.md").read_text()
    assert "figs/q10_feature_importance.png" in md and "nan" not in md.lower()
    for line in md.splitlines():
        assert LINE.match(line), line
    assert (cfg.out_dir / "figs" / "q10_feature_importance.png").exists()
    assert len((cfg.out_dir / "reports" / "q10_features.txt").read_text().splitlines()) <= 152
