"""Q10 features for predicting complaint volume: the full path on a synthetic multi-year daily table, the short path
(and the whole module) on the 20-file sample (fixture `cleaned_cfg`)."""
import re

import numpy as np
import pandas as pd
import pytest

from nyc311 import q10_features as q10
from nyc311.clean import clean_glob

SET = ("not is_test and not dup_key and not coalesce(is_info_request, false) "
       "and coalesce(category, '') <> 'Invalid entries'")


# ---------- synthetic data ----------

def synthetic_daily(start="2014-01-01", end="2021-06-30", seed=1, regime_effect=0.0):
    """Daily counts with a strong weekday pattern, a seasonal cycle, a holiday dip, a cold-weather effect and a
    trend; response times rise with volume and ignore rain."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    doy = d.date.dt.dayofyear.to_numpy()
    w = pd.DataFrame({"date": d.date})
    w["tmin_f"] = 45 + 18 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 6, len(d))
    w["tmax_f"] = w.tmin_f + 14 + rng.normal(0, 3, len(d))
    w["prcp_in"] = np.where(rng.random(len(d)) < 0.3, rng.exponential(0.4, len(d)), 0.0).round(2)
    w["snow_in"] = np.where((w.tmax_f < 36) & (w.prcp_in > 0), 10 * w.prcp_in, 0.0).round(1)
    dow_eff = np.array([0.15, 0.12, 0.10, 0.08, 0.05, -0.30, -0.35])
    hol = d.date.isin(pd.tseries.holiday.USFederalHolidayCalendar().holidays(start, end))
    trend = (d.date - d.date.min()).dt.days / 365.25
    regime = (d.date >= pd.Timestamp(q10.REGIME_START)).astype(float)
    log_n = (8 + dow_eff[d.date.dt.dayofweek] + 0.12 * np.sin(2 * np.pi * doy / 365.25) - 0.4 * hol
             - 0.006 * (w.tmin_f - 45) + 0.05 * trend + regime_effect * regime + rng.normal(0, 0.05, len(d)))
    d["n"] = np.round(np.exp(log_n)).astype(int)
    d["n_unresolved"] = (0.02 * d.n).round().astype(int)
    d["n_closed"] = (0.8 * d.n).round().astype(int)
    d["adj_log"] = 0.5 * (np.log(d.n) - np.log(d.n).mean()) + rng.normal(0, 0.05, len(d))
    d["median_h"] = 24 * np.exp(d.adj_log)
    return d, w


@pytest.fixture(scope="module")
def synth():
    d, w = synthetic_daily(regime_effect=0.3)
    return q10.make_features(d, w)


# ---------- features ----------

def test_lag7_is_the_count_exactly_seven_calendar_days_earlier():
    d, w = synthetic_daily("2019-01-01", "2019-03-31")
    d = d[d.date != "2019-02-01"]                     # a missing (incomplete) day
    f = q10.make_features(d, w).set_index("date")
    assert f.loc["2019-02-10", "lag7_log"] == pytest.approx(np.log(f.loc["2019-02-03", "n"]))
    assert np.isnan(f.loc["2019-02-08", "lag7_log"])  # 7 days earlier is missing, not the previous row
    assert np.isnan(f.loc["2019-01-05", "lag7_log"])


def test_calendar_weather_and_regime_features():
    d, w = synthetic_daily("2019-06-25", "2020-04-10")
    f = q10.make_features(d, w).set_index("date")
    assert f.loc["2019-07-04", "holiday"] == 1 and f.loc["2019-07-05", "holiday"] == 0
    assert f.loc["2019-07-06", "dow_5"] == 1 and f.loc["2019-07-06", "weekend"] == 1   # a Saturday
    assert f.loc["2019-07-08", ["dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]].sum() == 0   # Monday base
    assert f.loc["2019-12-15", "m_12"] == 1
    assert f.loc["2020-03-21", "regime"] == 0 and f.loc["2020-03-22", "regime"] == 1
    assert f.loc["2019-07-10", "prcp_lag1"] == pytest.approx(w.set_index("date").loc["2019-07-09", "prcp_in"])
    assert f.loc["2019-12-15", "heat_season"] == 1 and f.loc["2019-07-10", "heat_season"] == 0


# ---------- full path (synthetic, multi-year) ----------

def test_folds_are_the_last_years_each_trained_on_everything_before(synth):
    plan = q10.plan_folds(synth)
    assert [p["test_year"] for p in plan] == [2019, 2020, 2021]
    for p in plan:
        assert p["train"].date.max() < pd.Timestamp(f"{p['test_year']}-01-01") <= p["test"].date.min()
        assert len(p["train"]) >= q10.MIN_TRAIN_DAYS and len(p["test"]) >= q10.MIN_TEST_DAYS


def test_full_evaluation_ranks_known_signals(synth):
    ev = q10.evaluate(synth)
    assert ev["mode"] == "full"
    s = ev["scores"]
    assert s.mape.between(0, 100).all() and (s.r2 <= 1).all()
    full = s[s.model == "full"].set_index("test_year")
    base = s[s.model == "intercept only"].set_index("test_year")
    assert (full.mape < base.mape).all()
    imp = ev["importance"].set_index("group")
    assert imp.loc["dow", "d_mape"] > 1                 # median over test years: the weekday pattern matters
    assert imp.loc["dow", "d_mape"] > imp.loc["weather", "d_mape"]
    assert imp.loc["holiday", "d_mape"] > 0
    assert imp.loc["dow", "d_mape"] == pytest.approx(np.median([imp.loc["dow", f"d_mape_{y}"] for y in full.index]))
    # 2020 carries a level shift the training years never saw: last week's count beats the calendar model there
    assert ev["breaks"] == [2020]
    alone = s[(s.model == "alone") & (s.group == "dow")].set_index("test_year")
    assert alone.loc[2019, "mape"] < base.loc[2019, "mape"] - 5   # in a normal year the weekday alone helps
    # the regime can only be learned once training data include it: not estimable for the 2019 and 2020 folds
    fe = ev["fold_groups"]
    reg = fe[fe.group == "regime"].set_index("test_year").estimable
    assert not reg[2019] and not reg[2020] and reg[2021]
    assert imp.loc["regime", "n_folds"] == 1


def test_response_time_tests_find_the_built_in_volume_link(synth):
    rt = q10.response_tests(synth)
    sp = rt["spearman"].set_index(["feature", "outcome"])
    vol = sp.loc[("log_n", "adj_pct")]
    assert vol.n == synth.adj_log.notna().sum()
    assert vol.rho > 0.5 and vol.p < 0.001
    assert abs(sp.loc[("prcp_in", "adj_pct"), "rho"]) < 0.1
    hac = rt["univariate"].set_index("feature")
    assert hac.loc["log_n", "effect_pct"] > 0 and hac.loc["log_n", "p"] < 0.001
    assert rt["joint"] is not None and "log_n" in set(rt["joint"].group)


def test_full_path_writes_the_importance_chart_and_findings(synth, tmp_path):
    from types import SimpleNamespace
    ex = pd.DataFrame({"step": ["default analysis set", "analyzed (complete days)"],
                       "rows": [int(synth.n.sum())] * 2, "pct": [100.0, 100.0]})
    res = q10.analyze(synth, ex)
    cfg = SimpleNamespace(out_dir=tmp_path)
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "q10_feature_correlations.png").write_bytes(b"stale")   # from an earlier short run
    q10.write_outputs(cfg, res)
    assert (tmp_path / "figs" / "q10_feature_importance.png").stat().st_size > 10_000
    assert not (tmp_path / "figs" / "q10_feature_correlations.png").exists()
    md = (tmp_path / "answers" / "q10.md").read_text()
    assert md.startswith("# Q10") and "figs/q10_feature_importance.png" in md
    assert "Level shift in 2020" in md and "nan" not in md.lower()
    imp = res["evaluation"]["importance"]
    top = imp[imp.n_folds == imp.n_folds.max()].iloc[0]       # the title names a group seen in every test year
    assert top.label in res["figure_title"] and f"{top.d_mape:.1f}" in res["figure_title"]
    assert "estimable in only 1 of 3 test years" in md
    assert res["effects"] is not None and res["response"]["joint"] is not None
    for line in md.splitlines():
        assert re.match(r"^(#{1,3} |- |  - |\d+\. |!\[|\*[^ ]|\||$)", line), line


# ---------- short path ----------

def test_short_data_is_detected_and_explained():
    d, w = synthetic_daily("2010-01-01", "2010-02-03")
    f = q10.make_features(d, w)
    ev = q10.evaluate(f)
    assert ev["mode"] == "short"
    av = ev["availability"].set_index("group")
    assert not av.loc["season", "estimable"] and not av.loc["trend", "estimable"]
    assert not av.loc["regime", "estimable"]
    assert av.loc["dow", "estimable"]
    assert av.reason.str.len().gt(0).all()
    rt = q10.response_tests(f)                       # still runs; the joint model is skipped for lack of days
    assert rt["joint"] is None and rt["joint_reason"]
    assert rt["spearman"].rho.dropna().between(-1, 1).all()


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
    n = con.sql(f"select count(*) from {src} where {SET} and created::date in ({days})").fetchone()[0]
    assert daily.n.sum() == n
    ex = res["exclusions"]
    assert ex.rows.iloc[-1] == daily.n.sum()
    assert ex.rows.iloc[-1] == ex.rows.iloc[0] - ex.rows.iloc[1:-1].sum()


def test_sample_runs_in_short_mode_and_says_so(res, cleaned_cfg):
    assert res["evaluation"]["mode"] == "short"
    out = cleaned_cfg.out_dir
    md = (out / "answers" / "q10.md").read_text()
    assert md.startswith("# Q10")
    assert "not estimable" in md.lower() or "cannot" in md.lower()
    assert "nan" not in md.lower().replace("nanc", "")
    assert (out / "figs" / "q10_feature_correlations.png").stat().st_size > 10_000
    assert not (out / "figs" / "q10_feature_importance.png").exists()
    assert "figs/q10_feature_correlations.png" in md
    for line in md.splitlines():
        assert re.match(r"^(#{1,3} |- |  - |\d+\. |!\[|\*[^ ]|\||$)", line), line
    sp = res["response"]["spearman"]
    assert (sp.n <= len(res["daily"])).all() and sp.rho.dropna().between(-1, 1).all()
    rep = (out / "reports" / "q10_features.txt").read_text().splitlines()
    assert rep[0] == "===== BEGIN q10_features =====" and rep[-1] == "===== END q10_features ====="
