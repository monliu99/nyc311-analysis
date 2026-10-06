"""Q6: daily complaint volume and content vs Central Park weather."""
import re

import numpy as np
import pandas as pd
import pytest

from nyc311 import q06_weather as q6
from nyc311.clean import clean_glob


# ------------------------------------------------------------------ pure functions
def test_sparse_edge_bins_merge_inward_and_reference_is_the_mild_bin():
    # 3 days <20F, 10 in the 20s, 10 in the 30s, 2 in the 40s: both edge bins are too thin
    tmin = pd.Series([10] * 3 + [25] * 10 + [35] * 10 + [45] * 2)
    labels, ref = q6.tmin_bins(tmin, min_days=5)
    assert list(labels.cat.categories) == ["<30°F", "30°F+"]
    assert labels.value_counts().to_dict() == {"<30°F": 13, "30°F+": 12}
    assert ref == "30°F+"  # the open top bin contains the 55F reference temperature


def test_full_range_keeps_ten_degree_bins_with_50s_as_reference():
    tmin = pd.Series(np.repeat(np.arange(0, 100), 10))
    labels, ref = q6.tmin_bins(tmin, min_days=5)
    assert list(labels.cat.categories) == ["<20°F", "20-29°F", "30-39°F", "40-49°F", "50-59°F", "60-69°F", "70°F+"]
    assert ref == "50-59°F"
    assert labels.notna().all()


def test_weather_events_and_lags_use_the_calendar_day_before():
    w = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=5),
        "tmax_f": [40] * 5, "tmin_f": [30] * 5,
        "prcp_in": [0.0, 0.7, 0.2, 0.6, 0.0],
        "snow_in": [0.0, 0.0, 0.0, 6.0, 0.0],
        "snwd_in": [0.0] * 5})
    f = q6.weather_features(w).set_index("date")
    assert f.rain_heavy.tolist() == [False, True, False, False, False]  # day 4 is snow, not rain
    assert f.rain_light.tolist() == [False, False, True, False, False]
    assert f.snow.tolist() == [False, False, False, True, False]
    assert f.rain_heavy_lag1.tolist() == [False, False, True, False, False]
    assert f.snow_lag1.tolist() == [False, False, False, False, True]


def _synthetic_days(n=364, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.date_range("2015-03-02", periods=n)})
    d["tmin_bin"] = pd.Categorical(rng.choice(["cold", "mild"], size=n), categories=["cold", "mild"], ordered=True)
    d["rain_heavy"] = rng.random(n) < 0.15
    d["snow"] = False
    d.loc[[10, 50], "snow"] = True  # only two snow days: too few to estimate
    weekend = d.date.dt.dayofweek >= 5
    log_mu = 6 + 0.4 * (d.tmin_bin == "cold") + 0.25 * d.rain_heavy - 0.5 * weekend + 0.1 * d.date.dt.month / 12
    d["y"] = np.round(np.exp(log_mu + rng.normal(0, 0.05, n)))
    return d


def test_effects_recover_known_percent_changes_and_skip_rare_events():
    d = _synthetic_days()
    eff, skipped = q6.fit_effects(d, ["y"], "tmin_bin", "mild", ["rain_heavy", "snow"], min_event_days=5)
    eff = eff.set_index("term")
    assert skipped == {"snow": 2}
    cold, rain = eff.loc["cold"], eff.loc["rain_heavy"]
    assert cold.ci_low_pct <= 100 * (np.exp(0.4) - 1) <= cold.ci_high_pct
    assert rain.ci_low_pct <= 100 * (np.exp(0.25) - 1) <= rain.ci_high_pct
    assert abs(cold.effect_pct - 100 * (np.exp(0.4) - 1)) < 3
    assert (eff.ci_low_pct <= eff.effect_pct).all() and (eff.effect_pct <= eff.ci_high_pct).all()
    assert "snow" not in eff.index and "mild" not in eff.index  # reference bin has no row
    assert cold.days_with == int((d.tmin_bin == "cold").sum())


def test_wettest_day_is_compared_with_dry_days_of_the_same_month_and_day_type():
    d = pd.DataFrame({"date": pd.date_range("2015-06-01", periods=30), "tmax_f": 80, "tmin_f": 65,
                      "prcp_in": 0.0, "snow_in": 0.0, "snwd_in": 0.0})
    d.loc[9, "prcp_in"] = 2.0  # Wednesday 2015-06-10
    d = q6.weather_features(d)
    d["sewer"] = np.where(d.date.dt.dayofweek >= 5, 40, 100)
    d.loc[9, "sewer"], d.loc[10, "sewer"] = 300, 900  # the day after a storm is excluded from the baseline
    x = q6.extreme_day(d, "prcp_in", ["sewer"], q6.HEAVY_RAIN_IN)
    assert x["date"] == pd.Timestamp("2015-06-10") and x["value"] == 2.0
    assert x["rows"] == [("sewer", 300, 100.0)]
    assert q6.extreme_day(d.assign(prcp_in=0.2), "prcp_in", ["sewer"], q6.HEAVY_RAIN_IN) is None


def test_three_synthetic_years_estimate_every_bin_and_event_and_draw_every_figure(tmp_path):
    """The full-data path (all bins, rain/snow, seasons) on real 2015-2017 weather with simulated counts."""
    rng = np.random.default_rng(1)
    w = q6.load_weather()
    days = w[w.date.dt.year.between(2015, 2017)].reset_index(drop=True)
    cold = np.clip(50 - days.tmin_f, 0, None)
    mu = {"heat": 50 * np.exp(0.04 * cold), "sewer": 80 * np.exp(0.6 * days.rain_heavy),
          "snowice": 5 * np.exp(1.5 * (days.snow | days.snow_lag1))}
    for g in q6.GROUPS:
        days[g.key] = rng.poisson(mu.get(g.key, 100 + 0 * cold))
    days["total"] = days[[g.key for g in q6.GROUPS]].sum(axis=1) + rng.poisson(3000, len(days))
    cat = pd.concat([days[["date"]].assign(category="Heat & hot water", n=days.heat),
                     days[["date"]].assign(category="Everything else", n=days.total - days.heat)])
    groups = pd.DataFrame({"group": [g.key for g in q6.GROUPS], "label": [g.label for g in q6.GROUPS],
                           "category": "x", "canonical_type": "y", "n": 1})
    ex = pd.DataFrame({"step": ["default analysis set", "analyzed"], "rows": [days.total.sum()] * 2})
    out = q6.build(tmp_path, days, cat, groups, ex)
    assert len(out["bin_order"]) == 7 and out["ref"] == "50-59°F"
    assert {"rain_heavy", "snow", "snow_lag1"} <= set(out["effects"].term)
    eff = out["effects"].set_index(["series", "term"])
    assert eff.loc[("sewer", "rain_heavy"), "ci_low_pct"] > 50
    assert eff.loc[("heat", "<20°F"), "effect_pct"] > eff.loc[("heat", "40-49°F"), "effect_pct"] > 0
    for name in ["q06_volume_vs_weather", "q06_temperature_effects", "q06_rain_snow_effects",
                 "q06_content_by_temperature", "q06_seasonal_profile"]:
        assert (tmp_path / "figs" / f"{name}.png").exists()
        assert f"(figs/{name}.png)" in (tmp_path / "answers" / "q06.md").read_text()
    assert len(out["seasonal"]) == 12


# ------------------------------------------------------------------ the module on the sample
@pytest.fixture(scope="module")
def q6_out(cleaned_cfg):
    return q6.run(cleaned_cfg)


@pytest.fixture(scope="module")
def con_src(cleaned_cfg):
    return cleaned_cfg.connect(), clean_glob(cleaned_cfg)


def test_daily_table_drops_incomplete_and_uncovered_days(q6_out, con_src):
    con, src = con_src
    days = q6_out["days"]
    incomplete = set(con.sql(f"select distinct created::date d from {src} where possibly_incomplete_day").df().d)
    assert incomplete and not (set(days.date.dt.date) & {pd.Timestamp(d).date() for d in incomplete})
    # days that hold only out-of-file-order rows (2013-04-17, 2015-10-01 in the sample) are outside the files' period
    assert days.date.dt.year.eq(2010).all()
    assert days.date.is_unique and days[["tmin_f", "prcp_in", "snow_in"]].notna().all().all()


def test_daily_totals_reconcile_with_the_clean_table(q6_out, con_src):
    con, src = con_src
    days = q6_out["days"].set_index("date")
    d0 = days.index[0]
    n = con.sql(f"""select count(*) from {src} where not is_test and not dup_key
                    and not coalesce(is_info_request, false) and created::date = '{d0.date()}'""").fetchone()[0]
    assert days.loc[d0, "total"] == n
    cats = q6_out["category_daily"].groupby("date").n.sum()
    assert (cats.reindex(days.index) == days.total).all()
    for g in q6.GROUPS:
        assert (days[g.key] <= days.total).all()
    assert days.heat.sum() > 0 and days.snowice.sum() > 0


def test_exclusions_reconcile_to_the_default_set(q6_out, con_src):
    con, src = con_src
    ex = q6_out["exclusions"].set_index("step")
    default = con.sql(f"select count(*) from {src} where not is_test and not dup_key").fetchone()[0]
    assert ex.loc["default analysis set", "rows"] == default
    assert ex.rows.iloc[1:].sum() == default  # removed steps plus the rows analyzed add back to the default set
    assert ex.loc["analyzed", "rows"] == q6_out["days"].total.sum()


def test_group_membership_is_listed_and_heat_is_the_heat_category(q6_out):
    m = q6_out["groups"]
    heat = m[m.group == "heat"]
    assert set(heat.category) == {"Heat & hot water"}
    assert {"heat", "sewer", "street", "snowice", "trees", "noise"} <= set(m.group)
    trees = m[m.group == "trees"].canonical_type
    assert trees.str.contains("Tree").all() and not trees.str.contains("Street").any()


def test_effects_have_ordered_cis_and_cold_nights_raise_heat_complaints(q6_out):
    eff = q6_out["effects"]
    assert (eff.ci_low_pct <= eff.effect_pct).all() and (eff.effect_pct <= eff.ci_high_pct).all()
    temp = eff[(eff.series == "heat") & (eff.kind == "temperature")]
    coldest = temp[temp.term == q6_out["bin_order"][0]].iloc[0]
    assert coldest.effect_pct > 0
    # one month of sample has one heavy-rain day and one snow day: too few to estimate
    assert set(q6_out["skipped_events"]) >= {"rain_heavy", "snow"}


def test_category_shares_sum_to_100_in_every_band(q6_out):
    s = q6_out["shares"]
    totals = s.groupby("tmin_bin", observed=True).share_pct.sum()
    assert np.allclose(totals, 100)
    assert set(s.tmin_bin) == set(q6_out["bin_order"])


def test_outputs_written(q6_out, cleaned_cfg):
    out = cleaned_cfg.out_dir
    for name in ["q06_volume_vs_weather.png", "q06_temperature_effects.png", "q06_content_by_temperature.png"]:
        assert (out / "figs" / name).stat().st_size > 10_000
    for name in ["q06_daily.csv", "q06_effects.csv", "q06_shares_by_temperature.csv", "q06_groups.csv"]:
        assert (out / "agg" / name).exists()
    text = (out / "answers" / "q06.md").read_text()
    assert not re.search(r"\bnan\b", text, re.IGNORECASE)
    assert f"{q6_out['days'].total.sum():,}" in text
    report = (out / "reports" / "q06_weather.txt").read_text()
    assert report.startswith("===== BEGIN q06_weather =====")
