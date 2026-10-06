"""Q7: complaints per 1,000 residents by community district, disparities, and three computed insights."""
import re

import numpy as np
import pandas as pd
import pytest

from nyc311 import q07_geography as q7
from nyc311.clean import clean_glob


# ------------------------------------------------------------------ pure functions
def test_person_years_switch_population_by_period():
    demo = pd.DataFrame({"boro_cd": [101, 102], "population_2010": [1000, 500], "population_2020": [2000, 500]})
    py = q7.person_years(demo, days_2010=365.25, days_2020=730.5)
    assert py.loc[101] == pytest.approx(1000 + 2 * 2000)
    assert py.loc[102] == pytest.approx(500 * 3)


def test_district_rates_and_housing_split():
    demo = pd.DataFrame({"boro_cd": [101, 201], "borough": ["Manhattan", "Bronx"], "cd_number": [1, 1],
                         "cd_name": ["A", "B"], "population_2010": [1000, 2000], "population_2020": [1000, 2000],
                         "median_hh_income": [100000, 30000], "puma_2010": ["1", "2"]})
    counts = pd.DataFrame({"boro_cd": [101, 101, 201, 201], "category": ["Noise", "Heat & hot water"] * 2,
                           "n": [30, 10, 40, 160]})
    d = q7.district_table(counts, demo, q7.person_years(demo, 365.25, 0)).set_index("boro_cd")
    assert d.loc[101, "rate"] == pytest.approx(40.0) and d.loc[201, "rate"] == pytest.approx(100.0)
    assert d.loc[201, "housing_rate"] == pytest.approx(80.0) and d.loc[201, "housing_share_pct"] == pytest.approx(80)
    assert d.loc[201, "top_category"] == "Heat & hot water" and d.loc[101, "top_category"] == "Noise"


def test_housing_share_of_the_gap():
    top = {"rate": 300.0, "housing_rate": 200.0}
    bottom = {"rate": 100.0, "housing_rate": 50.0}
    assert q7.gap_share(top, bottom) == pytest.approx(75.0)


def test_quantile_classes_are_monotone_in_the_rate():
    rates = pd.Series([5, 50, 10, 40, 20, 30, 25, 15, 35, 45])
    cls, edges = q7.classify(rates, 5)
    assert sorted(set(cls)) == [0, 1, 2, 3, 4]
    order = rates.sort_values().index
    assert cls[order].is_monotonic_increasing
    assert edges[0] == rates.min() and edges[-1] == rates.max()


def test_pumas_pool_districts_that_share_one_income_value():
    d = pd.DataFrame({"boro_cd": [101, 102, 103], "puma_2010": ["03810", "03810", "03809"],
                      "median_hh_income": [118108, 118108, 42873], "n": [10, 30, 50], "n_housing": [1, 3, 40],
                      "person_years": [100.0, 100.0, 100.0]})
    p = q7.puma_units(d).set_index("puma_2010")
    assert len(p) == 2 and p.loc["03810", "rate"] == pytest.approx(200.0)
    assert p.loc["03810", "districts"] == "101, 102" and p.loc["03809", "housing_rate"] == pytest.approx(400.0)


def _synthetic_districts(k=12):
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"boro_cd": 200 + np.arange(1, k + 1), "borough": "Bronx", "cd_number": np.arange(1, k + 1),
                      "cd_name": [f"Area {i}" for i in range(1, k + 1)], "person_years": 1000.0,
                      "population": 1000, "puma_2010": [f"p{i}" for i in range(k)],
                      "median_hh_income": np.linspace(20000, 120000, k)})
    d["n_housing"] = np.linspace(400, 20, k).round()
    d["n"] = d.n_housing + rng.integers(100, 120, k)
    d["rate"], d["housing_rate"] = 1000 * d.n / d.person_years, 1000 * d.n_housing / d.person_years
    d["other_rate"] = d.rate - d.housing_rate
    return d


def test_insights_are_built_from_the_numbers_they_quote():
    d = _synthetic_districts()
    top, bottom = d.loc[d.rate.idxmax()], d.loc[d.rate.idxmin()]
    facts = q7.insight_facts(d, q7.puma_units(d), k=3)
    assert facts["top_name"] == f"{top.borough} {top.cd_number} ({top.cd_name})"
    assert facts["ratio"] == pytest.approx(top.rate / bottom.rate)
    texts = q7.insights(facts)
    assert len(texts) == 3
    assert f"{facts['ratio']:.1f}" in texts[0] and facts["top_name"] in texts[0]
    assert f"{facts['gap_housing_pct']:.0f}%" in texts[1]
    assert f"{facts['inc_low_housing']:,.0f}" in texts[2]


# ------------------------------------------------------------------ the module on the sample
@pytest.fixture(scope="module")
def q7_out(cleaned_cfg):
    return q7.run(cleaned_cfg)


@pytest.fixture(scope="module")
def con_src(cleaned_cfg):
    return cleaned_cfg.connect(), clean_glob(cleaned_cfg)


def test_all_59_districts_have_positive_rates(q7_out):
    d = q7_out["districts"]
    demo = pd.read_csv(q7.DEMO_CSV)
    assert sorted(d.boro_cd) == sorted(demo.boro_cd) and len(d) == 59
    assert (d.n > 0).all() and (d.person_years > 0).all()
    assert d.rate.between(1, 100_000).all()
    assert d.housing_share_pct.between(0, 100).all()
    assert np.allclose(d.rate, 1000 * d.n / d.person_years)


def test_exclusions_reconcile_to_the_default_set(q7_out, con_src):
    con, src = con_src
    ex = q7_out["exclusions"].set_index("step")
    default = con.sql(f"select count(*) from {src} where not is_test and not dup_key").fetchone()[0]
    assert ex.loc["default analysis set", "rows"] == default
    assert ex.rows.iloc[1:].sum() == default
    assert ex.loc["mapped to a community district", "rows"] == q7_out["districts"].n.sum()
    special = con.sql(f"""select count(*) from {src} where not is_test and not dup_key and special_area_cd is not null
                          and not coalesce(is_info_request, false) and not possibly_incomplete_day""").fetchone()[0]
    assert ex.loc["park or airport (special_area_cd)", "rows"] == special
    assert ex.pct.between(0, 100).all()


def test_borough_table_sums_the_districts(q7_out):
    b, d = q7_out["boroughs"].set_index("borough"), q7_out["districts"]
    assert b.n.sum() == d.n.sum()
    assert len(b) == 5
    bx = d[d.borough == "Bronx"]
    assert b.loc["Bronx", "rate"] == pytest.approx(1000 * bx.n.sum() / bx.person_years.sum())
    assert (b.n_with_borough_no_district >= 0).all()


def test_insights_compare_extremes_on_the_sample(q7_out):
    f = q7_out["facts"]
    d = q7_out["districts"]
    assert f["ratio"] == pytest.approx(d.rate.max() / d.rate.min())
    assert f["ratio"] > 1
    assert 0 < f["top_pop_share"] < 100 and f["top_n_share"] > f["top_pop_share"]
    assert len(q7_out["insights"]) == 3


def test_outputs_written(q7_out, cleaned_cfg):
    out = cleaned_cfg.out_dir
    for name in ["q07_district_map.png", "q07_district_rates.png", "q07_income.png"]:
        assert (out / "figs" / name).stat().st_size > 10_000
    for name in ["q07_districts.csv", "q07_boroughs.csv", "q07_exclusions.csv", "q07_pumas.csv"]:
        assert (out / "agg" / name).exists()
    text = (out / "answers" / "q07.md").read_text()
    assert text.startswith("# Q7")
    assert "![" in text and "(figs/q07_district_map.png)" in text
    assert not re.search(r"\bnan\b", text, re.IGNORECASE)
    for insight in q7_out["insights"]:
        assert insight in text
    assert (out / "reports" / "q07_geography.txt").read_text().startswith("===== BEGIN q07_geography =====")
