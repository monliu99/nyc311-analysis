"""Q9 equity in city services, on the 20-file sample (fixture `cleaned_cfg`) plus small hand-built tables."""
import math
import re

import numpy as np
import pandas as pd
import pytest

from nyc311 import q09_equity as q9
from nyc311.clean import clean_glob

SET = ("not is_test and not dup_key and not coalesce(is_info_request, false) "
       "and coalesce(category, '') <> 'Invalid entries'")


@pytest.fixture(scope="module")
def res(cleaned_cfg):
    return q9.run(cleaned_cfg)


@pytest.fixture(scope="module")
def con_src(cleaned_cfg):
    return cleaned_cfg.connect(), clean_glob(cleaned_cfg)


def _one(con_src, sql):
    con, src = con_src
    return con.sql(sql.format(src=src)).fetchone()[0]


# ---------- pure functions ----------

def test_population_is_linear_between_censuses_and_weighted_by_days():
    demo = pd.DataFrame({"boro_cd": [101, 102], "population_2010": [100, 1000], "population_2020": [200, 1000]})
    p = q9.interpolated_population(demo, {2015: 10}).set_index("boro_cd")
    assert p.loc[101, "avg_pop"] == pytest.approx(150)
    assert p.loc[102, "avg_pop"] == pytest.approx(1000)
    p = q9.interpolated_population(demo, {2010: 100, 2020: 100}).set_index("boro_cd")
    assert p.loc[101, "avg_pop"] == pytest.approx(150)
    p = q9.interpolated_population(demo, {2010: 300, 2020: 100}).set_index("boro_cd")
    assert p.loc[101, "avg_pop"] == pytest.approx(125)          # weighted toward the year with more days
    p = q9.interpolated_population(demo, {2021: 365}).set_index("boro_cd")
    assert p.loc[101, "avg_pop"] == pytest.approx(210)          # 2021 extrapolated on the same line
    assert p.loc[101, "resident_years"] == pytest.approx(210 * 365 / q9.DAYS_PER_YEAR)


def test_location_quotients_average_to_one_over_districts():
    counts = pd.DataFrame({"boro_cd": [1, 1, 2, 2, 3], "key": ["a", "b", "a", "b", "a"], "n": [10, 30, 40, 10, 10]})
    lq = q9.location_quotients(counts)
    a1 = lq.set_index(["boro_cd", "key"]).loc[(1, "a")]
    assert a1.share_pct == pytest.approx(25)
    assert a1.city_share_pct == pytest.approx(60)
    assert a1.lq == pytest.approx(25 / 60)
    for _, g in lq.groupby("key"):
        assert (g.n_district * g.lq).sum() / lq.drop_duplicates("boro_cd").n_district.sum() == pytest.approx(1)
    assert (lq.lq >= 0).all()


def test_concentration_names_only_districts_with_enough_complaints():
    counts = pd.DataFrame({"boro_cd": [1, 1, 2, 2, 3, 3], "key": ["a", "b"] * 3, "n": [5, 95, 300, 100, 100, 400]})
    c = q9.concentration(q9.location_quotients(counts), min_cell=10).set_index("key")
    assert 0 <= c.loc["a", "hoover_pct"] <= 100
    assert c.loc["a", "top_boro_cd"] == 2          # LQ of a in district 1 is lower; district 2 leads
    small = q9.concentration(q9.location_quotients(counts), min_cell=1000)
    assert small.top_boro_cd.isna().all()


def test_spearman_matches_pandas_and_reports_n():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)
    y[3] = np.nan
    s = q9.spearman(x, y)
    ok = ~np.isnan(y)
    assert s["n"] == ok.sum()
    assert s["rho"] == pytest.approx(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))
    assert s["ci_low"] < s["rho"] < s["ci_high"]
    assert 0 <= s["p"] < 0.05
    assert math.isnan(q9.spearman([1, 2], [2, 1])["rho"])   # too few points: no estimate, no error


def test_puma_pooling_keeps_shared_income_areas_together():
    d = pd.DataFrame({"boro_cd": [101, 102, 103], "puma_2010": ["03810", "03810", "03809"],
                      "median_hh_income": [118108, 118108, 42873], "n": [100, 300, 200],
                      "n_unresolved": [10, 30, 5], "expected_mix": [5.0, 15.0, 10.0],
                      "n_closed": [50, 150, 100], "adj_pct": [10.0, -10.0, 0.0],
                      "avg_pop": [1000.0, 3000.0, 2000.0], "resident_years": [10.0, 30.0, 20.0]})
    p = q9.puma_units(d).set_index("puma_2010")
    assert len(p) == 2
    assert p.loc["03810", "n"] == 400 and p.loc["03810", "n_unresolved"] == 40
    assert p.loc["03810", "excess_pts_mix"] == pytest.approx(100 * (40 - 20) / 400)
    lr = (50 * np.log(1.1) + 150 * np.log(0.9)) / 200
    assert p.loc["03810", "adj_pct"] == pytest.approx(100 * np.expm1(lr))


# ---------- the sample ----------

def test_exclusions_reconcile_with_the_clean_table(res, con_src):
    ex = res["exclusions"]
    assert ex.rows.iloc[0] == _one(con_src, "select count(*) from {src} where not is_test and not dup_key")
    assert ex.rows.iloc[-1] == ex.rows.iloc[0] - ex.rows.iloc[1:-1].sum()
    assert ex.set_index("step").loc["information requests (is_info_request)", "rows"] == _one(
        con_src, "select count(*) from {src} where not is_test and not dup_key and is_info_request")
    assert ex.rows.iloc[-1] == res["districts"].n.sum()


def test_districts_are_the_standard_59_and_counts_reconcile(res, con_src):
    d = res["districts"]
    demo = q9.load_demographics()
    assert set(d.boro_cd) <= set(demo.boro_cd) and len(d) <= 59
    assert d.boro_cd.is_unique
    assert d.n_unresolved.sum() == _one(
        con_src, f"select count(*) from {{src}} where {SET} and unresolved and boro_cd is not null "
                 "and created::date in (select distinct created::date from {src} where not out_of_file_order)")
    for c in ["pct_unresolved", "expected_pct_year", "expected_pct_mix"]:
        assert d[c].between(0, 100).all(), c
    assert np.allclose(d.pct_unresolved, 100 * d.n_unresolved / d.n)
    assert np.allclose(d.unresolved_per_1k, 1000 * d.n_unresolved / d.avg_pop)
    assert (d.n_unresolved_final <= d.n_unresolved).all()
    assert d.adj_pct.dropna().gt(-100).all()


def test_indirect_standardization_expected_equals_observed_citywide(res):
    u = res["unresolved_all_groups"]   # every group, including rows with no district
    assert u.expected_year.sum() == pytest.approx(u.n_unresolved.sum())
    assert u.expected_mix.sum() == pytest.approx(u.n_unresolved.sum())


def test_location_quotients_on_the_sample(res):
    lq = res["lq_categories"]
    tot = lq.drop_duplicates("boro_cd").n_district.sum()
    for _, g in lq.groupby("key"):
        assert (g.n_district * g.lq).sum() / tot == pytest.approx(1)
    conc = res["concentration_categories"]
    assert conc.hoover_pct.between(0, 100).all()
    assert (conc.city_share_pct >= q9.MAIN_CATEGORY_SHARE).all()


def test_origin_shares_add_up(res):
    o = res["origin"]
    cols = [f"pct_{k}" for k in q9.ORIGINS]
    assert np.allclose(o[cols].sum(axis=1), 100)
    assert set(o.dominant_origin) <= set(q9.ORIGINS)


def test_correlations_report_n_and_valid_values(res):
    c = res["correlations"]
    assert c.rho.dropna().between(-1, 1).all()
    dist = c[c.unit == "district"]
    assert (dist.n <= len(res["districts"])).all()
    puma = c[c.unit == "PUMA"]
    assert (puma.n <= res["districts"].puma_2010.nunique()).all()
    assert len(res["fifths"]) == q9.INCOME_GROUPS


def test_outputs_written_and_numbers_computed(res, cleaned_cfg):
    out = cleaned_cfg.out_dir
    assert (out / "figs" / "q09_equity.png").stat().st_size > 10_000
    assert len(list((out / "figs").glob("q09_*.png"))) == 1
    assert (out / "agg" / "q09_districts.csv").is_file()
    md = (out / "answers" / "q09.md").read_text()
    assert md.startswith("# Q9")
    assert "nan" not in md.lower().replace("nanc", "")
    assert "![" in md and "figs/q09_equity.png" in md
    top = res["districts"].sort_values("n_unresolved", ascending=False).iloc[0]
    assert f"{int(top.n_unresolved):,}" in md
    rep = (out / "reports" / "q09_equity.txt").read_text().splitlines()
    assert rep[0] == "===== BEGIN q09_equity =====" and rep[-1] == "===== END q09_equity ====="
    for line in md.splitlines():   # markdown subset only: headings, bullets, lists, images, italics, tables
        assert re.match(r"^(#{1,3} |- |  - |\d+\. |!\[|\*[^ ]|\||$)", line), line
