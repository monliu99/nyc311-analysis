"""Q9 equity in city services, on the 20-file sample (fixture `cleaned_cfg`), small hand-built tables, and a
synthetic pull that spans the 2015 population step."""
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyc311 import q07_geography as q7
from nyc311 import q09_equity as q9
from nyc311 import rules
from nyc311.clean import clean_glob

REPO = Path(__file__).resolve().parent.parent
DAYS = """(select created::date from {src} where not is_test and not dup_key and created is not null group by 1
           having bool_or(not out_of_file_order) and not bool_or(possibly_incomplete_day))"""


@pytest.fixture(scope="module")
def res(cleaned_cfg):
    return q9.run(cleaned_cfg)


@pytest.fixture(scope="module")
def con_src(cleaned_cfg):
    return cleaned_cfg.connect(), clean_glob(cleaned_cfg)


def _one(con_src, sql):
    con, src = con_src
    return con.sql(sql.format(src=src, days=DAYS.format(src=src))).fetchone()[0]


def _png_size(path):
    head = Path(path).read_bytes()[:24]
    return struct.unpack(">II", head[16:24])


# ---------- pure functions ----------

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


def _districts():
    return pd.DataFrame({"boro_cd": [104, 105, 103], "borough": "Manhattan", "puma_2010": ["03807", "03807", "03809"],
                         "median_hh_income": [92246, 92246, 42873], "n": [100, 300, 200],
                         "n_unresolved": [10, 30, 5], "expected_mix": [5.0, 15.0, 10.0],
                         "n_closed": [50, 150, 100], "adj_pct": [10.0, -10.0, 0.0],
                         "person_years": [10.0, 1.0, 20.0], "commuter": [False, True, False],
                         "n_city": [10, 30, 20], "n_private": [80, 250, 170], "n_natural": [5, 10, 5],
                         "n_other": [5, 10, 5]})


def test_puma_pooling_keeps_shared_income_areas_together():
    p = q9.puma_units(_districts()).set_index("puma_2010")
    assert len(p) == 2
    assert p.loc["03807", "n"] == 400 and p.loc["03807", "n_unresolved"] == 40
    assert p.loc["03807", "excess_pts_mix"] == pytest.approx(100 * (40 - 20) / 400)
    lr = (50 * np.log(1.1) + 150 * np.log(0.9)) / 200
    assert p.loc["03807", "adj_pct"] == pytest.approx(100 * np.expm1(lr))
    # per resident a year: the commuter district's complaints and residents are left out of the pooled rate
    assert p.loc["03807", "unresolved_per_1k_yr"] == pytest.approx(1000 * 10 / 10.0)
    assert p.loc["03809", "unresolved_per_1k_yr"] == pytest.approx(1000 * 5 / 20.0)
    assert p.loc["03807", "borough"] == "Manhattan"


def test_income_fifths_pool_origin_shares_and_rates():
    rng = np.random.default_rng(1)
    k = 10
    p = pd.DataFrame({"puma_2010": [f"p{i}" for i in range(k)], "median_hh_income": np.linspace(20e3, 120e3, k),
                      "n_districts": 1, "n": 1000, "n_unresolved": rng.integers(10, 50, k),
                      "expected_mix": 20.0, "n_closed": 500, "adj_pct": rng.normal(0, 5, k),
                      "n_res": 1000, "n_res_unresolved": rng.integers(10, 50, k), "person_years_res": 100.0,
                      "n_city": np.linspace(100, 300, k).astype(int)})
    p["n_private"] = 1000 - p.n_city - 20
    p["n_natural"], p["n_other"] = 10, 10
    f = q9.income_fifths(p)
    assert len(f) == 5 and f.pumas.sum() == k
    assert np.allclose(f[[f"pct_{o}" for o in q9.ORIGINS]].sum(axis=1), 100)
    assert f.pct_city.is_monotonic_increasing
    g = p.iloc[:2]
    assert f.unresolved_per_1k_yr.iloc[0] == pytest.approx(1000 * g.n_res_unresolved.sum() / 200.0)


def _s(rho, p, n=55):
    return {"rho": rho, "p": p, "n": n, "ci_low": rho - 0.25, "ci_high": rho + 0.25}


def test_verdict_grades_strength_significance_and_robustness():
    assert q9.verdict(_s(-0.19, 0.17))["strength"] == "no clear"
    v = q9.verdict(_s(-0.28, 0.03), within=(-4.3, 2.7))
    assert v["strength"] == "weak" and v["sig"] and v["fragile"] and v["sign"] < 0
    v = q9.verdict(_s(0.55, 0.0001), within=(1.0, 3.0))
    assert v["strength"] == "strong" and not v["fragile"] and v["sign"] > 0
    assert q9.verdict(_s(float("nan"), float("nan")))["strength"] == "no clear"


def test_title_follows_the_evidence_not_an_assertion():
    none = q9.equity_title(_s(-0.19, 0.17), _s(-0.05, 0.7), None, None)
    assert "no clear" in none.lower() and "take longer" not in none
    weak = q9.equity_title(_s(-0.19, 0.17), _s(-0.27, 0.03), None, (-0.17, 0.13))
    assert weak.lower().startswith("weak") and "within boroughs" in weak and "lower-income" in weak
    strong = q9.equity_title(_s(-0.6, 0.0001), _s(-0.1, 0.5), (-9.0, -3.0), None)
    assert strong.lower().startswith("strong") and "lower-income" in strong and "within boroughs" not in strong
    sub = q9.equity_subtitle(_s(-0.19, 0.17), _s(-0.30, 0.03))
    assert "\N{MINUS SIGN}0.19" in sub and "p 0.17" in sub and "55" in sub


# ---------- the sample ----------

def test_rows_and_totals_match_q7(res, cleaned_cfg):
    con, src = cleaned_cfg.connect(), clean_glob(cleaned_cfg)
    counts, ex7, _ = q7.district_counts(con, src)
    ex = res["exclusions"]
    assert ex.step.tolist() == ex7.step.tolist() and ex.rows.tolist() == ex7.rows.tolist()
    assert res["districts"].n.sum() == counts.n.sum() == ex.rows.iloc[-1]
    assert res["n_benchmark"] == ex.rows.iloc[0] - ex.rows.iloc[1:6].sum()   # through possibly_incomplete_day


def test_exclusions_reconcile_with_the_clean_table(res, con_src):
    ex = res["exclusions"].set_index("step")
    assert ex.rows.iloc[0] == _one(con_src, f"select count(*) from {{src}} where {rules.BASE_SQL}")
    assert ex.loc["information requests (is_info_request)", "rows"] == _one(
        con_src, "select count(*) from {src} where not is_test and not dup_key and is_info_request")
    assert ex.loc["possibly_incomplete_day", "rows"] == _one(
        con_src, f"select count(*) from {{src}} where {rules.COMPLAINTS_SQL} and possibly_incomplete_day")


def test_districts_use_complete_days_and_q7_population(res, con_src):
    d = res["districts"]
    demo = q9.load_demographics()
    assert set(d.boro_cd) <= set(demo.boro_cd) and len(d) <= 59
    assert d.boro_cd.is_unique
    assert d.n_unresolved.sum() == _one(
        con_src, f"select count(*) from {{src}} where {rules.COMPLAINTS_SQL} and unresolved and boro_cd is not null "
                 "and created::date in {days}")
    days = res["days"]
    n20 = int((days.dt.year >= q7.POP_2020_FROM_YEAR).sum())
    py = q7.person_years(demo, len(days) - n20, n20)
    assert np.allclose(d.person_years, py.reindex(d.boro_cd).to_numpy())
    assert np.allclose(d.unresolved_per_1k_yr, 1000 * d.n_unresolved / d.person_years)
    for c in ["pct_unresolved", "expected_pct_year", "expected_pct_mix"]:
        assert d[c].between(0, 100).all(), c
    assert np.allclose(d.pct_unresolved, 100 * d.n_unresolved / d.n)
    assert (d.n_unresolved_final <= d.n_unresolved).all()
    assert d.adj_pct.dropna().gt(-100).all()
    assert set(d.boro_cd[d.commuter]) <= set(q7.COMMUTER_CDS)
    by_origin = d[[f"nu_{o}" for o in q9.ORIGINS]].sum(axis=1)
    assert (by_origin == d.n_unresolved).all()


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
    assert set(conc.origin) <= set(q9.ORIGINS)


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
    per_res = dist[dist.metric == "unresolved_per_1k_yr"]
    assert (per_res.n <= len(res["districts"]) - res["districts"].commuter.sum()).all()
    puma = c[c.unit == "PUMA"]
    assert (puma.n <= res["districts"].puma_2010.nunique()).all()
    assert len(res["fifths"]) == q9.INCOME_GROUPS
    assert set(res["slopes"].unit) <= {"district", "PUMA"}


def test_outputs_written_and_numbers_computed(res, cleaned_cfg):
    out = cleaned_cfg.out_dir
    png = out / "figs" / "q09_equity.png"
    assert png.stat().st_size > 10_000
    w, h = _png_size(png)
    assert 8 * 300 <= w <= 12.5 * 300 and h <= 1.4 * w     # one figure that fits a ~1000 px report column
    assert len(list((out / "figs").glob("q09_*.png"))) == 1
    assert (out / "agg" / "q09_districts.csv").is_file()
    md = (out / "answers" / "q09.md").read_text()
    assert md.startswith("# Q9")
    lines = [x for x in md.splitlines() if x.strip()]
    n = int(res["districts"].n.sum())
    assert re.fullmatch(rf"\*Computed by `nyc311/q09_equity\.py` from {n:,} complaints, .+\.\*", lines[1])
    assert "write the prose" not in md and "Drafted" not in md
    assert "nan" not in md.lower().replace("nanc", "")
    assert "![" in md and "figs/q09_equity.png" in md
    assert f"across the {len(res['districts'])} districts" in md
    top = res["districts"].sort_values("n_unresolved", ascending=False).iloc[0]
    assert f"{int(top.n_unresolved):,}" in md
    assert res["facts"]["title"] in md
    rep = (out / "reports" / "q09_equity.txt").read_text().splitlines()
    assert rep[0] == "===== BEGIN q09_equity =====" and rep[-1] == "===== END q09_equity ====="
    for line in md.splitlines():   # markdown subset only: headings, bullets, lists, images, italics, tables
        assert re.match(r"^(#{1,3} |- |  - |\d+\. |!\[|\*[^ ]|\||$)", line), line


# ---------- several years, across the 2015 population step (synthetic pull) ----------

@pytest.fixture(scope="module")
def years(tmp_path_factory):
    sys.path.insert(0, str(REPO / "tools"))
    import make_synthetic_batches as synth
    from nyc311 import clean, ingest
    from nyc311.config import Config
    root = tmp_path_factory.mktemp("q09years")
    synth.main(["--files", "40", "--rows-per-file", "400", "--start", "2014-04-01", "--end", "2015-09-30",
                "--out", str(root / "data")])
    cfg = Config.from_args(["--data", str(root / "data" / "311_batch_*.json"), "--out", str(root / "out"),
                            "--work", str(root / "work"), "--threads", "2", "--memory-limit", "2GB"])
    ingest.run(cfg)
    clean.run(cfg)
    return cfg, q7.run(cfg), q9.run(cfg)


def test_several_years_step_population_and_matching_totals(years):
    cfg, r7, r9 = years
    days = r9["days"]
    n20 = int((days.dt.year >= 2015).sum())
    assert 0 < n20 < len(days)                                   # both Census counts are used
    d = r9["districts"].set_index("boro_cd")
    demo = q9.load_demographics().set_index("boro_cd")
    step = (demo.population_2010 * (len(days) - n20) + demo.population_2020 * n20) / q7.DAYS_PER_YEAR
    assert np.allclose(d.person_years, step.reindex(d.index))
    assert r9["districts"].n.sum() == r7["districts"].n.sum()     # same complaints as Q7
    assert r9["exclusions"].rows.tolist() == r7["exclusions"].rows.tolist()
    assert r9["exclusions"].set_index("step").loc["possibly_incomplete_day", "rows"] > 0
    assert days.max() <= r9["d_final"] + pd.Timedelta(days=q9.FINAL_DAYS)
    assert (cfg.out_dir / "figs" / "q09_equity.png").stat().st_size > 10_000
