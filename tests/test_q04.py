"""Q4 response times, on the 20-file sample (fixture `cleaned_cfg`) plus small hand-built tables."""
import math
import re

import duckdb
import numpy as np
import pandas as pd
import pytest

from nyc311 import q04_response_times as q4
from nyc311.clean import clean_glob

QS = ["p10_h", "p25_h", "median_h", "p75_h", "p90_h", "p99_h"]


@pytest.fixture(scope="module")
def res(cleaned_cfg):
    return q4.run(cleaned_cfg)


@pytest.fixture(scope="module")
def con_src(cleaned_cfg):
    return cleaned_cfg.connect(), clean_glob(cleaned_cfg)


def _one(con_src, sql):
    con, src = con_src
    return con.sql(sql.format(src=src)).fetchone()[0]


BASE = "not is_test and not dup_key and not coalesce(is_info_request, false)"


# ---------- a) summary statistics ----------

def test_overall_stats_count_every_complaint_duration_and_quantiles_are_ordered(res, con_src):
    s = res["summary"].set_index("set")
    n_dur = _one(con_src, f"select count(duration_h) from {{src}} where {BASE}")
    assert s.loc["all closed complaints", "n"] == n_dur
    n_zero = _one(con_src, f"select count(*) from {{src}} where {BASE} and duration_h is not null and closed_at_creation")
    assert s.loc["excluding closed at creation", "n"] == n_dur - n_zero
    for _, r in s.iterrows():
        assert list(r[QS]) == sorted(r[QS])
        assert r.sd_h > 0 and r.mean_h > 0
        assert math.isclose(r.median_d, r.median_h / 24) and math.isclose(r.mean_d, r.mean_h / 24)
    assert s.loc["excluding closed at creation", "median_h"] >= s.loc["all closed complaints", "median_h"]


def test_mean_far_above_median_because_durations_are_skewed(res):
    s = res["summary"].set_index("set").loc["all closed complaints"]
    assert s.mean_h > 2 * s.median_h


def test_outliers_reconcile_with_flags_and_fences_bracket_the_middle(res, con_src):
    o = res["outliers"].set_index("kind")
    n = res["summary"].set_index("set").loc["all closed complaints", "n"]
    assert o.loc["closed at creation (0 hours)", "n"] == _one(
        con_src, f"select count(*) from {{src}} where {BASE} and duration_h = 0")
    assert o.loc["closed more than a year after opening", "n"] == _one(
        con_src, f"select count(*) from {{src}} where {BASE} and closed_after_1yr")
    assert ((o.n >= 0) & (o.n <= n)).all()
    assert np.allclose(o.pct, 100 * o.n / n)
    f = res["fences"]
    s = res["summary"].set_index("set").loc["all closed complaints"]
    assert 0 < f["lower_h"] < s.median_h < f["upper_h"]
    assert o.loc["above citywide Tukey fence (log scale)", "n"] == _one(
        con_src, f"select count(*) from {{src}} where {BASE} and duration_h > {f['upper_h']}")


def test_within_type_outliers_use_each_types_own_fences(res):
    t = res["type_outliers"]
    assert (t.n >= q4.MIN_N).all()
    assert (t.upper_h > t.lower_h).all()
    assert ((t.pct_outside >= 0) & (t.pct_outside <= 100)).all()


def test_missing_durations_are_explained_and_add_up(res, con_src):
    r = res["no_duration"]
    total = _one(con_src, f"select count(*) from {{src}} where {BASE}")
    assert r.n.sum() == total
    assert set(r.reason) >= {"closed, with a duration", "still open (unresolved)"}
    assert r.set_index("reason").loc["still open (unresolved)", "n"] == _one(
        con_src, f"select count(*) from {{src}} where {BASE} and unresolved")
    assert math.isclose(r.pct.sum(), 100)


def test_by_agency_covers_large_agencies_with_unresolved_shares_as_percentages(res, con_src):
    a = res["by_agency"]
    big = con_src[0].sql(f"select agency from {con_src[1]} where {BASE} group by 1 "
                         f"having count(duration_h) >= {q4.MIN_N}").df().agency
    assert set(a.agency) == set(big)
    for c in ["pct_unresolved", "pct_zero_hours", "pct_day_precision"]:
        assert a[c].between(0, 100).all()
    assert (a.median_d <= a.p90_d).all()
    # HPD records dates only in 2010, so nearly all its durations are whole days
    assert a.set_index("agency").loc["HPD", "pct_day_precision"] > 90


def test_by_year_counts_add_up(res, con_src):
    y = res["by_year"]
    assert y.n_complaints.sum() == _one(con_src, f"select count(*) from {{src}} where {BASE}")
    assert y.year.is_unique and y.year.is_monotonic_increasing
    assert y.pct_unresolved.between(0, 100).all()


# ---------- b) neighborhoods ----------

def _toy(con, rows):
    df = pd.DataFrame(rows, columns=["grp", "canonical_type", "created", "duration_h"])
    df["created"] = pd.to_datetime(df.created)
    con.register("toy_df", df)
    con.execute("create or replace temp table toy as select * from toy_df")
    return "toy"


def test_mix_adjustment_removes_differences_due_only_to_complaint_type():
    con = duckdb.connect()
    # A gets only the slow type, B only the fast type; each is exactly typical for its type
    rows = ([("A", "slow", "2010-01-05", h) for h in (90, 100, 110)] +
            [("B", "fast", "2010-01-05", h) for h in (1, 2, 3)] +
            [("C", "slow", "2010-01-06", h) for h in (90, 100, 110)] +
            [("C", "fast", "2010-01-06", h) for h in (1, 2, 3)])
    out = q4.mix_adjusted(con, _toy(con, rows), "grp", min_n=1).set_index("grp")
    assert out.loc["A", "median_h"] > 10 * out.loc["B", "median_h"]
    assert np.allclose(out.adj_pct, 0, atol=1e-9)


def test_mix_adjustment_flags_a_group_that_is_slow_for_the_same_type():
    con = duckdb.connect()
    rows = ([("A", "t", "2011-03-01", h) for h in (10, 10, 10, 10)] +
            [("B", "t", "2011-03-01", h) for h in (40, 40, 40, 40)] +
            [("A", "u", "2012-03-01", h) for h in (5, 5)])
    out = q4.mix_adjusted(con, _toy(con, rows), "grp", min_n=1).set_index("grp")
    assert out.loc["B", "adj_pct"] > 0 > out.loc["A", "adj_pct"]
    # a cell (type x year) handled by one group only is its own benchmark
    out_u = q4.mix_adjusted(con, "(select * from toy where canonical_type = 'u')", "grp", min_n=1)
    assert np.allclose(out_u.adj_pct, 0, atol=1e-9)


def test_mix_adjustment_drops_small_groups():
    con = duckdb.connect()
    rows = [("A", "t", "2011-03-01", 1)] * 5 + [("B", "t", "2011-03-01", 2)] * 2
    out = q4.mix_adjusted(con, _toy(con, rows), "grp", min_n=3)
    assert out.grp.tolist() == ["A"]


def test_districts_are_standard_districts_with_both_rankings(res):
    d = res["districts"]
    assert d.boro_cd.between(101, 595).all()
    assert (d.boro_cd % 100).between(1, 18).all()
    assert (d.n >= q4.MIN_N).all() and d.boro_cd.is_unique
    k = len(d)
    assert sorted(d.rank_raw) == list(range(1, k + 1)) and sorted(d.rank_adjusted) == list(range(1, k + 1))
    slowest_raw = d.sort_values("rank_raw").iloc[0]
    assert slowest_raw.median_h == d.median_h.max()
    assert d.sort_values("rank_adjusted").iloc[0].adj_pct == d.adj_pct.max()
    assert d.district.str.len().gt(0).all()


def test_district_rankings_differ_after_adjustment_and_disagreements_are_listed(res):
    d, cmp = res["districts"], res["rank_compare"]
    assert -1 <= cmp["spearman"] <= 1
    k = min(q4.TOP_K, len(d))
    raw_top = set(d.nsmallest(k, "rank_raw").boro_cd)
    adj_top = set(d.nsmallest(k, "rank_adjusted").boro_cd)
    assert set(cmp["only_raw"]) == raw_top - adj_top
    assert set(cmp["only_adjusted"]) == adj_top - raw_top


def test_borough_summary_has_the_five_boroughs(res):
    b = res["boroughs"]
    assert set(b.borough) == {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}


# ---------- c) agencies ----------

def test_variance_explained_grows_as_terms_are_added(res):
    v = res["variance"].set_index("model")
    assert v.r2.between(0, 1).all()
    ladder = v.loc[q4.NESTED_MODELS, "r2"].tolist()
    assert ladder == sorted(ladder)
    assert v.n.nunique() == 1
    assert v.n.iloc[0] == min(q4.SAMPLE_ROWS, res["n_response_set"])


def test_agency_effects_only_where_another_agency_handles_the_same_type(res):
    e = res["agency_effects"]
    assert (e.n_shared >= q4.MIN_N_SHARED).all()
    assert (e.pct_vs_typical > -100).all()
    assert (e.ci_low <= e.pct_vs_typical).all() and (e.pct_vs_typical <= e.ci_high).all()
    cov = res["agency_coverage"]
    assert cov.pct_volume_in_shared_types.between(0, 100).all()
    assert set(e.agency) <= set(cov.agency[cov.pct_volume_in_shared_types > 0])


def test_within_type_comparison_lists_types_with_two_or_more_agencies(res):
    w = res["within_type"]
    assert len(w) > 0
    assert (w.n >= q4.MIN_N_SHARED).all()
    assert (w.groupby("canonical_type").agency.nunique() >= 2).all()
    assert (w.ratio_to_type_median > 0).all()


# ---------- outputs ----------

def test_outputs_are_written(res, cleaned_cfg):
    out = cleaned_cfg.out_dir
    for name in ["q04_district_adjusted.png", "q04_agency_variance.png"]:
        assert (out / "figs" / name).stat().st_size > 10_000
    for name in ["summary", "outliers", "by_agency", "by_year", "districts", "boroughs", "variance",
                 "agency_effects", "within_type"]:
        assert (out / "agg" / f"q04_{name}.csv").exists(), name
    block = (out / "reports" / "q04_response_times.txt").read_text()
    assert block.startswith("===== BEGIN q04_response_times =====")
    assert len(block.splitlines()) <= 152


def test_findings_quote_the_computed_numbers(res, cleaned_cfg):
    text = (cleaned_cfg.out_dir / "answers" / "q04.md").read_text()
    s = res["summary"].set_index("set").loc["all closed complaints"]
    assert f"{int(s.n):,}" in text
    assert q4.fmt_h(s.median_h) in text
    d = res["districts"].sort_values("rank_adjusted").iloc[0]
    assert d.district in text
    assert not re.search(r"\bnan\b|\bNone\b|\binf\b", text, flags=re.IGNORECASE)


def test_findings_use_the_report_markdown_subset_and_embed_both_figures(res, cleaned_cfg):
    text = (cleaned_cfg.out_dir / "answers" / "q04.md").read_text()
    assert text.splitlines()[0].startswith("# Q4")
    for name in ["q04_district_adjusted.png", "q04_agency_variance.png"]:
        assert re.search(r"!\[[^\]]+\]\(figs/" + re.escape(name) + r"\)", text), name
        assert (cleaned_cfg.out_dir / "figs" / name).exists()
    assert not re.search(r"<[a-zA-Z/][^>]*>", text)  # no raw HTML
    assert not re.search(r"(^|\s)_[^_\s][^_]*_(\s|$|[.,;])", text)  # italics with *, not _

