"""Q4 response times, on the 20-file sample (fixture `cleaned_cfg`) plus small hand-built tables."""
import math
import re

import duckdb
import numpy as np
import pandas as pd
import pytest

from nyc311 import q04_response_times as q4
from nyc311 import rules
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


BASE = rules.COMPLAINTS_SQL
INSTANT = rules.INSTANT_SQL


# ---------- a) summary statistics ----------

def test_overall_stats_count_every_complaint_duration_and_quantiles_are_ordered(res, con_src):
    s = res["summary"].set_index("set")
    n_dur = _one(con_src, f"select count(duration_h) from {{src}} where {BASE}")
    assert s.loc["all closed complaints", "n"] == n_dur
    n_inst = _one(con_src, f"select count(*) from {{src}} where {BASE} and duration_h is not null and {INSTANT}")
    assert s.loc["excluding instant closures", "n"] == n_dur - n_inst == res["n_response_set"]
    assert res["n_instant"] == n_inst
    for _, r in s.iterrows():
        assert list(r[QS]) == sorted(r[QS])
        assert r.sd_h > 0 and r.mean_h > 0
        assert math.isclose(r.median_d, r.median_h / 24) and math.isclose(r.mean_d, r.mean_h / 24)
    assert s.loc["excluding instant closures", "median_h"] >= s.loc["all closed complaints", "median_h"]


def test_one_instant_closure_rule_and_one_median_excluding_them(res, con_src):
    """Every 'excluding instant closures' figure uses rules.INSTANT_SQL (time-of-day closures at the second of
    opening); date-only same-day closures stay in. No set drops every closed_at_creation row."""
    s = res["summary"].set_index("set")
    assert set(s.index) == {"all closed complaints", "excluding instant closures", "exact timestamps, excluding instant"}
    med = _one(con_src, f"select quantile_cont(duration_h, 0.5) from {{src}} where {BASE} and duration_h is not null "
                        f"and not ({INSTANT})")
    assert math.isclose(s.loc["excluding instant closures", "median_h"], med)
    o = res["outliers"].set_index("kind")
    assert o.loc["  0 hours, exact second with a time of day", "n"] == res["n_instant"]
    assert o.loc["  0 hours, dates only (closed the same day)", "n"] == _one(
        con_src, f"select count(*) from {{src}} where {BASE} and duration_h is not null and closed_at_creation "
                 f"and date_only_created")
    y = res["by_year"]
    assert "median_h_excl_instant" in y.columns and "median_h_excl_zero" not in y.columns
    a = res["by_agency"].set_index("agency")
    hpd_x = _one(con_src, f"select quantile_cont(duration_h, 0.5) / 24 from {{src}} where {BASE} and agency = 'HPD' "
                          f"and duration_h is not null and not ({INSTANT})")
    assert math.isclose(a.loc["HPD", "median_d_excl_instant"], hpd_x)
    assert q4.INSTANT == rules.INSTANT_SQL and q4.COMPLAINTS == rules.COMPLAINTS_SQL


def test_summary_quantiles_match_single_quantile_queries(res, con_src):
    s = res["summary"].set_index("set").loc["all closed complaints"]
    for col, q in q4.QUANTILES.items():
        v = _one(con_src, f"select quantile_cont(duration_h, {q}) from {{src}} where {BASE} and duration_h is not null")
        assert math.isclose(s[col], v, rel_tol=1e-12), col


@pytest.mark.parametrize("h, text", [(0, "0 seconds"), (40 / 3600, "40 seconds"), (1 / 3600, "1 second"),
                                     (0.5, "30 minutes"), (1 / 60, "1 minute"), (1.0, "1.0 hours"),
                                     (30.3, "30.3 hours"), (72, "3.0 days"), (float("nan"), "n/a")])
def test_fmt_h_never_rounds_short_durations_to_zero_hours(h, text):
    assert q4.fmt_h(h) == text


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
    for c in ["pct_unresolved", "pct_instant", "pct_day_precision", "share_pct"]:
        assert a[c].between(0, 100).all()
    assert (a.median_d <= a.p90_d).all()
    assert a.share_pct.sum() <= 100 + 1e-9
    total = _one(con_src, f"select count(*) from {{src}} where {BASE} and duration_h is not null and not ({INSTANT})")
    assert np.allclose(a.share_pct, 100 * a.n_excl_instant / total)
    # HPD records dates only in 2010, so nearly all its durations are whole days
    assert a.set_index("agency").loc["HPD", "pct_day_precision"] > 90


def test_agency_chart_and_text_compare_the_same_agencies_with_meaningful_volume(res, cleaned_cfg):
    a, shown = res["by_agency"], res["agency_chart"]
    keep = a[(a.share_pct >= q4.MIN_AGENCY_SHARE) & (a.n >= q4.MIN_N)]
    assert set(shown.agency) == set(keep.agency) and len(shown) >= 2
    small = a[~a.agency.isin(shown.agency)]
    assert (small.share_pct < q4.MIN_AGENCY_SHARE).all()
    text = (cleaned_cfg.out_dir / "answers" / "q04.md").read_text()
    order = shown.sort_values("median_d_excl_instant")
    fast, slow = order.iloc[0], order.iloc[-1]
    raw_gap = next(line for line in text.splitlines() if line.startswith("- **Raw gap.**"))
    assert f"({fast.agency})" in raw_gap and f"({slow.agency})" in raw_gap
    assert q4.fmt_h(24 * fast.median_d_excl_instant) in raw_gap and q4.fmt_h(24 * slow.median_d_excl_instant) in raw_gap
    for r in small.itertuples():  # smaller agencies are never named as the chart's extremes
        assert f"({r.agency})" not in raw_gap
    assert "times longer" not in text
    assert not re.search(r"\b0\.0 hours\b", text)


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


def test_gap_drivers_decompose_a_groups_gap_into_complaint_types():
    con = duckdb.connect()
    # A is slow on type t (40 vs 10 h for B) and typical on u; B is fast on t
    rows = ([("A", "t", "2011-03-01", 40)] * 4 + [("B", "t", "2011-03-01", 10)] * 4 +
            [("A", "u", "2011-03-01", 5)] * 2 + [("B", "u", "2011-03-01", 5)] * 2)
    rel = _toy(con, rows)
    adj = q4.mix_adjusted(con, rel, "grp", min_n=1).set_index("grp").adj_pct
    dr = q4.gap_drivers(con, rel, "grp", ["A", "B"])
    for g, part in dr.groupby("grp"):
        total_log = np.log1p(adj[g] / 100)
        assert math.isclose(part.contrib_log.sum(), total_log, abs_tol=1e-12)
        assert np.allclose(part.group_adj_pct, adj[g])
        assert math.isclose(part.pct_of_closures.sum(), 100)
    a = dr[dr.grp == "A"].set_index("canonical_type")
    assert a.loc["t", "share_of_gap_pct"] == pytest.approx(100) and a.loc["u", "share_of_gap_pct"] == pytest.approx(0)
    assert a.loc["t", "adj_pct_without"] == pytest.approx(0, abs=1e-9)  # without t, A is typical
    assert a.loc["t", "type_adj_pct"] > 0 and a.loc["t", "city_type_median_h"] == pytest.approx(25)
    assert a.loc["t", "n"] == 4 and a.loc["t", "median_h"] == 40


def test_drivers_cover_the_most_extreme_districts(res, cleaned_cfg):
    d, dr = res["districts"], res["district_drivers"]
    k = min(q4.DRIVER_K, len(d) // 2)
    order = d.sort_values("adj_pct", ascending=False)
    want = set(order.head(k).boro_cd) | set(order.tail(k).boro_cd)
    assert set(dr.boro_cd) == want
    adj = d.set_index("boro_cd").adj_pct
    assert np.allclose(dr.group_adj_pct, dr.boro_cd.map(adj))
    assert (dr.groupby("boro_cd").size() <= q4.DRIVER_TYPES).all()
    assert set(dr.side) == {"slowest", "fastest"}
    # listed types push the district's gap in its own direction, biggest first
    for cd, part in dr.groupby("boro_cd"):
        sign = 1 if adj[cd] > 0 else -1
        assert list(sign * part.contrib_log) == sorted(sign * part.contrib_log, reverse=True)
    assert (cleaned_cfg.out_dir / "agg" / "q04_district_drivers.csv").exists()
    text = (cleaned_cfg.out_dir / "answers" / "q04.md").read_text()
    line = next(x for x in text.splitlines() if x.startswith("- **What drives the extremes.**"))
    top = order.iloc[0]
    assert top.district in line
    lead = dr[(dr.boro_cd == top.boro_cd) & ~dr.low_n].iloc[0]
    assert lead.canonical_type in line and f"{lead.share_of_gap_pct:.0f}%" in line


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
    assert w.pct_desc_shared.between(0, 100).all()
    like = w.groupby("canonical_type").pct_desc_shared.min() >= q4.LIKE_FOR_LIKE_PCT
    assert (w.like_for_like == w.canonical_type.map(like)).all()
    # the taxonomy keeps plated (NYPD) and unplated (DSNY) vehicles apart: no merged type shared by both
    assert not any({"NYPD", "DSNY"} <= set(g.agency) for t, g in w.groupby("canonical_type") if "Vehicle" in t)


def _toy_desc(con, rows):
    df = pd.DataFrame(rows, columns=["canonical_type", "agency", "descriptor"])
    con.register("toy_desc_df", df)
    con.execute("create or replace temp table toy_desc as select * from toy_desc_df")
    return "toy_desc"


def test_like_for_like_needs_each_agency_to_log_descriptors_the_other_also_uses():
    con = duckdb.connect()
    n = q4.MIN_N_SHARED
    rows = ([("same", "A", "x")] * n + [("same", "B", "x")] * (n - 2) + [("same", "B", "y")] * 2 +
            [("split", "A", "p")] * n + [("split", "B", "q")] * n + [("split", "B", "p")] * 1 +
            [("blank", "A", "x")] * n + [("blank", "B", None)] * n +
            [("alone", "A", "x")] * n + [("alone", "B", "x")] * (n - 1))
    o = q4.descriptor_overlap(con, _toy_desc(con, rows)).set_index(["canonical_type", "agency"])
    assert set(o.index.get_level_values(0)) == {"same", "split", "blank"}  # "alone": B has too few closures
    assert o.loc[("same", "A"), "pct_desc_shared"] == pytest.approx(100)
    assert o.loc[("same", "B"), "pct_desc_shared"] == pytest.approx(100 * (n - 2) / n)
    assert o.loc[("split", "A"), "pct_desc_shared"] == 0  # B's single 'p' is not a regular descriptor of B's
    assert o.loc[("blank", "B"), "pct_desc_shared"] == 0  # a missing descriptor never counts as a match
    like = o.groupby(level=0).like_for_like.first()
    assert like["same"] and not like["split"] and not like["blank"]
    assert o.loc[("split", "B"), "top_descriptor"] == "q"


def test_agency_benchmarks_use_like_for_like_types_only(res, cleaned_cfg):
    w, e = res["within_type"], res["agency_effects"]
    like_types = set(w.canonical_type[w.like_for_like])
    for r in e.itertuples():
        assert set(r.shared_types.split(", ")) <= like_types
    text = (cleaned_cfg.out_dir / "answers" / "q04.md").read_text()
    if not like_types:
        assert e.empty and "- **No like-for-like agency comparison.**" in text
        assert "- **Direct comparison, same complaint type:**" not in text
        for i, (t, g) in enumerate(w.groupby("canonical_type", sort=False)):
            assert t in text
            if i < 4:  # the first four types are listed with each agency's commonest descriptor
                assert all(f"{a} {d!r}" in text for a, d in zip(g.agency, g.top_descriptor) if isinstance(d, str))


# ---------- outputs ----------

def test_outputs_are_written(res, cleaned_cfg):
    out = cleaned_cfg.out_dir
    for name in ["q04_district_adjusted.png", "q04_agency_variance.png"]:
        assert (out / "figs" / name).stat().st_size > 10_000
    for name in ["summary", "outliers", "by_agency", "by_year", "districts", "boroughs", "variance",
                 "agency_effects", "within_type", "district_drivers"]:
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


def test_findings_open_with_a_neutral_provenance_line(res, cleaned_cfg):
    lines = (cleaned_cfg.out_dir / "answers" / "q04.md").read_text().splitlines()
    first = lines[2]
    assert first.startswith("*Computed by `nyc311/q04_response_times.py` from ") and first.endswith(".*")
    assert f"{res['excl'].n_complaints:,} complaints" in first and res["period_long"] in first
    assert "prose" not in first.lower() and "write" not in first.lower()


def test_findings_use_the_report_markdown_subset_and_embed_both_figures(res, cleaned_cfg):
    text = (cleaned_cfg.out_dir / "answers" / "q04.md").read_text()
    assert text.splitlines()[0].startswith("# Q4")
    for name in ["q04_district_adjusted.png", "q04_agency_variance.png"]:
        assert re.search(r"!\[[^\]]+\]\(figs/" + re.escape(name) + r"\)", text), name
        assert (cleaned_cfg.out_dir / "figs" / name).exists()
    assert not re.search(r"<[a-zA-Z/][^>]*>", text)  # no raw HTML
    assert not re.search(r"(^|\s)_[^_\s][^_]*_(\s|$|[.,;])", text)  # italics with *, not _

