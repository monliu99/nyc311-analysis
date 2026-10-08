"""Q2 (daily volume, closing times, week patterns): pure helpers, the 20-file sample, and a multi-year pull."""
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from nyc311 import clean, ingest, q02_timing as q2, rules
from nyc311.config import Config

REPO = Path(__file__).resolve().parent.parent
BASE = rules.BASE_SQL


# ---------------------------------------------------------------- pure helpers

def test_missing_batch_numbers_are_listed_as_ranges():
    files = ["311_batch_0001.json", "311_batch_0002.json", "311_batch_0005.json", "311_batch_0009.json"]
    assert q2.missing_batch_ranges(files) == ["3-4", "6-8"]


def _weekly_series(weeks=16, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-05", periods=7 * weeks, freq="D")  # starts on a Monday
    base = np.where(idx.dayofweek >= 5, 60.0, 100.0)
    return pd.Series(base * (1 + rng.normal(0, 0.03, len(idx))), index=idx).round()


def test_robust_score_flags_a_spike_and_a_dip_but_not_the_weekly_rhythm():
    s = _weekly_series()
    spike, dip, cut = s.index[50], s.index[71], s.index[30]
    s[spike], s[dip], s[cut] = 300, 20, 5
    excluded = pd.Series(s.index == cut, index=s.index)
    gap = s.index[85:92]  # a week with no rows at all: a gap, not seven "low" days, and not a baseline of zero
    s[gap] = 0
    out = q2.score_days(s, excluded).set_index("date")
    assert out.loc[spike, "kind"] == "high" and out.loc[dip, "kind"] == "low"
    assert out.loc[cut, "kind"] == "excluded"
    assert (out.loc[gap, "kind"] == "empty").all()
    assert out.loc[gap + pd.Timedelta(days=7), "baseline"].min() > 50
    assert set(out.drop([spike, dip, cut, *gap]).kind) <= {"", "no baseline"}
    # the baseline is the same weekday's level, so a Saturday is compared with Saturdays
    sat = out[out.index.dayofweek == 5].baseline.dropna()
    assert sat.between(50, 70).all()


def test_surge_day_excess_and_its_usual_level_use_the_same_baseline():
    """The 'usual' level printed beside a surge day is the mean of the same-weekday days whose per-type means give
    the excess shares, so the shares are of (day - usual) and can pass 100% when other types fall below usual."""
    mondays = pd.date_range("2015-01-05", periods=9, freq="7D")
    rows = []
    for i, d in enumerate(mondays):
        a, b = (40, 3) if i == 4 else (10, 5 + (i % 2))  # B alternates 5/6: mean 5.5, median differs from mean
        rows += [(d, "A")] * a + [(d, "B")] * b
    toy = pd.DataFrame(rows, columns=["created", "canonical_type"]).assign(
        complaint_type=lambda x: x.canonical_type, is_test=False, dup_key=False)
    con = duckdb.connect()
    con.register("toy", toy)
    totals = toy.groupby("created").size()
    scored = pd.DataFrame({"date": mondays, "n": totals.reindex(mondays).to_numpy(), "baseline": 99.0,
                           "z": np.where(mondays == mondays[4], 9.0, 0.0),
                           "kind": np.where(mondays == mondays[4], "high", ""), "holiday": None})
    out = q2.surge_days(con, "toy", scored)
    r = out.iloc[0]
    usual = totals.drop(mondays[4]).mean()
    assert r.n == 43 and r.baseline == pytest.approx(usual) and r.excess == pytest.approx(43 - usual)
    a_excess = 40 - 10
    assert r.drivers.startswith(f"A +{a_excess} ({100 * a_excess / (43 - usual):.0f}% of excess)")


def test_agency_spans_print_dates_in_the_findings_style():
    """DuckDB hands dates back as Timestamps; the findings print them as 'Jan 1, 2010', never '2010-01-01 00:00:00'."""
    spans = pd.DataFrame({"agency": ["DOF", "OSE"], "first_day": pd.to_datetime(["2010-01-01", "2020-06-08"]),
                          "last_day": pd.to_datetime(["2019-06-21", "2021-11-19"]), "n": [511020, 70555],
                          "change": ["stops early", "starts late"]})
    text = q2.agency_change_text(spans)
    assert "DOF stops early (Jan 1, 2010 to Jun 21, 2019, 511,020 requests)" in text
    assert "OSE starts late (Jun 8, 2020 to Nov 19, 2021, 70,555 requests)" in text
    assert "00:00:00" not in text and "2010-01-01" not in text
    one_day = spans.assign(last_day=spans.first_day).head(1)
    assert "(Jan 1, 2010, 511,020 requests)" in q2.agency_change_text(one_day)
    assert q2.agency_change_text(spans.head(0)) == "none"


def test_coverage_sentence_adds_up_on_its_own():
    s = q2.coverage_sentence(n_base=1000, n_closed=900, n_none=95, n_placeholder=5)
    assert s == "900 of 1,000 requests (90.0%) have a close date; 95 have none and 5 have a 1900-01-01 placeholder."
    s0 = q2.coverage_sentence(n_base=1000, n_closed=900, n_none=100, n_placeholder=0)
    assert s0 == "900 of 1,000 requests (90.0%) have a close date; 100 have none."
    # rows the three groups don't cover are counted rather than silently dropped
    s1 = q2.coverage_sentence(n_base=1000, n_closed=900, n_none=90, n_placeholder=5)
    assert s1.endswith("90 have none, 5 have a 1900-01-01 placeholder and 5 are neither (flags missing).")


def test_requests_outside_the_window_are_mentioned_only_when_there_are_some():
    assert q2.outside_sentence(0) == ""
    assert "43" in q2.outside_sentence(43) and "left out" in q2.outside_sentence(43)


def test_bulk_closure_needs_many_times_the_usual_volume_and_enough_rows():
    days = pd.date_range("2019-08-01", periods=60, freq="D")
    rows = [{"agency": "A", "d": d, "n": 50, "top_ts_n": 3, "top_ts": d, "n_ts": 40} for d in days]
    rows += [{"agency": "B", "d": d, "n": 3, "top_ts_n": 1, "top_ts": d, "n_ts": 3} for d in days]
    ad = pd.DataFrame(rows)
    big, mild, small = days[30], days[40], days[20]
    ad.loc[(ad.agency == "A") & (ad.d == big), ["n", "top_ts_n"]] = [2000, 1900]
    ad.loc[(ad.agency == "A") & (ad.d == mild), "n"] = 120
    ad.loc[(ad.agency == "B") & (ad.d == small), "n"] = 100
    out = q2.bulk_closures(ad, days[0], days[-1])
    assert list(zip(out.agency, out.date)) == [("A", big)]
    r = out.iloc[0]
    assert r.usual == 50 and r.ratio == 40 and r.top_share == pytest.approx(0.95)


# ---------------------------------------------------------------- the 20-file sample

@pytest.fixture(scope="module")
def q02(cleaned_cfg):
    out = q2.run(cleaned_cfg)
    return cleaned_cfg, out, cleaned_cfg.connect(), clean.clean_glob(cleaned_cfg)


def _one(con, sql):
    return con.sql(sql).fetchone()[0]


def test_daily_counts_cover_the_window_and_reconcile_with_the_table(q02):
    cfg, out, con, src = q02
    d = out["daily"]
    start, end = con.sql(f"""select min(created)::date, max(created)::date from {src}
                             where {BASE} and not out_of_file_order""").fetchone()
    assert d.date.min().date() == start and d.date.max().date() == end
    assert len(d) == (end - start).days + 1
    assert d.n_open.sum() + out["n_outside"] == _one(con, f"select count(*) from {src} where {BASE}")
    closed = _one(con, f"select count(*) from {src} where {BASE} and closed::date between '{start}' and '{end}'")
    assert d.n_close.sum() == closed


def test_every_flagged_day_has_a_reason(q02):
    cfg, out, con, src = q02
    inc = out["incomplete"]
    flagged = {pd.Timestamp(x) for x in con.sql(
        f"select distinct created::date from {src} where possibly_incomplete_day").df().iloc[:, 0]}
    assert set(inc.date) == flagged
    assert (inc.reason.str.len() > 0).all() and not inc.reason.str.contains(q2.UNEXPLAINED).any()
    by_day = inc.set_index("date").reason
    assert "311_batch_0010.json" in by_day[pd.Timestamp("2010-01-17")]  # the cut-off file
    assert "last file" in by_day[pd.Timestamp("2010-02-03")]
    assert inc.set_index("date").n_open[pd.Timestamp("2010-01-17")] == 337


def test_flagged_days_are_not_scored_and_holidays_show_up_as_low(q02):
    cfg, out, con, src = q02
    sc = out["scored"].set_index("date")
    inc = set(out["incomplete"].date)
    assert (sc.loc[sorted(inc), "kind"] == "excluded").all()
    low = sc[sc.kind == "low"]
    assert len(low) > 0 and low.holiday.notna().any()
    assert (low.n < low.baseline).all()


def test_closing_checks_match_the_cleaning_flags(q02):
    cfg, out, con, src = q02
    chk = out["closing_checks"].set_index("check").rows
    for check, flag in [("closed before opened", "closed_before_created"), ("closed at the second opened",
                         "closed_at_creation"), ("1900 placeholder close date", "closed_placeholder"),
                        ("closed more than a year after opening", "closed_after_1yr")]:
        assert chk[check] == _one(con, f"select count_if({flag}) from {src} where {BASE}"), check
    end = out["daily"].date.max().date()
    assert chk["closed after the last opening date"] == _one(
        con, f"select count_if(closed::date > '{end}') from {src} where {BASE}")
    assert chk["closed after the last opening date"] > 0


def test_closing_queries_run_directly_and_agree(cleaned_cfg):
    """The closing SQL (aliases like n_before, year) parses on its own and reconciles across helpers."""
    con, src = cleaned_cfg.connect(), clean.clean_glob(cleaned_cfg)
    start, end, _ = q2.analysis_window(con, src)
    checks, lo, hi = q2.closing_checks(con, src, start, end)
    chk = checks.set_index("check").rows
    assert checks.pct.between(0, 100).all() and lo <= hi
    by_ag = q2.closing_by_agency(con, src)
    assert by_ag.n.sum() == chk["requests (default set)"] and by_ag.n_closed.sum() == chk["with a close date"]
    outside = q2.closes_outside(con, src, start, end)
    assert outside[outside.side == "after last opening"].n.sum() == chk["closed after the last opening date"]
    assert outside[outside.side == "before first opening"].n.sum() == chk["closed before the first opening date"]
    assert q2.placeholder_status(con, src).n.sum() == chk["1900 placeholder close date"]


def test_week_patterns_use_only_complete_days_and_complete_weeks(q02):
    cfg, out, con, src = q02
    d, pat = out["daily"].set_index("date"), out["patterns"]
    ok = d[~d.incomplete]
    sundays = ok[ok.index.dayofweek == 6].n_open.mean()
    dow = pat["dow"].set_index("weekday")
    assert len(dow) == 7 and dow.loc[6, "open"] == pytest.approx(sundays)
    assert dow.loc[6, "n_days_open"] == (ok.index.dayofweek == 6).sum()
    for r in pat["woy"].dropna(subset=["open"]).itertuples():
        first = pd.Timestamp(year=r.year, month=1, day=1) + pd.Timedelta(days=7 * (r.week - 1))
        days = pd.date_range(first, periods=7)
        assert days.isin(ok.index).all(), (r.year, r.week)
    assert pat["avg_years"] == [2010] and not pat["full_years"]  # five weeks: no complete year


def test_outputs_charts_and_answer_are_written(q02):
    cfg, out, con, src = q02
    for name in ["q02_daily_volume", "q02_week_patterns"]:
        assert (cfg.out_dir / "figs" / f"{name}.png").stat().st_size > 10_000
        assert f"](figs/{name}.png)" in (cfg.out_dir / "answers" / "q02.md").read_text()
    for name in ["q02_daily", "q02_incomplete_days", "q02_closing_checks", "q02_week_patterns"]:
        assert len(pd.read_csv(cfg.out_dir / "agg" / f"{name}.csv")) > 0, name
    text = (cfg.out_dir / "answers" / "q02.md").read_text()
    assert text.startswith("# Q2") and "{" not in text and "nan" not in text.split()
    assert len((cfg.out_dir / "reports" / f"{q2.STAGE}.txt").read_text().splitlines()) <= 152
    note = text.splitlines()[2]
    assert note.startswith("*Computed by `nyc311/q02_timing.py` from ") and note.endswith("*")
    assert "write the prose" not in text and "Drafted" not in text
    n = con.sql(f"select count(*) from {src} where {BASE}").fetchone()[0]
    assert f"{n:,}" in note
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)  # no raw timestamps in the findings


def test_close_date_coverage_line_adds_up_to_the_requests(q02):
    cfg, out, con, src = q02
    text = (cfg.out_dir / "answers" / "q02.md").read_text()
    line = next(x for x in text.splitlines() if x.startswith("- **Coverage.**"))
    n, closed, none, ph = con.sql(f"""select count(*), count(closed), count_if(closed is null and not closed_placeholder),
                                     count_if(closed_placeholder) from {src} where {BASE}""").fetchone()
    assert ph > 0 and closed + none + ph == n   # the sample has 1900 placeholders, so all three groups show
    assert f"{closed:,} of {n:,} requests (" in line
    assert line.endswith(f"; {none:,} have none and {ph:,} have a 1900-01-01 placeholder.")


# ---------------------------------------------------------------- several years (synthetic pull)

@pytest.fixture(scope="module")
def years(tmp_path_factory):
    """Twelve synthetic years (2010 to Nov 2021) with planted problem files and a run of missing batch numbers
    near the end, which the cleaning stage's sparse-tail rule turns into flagged days from mid-2020 on."""
    sys.path.insert(0, str(REPO / "tools"))
    import make_synthetic_batches as synth
    root = tmp_path_factory.mktemp("years")
    synth.main(["--files", "140", "--rows-per-file", "150", "--start", "2010-01-01", "--end", "2021-11-20",
                "--missing-numbers", "6", "--out", str(root / "data")])
    cfg = Config.from_args(["--data", str(root / "data" / "311_batch_*.json"), "--out", str(root / "out"),
                            "--work", str(root / "work"), "--threads", "2", "--memory-limit", "2GB"])
    ingest.run(cfg)
    clean.run(cfg)
    return cfg, q2.run(cfg)


def test_several_years_use_yearly_periods_and_average_only_full_years(years):
    cfg, out = years
    con, src = cfg.connect(), clean.clean_glob(cfg)
    assert out["unit"] == "year"
    pat = out["patterns"]
    full = clean.full_years(con, src)
    assert pat["full_years"] == full
    assert pat["avg_years"] == [y for y in full if y <= rules.PROFILE_LAST_YEAR] and len(pat["avg_years"]) >= 2
    assert pat["woy"].dropna(subset=["open"]).year.nunique() >= 3
    assert (cfg.out_dir / "figs" / "q02_week_patterns.png").exists()


def test_typical_profiles_leave_out_years_after_the_profile_cutoff(years):
    cfg, out = years
    pat, d = out["patterns"], out["daily"]
    later = sorted(y for y in d.date.dt.year.unique() if y > rules.PROFILE_LAST_YEAR)
    assert later and pat["later_years"] == later
    assert not set(pat["avg_years"]) & set(later)
    # the typical week-of-year line is the mean over the averaged years only
    woy = pat["woy"]
    expect = woy[woy.year.isin(pat["avg_years"])].groupby("week").open.mean()
    got = pat["woy_avg"].set_index("week").open
    assert np.allclose(got.reindex(expect.index), expect, equal_nan=True)
    # and the typical weekday bar is the mean over usable days (not flagged, not empty) of those years
    ok = d[~d.incomplete & (d.n_open > 0) & d.date.dt.year.isin(pat["avg_years"])]
    tue = ok[ok.date.dt.dayofweek == 1].n_open.mean()
    assert pat["dow"].set_index("weekday").open[1] == pytest.approx(tue)
    # the peak-week statement comes from that profile
    wa = pat["woy_avg"].dropna(subset=["open"])
    peak = int(wa.loc[wa.open.idxmax(), "week"])
    text = (cfg.out_dir / "answers" / "q02.md").read_text()
    assert f"peak in week {peak} " in text
    for y in later:
        assert str(y) in text.split("## (c)")[1]
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)  # agency spans etc. print readable dates


def test_several_years_explain_flagged_days_with_missing_batches(years):
    cfg, out = years
    inc = out["incomplete"]
    assert len(inc) > 0 and not inc.reason.str.contains(q2.UNEXPLAINED).any()
    assert inc.reason.str.contains("missing").any()
    assert len(out["missing_batches"]) > 0
    # the sparse tail: every flagged day from its start to the end has the one tail reason (days with no rows at
    # all, inside the run of missing batches, have nothing to flag and show up as gaps instead)
    tail = inc[inc.reason.str.contains("end of the pull")]
    assert len(tail) > 0 and tail.date.max() == inc.date.max() and tail.reason.nunique() == 1
    assert inc[inc.date >= tail.date.min()].reason.str.contains("end of the pull").all()


def test_days_with_no_openings_do_not_count_as_usable_weeks(years):
    cfg, out = years
    d, woy = out["daily"], out["patterns"]["woy"]
    empty = d[(d.n_open == 0) & ~d.incomplete]
    assert len(empty) > 0  # the run of missing batch numbers leaves days with no rows at all
    for r in woy.dropna(subset=["open"]).itertuples():
        first = pd.Timestamp(year=r.year, month=1, day=1) + pd.Timedelta(days=7 * (r.week - 1))
        assert not pd.date_range(first, periods=7).isin(empty.date).any(), (r.year, r.week)
