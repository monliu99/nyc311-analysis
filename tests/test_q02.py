"""Q2 (daily volume, closing times, week patterns): pure helpers, the 20-file sample, and a multi-year pull."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyc311 import clean, ingest, q02_timing as q2
from nyc311.config import Config

REPO = Path(__file__).resolve().parent.parent
BASE = "not is_test and not dup_key"


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


# ---------------------------------------------------------------- several years (synthetic pull)

@pytest.fixture(scope="module")
def years(tmp_path_factory):
    sys.path.insert(0, str(REPO / "tools"))
    import make_synthetic_batches as synth
    root = tmp_path_factory.mktemp("years")
    synth.main(["--files", "70", "--rows-per-file", "300", "--start", "2010-01-01", "--end", "2013-12-31",
                "--out", str(root / "data")])
    cfg = Config.from_args(["--data", str(root / "data" / "311_batch_*.json"), "--out", str(root / "out"),
                            "--work", str(root / "work"), "--threads", "2", "--memory-limit", "2GB"])
    ingest.run(cfg)
    clean.run(cfg)
    return cfg, q2.run(cfg)


def test_several_years_use_yearly_periods_and_average_only_full_years(years):
    cfg, out = years
    assert out["unit"] == "year"
    pat = out["patterns"]
    assert len(pat["full_years"]) >= 2 and set(pat["avg_years"]) == set(pat["full_years"])
    assert pat["woy"].dropna(subset=["open"]).year.nunique() >= 3
    assert (cfg.out_dir / "figs" / "q02_week_patterns.png").exists()


def test_several_years_explain_flagged_days_with_missing_batches(years):
    cfg, out = years
    inc = out["incomplete"]
    assert len(inc) > 0 and not inc.reason.str.contains(q2.UNEXPLAINED).any()
    assert inc.reason.str.contains("missing").any()
    assert len(out["missing_batches"]) > 0
