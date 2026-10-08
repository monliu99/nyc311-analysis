"""Q5 top 10 complaint types, on the 20-file sample (fixture `cleaned_cfg`) and a small two-year synthetic table."""
import datetime as dt
import re

import duckdb
import numpy as np
import pandas as pd
import pytest

from nyc311 import clean, q05_top_types as q5, rules
from nyc311.clean import clean_glob
from nyc311.config import Config

COMPLAINTS = rules.COMPLAINTS_SQL


@pytest.fixture(scope="module")
def res(cleaned_cfg):
    return q5.run(cleaned_cfg)


def _one(cfg, sql):
    con = cfg.connect()  # keep a reference: duckdb 1.4 closes an unreferenced connection before the fetch
    return con.sql(sql.format(src=clean_glob(cfg))).fetchone()[0]


def test_top10_is_sorted_and_shares_are_of_all_complaints(res, cleaned_cfg):
    t = res["top"]
    assert len(t) == q5.TOP_N
    assert t.n.is_monotonic_decreasing and t["rank"].tolist() == list(range(1, q5.TOP_N + 1))
    total = _one(cleaned_cfg, f"select count(*) from {{src}} where {COMPLAINTS} and canonical_type is not null")
    assert res["total"] == total
    assert np.allclose(t.pct, 100 * t.n / total)
    assert np.allclose(t.cum_pct, t.pct.cumsum()) and t.cum_pct.iloc[-1] <= 100


def test_top_count_matches_an_independent_count(res, cleaned_cfg):
    top = res["top"].iloc[0]
    n = _one(cleaned_cfg, f"select count(*) from {{src}} where {COMPLAINTS} and canonical_type = "
                          f"'{top.canonical_type.replace(chr(39), chr(39) * 2)}'")
    assert top.n == n
    # nothing else is bigger
    biggest = _one(cleaned_cfg, f"select max(n) from (select count(*) n from {{src}} where {COMPLAINTS} "
                                f"and canonical_type is not null group by canonical_type)")
    assert top.n == biggest


def test_information_requests_and_tests_are_excluded_and_reconcile(res, cleaned_cfg):
    con = cleaned_cfg.connect()
    info_types = con.sql(
        f"select distinct canonical_type from {clean_glob(cleaned_cfg)} where is_info_request").df().canonical_type
    assert not set(res["top"].canonical_type) & set(info_types)
    x = res["excluded"]
    n_rows = _one(cleaned_cfg, "select count(*) from {src}")
    assert x.n_rows == n_rows
    assert x.n_rows == x.n_test + x.n_dup + x.n_info + x.n_invalid + x.n_no_type + res["total"]
    assert x.n_info > 0
    assert x.n_invalid == _one(cleaned_cfg, f"select count(*) from {{src}} where {rules.BASE_SQL} and not "
                                            f"coalesce(is_info_request, false) and category = '{rules.INVALID_CATEGORY}'")


def test_main_agency_is_the_agency_with_most_rows_of_that_type(res, cleaned_cfg):
    top = res["top"].iloc[0]
    q = top.canonical_type.replace("'", "''")
    agency = _one(cleaned_cfg, f"select agency from {{src}} where {COMPLAINTS} and canonical_type = '{q}' "
                               "group by 1 order by count(*) desc limit 1")
    assert top.main_agency == agency


@pytest.mark.parametrize("d0, d1, short, long", [
    (dt.date(2010, 1, 1), dt.date(2010, 2, 3), "Jan 1–Feb 3, 2010", "Jan 1, 2010 to Feb 3, 2010"),
    (dt.date(2010, 3, 2), dt.date(2010, 3, 30), "Mar 2–30, 2010", "Mar 2, 2010 to Mar 30, 2010"),
    (dt.date(2010, 1, 1), dt.date(2021, 11, 20), "2010–2021", "Jan 1, 2010 to Nov 20, 2021"),
])
def test_period_labels_come_from_the_dates(d0, d1, short, long):
    assert q5.period_label(d0, d1) == short
    assert q5.period_long(d0, d1) == long


def test_period_ignores_backfilled_rows(res, cleaned_cfg):
    d0, d1 = _one(cleaned_cfg, "select min(created)::date from {src} where not out_of_file_order"), \
        _one(cleaned_cfg, "select max(created)::date from {src} where not out_of_file_order")
    assert res["period"] == q5.period_label(d0, d1)
    assert res["n_outside_period"] == _one(
        cleaned_cfg, f"select count(*) from {{src}} where {COMPLAINTS} and canonical_type is not null and "
                     f"(created::date < '{d0}' or created::date > '{d1}')")


def _toy_clean(rows, invalid=0, flagged=()):
    """Rows are (created, canonical_type, agency, is_info_request[, raw complaint_type, descriptor]); without the
    last two, the raw type is the canonical one and the descriptor is missing."""
    rows = [tuple(r) + (r[1], None) if len(r) == 4 else tuple(r) for r in rows]
    df = pd.DataFrame(rows, columns=["created", "canonical_type", "agency", "is_info_request", "complaint_type",
                                     "descriptor"])
    df["descriptor"] = df.descriptor.astype("string")
    df["created"] = pd.to_datetime(df.created)
    df["category"] = "c"
    for c in ["is_test", "dup_key", "out_of_file_order", "possibly_incomplete_day"]:
        df[c] = False
    df.loc[df.created.dt.strftime("%Y-%m-%d").isin(flagged), "possibly_incomplete_day"] = True
    if invalid:  # rows that are not service requests (taxonomy category "Invalid entries")
        bad = df.iloc[:invalid].copy().assign(category=rules.INVALID_CATEGORY, canonical_type="Probe")
        df = pd.concat([df, bad], ignore_index=True)
    return df


@pytest.mark.parametrize("ratio, text", [(1.044, "4% more than"), (1.49, "49% more than"), (1.5, "1.5 times"),
                                          (2.0, "2.0 times")])
def test_leader_vs_next_type_reads_as_percent_more_when_close(ratio, text):
    assert q5.ratio_phrase(ratio).startswith(text)


@pytest.mark.parametrize("full, present, avg, later", [
    ([2011, 2012, 2019, 2020], range(2010, 2022), [2011, 2012, 2019], [2020, 2021]),
    ([], [2010], [2010], []),
    ([2020], [2020, 2021], [2020], []),
])
def test_profile_years_stop_at_the_profile_cutoff(full, present, avg, later):
    assert q5.profile_years(full, present, last=2019) == (avg, later)


def test_change_years_come_from_the_shared_full_year_rule():
    """A year with a flagged day is not full (clean.full_years), so the comparison skips it."""
    con = duckdb.connect()
    rows = [(f"{y}-{m}", "A", "X", False) for y in (2016, 2017, 2018, 2019, 2020) for m in ("01-01", "12-31")]
    rows += [("2017-06-01", "A", "X", False), ("2021-03-01", "A", "X", False)]
    con.register("toy", _toy_clean(rows, flagged=("2017-06-01",)))
    full = clean.full_years(con, "toy")
    assert full == [2016, 2018, 2019, 2020]
    assert q5.comparison_years(full) == (2016, 2019, 2020)
    assert q5.comparison_years([2016, 2018]) == (2016, 2018, None)
    assert q5.comparison_years([2020, 2021]) == (2020, 2021, None)
    assert q5.comparison_years([2018]) is None


@pytest.fixture(scope="module")
def two_years(tmp_path_factory):
    """A tiny clean table covering 2018-2019 in full and part of 2020, where type C overtakes type A."""
    root = tmp_path_factory.mktemp("q5two")
    rng = np.random.default_rng(0)
    rows = []
    for year, weights in [(2018, {"A": 50, "B": 30, "C": 10}), (2019, {"A": 40, "B": 30, "C": 25}),
                          (2020, {"A": 5, "B": 5, "C": 5})]:
        last = 12 if year < 2020 else 3
        for t, k in weights.items():
            for i in range(k):
                day = dt.date(year, 1, 1) + dt.timedelta(days=int(rng.integers(0, 365 if last == 12 else 80)))
                rows.append((day.isoformat(), t, "AG" + t, False))
        rows += [(f"{year}-01-01", "A", "AGA", False), (f"{year}-{last:02d}-28", "A", "AGA", False)]
    for i in range(12):  # more types so a top 10 exists, plus information requests
        rows += [("2019-06-01", f"T{i:02d}", "AGT", False)] * (i + 1)
    rows += [("2019-06-01", "Info", "AGI", True)] * 500
    work = root / "work"
    work.mkdir()
    _toy_clean(rows, invalid=7).to_parquet(work / "clean.parquet")
    cfg = Config.from_args(["--out", str(root / "out"), "--work", str(work), "--threads", "2"])
    return cfg, q5.run(cfg)


@pytest.fixture(scope="module")
def four_years(tmp_path_factory):
    """2017-2020 in full and part of 2021: the change chart compares 2017 with rules.PROFILE_LAST_YEAR (2019) and
    shows 2020, the last full year, as a third marker; type B surges in 2020 only."""
    root = tmp_path_factory.mktemp("q5four")
    rows = []
    for year, weights in [(2017, {"A": 50, "B": 20, "C": 10}), (2018, {"A": 45, "B": 20, "C": 15}),
                          (2019, {"A": 40, "B": 20, "C": 25}), (2020, {"A": 40, "B": 90, "C": 25}),
                          (2021, {"A": 5, "B": 5, "C": 5})]:
        for t, k in weights.items():
            rows += [(f"{year}-06-15", t, "AG" + t, False)] * k
        rows += [(f"{year}-01-01", "A", "AGA", False)] + ([(f"{year}-12-31", "A", "AGA", False)] if year < 2021 else [])
    for i in range(12):
        rows += [("2019-06-01", f"T{i:02d}", "AGT", False)] * (i + 1)
    work = root / "work"
    work.mkdir()
    _toy_clean(rows).to_parquet(work / "clean.parquet")
    cfg = Config.from_args(["--out", str(root / "out"), "--work", str(work), "--threads", "2"])
    return cfg, q5.run(cfg)


def test_change_compares_first_full_year_with_profile_cutoff_and_shows_last_full_year(four_years):
    cfg, r = four_years
    assert rules.PROFILE_LAST_YEAR == 2019
    assert r["years"] == (2017, 2019) and r["later_year"] == 2020
    ch = r["change"].set_index("canonical_type")
    for y in (2017, 2019, 2020):
        assert 0 < ch[f"pct_{y}"].sum() <= 100 + 1e-9
    assert ch.loc["B", "pct_2020"] > ch.loc["B", "pct_2019"]
    # the headline change is 2017 -> 2019, not driven by 2020's jump in B
    assert ch.loc["C", "change_pts"] == pytest.approx(ch.loc["C", "pct_2019"] - ch.loc["C", "pct_2017"])
    assert (cfg.out_dir / "figs" / "q05_top10_change.png").stat().st_size > 10_000
    text = (cfg.out_dir / "answers" / "q05.md").read_text()
    assert "2017 and 2019" in text and "2020" in text.split("figs/q05_top10.png")[1]
    assert "figs/q05_top10_change.png" in text


def test_change_between_first_and_last_full_year(two_years):
    cfg, r = two_years
    ch = r["change"]
    assert r["years"] == (2018, 2019) and r["later_year"] is None
    assert r["excluded"].n_invalid == 7 and "Probe" not in set(r["counts"].canonical_type)
    assert set(ch.canonical_type) == set(r["top"].canonical_type)
    for y in (2018, 2019):
        assert 0 < ch[f"pct_{y}"].sum() <= 100 + 1e-9
    a = ch.set_index("canonical_type")
    assert a.loc["C", "pct_2019"] > a.loc["C", "pct_2018"]
    assert (cfg.out_dir / "figs" / "q05_top10_change.png").stat().st_size > 10_000
    assert "Info" not in set(r["top"].canonical_type)
    text = (cfg.out_dir / "answers" / "q05.md").read_text()
    assert "figs/q05_top10_change.png" in text


def test_no_change_chart_without_two_full_years(res, cleaned_cfg):
    assert res["change"] is None
    assert not (cleaned_cfg.out_dir / "figs" / "q05_top10_change.png").exists()
    text = (cleaned_cfg.out_dir / "answers" / "q05.md").read_text()
    assert "full calendar year" in text


def test_outputs_and_findings(res, cleaned_cfg):
    out = cleaned_cfg.out_dir
    assert (out / "figs" / "q05_top10.png").stat().st_size > 10_000
    assert (out / "agg" / "q05_top10.csv").exists()
    block = (out / "reports" / "q05_top_types.txt").read_text()
    assert block.startswith("===== BEGIN q05_top_types =====")
    text = (out / "answers" / "q05.md").read_text()
    assert text.splitlines()[0].startswith("# Q5")
    note = text.splitlines()[2]
    assert note == f"*Computed by `nyc311/q05_top_types.py` from {res['total']:,} complaints, {res['period_long']}.*"
    assert "write the prose" not in text
    t = res["top"]
    assert q5.ratio_phrase(t.n.iloc[0] / t.n.iloc[1]) in text
    top = res["top"].iloc[0]
    assert top.canonical_type in text and f"{int(top.n):,}" in text
    assert f"{int(res['excluded'].n_info):,}" in text
    assert re.search(r"!\[[^\]]+\]\(figs/q05_top10\.png\)", text)
    assert not re.search(r"<[a-zA-Z/][^>]*>", text)
    assert not re.search(r"\bnan\b|\bNone\b", text)


# ---------- like-for-like check: descriptors moved to another type by an agency recoding ----------

def test_name_key_reads_plural_and_separator_variants_as_one_name():
    assert q5.name_key("WATER-LEAKS") == q5.name_key("WATER LEAK") == "water leak"
    assert q5.name_key("JANITOR/SUPER") == q5.name_key("JANITOR-SUPER")
    assert q5.name_key(None) == "" and q5.name_key(float("nan")) == ""


def _rel_rows():
    """2016-2020 in full and part of 2021 (HPD and DOT rows; raw type and descriptor given).
    - Plumbing: BASIN 30 a year; WATER-LEAKS 40 in 2016 and 10 up to 2017-03-10, then the new raw type WATER LEAK
      (canonical Water Leak) from 2017-03-11: a descriptor that became its own type.
    - General MOLD (20 in 2016, last 2017-03-03) continues as Unsanitary MOLD from 2017-03-11: it joined a top type.
    - Street CAVE-IN stops 2016-06-01 and a 'Cave In' type starts 2016-09-01: too late to be the same thing.
    - Paint drops for real (60 -> 25); Noise grows (50 -> 80)."""
    r = []
    add = lambda d, canon, ag, raw, desc, k: r.extend([(d, canon, ag, False, raw, desc)] * k)  # noqa: E731
    for y in range(2016, 2021):
        add(f"{y}-06-15", "Plumbing", "HPD", "PLUMBING", "BASIN", 30)
        add(f"{y}-06-15", "Unsanitary", "HPD", "UNSANITARY CONDITION", "PESTS", 30)
        add(f"{y}-06-15", "Street", "DOT", "Street Condition", "POTHOLE", 50)
        add(f"{y}-06-15", "Paint", "HPD", "PAINT/PLASTER", "WALL", {2016: 60, 2017: 50, 2018: 40}.get(y, 25))
        add(f"{y}-06-15", "Noise", "NYPD", "Noise", "Loud Music", {2016: 50, 2017: 60, 2018: 70}.get(y, 80))
        if y >= 2018:
            add(f"{y}-06-15", "Water Leak", "HPD", "WATER LEAK", "SLOW LEAK", 40)
            add(f"{y}-06-15", "Unsanitary", "HPD", "UNSANITARY CONDITION", "MOLD", 20)
        if y >= 2017:
            add(f"{y}-06-15", "Cave In", "DOT", "Cave In", "Cave-In", 10)
    add("2016-06-15", "Plumbing", "HPD", "PLUMBING", "WATER-LEAKS", 40)
    add("2017-03-10", "Plumbing", "HPD", "PLUMBING", "WATER-LEAKS", 10)
    add("2017-03-11", "Water Leak", "HPD", "WATER LEAK", "HEAVY FLOW", 30)
    add("2016-06-15", "General", "HPD", "GENERAL CONSTRUCTION", "MOLD", 20)
    add("2017-03-03", "General", "HPD", "GENERAL CONSTRUCTION", "MOLD", 5)
    add("2017-03-11", "Unsanitary", "HPD", "UNSANITARY CONDITION", "MOLD", 15)
    add("2016-06-01", "Street", "DOT", "Street Condition", "CAVE-IN", 10)
    add("2016-09-01", "Cave In", "DOT", "Cave In", "Cave-In", 10)
    add("2016-01-01", "Noise", "NYPD", "Noise", "Loud Music", 1)
    add("2020-12-31", "Noise", "NYPD", "Noise", "Loud Music", 1)
    add("2021-03-01", "Noise", "NYPD", "Noise", "Loud Music", 1)
    return r


@pytest.fixture(scope="module")
def relabeled(tmp_path_factory):
    root = tmp_path_factory.mktemp("q5relabel")
    work = root / "work"
    work.mkdir()
    _toy_clean(_rel_rows()).to_parquet(work / "clean.parquet")
    cfg = Config.from_args(["--out", str(root / "out"), "--work", str(work), "--threads", "2"])
    return cfg, q5.run(cfg)


def test_relabelings_link_a_stopped_descriptor_to_a_same_name_successor_that_starts_at_once(relabeled):
    cfg, r = relabeled
    mv = r["relabelings"]
    pairs = {(m.old_type, m.old_descriptor, m.new_type, m.via) for m in mv.itertuples()}
    assert ("Plumbing", "WATER-LEAKS", "Water Leak", "type") in pairs
    assert ("General", "MOLD", "Unsanitary", "descriptor") in pairs
    assert not any(m.old_type == "Street" for m in mv.itertuples())  # successor started 92 days later
    p = mv[mv.old_descriptor == "WATER-LEAKS"].iloc[0]
    assert p.n_old == 50 and str(p.last_old)[:10] == "2017-03-10" and str(p.first_new)[:10] == "2017-03-11"
    assert p.weight == pytest.approx(1.0)


def test_relabelings_respect_the_window():
    c = pd.DataFrame([("A", "OLD", "T1", "THING", 100, "2015-01-01", "2016-05-01"),
                      ("A", "OLD", "T1", "KEEP", 100, "2015-01-01", "2017-12-31"),
                      ("A", "THING", "T2", "X", 80, f"2016-05-{1 + q5.MOVE_WINDOW_DAYS:02d}", "2017-12-31"),
                      ("A", "OTHER", "T3", "THING", 80, f"2016-05-{2 + q5.MOVE_WINDOW_DAYS:02d}", "2017-12-31"),
                      ("B", "THING", "T4", "X", 80, "2016-05-02", "2017-12-31")],
                     columns=["agency", "complaint_type", "canonical_type", "descriptor", "n", "first", "last"])
    for col in ["first", "last"]:
        c[col] = pd.to_datetime(c[col])
    mv = q5.relabelings(c, pd.Timestamp("2015-01-01"), pd.Timestamp("2017-12-31"))
    assert list(mv.new_type) == ["T2"]  # T3 starts one day too late; T4 belongs to another agency


def test_like_for_like_counts_moved_descriptors_under_their_first_year_type(relabeled):
    cfg, r = relabeled
    assert r["years"] == (2016, 2019) and r["later_year"] == 2020
    t = r["totals"]
    ll = r["like_for_like"]
    ll = ll[(ll.year_a == 2016) & (ll.year_b == 2019)].set_index("canonical_type")
    assert ll.loc["Plumbing", "pct_b_same_labels"] == pytest.approx(100 * (30 + 40) / t[2019])
    assert ll.loc["Water Leak", "pct_b_same_labels"] == pytest.approx(0)
    assert ll.loc["Unsanitary", "pct_b_same_labels"] == pytest.approx(100 * 30 / t[2019])
    assert ll.loc["Paint", "pct_b_same_labels"] == pytest.approx(ll.loc["Paint", "pct_b"])
    assert ll.loc["Plumbing", "relabeled"] and not ll.loc["Paint", "relabeled"]
    assert ll.loc["Plumbing", "pct_a"] == pytest.approx(100 * 70 / t[2016])  # BASIN + WATER-LEAKS
    ch = r["change"].set_index("canonical_type")
    assert ch.change_pts.idxmin() == "Plumbing"
    assert ll.change_same_labels_pts.idxmin() == "Paint"
    assert (cfg.out_dir / "agg" / "q05_like_for_like.csv").exists()
    assert (cfg.out_dir / "agg" / "q05_relabelings.csv").exists()


def test_findings_do_not_overstate_a_drop_that_is_partly_a_relabeling(relabeled):
    cfg, r = relabeled
    text = (cfg.out_dir / "answers" / "q05.md").read_text()
    line = next(x for x in text.splitlines() if x.startswith("- **Like-for-like check"))
    assert "PLUMBING 'WATER-LEAKS'" in line and "WATER LEAK" in line
    assert "2017-03-10" in line and "2017-03-11" in line and "50 complaints" in line
    assert "GENERAL CONSTRUCTION 'MOLD'" in line
    t = r["totals"]
    assert f"{100 * 70 / t[2016]:.1f}% to {100 * 70 / t[2019]:.1f}%" in line
    assert "the biggest drop is Paint" in line and "the biggest gain is Noise" in line
    assert "Cave" not in line
    assert not re.search(r"\bnan\b|\bNone\b", text)
    block = (cfg.out_dir / "reports" / "q05_top_types.txt").read_text()
    assert "like for like" in block.lower()


def test_no_relabeling_found_is_said_plainly(four_years):
    cfg, r = four_years
    assert r["relabelings"].empty
    text = (cfg.out_dir / "answers" / "q05.md").read_text()
    line = next(x for x in text.splitlines() if x.startswith("- **Like-for-like check"))
    assert "No top type" in line
