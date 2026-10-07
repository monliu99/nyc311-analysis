import re

import pandas as pd
import pytest

from nyc311 import clean, rules
from nyc311 import q04_response_times as q4
from nyc311 import q11_summary
from nyc311.clean import clean_glob


@pytest.fixture(scope="module")
def ran(cleaned_cfg):
    q11_summary.run(cleaned_cfg)
    return cleaned_cfg


def _nums(cfg):
    return pd.read_csv(cfg.out_dir / "agg" / "q11_key_numbers.csv", dtype=str, keep_default_na=False).set_index("metric")


def _one(cfg, sql):
    con = cfg.connect()  # keep a reference: duckdb 1.4 closes an unreferenced connection before the fetch
    return con.sql(sql.format(src=clean_glob(cfg))).fetchone()[0]


def test_key_numbers_match_the_clean_table(ran):
    nums = _nums(ran).value
    assert int(nums["requests"]) == _one(ran, f"select count(*) from {{src}} where {rules.BASE_SQL}")
    assert int(nums["complaints"]) == _one(ran, f"select count(*) from {{src}} where {rules.COMPLAINTS_SQL}")
    assert 0 < float(nums["median_hours_to_close"]) < 365 * 24
    assert 0 <= float(nums["pct_unresolved"]) <= 100
    assert int(nums["unresolved"]) == _one(ran, f"select count_if(unresolved) from {{src}} where {rules.COMPLAINTS_SQL}")


@pytest.mark.parametrize("value, kind, text", [(2011, "year", "2011"), ("2011", "year", "2011"), (3.0, "pct", "3.0"),
                                               (39.8, "pct", "39.8"), (28239945, "count", "28,239,945"),
                                               (40.73, "hours", "40.7"), ("Noise", "text", "Noise"),
                                               ("2010-01-01", "date", "2010-01-01")])
def test_values_are_formatted_by_kind(value, kind, text):
    assert q11_summary.fmt_value(value, kind) == text


def test_years_are_plain_and_percentages_have_one_decimal(ran):
    nums = _nums(ran)
    assert set(nums.kind) <= {"count", "pct", "hours", "year", "date", "text"}
    for m, r in nums.iterrows():
        if r.kind == "pct":
            assert re.fullmatch(r"-?\d+\.\d", r.value) and r.display == r.value, m
        if r.kind == "year":
            assert re.fullmatch(r"\d{4}", r.value) and r.display == r.value, m
    text = (ran.out_dir / "answers" / "q11.md").read_text()
    for r in nums.itertuples():
        assert f"| {r.display} |" in text
    assert not re.search(r"\b\d,\d{3}\b(?!,)", "\n".join(l for l in text.splitlines() if "year" in l.lower()))


def test_growth_to_the_last_pre_pandemic_year_and_to_the_last_full_year():
    counts = {2011: 1000, 2019: 1237, 2020: 1398}
    rows = {r["metric"]: r for r in q11_summary.growth_rows(counts, list(range(2011, 2021)), 2019)}
    assert rows["growth_to_profile_year_pct"]["value"] == pytest.approx(23.7)
    assert "2011 to 2019" in rows["growth_to_profile_year_pct"]["label"]
    assert rows["growth_pct"]["value"] == pytest.approx(39.8) and "2011 to 2020" in rows["growth_pct"]["label"]
    assert rows["first_full_year"]["value"] == "2011" and rows["last_full_year"]["value"] == "2020"
    assert rows["complaints_2019"]["value"] == 1237
    short = {r["metric"] for r in q11_summary.growth_rows({2011: 10, 2015: 12}, list(range(2011, 2016)), 2019)}
    assert "growth_pct" in short and "growth_to_profile_year_pct" not in short
    assert q11_summary.growth_rows({2011: 10}, [2011], 2019) == []


def test_full_years_come_from_the_shared_rule(ran):
    con = ran.connect()
    _, full = q11_summary.key_numbers(ran)
    assert list(full) == clean.full_years(con, clean_glob(ran))


def test_no_district_share_uses_q1s_rule(ran):
    from nyc311.q01_missing import NO_DISTRICT_SQL
    nums = _nums(ran)
    pct = _one(ran, f"select 100 * avg(({NO_DISTRICT_SQL})::int) from {{src}} where {rules.BASE_SQL}")
    assert float(nums.value["pct_no_district"]) == round(pct, 1)
    assert int(nums.value["no_district"]) == _one(
        ran, f"select count_if({NO_DISTRICT_SQL}) from {{src}} where {rules.BASE_SQL}")
    assert q11_summary.NO_DISTRICT_SQL is NO_DISTRICT_SQL
    assert nums.set["pct_no_district"].startswith("requests") and "Q1" in nums.label["no_district"]


def test_median_excluding_instant_closures_equals_q4s(ran):
    con = ran.connect()
    q4._load(con, ran)
    s = q4.summary_stats(con).set_index("set")
    nums = _nums(ran).value
    assert float(nums["median_hours_to_close_excl_instant"]) == round(s.loc[q4.RESPONSE_SET, "median_h"], 1)
    assert float(nums["median_hours_to_close"]) == round(s.loc["all closed complaints", "median_h"], 1)


def test_every_number_names_the_set_it_covers(ran):
    nums = _nums(ran)
    assert (nums["set"].str.len() > 0).all()
    text = (ran.out_dir / "answers" / "q11.md").read_text()
    assert "| Metric | Value | Counted over |" in text


def test_answers_file_opens_with_a_neutral_provenance_line(ran):
    text = (ran.out_dir / "answers" / "q11.md").read_text()
    lines = text.splitlines()
    assert lines[0].startswith("# Q11")
    first = lines[2]
    assert first.startswith("*Computed by `nyc311/q11_summary.py` from ") and first.endswith(".*")
    nums = _nums(ran).display
    assert f"{nums['requests']} requests" in first and f"{nums['complaints']} complaints" in first
    assert "report/prose" not in text and "write" not in first.lower()
    assert "|" in text


def test_periods_without_a_full_year_are_reported_not_compared(ran):
    text = (ran.out_dir / "answers" / "q11.md").read_text()
    assert "full calendar year" in text


def test_report_block_is_written(ran):
    assert (ran.out_dir / "reports" / "q11_summary.txt").exists()
