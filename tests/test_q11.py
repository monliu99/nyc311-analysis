import pandas as pd
import pytest

from nyc311 import q11_summary
from nyc311.clean import clean_glob


@pytest.fixture(scope="module")
def ran(cleaned_cfg):
    q11_summary.run(cleaned_cfg)
    return cleaned_cfg


def test_key_numbers_match_the_clean_table(ran):
    con = ran.connect()
    n = con.sql(f"select count(*) from {clean_glob(ran)} where not is_test and not dup_key").fetchone()[0]
    nums = pd.read_csv(ran.out_dir / "agg" / "q11_key_numbers.csv").set_index("metric").value
    assert int(float(nums["requests"])) == n
    assert 0 < float(nums["median_days_to_close"]) < 365
    assert 0 <= float(nums["pct_unresolved"]) <= 100


def test_answers_file_has_heading_and_points_to_the_prose(ran):
    text = (ran.out_dir / "answers" / "q11.md").read_text()
    assert text.startswith("# Q11")
    assert "report/prose/q11.md" in text
    assert "|" in text


def test_periods_without_a_full_year_are_reported_not_compared(ran):
    text = (ran.out_dir / "answers" / "q11.md").read_text()
    assert "full calendar year" in text


def test_report_block_is_written(ran):
    assert (ran.out_dir / "reports" / "q11_summary.txt").exists()
