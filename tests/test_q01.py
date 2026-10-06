"""Q1 (missing values) on the 20-file sample: measures reconcile with the raw and clean tables."""
import re

import pandas as pd
import pytest

from nyc311 import clean, ingest, q01_missing as q1


@pytest.fixture(scope="module")
def q01(cleaned_cfg):
    out = q1.run(cleaned_cfg)
    con = cleaned_cfg.connect()
    raw = ingest.raw_source(cleaned_cfg)
    return cleaned_cfg, out, con, raw


def _one(con, sql):
    return con.sql(sql).fetchone()[0]


def test_every_raw_column_has_one_row_with_rates_between_0_and_100(q01):
    cfg, out, con, raw = q01
    cols = out["columns"].set_index("column")
    raw_cols = [c for c in con.sql(f"describe select * from {raw}").df().column_name if c != "_file"]
    assert sorted(cols.index) == sorted(raw_cols)
    for c in ["pct_stored", "pct_with_placeholders", "pct_missing_where_applies"]:
        assert cols[c].between(0, 100).all(), c
    assert (cols.pct_stored <= cols.pct_with_placeholders + 1e-9).all()
    assert (cols.n_applies <= cols.n_rows).all()
    assert (cols.n_rows == _one(con, f"select count(*) from {raw}")).all()


def test_stored_missing_is_null_or_empty_text(q01):
    cfg, out, con, raw = q01
    cols = out["columns"].set_index("column")
    for c in ["landmark", "due_date", "borough"]:
        expected = _one(con, f"select count_if({c} is null or trim({c}) = '') from {raw}")
        assert cols.loc[c, "n_null"] + cols.loc[c, "n_empty"] == expected


def test_each_encoding_is_listed_with_its_rows_and_they_add_up(q01):
    cfg, out, con, raw = q01
    enc, cols = out["encodings"], out["columns"].set_index("column")
    counted = enc[enc.counts_as_missing].groupby("column").n.sum()
    for c, row in cols.iterrows():
        assert counted.get(c, 0) == row.n_missing, c
    get = lambda col, value: int(enc[(enc.column == col) & (enc.value == value)].n.sum())
    assert get("borough", "Unspecified") == _one(con, f"select count_if(borough = 'Unspecified') from {raw}")
    assert get("open_data_channel_type", "UNKNOWN") == _one(
        con, f"select count_if(open_data_channel_type = 'UNKNOWN') from {raw}")
    assert get("community_board", "0 Unspecified") > 0
    partial = enc[(enc.column == "community_board") & (enc.kind == q1.KIND_PARTIAL)]
    assert set(partial.value) >= {"Unspecified QUEENS", "Unspecified BROOKLYN"}
    ph = enc[(enc.column == "closed_date") & (enc.kind == q1.KIND_DATE)]
    assert int(ph.n.sum()) == 367  # README: 367 close dates are 1900-01-01
    assert int(enc[(enc.column == "due_date") & (enc.kind == q1.KIND_DATE)].n.sum()) == 356


def test_invalid_zips_match_what_cleaning_set_to_null(q01):
    cfg, out, con, raw = q01
    cols = out["columns"].set_index("column")
    nulled = _one(con, f"select count_if(zip is null) from {clean.clean_glob(cfg)}")
    assert cols.loc["incident_zip", "n_missing"] == nulled
    enc = out["encodings"]
    zip_kinds = set(enc[enc.column == "incident_zip"].kind)
    assert q1.KIND_ZIP in zip_kinds and q1.KIND_TOKEN in zip_kinds


def test_time_of_day_missing_is_listed_but_not_counted_as_missing(q01):
    cfg, out, con, raw = q01
    enc = out["encodings"]
    mid = enc[(enc.column == "created_date") & (enc.kind == q1.KIND_DATE_ONLY)]
    assert len(mid) == 1 and not mid.counts_as_missing.iloc[0]
    assert int(mid.n.iloc[0]) == _one(con, f"select count_if(date_only_created) from {clean.clean_glob(cfg)}")


def test_intersection_streets_apply_only_to_intersection_addresses(q01):
    cfg, out, con, raw = q01
    cols = out["columns"].set_index("column")
    n_inter = _one(con, f"select count_if(address_type = 'INTERSECTION') from {raw}")
    r = cols.loc["intersection_street_1"]
    assert 0 < r.n_applies <= n_inter
    assert r.pct_stored > 50 and r.pct_missing_where_applies < 5


def test_close_date_applies_only_to_closed_requests(q01):
    cfg, out, con, raw = q01
    r = out["columns"].set_index("column").loc["closed_date"]
    assert r.n_applies <= _one(con, f"select count_if(trim(status) = 'Closed') from {raw}")
    # the sample's 1900 placeholders sit on open requests, so they don't count here; closed rows without a date do
    no_date = _one(con, f"""select count_if(trim(status) = 'Closed' and (closed_date is null
                             or trim(closed_date) = '' or closed_date like '1900-%')) from {raw}""")
    assert r.n_missing_where_applies == no_date
    assert r.pct_missing_where_applies < r.pct_with_placeholders


def test_type_specific_columns_apply_to_few_complaint_types(q01):
    cfg, out, con, raw = q01
    cols = out["columns"].set_index("column")
    n_types = _one(con, f"select count(distinct coalesce(complaint_type, '(null)')) from {raw}")
    for c in ["bridge_highway_name", "taxi_pick_up_location"]:
        assert 0 < cols.loc[c, "n_types_apply"] < 0.1 * n_types, c
        assert cols.loc[c, "pct_missing_where_applies"] < cols.loc[c, "pct_with_placeholders"]
    app = out["applicability"]
    assert app.fill_pct.between(0, 100).all()
    assert (app.applies == (app.fill_pct >= 100 * q1.APPLIES_MIN_FILL)).all()


def test_location_recovery_reconciles_with_the_clean_table(q01):
    cfg, out, con, raw = q01
    rec = out["recovery"]
    n = _one(con, f"select count(*) from {raw}")
    for field in ["borough", "community district", "zip"]:
        assert rec[rec.field == field].n.sum() == n, field
    b = rec[rec.field == "borough"].set_index("status").n
    raw_missing = _one(con, f"select count_if(borough is null or trim(borough) in ('', 'Unspecified')) from {raw}")
    assert b.drop(q1.NOT_MISSING).sum() == raw_missing
    still = _one(con, f"select count_if(borough is null) from {clean.clean_glob(cfg)}")
    assert b[q1.STILL_MISSING] == still
    assert still < 0.1 * raw_missing  # most Unspecified boroughs are recovered


def test_missingness_over_time_uses_periods_that_fit_the_data(q01):
    cfg, out, con, raw = q01
    t = out["by_period"]
    assert out["period_unit"] == "week"  # five weeks of data: weekly, not yearly
    assert (t.n >= q1.MIN_PERIOD_ROWS).all()
    pct = [c for c in t.columns if c.startswith("pct_")]
    assert len(pct) >= 8 and t[pct].stack().between(0, 100).all()


def test_outputs_and_answer_are_written_with_computed_numbers(q01):
    cfg, out, con, raw = q01
    for name in ["q01_columns", "q01_encodings", "q01_applicability", "q01_recovery", "q01_by_period"]:
        assert len(pd.read_csv(cfg.out_dir / "agg" / f"{name}.csv")) > 0, name
    text = (cfg.out_dir / "answers" / "q01.md").read_text()
    assert "{" not in text and not re.search(r"\bnan\b|\bNaT\b|\bNone\b", text)
    assert f"{_one(con, f'select count(*) from {raw}'):,}" in text
    report = (cfg.out_dir / "reports" / f"{q1.STAGE}.txt").read_text().splitlines()
    assert len(report) <= 152
