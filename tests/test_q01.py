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


# ---------------------------------------------------------------- answer text for the client report

def test_answer_opens_with_a_neutral_note_on_rows_and_period(q01):
    cfg, out, con, raw = q01
    lines = (cfg.out_dir / "answers" / "q01.md").read_text().splitlines()
    note = lines[2]
    n = _one(con, f"select count(*) from {raw}")
    first, last = con.sql(f"""select min(created)::date, max(created)::date from {clean.clean_glob(cfg)}
                              where created is not null and not out_of_file_order""").fetchone()
    assert note.startswith("*Computed by `nyc311/q01_missing.py` from ") and note.endswith("*")
    assert f"{n:,}" in note and f"{first:%Y-%m-%d}" in note and f"{last:%Y-%m-%d}" in note
    text = "\n".join(lines)
    assert "write the prose" not in text and "Drafted" not in text and "Read the last column" not in text


def test_no_district_rule_is_shared_and_matches_the_recovery_table(q01):
    """'No community district' = neither a standard district nor a park/airport area, over all rows; Q11 imports
    NO_DISTRICT_SQL so both report the same share."""
    cfg, out, con, raw = q01
    src = clean.clean_glob(cfg)
    assert q1.NO_DISTRICT_SQL == "boro_cd is null and special_area_cd is null"
    rec = out["recovery"]
    still = rec[(rec.field == "community district") & (rec.status == q1.STILL_MISSING)].n.sum()
    assert still == _one(con, f"select count_if({q1.NO_DISTRICT_SQL}) from {src}")
    parks = _one(con, f"select count_if(boro_cd is null and special_area_cd is not null) from {src}")
    assert parks > 0
    text = (cfg.out_dir / "answers" / "q01.md").read_text()
    pct = 100 * still / _one(con, f"select count(*) from {src}")
    assert f"{still:,} ({pct:.1f}%)" in text and f"{parks:,}" in text


@pytest.mark.parametrize("x, shown", [(0, "0.0%"), (0.004, "<0.1%"), (0.05, "0.1%"), (12.345, "12.3%"),
                                      (99.96, ">99.9%"), (100, "100.0%")])
def test_percentages_never_round_a_nonzero_share_to_zero_or_a_partial_one_to_100(x, shown):
    assert q1._f(x) == shown


def test_encoding_text_formats_value_counts_with_separators():
    enc = pd.DataFrame({"column": "incident_zip", "kind": q1.KIND_ZIP, "value": [f"0{i:04d}" for i in range(1500)],
                        "n": 2, "counts_as_missing": True})
    assert "invalid ZIP 3,000 (1,500 values, e.g. 00000, 00001)" in q1._encoding_text(enc)


def test_lasting_changes_find_drops_and_rises():
    per = pd.DataFrame({"period": ["2010", "2011", "2012", "2013"],
                        "pct_borough_raw": [30.0, 10.0, 5.0, 1.0],      # falls below half from 2011
                        "pct_address_type": [4.0, 5.0, 9.0, 12.0],     # above double from 2012
                        "pct_zip": [6.0, 7.0, 6.5, 2.0],               # below half only in the last period
                        "pct_closed_missing": [0.0, 0.01, 0.0, 0.0]})  # never visible at one decimal
    d = q1.lasting_drops(per).set_index("measure")
    lab = q1.PERIOD_MEASURES
    assert d.loc[lab["pct_borough_raw"], "drop_from"] == "2011"
    assert d.loc[lab["pct_address_type"], "rise_from"] == "2012" and d.loc[lab["pct_address_type"], "drop_from"] is None
    tab, tiny = q1.time_table(d.reset_index(), "year")
    assert lab["pct_closed_missing"] in tiny and lab["pct_closed_missing"] not in set(tab.field)
    change = tab.set_index("field")["lasting change"]
    assert change[lab["pct_borough_raw"]] == "below half from 2011"
    assert change[lab["pct_address_type"]] == "above double from 2012"
    assert change[lab["pct_zip"]] == "below half in 2013 only (last year)"
