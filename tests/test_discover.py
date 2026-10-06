import pandas as pd
import pytest

from nyc311 import discover, ingest


@pytest.fixture(scope="module")
def ctx(sample_cfg, ingested):
    return sample_cfg.connect(), ingest.raw_glob(sample_cfg), ingested["manifest"]


def test_rows_far_from_their_file_window_are_found(ctx):
    con, raw, _ = ctx
    out = discover.out_of_window_rows(con, raw)
    assert out.n_outside.sum() == 43
    assert out[out.n_outside > 0].file.tolist() == ["311_batch_0001.json"]


def test_no_duplicate_keys_in_sample(ctx):
    con, raw, _ = ctx
    assert discover.key_duplicates(con, raw)["dup_keys"] == 0


def test_low_volume_day_is_flagged(ctx):
    con, raw, _ = ctx
    cov = discover.coverage(con, raw)
    assert "2010-01-17" in cov["low_days"].date.astype(str).tolist()


def test_stray_rows_do_not_create_empty_stretches(ctx):
    con, raw, _ = ctx
    assert len(discover.coverage(con, raw)["empty_stretches"]) == 0  # the 2013/2015 rows used to add two


def test_long_empty_stretches_are_reported_not_listed_daily(tmp_path):
    import duckdb

    path = tmp_path / "gap.parquet"
    days = [f"2010-01-{d:02d}" for d in range(1, 11)] + [f"2010-02-{d:02d}" for d in range(1, 11)]
    rows = ", ".join(f"('{d}T08:00:00', 'HPD', 'f1')" for d in days for _ in range(5))
    duckdb.sql(f"""copy (select * from (values {rows}) t(created_date, agency, _file)) to '{path.as_posix()}' (format parquet)""")
    cov = discover.coverage(duckdb.connect(), path.as_posix(), days=60)
    st = cov["empty_stretches"]
    assert len(st) == 1 and st.days.item() == 21 and str(st.start.item()) == "2010-01-11"


def test_closed_before_created_count_leaves_out_1900_placeholders(ctx):
    con, raw, _ = ctx
    ts = discover.timestamp_checks(con, raw)
    assert ts["closed_before_created"] == 3288 - 367  # the 367 closed_date 1900-01-01 rows are counted as closed_1900
    assert ts["closed_1900"] == 367
    assert ts["due_before_created"] == 404 - 356
    assert ts["due_1900"] == 356


def test_complaint_type_count_and_variant_groups(ctx):
    con, raw, _ = ctx
    vals = discover.value_lists(con, raw)
    assert vals["n_complaint_types"] == 174
    groups = vals["type_variants"]
    plumbing = groups[groups.key == "plumbing"].spellings.item()
    assert "PLUMBING" in plumbing and "Plumbing" in plumbing


def test_column_profile_counts_placeholders(ctx):
    con, raw, manifest = ctx
    prof = discover.column_profile(con, raw).set_index("column")
    assert len(prof) == 41
    assert prof.loc["borough", "n_placeholder"] == 89_164
    assert prof.loc["closed_date", "n_null"] == 3100
    assert prof.loc["facility_type", "n_placeholder"] > 0


def test_run_writes_report_blocks_and_aggregates(sample_cfg, ingested):
    discover.run(sample_cfg)
    reports = sorted(p.stem for p in (sample_cfg.out_dir / "reports").glob("02*.txt"))
    assert reports == ["02a_files", "02b_coverage", "02c_columns", "02d_values", "02e_timestamps", "02f_sparse"]
    agg = sample_cfg.out_dir / "agg"
    for name in ["column_profile.csv", "complaint_types_raw.csv", "daily_by_agency.csv"]:
        assert (agg / name).exists(), name


def test_duplicate_keys_split_into_identical_and_conflicting(tmp_path):
    import duckdb

    path = tmp_path / "dups.parquet"
    duckdb.sql("""copy (select * from (values ('1','x','f1'), ('1','x','f2'), ('2','y','f1'), ('2','z','f2'),
                                               ('3','w','f1')) t(unique_key, a, _file))
                  to '%s' (format parquet)""" % path.as_posix())
    assert discover.key_duplicates(duckdb.connect(), path.as_posix()) == {
        "dup_keys": 2, "dup_rows": 4, "identical_copies": 1, "conflicting_versions": 1}


def test_low_day_next_to_stray_rows_is_flagged(ctx):
    con, raw, _ = ctx
    cov = discover.coverage(con, raw)
    low = cov["low_days"].date.astype(str).tolist()
    assert "2010-02-03" in low  # 90 rows; zero days stretched out to 2015 by stray rows must not lower the median
    assert cov["daily"].index.max() < pd.Timestamp("2010-03-01")  # stray rows are not part of the date spine


def test_closed_after_last_created_uses_the_in_window_end(ctx):
    con, raw, _ = ctx
    ts = discover.timestamp_checks(con, raw)
    assert str(ts["last_in_window_created"]) == "2010-02-03 02:27:00"  # not the stray 2015-10-01 row
    assert ts["closed_after_last_created"] == 46_544


def test_column_profile_distinct_counts_are_exact(ctx):
    con, raw, _ = ctx
    prof = discover.column_profile(con, raw).set_index("column")
    assert prof.loc["unique_key", "n_distinct"] == 196_270
    assert prof.loc["complaint_type", "n_distinct"] == 174


def test_offsets_from_file_median_are_summarised_by_agency(ctx):
    con, raw, _ = ctx
    off = discover.window_offsets(con, raw).set_index("agency")
    assert off.loc["(all)", "n"] == 196_270
    assert off.loc["(all)", "beyond_30d"] == 43
    assert off.loc["DOB", "beyond_7d"] >= 1_089
    assert off.index[0] == "(all)"


def test_sparse_columns_tolerate_null_complaint_type(tmp_path):
    import duckdb

    path = tmp_path / "r.parquet"
    duckdb.sql(f"""copy (select * from (values (null, null), (null, 'x'), ('HEAT', null)) t(complaint_type, landmark))
                   to '{path.as_posix()}' (format parquet)""")
    con = duckdb.connect()
    prof = discover.column_profile(con, path.as_posix())
    out = discover.sparse_columns(con, path.as_posix(), prof, min_rows=1)
    assert "landmark" in out.column.tolist()


def test_batch_number_gaps_are_listed_as_ranges():
    files = [f"311_batch_{i:04d}.json" for i in [1, 2, 3, 4, 5, 8, 10, 11, 12, 20]]
    files += ["311_batch_0003 (1).json", "notes.json"]
    seq = discover.batch_sequence(files)
    assert seq["n_missing"] == 2 + 1 + 7
    assert seq["missing_ranges"] == ["6-7", "9", "13-19"]
    assert seq["odd_names"] == ["311_batch_0003 (1).json", "notes.json"]
    assert seq["first"] == 1 and seq["last"] == 20


def test_duplicate_keys_by_file_pair(tmp_path):
    import duckdb

    path = tmp_path / "dups.parquet"
    duckdb.sql("""copy (select * from (values ('1','x','f1'), ('1','x','f2'), ('2','y','f1'), ('2','z','f2'),
                                               ('3','w','f1'), ('4','q','f3'), ('4','q','f3')) t(unique_key, a, _file))
                  to '%s' (format parquet)""" % path.as_posix())
    pairs = discover.duplicate_file_pairs(duckdb.connect(), path.as_posix())
    top = pairs.iloc[0]
    assert (top.file_a, top.file_b, top.n_keys, top.identical, top.conflicting) == ("f1", "f2", 2, 1, 1)
    assert ((pairs.file_a == "f3") & (pairs.file_b == "f3")).any()  # repeated inside one file


def test_agency_table_shows_name_count_and_top_names(ctx):
    con, raw, _ = ctx
    ag = discover.value_lists(con, raw)["agency"].set_index("agency")
    dof = ag.loc["DOF"]
    assert dof.n_names == 17
    assert dof.top_names.count(" | ") == 2  # the three most common spellings, not a cut-off list
