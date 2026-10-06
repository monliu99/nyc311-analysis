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


def test_long_empty_stretches_are_reported_not_listed_daily(ctx):
    con, raw, _ = ctx
    stretches = discover.coverage(con, raw)["empty_stretches"]
    assert (stretches.days >= 7).all()
    assert len(stretches) == 2


def test_closed_before_created_count(ctx):
    con, raw, _ = ctx
    assert discover.timestamp_checks(con, raw)["closed_before_created"] == 3288


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
