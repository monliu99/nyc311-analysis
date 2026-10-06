import datetime as dt
import json

import pandas as pd
import pytest

from nyc311 import clean, geo


def test_community_board_gives_borough_district_or_special_area():
    out = clean.parse_community_board(
        pd.Series(["01 MANHATTAN", "18 BROOKLYN", "83 QUEENS", "Unspecified QUEENS", "0 Unspecified", None])
    ).set_index("community_board")
    assert out.loc["01 MANHATTAN"].tolist() == ["MANHATTAN", 101, None]
    assert out.loc["18 BROOKLYN"].tolist() == ["BROOKLYN", 318, None]
    assert out.loc["83 QUEENS"].tolist() == ["QUEENS", None, 483]
    assert out.loc["Unspecified QUEENS"].tolist() == ["QUEENS", None, None]
    assert out.loc["0 Unspecified"].tolist() == [None, None, None]


def test_zip_keeps_only_nyc_five_digit_codes():
    out = clean.clean_zip(pd.Series(["10467", "11226-1234", "33771-5351", "N/A", "0000", "10594", None]))
    assert out.tolist() == ["10467", "11226", None, None, None, None, None]


def test_points_are_assigned_to_the_polygon_that_contains_them(tmp_path):
    square = lambda x0, y0: [[[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1], [x0, y0]]]
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"boro_cd": 101, "is_jia": False},
         "geometry": {"type": "Polygon", "coordinates": square(0, 0)}},
        {"type": "Feature", "properties": {"boro_cd": 164, "is_jia": True},
         "geometry": {"type": "MultiPolygon", "coordinates": [square(5, 5)]}}]}
    path = tmp_path / "cd.geojson"
    path.write_text(json.dumps(gj))
    codes = geo.assign_districts(lat=[0.5, 5.5, 9.0], lon=[0.5, 5.5, 9.0], geojson_path=path)
    assert codes.tolist() == [101, 164, None]


@pytest.fixture(scope="module")
def cleaned(sample_cfg, ingested):
    clean.run(sample_cfg)
    con = sample_cfg.connect()
    return con, f"read_parquet('{clean.clean_path(sample_cfg).as_posix()}')"


def _count(cleaned, where):
    con, src = cleaned
    return con.sql(f"select count(*) from {src} where {where}").fetchone()[0]


def test_cleaning_flags_rows_instead_of_dropping_them(cleaned):
    assert _count(cleaned, "true") == 196_270


@pytest.mark.parametrize("flag, expected", [
    ("out_of_file_order", 43),
    ("closed_placeholder", 367),
    ("closed_before_created", 2_921),
    ("is_test", 6),
    ("unresolved", 6_435),
    ("status_conflict", 2_983),
    ("date_only_closed", 94_435),
    ("closed_at_creation", 8_600),
    ("date_only_created", 84_883),
])
def test_flag_counts_match_discovery(cleaned, flag, expected):
    assert _count(cleaned, flag) == expected


def test_durations_exist_only_for_closed_requests_closed_after_opening(cleaned):
    assert _count(cleaned, "duration_h < 0") == 0
    assert _count(cleaned, "duration_h is not null and (closed_before_created or status <> 'Closed')") == 0
    assert _count(cleaned, "duration_h is not null") > 180_000


def test_placeholders_become_null(cleaned):
    assert _count(cleaned, "channel = 'UNKNOWN' or facility_type = 'N/A' or borough = 'Unspecified'") == 0


def test_most_unspecified_boroughs_are_recovered(cleaned):
    assert _count(cleaned, "borough is null") < 5_000
    con, src = cleaned
    sources = set(con.sql(f"select distinct borough_source from {src} where borough_source is not null").df().borough_source)
    assert sources <= {"reported", "community_board", "latlon", "zip"}


def test_incomplete_days_cover_the_tail_and_the_truncated_file(cleaned):
    con, src = cleaned
    days = {pd.Timestamp(d).date() for d in
            con.sql(f"select distinct created::date d from {src} where possibly_incomplete_day").df().d}
    assert {dt.date(2010, 1, 17), dt.date(2010, 2, 2), dt.date(2010, 2, 3)} <= days
    assert dt.date(2010, 2, 1) not in days


def test_complaint_types_get_taxonomy_columns(cleaned):
    con, src = cleaned
    assert _count(cleaned, "canonical_type is null") == 0
    origins = set(con.sql(f"select distinct origin from {src}").df().origin)
    assert origins <= {"city", "private", "natural", "other"}


def test_placeholder_close_dates_get_no_other_close_flags(cleaned):
    assert _count(cleaned, "closed_placeholder and (status_conflict or date_only_closed or closed_at_creation)") == 0


def test_special_areas_only_where_no_standard_district(cleaned):
    assert _count(cleaned, "boro_cd is not null and special_area_cd is not null") == 0


def test_clean_reads_raw_parts_with_different_columns(tmp_path, sample_cfg, ingested):
    import shutil

    import duckdb

    from nyc311 import ingest
    from nyc311.config import Config
    cfg = Config.from_args(["--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"), "--threads", "2"])
    (cfg.work_dir / "raw").mkdir(parents=True)
    shutil.copy(sample_cfg.work_dir / "manifest.parquet", cfg.work_dir / "manifest.parquet")
    src = ingest.raw_glob(sample_cfg)
    duckdb.sql(f"copy (select * exclude (landmark) from read_parquet('{src}') where _file < '311_batch_0010.json') "
               f"to '{(cfg.work_dir / 'raw' / 'a.parquet').as_posix()}' (format parquet)")
    duckdb.sql(f"copy (select * from read_parquet('{src}') where _file >= '311_batch_0010.json') "
               f"to '{(cfg.work_dir / 'raw' / 'b.parquet').as_posix()}' (format parquet)")
    clean.run(cfg)
    con = cfg.connect()
    out = con.sql(f"select count(*), count(landmark) from {clean.clean_glob(cfg)}").fetchone()
    assert out[0] == 196_270
    assert out[1] > 0


def test_days_covered_by_a_missing_batch_are_flagged(tmp_path, sample_cfg, ingested):
    import duckdb

    from nyc311 import ingest
    manifest = ingested["manifest"]
    manifest = manifest[manifest.file != "311_batch_0010.json"]
    part = tmp_path / "raw.parquet"
    duckdb.sql(f"copy (select * from read_parquet('{ingest.raw_glob(sample_cfg)}') "
               f"where _file <> '311_batch_0010.json') to '{part.as_posix()}' (format parquet)")
    con = duckdb.connect()
    days = clean.incomplete_days(con, f"read_parquet('{part.as_posix()}')", manifest)
    assert {dt.date(2010, 1, 15), dt.date(2010, 1, 16), dt.date(2010, 1, 17)} <= days
    assert dt.date(2010, 1, 13) not in days
    assert dt.date(2010, 1, 20) not in days


def test_unordered_neighbor_files_do_not_flag_years_of_days(tmp_path):
    import duckdb
    part = tmp_path / "raw.parquet"
    duckdb.sql(f"""copy (
        select '311_batch_0001.json' as _file, strftime(timestamp '2015-01-01' + to_days(i % 3), '%Y-%m-%dT%H:%M:%S.000') as created_date
        from range(100) t(i)
        union all
        select '311_batch_0003.json', strftime(timestamp '2015-01-01' + to_days(i * 14), '%Y-%m-%dT%H:%M:%S.000')
        from range(100) t(i)
        union all
        select '311_batch_0004.json', strftime(timestamp '2019-01-05' + to_days(i % 2), '%Y-%m-%dT%H:%M:%S.000')
        from range(100) t(i)
    ) to '{part.as_posix()}' (format parquet)""")
    manifest = pd.DataFrame({"file": ["311_batch_0001.json", "311_batch_0003.json", "311_batch_0004.json"],
                             "parse_ok": [True, True, True]})
    days = clean.incomplete_days(duckdb.connect(), f"read_parquet('{part.as_posix()}')", manifest)
    assert len(days) < 10


def test_any_open_status_counts_as_unresolved(cleaned):
    from nyc311 import rules
    assert "Closed" in rules.CLOSED_STATUSES and "Cancelled" in rules.NOT_UNRESOLVED_STATUSES
    sql = clean.unresolved_sql("s")
    con, _ = cleaned
    got = con.sql(f"""select s, {sql} u from (values ('Closed'), ('In Progress'), ('Email Sent'), ('Cancelled'),
                                                     ('Closed - Testing'), ('Draft'), (null), ('Pending')) t(s)""").df()
    assert dict(zip(got.s.fillna("NULL"), got.u)) == {
        "Closed": False, "In Progress": True, "Email Sent": True, "Cancelled": False,
        "Closed - Testing": False, "Draft": False, "NULL": False, "Pending": True}


def test_shared_cleaned_fixture_has_the_clean_table(cleaned_cfg):
    assert clean.clean_path(cleaned_cfg).exists()
