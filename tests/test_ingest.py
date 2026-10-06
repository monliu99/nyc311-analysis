import pandas as pd

from nyc311 import ingest


def test_recover_records_keeps_complete_objects_before_the_cut():
    text = '[{"a":"1"}\n,{"a":"2"}\n,{"a":"3", "b": "trunc'
    assert ingest.recover_records(text) == [{"a": "1"}, {"a": "2"}]


def test_recover_records_handles_text_without_array():
    assert ingest.recover_records("garbage") == []


def test_manifest_flags_only_batch_0010_as_truncated(ingested):
    m = ingested["manifest"]
    assert len(m) == 20
    bad = m[~m.parse_ok]
    assert bad.file.tolist() == ["311_batch_0010.json"]
    assert bad.n_records.item() == 6270


def test_manifest_counts_all_records(ingested):
    assert ingested["manifest"].n_records.sum() == 196_270


def test_manifest_records_created_range_per_file(ingested):
    row = ingested["manifest"].set_index("file").loc["311_batch_0001.json"]
    assert row.min_created == "2010-01-01T00:00:00.000"
    assert row.max_created == "2015-10-01T00:00:00.000"


def test_raw_parquet_has_every_record_as_text(sample_cfg, ingested):
    con = sample_cfg.connect()
    raw = ingest.raw_glob(sample_cfg)
    assert con.sql(f"select count(*) from read_parquet('{raw}')").fetchone()[0] == 196_270
    types = con.sql(f"describe select * from read_parquet('{raw}')").df()
    assert len(types) == 42
    assert set(types.column_type) == {"VARCHAR"}


def test_raw_parquet_tags_rows_with_source_file(sample_cfg, ingested):
    con = sample_cfg.connect()
    raw = ingest.raw_glob(sample_cfg)
    per_file = con.sql(f"select _file, count(*) n from read_parquet('{raw}') group by 1").df()
    assert len(per_file) == 20
    assert per_file.set_index("_file").n["311_batch_0010.json"] == 6270


def test_location_object_survives_as_json_text(sample_cfg, ingested):
    con = sample_cfg.connect()
    raw = ingest.raw_glob(sample_cfg)
    loc = con.sql(f"select location from read_parquet('{raw}') where location is not null limit 1").fetchone()[0]
    assert '"latitude"' in loc


def test_second_run_skips_known_files(sample_cfg, ingested):
    again = ingest.run(sample_cfg)
    assert again["new_files"] == 0
    assert len(again["manifest"]) == 20
    assert isinstance(again["manifest"], pd.DataFrame)


def test_ingest_writes_downloadable_manifest(sample_cfg, ingested):
    assert (sample_cfg.out_dir / "agg" / "manifest.csv").exists()


def test_scan_falls_back_to_one_process_when_pool_breaks(monkeypatch, tmp_path):
    from concurrent.futures.process import BrokenProcessPool

    from nyc311.config import Config

    class Broken:
        def __init__(self, *a, **k):
            raise BrokenProcessPool("no workers on this node")

    monkeypatch.setattr(ingest, "ProcessPoolExecutor", Broken)
    cfg = Config.from_args(["--data", "data/sample/311_batch_000[12].json", "--out", str(tmp_path / "o"),
                            "--work", str(tmp_path / "w"), "--threads", "2"])
    assert ingest.run(cfg)["manifest"].n_records.sum() == 20_000
