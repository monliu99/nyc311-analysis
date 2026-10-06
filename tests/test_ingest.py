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


# ---- Problem files a 2,000-file pull can contain (each must be recorded, never crash ingest) ----
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from nyc311.config import Config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _rec(key, created="2010-01-01T08:00:00.000", **extra):
    r = {"unique_key": str(key), "created_date": created, "agency": "HPD", "complaint_type": "HEATING",
         "status": "Closed", "location": {"latitude": "40.8", "longitude": "-73.9"}}
    r.update(extra)
    return r


def _batch(path, keys, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[" + "\n,".join(json.dumps(_rec(k, **extra)) for k in keys) + "]\n")


def _cfg(tmp_path, data=None, pattern="311_batch_*.json"):
    data = data or tmp_path / "data"
    return Config.from_args(["--data", str(Path(data) / pattern), "--out", str(tmp_path / "out"),
                             "--work", str(tmp_path / "work"), "--threads", "2", "--memory-limit", "1GB"])


def _rows(cfg, where="true"):
    con = cfg.connect()  # keep a reference: DuckDB < 1.5 closes a connection that is only a temporary
    return con.sql(f"select count(*) from {ingest.raw_source(cfg)} where {where}").fetchone()[0]


def _manifest(cfg):
    return pd.read_parquet(cfg.work_dir / "manifest.parquet").set_index("file")


def test_top_level_object_is_recorded_not_a_crash(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(3))
    (d / "311_batch_0002.json").write_text('{"error": true, "message": "rate limit [1, 2]", "data": {"x": [3, 4]}}')
    cfg = _cfg(tmp_path)
    ingest.run(cfg)
    row = _manifest(cfg).loc["311_batch_0002.json"]
    assert not row.parse_ok and row.n_records == 0 and "array" in row.error
    assert _rows(cfg) == 3


def test_non_object_array_elements_are_dropped_and_counted(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "311_batch_0001.json").write_text('[{"unique_key": "1", "created_date": "2010-01-01"}, 5, "ab", [1]]')
    cfg = _cfg(tmp_path)
    m = ingest.run(cfg)["manifest"].set_index("file")
    assert m.loc["311_batch_0001.json", "n_records"] == 1
    assert "3 array elements" in m.loc["311_batch_0001.json", "error"]
    assert m.loc["311_batch_0001.json", "keys"] == "created_date|unique_key"
    assert _rows(cfg) == 1


def test_zero_byte_and_empty_array_files_are_recorded(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(4))
    (d / "311_batch_0002.json").write_bytes(b"")
    (d / "311_batch_0003.json").write_text("[]\n")
    cfg = _cfg(tmp_path)
    m = ingest.run(cfg)["manifest"].set_index("file")
    assert m.loc["311_batch_0002.json", "n_records"] == 0 and not m.loc["311_batch_0002.json", "parse_ok"]
    assert "empty file" in m.loc["311_batch_0002.json", "error"]
    assert m.loc["311_batch_0003.json", "n_records"] == 0 and m.loc["311_batch_0003.json", "parse_ok"]
    assert _rows(cfg) == 4


def test_truncated_file_keeps_records_before_the_cut(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(10))
    text = (d / "311_batch_0001.json").read_text()
    cut = text.index('"unique_key": "7"') + 5
    (d / "311_batch_0001.json").write_text(text[:cut])
    cfg = _cfg(tmp_path)
    m = ingest.run(cfg)["manifest"].set_index("file")
    assert m.loc["311_batch_0001.json", "n_records"] == 7
    assert _rows(cfg) == 7
    assert _rows(cfg, "_file = '311_batch_0001.json'") == 7


def test_truncated_file_that_starts_with_an_object_recovers_nothing(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "311_batch_0001.json").write_text('{"meta": {"view": {"columns": [{"name": "unique_key"}, {"name": "x"')
    cfg = _cfg(tmp_path)
    m = ingest.run(cfg)["manifest"].set_index("file")
    assert m.loc["311_batch_0001.json", "n_records"] == 0


def test_utf8_bom_file_is_read_normally(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(5))
    p = d / "311_batch_0001.json"
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    cfg = _cfg(tmp_path)
    m = ingest.run(cfg)["manifest"].set_index("file")
    assert m.loc["311_batch_0001.json", "n_records"] == 5
    assert _rows(cfg) == 5


def test_invalid_utf8_byte_does_not_stop_the_chunk(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(5))
    _batch(d / "311_batch_0002.json", range(5, 8), incident_address="100 CAFE STREET")
    p = d / "311_batch_0002.json"
    p.write_bytes(p.read_bytes().replace(b"CAFE", b"CAF\xe9"))
    cfg = _cfg(tmp_path)
    m = ingest.run(cfg)["manifest"].set_index("file")
    assert m.loc["311_batch_0002.json", "n_records"] == 3
    assert "UTF-8" in m.loc["311_batch_0002.json", "error"]
    assert _rows(cfg) == 8


def test_unreadable_file_is_recorded(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can read anything")
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(2))
    _batch(d / "311_batch_0002.json", range(2, 4))
    (d / "311_batch_0002.json").chmod(0)
    try:
        cfg = _cfg(tmp_path)
        m = ingest.run(cfg)["manifest"].set_index("file")
    finally:
        (d / "311_batch_0002.json").chmod(0o644)
    assert m.loc["311_batch_0002.json", "n_records"] == 0 and "ermission" in m.loc["311_batch_0002.json", "error"]
    assert _rows(cfg) == 2
    again = ingest.run(cfg)  # readable now: retried, not skipped as already known
    assert again["new_files"] == 1 and _rows(cfg) == 4


def test_schema_drift_within_one_run_keeps_every_column(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(3))
    _batch(d / "311_batch_0002.json", range(3, 6), **{":@computed_region_efsh_h5xi": "12", "new_field": "y"})
    p = d / "311_batch_0003.json"
    p.write_text(json.dumps([{"unique_key": "9", "created_date": "2010-01-02T00:00:00.000"}]))
    cfg = _cfg(tmp_path)
    ingest.run(cfg)
    con = cfg.connect()
    src = ingest.raw_source(cfg)
    assert con.sql(f'select count("new_field"), count(":@computed_region_efsh_h5xi") from {src}').fetchone() == (3, 3)
    assert con.sql(f"select count(*), count(agency) from {src}").fetchone() == (7, 6)


@pytest.mark.parametrize("new_col", ["aaa_new", "zzz_new"])
def test_schema_drift_across_runs_keeps_new_and_old_columns(tmp_path, new_col):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(2))
    cfg = _cfg(tmp_path)
    ingest.run(cfg)
    _batch(d / "311_batch_0002.json", range(2, 4), **{new_col: "x"})
    ingest.run(cfg)
    con = cfg.connect()
    assert con.sql(f"select count(*), count({new_col}), count(agency) from {ingest.raw_source(cfg)}").fetchone() == (4, 2, 4)


def test_paths_with_spaces_brackets_quotes_and_ampersands(tmp_path):
    d = tmp_path / "it's a [test] & data"
    _batch(d / "311_batch_0001.json", range(3))
    _batch(d / "311_batch_0001 (1).json", range(3))
    cfg = _cfg(tmp_path, data=d)
    m = ingest.run(cfg)["manifest"]
    assert sorted(m.file) == ["311_batch_0001 (1).json", "311_batch_0001.json"]
    assert _rows(cfg) == 6


def test_home_shorthand_in_data_pattern_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _batch(tmp_path / "d" / "311_batch_0001.json", range(2))
    cfg = _cfg(tmp_path, data="~/d")
    assert ingest.run(cfg)["manifest"].n_records.sum() == 2


def test_same_file_name_in_two_folders_is_refused(tmp_path):
    _batch(tmp_path / "data" / "a" / "311_batch_0001.json", range(2))
    _batch(tmp_path / "data" / "b" / "311_batch_0001.json", range(2, 4))
    cfg = _cfg(tmp_path, pattern="*/311_batch_*.json")
    with pytest.raises(ValueError, match="same name"):
        ingest.run(cfg)


def test_manifest_records_rows_landed_in_parquet_per_file(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(3))
    _batch(d / "311_batch_0002.json", range(3, 8))
    cfg = _cfg(tmp_path)
    m = ingest.run(cfg)["manifest"]
    assert (m.n_parquet == m.n_records).all()
    assert m.part.str.endswith(".parquet").all()


def _five_files(tmp_path):
    d = tmp_path / "data"
    for i in range(5):
        _batch(d / f"311_batch_{i + 1:04d}.json", range(10 * i, 10 * i + 10))
    return _cfg(tmp_path)


def test_crash_while_writing_parquet_leaves_a_consistent_manifest(tmp_path, monkeypatch):
    cfg = _five_files(tmp_path)
    monkeypatch.setattr(ingest, "CHUNK_FILES", 2)
    real, calls = ingest._write_part, []

    def dies_on_second_chunk(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("node lost")
        return real(*a, **k)

    monkeypatch.setattr(ingest, "_write_part", dies_on_second_chunk)
    with pytest.raises(RuntimeError, match="node lost"):
        ingest.run(cfg)
    m = pd.read_parquet(cfg.work_dir / "manifest.parquet")
    assert len(m) == 2 and _rows(cfg) == 20  # only what is really in parquet
    monkeypatch.setattr(ingest, "_write_part", real)
    again = ingest.run(cfg)
    assert again["new_files"] == 3
    assert _rows(cfg) == 50
    con = cfg.connect()
    assert con.sql(f"select count(distinct unique_key) from {ingest.raw_source(cfg)}").fetchone()[0] == 50


def test_crash_between_parquet_and_manifest_does_not_double_count(tmp_path, monkeypatch):
    cfg = _five_files(tmp_path)
    monkeypatch.setattr(ingest, "CHUNK_FILES", 2)
    real, calls = ingest._save_manifest, []

    def dies_on_second_save(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(ingest, "_save_manifest", dies_on_second_save)
    with pytest.raises(KeyboardInterrupt):
        ingest.run(cfg)
    monkeypatch.setattr(ingest, "_save_manifest", real)
    ingest.run(cfg)
    assert _rows(cfg) == 50
    assert len(list((cfg.work_dir / "raw").glob("*.parquet"))) == 3


def _make_legacy(cfg):
    """Turn a work dir into what the first committed ingest wrote: no part/n_parquet columns, parts named
    part_<first file stem>_<number of files>.parquet, every part written before the manifest."""
    m = pd.read_parquet(cfg.work_dir / "manifest.parquet")
    for i, part in enumerate(sorted(m.part.dropna().unique())):
        files = m[m.part == part]
        (cfg.work_dir / "raw" / part).rename(cfg.work_dir / "raw" / f"part_{Path(files.file.iloc[0]).stem}_{len(files)}.parquet")
    m.drop(columns=["part", "n_parquet"]).to_parquet(cfg.work_dir / "manifest.parquet", index=False)


def test_work_dir_from_the_first_ingest_is_upgraded_without_rescanning(tmp_path, monkeypatch):
    cfg = _five_files(tmp_path)
    monkeypatch.setattr(ingest, "CHUNK_FILES", 2)
    ingest.run(cfg)
    _make_legacy(cfg)
    out = ingest.run(cfg)
    assert out["new_files"] == 0 and _rows(cfg) == 50  # every file known: nothing rescanned
    m = out["manifest"]
    assert (m.n_parquet == m.n_records).all() and m.part.str.startswith("part_311_batch_").all()
    assert sorted(p.name for p in (cfg.work_dir / "raw").glob("*.parquet")) == sorted(m.part.unique())


def test_upgraded_legacy_work_dir_takes_new_files(tmp_path):
    cfg = _five_files(tmp_path)
    ingest.run(cfg)
    _make_legacy(cfg)
    _batch(tmp_path / "data" / "311_batch_0006.json", range(50, 60))
    out = ingest.run(cfg)
    assert out["new_files"] == 1 and _rows(cfg) == 60


def test_legacy_parts_without_a_manifest_are_redone(tmp_path):
    cfg = _five_files(tmp_path)
    ingest.run(cfg)
    _make_legacy(cfg)
    (cfg.work_dir / "manifest.parquet").unlink()  # the first ingest died before writing its manifest
    out = ingest.run(cfg)
    assert out["new_files"] == 5 and _rows(cfg) == 50


def test_report_groups_bad_files_by_problem(tmp_path):
    d = tmp_path / "data"
    _batch(d / "311_batch_0001.json", range(3))
    for i in range(2, 40):
        (d / f"311_batch_{i:04d}.json").write_bytes(b"")
    (d / "311_batch_0040.json").write_text('[{"unique_key": "77", "created_')
    cfg = _cfg(tmp_path)
    ingest.run(cfg)
    text = (cfg.out_dir / "reports" / "01_ingest.txt").read_text()
    lines = text.splitlines()
    assert len(lines) <= 152
    kinds = text.index("## Problem files by kind")
    assert "empty file" in text[kinds:kinds + 400] and "38" in text[kinds:kinds + 400]
    assert "311_batch_0040.json" in text  # the truncated file is listed before the empty ones


START_METHOD_SCRIPT = """
import multiprocessing as mp, sys
mp.set_start_method({method!r}, force=True)
sys.path.insert(0, {repo!r})
from nyc311 import ingest
from nyc311.config import Config
cfg = Config.from_args(["--data", {data!r}, "--work", "w", "--out", "o", "--threads", "2"])
print("ROWS", ingest.run(cfg)["manifest"].n_records.sum())
"""


@pytest.mark.parametrize("method", ["spawn", "forkserver", "fork"])
def test_ingest_runs_under_each_start_method_from_python_c(tmp_path, method):
    import multiprocessing as mp
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} not available here")
    _batch(tmp_path / "data" / "311_batch_0001.json", range(3))
    _batch(tmp_path / "data" / "311_batch_0002.json", range(3, 7))
    run_dir = tmp_path / "elsewhere"
    run_dir.mkdir()
    code = START_METHOD_SCRIPT.format(method=method, repo=str(REPO), data=str(tmp_path / "data" / "311_batch_*.json"))
    r = subprocess.run([sys.executable, "-c", code], cwd=run_dir, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ROWS 7" in r.stdout
    assert "one at a time" not in r.stdout  # the pool really worked


def test_discovery_reads_a_work_dir_from_the_first_ingest_as_is(tmp_path):
    from nyc311 import discover

    cfg = _five_files(tmp_path)
    ingest.run(cfg)
    _make_legacy(cfg)
    discover.run(cfg)
    assert "rows in raw parquet: 50" in (cfg.out_dir / "reports" / "02a_files.txt").read_text()
