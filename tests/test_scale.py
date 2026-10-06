"""A small multi-year pull with dozens of problem files, built by tools/make_synthetic_batches.py.

Checks that ingest + discovery survive every planted problem with consistent counts, and that every report
block fits in its 150-line cap with the important sections printed (nothing cut off at the end).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from nyc311 import discover, ingest
from nyc311.config import Config

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import make_synthetic_batches as synth  # noqa: E402

N_FILES, ROWS, REPEATS = 120, 150, 3


@pytest.fixture(scope="module")
def pull(tmp_path_factory):
    root = tmp_path_factory.mktemp("pull")
    synth.main(["--files", str(N_FILES), "--rows-per-file", str(ROWS), "--problem-repeats", str(REPEATS),
                "--out", str(root / "data")])
    planted = json.loads((root / "planted.json").read_text())
    cfg = Config.from_args(["--data", str(root / "data" / "311_batch_*.json"), "--out", str(root / "out"),
                            "--work", str(root / "work"), "--threads", "2", "--memory-limit", "2GB"])
    ingest.run(cfg)
    discover.run(cfg)
    return cfg, planted


def _block(cfg, name):
    return (cfg.out_dir / "reports" / f"{name}.txt").read_text()


def _roles(planted):
    roles = pd.Series(planted["planted"]).value_counts()
    return roles


def test_every_planted_problem_file_is_in_the_manifest_with_the_right_kind(pull):
    cfg, planted = pull
    m = pd.read_parquet(cfg.work_dir / "manifest.parquet").set_index("file")
    kinds = m.error.map(ingest.problem_kind)
    for f, role in planted["planted"].items():
        expected = {"truncated": "cut off", "top_level_object": "not an array", "zero_bytes": "empty file",
                    "bad_utf8": "invalid UTF-8"}.get(role)
        if expected:
            assert kinds[f].startswith(expected), (f, role, m.loc[f, "error"])
        elif role in ("empty_array", "extra_column", "missing_column", "bom", "identical_dups", "conflicting_dups"):
            assert m.loc[f, "parse_ok"], (f, role, m.loc[f, "error"])
    assert (m.n_parquet == m.n_records).all()


def test_rows_in_parquet_match_the_manifest(pull):
    cfg, _ = pull
    m = pd.read_parquet(cfg.work_dir / "manifest.parquet")
    con = cfg.connect()
    assert con.sql(f"select count(*) from {ingest.raw_source(cfg)}").fetchone()[0] == m.n_records.sum()
    cols = con.sql(f"describe select * from {ingest.raw_source(cfg)}").df().column_name.tolist()
    assert ":@computed_region_efsh_h5xi" in cols and "open_data_channel_type" in cols


def test_duplicates_from_overlapping_pages_and_redownloads_are_found(pull):
    cfg, planted = pull
    con = cfg.connect()
    d = discover.key_duplicates(con, ingest.raw_glob(cfg))
    assert d["identical_copies"] >= REPEATS * ROWS  # each re-downloaded file repeats every key
    assert d["conflicting_versions"] >= REPEATS


def test_every_block_fits_its_cap_without_cutting(pull):
    cfg, _ = pull
    for p in sorted((cfg.out_dir / "reports").glob("0*.txt")):
        n = len(p.read_text().splitlines())
        assert n <= 152, f"{p.name} has {n} lines; the printed copy would be cut"


def test_ingest_block_summarises_problem_files_first(pull):
    cfg, planted = pull
    text = _block(cfg, "01_ingest")
    kinds = text.index("## Problem files by kind")
    assert kinds < text.index("## Problem files\n")
    roles = _roles(planted)
    assert re.search(rf"cut off / malformed .*\s{roles['truncated']}\s", text)


def test_files_block_lists_missing_batch_numbers_and_odd_names(pull):
    cfg, planted = pull
    text = _block(cfg, "02a_files")
    nums = planted["missing_numbers"]
    assert f"{len(nums)} numbers missing" in text
    assert re.search(rf"missing: .*\b{nums[0]}\b", text)
    assert "(1).json" in text  # the re-downloaded copies don't follow the naming pattern
    assert "Days from the file's median created date" in text


def test_coverage_block_shows_every_year(pull):
    cfg, _ = pull
    text = _block(cfg, "02b_coverage")
    for year in range(2010, 2018):
        assert re.search(rf"^\s*{year}\s", text, re.M), year
    assert "1900" not in text.split("Requests created per month")[1].split("##")[0]  # stray rows stay out


def test_values_block_reports_spelling_groups(pull):
    cfg, _ = pull
    text = _block(cfg, "02d_values")
    assert "groups of spellings" in text and "Largest spelling groups" in text


def test_second_ingest_of_more_files_matches_one_run(pull, tmp_path):
    cfg, _ = pull
    files = ingest.match_files(cfg.data_glob)
    half = tmp_path / "half"
    half.mkdir()
    for f in files[: len(files) // 2]:
        (half / Path(f).name).symlink_to(f)
    inc = Config.from_args(["--data", str(half / "311_batch_*.json"), "--out", str(tmp_path / "o"),
                            "--work", str(tmp_path / "w"), "--threads", "2"])
    ingest.run(inc)
    for f in files[len(files) // 2:]:
        (half / Path(f).name).symlink_to(f)
    ingest.run(inc)
    con = inc.connect()
    con1 = cfg.connect()
    one = con1.sql(f"select count(*), count(distinct _file), count(open_data_channel_type) "
                            f"from {ingest.raw_source(cfg)}").fetchone()
    two = con.sql(f"select count(*), count(distinct _file), count(open_data_channel_type) "
                  f"from {ingest.raw_source(inc)}").fetchone()
    assert one == two
