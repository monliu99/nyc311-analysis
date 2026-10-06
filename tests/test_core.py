import pandas as pd

from nyc311.config import Config
from nyc311.report import Report


def _cfg(tmp_path, *extra):
    return Config.from_args(
        ["--data", "x/*.json", "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"), *extra]
    )


def test_threads_follow_slurm_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    assert _cfg(tmp_path).threads == 7


def test_explicit_threads_override_slurm(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    assert _cfg(tmp_path, "--threads", "3").threads == 3


def test_config_keeps_paths_and_pattern(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.data_glob == "x/*.json"
    assert cfg.out_dir == tmp_path / "out"
    assert cfg.work_dir == tmp_path / "work"


def test_connect_applies_thread_setting(tmp_path):
    cfg = _cfg(tmp_path, "--threads", "2")
    con = cfg.connect()
    assert con.sql("select current_setting('threads')").fetchone()[0] == 2


def test_report_wraps_block_in_markers_and_saves(tmp_path, capsys):
    rep = Report("02a_files", tmp_path)
    rep.add("rows: 10")
    text = rep.emit()
    lines = text.splitlines()
    assert lines[0] == "===== BEGIN 02a_files ====="
    assert lines[-1] == "===== END 02a_files ====="
    assert (tmp_path / "reports" / "02a_files.txt").read_text() == text
    assert "rows: 10" in capsys.readouterr().out


def test_report_truncates_long_blocks(tmp_path):
    rep = Report("long", tmp_path, max_lines=10)
    for i in range(50):
        rep.add(f"line {i}")
    lines = rep.emit().splitlines()
    assert len(lines) == 12
    assert "41 more lines" in lines[-2]
    assert (tmp_path / "reports" / "long.txt").read_text().count("\n") == 51


def test_report_empty_table_says_none(tmp_path):
    rep = Report("empty", tmp_path)
    rep.table(pd.DataFrame({"a": []}), title="Nothing")
    text = rep.emit()
    assert "(none)" in text
    assert "Empty DataFrame" not in text


def test_report_left_aligns_text_columns(tmp_path):
    rep = Report("align", tmp_path)
    rep.table(pd.DataFrame({"name": ["a", "a much longer value"], "n": [1, 2]}))
    first_row = rep.emit().splitlines()[2]
    assert first_row.startswith("a ")


def test_report_table_says_how_many_rows_were_cut(tmp_path):
    rep = Report("tbl", tmp_path)
    rep.table(pd.DataFrame({"a": [1, 2, 3]}), max_rows=2, title="Counts")
    text = rep.emit()
    assert "Counts" in text
    assert "(2 of 3 rows shown)" in text
