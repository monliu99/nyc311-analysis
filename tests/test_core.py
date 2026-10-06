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


def test_threads_default_is_capped_when_slurm_is_silent(monkeypatch, tmp_path):
    import os
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 128, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert _cfg(tmp_path).threads == 16


def test_memory_limit_follows_slurm_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16000")
    assert _cfg(tmp_path).memory_limit == "11GB"


def test_memory_limit_from_per_cpu_allocation(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4000")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert _cfg(tmp_path).memory_limit == "11GB"


def test_explicit_memory_limit_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16000")
    assert _cfg(tmp_path, "--memory-limit", "48GB").memory_limit == "48GB"


def test_cgroup_limit_is_read_and_unlimited_is_ignored(tmp_path):
    from nyc311 import config
    capped = tmp_path / "memory.max"
    capped.write_text("17179869184\n")
    unlimited = tmp_path / "unlimited"
    unlimited.write_text("max\n")
    assert config.cgroup_limit_bytes([unlimited, capped]) == 17179869184
    assert config.cgroup_limit_bytes([unlimited, tmp_path / "missing"]) is None


def test_report_shows_missing_text_as_null_left_aligned(tmp_path):
    rep = Report("nulls", tmp_path)
    rep.table(pd.DataFrame({"value": [None, "Closed"], "n": [5, 7]}))
    lines = rep.emit().splitlines()
    assert lines[2].startswith("NULL ")
    assert "NaN" not in "\n".join(lines) and "None" not in "\n".join(lines)


def test_connect_works_with_a_quote_in_the_work_path(tmp_path):
    cfg = Config.from_args(["--out", str(tmp_path / "o"), "--work", str(tmp_path / "it's work")])
    con = cfg.connect()
    assert con.sql("select 1").fetchone()[0] == 1


def test_paths_expand_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = Config.from_args(["--out", "~/o", "--work", "~/w"])
    assert cfg.out_dir == tmp_path / "o" and cfg.work_dir == tmp_path / "w"
