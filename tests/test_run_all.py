import json
import re
import zipfile
from pathlib import Path

import run_all
from nyc311 import bundle, env
from nyc311.config import Config


def test_env_block_reports_python_and_cpus(tmp_path):
    cfg = Config.from_args(["--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"), "--threads", "2"])
    text = env.run(cfg)
    assert "python" in text
    assert "threads used: 2" in text


def test_bundle_zips_outputs_for_download(tmp_path):
    cfg = Config.from_args(["--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")])
    (cfg.out_dir / "reports").mkdir(parents=True)
    (cfg.out_dir / "reports" / "x.txt").write_text("hi")
    path = bundle.zip_outputs(cfg)
    assert zipfile.ZipFile(path).namelist() == ["outputs/reports/x.txt"]


def test_run_all_is_one_command_for_every_stage(tmp_path):
    run_all.main(["--data", "data/sample/311_batch_*.json", "--out", str(tmp_path / "outputs"),
                  "--work", str(tmp_path / "work"), "--threads", "4"])
    reports = sorted(p.stem for p in (tmp_path / "outputs" / "reports").glob("*.txt"))
    assert reports[:3] == ["00_env", "01_ingest", "02a_files"]
    assert "03_clean" in reports
    assert (tmp_path / "outputs_bundle.zip").exists()


def test_every_stage_has_a_module_name():
    from nyc311.config import STAGE_MODULES, STAGES
    assert STAGES[:3] == ["ingest", "discover", "clean"]
    assert [s for s in STAGES if s.startswith("q")] == [f"q{i:02d}" for i in range(1, 12)]
    assert set(STAGES) == set(STAGE_MODULES)


def test_stages_not_built_yet_are_skipped_with_a_note(tmp_path, capsys, monkeypatch):
    from nyc311 import config
    monkeypatch.setitem(config.STAGE_MODULES, "q11", "nyc311.does_not_exist")
    run_all.run_stage("q11", None)
    assert "not built yet" in capsys.readouterr().out


def test_notebook_cells_call_the_same_stages_as_run_all():
    nb = json.loads(Path("lab.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for call in ["env.run(cfg)", "run_stage(", "config.STAGES", "bundle.zip_outputs(cfg)"]:
        assert call in code, call
    assert re.search(r"%pip install .*-r requirements.txt", code)


def test_env_block_reports_what_data_matches_and_how_workers_start(tmp_path):
    cfg = Config.from_args(["--data", "data/sample/311_batch_000[1-3].json", "--out", str(tmp_path / "out"),
                            "--work", str(tmp_path / "work"), "--threads", "2"])
    text = env.run(cfg)
    assert re.search(r"files matching DATA: 3 \(0\.04 GB; 311_batch_0001\.json \.\. 311_batch_0003\.json\)", text)
    assert "process start method:" in text
    assert "CPUs this job may use:" in text


def test_env_block_warns_when_data_matches_nothing(tmp_path):
    cfg = Config.from_args(["--data", str(tmp_path / "nope" / "*.json"), "--out", str(tmp_path / "out"),
                            "--work", str(tmp_path / "work")])
    assert "WARNING: no files match DATA" in env.run(cfg)


def test_env_block_warns_when_work_disk_is_too_small(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**12, 10**12 - 10**6, 10**6))
    cfg = Config.from_args(["--data", "data/sample/311_batch_000[1-3].json", "--out", str(tmp_path / "out"),
                            "--work", str(tmp_path / "work")])
    assert "WARNING: WORK has" in env.run(cfg)
