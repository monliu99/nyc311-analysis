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
    assert len(reports) == 8
    assert (tmp_path / "outputs_bundle.zip").exists()


def test_notebook_cells_call_the_same_stages_as_run_all():
    nb = json.loads(Path("lab.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for call in ["env.run(cfg)", "ingest.run(cfg)", "discover.run(cfg)", "bundle.zip_outputs(cfg)"]:
        assert call in code, call
    assert re.search(r"%pip install .*-r requirements.txt", code)
