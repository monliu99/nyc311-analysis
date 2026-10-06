"""tools/make_upload_zip.py: the cluster upload holds exactly what the cluster needs, from the last commit."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "make_upload_zip.py"
TOP = "nyc311_lab/"
EXCLUDED_PREFIXES = ("tests/", "tools/", "brief/", "data/sample/", "work/", "outputs/", "cluster_outputs/", ".git/",
                     ".superpowers/", ".pytest_cache/")
ALLOWED_FILES = {"VERSION.txt", "run_all.py", "lab.ipynb", "requirements.txt", "README.md"}
ALLOWED_DIRS = ("nyc311/", "data/external/", "report/prose/")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True).stdout


def run_script(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def entries(zip_path):
    names = zipfile.ZipFile(zip_path).namelist()
    assert all(n.startswith(TOP) for n in names), [n for n in names if not n.startswith(TOP)]
    return {n[len(TOP):] for n in names}


def read(zip_path, rel):
    return zipfile.ZipFile(zip_path).read(TOP + rel).decode()


def assert_only_allowed(names):
    bad = [n for n in names if not (n in ALLOWED_FILES or n.startswith(ALLOWED_DIRS))
           or "__pycache__" in n or n.endswith((".pyc", ".zip")) or n.startswith(EXCLUDED_PREFIXES)]
    assert not bad, bad


# ---- the real repo: package HEAD into a temp folder (never touches the session folder) ----

@pytest.fixture(scope="module")
def real_zip(tmp_path_factory):
    out = tmp_path_factory.mktemp("upload") / "nyc311_lab_upload.zip"
    r = run_script(SCRIPT, "--out", str(out), "--allow-dirty")  # other work may be in progress; HEAD is packaged
    assert r.returncode == 0, r.stderr
    return out, r.stdout


def test_real_zip_has_everything_the_cluster_needs(real_zip):
    out, _ = real_zip
    names = entries(out)
    committed = git(REPO, "ls-tree", "-r", "--name-only", "--full-tree", "HEAD").splitlines()
    required = {"VERSION.txt", "run_all.py", "lab.ipynb", "requirements.txt", "README.md", "report/prose/README.md",
                "nyc311/__init__.py", "nyc311/config.py", "nyc311/nytplot.py", "nyc311/nyt.mplstyle",
                "nyc311/complaint_type_map.csv", "nyc311/build_report.py"}
    required |= {p for p in committed if p.startswith(("nyc311/", "data/external/"))}
    assert required <= names, sorted(required - names)
    assert any(n.startswith("data/external/") for n in names)


def test_real_zip_leaves_out_tests_tools_data_and_run_state(real_zip):
    out, _ = real_zip
    names = entries(out)
    assert_only_allowed(names)
    assert not {"pytest.ini", ".gitignore"} & names


def test_real_zip_version_names_the_head_commit_and_stages(real_zip):
    out, stdout = real_zip
    head = git(REPO, "rev-parse", "HEAD").strip()
    version = read(out, "VERSION.txt")
    assert f"commit:          {head}" in version
    assert "built:" in version and "commit date:" in version
    stages_line = next(line for line in version.splitlines() if line.startswith("stages present:"))
    for stage in ("ingest", "discover", "clean", "report"):
        assert stage in stages_line
    assert str(out) in stdout and "files" in stdout and "stages included:" in stdout


def test_real_zip_files_match_the_commit(real_zip):
    out, _ = real_zip
    z = zipfile.ZipFile(out)
    for rel in ("run_all.py", "nyc311/config.py", "requirements.txt"):
        committed = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True).stdout
        assert z.read(TOP + rel) == committed, rel


# ---- a throwaway repo in tmp_path, to simulate uncommitted work ----

FAKE_CONFIG = '''STAGE_MODULES = {
    "ingest": "nyc311.ingest",
    "q01": "nyc311.q01_missing",
    "q03": "nyc311.q03_complaint_types",
    "report": "nyc311.build_report",
}
'''
FAKE_FILES = {
    "nyc311/__init__.py": "",
    "nyc311/config.py": FAKE_CONFIG,
    "nyc311/ingest.py": "def run(cfg):\n    pass\n",
    "nyc311/q03_complaint_types.py": "def run(cfg):\n    pass\n",
    "nyc311/build_report.py": "def run(cfg):\n    pass\n",
    "nyc311/nytplot.py": "",
    "nyc311/nyt.mplstyle": "axes.grid: True\n",
    "nyc311/complaint_type_map.csv": "raw,canonical\n",
    "nyc311/__pycache__/config.cpython-39.pyc": "junk",
    "run_all.py": "print('run')\n",
    "lab.ipynb": "{}\n",
    "requirements.txt": "duckdb>=1.1\n",
    "README.md": "# lab\n",
    "data/external/weather.csv": "date,tmax\n",
    "data/sample/311_batch_0001.json": "[]",
    "report/prose/README.md": "how to write prose\n",
    "brief/brief.pdf": "pdf",
    "tests/test_x.py": "def test_x():\n    pass\n",
    "work/manifest.parquet": "x",
    "outputs/reports/00_env.txt": "x",
    "cluster_outputs/run1/x.txt": "x",
    "pytest.ini": "[pytest]\n",
    ".gitignore": "*.zip\n",
    "old_upload.zip": "zip",
}
FAKE_EXPECTED = {"VERSION.txt", "nyc311/__init__.py", "nyc311/config.py", "nyc311/ingest.py",
                 "nyc311/q03_complaint_types.py", "nyc311/build_report.py", "nyc311/nytplot.py",
                 "nyc311/nyt.mplstyle", "nyc311/complaint_type_map.csv", "run_all.py", "lab.ipynb",
                 "requirements.txt", "README.md", "data/external/weather.csv", "report/prose/README.md"}


@pytest.fixture
def fake_repo(tmp_path):
    """tmp_path/nyc311_lab: a committed lab folder with this script in tools/; the zip lands in tmp_path."""
    repo = tmp_path / "nyc311_lab"
    for rel, text in FAKE_FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    (repo / "tools").mkdir()
    shutil.copy(SCRIPT, repo / "tools" / SCRIPT.name)
    git(repo, "init", "-q")
    git(repo, "add", "-A", "-f")
    git(repo, "commit", "-q", "--no-verify", "-m", "fake lab")
    return repo


def build(repo, *args):
    return run_script(repo / "tools" / SCRIPT.name, *args)


def test_default_output_sits_next_to_the_folder_and_holds_only_packaged_files(fake_repo):
    r = build(fake_repo)
    assert r.returncode == 0, r.stderr
    out = fake_repo.parent / "nyc311_lab_upload.zip"
    assert str(out) in r.stdout
    assert entries(out) == FAKE_EXPECTED
    version = read(out, "VERSION.txt")
    assert git(fake_repo, "rev-parse", "HEAD").strip() in version
    assert "stages present:  ingest, q03, report" in version
    assert "not built yet:   q01" in version
    assert "stages included: ingest, q03, report" in r.stdout
    assert git(fake_repo, "status", "--porcelain", "--ignored", "--untracked-files=all") == "", \
        "the lab folder itself must stay untouched"


def test_refuses_uncommitted_changes_to_packaged_files(fake_repo):
    (fake_repo / "nyc311" / "ingest.py").write_text("def run(cfg):\n    raise SystemExit('edited')\n")
    r = build(fake_repo)
    assert r.returncode != 0
    assert "uncommitted" in r.stderr and "nyc311/ingest.py" in r.stderr and "--allow-dirty" in r.stderr
    assert not (fake_repo.parent / "nyc311_lab_upload.zip").exists()

    git(fake_repo, "add", "nyc311/ingest.py")  # staged but not committed: still refused
    assert build(fake_repo).returncode != 0


def test_allow_dirty_packages_the_commit_not_the_edit(fake_repo):
    (fake_repo / "README.md").write_text("# edited, not committed\n")
    r = build(fake_repo, "--allow-dirty")
    assert r.returncode == 0, r.stderr
    out = fake_repo.parent / "nyc311_lab_upload.zip"
    assert read(out, "README.md") == "# lab\n"
    assert "README.md" in read(out, "VERSION.txt").split("--allow-dirty")[1]
    assert "NOT included" in r.stdout


def test_prose_comes_from_the_working_tree_even_if_uncommitted(fake_repo):
    (fake_repo / "report" / "prose" / "README.md").write_text("edited guide\n")
    (fake_repo / "report" / "prose" / "q01.md").write_text("My Q1 answer.\n")
    r = build(fake_repo)
    assert r.returncode == 0, r.stderr
    out = fake_repo.parent / "nyc311_lab_upload.zip"
    assert read(out, "report/prose/q01.md") == "My Q1 answer.\n"
    assert read(out, "report/prose/README.md") == "edited guide\n"
    assert "working tree" in r.stdout and "q01.md (new)" in r.stdout and "README.md (edited)" in r.stdout


def test_changes_outside_the_package_do_not_block(fake_repo):
    (fake_repo / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    (fake_repo / "tests" / "test_new.py").write_text("")
    (fake_repo / "data" / "sample" / "311_batch_0002.json").write_text("[]")
    r = build(fake_repo)
    assert r.returncode == 0, r.stderr
    assert entries(fake_repo.parent / "nyc311_lab_upload.zip") == FAKE_EXPECTED


def test_uncommitted_new_module_is_left_out_and_named_until_committed(fake_repo):
    (fake_repo / "nyc311" / "q01_missing.py").write_text("def run(cfg):\n    pass\n")
    r = build(fake_repo)
    assert r.returncode == 0, r.stderr
    out = fake_repo.parent / "nyc311_lab_upload.zip"
    assert "nyc311/q01_missing.py" not in entries(out)
    assert "NOT included" in r.stdout and "nyc311/q01_missing.py" in r.stdout

    git(fake_repo, "add", "nyc311/q01_missing.py")
    git(fake_repo, "commit", "-q", "--no-verify", "-m", "add q01")
    r = build(fake_repo, "--out", str(fake_repo.parent / "second.zip"))
    assert r.returncode == 0, r.stderr
    assert "nyc311/q01_missing.py" in entries(fake_repo.parent / "second.zip")
    assert "stages present:  ingest, q01, q03, report" in read(fake_repo.parent / "second.zip", "VERSION.txt")


def test_outside_a_git_repo_it_stops_with_a_clear_message(tmp_path):
    folder = tmp_path / "plain" / "nyc311_lab"
    (folder / "tools").mkdir(parents=True)
    shutil.copy(SCRIPT, folder / "tools" / SCRIPT.name)
    r = build(folder)
    assert r.returncode != 0
    assert "not a git repository" in r.stderr
