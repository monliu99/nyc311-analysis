#!/usr/bin/env python3
"""Build the cluster upload zip: only what the cluster needs to produce the results, from the last commit.

    python tools/make_upload_zip.py                 # writes ../nyc311_lab_upload.zip, next to this folder
    python tools/make_upload_zip.py --out PATH      # somewhere else
    python tools/make_upload_zip.py --allow-dirty   # package the last commit even if packaged files have edits

Everything sits under nyc311_lab/ inside the zip: nyc311/ (the package, no __pycache__), run_all.py, lab.ipynb,
requirements.txt, README.md, data/external/, report/prose/, plus VERSION.txt (commit, build time, stages present).
Tests, tools, the brief, data/sample, work, outputs and cluster_outputs stay out. File contents come from the
committed HEAD, so the zip always matches a commit; if any packaged file has uncommitted changes the script stops
(--allow-dirty packages the commit without them). Exception: report/prose/*.md comes from the working tree,
committed or not, because the student edits prose. Files never committed are not packaged (the script lists them).
The local folder is only read. Standard library only; Python 3.9+.
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

TOP = "nyc311_lab"
DEFAULT_NAME = "nyc311_lab_upload.zip"
FILES = ("run_all.py", "lab.ipynb", "requirements.txt", "README.md")
DIRS = ("nyc311/", "data/external/", "report/prose/")
PROSE_DIR = "report/prose/"
REQUIRED = FILES + ("nyc311/__init__.py", "nyc311/config.py")
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}
SKIP_NAMES = {".DS_Store"}
REPO = Path(__file__).resolve().parents[1]


class UploadError(Exception):
    pass


def is_packaged(path):
    """True for a repo-relative path that belongs in the upload."""
    p = PurePosixPath(path)
    if SKIP_PARTS.intersection(p.parts) or p.suffix in SKIP_SUFFIXES or p.name in SKIP_NAMES:
        return False
    return path in FILES or path.startswith(DIRS)


def is_prose(path):
    """report/prose/<name>.md: the student's writing, taken from the working tree."""
    rest = path[len(PROSE_DIR):]
    return path.startswith(PROSE_DIR) and "/" not in rest and rest.endswith(".md")


def git(repo, *args, binary=False):
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True)
    except FileNotFoundError:
        raise UploadError("git is not installed or not on PATH") from None
    if r.returncode:
        raise UploadError(f"git {' '.join(args)} failed in {repo}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode("utf-8", errors="surrogateescape")


def repo_root(repo):
    try:
        return Path(git(repo, "rev-parse", "--show-toplevel").strip())
    except UploadError:
        raise UploadError(f"{repo} is not a git repository; the zip is built from its last commit") from None


def head_info(repo):
    try:
        sha, epoch, date, subject = git(repo, "show", "-s", "--format=%H%n%ct%n%cI%n%s", "HEAD").split("\n")[:4]
    except (UploadError, ValueError):
        raise UploadError(f"{repo} has no commits yet; commit the code first") from None
    return {"sha": sha, "epoch": int(epoch), "date": date, "subject": subject}


def head_files(repo, commit):
    """{path: (mode, blob sha)} for every regular file in the commit."""
    files = {}
    for entry in git(repo, "ls-tree", "-r", "-z", "--full-tree", commit).split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind == "blob" and mode in ("100644", "100755"):
            files[path] = (mode, sha)
    return files


def working_tree_changes(repo):
    """[(two-letter status, path)] from git status; a rename contributes both its old and new path."""
    fields = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0")
    changes, i = [], 0
    while i < len(fields):
        field = fields[i]
        i += 1
        if not field:
            continue
        code, path = field[:2], field[3:]
        if code[0] in "RC" and i < len(fields):
            changes.append((code, fields[i]))
            i += 1
        changes.append((code, path))
    return changes


def stage_modules(config_source):
    tree = ast.parse(config_source)
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == "STAGE_MODULES" for t in targets) and node.value is not None:
            return ast.literal_eval(node.value)
    raise UploadError("nyc311/config.py has no STAGE_MODULES dict")


def split_stages(modules, paths):
    present, missing = [], []
    for stage, module in modules.items():
        rel = module.replace(".", "/")
        (present if f"{rel}.py" in paths or f"{rel}/__init__.py" in paths else missing).append(stage)
    return present, missing


def prose_label(repo, code, path):
    if not (repo / path).is_file():
        return "deleted, not included"
    return "new" if code == "??" or code[0] == "A" else "edited"


def zip_time(epoch):
    return max(time.localtime(epoch)[:6], (1980, 1, 1, 0, 0, 0))


def add(z, name, data, epoch, mode=0o644):
    info = zipfile.ZipInfo(f"{TOP}/{name}", date_time=zip_time(epoch))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | mode) << 16
    z.writestr(info, data)


def build(repo, out, allow_dirty=False):
    repo = repo_root(repo)
    head = head_info(repo)
    committed = head_files(repo, head["sha"])
    missing = [p for p in REQUIRED if p not in committed]
    if missing:
        raise UploadError(f"the last commit in {repo} has no {', '.join(missing)}; is this the nyc311_lab folder?")

    dirty, untracked, prose_changes = [], [], []
    for code, path in working_tree_changes(repo):
        if is_prose(path):
            prose_changes.append((code, path))
        elif not is_packaged(path):
            continue
        elif code == "??":
            untracked.append(path)
        else:
            dirty.append((code, path))
    if dirty and not allow_dirty:
        lines = "\n".join(f"  {code} {path}" for code, path in dirty)
        raise UploadError(
            "these packaged files have uncommitted changes, and the zip is built from the last commit:\n"
            f"{lines}\nCommit them first (or pass --allow-dirty to package the last commit without them).")

    from_head = {p: v for p, v in committed.items() if is_packaged(p) and not is_prose(p)}
    prose = sorted(p for p in (repo / PROSE_DIR).glob("*.md") if p.is_file())
    paths = set(from_head) | {PROSE_DIR + p.name for p in prose}
    present, not_built = split_stages(stage_modules(git(repo, "cat-file", "blob", from_head["nyc311/config.py"][1])),
                                      paths)

    built = time.time()
    prose_note = ", ".join(f"{path[len(PROSE_DIR):]} ({prose_label(repo, code, path)})" for code, path in prose_changes)
    version = [
        "nyc311_lab upload package (built by tools/make_upload_zip.py)",
        f"commit:          {head['sha']}",
        f"commit date:     {head['date']}",
        f"commit message:  {head['subject']}",
        f"built:           {time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(built))}",
        f"stages present:  {', '.join(present)}",
        f"not built yet:   {', '.join(not_built) or 'none'} (run_all.py skips these)",
        f"report/prose (working tree): {', '.join(p.name for p in prose) or 'none'}"
        + (f"; uncommitted: {prose_note}" if prose_note else ""),
    ]
    if dirty:
        version.append("WARNING: built with --allow-dirty; uncommitted changes to these files are NOT in this zip: "
                       + ", ".join(path for _, path in dirty))

    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".nyc311_upload_", suffix=".zip", dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            add(z, "VERSION.txt", "\n".join(version) + "\n", built)
            for path in sorted(from_head):
                mode, sha = from_head[path]
                add(z, path, git(repo, "cat-file", "blob", sha, binary=True), head["epoch"],
                    0o755 if mode == "100755" else 0o644)
            for p in prose:
                add(z, PROSE_DIR + p.name, p.read_bytes(), p.stat().st_mtime)
            count = len(z.infolist())
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"out": out, "files": count, "head": head, "present": present, "not_built": not_built,
            "prose": [p.name for p in prose], "prose_note": prose_note, "untracked": untracked, "dirty": dirty}


def main(argv=None):
    p = argparse.ArgumentParser(description="Zip what the cluster needs (from the last commit) for upload.")
    p.add_argument("--out", help=f"zip to write (default: {DEFAULT_NAME} next to the nyc311_lab folder)")
    p.add_argument("--allow-dirty", action="store_true",
                   help="package the last commit even if packaged files have uncommitted changes")
    a = p.parse_args(argv)
    try:
        r = build(REPO, a.out or REPO.parent / DEFAULT_NAME, a.allow_dirty)
    except UploadError as e:
        print(f"make_upload_zip: {e}", file=sys.stderr)
        return 1
    size = r["out"].stat().st_size
    shown = f"{size / 1e6:,.1f} MB" if size >= 1e6 else f"{size / 1e3:,.0f} KB"
    print(f"Wrote {r['out']}")
    print(f"  {shown}, {r['files']} files, all under {TOP}/; commit {r['head']['sha'][:10]} "
          f"({r['head']['date']}): {r['head']['subject']}")
    print(f"  stages included: {', '.join(r['present'])}")
    if r["not_built"]:
        print(f"  not built yet (run_all.py skips them): {', '.join(r['not_built'])}")
    print(f"  report/prose: {', '.join(r['prose']) or 'no files'} taken from the working tree"
          + (f" (uncommitted: {r['prose_note']})" if r["prose_note"] else ""))
    if r["untracked"]:
        print(f"  NOT included (never committed; commit to include): {', '.join(r['untracked'])}")
    if r["dirty"]:
        print("  WARNING (--allow-dirty): uncommitted changes NOT included: "
              + ", ".join(path for _, path in r["dirty"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
