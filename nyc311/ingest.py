"""Stage 0 (manifest) and stage 1 (JSON -> all-VARCHAR parquet), incremental by file and safe to rerun.

Files are processed in chunks of CHUNK_FILES. For each chunk: a process pool checks every file (size, SHA-256,
does it parse, which columns, created-date range) and rewrites any file DuckDB can't read directly (cut off,
invalid UTF-8, stray non-object elements) as clean newline-delimited JSON; DuckDB then writes the chunk to one
Parquet part (to a temporary name, renamed when complete); only then is the chunk added to the manifest.
So a crash at any point leaves a manifest that lists exactly the files whose rows are in Parquet, and a
rerun picks up where it stopped. Parts not listed in the manifest (a crash between the two steps) are deleted.
"""
import glob
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import pandas as pd

from nyc311.report import Report

CHUNK_FILES = 100
_DECODER = json.JSONDecoder()
READ_FAILED = "read failed (retried next run): "
MANIFEST_COLUMNS = ["file", "path", "size_bytes", "sha256", "parse_ok", "error", "source", "n_records", "keys",
                    "min_created", "max_created", "part", "n_parquet"]


def recover_records(text):
    """Every complete element of a JSON array that may be cut off mid-record ([] if it isn't an array)."""
    start = len(text) - len(text.lstrip())
    if not text[start:start + 1] == "[":
        return []
    out, i, n = [], start + 1, len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            return out
        try:
            obj, i = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            return out
        out.append(obj)


def _read(raw):
    """(records, problem, needs_copy) for one file's bytes. problem is '' for a sound file; needs_copy says
    DuckDB can't read the file as it is (problems, or a UTF-8 byte-order mark, which DuckDB < 1.4 rejects)."""
    if not raw.strip():
        return [], "empty file (0 bytes or only whitespace)", False
    bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text, problem = raw.decode("utf-8-sig"), ""
    except UnicodeDecodeError as e:
        text, problem = raw.decode("utf-8-sig", errors="replace"), f"invalid UTF-8 at byte {e.start} (replaced)"
    try:
        recs = json.loads(text)
    except json.JSONDecodeError as e:
        recs = recover_records(text)
        return recs, "; ".join(p for p in (problem, f"cut off or malformed: {e}") if p), True
    if not isinstance(recs, list):
        return [], "; ".join(p for p in (problem, f"top level is a JSON {type(recs).__name__}, not an array") if p), True
    return recs, problem, bool(problem) or bom


def scan_file(path, recovered_dir):
    """Manifest row for one file. Never raises: any problem goes into 'error' with parse_ok False."""
    name = os.path.basename(path)
    info = {"file": name, "path": str(path), "size_bytes": 0, "sha256": "", "parse_ok": False, "error": "",
            "source": str(path), "n_records": 0, "keys": "", "min_created": None, "max_created": None}
    try:
        info["size_bytes"] = os.path.getsize(path)
        raw = Path(path).read_bytes()
        info.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        recs, problem, needs_copy = _read(raw)
        objs = [r for r in recs if isinstance(r, dict)]
        if len(objs) < len(recs):
            problem = "; ".join(p for p in (problem, f"{len(recs) - len(objs)} array elements are not objects (dropped)") if p)
            needs_copy = True
        if needs_copy and objs:  # DuckDB can't read this file as it is: hand it a clean copy
            rec_path = Path(recovered_dir) / f"{name}.ndjson"
            with open(rec_path, "w", encoding="utf-8") as fh:
                for r in objs:
                    fh.write(json.dumps(r) + "\n")
            info["source"] = str(rec_path)
        keys, created = set(), []
        for r in objs:
            keys.update(r)
            c = r.get("created_date")
            if isinstance(c, str) and c:
                created.append(c)
        info.update(parse_ok=not problem, error=problem[:300], n_records=len(objs), keys="|".join(sorted(keys)),
                    min_created=min(created) if created else None, max_created=max(created) if created else None)
    except Exception as e:  # unreadable file, out of memory on one file, ...: record it, keep going
        info.update(parse_ok=False, n_records=0, error=f"{READ_FAILED}{type(e).__name__}: {e}"[:300])
    return info


def raw_glob(cfg):
    return (cfg.work_dir / "raw" / "*.parquet").as_posix()


def raw_source(cfg):
    """SQL table expression for the raw Parquet. union_by_name: parts written at different times can have
    different columns (Socrata leaves out keys whose value is null, and new columns appear over the years)."""
    return f"read_parquet({_sql_str(raw_glob(cfg))}, union_by_name=true)"


def _sql_str(s):
    return "'" + str(s).replace("'", "''") + "'"


def match_files(pattern):
    """Files matching the DATA pattern; '~' is expanded and a literal folder with [ ] in its name still works."""
    pattern = os.path.expanduser(str(pattern))
    folder, base = os.path.split(pattern)
    if folder and os.path.isdir(folder) and glob.has_magic(folder):
        pattern = os.path.join(glob.escape(folder), base)
    files = sorted(p for p in glob.glob(pattern) if os.path.isfile(p))
    names = pd.Series([os.path.basename(f) for f in files])
    clash = names[names.duplicated()]
    if len(clash):
        raise ValueError(f"{len(clash)} files have the same name in different folders (e.g. {clash.iloc[0]}); "
                         "rows are tagged by file name, so point DATA at one folder")
    return files


def _write_part(con, cfg, chunk, columns):
    """Write the chunk's rows to one Parquet part; returns (part path or None, rows per file)."""
    cols = "{" + ", ".join(f"{_sql_str(c)}: 'VARCHAR'" for c in columns) + "}"
    selects = []
    readable = chunk[chunk.n_records > 0]
    direct = readable[readable.source == readable.path].source.tolist()
    rewritten = readable[readable.source != readable.path].source.tolist()
    for paths, fmt in ((direct, "array"), (rewritten, "newline_delimited")):
        if paths:
            lst = "[" + ", ".join(_sql_str(Path(p).as_posix()) for p in paths) + "]"
            selects.append(
                f"select * exclude (filename), "
                f"regexp_replace(regexp_extract(filename, '[^/]+$'), '\\.ndjson$', '') as _file "
                f"from read_json({lst}, format='{fmt}', columns={cols}, filename=true)")
    if not selects:
        return None, {}
    tag = hashlib.sha256("|".join(chunk.file).encode()).hexdigest()[:10]
    out = cfg.work_dir / "raw" / f"part_{Path(chunk.file.iloc[0]).stem.replace(' ', '_')}_{tag}.parquet"
    tmp = out.with_name(out.name + ".tmp")
    con.execute(f"copy ({' union all by name '.join(selects)}) to {_sql_str(tmp.as_posix())} "
                f"(format parquet, compression zstd)")
    per_file = dict(con.execute(f"select _file, count(*) from read_parquet({_sql_str(tmp.as_posix())}) group by 1")
                    .fetchall())
    os.replace(tmp, out)
    return out, per_file


def _save_manifest(manifest, path):
    tmp = path.with_name(path.name + ".tmp")
    manifest.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _upgrade_legacy(cfg, old):
    """Add part/n_parquet to a manifest written by the first version of this code (which wrote every part
    before the manifest), by counting each file's rows in the existing parts. No file is rescanned."""
    raw_dir = cfg.work_dir / "raw"
    con = cfg.connect()
    try:
        counts = con.sql(f"""select _file, regexp_extract(filename, '[^/]+$') part, count(*) n
                             from read_parquet({_sql_str((raw_dir / "*.parquet").as_posix())}, filename=true,
                                               union_by_name=true)
                             group by all""").df() if any(raw_dir.glob("*.parquet")) else pd.DataFrame(
            columns=["_file", "part", "n"])
    finally:
        con.close()
    if counts._file.duplicated().any() or not set(counts._file) <= set(old.file):
        print("[ingest] the Parquet copy doesn't match the old manifest; rebuilding it")
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    by_file = counts.set_index("_file")
    old = old.copy()
    old["part"] = old.file.map(by_file.part)
    old["n_parquet"] = old.file.map(by_file.n).fillna(0).astype("int64")
    print(f"[ingest] upgraded the manifest of {len(old):,} files to the current layout (no rescan)")
    return old[MANIFEST_COLUMNS]


def _load_manifest(cfg):
    """The manifest, after removing anything a crash left behind (and upgrading one from the first version)."""
    raw_dir, mpath = cfg.work_dir / "raw", cfg.work_dir / "manifest.parquet"
    old = pd.read_parquet(mpath) if mpath.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    if "part" not in old.columns:
        old = _upgrade_legacy(cfg, old)
        _save_manifest(old, mpath)
    old = old[~old.error.fillna("").astype(str).str.startswith(READ_FAILED)].reset_index(drop=True)
    keep = set(old.part.dropna())
    for p in list(raw_dir.glob("*.parquet")) + list(raw_dir.glob("*.tmp")):
        if p.name not in keep:
            print(f"[ingest] removing {p.name}: not in the manifest (left by an interrupted run)")
            p.unlink()
    return old


class _Scanner:
    """Scans files in a process pool; falls back to one process if the pool can't start or breaks."""

    def __init__(self, workers, recovered_dir):
        self.fn = partial(scan_file, recovered_dir=recovered_dir)
        self.ex = None
        if workers > 1:
            try:
                self.ex = ProcessPoolExecutor(max_workers=workers)
            except (BrokenProcessPool, OSError, NotImplementedError, PermissionError) as e:
                print(f"[ingest] process pool unavailable ({e}); scanning one file at a time")

    def __call__(self, paths):
        if self.ex is not None:
            try:
                return list(self.ex.map(self.fn, paths, chunksize=2))
            except (BrokenProcessPool, OSError, PermissionError) as e:
                print(f"[ingest] process pool unavailable ({type(e).__name__}: {e}); scanning one file at a time")
                self.close()
        return [self.fn(p) for p in paths]

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(cancel_futures=True)
            self.ex = None


def _keys(manifest):
    return {k for ks in manifest["keys"].dropna() for k in str(ks).split("|") if k}


def run(cfg):
    t0 = time.time()
    files = match_files(cfg.data_glob)
    if not files:
        raise FileNotFoundError(f"no files match {cfg.data_glob!r}")
    recovered_dir = cfg.work_dir / "recovered"
    for d in (cfg.work_dir / "raw", recovered_dir, cfg.out_dir / "agg"):
        d.mkdir(parents=True, exist_ok=True)

    mpath = cfg.work_dir / "manifest.parquet"
    manifest = _load_manifest(cfg)
    known = dict(zip(manifest.file, manifest.size_bytes))
    changed = [f for f in files if os.path.basename(f) in known and known[os.path.basename(f)] != os.path.getsize(f)]
    if changed:
        raise RuntimeError(f"{len(changed)} files changed size since they were converted (e.g. {changed[0]}); "
                           f"delete {cfg.work_dir} and rerun")
    new = [f for f in files if os.path.basename(f) not in known]
    if new:
        print(f"[ingest] {len(new):,} new files ({len(known):,} already converted); "
              f"{CHUNK_FILES} files per Parquet part, {cfg.threads} workers", flush=True)
    scanner = _Scanner(max(1, min(cfg.threads, len(new))), recovered_dir) if new else None
    con = None
    try:
        for i in range(0, len(new), CHUNK_FILES):
            chunk = pd.DataFrame(scanner(new[i:i + CHUNK_FILES]))
            if con is None:  # after the pool has started, so workers never inherit DuckDB's threads
                con = cfg.connect()
            columns = sorted(_keys(manifest) | _keys(chunk))
            part, per_file = _write_part(con, cfg, chunk, columns)
            chunk["part"] = part.name if part else None
            chunk["n_parquet"] = chunk.file.map(per_file).fillna(0).astype("int64")
            manifest = pd.concat([manifest, chunk], ignore_index=True) if len(manifest) else chunk
            _save_manifest(manifest[MANIFEST_COLUMNS], mpath)
            done = min(i + CHUNK_FILES, len(new))
            print(f"[ingest] {done:,}/{len(new):,} files | {int(manifest.n_records.sum()):,} rows | "
                  f"{time.time() - t0:,.0f} s", flush=True)
    finally:
        if scanner is not None:
            scanner.close()
    manifest = manifest[MANIFEST_COLUMNS].reset_index(drop=True)
    manifest.drop(columns=["path", "source"]).to_csv(cfg.out_dir / "agg" / "manifest.csv", index=False)
    _report(cfg, files, new, manifest, time.time() - t0)
    return {"manifest": manifest, "new_files": len(new)}


def problem_kind(error):
    e = str(error)
    for key, label in [("empty file", "empty file"), ("top level", "not an array (top level is an object/value)"),
                       ("cut off", "cut off / malformed (records before the cut kept)"),
                       ("Unterminated", "cut off / malformed (records before the cut kept)"),  # first-version wording
                       ("Expecting", "cut off / malformed (records before the cut kept)"),
                       ("not objects", "array elements that are not objects"), ("UTF-8", "invalid UTF-8 bytes"),
                       ("ermission", "cannot be read (permission)")]:
        if key in e:
            return label
    return "other: " + e.split(":")[0][:40]


def _report(cfg, files, new, manifest, seconds):
    rep = Report("01_ingest", cfg.out_dir)
    matched = {os.path.basename(f) for f in files}
    rep.add(f"data pattern: {cfg.data_glob}",
            f"files matched: {len(files):,} | new this run: {len(new):,} | in manifest: {len(manifest):,} "
            f"| in manifest but not matched now: {len(set(manifest.file) - matched):,}",
            f"records: {int(manifest.n_records.sum()):,} | rows in Parquet: {int(manifest.n_parquet.sum()):,} "
            f"| total JSON size: {manifest.size_bytes.sum() / 1e9:,.2f} GB",
            f"threads: {cfg.threads} | seconds: {seconds:,.1f}")
    mism = manifest[manifest.n_parquet != manifest.n_records]
    if len(mism):
        rep.add(f"WARNING: {len(mism)} files have a different row count in Parquet than in the JSON check")
        rep.table(mism[["file", "n_records", "n_parquet"]], max_rows=10)
    bad = manifest[~manifest.parse_ok.astype(bool)].copy()
    rep.add(f"files with a problem: {len(bad):,} (records kept from them: {int(bad.n_records.sum()):,})")
    if len(bad):
        bad["kind"] = bad.error.map(problem_kind)
        kinds = (bad.groupby("kind").agg(files=("file", "size"), records_kept=("n_records", "sum"),
                                         example=("file", "first"))
                 .sort_values("files", ascending=False).reset_index())
        rep.table(kinds, max_rows=12, title="Problem files by kind")
        # Files that lost rows matter most: cut-off files first, then the rest by size.
        bad["order"] = (~bad.kind.str.startswith("cut off")).astype(int)
        bad = bad.sort_values(["order", "size_bytes"], ascending=[True, False])
        rep.table(bad[["file", "size_bytes", "n_records", "error"]], max_rows=40, title="Problem files")
    rep.emit()
