"""Stage 0 (manifest) and stage 1 (JSON -> all-VARCHAR parquet), incremental by file."""
import glob
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from nyc311.report import Report

CHUNK_FILES = 100
_DECODER = json.JSONDecoder()


def recover_records(text):
    """Every complete object in a JSON array that may be cut off mid-record."""
    start = text.find("[")
    if start < 0:
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


def scan_file(path, recovered_dir):
    raw = Path(path).read_bytes()
    name = os.path.basename(path)
    info = {"file": name, "path": str(path), "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "parse_ok": True, "error": "", "source": str(path)}
    text = raw.decode("utf-8", errors="replace")
    try:
        recs = json.loads(text)
        if not isinstance(recs, list):
            raise ValueError(f"top level is {type(recs).__name__}, not an array")
    except (json.JSONDecodeError, ValueError) as e:
        recs = recover_records(text)
        info.update(parse_ok=False, error=str(e)[:200])
        rec_path = Path(recovered_dir) / f"{name}.ndjson"
        with open(rec_path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        info["source"] = str(rec_path)
    keys, created = set(), []
    for r in recs:
        keys.update(r)
        if r.get("created_date"):
            created.append(r["created_date"])
    info.update(n_records=len(recs), keys="|".join(sorted(keys)),
                min_created=min(created) if created else None, max_created=max(created) if created else None)
    return info


def raw_glob(cfg):
    return (cfg.work_dir / "raw" / "*.parquet").as_posix()


def _sql_str(s):
    return "'" + str(s).replace("'", "''") + "'"


def _write_part(con, cfg, chunk, columns):
    cols = "{" + ", ".join(f"{_sql_str(c)}: 'VARCHAR'" for c in columns) + "}"
    selects = []
    good = chunk[chunk.parse_ok].source.tolist()
    rec = chunk[~chunk.parse_ok & (chunk.n_records > 0)].source.tolist()
    for paths, fmt in ((good, "array"), (rec, "newline_delimited")):
        if paths:
            lst = "[" + ", ".join(_sql_str(Path(p).as_posix()) for p in paths) + "]"
            selects.append(
                f"select * exclude (filename), "
                f"regexp_replace(regexp_extract(filename, '[^/]+$'), '\\.ndjson$', '') as _file "
                f"from read_json({lst}, format='{fmt}', columns={cols}, filename=true)")
    if not selects:
        return None
    out = cfg.work_dir / "raw" / f"part_{Path(chunk.file.iloc[0]).stem}_{len(chunk)}.parquet"
    con.execute(f"copy ({' union all by name '.join(selects)}) to {_sql_str(out.as_posix())} "
                f"(format parquet, compression zstd)")
    return out


def run(cfg):
    t0 = time.time()
    files = sorted(glob.glob(cfg.data_glob))
    if not files:
        raise FileNotFoundError(f"no files match {cfg.data_glob!r}")
    recovered_dir = cfg.work_dir / "recovered"
    for d in (cfg.work_dir / "raw", recovered_dir, cfg.out_dir / "agg"):
        d.mkdir(parents=True, exist_ok=True)

    mpath = cfg.work_dir / "manifest.parquet"
    old = pd.read_parquet(mpath) if mpath.exists() else pd.DataFrame()
    known = dict(zip(old.file, old.size_bytes)) if len(old) else {}
    changed = [f for f in files if os.path.basename(f) in known and known[os.path.basename(f)] != os.path.getsize(f)]
    if changed:
        raise RuntimeError(f"{len(changed)} files changed size since last ingest (e.g. {changed[0]}); "
                           f"delete {cfg.work_dir} and rerun")
    new = [f for f in files if os.path.basename(f) not in known]

    if new:
        with ProcessPoolExecutor(max_workers=max(1, min(cfg.threads, len(new)))) as ex:
            infos = list(ex.map(partial(scan_file, recovered_dir=recovered_dir), new, chunksize=4))
        scanned = pd.DataFrame(infos)
        manifest = pd.concat([old, scanned], ignore_index=True) if len(old) else scanned
        columns = sorted({k for ks in manifest["keys"] for k in ks.split("|") if k})
        con = cfg.connect()
        for i in range(0, len(scanned), CHUNK_FILES):
            _write_part(con, cfg, scanned.iloc[i:i + CHUNK_FILES], columns)
        manifest.to_parquet(mpath, index=False)
    else:
        manifest = old
    manifest.drop(columns=["path", "source"]).to_csv(cfg.out_dir / "agg" / "manifest.csv", index=False)

    rep = Report("01_ingest", cfg.out_dir)
    rep.add(f"data pattern: {cfg.data_glob}",
            f"files matched: {len(files):,} | new this run: {len(new):,} | in manifest: {len(manifest):,}",
            f"records: {int(manifest.n_records.sum()):,} | total JSON size: {manifest.size_bytes.sum() / 1e9:,.2f} GB",
            f"threads: {cfg.threads} | seconds: {time.time() - t0:,.1f}")
    bad = manifest[~manifest.parse_ok]
    rep.add(f"files that failed to parse: {len(bad):,} (records recovered from them: {int(bad.n_records.sum()):,})")
    if len(bad):
        rep.table(bad[["file", "size_bytes", "n_records", "error"]], max_rows=30, title="Unparseable files")
    rep.emit()
    return {"manifest": manifest, "new_files": len(new)}
