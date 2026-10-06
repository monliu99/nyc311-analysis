"""Build N batch files shaped like the cluster pull, from the 20 local sample files, for scale tests.

    python tools/make_synthetic_batches.py --files 200 --out /tmp/nyc311_synth/data

Every output file is a JSON array of ~10,000 Socrata-style records (same field set and formatting as the
sample). unique_key is rewritten so keys are unique, and dates are shifted so file i covers its slice of
--start..--end, in order, the way the real pull is sorted by created_date. closed/due/resolution dates move by
the same number of days as created_date (durations are kept); 1900 placeholders stay as they are, and the
sample's own stray 2013/2015 rows keep their dates (so they stay out of their file's window).

Planted problems (file numbers are printed and written to <out>/../planted.json; --problem-repeats K plants
each one K times):
  truncated files (3), a file with an extra column, one missing a column, an empty array, a top-level object,
  a zero-byte file, a UTF-8 BOM, an invalid UTF-8 byte, rows copied from the previous file (identical) and
  re-pulled with changes (conflicting), an exact re-download under another name ("... (1).json"), stray
  out-of-window rows (2009, 2021, 1900), complaint_type spelling drift in later years, and batch numbers
  missing from the sequence (--missing-numbers, as in the real pull).

Writes only outside the repository (the repo sits in a synced folder; the data is GBs).
"""
import argparse
import datetime as dt
import json
import os
import random
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATE_FIELDS = ["created_date", "closed_date", "due_date", "resolution_action_updated_date"]


def _load(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        dec, out, i = json.JSONDecoder(), [], text.find("[") + 1
        while True:
            while i < len(text) and text[i] in " \t\r\n,":
                i += 1
            try:
                obj, i = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                return out
            out.append(obj)


def load_templates(sample_dir):
    """Records per sample file, split into in-window rows (sorted by created) and stray rows."""
    templates = []
    for p in sorted(Path(sample_dir).glob("311_batch_*.json")):
        recs = _load(p)
        days = sorted(r["created_date"][:10] for r in recs if r.get("created_date"))
        med = dt.date.fromisoformat(days[len(days) // 2])
        inside, stray = [], []
        for r in recs:
            c = r.get("created_date")
            far = c and abs((dt.date.fromisoformat(c[:10]) - med).days) > 30
            (stray if far else inside).append(r)
        inside.sort(key=lambda r: r.get("created_date") or "")
        if len(inside) >= 9000:  # the cut-off batch 0010 is not a good template
            templates.append((inside, stray))
    if not templates:
        sys.exit(f"no usable sample files in {sample_dir}")
    return templates


def _shift(value, days):
    if not value or value.startswith("1900-") or days == 0:
        return value
    d = dt.date.fromisoformat(value[:10]) + dt.timedelta(days=days)
    return d.isoformat() + value[10:]


def _drift_type(ct, year, rng_seed):
    """Later years spell some complaint types differently (upper case, plural, extra space)."""
    if year < 2014 or not ct:
        return ct
    h = zlib.crc32(f"{rng_seed}|{ct}".encode()) % 10
    if h == 0:
        return ct.upper()
    if h == 1 and not ct.endswith("s"):
        return ct + "s"
    if h == 2:
        return ct.replace(" ", "  ", 1) if " " in ct else ct.lower()
    return ct


def build_file(i, n_files, templates, start, total_days, key_base, seed, rows=None):
    inside, stray = templates[i % len(templates)]
    if rows and rows < len(inside):
        inside = inside[:: max(1, len(inside) // rows)][:rows]
    lo = start + dt.timedelta(days=total_days * i / n_files)
    width = total_days / n_files
    out = []
    n = len(inside)
    for j, r in enumerate(inside):
        r = dict(r)
        old = dt.date.fromisoformat(r["created_date"][:10])
        new = lo + dt.timedelta(days=int(width * j / n))
        delta = (new - old).days
        for f in DATE_FIELDS:
            if f in r:
                r[f] = _shift(r[f], delta)
        r["complaint_type"] = _drift_type(r.get("complaint_type"), new.year, seed)
        out.append(r)
    out[:0] = [dict(r) for r in stray]  # the sample's stray rows sit at the top of their file
    for j, r in enumerate(out):
        r["unique_key"] = str(key_base + i * 20_000 + j)
    return out


def dumps(records):
    """Same layout as the Socrata export: '[{...}\\n,{...}\\n,...]' plus a trailing newline."""
    return "[" + "\n,".join(json.dumps(r, separators=(",", ":")) for r in records) + "]\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default=str(REPO / "data" / "sample"))
    p.add_argument("--out", required=True, help="output folder (must be outside the repo, e.g. /tmp/...)")
    p.add_argument("--files", type=int, default=200)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2017-12-31")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-problems", action="store_true", help="clean files only")
    p.add_argument("--rows-per-file", type=int, default=None, help="keep only this many rows per file (fast tests)")
    p.add_argument("--problem-repeats", type=int, default=1, help="plant each problem this many times")
    p.add_argument("--missing-numbers", type=int, default=4,
                   help="batch numbers left out of the sequence (one single number, the rest as one run)")
    a = p.parse_args(argv)

    out = Path(a.out).resolve()
    if out == REPO or REPO in out.parents:
        sys.exit(f"refusing to write inside the repo ({REPO}); use e.g. /tmp/nyc311_synth/data")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("311_batch_*.json")):
        sys.exit(f"{out} already has batch files; use an empty folder")
    rng = random.Random(a.seed)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    total_days = (end - start).days + 1
    templates = load_templates(a.sample)
    n = a.files
    name = lambda k: f"311_batch_{k + 1:04d}.json"  # noqa: E731

    # Problem files, spread over the run so they land in different ingest chunks.
    roles = ["truncated", "truncated", "truncated", "extra_column", "missing_column", "empty_array",
             "top_level_object", "zero_bytes", "bom", "bad_utf8", "identical_dups", "conflicting_dups",
             "stray_rows", "redownload"] * a.problem_repeats
    spots = sorted(rng.sample(range(5, n - 2), min(n - 7, len(roles)))) if n >= 30 and not a.no_problems else []
    planted = dict(zip(spots, roles))
    free = [k for k in range(5, n - 2) if k not in planted and k - 1 not in planted]
    missing = set()
    if a.missing_numbers and not a.no_problems and len(free) > a.missing_numbers + 2:
        missing.add(free[len(free) // 4])
        width = a.missing_numbers - 1
        runs = [f for f in free[(3 * len(free)) // 4:] if all(f + d in free for d in range(width))]
        if width and runs:
            missing |= set(range(runs[0], runs[0] + width))

    prev = None
    total_rows = 0
    for k in range(n):
        if k in missing:
            prev = None
            continue
        role = planted.get(k)
        path = out / name(k)
        if role == "zero_bytes":
            path.write_bytes(b"")
            prev = None
            continue
        if role == "empty_array":
            path.write_text("[]\n")
            prev = None
            continue
        if role == "top_level_object":
            path.write_text('{"error": true, "code": "query.execution.queryTimeout", '
                            '"message": "Query timed out after [600] seconds", "data": {"retry": [1, 2]}}\n')
            prev = None
            continue
        recs = build_file(k, n, templates, start, total_days, 30_000_000, a.seed, a.rows_per_file)
        if role == "extra_column":
            for r in recs:
                r[":@computed_region_efsh_h5xi"] = "1234"
                r["new_survey_field"] = "Y"
        elif role == "missing_column":
            for r in recs:
                r.pop("open_data_channel_type", None)
        elif role == "identical_dups" and prev:
            recs = recs + [dict(r) for r in prev[-max(1, len(prev) // 20):]]  # page overlap: the last 500 rows of the previous file again
        elif role == "conflicting_dups" and prev:
            changed = []
            for r in prev[-max(1, len(prev) // 33):]:
                r = dict(r)
                r["status"] = "Closed" if r.get("status") != "Closed" else "Open"
                r["resolution_action_updated_date"] = _shift(r.get("created_date"), 30)
                changed.append(r)
            recs = recs + changed
        elif role == "stray_rows":
            tmpl = recs[0]
            for j, when in enumerate(["2009-06-15T09:00:00.000"] * 5 + ["2021-03-01T12:00:00.000"] * 3
                                     + ["1900-01-01T00:00:00.000"] * 2):
                r = dict(tmpl, unique_key=str(29_000_000 + j), created_date=when)
                r.pop("closed_date", None)
                recs.insert(0, r)
        if k % 7 == 3 and recs:  # some valid non-ASCII text, as in real addresses
            recs[1]["incident_address"] = "100 CAFÉ STREET"
        text = dumps(recs)
        data = text.encode("utf-8")
        if role == "truncated":
            frac = [0.31, 0.62, 0.97][sum(1 for s, r in planted.items() if r == "truncated" and s < k) % 3]
            data = data[: int(len(data) * frac)]
        elif role == "bom":
            data = b"\xef\xbb\xbf" + data
        elif role == "bad_utf8":
            data = data.replace(b"100 CAF\xc3\x89 STREET", b"100 CAF\xe9 STREET", 1)
            if b"\xe9" not in data:
                cut = data.index(b'"incident_address":"') + len(b'"incident_address":"')
                data = data[:cut] + b"\xe9" + data[cut:]
        path.write_bytes(data)
        if role == "redownload":
            (out / f"{path.stem} (1).json").write_bytes(data)
        total_rows += len(recs)
        prev = recs
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{n} files written", flush=True)

    info = {"files": n, "rows_written_before_truncation": total_rows, "start": a.start, "end": a.end,
            "planted": {name(k): r for k, r in sorted(planted.items())},
            "missing_numbers": sorted(k + 1 for k in missing)}
    (out.parent / "planted.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
