"""Stage 2: describe the data's structure before answering any question.

Every function takes a DuckDB connection and the raw Parquet glob, and reads it with union_by_name (parts
written at different times can have different columns). Blocks 02a-02f put the most important lines first,
because each printed block is capped at 150 lines; the full tables are in outputs/agg/.
"""
import re

import duckdb
import pandas as pd

from nyc311 import ingest, rules
from nyc311.report import Report

CR = "try_cast(created_date as timestamp)"
CL = "try_cast(closed_date as timestamp)"
SPARSE_FILL = 0.5
NEAR_UNIQUE = 0.5  # columns with more distinct values than this share of rows get no top-values list
OFFSET_DAYS = (7, 14, 30, 90, 365)
BATCH_NAME = re.compile(r"^311_batch_(\d+)\.json$")


def _sql_str(s):
    return "'" + str(s).replace("'", "''") + "'"


# Columns the queries below name. If the export ever drops one from every file, it reads as NULL instead of failing.
NAMED_COLUMNS = ["unique_key", "created_date", "closed_date", "due_date", "resolution_action_updated_date", "agency",
                 "agency_name", "complaint_type", "descriptor", "status", "open_data_channel_type", "borough",
                 "address_type", "location_type"]


def _src(raw, con=None):
    """SQL source for the raw Parquet (union_by_name). With a connection, columns from NAMED_COLUMNS that no part
    has are added as NULL."""
    base = f"read_parquet({_sql_str(raw)}, union_by_name=true)"
    if con is None:
        return base
    have = set(_columns(con, raw))
    missing = [c for c in NAMED_COLUMNS if c not in have]
    if not missing:
        return base
    return "(select *, " + ", ".join(f"null::varchar as {_q(c)}" for c in missing) + f" from {base})"


def _columns(con, raw):
    return [c for c in con.sql(f"describe select * from {_src(raw)}").df().column_name if c != "_file"]


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def _placeholder_pred(col):
    toks = ", ".join("'" + t.replace("'", "''") + "'" for t in rules.PLACEHOLDER_TOKENS)
    return f"(lower(trim({col})) in ({toks}) or {col} like '{rules.PLACEHOLDER_DATE_PREFIX}%')"


def _offsets(raw, con):
    """One row per row with a parseable created_date: its file, agency, timestamp and days from the file's median."""
    return f"""(select t._file, t.agency, t.ts, m.med, date_diff('day', m.med, t.ts) as off_days
        from (select _file, agency, {CR} ts from {_src(raw, con)} where {CR} is not null) t
        join (select _file, quantile_disc({CR}, 0.5) med from {_src(raw, con)} where {CR} is not null group by 1) m
        using (_file))"""


def _in_window(days=rules.OUT_OF_WINDOW_DAYS):
    return f"abs(off_days) <= {int(days)}"


# ---------------------------------------------------------------- 02a: files, keys, rows outside their file's dates

def batch_sequence(files):
    """Gaps in the 311_batch_NNNN numbering, as compact ranges, plus names that don't follow the pattern."""
    nums, odd = [], []
    for f in files:
        m = BATCH_NAME.match(str(f))
        if m:
            nums.append(int(m.group(1)))
        else:
            odd.append(str(f))
    nums = sorted(set(nums))
    ranges, n_missing = [], 0
    for a, b in zip(nums, nums[1:]):
        if b - a > 1:
            lo, hi = a + 1, b - 1
            ranges.append(str(lo) if lo == hi else f"{lo}-{hi}")
            n_missing += hi - lo + 1
    return {"first": nums[0] if nums else None, "last": nums[-1] if nums else None, "n_numbered": len(nums),
            "n_missing": n_missing, "missing_ranges": ranges, "odd_names": sorted(odd)}


def out_of_window_rows(con, raw, days=rules.OUT_OF_WINDOW_DAYS):
    return con.sql(f"""
        select _file as file, count(*) as n_rows, count_if(not {_in_window(days)})::bigint as n_outside,
               min(ts) as min_ts, any_value(med) as median_ts, max(ts) as max_ts
        from {_offsets(raw, con)} group by 1 order by n_outside desc, file""").df()


def outside_detail(con, raw, days=rules.OUT_OF_WINDOW_DAYS):
    return con.sql(f"""
        select agency, year(ts) as year, count(*) as n
        from {_offsets(raw, con)} where not {_in_window(days)}
        group by all order by n desc""").df()


def window_offsets(con, raw):
    """How far rows sit from their file's median created date: counts beyond each threshold and quantiles,
    for all rows and for the agencies with the most rows more than a week away."""
    beyond = ", ".join(f"count_if(abs(off_days) > {d})::bigint as beyond_{d}d" for d in OFFSET_DAYS)
    qs = ", ".join(f"quantile_cont(off_days, {q}) as {name}"
                   for q, name in [(0.001, "p0_1"), (0.01, "p1"), (0.99, "p99"), (0.999, "p99_9")])
    df = con.sql(f"""
        select case when grouping(agency) = 1 then '(all)' else coalesce(agency, '(null)') end as agency,
               count(*) as n, {beyond}, {qs}
        from {_offsets(raw, con)} group by grouping sets ((), (agency))""").df()
    first = df[df.agency == "(all)"]
    rest = df[df.agency != "(all)"].sort_values(["beyond_7d", "n"], ascending=False)
    return pd.concat([first, rest], ignore_index=True)


def key_duplicates(con, raw):
    cols = _columns(con, raw)
    sig = "md5(concat_ws('|', " + ", ".join(f"coalesce({_q(c)}, '<null>')" for c in cols) + "))"
    r = con.sql(f"""
        with k as (select unique_key from {_src(raw, con)} group by 1 having count(*) > 1),
        d as (select unique_key, count(*) n, count(distinct {sig}) nv
              from {_src(raw, con)} semi join k using (unique_key) group by 1)
        select count(*), sum(n), count_if(nv = 1), count_if(nv > 1) from d""").fetchone()
    r = [int(x or 0) for x in r]
    return {"dup_keys": r[0], "dup_rows": r[1], "identical_copies": r[2], "conflicting_versions": r[3]}


def duplicate_file_pairs(con, raw):
    """For keys that appear more than once: which files share them, and whether the copies are identical."""
    cols = _columns(con, raw)
    sig = "md5(concat_ws('|', " + ", ".join(f"coalesce({_q(c)}, '<null>')" for c in cols) + "))"
    return con.sql(f"""
        with k as (select unique_key from {_src(raw, con)} group by 1 having count(*) > 1),
        d as (select unique_key, _file, {sig} sig, row_number() over (partition by unique_key order by _file) rn
              from {_src(raw, con)} semi join k using (unique_key))
        select a._file file_a, b._file file_b, count(distinct a.unique_key) as n_keys,
               count_if(a.sig = b.sig)::bigint as identical, count_if(a.sig <> b.sig)::bigint as conflicting
        from d a join d b on a.unique_key = b.unique_key and a.rn < b.rn
        group by 1, 2 order by n_keys desc, file_a, file_b""").df()


def _date_order(win, seq_files):
    """Numbered files whose median created date is more than a day before the previous numbered file's."""
    w = win.set_index("file").median_ts
    order = [f for f in seq_files if f in w.index]
    med = pd.to_datetime(w.reindex(order))
    prev = med.shift()
    back = med < prev - pd.Timedelta(days=1)
    return pd.DataFrame({"file": med.index[back], "median_ts": med[back].values, "previous_file_median": prev[back].values})


def _gap_table(seq, manifest, win):
    """Each missing batch-number range with the dates of the files on either side."""
    med = win.set_index("file").median_ts if len(win) else pd.Series(dtype="datetime64[ns]")
    by_num = {int(BATCH_NAME.match(f).group(1)): f for f in manifest.file if BATCH_NAME.match(f)}
    rows = []
    for r in seq["missing_ranges"]:
        lo, _, hi = r.partition("-")
        lo, hi = int(lo), int(hi or lo)
        before, after = by_num.get(lo - 1), by_num.get(hi + 1)
        rows.append({"missing": r, "files": hi - lo + 1,
                     "median_date_before": med.get(before, pd.NaT), "median_date_after": med.get(after, pd.NaT)})
    out = pd.DataFrame(rows, columns=["missing", "files", "median_date_before", "median_date_after"])
    for c in ["median_date_before", "median_date_after"]:
        out[c] = pd.to_datetime(out[c]).dt.date
    return out


def _compact(items, width=600):
    s = ", ".join(items)
    return s if len(s) <= width else s[:width].rsplit(", ", 1)[0] + f", ... ({len(items)} in all)"


def files_block(con, raw, manifest, out, agg):
    rep = Report("02a_files", out)
    n_rows = con.sql(f"select count(*) from {_src(raw, con)}").fetchone()[0]
    rep.add(f"duckdb {duckdb.__version__} | rows in raw parquet: {n_rows:,} "
            f"| manifest records: {int(manifest.n_records.sum()):,} | files: {len(manifest):,}")
    per_file = manifest.n_records.value_counts()
    rep.add(f"rows per file: most common {per_file.index[0]:,} ({per_file.iloc[0]:,} files), "
            f"min {manifest.n_records.min():,}, max {manifest.n_records.max():,}")

    seq = batch_sequence(manifest.file)
    win = out_of_window_rows(con, raw)
    win.to_csv(agg / "file_windows.csv", index=False)
    rep.add("", f"batch numbers {seq['first']}-{seq['last']}: {seq['n_numbered']:,} files, "
                f"{seq['n_missing']:,} numbers missing in {len(seq['missing_ranges']):,} gaps")
    if seq["missing_ranges"]:
        rep.add("missing: " + _compact(seq["missing_ranges"]))
        gaps = _gap_table(seq, manifest, win)
        gaps.to_csv(agg / "batch_gaps.csv", index=False)
        rep.table(gaps.sort_values("files", ascending=False, kind="stable"), max_rows=12,
                  title="Largest gaps in the batch numbers, with the dates on either side")
    if seq["odd_names"]:
        rep.add(f"files whose name is not 311_batch_NNNN.json: {len(seq['odd_names'])}: "
                + _compact(seq["odd_names"], 300))
    numbered = sorted((f for f in manifest.file if BATCH_NAME.match(f)), key=lambda f: int(BATCH_NAME.match(f).group(1)))
    order = _date_order(win, numbered)
    rep.add(f"files whose median created date is more than a day before the previous file's: {len(order):,}")
    if len(order):
        rep.table(order, max_rows=8)

    odd = manifest[manifest.n_records != per_file.index[0]].sort_values(["n_records", "file"])
    rep.table(odd[["file", "n_records", "parse_ok", "min_created", "max_created"]], max_rows=12,
              title=f"Files with a row count other than {per_file.index[0]:,} (fewest first)")

    dups = key_duplicates(con, raw)
    rep.add("", "duplicate unique_key: " + ", ".join(f"{k}={v:,}" for k, v in dups.items()))
    if dups["dup_keys"]:
        pairs = duplicate_file_pairs(con, raw)
        pairs.to_csv(agg / "duplicate_file_pairs.csv", index=False)
        rep.table(pairs, max_rows=8, title="Files sharing keys (same file twice = repeated inside one file)")

    rep.add("", f"rows created >{rules.OUT_OF_WINDOW_DAYS} days from their file's median date "
                f"(OUT_OF_WINDOW_DAYS): {int(win.n_outside.sum()):,} in {int((win.n_outside > 0).sum()):,} files")
    off = window_offsets(con, raw)
    off.to_csv(agg / "window_offsets.csv", index=False)
    rep.table(off, max_rows=9, title="Days from the file's median created date: rows beyond each distance, "
                                     "and quantiles (all rows, then agencies with most rows >7 days away)")
    rep.table(win[win.n_outside > 0], max_rows=10, title="Files holding out-of-window rows (most first)")
    rep.table(outside_detail(con, raw), max_rows=10, title="Out-of-window rows by agency and year")
    rep.emit()


# ---------------------------------------------------------------- 02b: date coverage

def coverage(con, raw, days=rules.OUT_OF_WINDOW_DAYS):
    """Daily counts over rows inside their file's window (stray rows are reported in 02a, not here)."""
    daily = con.sql(f"""select cast(ts as date) d, count(*) n from {_offsets(raw, con)}
                        where {_in_window(days)} group by 1 order by 1""").df()
    if daily.empty:
        e = pd.DataFrame()
        return {"daily": pd.Series(dtype="int64"), "low_days": pd.DataFrame(columns=["date", "n", "rolling_median"]),
                "empty_stretches": pd.DataFrame(columns=["start", "end", "days"]), "zero_days_short": 0, "monthly": e}
    spine = pd.date_range(daily.d.min(), daily.d.max(), freq="D")
    s = daily.set_index(pd.to_datetime(daily.d)).n.reindex(spine, fill_value=0)
    med = s.where(s > 0).rolling(29, center=True, min_periods=1).median()  # zero days (gaps) don't drag it down
    low_mask = (s > 0) & (s < rules.LOW_DAY_RATIO * med)
    low = pd.DataFrame({"date": s.index[low_mask].date, "n": s[low_mask].values, "rolling_median": med[low_mask].values})
    zero = s == 0
    run_id = (zero != zero.shift()).cumsum()
    runs = (pd.DataFrame({"zero": zero, "run": run_id, "day": s.index})[zero]
            .groupby("run").day.agg(["min", "max", "count"]))
    stretches = runs[runs["count"] >= rules.EMPTY_STRETCH_DAYS].rename(
        columns={"min": "start", "max": "end", "count": "days"}).reset_index(drop=True)
    stretches["start"] = stretches.start.dt.date
    stretches["end"] = stretches.end.dt.date
    short_zero = runs[runs["count"] < rules.EMPTY_STRETCH_DAYS]
    sd = pd.Series(s.values, index=s.index)
    monthly = sd.groupby([sd.index.year, sd.index.month]).sum().unstack(fill_value=0)
    monthly = monthly.reindex(columns=range(1, 13), fill_value=0)
    monthly.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["total"] = monthly.sum(axis=1)
    monthly.index.name = "year"
    return {"daily": sd, "low_days": low, "empty_stretches": stretches, "zero_days_short": int(short_zero["count"].sum()),
            "monthly": monthly}


def coverage_block(con, raw, out, agg):
    cov = coverage(con, raw)
    rep = Report("02b_coverage", out)
    d = cov["daily"]
    if d.empty:
        rep.add("no parseable created dates")
        return rep.emit()
    d.rename("n").rename_axis("date").reset_index().to_csv(agg / "daily_counts.csv", index=False)
    rep.add(f"rows inside their file's window only (out-of-window rows: see 02a) | created dates: "
            f"{d.index.min().date()} to {d.index.max().date()} | days with data: {(d > 0).sum():,} of {len(d):,} "
            f"| zero days inside short gaps: {cov['zero_days_short']:,}")
    rep.table(cov["empty_stretches"].sort_values("days", ascending=False), max_rows=15,
              title=f"Empty stretches (>= {rules.EMPTY_STRETCH_DAYS} days with no requests), longest first")
    rep.table(cov["monthly"].reset_index(), max_rows=40, title="Requests created per month")
    low = cov["low_days"].sort_values("n")
    rep.add("", f"low days: {len(low):,} (all in agg/daily_counts.csv)")
    rep.table(low, max_rows=30, title=f"Low days (< {rules.LOW_DAY_RATIO:.0%} of the 29-day rolling median "
                                      f"of non-empty days), lowest first")
    return rep.emit()


# ---------------------------------------------------------------- 02c: columns

def column_profile(con, raw):
    cols = _columns(con, raw)
    parts = ["count(*)"]
    for c in cols:
        qc = _q(c)
        parts += [f"count({qc})", f"count_if(trim({qc}) = '')", f"count_if({_placeholder_pred(qc)})"]
    r = con.sql(f"select {', '.join(parts)} from {_src(raw, con)}").fetchone()
    n = r[0]
    rows = []
    for i, c in enumerate(cols):
        nn, empty, ph = r[1 + 3 * i: 4 + 3 * i]
        nd, top3 = con.sql(f"""select count(*), max_by({{'v': v, 'k': k}}, k, 3)
                               from (select {_q(c)} v, count(*) k from {_src(raw, con)}
                                     where {_q(c)} is not null group by 1)""").fetchone()
        top = ""
        if nd <= NEAR_UNIQUE * max(n, 1) and top3:
            top = "; ".join(f"{str(t['v'])[:30]}={100 * t['k'] / n:.1f}%" for t in top3)
        rows.append({"column": c, "n_rows": n, "n_null": n - nn, "n_empty": empty, "n_placeholder": ph,
                     "pct_null": 100 * (n - nn) / max(n, 1),
                     "pct_missing_incl_placeholders": 100 * (n - nn + empty + ph) / max(n, 1),
                     "n_distinct": int(nd), "top_values": top})
    return pd.DataFrame(rows)


def column_presence(manifest):
    """Columns missing from some files, with the first and last file (by name) that has them."""
    key_sets = manifest.sort_values("file")[["file", "keys"]]
    sets = [(f, set(str(k).split("|")) - {""}) for f, k in zip(key_sets.file, key_sets["keys"].fillna(""))]
    files_with_rows = [(f, ks) for f, ks in sets if ks]
    all_keys = sorted(set().union(*[ks for _, ks in files_with_rows])) if files_with_rows else []
    rows = []
    for k in all_keys:
        having = [f for f, ks in files_with_rows if k in ks]
        rows.append({"column": k, "files_with_column": len(having), "files_without": len(files_with_rows) - len(having),
                     "first_file": having[0], "last_file": having[-1]})
    df = pd.DataFrame(rows, columns=["column", "files_with_column", "files_without", "first_file", "last_file"])
    return df, len(files_with_rows)


def columns_block(con, raw, manifest, out, agg):
    prof = column_profile(con, raw)
    prof.to_csv(agg / "column_profile.csv", index=False)
    rep = Report("02c_columns", out)
    rep.add(f"placeholder tokens (PROVISIONAL, lowercased): {rules.PLACEHOLDER_TOKENS} + dates starting "
            f"'{rules.PLACEHOLDER_DATE_PREFIX}'")
    rep.table(prof[["column", "pct_null", "n_empty", "n_placeholder", "pct_missing_incl_placeholders",
                    "n_distinct", "top_values"]], max_rows=60, title="Per-column missingness and top values")
    presence, n_files = column_presence(manifest)
    presence.to_csv(agg / "column_presence.csv", index=False)
    rep.add("", f"distinct columns across files: {len(presence)} (Socrata leaves out keys whose value is null, so a "
                f"rarely used column can be absent from a whole file)")
    rep.table(presence[presence.files_without > 0].sort_values("files_without", ascending=False), max_rows=30,
              title=f"Columns missing from some of the {n_files:,} files with rows (most often missing first)")
    rep.emit()
    return prof


# ---------------------------------------------------------------- 02d: values

def value_lists(con, raw):
    src = _src(raw, con)
    out = {}
    for col in ["status", "open_data_channel_type", "borough", "address_type"]:
        out[col] = con.sql(f"select {col} as value, count(*) n from {src} group by 1 order by n desc").df()
    out["agency"] = con.sql(f"""
        with a as (select agency, agency_name, count(*) n from {src} group by all),
        r as (select *, row_number() over (partition by agency order by n desc, agency_name) rn from a)
        select agency, sum(n)::bigint n, count(agency_name) n_names,
               string_agg(case when rn <= 3 then agency_name end, ' | ' order by n desc) top_names
        from r group by agency order by n desc""").df()
    out["agency_names"] = con.sql(f"""select agency, agency_name, count(*) n from {src}
                                      group by all order by agency, n desc""").df()
    out["location_type"] = con.sql(f"select location_type as value, count(*) n from {src} group by 1 order by n desc").df()
    types = con.sql(f"select complaint_type, count(*) n from {src} group by 1 order by n desc").df()
    out["n_complaint_types"] = len(types)
    out["complaint_types"] = types
    out["type_variants"] = con.sql(f"""
        with t as (select complaint_type ct, count(*) n from {src} where complaint_type is not null group by 1),
        k as (select regexp_replace(lower(trim(regexp_replace(ct, '\\s+', ' ', 'g'))), 's$', '') as key, ct, n from t)
        select key, count(*) n_spellings, sum(n)::bigint n,
               string_agg(ct || ' (' || n || ')', ' | ' order by n desc) spellings
        from k group by all having count(*) > 1 order by n desc""").df()
    return out


def values_block(con, raw, out, agg):
    vals = value_lists(con, raw)
    vals["type_variants"].to_csv(agg / "type_variants.csv", index=False)
    vals["agency_names"].to_csv(agg / "agency_names.csv", index=False)
    rep = Report("02d_values", out)
    tv = vals["type_variants"]
    rep.add(f"distinct complaint_type values: {vals['n_complaint_types']:,} | groups of spellings that differ only "
            f"by case/spacing/trailing s: {len(tv):,} covering {int(tv.n_spellings.sum()) if len(tv) else 0:,} "
            f"spellings (all in agg/type_variants.csv, agg/complaint_type_counts.csv)")
    rep.table(tv[["key", "n_spellings", "n", "spellings"]], max_rows=20, title="Largest spelling groups")
    rep.table(vals["agency"], max_rows=25, title="agency codes: how many agency_name spellings, and the 3 most "
                                                 "common (all in agg/agency_names.csv)")
    for col in ["status", "open_data_channel_type", "borough", "address_type"]:
        rep.table(vals[col], max_rows=10, title=col)
    rep.table(vals["location_type"], max_rows=10, title="location_type (top)")
    rep.emit()
    return vals


# ---------------------------------------------------------------- 02e: timestamps

def timestamp_checks(con, raw):
    ph = rules.PLACEHOLDER_DATE_PREFIX
    r = con.sql(f"""
        with t as (select status, created_date, closed_date, due_date, resolution_action_updated_date ru,
                          {CR} cr, {CL} cl, try_cast(due_date as timestamp) du,
                          try_cast(resolution_action_updated_date as timestamp) rut from {_src(raw, con)}),
        mx as (select max(ts) m from {_offsets(raw, con)} where {_in_window()})
        select count(*) n,
               count_if(created_date is null) created_null,
               count_if(created_date is not null and cr is null) created_unparseable,
               count_if(cr = date_trunc('day', cr)) created_midnight,
               count_if(closed_date is null) closed_null,
               count_if(closed_date is null and status = 'Closed') closed_null_status_closed,
               count_if(closed_date is not null and status <> 'Closed') closed_set_status_not_closed,
               count_if(closed_date like '{ph}%') closed_1900,
               count_if(cl < cr and closed_date not like '{ph}%') closed_before_created,
               count_if(cl = cr) closed_equals_created,
               count_if(cl > (select m from mx)) closed_after_last_created,
               count_if(cl = date_trunc('day', cl) and closed_date not like '{ph}%') closed_midnight,
               count_if(due_date is null) due_null,
               count_if(due_date like '{ph}%') due_1900,
               count_if(du < cr and due_date not like '{ph}%') due_before_created,
               count_if(rut < cr and ru not like '{ph}%') resolution_update_before_created
        from t""").df().iloc[0].to_dict()
    out = {k: int(v) if pd.notna(v) else 0 for k, v in r.items()}  # count_if over an all-NULL column is NULL
    out["last_in_window_created"] = con.sql(f"select max(ts) from {_offsets(raw, con)} where {_in_window()}").fetchone()[0]
    out["midnight_by_agency"] = con.sql(f"""
        select agency, count(*) n, round(100 * avg(({CR} = date_trunc('day', {CR}))::int), 1) pct_midnight
        from {_src(raw, con)} group by 1 order by n desc""").df()
    out["midnight_by_year"] = con.sql(f"""
        select year({CR}) as year, count(*) as n, round(100 * avg(({CR} = date_trunc('day', {CR}))::int), 1) pct_midnight
        from {_src(raw, con)} where {CR} is not null group by 1 order by 1""").df()
    out["closed_by_year"] = con.sql(f"""
        select year({CL}) closed_year, count(*) n from {_src(raw, con)} where {CL} is not null
        group by 1 order by 1""").df()
    return out


TIMESTAMP_NOTES = {
    "closed_before_created": "excludes 1900 placeholders",
    "closed_after_last_created": "closed after the last in-window created date",
    "closed_midnight": "excludes 1900 placeholders",
    "due_before_created": "excludes 1900 placeholders",
    "resolution_update_before_created": "excludes 1900 placeholders",
}


def timestamps_block(con, raw, out):
    ts = timestamp_checks(con, raw)
    rep = Report("02e_timestamps", out)
    n = max(ts["n"], 1)
    checks = [(k, v) for k, v in ts.items() if isinstance(v, int) and k != "n"]
    rep.add(f"rows: {ts['n']:,} | last created date inside its file's window: {ts['last_in_window_created']}")
    rep.table(pd.DataFrame({"check": [k for k, _ in checks], "rows": [v for _, v in checks],
                            "pct": [round(100 * v / n, 2) for _, v in checks],
                            "note": [TIMESTAMP_NOTES.get(k, "") for k, _ in checks]}), max_rows=20,
              title="Timestamp checks")
    rep.table(ts["midnight_by_year"], max_rows=25, title="Share of created timestamps at exactly midnight, by year")
    rep.table(ts["midnight_by_agency"], max_rows=25, title="... by agency (most rows first)")
    rep.table(ts["closed_by_year"], max_rows=30, title="closed_date by year")
    rep.emit()
    return ts


# ---------------------------------------------------------------- 02f: sparse columns

def sparse_columns(con, raw, profile, min_rows=100):
    sparse = profile[profile.pct_missing_incl_placeholders > 100 * (1 - SPARSE_FILL)].column.tolist()
    sparse = [c for c in sparse if c != "complaint_type"]
    if not sparse:
        return pd.DataFrame(columns=["column", "types_with_fill_ge_5pct", "types_checked", "fill_where_applies_pct",
                                     "top_types_by_fill"])
    fills = ", ".join(
        f"avg(({_q(c)} is not null and trim({_q(c)}) <> '' and not {_placeholder_pred(_q(c))})::int) as f{i}"
        for i, c in enumerate(sparse))
    df = con.sql(f"""select coalesce(complaint_type, '(null)') as complaint_type, count(*) n, {fills}
                     from {_src(raw, con)} group by 1 having count(*) >= {int(min_rows)}""").df()
    out = []
    for i, c in enumerate(sparse):
        t = df[["complaint_type", "n", f"f{i}"]].rename(columns={f"f{i}": "fill"})
        t = t.sort_values(["fill", "n"], ascending=False)
        applies = t[t.fill >= 0.05]
        top = "; ".join(f"{str(r.complaint_type)[:28]}={100 * r.fill:.0f}%" for r in applies.head(4).itertuples())
        out.append({"column": c, "types_with_fill_ge_5pct": len(applies), "types_checked": len(t),
                    "fill_where_applies_pct": 100 * (applies.fill * applies.n).sum() / max(applies.n.sum(), 1),
                    "top_types_by_fill": top})
    return pd.DataFrame(out)


def sparse_block(con, raw, prof, out, agg):
    sparse = sparse_columns(con, raw, prof)
    sparse.to_csv(agg / "sparse_columns.csv", index=False)
    rep = Report("02f_sparse", out)
    rep.add("Columns missing for more than half the rows: which complaint types fill them in (>= 5% fill = applies).")
    rep.table(sparse, max_rows=45)
    rep.emit()


def _write_aggregates(con, raw, agg):
    con.sql(f"""copy (select agency, complaint_type, descriptor, count(*) n,
                             min(created_date) first_seen, max(created_date) last_seen
                      from {_src(raw, con)} group by all order by n desc)
                to {_sql_str((agg / "complaint_types_raw.csv").as_posix())} (header)""")
    con.sql(f"""copy (
                select cast({CR} as date) date, agency, 'created' kind, count(*) n,
                       count_if({CR} = date_trunc('day', {CR})) n_midnight
                from {_src(raw, con)} where {CR} is not null group by all
                union all
                select cast({CL} as date), agency, 'closed', count(*), count_if({CL} = date_trunc('day', {CL}))
                from {_src(raw, con)} where {CL} is not null group by all
                order by 1, 2, 3)
                to {_sql_str((agg / "daily_by_agency.csv").as_posix())} (header)""")


def run(cfg):
    con = cfg.connect()
    raw = ingest.raw_glob(cfg)
    manifest = pd.read_parquet(cfg.work_dir / "manifest.parquet")
    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir
    files_block(con, raw, manifest, out, agg)
    coverage_block(con, raw, out, agg)
    prof = columns_block(con, raw, manifest, out, agg)
    vals = values_block(con, raw, out, agg)
    timestamps_block(con, raw, out)
    sparse_block(con, raw, prof, out, agg)
    _write_aggregates(con, raw, agg)
    vals["complaint_types"].to_csv(agg / "complaint_type_counts.csv", index=False)
