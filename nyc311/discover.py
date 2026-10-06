"""Stage 2: describe the data's structure before answering any question."""
import duckdb
import pandas as pd

from nyc311 import ingest, rules
from nyc311.report import Report

CR = "try_cast(created_date as timestamp)"
CL = "try_cast(closed_date as timestamp)"
TOP_VALUES_MAX_DISTINCT = 200_000
SPARSE_FILL = 0.5


def _src(raw):
    return f"read_parquet('{raw}')"


def _columns(con, raw):
    return [c for c in con.sql(f"describe select * from {_src(raw)}").df().column_name if c != "_file"]


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def _placeholder_pred(col):
    toks = ", ".join("'" + t.replace("'", "''") + "'" for t in rules.PLACEHOLDER_TOKENS)
    return f"(lower(trim({col})) in ({toks}) or {col} like '{rules.PLACEHOLDER_DATE_PREFIX}%')"


def out_of_window_rows(con, raw, days=rules.OUT_OF_WINDOW_DAYS):
    return con.sql(f"""
        with t as (select _file, {CR} ts from {_src(raw)} where {CR} is not null),
        m as (select _file, quantile_disc(ts, 0.5) med from t group by 1)
        select t._file as file, count(*) as n_rows,
               count_if(abs(date_diff('day', m.med, t.ts)) > {days})::bigint as n_outside,
               min(t.ts) as min_ts, any_value(m.med) as median_ts, max(t.ts) as max_ts
        from t join m using (_file) group by 1 order by n_outside desc, file""").df()


def outside_detail(con, raw, days=rules.OUT_OF_WINDOW_DAYS):
    return con.sql(f"""
        with t as (select _file, agency, {CR} ts from {_src(raw)} where {CR} is not null),
        m as (select _file, quantile_disc(ts, 0.5) med from t group by 1)
        select agency, year(ts) as year, count(*) as n
        from t join m using (_file) where abs(date_diff('day', m.med, t.ts)) > {days}
        group by all order by n desc""").df()


def key_duplicates(con, raw):
    cols = _columns(con, raw)
    sig = "md5(concat_ws('|', " + ", ".join(f"coalesce({_q(c)}, '<null>')" for c in cols) + "))"
    r = con.sql(f"""
        with k as (select unique_key from {_src(raw)} group by 1 having count(*) > 1),
        d as (select unique_key, count(*) n, count(distinct {sig}) nv
              from {_src(raw)} semi join k using (unique_key) group by 1)
        select count(*), sum(n), count_if(nv = 1), count_if(nv > 1) from d""").fetchone()
    r = [int(x or 0) for x in r]
    return {"dup_keys": r[0], "dup_rows": r[1], "identical_copies": r[2], "conflicting_versions": r[3]}


def coverage(con, raw):
    daily = con.sql(f"""select cast({CR} as date) d, count(*) n from {_src(raw)}
                        where {CR} is not null group by 1 order by 1""").df()
    spine = pd.date_range(daily.d.min(), daily.d.max(), freq="D")
    s = daily.set_index(pd.to_datetime(daily.d)).n.reindex(spine, fill_value=0)
    med = s.rolling(29, center=True, min_periods=1).median()
    low_mask = (med > 0) & (s < rules.LOW_DAY_RATIO * med) & (s > 0)
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
    monthly.index.name, monthly.columns.name = "year", None
    return {"daily": sd, "low_days": low, "empty_stretches": stretches, "zero_days_short": int(short_zero["count"].sum()),
            "monthly": monthly}


def timestamp_checks(con, raw):
    r = con.sql(f"""
        with t as (select status, created_date, closed_date, due_date, resolution_action_updated_date ru,
                          {CR} cr, {CL} cl, try_cast(due_date as timestamp) du from {_src(raw)}),
        mx as (select max(cr) m from t)
        select count(*) n,
               count_if(created_date is null) created_null,
               count_if(created_date is not null and cr is null) created_unparseable,
               count_if(cr = date_trunc('day', cr)) created_midnight,
               count_if(closed_date is null) closed_null,
               count_if(closed_date is null and status = 'Closed') closed_null_status_closed,
               count_if(closed_date is not null and status <> 'Closed') closed_set_status_not_closed,
               count_if(cl < cr) closed_before_created,
               count_if(cl = cr) closed_equals_created,
               count_if(year(cl) = 1900) closed_1900,
               count_if(cl > (select m from mx)) closed_after_last_created,
               count_if(cl = date_trunc('day', cl)) closed_midnight,
               count_if(due_date is null) due_null,
               count_if(year(du) = 1900) due_1900,
               count_if(du < cr) due_before_created,
               count_if(try_cast(ru as timestamp) < cr) resolution_update_before_created
        from t""").df().iloc[0].to_dict()
    out = {k: int(v) for k, v in r.items()}
    out["midnight_by_agency"] = con.sql(f"""
        select agency, count(*) n, round(100 * avg(({CR} = date_trunc('day', {CR}))::int), 1) pct_midnight
        from {_src(raw)} group by 1 order by n desc""").df()
    out["midnight_by_year"] = con.sql(f"""
        select year({CR}) as year, count(*) as n, round(100 * avg(({CR} = date_trunc('day', {CR}))::int), 1) pct_midnight
        from {_src(raw)} where {CR} is not null group by 1 order by 1""").df()
    out["closed_by_year"] = con.sql(f"""
        select year({CL}) closed_year, count(*) n from {_src(raw)} where {CL} is not null
        group by 1 order by 1""").df()
    return out


def value_lists(con, raw):
    src = _src(raw)
    out = {}
    for col in ["status", "open_data_channel_type", "borough", "address_type"]:
        out[col] = con.sql(f"select {col} as value, count(*) n from {src} group by 1 order by n desc").df()
    out["agency"] = con.sql(f"""
        select agency, count(*) n, count(distinct agency_name) n_names,
               string_agg(distinct agency_name, ' | ') agency_names
        from {src} group by 1 order by n desc""").df()
    out["location_type"] = con.sql(f"select location_type as value, count(*) n from {src} group by 1 order by n desc").df()
    types = con.sql(f"select complaint_type, count(*) n from {src} group by 1 order by n desc").df()
    out["n_complaint_types"] = len(types)
    out["complaint_types"] = types
    out["type_variants"] = con.sql(f"""
        with t as (select complaint_type ct, count(*) n from {src} group by 1),
        k as (select regexp_replace(lower(trim(regexp_replace(ct, '\\s+', ' ', 'g'))), 's$', '') as key, ct, n from t)
        select key, count(*) n_spellings, sum(n)::bigint n,
               string_agg(ct || ' (' || n || ')', ' | ' order by n desc) spellings
        from k group by all having count(*) > 1 order by n desc""").df()
    return out


def column_profile(con, raw):
    cols = _columns(con, raw)
    parts = ["count(*)"]
    for c in cols:
        qc = _q(c)
        parts += [f"count({qc})", f"count_if(trim({qc}) = '')", f"count_if({_placeholder_pred(qc)})",
                  f"approx_count_distinct({qc})"]
    r = con.sql(f"select {', '.join(parts)} from {_src(raw)}").fetchone()
    n = r[0]
    rows = []
    for i, c in enumerate(cols):
        nn, empty, ph, distinct = r[1 + 4 * i: 5 + 4 * i]
        top = ""
        if distinct <= TOP_VALUES_MAX_DISTINCT:
            tv = con.sql(f"""select {_q(c)} v, count(*) k from {_src(raw)} where {_q(c)} is not null
                             group by 1 order by 2 desc limit 3""").fetchall()
            top = "; ".join(f"{str(v)[:30]}={100 * k / n:.1f}%" for v, k in tv)
        rows.append({"column": c, "n_rows": n, "n_null": n - nn, "n_empty": empty, "n_placeholder": ph,
                     "pct_null": 100 * (n - nn) / n, "pct_missing_incl_placeholders": 100 * (n - nn + empty + ph) / n,
                     "approx_distinct": distinct, "top_values": top})
    return pd.DataFrame(rows)


def sparse_columns(con, raw, profile, min_rows=100):
    out = []
    for c in profile[profile.pct_missing_incl_placeholders > 100 * (1 - SPARSE_FILL)].column:
        qc = _q(c)
        filled = f"({qc} is not null and trim({qc}) <> '' and not {_placeholder_pred(qc)})"
        df = con.sql(f"""
            with t as (select complaint_type, count(*) n, avg({filled}::int) fill
                       from {_src(raw)} group by 1 having count(*) >= {min_rows})
            select * from t order by fill desc, n desc""").df()
        applies = df[df.fill >= 0.05]
        top = "; ".join(f"{r.complaint_type[:28]}={100 * r.fill:.0f}%" for r in applies.head(4).itertuples())
        out.append({"column": c, "types_with_fill_ge_5pct": len(applies), "types_checked": len(df),
                    "fill_where_applies_pct": 100 * (applies.fill * applies.n).sum() / max(applies.n.sum(), 1),
                    "top_types_by_fill": top})
    return pd.DataFrame(out)


def _write_aggregates(con, raw, agg):
    con.sql(f"""copy (select agency, complaint_type, descriptor, count(*) n,
                             min(created_date) first_seen, max(created_date) last_seen
                      from {_src(raw)} group by all order by n desc)
                to '{(agg / "complaint_types_raw.csv").as_posix()}' (header)""")
    con.sql(f"""copy (
                select cast({CR} as date) date, agency, 'created' kind, count(*) n,
                       count_if({CR} = date_trunc('day', {CR})) n_midnight
                from {_src(raw)} where {CR} is not null group by all
                union all
                select cast({CL} as date), agency, 'closed', count(*), count_if({CL} = date_trunc('day', {CL}))
                from {_src(raw)} where {CL} is not null group by all
                order by 1, 2, 3)
                to '{(agg / "daily_by_agency.csv").as_posix()}' (header)""")


def run(cfg):
    con = cfg.connect()
    raw = ingest.raw_glob(cfg)
    manifest = pd.read_parquet(cfg.work_dir / "manifest.parquet")
    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir

    rep = Report("02a_files", out)
    n_rows = con.sql(f"select count(*) from {_src(raw)}").fetchone()[0]
    rep.add(f"duckdb {duckdb.__version__} | threads {cfg.threads} | rows in raw parquet: {n_rows:,} "
            f"| manifest records: {int(manifest.n_records.sum()):,}")
    per_file = manifest.n_records.value_counts()
    rep.add(f"files: {len(manifest):,} | rows per file: most common {per_file.index[0]:,} "
            f"({per_file.iloc[0]:,} files), min {manifest.n_records.min():,}, max {manifest.n_records.max():,}")
    odd = manifest[manifest.n_records != per_file.index[0]][["file", "n_records", "parse_ok", "min_created", "max_created"]]
    rep.table(odd, max_rows=15, title="Files with an unusual row count")
    key_sets = manifest["keys"].str.split("|")
    all_keys = sorted({k for ks in key_sets for k in ks if k})
    presence = pd.DataFrame({"column": all_keys,
                             "files_with_column": [int(key_sets.apply(lambda ks, k=k: k in ks).sum()) for k in all_keys]})
    rep.add("", f"distinct columns across files: {len(all_keys)}")
    rep.table(presence[presence.files_with_column < len(manifest)], max_rows=45,
              title="Columns missing from some files (blank = every file has every column)")
    dups = key_duplicates(con, raw)
    rep.add("", "duplicate unique_key: " + ", ".join(f"{k}={v:,}" for k, v in dups.items()))
    win = out_of_window_rows(con, raw)
    win.to_csv(agg / "file_windows.csv", index=False)
    rep.add("", f"rows created >{rules.OUT_OF_WINDOW_DAYS} days from their file's median date: "
                f"{int(win.n_outside.sum()):,} in {int((win.n_outside > 0).sum()):,} files")
    rep.table(win[win.n_outside > 0], max_rows=15, title="Files holding out-of-window rows")
    rep.table(outside_detail(con, raw), max_rows=15, title="Out-of-window rows by agency and year")
    rep.emit()

    cov = coverage(con, raw)
    rep = Report("02b_coverage", out)
    d = cov["daily"]
    rep.add(f"created dates: {d.index.min().date()} to {d.index.max().date()} | days with data: {(d > 0).sum():,} "
            f"| zero days inside short gaps: {cov['zero_days_short']:,}")
    rep.table(cov["monthly"].reset_index(), max_rows=40, title="Requests created per month (rows = year, columns = month)")
    rep.table(cov["empty_stretches"], max_rows=15, title=f"Empty stretches (>= {rules.EMPTY_STRETCH_DAYS} days with no requests)")
    rep.table(cov["low_days"].sort_values("n"), max_rows=30,
              title=f"Low days (< {rules.LOW_DAY_RATIO:.0%} of the 29-day rolling median)")
    rep.emit()

    prof = column_profile(con, raw)
    prof.to_csv(agg / "column_profile.csv", index=False)
    rep = Report("02c_columns", out)
    rep.add(f"placeholder tokens (PROVISIONAL, lowercased): {rules.PLACEHOLDER_TOKENS} + dates starting "
            f"'{rules.PLACEHOLDER_DATE_PREFIX}'")
    rep.table(prof[["column", "pct_null", "n_empty", "n_placeholder", "pct_missing_incl_placeholders",
                    "approx_distinct", "top_values"]], max_rows=60, title="Per-column missingness and top values")
    rep.emit()

    vals = value_lists(con, raw)
    vals["type_variants"].to_csv(agg / "type_variants.csv", index=False)
    rep = Report("02d_values", out)
    for col in ["status", "open_data_channel_type", "borough", "address_type"]:
        rep.table(vals[col], max_rows=12, title=col)
    rep.table(vals["agency"], max_rows=30, title="agency codes and their agency_name spellings")
    rep.table(vals["location_type"], max_rows=12, title="location_type (top)")
    rep.add("", f"distinct complaint_type values: {vals['n_complaint_types']:,}")
    rep.table(vals["type_variants"][["key", "n_spellings", "n", "spellings"]], max_rows=25,
              title="complaint_type spellings that differ only by case/spacing/trailing s")
    rep.emit()

    ts = timestamp_checks(con, raw)
    rep = Report("02e_timestamps", out)
    rep.add(", ".join(f"{k}={v:,}" for k, v in ts.items() if isinstance(v, int)))
    rep.table(ts["midnight_by_agency"], max_rows=30, title="Share of created timestamps at exactly midnight, by agency")
    rep.table(ts["midnight_by_year"], max_rows=20, title="... by year")
    rep.table(ts["closed_by_year"], max_rows=30, title="closed_date by year")
    rep.emit()

    sparse = sparse_columns(con, raw, prof)
    sparse.to_csv(agg / "sparse_columns.csv", index=False)
    rep = Report("02f_sparse", out)
    rep.add("Columns missing for more than half the rows: which complaint types fill them in (>= 5% fill = applies).")
    rep.table(sparse, max_rows=40)
    rep.emit()

    _write_aggregates(con, raw, agg)
    vals["complaint_types"].to_csv(agg / "complaint_type_counts.csv", index=False)
