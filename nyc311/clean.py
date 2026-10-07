"""Stage 3: one typed, flagged row per raw row (nothing dropped). Decisions live in rules.py."""
import re

import pandas as pd

from nyc311 import geo, ingest, rules
from nyc311.report import Report

CR = "try_cast(r.created_date as timestamp)"
_ZIP = re.compile(r"^(\d{5})(?:-\d{4})?$")
BORO_NAME = {v: k for k, v in rules.BORO_CODE.items()}

TEXT_COLUMNS = ["agency", "agency_name", "complaint_type", "descriptor", "location_type", "incident_address",
                "street_name", "cross_street_1", "cross_street_2", "intersection_street_1", "intersection_street_2",
                "address_type", "city", "landmark", "facility_type", "community_board", "bbl", "park_facility_name",
                "park_borough", "resolution_description", "vehicle_type", "taxi_company_borough",
                "taxi_pick_up_location", "bridge_highway_name", "bridge_highway_direction", "road_ramp",
                "bridge_highway_segment"]


def clean_path(cfg):
    return cfg.work_dir / "clean.parquet"


def clean_glob(cfg):
    return "read_parquet('" + clean_path(cfg).as_posix().replace("'", "''") + "')"


def parse_community_board(values):
    rows = []
    for v in pd.unique(pd.Series(values, dtype=object)):
        boro = cd = special = None
        if isinstance(v, str):
            head, _, tail = v.strip().partition(" ")
            if tail in rules.CD_COUNT:
                boro = tail
                if head.isdigit():
                    n, code = int(head), rules.BORO_CODE[tail] * 100 + int(head)
                    if 1 <= n <= rules.CD_COUNT[tail]:
                        cd = code
                    elif n > rules.CD_COUNT[tail]:
                        special = code
        rows.append({"community_board": v, "cb_borough": boro, "cb_boro_cd": cd, "cb_special_cd": special})
    return pd.DataFrame(rows, dtype=object)


def clean_zip(values):
    def one(v):
        if not isinstance(v, str):
            return None
        m = _ZIP.match(v.strip())
        return m.group(1) if m and re.match(rules.NYC_ZIP_PATTERN, m.group(1)) else None
    s = pd.Series(values, dtype=object)
    return pd.Series([one(v) for v in s], index=s.index, dtype=object)


def incomplete_days(con, raw, manifest):
    """Days whose rows are partly in files we don't have (each day's rows are split across about two
    consecutive files): after the last file's median date, the late days of any truncated file, and the
    days between the median dates of the files on either side of a missing batch number."""
    d = con.sql(f"""
        with r as (select _file, {CR} ts from {raw} r where r.created_date is not null),
        m as (select _file, quantile_disc(ts, 0.5) med from r group by 1),
        j as (select _file, ts, med, abs(date_diff('day', med, ts)) <= {rules.OUT_OF_WINDOW_DAYS} inwin
              from r join m using (_file))
        select _file, cast(any_value(med) as date) p50, max(case when inwin then cast(ts as date) end) mx,
               avg(inwin::int) ordered_share
        from j group by 1 order by 1""").df()
    days = set()
    d = d[d.ordered_share >= rules.ORDERED_FILE_SHARE]
    if d.empty:
        return days
    last = d.iloc[-1]
    days |= set(pd.date_range(last.p50 + pd.Timedelta(days=1), last.mx).date)
    for f in manifest.loc[~manifest.parse_ok, "file"]:
        row = d[d._file == f]
        if len(row):
            days |= set(pd.date_range(row.p50.iloc[0], row.mx.iloc[0]).date)
    d["num"] = d._file.str.extract(r"_(\d+)\.json$", expand=False).astype(float)
    numbered = d.dropna(subset=["num"]).sort_values("num")
    # d holds ordered files only, so a gap next to an out-of-order file (or between two ordered files that are
    # not neighbors) is checked against the batch numbers actually present in the manifest.
    present = set(manifest.file.str.extract(r"_(\d+)\.json$", expand=False).dropna().astype(int))
    for prev, nxt in zip(numbered.itertuples(), numbered.iloc[1:].itertuples()):
        between = set(range(int(prev.num) + 1, int(nxt.num)))
        if not between or (between & present):
            continue
        days |= set(pd.date_range(prev.p50 + pd.Timedelta(days=1), nxt.p50 - pd.Timedelta(days=1)).date)
    return days


def unresolved_sql(col):
    not_open = rules.CLOSED_STATUSES + rules.NOT_UNRESOLVED_STATUSES + rules.TEST_STATUSES + rules.NULL_IF["status"]
    vals = ", ".join("'" + v.replace("'", "''") + "'" for v in not_open)
    return f"coalesce({col} is not null and trim({col}) not in ({vals}), false)"


def _nullif(col, values):
    vals = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"case when trim({col}) in ({vals}) then null else {col} end"


def _in(col, values):
    return f"coalesce(trim({col}) in (" + ", ".join(f"'{v}'" for v in values) + "), false)"


def _register(con, name, df, int_cols=()):
    df = df.copy()
    for c in int_cols:
        df[c] = pd.array(df[c], dtype="Int64")
    con.register(name, df)


def _raw_rows(cfg):
    """The raw Parquet plus the part file and row number, which together identify each raw row."""
    path = ingest.raw_glob(cfg).replace("'", "''")
    return f"read_parquet('{path}', union_by_name=true, filename=true, file_row_number=true)"


def run(cfg):
    from nyc311 import taxonomy

    con = cfg.connect()
    raw = ingest.raw_source(cfg)
    manifest = pd.read_parquet(cfg.work_dir / "manifest.parquet")

    _register(con, "cb_lookup", parse_community_board(
        con.sql(f"select distinct community_board from {raw}").df().community_board),
        int_cols=["cb_boro_cd", "cb_special_cd"])
    zips = con.sql(f"select distinct incident_zip from {raw} where incident_zip is not null").df()
    zips["zip"] = clean_zip(zips.incident_zip)
    _register(con, "zip_lookup", zips)
    tax = taxonomy.classify(con.sql(f"select distinct complaint_type from {raw}").df().complaint_type)
    if tax.complaint_type.duplicated().any():
        raise ValueError("taxonomy.classify returned more than one row for a complaint_type")
    _register(con, "tax_lookup", tax)
    inc = pd.DataFrame({"d": sorted(incomplete_days(con, raw, manifest))})
    inc["d"] = pd.to_datetime(inc.d)
    _register(con, "incomplete", inc)

    # Everything that needs all rows at once (file medians, duplicate keys, coordinates to place, the usual
    # borough of each ZIP) is computed first from a few columns. The full-width pass below then only joins
    # small tables, so it streams to Parquet instead of holding all rows in memory.
    con.execute(f"create or replace temp table file_med as "
                f"select _file, quantile_disc({CR}, 0.5) med from {raw} r group by 1")
    rows = _raw_rows(cfg)
    con.execute(f"""
        create or replace temp table dup_rows as
        select filename, file_row_number from (
            select filename, file_row_number,
                   row_number() over (partition by unique_key order by _file, filename, file_row_number) rn
            from {rows} semi join (select unique_key from {raw} group by 1 having count(*) > 1) k
            using (unique_key))
        where rn > 1""")
    pts = con.sql(f"""
        select distinct lat as latitude, lon as longitude from (
            select try_cast(r.latitude as double) lat, try_cast(r.longitude as double) lon from {raw} r
            left join cb_lookup cb on cb.community_board = r.community_board where cb.cb_boro_cd is null)
        where lat is not null and lon is not null""").df()
    pts["geo_cd"] = geo.assign_districts(pts.latitude, pts.longitude) if geo.DISTRICTS_GEOJSON.exists() else None
    _register(con, "geo_lookup", pts, int_cols=["geo_cd"])
    con.execute(f"""
        create or replace temp table zip_boro as
        select zip, arg_max(b, n) as zip_borough from (
            select z.zip, {_nullif('r.borough', rules.NULL_IF['borough'])} b, count(*) n
            from {raw} r join zip_lookup z on z.incident_zip = r.incident_zip group by all)
        where zip is not null and b is not null group by zip""")

    text = ",\n".join(f"{_nullif('r.' + c, rules.NULL_IF[c]) if c in rules.NULL_IF else 'r.' + c} as {c}"
                      for c in TEXT_COLUMNS)
    ph = rules.PLACEHOLDER_YEAR
    geo_boro = "case cast(g.geo_cd // 100 as int) " + " ".join(
        f"when {k} then '{v}'" for k, v in BORO_NAME.items()) + " end"
    con.execute(f"""
        copy (
        select c.* exclude (borough_reported, cb_borough, cb_boro_cd, cb_special_cd),
               coalesce(c.borough_reported, c.cb_borough, {geo_boro}, zb.zip_borough) as borough,
               case when c.borough_reported is not null then 'reported'
                    when c.cb_borough is not null then 'community_board'
                    when g.geo_cd is not null then 'latlon'
                    when zb.zip_borough is not null then 'zip' end as borough_source,
               coalesce(c.cb_boro_cd, case when c.cb_special_cd is null and g.geo_cd % 100 <= 18
                                           then g.geo_cd end) as boro_cd,
               case when c.cb_boro_cd is not null then 'community_board'
                    when c.cb_special_cd is null and g.geo_cd % 100 <= 18 then 'latlon' end as cd_source,
               coalesce(c.cb_special_cd, case when c.cb_boro_cd is null and g.geo_cd % 100 > 18
                                              then g.geo_cd end) as special_area_cd,
               coalesce(c.duration_h > {rules.LONG_DURATION_HOURS}, false) as closed_after_1yr
        from (
        select r.unique_key, r._file,
               r.cr as created,
               r.cl as closed,
               case when year(r.du0) = {ph} then null else r.du0 end as due,
               try_cast(r.resolution_action_updated_date as timestamp) as resolution_updated,
               {text},
               {_nullif('r.status', rules.NULL_IF['status'])} as status,
               {_nullif('r.open_data_channel_type', rules.NULL_IF['open_data_channel_type'])} as channel,
               r.incident_zip as zip_raw, z.zip,
               try_cast(r.latitude as double) as latitude, try_cast(r.longitude as double) as longitude,
               {_nullif('r.borough', rules.NULL_IF['borough'])} as borough_reported,
               cb.cb_borough, cb.cb_boro_cd, cb.cb_special_cd,
               t.canonical_type, t.category, t.origin, t.is_info_request, t.needs_review as type_needs_review,
               coalesce(abs(date_diff('day', m.med, r.cr)) > {rules.OUT_OF_WINDOW_DAYS}, false) as out_of_file_order,
               coalesce(cast(r.cr as date) in (select cast(d as date) from incomplete), false) as possibly_incomplete_day,
               coalesce(year(r.cl0) = {ph}, false) as closed_placeholder,
               coalesce(r.cl < r.cr, false) as closed_before_created,
               coalesce(r.cl = r.cr, false) as closed_at_creation,
               coalesce(r.cr = date_trunc('day', r.cr), false) as date_only_created,
               coalesce(r.cl = date_trunc('day', r.cl), false) as date_only_closed,
               {_in('r.status', rules.TEST_STATUSES)} as is_test,
               {unresolved_sql('r.status')} as unresolved,
               coalesce(r.cl is not null and trim(r.status) <> 'Closed', false) as status_conflict,
               d.file_row_number is not null as dup_key,
               case when trim(r.status) = 'Closed' and r.cl >= r.cr
                    then date_diff('second', r.cr, r.cl) / 3600.0 end as duration_h
        from (select *, case when year(cl0) = {ph} then null else cl0 end cl from
                (select *, {CR.replace('r.', '')} cr, try_cast(closed_date as timestamp) cl0,
                        try_cast(due_date as timestamp) du0 from {rows})) r
        join file_med m using (_file)
        left join cb_lookup cb on cb.community_board = r.community_board
        left join zip_lookup z on z.incident_zip = r.incident_zip
        left join tax_lookup t on t.complaint_type = r.complaint_type
        left join dup_rows d on d.filename = r.filename and d.file_row_number = r.file_row_number
        ) c left join geo_lookup g using (latitude, longitude) left join zip_boro zb using (zip)
        ) to '{clean_path(cfg).as_posix().replace("'", "''")}' (format parquet, compression zstd)""")
    _report(con, cfg)


def _report(con, cfg):
    src = clean_glob(cfg)
    rep = Report("03_clean", cfg.out_dir)
    n = con.sql(f"select count(*) from {src}").fetchone()[0]
    rep.add(f"rows in clean table: {n:,} (nothing dropped; flags below)")
    flags = ["dup_key", "out_of_file_order", "possibly_incomplete_day", "is_test", "unresolved", "status_conflict",
             "closed_placeholder", "closed_before_created", "closed_at_creation", "closed_after_1yr",
             "date_only_created",
             "date_only_closed", "is_info_request", "type_needs_review"]
    counts = con.sql("select " + ", ".join(f"count_if({f})" for f in flags) + f" from {src}").fetchone()
    rep.table(pd.DataFrame({"flag": flags, "rows": counts, "pct": [100 * c / n for c in counts]}), title="Flags")
    rep.table(con.sql(f"""select coalesce(borough_source, '(unrecovered)') as borough_source, count(*) as n,
                          round(100 * count(*) / {n}, 2) as pct from {src} group by 1 order by n desc""").df(),
              title="Where borough came from")
    rep.table(con.sql(f"""select coalesce(cd_source, '(none)') as cd_source, count(*) as n,
                          round(100 * count(*) / {n}, 2) as pct from {src} group by 1 order by n desc""").df(),
              title="Where community district came from")
    rep.table(con.sql(f"""select agency, count(*) n, round(100 * avg((boro_cd is not null)::int), 1) pct_with_cd,
                          round(100 * avg((borough is not null)::int), 1) pct_with_borough
                          from {src} group by 1 order by n desc""").df(), max_rows=12,
              title="District coverage by agency (after recovery)")
    rep.add("", "possibly incomplete days: " + ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in con.sql(
        f"select distinct created::date d from {src} where possibly_incomplete_day order by 1").df().d))
    rep.emit()
