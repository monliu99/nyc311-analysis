"""Q1: which columns have missing values, and how much, counting every way the data writes "missing".

Four measures per raw column:
  1. missing as stored: NULL or empty text;
  2. missing once placeholders count: placeholder text (rules.PLACEHOLDER_TOKENS), 1900 dates, community_board
     'Unspecified <BOROUGH>' (borough known, district not), ZIPs that are not NYC 5-digit codes; every encoding
     is listed with its row count;
  3. missing where the column applies: rows of complaint types that fill the column for at least APPLIES_MIN_FILL
     of their rows, after the structural rules in STRUCTURAL (intersection streets only for intersection
     addresses, close dates only for closed requests, ...);
  4. for location fields, how much the cleaning stage recovered (borough_source, cd_source, ZIP+4).
Plus how missingness of the key fields changes over time. Raw columns come from the raw Parquet, recovery and
time trends from the clean table (one row per raw row). Every number is computed.
"""
from __future__ import annotations

import pandas as pd

from nyc311 import clean, ingest, rules
from nyc311.report import Report

STAGE = "q01_missing"
APPLIES_MIN_FILL = 0.05     # a complaint type uses a column if it fills it for at least 5% of its rows
SMALL_TYPE_ROWS = 100       # types smaller than this get a noisy applicability call (reported, not dropped)
MIN_PERIOD_ROWS = 500       # periods with fewer rows are left out of the time table (stray backfilled rows)
DATE_COLUMNS = ("created_date", "closed_date", "due_date", "resolution_action_updated_date")
ZIP_COLUMN = "incident_zip"
PARTIAL_PATTERNS = {"community_board": "unspecified %"}  # 'Unspecified QUEENS': borough known, district missing

_INTERSECTION = "coalesce(trim(address_type), '') = 'INTERSECTION'"
_NOT_INTERSECTION = f"not ({_INTERSECTION})"
STRUCTURAL = {
    "intersection_street_1": (_INTERSECTION, "intersection addresses only"),
    "intersection_street_2": (_INTERSECTION, "intersection addresses only"),
    "incident_address": (_NOT_INTERSECTION, "not for intersection addresses"),
    "street_name": (_NOT_INTERSECTION, "not for intersection addresses"),
    "cross_street_1": (_NOT_INTERSECTION, "not for intersection addresses"),
    "cross_street_2": (_NOT_INTERSECTION, "not for intersection addresses"),
    "closed_date": ("coalesce(trim(status), '') = 'Closed'", "closed requests only"),
}

KIND_NULL, KIND_EMPTY = "NULL", "empty text"
KIND_TOKEN = "placeholder text"
KIND_DATE = "placeholder date (1900)"
KIND_PARTIAL = "partial placeholder"
KIND_ZIP = "invalid ZIP (not NYC 5-digit)"
KIND_DATE_ONLY = "time of day missing (00:00:00)"  # listed, not counted as missing
MISSING_KINDS = [KIND_NULL, KIND_EMPTY, KIND_TOKEN, KIND_DATE, KIND_PARTIAL, KIND_ZIP]

NOT_MISSING = "present in raw data"
STILL_MISSING = "still missing"

PERIOD_MEASURES = {  # clean-table expression (true = missing) -> label; denominators in _period_sql
    "pct_borough_raw": "borough missing in raw",
    "pct_borough_after": "borough missing after recovery",
    "pct_district_raw": "community district missing in raw",
    "pct_district_after": "community district missing after recovery",
    "pct_zip": "ZIP missing or invalid",
    "pct_coords": "coordinates missing",
    "pct_address_type": "address_type missing",
    "pct_location_type": "location_type missing",
    "pct_channel": "channel missing (UNKNOWN or NULL)",
    "pct_created_time": "opening time of day missing",
    "pct_closed_time": "closing time of day missing (of closed)",
    "pct_closed_missing": "close date missing (of status Closed)",
}


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def _lit(values):
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _columns(con, raw):
    return [c for c in con.sql(f"describe select * from {raw}").df().column_name if c != "_file"]


def _register_bad_zips(con, raw, columns):
    """Distinct incident_zip values (trimmed) that are text but not an NYC ZIP and not already a placeholder."""
    if ZIP_COLUMN in columns:
        z = con.sql(f"select distinct trim({ZIP_COLUMN}) as v from {raw} where trim({ZIP_COLUMN}) <> ''").df()
    else:
        z = pd.DataFrame({"v": pd.Series([], dtype=object)})
    tokens = set(rules.PLACEHOLDER_TOKENS)
    bad = z[clean.clean_zip(z.v).isna().values & ~z.v.str.lower().isin(tokens).values
            & ~z.v.str.startswith(rules.PLACEHOLDER_DATE_PREFIX).values]
    con.register("q01_bad_zip", bad.reset_index(drop=True))


def _kind_sql(col):
    """CASE expression naming how a value encodes missingness (NULL when it is a real value)."""
    q = _q(col)
    parts = [f"when {q} is null then '{KIND_NULL}'", f"when trim({q}) = '' then '{KIND_EMPTY}'",
             f"when starts_with(trim({q}), '{rules.PLACEHOLDER_DATE_PREFIX}') then '{KIND_DATE}'",
             f"when lower(trim({q})) in ({_lit(rules.PLACEHOLDER_TOKENS)}) then '{KIND_TOKEN}'"]
    if col in PARTIAL_PATTERNS:
        parts.append(f"when lower(trim({q})) like '{PARTIAL_PATTERNS[col]}' then '{KIND_PARTIAL}'")
    if col == ZIP_COLUMN:
        parts.append(f"when trim({q}) in (select v from q01_bad_zip) then '{KIND_ZIP}'")
    if col in DATE_COLUMNS:
        ts = f"try_cast({q} as timestamp)"
        parts.append(f"when {ts} = date_trunc('day', {ts}) then '{KIND_DATE_ONLY}'")
    return "case " + " ".join(parts) + " end"


def _missing_sql(col):
    """True when the value is missing in any encoding (measure 2)."""
    q = _q(col)
    ors = [f"{q} is null", f"trim({q}) = ''", f"starts_with(trim({q}), '{rules.PLACEHOLDER_DATE_PREFIX}')",
           f"lower(trim({q})) in ({_lit(rules.PLACEHOLDER_TOKENS)})"]
    if col in PARTIAL_PATTERNS:
        ors.append(f"lower(trim({q})) like '{PARTIAL_PATTERNS[col]}'")
    if col == ZIP_COLUMN:
        ors.append(f"trim({q}) in (select v from q01_bad_zip)")
    return "(" + " or ".join(ors) + ")"


# ---------------------------------------------------------------- measures 1-2: stored and encoded missingness

def encodings(con, raw, columns):
    """One row per (column, kind, value): every way a column encodes missing, with its rows."""
    frames = []
    for c in columns:
        kind = _kind_sql(c)
        df = con.sql(f"""
            select kind, case when kind in ('{KIND_NULL}', '{KIND_DATE_ONLY}') then null else trim(v) end as value,
                   count(*)::bigint as n
            from (select {kind} as kind, {_q(c)} as v from {raw}) where kind is not null
            group by all""").df()
        df.insert(0, "column", c)
        frames.append(df)
    enc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["column", "kind", "value", "n"])
    enc["counts_as_missing"] = enc.kind.isin(MISSING_KINDS)
    return enc.sort_values(["column", "n"], ascending=[True, False]).reset_index(drop=True)


def _encoding_text(e, k=3):
    """Short list of a column's placeholder encodings, largest first: 'Unspecified' 89,164; ..."""
    items = []
    ph = e[e.counts_as_missing & ~e.kind.isin([KIND_NULL, KIND_EMPTY, KIND_ZIP])]
    for r in ph.head(k).itertuples(index=False):
        items.append(f"'{r.value}' {r.n:,}")
    if len(ph) > k:
        items.append(f"+{len(ph) - k} more ({int(ph.n.iloc[k:].sum()):,})")
    z = e[e.kind == KIND_ZIP]
    if len(z):
        items.append(f"invalid ZIP {int(z.n.sum()):,} ({len(z)} values, e.g. {', '.join(z.value.head(2))})")
    d = e[e.kind == KIND_DATE_ONLY]
    if len(d):
        items.append(f"date only {int(d.n.sum()):,} (not counted)")
    return "; ".join(items)


def column_summary(enc, columns, n_rows):
    rows = []
    for c in columns:
        e = enc[enc.column == c]
        get = lambda kinds: int(e[e.kind.isin(kinds)].n.sum())
        n_null, n_empty = get([KIND_NULL]), get([KIND_EMPTY])
        n_missing = get(MISSING_KINDS)
        rows.append({"column": c, "n_rows": n_rows, "n_null": n_null, "n_empty": n_empty,
                     "n_placeholder": n_missing - n_null - n_empty, "n_missing": n_missing,
                     "pct_stored": 100 * (n_null + n_empty) / max(n_rows, 1),
                     "pct_with_placeholders": 100 * n_missing / max(n_rows, 1),
                     "encodings": _encoding_text(e)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- measure 3: missing where the column applies

def applicability(con, raw, columns):
    """Per (column, complaint type): rows the structural rule allows, rows filled, fill %, and whether it applies."""
    sel, agg = [], []
    for i, c in enumerate(columns):
        struct = STRUCTURAL.get(c, ("true", ""))[0]
        sel.append(f"({struct}) as s{i}, {_missing_sql(c)} as m{i}")
        agg.append(f"count_if(s{i})::bigint as a{i}, count_if(s{i} and not m{i})::bigint as f{i}")
    wide = con.sql(f"""select ct, count(*)::bigint as n_type_rows, {', '.join(agg)}
                       from (select coalesce(complaint_type, '(null)') as ct, {', '.join(sel)} from {raw})
                       group by ct""").df()
    frames = []
    for i, c in enumerate(columns):
        t = pd.DataFrame({"column": c, "complaint_type": wide.ct, "n_type_rows": wide.n_type_rows,
                          "n_structural": wide[f"a{i}"], "n_filled": wide[f"f{i}"]})
        frames.append(t[t.n_structural > 0])
    app = pd.concat(frames, ignore_index=True)
    app["fill_pct"] = 100 * app.n_filled / app.n_structural
    app["applies"] = app.fill_pct >= 100 * APPLIES_MIN_FILL
    return app.sort_values(["column", "n_structural"], ascending=[True, False]).reset_index(drop=True), len(wide)


def applies_summary(app, columns, n_types):
    rows = []
    for c in columns:
        a = app[(app.column == c) & app.applies]
        n_app = int(a.n_structural.sum())
        miss = int((a.n_structural - a.n_filled).sum())
        filled_total = int(app[app.column == c].n_filled.sum())
        rule = STRUCTURAL.get(c, ("", ""))[1]
        scope = "all types" if len(a) == n_types else f"{len(a)} of {n_types} types"
        rows.append({"column": c, "n_types_apply": len(a), "n_applies": n_app, "n_missing_where_applies": miss,
                     "pct_missing_where_applies": 100 * miss / n_app if n_app else 0.0,
                     "n_filled_where_not_applies": filled_total - int(a.n_filled.sum()),
                     "applies_to": f"{rule}; {scope}" if rule else scope,
                     "top_types": "; ".join(f"{r.complaint_type} {r.fill_pct:.0f}%"
                                            for r in a.sort_values("n_structural", ascending=False)
                                            .head(3).itertuples(index=False))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- measure 4: what cleaning recovered

def _register_cb(con, src):
    cb = clean.parse_community_board(con.sql(f"select distinct community_board from {src}").df().community_board)
    for c in ["cb_boro_cd", "cb_special_cd"]:
        cb[c] = pd.array(cb[c], dtype="Int64")
    con.register("q01_cb", cb[["community_board", "cb_boro_cd", "cb_special_cd"]])


def recovery(con, src):
    """For borough, community district, ZIP and coordinates: present, recovered (by source), or still missing."""
    df = con.sql(f"""
        with t as (select c.*, cb.cb_boro_cd, cb.cb_special_cd from {src} c
                   left join q01_cb cb on cb.community_board is not distinct from c.community_board)
        select 'borough' as field,
               case when borough_source = 'reported' then '{NOT_MISSING}'
                    when borough_source = 'community_board' then 'recovered: community_board text'
                    when borough_source = 'latlon' then 'recovered: coordinates (district polygon)'
                    when borough_source = 'zip' then 'recovered: usual borough of the ZIP'
                    else '{STILL_MISSING}' end as status, count(*)::bigint as n from t group by all
        union all
        select 'community district',
               case when cb_boro_cd is not null then '{NOT_MISSING}'
                    when cb_special_cd is not null then 'park/airport code in community_board (not a district)'
                    when cd_source = 'latlon' then 'recovered: coordinates (district polygon)'
                    when special_area_cd is not null then 'coordinates fall in a park/airport area'
                    else '{STILL_MISSING}' end, count(*)::bigint from t group by all
        union all
        select 'zip',
               case when zip_raw is null or trim(zip_raw) = '' then '{STILL_MISSING}: NULL in raw'
                    when zip is null then '{STILL_MISSING}: placeholder or invalid (set to NULL)'
                    when zip <> trim(zip_raw) then 'ZIP+4 cut to 5 digits'
                    else '{NOT_MISSING}' end, count(*)::bigint from t group by all
        union all
        select 'coordinates', case when latitude is null or longitude is null then '{STILL_MISSING}'
                                   else '{NOT_MISSING}' end, count(*)::bigint from t group by all""").df()
    tot = df.groupby("field").n.transform("sum")
    df["pct"] = 100 * df.n / tot
    order = {"borough": 0, "community district": 1, "zip": 2, "coordinates": 3}
    df["o"] = df.field.map(order)
    df["s"] = df.status.map(lambda s: 0 if s == NOT_MISSING else 2 if s.startswith(STILL_MISSING) else 1)
    return df.sort_values(["o", "s", "n"], ascending=[True, True, False]).drop(columns=["o", "s"]).reset_index(drop=True)


# ---------------------------------------------------------------- over time

def period_unit(con, src):
    """year for two or more years of data, month for three or more months, else week (span of in-order rows)."""
    span = con.sql(f"""select date_diff('day', min(created), max(created)) from {src}
                       where created is not null and not out_of_file_order""").fetchone()[0] or 0
    return "year" if span >= 730 else "month" if span >= 90 else "week"


def period_sql(unit, col="created"):
    if unit == "year":
        return f"strftime({col}, '%Y')"
    if unit == "month":
        return f"strftime({col}, '%Y-%m')"
    return f"strftime(date_trunc('week', {col}), '%Y-%m-%d')"


def by_period(con, src, unit):
    p = period_sql(unit)
    pct = lambda cond, among="true": f"round(100 * count_if(({among}) and ({cond})) / nullif(count_if({among}), 0), 2)"
    df = con.sql(f"""
        with t as (select c.*, cb.cb_boro_cd, cb.cb_special_cd from {src} c
                   left join q01_cb cb on cb.community_board is not distinct from c.community_board
                   where c.created is not null)
        select {p} as period, count(*)::bigint as n,
               {pct("borough_source is distinct from 'reported'")} as pct_borough_raw,
               {pct("borough is null")} as pct_borough_after,
               {pct("cb_boro_cd is null and cb_special_cd is null")} as pct_district_raw,
               {pct("boro_cd is null and special_area_cd is null")} as pct_district_after,
               {pct("zip is null")} as pct_zip,
               {pct("latitude is null or longitude is null")} as pct_coords,
               {pct("address_type is null or trim(address_type) = ''")} as pct_address_type,
               {pct("location_type is null or trim(location_type) = ''")} as pct_location_type,
               {pct("channel is null")} as pct_channel,
               {pct("date_only_created")} as pct_created_time,
               {pct("date_only_closed", "closed is not null")} as pct_closed_time,
               {pct("closed is null", "coalesce(trim(status), '') = 'Closed'")} as pct_closed_missing
        from t group by 1 order by 1""").df()
    small = df[df.n < MIN_PERIOD_ROWS]
    return df[df.n >= MIN_PERIOD_ROWS].reset_index(drop=True), int(small.n.sum()), len(small)


def lasting_drops(t):
    """Per measure: first and last period, and the first period from which it stays below half its first value."""
    rows = []
    if t.empty:
        return pd.DataFrame(columns=["measure", "first_period", "first_pct", "last_period", "last_pct", "drop_from"])
    for c, label in PERIOD_MEASURES.items():
        s = t.set_index("period")[c].dropna()
        if s.empty:
            continue
        drop = None
        if s.iloc[0] > 0:
            below = (s < s.iloc[0] / 2).to_numpy()
            for i in range(1, len(s)):
                if below[i:].all():
                    drop = s.index[i]
                    break
        rows.append({"measure": label, "first_period": s.index[0], "first_pct": s.iloc[0],
                     "last_period": s.index[-1], "last_pct": s.iloc[-1], "max_pct": s.max(),
                     "max_period": s.idxmax(), "drop_from": drop})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- run

def run(cfg):
    con = cfg.connect()
    raw = ingest.raw_source(cfg)
    src = clean.clean_glob(cfg)
    columns = _columns(con, raw)
    n_rows = con.sql(f"select count(*) from {raw}").fetchone()[0]
    _register_bad_zips(con, raw, columns)
    _register_cb(con, src)

    enc = encodings(con, raw, columns)
    cols = column_summary(enc, columns, n_rows)
    app, n_types = applicability(con, raw, columns)
    cols = cols.merge(applies_summary(app, columns, n_types), on="column")
    small_types = int(app[app.column == columns[0]].n_type_rows.lt(SMALL_TYPE_ROWS).sum()) if columns else 0
    rec = recovery(con, src)
    unit = period_unit(con, src)
    per, n_small_rows, n_small_periods = by_period(con, src, unit)
    drops = lasting_drops(per)

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    cols.to_csv(agg / "q01_columns.csv", index=False)
    enc.to_csv(agg / "q01_encodings.csv", index=False)
    app.to_csv(agg / "q01_applicability.csv", index=False)
    rec.to_csv(agg / "q01_recovery.csv", index=False)
    per.assign(period_unit=unit).to_csv(agg / "q01_by_period.csv", index=False)

    v = dict(cols=cols, enc=enc, app=app, rec=rec, per=per, drops=drops, unit=unit, n_rows=n_rows,
             n_types=n_types, small_types=small_types, n_small_rows=n_small_rows, n_small_periods=n_small_periods)
    _report(cfg, v)
    _write_answer(cfg, v)
    return {"columns": cols, "encodings": enc, "applicability": app, "recovery": rec, "by_period": per,
            "period_unit": unit, "drops": drops}


def _report(cfg, v):
    cols, rec, per = v["cols"], v["rec"], v["per"]
    rep = Report(STAGE, cfg.out_dir)
    rep.add(f"Q1. Missing values in {v['n_rows']:,} raw rows x {len(cols)} columns ({v['n_types']} complaint types)",
            "stored = NULL or empty text | +placeholders = also placeholder text, 1900 dates, 'Unspecified <BORO>' "
            "districts, non-NYC ZIPs",
            f"where applies = rows of complaint types that fill the column >= {APPLIES_MIN_FILL:.0%}, after the "
            "structural rules below")
    t = cols.sort_values(["pct_with_placeholders", "pct_stored"], ascending=False)
    t = t[["column", "pct_stored", "pct_with_placeholders", "pct_missing_where_applies", "n_applies", "applies_to",
           "encodings"]].rename(columns={"pct_stored": "stored%", "pct_with_placeholders": "+placeh%",
                                         "pct_missing_where_applies": "applies%"})
    rep.table(t, max_rows=len(t), title="Per column, most missing first (encodings: largest first)")
    rep.add("", "## Structural rules", *[f"- {c}: {d}" for c, (_, d) in STRUCTURAL.items()],
            f"- {v['small_types']} of {v['n_types']} complaint types have < {SMALL_TYPE_ROWS} rows; their "
            "applicability call is noisy (all rows: agg/q01_applicability.csv)")
    rep.table(rec, max_rows=30, title="Location fields: present, recovered by cleaning, still missing")
    t = per.rename(columns={c: c.replace("pct_", "") for c in PERIOD_MEASURES})
    rep.table(t, max_rows=40, title=f"Missing % by {v['unit']} of opening (key fields)")
    if v["n_small_periods"]:
        rep.add(f"({v['n_small_periods']} {v['unit']}s with < {MIN_PERIOD_ROWS} rows left out: "
                f"{v['n_small_rows']:,} rows, stray backfilled records)")
    rep.emit()


def _f(x):
    return f"{x:.1f}%"


def md_table(df):
    """Simple pipe table (header row + separator row); '|' inside cells is escaped."""
    cell = lambda x: str(x).replace("|", "\\|")
    head = "| " + " | ".join(map(cell, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return [head, sep] + ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False)]


def _write_answer(cfg, v):
    cols, rec, drops, n, unit = v["cols"].set_index("column"), v["rec"], v["drops"], v["n_rows"], v["unit"]
    stored = cols[cols.pct_stored > 0]
    anymiss = cols[cols.pct_with_placeholders > 0].sort_values(["pct_with_placeholders", "pct_stored"],
                                                                ascending=False)
    placeh = cols[cols.n_placeholder > 0].sort_values("n_placeholder", ascending=False)
    gap = (cols.pct_with_placeholders - cols.pct_stored).sort_values(ascending=False)
    gap = gap[gap > 0]
    by_design = cols[cols.pct_with_placeholders >= 50].sort_values("pct_with_placeholders", ascending=False)
    gaps_where = cols[(cols.n_applies > 0) & (cols.pct_missing_where_applies > 0)].sort_values(
        "pct_missing_where_applies", ascending=False)

    per_col = pd.DataFrame({
        "column": [f"`{c}`" for c in anymiss.index],
        "missing as stored": anymiss.pct_stored.map(_f).values,
        "with placeholders": anymiss.pct_with_placeholders.map(_f).values,
        "where it applies": anymiss.pct_missing_where_applies.map(_f).values,
        "applies to": anymiss.applies_to.values})

    rec_rows = []
    for field in ["borough", "community district", "zip", "coordinates"]:
        r = rec[rec.field == field].set_index("status")
        tot = int(r.n.sum())
        raw_missing = tot - int(r.n.get(NOT_MISSING, 0))
        still = int(r[r.index.str.startswith(STILL_MISSING)].n.sum())
        how = "; ".join(f"{s} {int(x.n):,}" for s, x in r.iterrows()
                        if s != NOT_MISSING and not s.startswith(STILL_MISSING)) or "none"
        rec_rows.append({"field": field, "no usable raw value": f"{raw_missing:,} ({100 * raw_missing / tot:.1f}%)",
                         "recovered or explained by cleaning": how,
                         "missing after cleaning": f"{still:,} ({100 * still / tot:.1f}%)"})

    shown = drops[drops.max_pct > 0]
    time_tab = pd.DataFrame({
        "field": shown.measure.values,
        f"first {unit}": [f"{p}: {_f(x)}" for p, x in zip(shown.first_period, shown.first_pct)],
        f"last {unit}": [f"{p}: {_f(x)}" for p, x in zip(shown.last_period, shown.last_pct)],
        "peak": [f"{p}: {_f(x)}" for p, x in zip(shown.max_period, shown.max_pct)],
        "below half its first level from": [d if isinstance(d, str) else "no lasting drop" for d in shown.drop_from]})
    n_periods = len(v["per"])
    short = (f"The data spans only {n_periods} {unit}s, too short to show shifts across years; the same code "
             "reports by year when the data covers two years or more."
             if unit != "year" else
             "Read the last column as when a recording practice changed (computed, not assumed).")

    lines = [
        "# Q1. Missing values",
        "",
        f"*Drafted by `nyc311/q01_missing.py` from {n:,} raw rows and {len(cols)} columns. Numbers are computed; "
        "write the prose from these bullets. Full tables: `outputs/agg/q01_*.csv`.*",
        "",
        "## Three ways to count missing",
        "",
        f"- **The answer depends on the definition.** {len(stored)} of {len(cols)} columns have values stored as NULL "
        f"or empty text; counting placeholder values, {len(anymiss)} columns have missing values.",
        "  - *As stored*: NULL or empty text.",
        "  - *With placeholders*: also placeholder text (`Unspecified`, `N/A`, `UNKNOWN`, `0000`, ...), 1900 dates, "
        "`Unspecified <BOROUGH>` community boards (borough known, district not) and ZIPs that are not NYC 5-digit codes.",
        f"  - *Where it applies*: only rows of complaint types that fill the column for at least "
        f"{APPLIES_MIN_FILL:.0%} of their rows, after structural rules: " + "; ".join(
            f"`{c}` {d}" for c, (_, d) in STRUCTURAL.items()) + ".",
        "",
        *md_table(per_col),
        "",
        "## Missing written as text",
        "",
        "- **Placeholders by column** (value and rows, largest first):",
        *[f"  - `{c}`: {r.encodings}" for c, r in placeh.head(10).iterrows()],
        "- **Where stored NULLs understate missingness most** (as stored -> with placeholders): " + "; ".join(
            f"`{c}` {_f(cols.loc[c, 'pct_stored'])} -> {_f(cols.loc[c, 'pct_with_placeholders'])}"
            for c in gap.head(6).index) + ".",
        "",
        "## Blank for a reason vs. truly missing",
        "",
        f"- **Blank by design.** {len(by_design)} columns are missing for at least half the rows, mostly because "
        "they don't apply (overall -> where it applies): " + "; ".join(
            f"`{c}` {_f(r.pct_with_placeholders)} -> {_f(r.pct_missing_where_applies)}"
            for c, r in by_design.iterrows()) + ".",
        "- **Still missing where the column applies** (the real gaps): " + "; ".join(
            f"`{c}` {_f(r.pct_missing_where_applies)} of {int(r.n_applies):,} rows"
            for c, r in gaps_where.head(8).iterrows()) + ".",
        "",
        "## Location fields after cleaning",
        "",
        *md_table(pd.DataFrame(rec_rows)),
        "",
        f"## Missingness over time (by {unit} of opening)",
        "",
        *md_table(time_tab),
        "",
        f"- {short}",
        "",
        "## Caveats",
        "",
        f"- The {APPLIES_MIN_FILL:.0%} rule is a heuristic: {v['small_types']} of {v['n_types']} complaint types have "
        f"fewer than {SMALL_TYPE_ROWS} rows, so their call can flip on a few rows.",
        "- Time of day missing (opening or closing at exactly 00:00:00) is listed as an encoding, not counted as a "
        "missing value; it limits durations to whole days (Q4).",
        *([f"- {v['n_small_periods']} {unit}s with fewer than {MIN_PERIOD_ROWS} rows ({v['n_small_rows']:,} stray "
           "backfilled rows) are left out of the time table."] if v["n_small_periods"] else []),
        "",
    ]
    path = cfg.out_dir / "answers" / "q01.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
