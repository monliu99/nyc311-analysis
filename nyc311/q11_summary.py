"""Q11: headline numbers for the executive summary, computed straight from the clean table.

Choosing the three findings, two actions and one recommendation is a judgment made after reading Q1-Q10;
that text lives in report/prose/q11.md. This module supplies the numbers it should cite, each labeled with the set
of rows it covers. The sets are the shared ones: rules.BASE_SQL (requests), rules.COMPLAINTS_SQL (complaints),
rules.INSTANT_SQL (instant closures, as in Q4), clean.full_years (full calendar years), and Q1's
q01_missing.NO_DISTRICT_SQL for a request with no community district (Q1 counts it over all rows; here over requests).
"""
import pandas as pd

from nyc311 import clean, rules
from nyc311.clean import clean_glob
from nyc311.q01_missing import NO_DISTRICT_SQL  # Q1's rule: parks and airports count as located
from nyc311.report import Report

KINDS = ("count", "pct", "hours", "year", "date", "text")
# The set each number covers (the "Counted over" column).
S_REQ = "requests: all rows except test records (is_test) and duplicate keys (dup_key)"
S_CMP = "complaints: requests except information requests and invalid entries"
S_CMP_COMPLETE = "complaints opened on days not flagged possibly_incomplete_day"
S_CLOSED = "complaints closed with a duration (status Closed, closed on or after opening)"
S_CLOSED_X = "complaints closed with a duration, minus instant closures (rules.INSTANT_SQL, as in Q4)"
S_BORO = "requests with a borough after cleaning"
S_YEARS = "calendar years with every day present and no day flagged possibly_incomplete_day"
S_ORDER = "requests in file order (not out_of_file_order)"
S_DAYS = "dates with requests"
S_FILES = "batch numbers from the first to the last file"


def fmt_value(v, kind):
    """Display text: counts with thousands separators, percentages and hours with one decimal, years plain."""
    if kind == "count":
        return f"{int(v):,}"
    if kind in ("pct", "hours"):
        return f"{float(v):.1f}"
    if kind == "year":
        return str(int(v)) if not isinstance(v, str) else v
    return str(v)


def _row(metric, value, kind, label, covers):
    assert kind in KINDS, kind
    return {"metric": metric, "value": value, "kind": kind, "label": label, "set": covers}


def growth_rows(counts, full, profile_last=rules.PROFILE_LAST_YEAR):
    """Rows for the first and last full year, the complaints in each, and the change between them; plus the change
    to profile_last (the last year before the pandemic) when it falls strictly between them."""
    if len(full) < 2:
        return []
    y0, y1 = int(full[0]), int(full[-1])

    def change(a, b):
        return round(100 * (counts[b] / counts[a] - 1), 1)

    rows = [_row("first_full_year", str(y0), "year", "First full calendar year", S_YEARS),
            _row("last_full_year", str(y1), "year", "Last full calendar year", S_YEARS),
            _row(f"complaints_{y0}", int(counts[y0]), "count", f"Complaints opened in {y0}", S_CMP)]
    if y0 < profile_last < y1 and profile_last in full:
        rows += [_row(f"complaints_{profile_last}", int(counts[profile_last]), "count",
                      f"Complaints opened in {profile_last}", S_CMP),
                 _row("growth_to_profile_year_pct", change(y0, profile_last), "pct",
                      f"Change in complaints, {y0} to {profile_last} (last year before the pandemic) (%)", S_CMP)]
    pandemic = f"; includes the pandemic years after {profile_last}" if y1 > profile_last else ""
    rows += [_row(f"complaints_{y1}", int(counts[y1]), "count", f"Complaints opened in {y1}", S_CMP),
             _row("growth_pct", change(y0, y1), "pct",
                  f"Change in complaints, {y0} to {y1} (last full year{pandemic}) (%)", S_CMP)]
    return rows


def key_numbers(cfg):
    from nyc311.q05_top_types import data_period

    con = cfg.connect()
    src = clean_glob(cfg)
    base, cmp_, inst = rules.BASE_SQL, rules.COMPLAINTS_SQL, rules.INSTANT_SQL
    rows = []

    def add(*args):
        rows.append(_row(*args))

    n = con.sql(f"select count(*) from {src} where {base}").fetchone()[0]
    add("requests", int(n), "count", "Service requests analyzed", S_REQ)
    lo, hi = data_period(con, src)
    add("first_date", str(lo), "date", "First request opened", S_ORDER)
    add("last_date", str(hi), "date", "Last request opened", S_ORDER)
    complaints = con.sql(f"select count(*) from {src} where {cmp_}").fetchone()[0]
    add("complaints", int(complaints), "count", "Complaints", S_CMP)

    full = clean.full_years(con, src)
    counts = dict(con.sql(f"select year(created), count(*) from {src} where {cmp_} and created is not null "
                          "group by 1").fetchall())
    grow = growth_rows(counts, full)
    if grow:
        rows.extend(grow)
    else:
        add("full_years", len(full), "count", "Full calendar years in the data (fewer than two: no year-over-year "
            "comparison)", S_YEARS)

    month = con.sql(f"""select strftime(created, '%Y-%m') as m, count(*) as n from {src}
                        where {cmp_} and not coalesce(possibly_incomplete_day, false) and created is not null
                        group by 1 order by n desc, m limit 1""").fetchone()
    if month:
        add("peak_month", month[0], "text", "Busiest month", S_CMP_COMPLETE)
        add("peak_month_complaints", int(month[1]), "count", "Complaints in the busiest month", S_CMP_COMPLETE)

    cat = con.sql(f"""select category, count(*) as n from {src} where {cmp_}
                      group by 1 order by n desc, category limit 1""").fetchone()
    add("top_category", cat[0], "text", "Largest complaint category", S_CMP)
    add("top_category_pct", round(100 * cat[1] / complaints, 1), "pct", "Largest category's share of complaints (%)",
        S_CMP)
    typ = con.sql(f"""select canonical_type, count(*) as n from {src} where {cmp_}
                      group by 1 order by n desc, canonical_type limit 1""").fetchone()
    add("top_type", typ[0], "text", "Most common complaint type", S_CMP)
    add("top_type_pct", round(100 * typ[1] / complaints, 1), "pct", "Most common type's share of complaints (%)",
        S_CMP)

    n_closed, med, n_x, med_x = con.sql(f"""
        select count(duration_h), quantile_cont(duration_h, 0.5),
               count(duration_h) filter (where not ({inst})), quantile_cont(duration_h, 0.5) filter (where not ({inst}))
        from {src} where {cmp_} and duration_h is not null""").fetchone()
    add("closed_with_duration", int(n_closed), "count", "Closed complaints with a duration", S_CLOSED)
    add("median_hours_to_close", round(med, 1), "hours", "Median hours to close", S_CLOSED)
    add("closed_excl_instant", int(n_x), "count", "Closed complaints with a duration, excluding instant closures",
        S_CLOSED_X)
    add("median_hours_to_close_excl_instant", round(med_x, 1), "hours",
        "Median hours to close, excluding instant closures (as in Q4)", S_CLOSED_X)

    unres, pct_unres = con.sql(f"select count_if(unresolved), 100 * avg(unresolved::int) from {src} "
                               f"where {cmp_}").fetchone()
    add("unresolved", int(unres), "count", "Complaints still unresolved", S_CMP)
    add("pct_unresolved", round(pct_unres, 1), "pct", "Share of complaints still unresolved (%)", S_CMP)

    rec, no_cd, pct_no_cd, inc_days = con.sql(f"""
        select 100 * avg((borough_source <> 'reported')::int) filter (where borough is not null),
               count_if({NO_DISTRICT_SQL}),
               100 * avg(({NO_DISTRICT_SQL})::int),
               count(distinct created::date) filter (where possibly_incomplete_day)
        from {src} where {base}""").fetchone()
    add("pct_borough_recovered", round(rec, 1), "pct", "Requests whose borough was recovered by cleaning (%)", S_BORO)
    add("no_district", int(no_cd), "count", "Requests with no community district after cleaning (Q1's rule: parks "
        "and airports count as located; Q1 counts all rows)", S_REQ)
    add("pct_no_district", round(pct_no_cd, 1), "pct", "Share of requests with no community district after cleaning "
        "(%)", S_REQ)
    add("incomplete_days", int(inc_days), "count", "Days flagged as possibly incomplete", S_DAYS)
    manifest = cfg.work_dir / "manifest.parquet"
    if manifest.exists():
        nums = pd.read_parquet(manifest, columns=["file"]).file.str.extract(r"_(\d+)\.json$", expand=False)
        nums = nums.dropna().astype(int)
        missing = len(set(range(int(nums.min()), int(nums.max()) + 1)) - set(nums)) if len(nums) else 0
        add("missing_batches", int(missing), "count", "Missing batch files in the numbered sequence", S_FILES)
    df = pd.DataFrame(rows)
    df["display"] = [fmt_value(v, k) for v, k in zip(df.value, df.kind)]
    return df[["metric", "value", "display", "kind", "label", "set"]], full


def run(cfg):
    from nyc311.q05_top_types import period_long

    nums, full = key_numbers(cfg)
    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    nums.to_csv(agg / "q11_key_numbers.csv", index=False)

    v = nums.set_index("metric")
    table = "\n".join(f"| {r.label} | {r.display} | {r.set} |" for r in nums.itertuples())
    lo, hi = (pd.Timestamp(v.value[m]).date() for m in ("first_date", "last_date"))
    if len(full) >= 2:
        y0, y1 = v.display["first_full_year"], v.display["last_full_year"]
        period = (f"- Full calendar years in the data: {y0} to {y1}; year-over-year figures use only those."
                  + (f" {rules.PROFILE_LAST_YEAR} is the last year before the pandemic, so growth is shown both to it "
                     f"and to {y1}." if "growth_to_profile_year_pct" in v.index else ""))
    else:
        period = ("- These data cover fewer than two full calendar years, so no full calendar year comparison is "
                  "made; seasonal and trend claims need a longer period.")
    text = f"""# Q11 Executive summary: key numbers

*Computed by `nyc311/q11_summary.py` from {v.display['requests']} requests ({v.display['complaints']} complaints), \
{period_long(lo, hi)}.*

| Metric | Value | Counted over |
|---|---|---|
{table}

{period}
- Instant closures (closed at the second of opening, with a time of day) are defined as in Q4
  (`rules.INSTANT_SQL`), so the median excluding them matches Q4's.
"""
    answers = cfg.out_dir / "answers"
    answers.mkdir(parents=True, exist_ok=True)
    (answers / "q11.md").write_text(text)

    rep = Report("q11_summary", cfg.out_dir)
    rep.table(nums[["metric", "display", "set"]].rename(columns={"display": "value"}), max_rows=40,
              title="Key numbers for the executive summary")
    rep.emit()
