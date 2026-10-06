"""Q11: headline numbers for the executive summary, computed straight from the clean table.

Choosing the three findings, two actions and one recommendation is a judgment made after reading Q1-Q10;
that text lives in report/prose/q11.md. This module supplies the numbers it should cite.
"""
import pandas as pd

from nyc311.clean import clean_glob
from nyc311.report import Report

BASE = "not is_test and not dup_key"
COMPLAINT = f"{BASE} and not coalesce(is_info_request, false) and coalesce(category, '') <> 'Invalid entries'"
FULL_YEAR_MIN_DAYS = 360


def key_numbers(cfg):
    con = cfg.connect()
    src = clean_glob(cfg)
    rows = []

    def add(metric, value, label, note=""):
        rows.append({"metric": metric, "value": value, "label": label, "note": note})

    n, lo, hi = con.sql(f"select count(*), min(created)::date, max(created)::date from {src} where {BASE}").fetchone()
    add("requests", n, "Service requests analyzed", "excludes test records and duplicate keys")
    add("first_date", str(lo), "First request opened")
    add("last_date", str(hi), "Last request opened")
    complaints = con.sql(f"select count(*) from {src} where {COMPLAINT}").fetchone()[0]
    add("complaints", complaints, "Complaints (excluding information requests and invalid entries)")

    years = con.sql(f"""select year(created) as y, count(distinct created::date) as days, count(*) as n,
                               count_if(possibly_incomplete_day) as n_incomplete
                        from {src} where {COMPLAINT} group by 1 order by 1""").df()
    full = years[(years.days >= FULL_YEAR_MIN_DAYS) & (years.n_incomplete == 0)]
    if len(full) >= 2:
        first, last = full.iloc[0], full.iloc[-1]
        add("first_full_year", int(first.y), "First full calendar year")
        add("last_full_year", int(last.y), "Last full calendar year")
        add("growth_pct", round(100 * (last.n / first.n - 1), 1),
            f"Change in complaints, {int(first.y)} to {int(last.y)} (%)")
    else:
        add("full_years", len(full), "Full calendar years in the data",
            "fewer than two, so no year-over-year comparison is made")

    month = con.sql(f"""select strftime(created, '%Y-%m') as m, count(*) as n from {src}
                        where {COMPLAINT} and not possibly_incomplete_day group by 1 order by n desc limit 1""").fetchone()
    if month:
        add("peak_month", month[0], "Busiest month")
        add("peak_month_complaints", month[1], "Complaints in the busiest month")

    cat = con.sql(f"""select category, count(*) as n from {src} where {COMPLAINT}
                      group by 1 order by n desc limit 1""").fetchone()
    add("top_category", cat[0], "Largest complaint category")
    add("top_category_pct", round(100 * cat[1] / complaints, 1), "Largest category's share of complaints (%)")
    typ = con.sql(f"""select canonical_type, count(*) as n from {src} where {COMPLAINT}
                      group by 1 order by n desc limit 1""").fetchone()
    add("top_type", typ[0], "Most common complaint type")
    add("top_type_pct", round(100 * typ[1] / complaints, 1), "Most common type's share of complaints (%)")

    med, med_real = con.sql(f"""select median(duration_h) / 24,
                                       median(duration_h) filter (where not closed_at_creation) / 24
                                from {src} where {COMPLAINT}""").fetchone()
    add("median_days_to_close", round(med, 2), "Median days to close")
    add("median_days_to_close_excl_instant", round(med_real, 2),
        "Median days to close, excluding closures at the second of opening")

    unres, pct_unres = con.sql(f"""select count_if(unresolved), 100 * avg(unresolved::int)
                                   from {src} where {COMPLAINT}""").fetchone()
    add("unresolved", unres, "Complaints still unresolved")
    add("pct_unresolved", round(pct_unres, 2), "Share still unresolved (%)")

    rec, no_cd, inc_days = con.sql(f"""select 100 * avg((borough_source <> 'reported')::int) filter (where borough is not null),
                                              100 * avg((boro_cd is null)::int),
                                              count(distinct created::date) filter (where possibly_incomplete_day)
                                       from {src} where {BASE}""").fetchone()
    add("pct_borough_recovered", round(rec, 1), "Requests whose borough had to be recovered (%)")
    add("pct_no_district", round(no_cd, 1), "Requests with no community district (%)")
    add("incomplete_days", inc_days, "Days flagged as possibly incomplete")
    manifest = cfg.work_dir / "manifest.parquet"
    if manifest.exists():
        nums = pd.read_parquet(manifest, columns=["file"]).file.str.extract(r"_(\d+)\.json$", expand=False)
        nums = nums.dropna().astype(int)
        missing = len(set(range(int(nums.min()), int(nums.max()) + 1)) - set(nums)) if len(nums) else 0
        add("missing_batches", missing, "Missing batch files in the numbered sequence")
    return pd.DataFrame(rows), full


def _fmt(v):
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, int) or (hasattr(v, "dtype") and str(getattr(v, "dtype", "")).startswith("int")):
        return f"{int(v):,}"
    return str(v)


def run(cfg):
    nums, full = key_numbers(cfg)
    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    nums.to_csv(agg / "q11_key_numbers.csv", index=False)

    table = "\n".join(f"| {r.label} | {_fmt(r.value)} |" for r in nums.itertuples())
    v = nums.set_index("metric").value
    if len(full) >= 2:
        period = (f"- Full calendar years in the data: {int(v['first_full_year'])} to {int(v['last_full_year'])}; "
                  f"year-over-year figures use only those.")
    else:
        period = ("- This data covers less than two full calendar years, so no full calendar year comparison is "
                  "made; seasonal and trend claims need the full cluster data.")
    text = f"""# Q11 Executive summary: key numbers

The three findings, two immediate actions and one long-term recommendation are written in
`report/prose/q11.md`, using these numbers and the findings in Q1-Q10.

| Metric | Value |
|---|---|
{table}

{period}
- All figures exclude test records and duplicate keys; complaint figures also exclude information requests and
  invalid entries.
"""
    answers = cfg.out_dir / "answers"
    answers.mkdir(parents=True, exist_ok=True)
    (answers / "q11.md").write_text(text)

    rep = Report("q11_summary", cfg.out_dir)
    rep.table(nums[["metric", "value", "note"]].assign(value=nums.value.map(_fmt)), max_rows=40,
              title="Key numbers for the executive summary")
    rep.emit()
