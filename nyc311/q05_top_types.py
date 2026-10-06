"""Q5: the top 10 complaint types by volume as a board-ready chart.

Counts canonical complaint types (Q3's map, so spelling variants are merged) in the default analysis set (not
is_test, not dup_key), leaving out information requests (is_info_request). When the data hold at least two full
calendar years, a second chart shows how the top 10 changed between the first and the last full year.
Every number in the report, the charts and outputs/answers/q05.md is computed here.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from nyc311.clean import clean_glob
from nyc311.report import Report

STAGE = "q05_top_types"
TOP_N = 10
FULL_YEAR_EDGE_DAYS = 7     # a year is "full" if its date-ordered rows reach into its first and its last week
AGENCY_LABEL_SHARE = 0.10   # agencies with at least this share of a type's rows are named in its label
COMPLAINTS = "not is_test and not dup_key and not coalesce(is_info_request, false)"
EN = "–"


# ---------- period ----------

def data_period(con, src):
    """First and last opening date among rows in file order (out_of_file_order rows are backfilled records)."""
    return con.sql(f"select min(created)::date, max(created)::date from {src} where not out_of_file_order").fetchone()


def period_label(d0, d1):
    if d0.year != d1.year:
        return f"{d0.year}{EN}{d1.year}"
    if d0.month == d1.month:
        return f"{d0:%b} {d0.day}{EN}{d1.day}, {d1.year}"
    return f"{d0:%b} {d0.day}{EN}{d1:%b} {d1.day}, {d1.year}"


def period_long(d0, d1):
    return f"{d0:%b} {d0.day}, {d0.year} to {d1:%b} {d1.day}, {d1.year}"


def full_years(con, rel):
    """Calendar years whose date-ordered rows start in the first week and end in the last week of the year."""
    y = con.sql(f"""select year(created) y, min(created)::date d0, max(created)::date d1 from {rel}
                    where not out_of_file_order and created is not null group by 1 order by 1""").df()
    edge = dt.timedelta(days=FULL_YEAR_EDGE_DAYS - 1)
    return [int(r.y) for r in y.itertuples()
            if pd.Timestamp(r.d0).date() <= dt.date(int(r.y), 1, 1) + edge
            and pd.Timestamp(r.d1).date() >= dt.date(int(r.y), 12, 31) - edge]


# ---------- counts ----------

def exclusions(con, src):
    return con.sql(f"""select count(*) n_rows, count_if(is_test) n_test, count_if(dup_key and not is_test) n_dup,
                       count_if(not is_test and not dup_key and coalesce(is_info_request, false)) n_info,
                       count_if({COMPLAINTS} and canonical_type is null) n_no_type from {src}""").df().iloc[0].astype(int)


def type_counts(con, src):
    """One row per canonical type: rows, category, main agency, and a label naming the agencies that matter."""
    by = con.sql(f"""select canonical_type, coalesce(agency, '(none)') agency, any_value(category) category,
                     count(*) n from {src} where {COMPLAINTS} and canonical_type is not null
                     group by canonical_type, agency""").df()
    rows = []
    for t, g in by.groupby("canonical_type", sort=False):
        g = g.sort_values(["n", "agency"], ascending=[False, True])
        n = int(g.n.sum())
        named = g[g.n >= AGENCY_LABEL_SHARE * n].agency.tolist()
        rows.append({"canonical_type": t, "category": g.category.iloc[0], "main_agency": g.agency.iloc[0],
                     "n_agencies": len(g), "n": n, "label": f"{t} ({'/'.join(named)})"})
    out = pd.DataFrame(rows, columns=["canonical_type", "category", "main_agency", "n_agencies", "n", "label"])
    return out.sort_values(["n", "canonical_type"], ascending=[False, True]).reset_index(drop=True)


def top_table(counts, total):
    top = counts.head(TOP_N).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["pct"] = 100 * top.n / total
    top["cum_pct"] = top.pct.cumsum()
    return top


def change_table(con, src, top, y0, y1):
    """Share of each year's complaints for the overall top types, first vs last full year."""
    d = con.sql(f"""select year(created) y, canonical_type, count(*) n from {src}
                    where {COMPLAINTS} and canonical_type is not null and year(created) in ({y0}, {y1})
                    group by all""").df()
    out = top[["rank", "canonical_type", "label"]].copy()
    totals = {}
    for y in (y0, y1):
        dy = d[d.y == y].sort_values(["n", "canonical_type"], ascending=[False, True]).reset_index(drop=True)
        dy["rank_y"] = np.arange(1, len(dy) + 1)
        totals[y] = int(dy.n.sum())
        m = dy.set_index("canonical_type")
        out[f"n_{y}"] = out.canonical_type.map(m.n).fillna(0).astype(int)
        out[f"pct_{y}"] = 100 * out[f"n_{y}"] / max(totals[y], 1)
        out[f"rank_{y}"] = out.canonical_type.map(m.rank_y)
    out["change_pts"] = out[f"pct_{y1}"] - out[f"pct_{y0}"]
    return out, totals


# ---------- charts ----------

def _scale(vmax):
    if vmax >= 1e6:
        return 1e6, "M"
    if vmax >= 1e4:
        return 1e3, "K"
    return 1.0, ""


def _count(n):
    return f"{n / 1e6:.2f} million" if n >= 1e6 else f"{n:,}"


def _hbar_axes(ax):
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", labelcolor="#333333", labelsize=10.5)


def chart_top(top, path, title, note):
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    fig = Figure(figsize=(8, 0.42 * len(top) + 1.7))
    ax = fig.subplots()
    scale, suffix = _scale(top.n.max())
    y = np.arange(len(top))
    vals = top.n.to_numpy() / scale
    ax.barh(y, vals, color=[nyt.SAGE.line if i == 0 else nyt.SAGE.band for i in y], height=0.72, zorder=2)
    for i, (v, n, p) in enumerate(zip(vals, top.n, top.pct)):
        ax.text(v + 0.012 * vals.max(), i, f"{_count(int(n))}  ({p:.1f}%)", va="center", ha="left",
                color=nyt.SAGE.line, fontfamily=nyt.SANS, fontweight="bold" if i == 0 else "normal", fontsize=9.5)
    ax.set_yticks(y, top.label)
    ax.invert_yaxis()
    ax.set_xlim(0, vals.max() * 1.32)
    _hbar_axes(ax)
    nyt.format_x(ax, suffix=suffix)
    nyt.title(fig, title)
    nyt.note(fig, note)
    return nyt.save(fig, path)


def chart_change(ch, y0, y1, path, title, note):
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    fig = Figure(figsize=(8, 0.42 * len(ch) + 1.9))
    ax = fig.subplots()
    y = np.arange(len(ch))
    a, b = ch[f"pct_{y0}"].to_numpy(), ch[f"pct_{y1}"].to_numpy()
    for i in y:
        ax.plot([a[i], b[i]], [i, i], color="#cccccc", linewidth=2.2, zorder=1, solid_capstyle="butt")
    ax.scatter(a, y, color=nyt.SAGE.line, s=46, zorder=3, linewidths=0)
    ax.scatter(b, y, color=nyt.CORAL.line, s=46, zorder=3, linewidths=0)
    gap = abs(a[0] - b[0]) < 0.06 * max(a.max(), b.max())
    for x, label, color, dx in [(a[0], str(y0), nyt.SAGE.line, -8 if gap else 0),
                                (b[0], str(y1), nyt.CORAL.line, 8 if gap else 0)]:
        ax.annotate(label, (x, 0), xytext=(dx, 9), textcoords="offset points", ha="center", va="bottom",
                    color=color, fontfamily=nyt.SANS, fontweight="bold", fontsize=10.5)
    ax.set_yticks(y, ch.label)
    ax.set_ylim(len(ch) - 0.4, -1.0)
    ax.set_xlim(0, max(a.max(), b.max()) * 1.12)
    _hbar_axes(ax)
    nyt.format_x(ax, suffix="%")
    nyt.title(fig, title)
    nyt.note(fig, note)
    return nyt.save(fig, path)


# ---------- run ----------

def run(cfg):
    con = cfg.connect()
    src = clean_glob(cfg)
    excl = exclusions(con, src)
    d0, d1 = data_period(con, src)
    period, period_l = period_label(d0, d1), period_long(d0, d1)
    n_outside = con.sql(f"""select count(*) from {src} where {COMPLAINTS} and canonical_type is not null
                            and (created::date < '{d0}' or created::date > '{d1}')""").fetchone()[0]
    counts = type_counts(con, src)
    total = int(counts.n.sum())
    top = top_table(counts, total)
    years = full_years(con, src)
    change, totals = (None, {})
    if len(years) >= 2:
        change, totals = change_table(con, src, top, years[0], years[-1])

    res = {"top": top, "counts": counts, "total": total, "excluded": excl, "period": period, "period_long": period_l,
           "n_outside_period": int(n_outside), "n_days": (d1 - d0).days + 1, "full_years": years, "change": change, "totals": totals,
           "years": (years[0], years[-1]) if change is not None else None}

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    top.to_csv(agg / "q05_top10.csv", index=False)
    counts.to_csv(agg / "q05_type_counts.csv", index=False)
    figs = cfg.out_dir / "figs"
    chart_top(top, figs / "q05_top10.png", f"The {len(top)} most common 311 complaints, {period}", _note(res))
    if change is not None:
        y0, y1 = res["years"]
        change.to_csv(agg / "q05_top10_change.csv", index=False)
        chart_change(change, y0, y1, figs / "q05_top10_change.png",
                     f"How the top {len(top)} complaint types changed, {y0} vs. {y1}",
                     f"Note: Share of each year's complaints ({totals[y0]:,} in {y0}, {totals[y1]:,} in {y1}); "
                     f"first and last full calendar years in the data.\nTypes ranked by volume over {period}; "
                     "information requests and test records excluded.")
    else:
        (figs / "q05_top10_change.png").unlink(missing_ok=True)
        (agg / "q05_top10_change.csv").unlink(missing_ok=True)
    _report(cfg, res)
    _write_answer(cfg, res)
    return res


def _note(res):
    e, top = res["excluded"], res["top"]
    outside = (f" Includes {res['n_outside_period']:,} backfilled requests dated outside this period."
               if res["n_outside_period"] else "")
    return (f"Note: Complaints and share of all {res['total']:,} complaints, {res['period_long']}; the top "
            f"{len(top)} are {top.cum_pct.iloc[-1]:.0f}% of the total. Spelling variants merged.\n"
            f"Excludes {e.n_info:,} information requests and {e.n_test:,} test records; agencies in parentheses."
            + outside)


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    e = res["excluded"]
    rep.add(f"Q5. Top {TOP_N} complaint types by volume, {res['period_long']}",
            f"rows {e.n_rows:,}; left out: is_test {e.n_test:,}, dup_key {e.n_dup:,}, is_info_request {e.n_info:,}, "
            f"no complaint type {e.n_no_type:,}; complaints counted {res['total']:,} in "
            f"{len(res['counts'])} canonical types",
            f"backfilled rows dated outside the period (counted): {res['n_outside_period']:,}; full calendar years: "
            + (", ".join(map(str, res["full_years"])) or "none"))
    rep.table(res["top"][["rank", "canonical_type", "category", "main_agency", "n", "pct", "cum_pct"]],
              title=f"Top {TOP_N} (pct of complaints)")
    if res["change"] is not None:
        rep.table(res["change"].drop(columns="label"), title="First vs last full year (pct of that year's complaints)")
    else:
        rep.add("", "fewer than two full calendar years: no change chart")
    rep.emit()


def _write_answer(cfg, res):
    e, top, total = res["excluded"], res["top"], res["total"]
    t1 = top.iloc[0]
    cats = top.groupby("category").agg(types=("n", "size"), n=("n", "sum")).sort_values("n", ascending=False)
    cat_line = "; ".join(f"{c} {int(r.types)} of {len(top)} ({100 * r.n / total:.1f}% of complaints)"
                         for c, r in cats.iterrows())
    table = ["| Rank | Complaint type | Main agency | Complaints | Share |", "|---|---|---|---|---|"]
    table += [f"| {r.rank} | {r.canonical_type.replace('|', '/')} | {r.main_agency} | {r.n:,} | {r.pct:.1f}% |"
              for r in top.itertuples()]
    lines = [
        f"# Q5. Top {len(top)} complaint types by volume",
        "",
        f"*Drafted by `nyc311/q05_top_types.py` from {total:,} complaints, {res['period_long']}. Every number is "
        "computed; write the prose from these bullets.*",
        "",
        f"- **What is counted.** Complaints in the default analysis set, grouped by canonical complaint type (Q3's "
        f"map merges spelling variants, so {len(res['counts'])} types remain). Left out: {e.n_info:,} information "
        f"requests (`is_info_request`), {e.n_test:,} test records (`is_test`), {e.n_dup:,} duplicate keys "
        f"(`dup_key`) and {e.n_no_type:,} rows with no complaint type."
        + (f" {res['n_outside_period']:,} backfilled requests dated outside {res['period']} are counted."
           if res["n_outside_period"] else ""),
        f"- **The leader.** {t1.canonical_type} ({t1.main_agency}) has {int(t1.n):,} complaints, {t1.pct:.1f}% of "
        f"the total and {t1.n / top.n.iloc[1]:.1f} times the next type ({top.canonical_type.iloc[1]}).",
        f"- **Concentration.** The top 3 are {top.cum_pct.iloc[min(2, len(top) - 1)]:.1f}% of complaints and the top "
        f"{len(top)} are {top.cum_pct.iloc[-1]:.1f}%; the other {len(res['counts']) - len(top)} types share the "
        f"remaining {100 - top.cum_pct.iloc[-1]:.1f}%.",
        f"- **By category,** the top {len(top)} are: {cat_line}.",
        *([f"- **Short period.** The data cover only {res['n_days']} days ({res['period_long']}), so seasonal "
           "types are over- or under-represented (heat complaints peak in winter, noise in summer); a ranking over "
           "full years will differ."] if res["n_days"] < 365 else []),
        "",
        *table,
        "",
        f"![The {len(top)} most common 311 complaints, {res['period']}](figs/q05_top10.png)",
        "",
    ]
    ch = res["change"]
    if ch is not None:
        y0, y1 = res["years"]
        up = ch.sort_values("change_pts", ascending=False).iloc[0]
        down = ch.sort_values("change_pts").iloc[0]
        lines += [
            f"- **Change between the first and last full calendar years ({y0} and {y1}).** Biggest gain: "
            f"{up.canonical_type}, {up[f'pct_{y0}']:.1f}% to {up[f'pct_{y1}']:.1f}% of that year's complaints "
            f"(rank {_rank(up[f'rank_{y0}'])} to {_rank(up[f'rank_{y1}'])}). Biggest drop: {down.canonical_type}, "
            f"{down[f'pct_{y0}']:.1f}% to {down[f'pct_{y1}']:.1f}% (rank {_rank(down[f'rank_{y0}'])} to "
            f"{_rank(down[f'rank_{y1}'])}). Complaints: {res['totals'][y0]:,} in {y0}, {res['totals'][y1]:,} in {y1}.",
            "",
            f"![How the top {len(top)} complaint types changed, {y0} vs. {y1}](figs/q05_top10_change.png)",
            "",
        ]
    else:
        n_full = len(res["full_years"])
        lines += [f"- **Change over time.** The data contain {'only one' if n_full == 1 else 'no'} full calendar "
                  f"year ({res['period_long']}), so the chart comparing the first and last full years is left out; "
                  "it is drawn automatically when the data cover two or more full years.", ""]
    lines += [
        "- **Board-ready choices.** Horizontal bars sorted by volume so the long type names read without "
        "rotation; each bar is labeled with its count and share, so no axis reading is needed; the agency that "
        "handles each type is in parentheses; the title states the period and the note states what was excluded.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q05.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _rank(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v))
