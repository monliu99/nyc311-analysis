"""Q5: the top 10 complaint types by volume as a board-ready chart.

Counts canonical complaint types (Q3's map, so spelling variants are merged) among complaints (rules.COMPLAINTS_SQL:
not is_test, not dup_key, not is_info_request, not taxonomy category "Invalid entries"), so totals match Q9-Q11.
When the data hold at least two full calendar years (clean.full_years), a second chart compares the first full year
with the last full year up to rules.PROFILE_LAST_YEAR (2011 vs 2019 on the full data) and shows the last full year
after it (2020) as a lighter third marker, so the pandemic shift is visible without driving the comparison.
Every number in the report, the charts and outputs/answers/q05.md is computed here.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from nyc311 import clean, rules
from nyc311.clean import clean_glob
from nyc311.report import Report

STAGE = "q05_top_types"
TOP_N = 10
AGENCY_LABEL_SHARE = 0.10   # agencies with at least this share of a type's rows are named in its label
COMPLAINTS = rules.COMPLAINTS_SQL
TIMES_FROM = 1.5            # "N times the next type" from this ratio up; below it, "N% more"
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


def profile_years(full, present, last=rules.PROFILE_LAST_YEAR):
    """(years averaged into a typical profile, later years shown separately).

    Typical seasonal and weekly profiles use the full calendar years up to `last` (rules.PROFILE_LAST_YEAR); every
    year after it that has data is shown on its own. Without a full year up to `last`, the full years are used as
    they are; without any full year, every year up to `last` with data (or every year, if none)."""
    present = sorted({int(y) for y in present})
    early = [int(y) for y in full if y <= last]
    if early:
        return early, [y for y in present if y > last]
    if full:
        return [int(y) for y in full], []
    pre = [y for y in present if y <= last]
    return (pre, [y for y in present if y > last]) if pre else (present, [])


def years_label(years):
    """'2011–2019' for a run of years, '2010, 2012' otherwise."""
    years = sorted(years)
    if not years:
        return "none"
    if len(years) > 1 and years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}{EN}{years[-1]}"
    return ", ".join(map(str, years))


def comparison_years(full, last=rules.PROFILE_LAST_YEAR):
    """(first year, comparison year, later full year or None) for the change chart, from the full years: the first
    full year vs the last full year up to `last`, plus the last full year after `last`. With fewer than two full
    years up to `last`, the first and last full years; None with fewer than two full years."""
    full = sorted(full)
    early = [y for y in full if y <= last]
    if len(early) >= 2:
        return early[0], early[-1], (full[-1] if full[-1] > early[-1] else None)
    return (full[0], full[-1], None) if len(full) >= 2 else None


def ratio_phrase(r):
    """How much bigger the leader is than the next type: '4% more than' below TIMES_FROM, else '2.3 times'."""
    return f"{100 * (r - 1):.0f}% more than" if r < TIMES_FROM else f"{r:.1f} times"


# ---------- counts ----------

def exclusions(con, src):
    base, info = rules.BASE_SQL, "coalesce(is_info_request, false)"
    return con.sql(f"""select count(*) n_rows, count_if(is_test) n_test, count_if(dup_key and not is_test) n_dup,
                       count_if({base} and {info}) n_info,
                       count_if({base} and not {info} and not ({COMPLAINTS})) n_invalid,
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


def change_table(con, src, top, y0, y1, y2=None):
    """Share of each year's complaints for the overall top types in y0, y1 (and the later year y2, if any);
    change_pts is y1 minus y0, change_later_pts y2 minus y1."""
    years = [y for y in (y0, y1, y2) if y is not None]
    d = con.sql(f"""select year(created) y, canonical_type, count(*) n from {src}
                    where {COMPLAINTS} and canonical_type is not null
                      and year(created) in ({", ".join(map(str, years))})
                    group by all""").df()
    out = top[["rank", "canonical_type", "label"]].copy()
    totals = {}
    for y in years:
        dy = d[d.y == y].sort_values(["n", "canonical_type"], ascending=[False, True]).reset_index(drop=True)
        dy["rank_y"] = np.arange(1, len(dy) + 1)
        totals[y] = int(dy.n.sum())
        m = dy.set_index("canonical_type")
        out[f"n_{y}"] = out.canonical_type.map(m.n).fillna(0).astype(int)
        out[f"pct_{y}"] = 100 * out[f"n_{y}"] / max(totals[y], 1)
        out[f"rank_{y}"] = out.canonical_type.map(m.rank_y)
    out["change_pts"] = out[f"pct_{y1}"] - out[f"pct_{y0}"]
    if y2 is not None:
        out["change_later_pts"] = out[f"pct_{y2}"] - out[f"pct_{y1}"]
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


def _label_rows(xs, span, min_gap=0.09):
    """Row on which to label each year's markers: the first row where its marker is at least min_gap of the axis
    span from every other year's marker (else the row where it stands furthest from them)."""
    rows = {}
    for k, x in xs.items():
        others = [v for j, v in xs.items() if j != k]
        gap = (np.min([np.abs(x - o) for o in others], axis=0) / span) if others else np.full(len(x), np.inf)
        ok = np.flatnonzero(gap >= min_gap)
        rows[k] = int(ok[0]) if len(ok) else int(np.argmax(gap))
    return rows


def chart_change(ch, y0, y1, path, title, note, y2=None):
    """Dumbbells from y0 (sage) to y1 (coral) per type; the later year y2, if any, as a lighter gray ring."""
    from matplotlib.figure import Figure

    from nyc311 import nytplot as nyt

    nyt.use()
    gray = nyt.HUES[-1]
    fig = Figure(figsize=(8, 0.42 * len(ch) + 2.1))
    ax = fig.subplots()
    y = np.arange(len(ch))
    a, b = ch[f"pct_{y0}"].to_numpy(), ch[f"pct_{y1}"].to_numpy()
    for i in y:
        ax.plot([a[i], b[i]], [i, i], color="#cccccc", linewidth=2.2, zorder=1, solid_capstyle="butt")
    ax.scatter(a, y, color=nyt.SAGE.line, s=46, zorder=3, linewidths=0)
    ax.scatter(b, y, color=nyt.CORAL.line, s=46, zorder=3, linewidths=0)
    xs, colors = {str(y0): a, str(y1): b}, {str(y0): nyt.SAGE.line, str(y1): nyt.CORAL.line}
    if y2 is not None:
        c = ch[f"pct_{y2}"].to_numpy()
        ax.scatter(c, y, facecolor="none", edgecolor=gray.line, s=46, zorder=4, linewidths=1.4)
        xs[str(y2)], colors[str(y2)] = c, gray.line
    xmax = max(v.max() for v in xs.values()) * 1.12
    rows = _label_rows(xs, xmax)
    anns = {label: ax.annotate(label, (xs[label][row], row), xytext=(0, 9), textcoords="offset points",
                               ha="center", va="bottom", color=colors[label], fontfamily=nyt.SANS, fontweight="bold",
                               fontsize=10.5, zorder=5) for label, row in rows.items()}
    ax.set_yticks(y, ch.label)
    ax.set_ylim(len(ch) - 0.4, -1.0)
    ax.set_xlim(0, xmax)
    _hbar_axes(ax)
    nyt.format_x(ax, suffix="%")
    nyt.title(fig, title)
    nyt.note(fig, note)
    _separate_labels(fig, anns, rows, {k: xs[k][r] for k, r in rows.items()})
    return nyt.save(fig, path)


def _separate_labels(fig, anns, rows, x, pad_px=6):
    """Labels that share a row and overlap once drawn are pushed apart sideways, keeping their left-to-right order
    (that of their markers) and their mean position."""
    fig.draw_without_rendering()
    for row in set(rows.values()):
        labs = sorted((k for k, r in rows.items() if r == row), key=lambda k: x[k])
        if len(labs) < 2:
            continue
        boxes = [anns[k].get_window_extent() for k in labs]
        shift = [0.0] * len(labs)
        for i in range(1, len(labs)):
            overlap = boxes[i - 1].x1 + shift[i - 1] + pad_px - (boxes[i].x0 + shift[i])
            if overlap > 0:
                shift[i] += overlap
        mid = sum(shift) / len(shift)
        for k, dx in zip(labs, shift):
            anns[k].xyann = ((dx - mid) * 72 / fig.dpi, 9)


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
    years = clean.full_years(con, src)
    pick = comparison_years(years)
    change, totals = (None, {})
    if pick is not None:
        change, totals = change_table(con, src, top, *pick)

    res = {"top": top, "counts": counts, "total": total, "excluded": excl, "period": period, "period_long": period_l,
           "n_outside_period": int(n_outside), "n_days": (d1 - d0).days + 1, "full_years": years, "change": change,
           "totals": totals, "years": pick[:2] if pick else None, "later_year": pick[2] if pick else None}

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
                     f"How the top {len(top)} complaint types changed, {y0} vs. {y1}", _change_note(res),
                     y2=res["later_year"])
    else:
        (figs / "q05_top10_change.png").unlink(missing_ok=True)
        (agg / "q05_top10_change.csv").unlink(missing_ok=True)
    _report(cfg, res)
    _write_answer(cfg, res)
    return res


def _excluded_text(e):
    return (f"{e.n_info:,} information requests, {e.n_invalid:,} invalid entries and {e.n_test:,} test records"
            if e.n_invalid else f"{e.n_info:,} information requests and {e.n_test:,} test records")


def _note(res):
    e, top = res["excluded"], res["top"]
    outside = (f" Includes {res['n_outside_period']:,} backfilled requests dated outside this period."
               if res["n_outside_period"] else "")
    return (f"Note: Complaints and share of all {res['total']:,} complaints, {res['period_long']}; the top "
            f"{len(top)} are {top.cum_pct.iloc[-1]:.0f}% of the total. Spelling variants merged.\n"
            f"Excludes {_excluded_text(e)}; agencies in parentheses." + outside)


def _years_role(res):
    """How the compared years were chosen, for the chart note and the findings."""
    y0, y1 = res["years"]
    if res["later_year"] is not None:
        last = ("the last before the pandemic years" if y1 == rules.PROFILE_LAST_YEAR else
                f"the last one up to {rules.PROFILE_LAST_YEAR}, before the pandemic years")
        return f"{y0} is the first full calendar year in the data and {y1} {last}"
    return f"{y0} and {y1} are the first and last full calendar years in the data"


def _change_note(res):
    y0, y1 = res["years"]
    y2, t = res["later_year"], res["totals"]
    counts = ", ".join(f"{t[y]:,} in {y}" for y in (y0, y1, y2) if y is not None)
    ring = (f" Gray rings: {y2}, the last full year, shown for reference." if y2 is not None else "")
    return textwrap.fill(
        f"Note: Share of each year's complaints ({counts}). {_years_role(res)}.{ring} Types ranked by volume over "
        f"{res['period']}; information requests, invalid entries and test records excluded.", 125)


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    e = res["excluded"]
    rep.add(f"Q5. Top {TOP_N} complaint types by volume, {res['period_long']}",
            f"rows {e.n_rows:,}; left out: is_test {e.n_test:,}, dup_key {e.n_dup:,}, is_info_request {e.n_info:,}, "
            f"category '{rules.INVALID_CATEGORY}' {e.n_invalid:,}, no complaint type {e.n_no_type:,}; complaints "
            f"counted {res['total']:,} in {len(res['counts'])} canonical types",
            (f"backfilled rows dated outside the period (counted): {res['n_outside_period']:,}; "
             if res["n_outside_period"] else "") + "full calendar years: "
            + (", ".join(map(str, res["full_years"])) or "none"))
    rep.table(res["top"][["rank", "canonical_type", "category", "main_agency", "n", "pct", "cum_pct"]],
              title=f"Top {TOP_N} (pct of complaints)")
    if res["change"] is not None:
        y0, y1 = res["years"]
        later = f", plus {res['later_year']}" if res["later_year"] is not None else ""
        rep.table(res["change"].drop(columns="label"),
                  title=f"{y0} vs {y1}{later} (pct of that year's complaints)")
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
    lead = (f" and {ratio_phrase(t1.n / top.n.iloc[1])} the next type ({top.canonical_type.iloc[1]})"
            if len(top) > 1 else "")
    lines = [
        f"# Q5. Top {len(top)} complaint types by volume",
        "",
        f"*Computed by `nyc311/q05_top_types.py` from {total:,} complaints, {res['period_long']}.*",
        "",
        f"- **What is counted.** Complaints in the default analysis set, grouped by canonical complaint type (Q3's "
        f"map merges spelling variants, so {len(res['counts'])} types remain). Left out: {e.n_info:,} information "
        f"requests (`is_info_request`), {e.n_invalid:,} invalid entries (category \"{rules.INVALID_CATEGORY}\": test "
        f"records and web-form probes), {e.n_test:,} test records (`is_test`), {e.n_dup:,} duplicate keys "
        f"(`dup_key`) and {e.n_no_type:,} rows with no complaint type."
        + (f" {res['n_outside_period']:,} backfilled requests dated outside {res['period']} are counted."
           if res["n_outside_period"] else ""),
        f"- **The leader.** {t1.canonical_type} ({t1.main_agency}) has {int(t1.n):,} complaints, {t1.pct:.1f}% of "
        f"the total{lead}.",
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
        lines += [f"- **Change between {y0} and {y1}** ({_years_role(res)}). "
                  + _moves(ch, y0, y1, "change_pts")
                  + f" Complaints: {res['totals'][y0]:,} in {y0}, {res['totals'][y1]:,} in {y1}."]
        y2 = res["later_year"]
        if y2 is not None:
            lines += [f"- **{y2}, the last full year (gray rings in the chart), against {y1}.** "
                      + _moves(ch, y1, y2, "change_later_pts")
                      + f" Complaints: {res['totals'][y2]:,} in {y2}. It is shown apart so the pandemic shift does "
                      "not drive the comparison."]
        lines += ["", f"![How the top {len(top)} complaint types changed, {y0} vs. {y1}](figs/q05_top10_change.png)",
                  ""]
    else:
        n_full = len(res["full_years"])
        lines += [f"- **Change over time.** The data contain {'only one' if n_full == 1 else 'no'} full calendar "
                  f"year ({res['period_long']}), so the chart comparing full years is left out; it is drawn "
                  "automatically when the data cover two or more full years.", ""]
    lines += [
        "- **Board-ready choices.** Horizontal bars sorted by volume so the long type names read without "
        "rotation; each bar is labeled with its count and share, so no axis reading is needed; the agency that "
        "handles each type is in parentheses; the title states the period and the note states what was excluded.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q05.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _moves(ch, ya, yb, col):
    """'Biggest gain: X, a% to b% of that year's complaints (rank i to j). Biggest drop: ...' between two years."""
    up = ch.sort_values(col, ascending=False).iloc[0]
    down = ch.sort_values(col).iloc[0]

    def one(r):
        ra, rb = _rank(r[f'rank_{ya}']), _rank(r[f'rank_{yb}'])
        rank = f"rank {ra} in both years" if ra == rb else f"rank {ra} to {rb}"
        return f"{r.canonical_type}, {r[f'pct_{ya}']:.1f}% to {r[f'pct_{yb}']:.1f}% of that year's complaints ({rank})"
    return f"Biggest gain: {one(up)}. Biggest drop: {one(down)}."


def _rank(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v))
