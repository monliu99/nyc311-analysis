"""Q5: the top 10 complaint types by volume as a board-ready chart.

Counts canonical complaint types (Q3's map, so spelling variants are merged) among complaints (rules.COMPLAINTS_SQL:
not is_test, not dup_key, not is_info_request, not taxonomy category "Invalid entries"), so totals match Q9-Q11.
When the data hold at least two full calendar years (clean.full_years), a second chart compares the first full year
with the last full year up to rules.PROFILE_LAST_YEAR (2011 vs 2019 on the full data) and shows the last full year
after it (2020) as a lighter third marker, so the pandemic shift is visible without driving the comparison.
A like-for-like check guards the change findings against relabeling: when an agency recoding moved a descriptor
from one type to another between the compared years (HPD's PLUMBING 'WATER-LEAKS' became its own WATER LEAK type in
2014), the moved complaints are counted under their first-year type in both years (relabelings, like_for_like).
Every number in the report, the charts and outputs/answers/q05.md is computed here.
"""
from __future__ import annotations

import re
import textwrap

import numpy as np
import pandas as pd

from nyc311 import clean, rules, taxonomy
from nyc311.clean import clean_glob
from nyc311.report import Report

STAGE = "q05_top_types"
TOP_N = 10
AGENCY_LABEL_SHARE = 0.10   # agencies with at least this share of a type's rows are named in its label
COMPLAINTS = rules.COMPLAINTS_SQL
TIMES_FROM = 1.5            # "N times the next type" from this ratio up; below it, "N% more"
EN = "–"
MOVE_WINDOW_DAYS = 14       # a descriptor that stops and a same-agency type or descriptor of the same name that starts
                            # within this many days of it are one relabeling (HPD's 2014 recoding retired descriptors
                            # from late February and started the new types on March 11)
MOVE_MIN_PCT = 5            # a relabeling is applied to a top type when the moved descriptor held at least this % of
                            # the type's complaints in the first compared year
MOVES_SHOWN = 3             # relabelings spelled out per top type in the findings


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


# ---------- like-for-like check: relabeled descriptors ----------

def name_key(s):
    """Matching key for descriptor and type names: taxonomy.normalize (case, spacing, separators, plural), with '-'
    read as a space, so 'WATER-LEAKS' and 'WATER LEAK' match; '' for a missing name."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[-\s]+", " ", taxonomy.normalize(s)).strip()


def descriptor_combos(con, src, years=()):
    """One row per (agency, raw complaint_type, canonical_type, descriptor) among complaints: rows, first and last
    opening time, and rows opened in each of `years` (n_<year>). A grouped aggregate: a few thousand rows."""
    ny = "".join(f", count_if(year(created) = {int(y)}) as n_{int(y)}" for y in years)
    df = con.sql(f"""select coalesce(agency, '(none)') as agency, complaint_type, canonical_type, descriptor,
                            count(*) as n, min(created) as first, max(created) as last{ny}
                     from {src} where {COMPLAINTS} and canonical_type is not null and created is not null
                     group by all""").df()
    for c in ["first", "last"]:
        df[c] = pd.to_datetime(df[c])
    return df


def _year_cols(df):
    return [c for c in df.columns if re.fullmatch(r"n_\d{4}", c)]


MOVE_COLUMNS = ["agency", "old_type", "old_raw_type", "old_descriptor", "n_old", "last_old", "new_type",
                "new_raw_type", "new_descriptor", "via", "n_new", "first_new", "days", "weight"]


def relabelings(combos, d0, d1, window=MOVE_WINDOW_DAYS):
    """Descriptors an agency moved from one canonical type to another: one row per (old, new) pair.

    Old: an (agency, canonical type, descriptor) group (descriptor names matched with name_key) whose rows stop more
    than `window` days before the data end (d1). New: a group of the same agency under another canonical type that
    starts within `window` days of that stop, and more than `window` days after the data start (d0): either a
    descriptor of the same name (via 'descriptor') or a whole raw complaint type named like the old descriptor (via
    'type'; it then supersedes descriptor matches in the same canonical type). weight: the old group's share (by
    rows) of all old groups feeding the same new group, so a successor fed by several is split among them. Year
    columns n_<year> of `combos` come through as n_<year>_old and n_<year>_new."""
    if combos.empty:
        return pd.DataFrame(columns=MOVE_COLUMNS)
    c = combos.assign(dkey=combos.descriptor.map(name_key), tkey=combos.complaint_type.map(name_key))
    c = c.sort_values(["n", "complaint_type"], ascending=[False, True], na_position="last")
    win = pd.Timedelta(days=window)
    ys = _year_cols(c)
    agg = {"n": "sum", "first": "min", "last": "max", **{y: "sum" for y in ys}}
    by_desc = c[c.dkey != ""].groupby(["agency", "canonical_type", "dkey"], as_index=False).agg(
        {**agg, "complaint_type": "first", "descriptor": "first"}).rename(columns={"dkey": "key"})
    old = by_desc[by_desc["last"] < pd.Timestamp(d1) - win]
    by_desc["via"] = "descriptor"
    by_type = c.groupby(["agency", "canonical_type", "complaint_type"], as_index=False).agg(
        {**agg, "tkey": "first"}).rename(columns={"tkey": "key"})
    by_type["descriptor"], by_type["via"] = None, "type"
    new = pd.concat([by_desc, by_type], ignore_index=True)
    new = new[(new.key != "") & (new["first"] > pd.Timestamp(d0) + win)]
    m = old.merge(new, on=["agency", "key"], suffixes=("_old", "_new"))
    m = m[m.canonical_type_old != m.canonical_type_new]
    m = m.assign(days=(m.first_new.dt.normalize() - m.last_old.dt.normalize()).dt.days)
    m = m[m.days.abs() <= window]
    has_type = m[m.via == "type"][["agency", "key", "canonical_type_old", "canonical_type_new"]].drop_duplicates()
    m = m.merge(has_type.assign(_t=True), how="left", on=list(has_type.columns))
    m = m[(m.via == "type") | m._t.isna()].drop(columns="_t")
    if m.empty:
        return pd.DataFrame(columns=MOVE_COLUMNS)
    succ = ["agency", "canonical_type_new", "via", "complaint_type_new", "key"]
    m["weight"] = m.n_old / m.groupby(succ, dropna=False).n_old.transform("sum")
    m = m.rename(columns={"canonical_type_old": "old_type", "complaint_type_old": "old_raw_type",
                          "descriptor_old": "old_descriptor", "canonical_type_new": "new_type",
                          "complaint_type_new": "new_raw_type", "descriptor_new": "new_descriptor"})
    for c_ in ["n_old", "n_new"] + [f"{y}_old" for y in ys] + [f"{y}_new" for y in ys]:
        m[c_] = m[c_].astype(int)
    keep = MOVE_COLUMNS + [f"{y}_old" for y in ys] + [f"{y}_new" for y in ys]
    return m.sort_values(["n_old", "old_type"], ascending=[False, True])[keep].reset_index(drop=True)


LL_COLUMNS = ["year_a", "year_b", "canonical_type", "n_a", "pct_a", "n_b", "pct_b", "change_pts", "n_b_same_labels",
              "pct_b_same_labels", "change_same_labels_pts", "rows_moved_out_b", "rows_moved_in_b", "relabeled"]


def like_for_like(change, moves, ya, yb, totals, min_pct=MOVE_MIN_PCT):
    """Each top type's share in ya and yb, with relabeled descriptors counted under their ya type in yb too.

    A relabeling counts for top type T between ya and yb when the old descriptor has rows in ya and none from Jan 1
    of yb, its successor has rows in yb, it left T or joined T, and the old descriptor held at least `min_pct` % of
    T's ya complaints. T's yb count on ya's labels = its yb count + the weighted yb rows of successors of
    descriptors that left T - those of descriptors that joined T. Returns (one row per top type, the relabelings
    applied with the top type and direction)."""
    a_col, b_col = f"n_{ya}_old", f"n_{yb}_new"
    if len(moves) and {a_col, b_col} <= set(moves.columns):
        mv = moves[(moves[a_col] > 0) & (moves.last_old < pd.Timestamp(int(yb), 1, 1)) & (moves[b_col] > 0)]
    else:
        mv = pd.DataFrame(columns=MOVE_COLUMNS + [a_col, b_col])
    rows, applied = [], []
    for r in change.itertuples(index=False):
        t, n_a, n_b = r.canonical_type, getattr(r, f"n_{ya}"), getattr(r, f"n_{yb}")
        big = mv[a_col] >= min_pct / 100 * n_a
        out, inn = mv[(mv.old_type == t) & big], mv[(mv.new_type == t) & big]
        add = float((out.weight * out[b_col]).sum()) if len(out) else 0.0
        sub = float((inn.weight * inn[b_col]).sum()) if len(inn) else 0.0
        n_ll = n_b + add - sub
        pa, pb = 100 * n_a / max(totals[ya], 1), 100 * n_b / max(totals[yb], 1)
        pll = 100 * n_ll / max(totals[yb], 1)
        rows.append({"year_a": int(ya), "year_b": int(yb), "canonical_type": t, "n_a": int(n_a), "pct_a": pa,
                     "n_b": int(n_b), "pct_b": pb, "change_pts": pb - pa, "n_b_same_labels": n_ll,
                     "pct_b_same_labels": pll, "change_same_labels_pts": pll - pa, "rows_moved_out_b": add,
                     "rows_moved_in_b": sub, "relabeled": bool(len(out) or len(inn))})
        applied += [x.assign(top_type=t, direction=d, year_a=int(ya), year_b=int(yb))
                    for x, d in ((out, "out"), (inn, "in")) if len(x)]
    ll = pd.DataFrame(rows, columns=LL_COLUMNS)
    used = (pd.concat(applied, ignore_index=True) if applied else
            pd.DataFrame(columns=MOVE_COLUMNS + ["top_type", "direction", "year_a", "year_b"]))
    return ll, used


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
    moves, ll, used = pd.DataFrame(columns=MOVE_COLUMNS), None, None
    if pick is not None:
        change, totals = change_table(con, src, top, *pick)
        moves = relabelings(descriptor_combos(con, src, [y for y in pick if y is not None]), d0, d1)
        pairs = [(pick[0], pick[1])] + ([(pick[1], pick[2])] if pick[2] is not None else [])
        parts = [like_for_like(change, moves, a, b, totals) for a, b in pairs]
        ll = pd.concat([x for x, _ in parts], ignore_index=True)
        used = pd.concat([u for _, u in parts if len(u)] or [parts[0][1]], ignore_index=True)

    res = {"top": top, "counts": counts, "total": total, "excluded": excl, "period": period, "period_long": period_l,
           "n_outside_period": int(n_outside), "n_days": (d1 - d0).days + 1, "full_years": years, "change": change,
           "totals": totals, "years": pick[:2] if pick else None, "later_year": pick[2] if pick else None,
           "relabelings": moves, "like_for_like": ll, "relabelings_applied": used}

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    top.to_csv(agg / "q05_top10.csv", index=False)
    counts.to_csv(agg / "q05_type_counts.csv", index=False)
    figs = cfg.out_dir / "figs"
    chart_top(top, figs / "q05_top10.png", f"The {len(top)} most common 311 complaints, {period}", _note(res))
    if change is not None:
        y0, y1 = res["years"]
        change.to_csv(agg / "q05_top10_change.csv", index=False)
        ll.to_csv(agg / "q05_like_for_like.csv", index=False)
        used.to_csv(agg / "q05_relabelings.csv", index=False)
        chart_change(change, y0, y1, figs / "q05_top10_change.png",
                     f"How the top {len(top)} complaint types changed, {y0} vs. {y1}", _change_note(res),
                     y2=res["later_year"])
    else:
        (figs / "q05_top10_change.png").unlink(missing_ok=True)
        for name in ["q05_top10_change.csv", "q05_like_for_like.csv", "q05_relabelings.csv"]:
            (agg / name).unlink(missing_ok=True)
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
        ll, used = res["like_for_like"], res["relabelings_applied"]
        if ll.relabeled.any():
            rep.table(ll[ll.relabeled][["year_a", "year_b", "canonical_type", "pct_a", "pct_b", "pct_b_same_labels",
                                        "change_pts", "change_same_labels_pts"]],
                      title="Like for like: top types with relabeled descriptors, counted under their year_a type")
            rep.table(used[["year_a", "top_type", "direction", "agency", "old_raw_type", "old_descriptor", "n_old",
                            "last_old", "new_raw_type", "new_descriptor", "first_new"]].assign(
                last_old=lambda x: x.last_old.dt.strftime("%Y-%m-%d"),
                first_new=lambda x: x.first_new.dt.strftime("%Y-%m-%d")),
                title=f"Relabelings applied (window {MOVE_WINDOW_DAYS} days, >= {MOVE_MIN_PCT:g}% of the top type)")
        else:
            rep.add("", f"like for like: no top type gained or lost a descriptor through a recoding "
                        f"({len(res['relabelings'])} relabelings detected among all types)")
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
        ll, used = res["like_for_like"], res["relabelings_applied"]
        lines += [f"- **Change between {y0} and {y1}** ({_years_role(res)}). "
                  + _moves(ch, y0, y1, "change_pts")
                  + f" Complaints: {res['totals'][y0]:,} in {y0}, {res['totals'][y1]:,} in {y1}.",
                  _relabel_line(ll, used, y0, y1)]
        y2 = res["later_year"]
        if y2 is not None:
            lines += [f"- **{y2}, the last full year (gray rings in the chart), against {y1}.** "
                      + _moves(ch, y1, y2, "change_later_pts")
                      + f" Complaints: {res['totals'][y2]:,} in {y2}. It is shown apart so the pandemic shift does "
                      "not drive the comparison."]
            u2 = used[used.year_a == y1] if len(used) else used
            if len(u2):
                lines += [_relabel_line(ll, used, y1, y2)]
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


def _pts(x):
    return f"{x:+.1f}".replace("-", "\N{MINUS SIGN}") + " points"


def _move_text(m):
    """One relabeling, from the top type's side: 'lost HPD PLUMBING 'WATER-LEAKS' (...) to the WATER LEAK type ...'."""
    old = (f"{m.agency} {m.old_raw_type} '{m.old_descriptor}' ({int(m.n_old):,} complaints, last "
           f"{pd.Timestamp(m.last_old):%Y-%m-%d})")
    new = f"the {m.new_raw_type} type" if m.via == "type" else f"{m.new_raw_type} '{m.new_descriptor}'"
    first = f"from {pd.Timestamp(m.first_new):%Y-%m-%d}"
    if m.direction == "out":
        return f"lost {old} to {new} ({m.new_type}) {first}"
    old = old.replace(" (", f" ({m.old_type}; ", 1)
    return f"took over {old} as {new} {first}"


def _relabel_line(ll, used, ya, yb):
    """The like-for-like bullet for one comparison: which top types gained or lost descriptors through a recoding,
    their shares with moved complaints counted under their `ya` type, and the biggest drop (and gain) on that basis."""
    head = f"- **Like-for-like check ({ya} vs. {yb}).** "
    rule = (f"a descriptor with at least {MOVE_MIN_PCT:g}% of a top type's {ya} complaints that stops while a "
            f"same-agency type or descriptor of the same name starts within {MOVE_WINDOW_DAYS} days")
    u = used[(used.year_a == ya) & (used.year_b == yb)] if len(used) else used
    if not len(u):
        return (head + f"No top type gained or lost a descriptor through an agency recoding between {ya} and {yb} "
                f"(checked: {rule}), so these changes are not relabelings of that kind.")
    x = ll[(ll.year_a == ya) & (ll.year_b == yb)]
    shift = (x.change_pts - x.change_same_labels_pts).abs()
    order = list(x.assign(_s=shift).sort_values("_s", ascending=False).canonical_type)
    parts = []
    for t in [t for t in order if t in set(u.top_type)]:  # the type whose number moves most first
        g = u[u.top_type == t]
        r = x[x.canonical_type == t].iloc[0]
        moves = [_move_text(m) for m in g.itertuples()]
        more = len(moves) - MOVES_SHOWN
        moves = moves[:MOVES_SHOWN] + ([f"{more} more"] if more > 0 else [])
        said = moves[0] if len(moves) == 1 else ", ".join(moves[:-1]) + " and " + moves[-1]
        parts.append(f"{t} {said}: counted on {ya}'s labels, it goes from {r.pct_a:.1f}% to "
                     f"{r.pct_b_same_labels:.1f}% of complaints ({_pts(r.change_same_labels_pts)}; "
                     f"{_pts(r.change_pts)} as recorded).")
    lo, lo_raw = x.loc[x.change_same_labels_pts.idxmin()], x.loc[x.change_pts.idxmin()]
    hi, hi_raw = x.loc[x.change_same_labels_pts.idxmax()], x.loc[x.change_pts.idxmax()]

    def one(r):
        return f"{r.canonical_type}, {r.pct_a:.1f}% to {r.pct_b_same_labels:.1f}% ({_pts(r.change_same_labels_pts)})"
    end = (f"On that basis the biggest drop is {one(lo)}, not {lo_raw.canonical_type}" if lo.canonical_type !=
           lo_raw.canonical_type else f"On that basis the biggest drop is still {one(lo)}")
    if hi.canonical_type != hi_raw.canonical_type:
        end += f"; the biggest gain is {one(hi)}, not {hi_raw.canonical_type}"
    return (head + f"Part of these changes is relabeling (found by the rule: {rule}). " + " ".join(parts) + " "
            + end + ".")
