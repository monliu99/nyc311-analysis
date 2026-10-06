"""Q7: where complaints concentrate. Complaints per 1,000 residents a year by community district, the category
mix behind the gaps, and the link to median household income.

Rates divide complaints by resident-years: 2010 Census population for days in 2010-2014 and 2020 Census
population for days from 2015 on, summed over the complete days in the data. Parks/airports (special_area_cd)
and complaints with no district are excluded and reported. The choropleth is drawn with matplotlib patches from
the DCP GeoJSON (no geopandas). The three insights in the figure and in outputs/answers/q07.md are f-strings
over computed values; the same code runs on one month or twelve years.
"""
from __future__ import annotations

import re
import textwrap

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from nyc311 import geo, rules
from nyc311 import nytplot as nyt
from nyc311.clean import clean_glob
from nyc311.report import Report

STAGE = "q07_geography"
DEMO_CSV = geo.DISTRICTS_GEOJSON.parent / "cd_demographics.csv"
HOUSING = ("Heat & hot water", "Housing conditions")
POP_2020_FROM_YEAR = 2015   # days before 2015 use population_2010; days from 2015 use population_2020
DAYS_PER_YEAR = 365.25
TOP_K = 10                  # "top" and "bottom" district groups in the insights
N_CLASSES = 5               # map classes: quintiles of the 59 districts
MIN_N = 100                 # districts with fewer complaints are hatched on the map and flagged in tables
INCOME_GROUPS = 5           # income fifths of PUMAs
HATCH_GRAY = "#a6a6a6"
MAP_LAT = 40.7              # aspect correction for lon/lat


# ------------------------------------------------------------------ pure functions
def load_demographics(path=DEMO_CSV):
    return pd.read_csv(path, dtype={"puma_2010": str})


def person_years(demo, days_2010, days_2020):
    """Resident-years per district: population_2010 for days before POP_2020_FROM_YEAR, population_2020 after."""
    d = demo.set_index("boro_cd")
    return (d.population_2010 * days_2010 + d.population_2020 * days_2020) / DAYS_PER_YEAR


def district_table(counts, demo, py):
    """One row per district: complaints, housing complaints, rates per 1,000 residents a year, top category."""
    tot = counts.groupby("boro_cd").n.sum()
    hous = counts[counts.category.isin(HOUSING)].groupby("boro_cd").n.sum()
    top = counts.sort_values("n", ascending=False).drop_duplicates("boro_cd").set_index("boro_cd")
    keep = ["borough", "cd_number", "cd_name", "median_hh_income", "puma_2010"]
    d = demo.set_index("boro_cd")[[c for c in keep if c in demo.columns]].copy()
    d["n"] = tot.reindex(d.index).fillna(0).astype(int)
    d["n_housing"] = hous.reindex(d.index).fillna(0).astype(int)
    d["person_years"] = py.reindex(d.index)
    d["rate"] = 1000 * d.n / d.person_years
    d["housing_rate"] = 1000 * d.n_housing / d.person_years
    d["other_rate"] = d.rate - d.housing_rate
    d["housing_share_pct"] = 100 * d.n_housing / d.n.where(d.n > 0)
    d["top_category"] = top.category.reindex(d.index)
    d["top_category_pct"] = 100 * top.n.reindex(d.index) / d.n.where(d.n > 0)
    d["low_n"] = d.n < MIN_N
    return d.reset_index()


def gap_share(top, bottom):
    """Percent of the total-rate gap between two groups that housing complaints account for."""
    return 100 * (top["housing_rate"] - bottom["housing_rate"]) / (top["rate"] - bottom["rate"])


def classify(rates, k=N_CLASSES):
    """Quantile classes 0..k-1 (ties broken by order, so every class is filled) and the k+1 class edges."""
    rates = pd.Series(rates)
    k = min(k, len(rates))
    cls = pd.Series(pd.qcut(rates.rank(method="first"), k, labels=False), index=rates.index).astype(int)
    edges = [float(rates[cls == c].min()) for c in range(k)] + [float(rates.max())]
    return cls, edges


def puma_units(d):
    """Pool districts that share one PUMA (and so one income value): 59 districts -> 55 income areas."""
    p = d.groupby("puma_2010").agg(median_hh_income=("median_hh_income", "first"), n=("n", "sum"),
                                   n_housing=("n_housing", "sum"), person_years=("person_years", "sum"),
                                   districts=("boro_cd", lambda s: ", ".join(str(x) for x in sorted(s))))
    p["rate"] = 1000 * p.n / p.person_years
    p["housing_rate"] = 1000 * p.n_housing / p.person_years
    p["other_rate"] = p.rate - p.housing_rate
    return p.reset_index()


def _group_rates(g):
    py = g.person_years.sum()
    return {"rate": 1000 * g.n.sum() / py, "housing_rate": 1000 * g.n_housing.sum() / py,
            "housing_share": 100 * g.n_housing.sum() / g.n.sum()}


def _boro(b):
    return "the Bronx" if b == "Bronx" else b


def _name(r):
    return f"{r.borough} {r.cd_number} ({r.cd_name})"


def insight_facts(d, p, k=TOP_K, counts=None):
    """Every number the three insights quote, computed from the district and PUMA tables."""
    top, bottom = d.loc[d.rate.idxmax()], d.loc[d.rate.idxmin()]
    topk, botk = d.nlargest(k, "rate"), d.nsmallest(k, "rate")
    gt, gb = _group_rates(topk), _group_rates(botk)
    boro = topk.borough.value_counts()
    f = {"k": k, "n_districts": len(d), "top_name": _name(top), "top_rate": top.rate, "bottom_name": _name(bottom),
         "bottom_rate": bottom.rate, "ratio": top.rate / bottom.rate,
         "top_pop_share": 100 * topk.person_years.sum() / d.person_years.sum(),
         "top_n_share": 100 * topk.n.sum() / d.n.sum(), "top_boro": boro.index[0], "top_boro_count": int(boro.iloc[0]),
         "rate_top": gt["rate"], "rate_bot": gb["rate"], "h_top": gt["housing_share"], "h_bot": gb["housing_share"],
         "other_top": gt["rate"] - gt["housing_rate"], "other_bot": gb["rate"] - gb["housing_rate"],
         "gap_housing_pct": gap_share(gt, gb), "gap_cats": []}
    if counts is not None:
        c = counts.pivot_table(index="boro_cd", columns="category", values="n", aggfunc="sum", fill_value=0)
        py = d.set_index("boro_cd").person_years
        rate_t = 1000 * c.reindex(topk.boro_cd).sum() / py.reindex(topk.boro_cd).sum()
        rate_b = 1000 * c.reindex(botk.boro_cd).sum() / py.reindex(botk.boro_cd).sum()
        contrib = (100 * (rate_t - rate_b) / (gt["rate"] - gb["rate"])).sort_values(ascending=False)
        f["gap_cats"] = list(contrib.head(3).items())
    p = p.sort_values("median_hh_income").reset_index(drop=True)
    q = pd.qcut(p.median_hh_income.rank(method="first"), INCOME_GROUPS, labels=False)
    lo, hi = p[q == 0], p[q == INCOME_GROUPS - 1]
    f.update({"n_units": len(p), "inc_low_max": lo.median_hh_income.max(), "inc_high_min": hi.median_hh_income.min(),
              "inc_low_housing": 1000 * lo.n_housing.sum() / lo.person_years.sum(),
              "inc_high_housing": 1000 * hi.n_housing.sum() / hi.person_years.sum(),
              "inc_low_other": 1000 * (lo.n - lo.n_housing).sum() / lo.person_years.sum(),
              "inc_high_other": 1000 * (hi.n - hi.n_housing).sum() / hi.person_years.sum(),
              "rho_housing": p.median_hh_income.corr(p.housing_rate, method="spearman"),
              "rho_other": p.median_hh_income.corr(p.other_rate, method="spearman")})
    return f


def insights(f):
    """The three insights for the Mayor, as text built only from insight_facts()."""
    one = (f"{f['top_name']} files {f['top_rate']:,.0f} complaints per 1,000 residents a year, {f['ratio']:.1f}x "
           f"{f['bottom_name']}, the lowest at {f['bottom_rate']:,.0f}. The {f['k']} highest-rate districts hold "
           f"{f['top_pop_share']:.0f}% of residents but {f['top_n_share']:.0f}% of complaints; "
           f"{f['top_boro_count']} of the {f['k']} are in {_boro(f['top_boro'])}.")
    two = (f"Housing drives the gap. Heat, hot water and housing-condition complaints are {f['h_top']:.0f}% of "
           f"complaints in the top {f['k']} districts vs {f['h_bot']:.0f}% in the bottom {f['k']}, and account for "
           f"{f['gap_housing_pct']:.0f}% of the gap in rates ({f['rate_top']:,.0f} vs {f['rate_bot']:,.0f} per "
           f"1,000). All other complaints: {f['other_top']:,.0f} vs {f['other_bot']:,.0f}.")
    if f["gap_cats"]:
        two += " Largest contributors: " + ", ".join(f"{c} {v:.0f}%" for c, v in f["gap_cats"]) + "."
    three = (f"Income. In the poorest fifth of the {f['n_units']} income areas (median household income up to "
             f"${f['inc_low_max'] / 1000:,.0f}k), residents file {f['inc_low_housing']:,.0f} housing complaints per "
             f"1,000 a year vs {f['inc_high_housing']:,.0f} in the richest fifth (${f['inc_high_min'] / 1000:,.0f}k "
             f"and up). Other complaints: {f['inc_low_other']:,.0f} vs {f['inc_high_other']:,.0f}. Rank correlation "
             f"with income: housing {f['rho_housing']:+.2f}, other {f['rho_other']:+.2f}.")
    return [one, two, three]


# ------------------------------------------------------------------ data
def analysis_days(con, src):
    """Complete days: dates with at least one in-order row, minus possibly_incomplete_day."""
    days = con.sql(f"""select created::date as d, bool_or(not out_of_file_order) as covered,
                       bool_or(possibly_incomplete_day) as inc
                       from {src} where not is_test and not dup_key and created is not null group by 1""").df()
    days["d"] = pd.to_datetime(days.d)
    return days[days.covered & ~days.inc].d.sort_values().reset_index(drop=True)


def district_counts(con, src, days):
    con.register("q07_days", pd.DataFrame({"d": days}))
    raw = con.sql(f"""
        with b as (select created::date as d, coalesce(is_info_request, false) as info, possibly_incomplete_day as inc,
                          boro_cd, special_area_cd is not null as special, borough,
                          coalesce(category, '(unmapped)') as category
                   from {src} where not is_test and not dup_key),
        cov as (select distinct created::date as cd from {src}
                where not is_test and not dup_key and not out_of_file_order and created is not null)
        select b.d is null as nodate, cov.cd is not null as covered, info, inc, boro_cd, special, borough, category,
               count(*)::bigint as n
        from b left join cov on b.d = cov.cd group by all""").df()
    con.unregister("q07_days")
    default = int(raw.n.sum())
    steps, keep = [("default analysis set", default)], raw

    def drop(label, mask):
        nonlocal keep
        steps.append((label, int(keep.n[mask].sum())))
        keep = keep[~mask]

    drop("information requests (is_info_request)", keep["info"])
    drop("no valid created date", keep.nodate)
    drop("dates with only out_of_file_order rows (outside the files' period)", ~keep.covered)
    drop("possibly_incomplete_day", keep.inc)
    drop("park or airport (special_area_cd)", keep.special)
    no_cd = keep.boro_cd.isna()
    no_district = keep[no_cd]
    drop("no community district", no_cd)
    steps.append(("mapped to a community district", int(keep.n.sum())))
    ex = pd.DataFrame(steps, columns=["step", "rows"])
    ex["pct"] = 100 * ex.rows / default
    counts = keep.assign(boro_cd=keep.boro_cd.astype(int)).groupby(["boro_cd", "category"], as_index=False).n.sum()
    return counts, ex, no_district


def unknown_by_agency(con, src):
    return con.sql(f"""select agency, count(*)::bigint as n from {src}
                       where not is_test and not dup_key and not coalesce(is_info_request, false)
                       and not possibly_incomplete_day and boro_cd is null and special_area_cd is null
                       group by 1 order by n desc""").df()


def borough_table(d, no_district, demo):
    names = demo.assign(code=demo.boro_cd // 100).groupby("code").borough.first()
    up = {k: names[v] for k, v in rules.BORO_CODE.items() if v in names.index}
    b = d.groupby("borough").agg(districts=("boro_cd", "size"), n=("n", "sum"), n_housing=("n_housing", "sum"),
                                 person_years=("person_years", "sum"))
    b["rate"] = 1000 * b.n / b.person_years
    b["housing_rate"] = 1000 * b.n_housing / b.person_years
    b["housing_share_pct"] = 100 * b.n_housing / b.n
    nd = no_district.assign(b=no_district.borough.map(up)).groupby("b").n.sum()
    b["n_with_borough_no_district"] = nd.reindex(b.index).fillna(0).astype(int)
    b["rate_incl_no_district"] = 1000 * (b.n + b.n_with_borough_no_district) / b.person_years
    b["highest_district"] = d.loc[d.groupby("borough").rate.idxmax()].set_index("borough").cd_name
    return b.sort_values("rate", ascending=False).reset_index()


# ------------------------------------------------------------------ figures
def ramp(k, hue=nyt.SAGE):
    rgb = np.array(plt.matplotlib.colors.to_rgb(hue.line))
    darkest = rgb * 0.72
    shades = np.linspace(0.12, 1.0, k)
    return [plt.matplotlib.colors.to_hex(1 - a * (1 - darkest)) for a in shades]


def _area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def polygon_path(outer, holes):
    """One compound path per polygon part; holes are wound opposite to the outer ring so they stay empty."""
    rings = [outer.vertices]
    for h in holes:
        v = h.vertices
        rings.append(v[::-1] if np.sign(_area(v)) == np.sign(_area(outer.vertices)) else v)
    verts = np.concatenate(rings)
    codes = np.concatenate([[MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 1) for r in rings])
    return MplPath(verts, codes)


def _fig_text(s):
    return s.replace("$", r"\$")


def fig_map(d, texts, path, note):
    nyt.use()
    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax, tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    cls, edges = classify(d.rate)
    colors = ramp(len(edges) - 1)
    fill = dict(zip(d.boro_cd, [colors[c] for c in cls]))
    low = set(d.boro_cd[d.low_n])
    parts = {}
    for code, outer, holes, lo, hi in geo.load_polygons():
        shape = polygon_path(outer, holes)
        ax.add_patch(PathPatch(shape, facecolor=fill.get(code, "white"), edgecolor="white", lw=0.5))
        hatch = "...." if code not in fill else "////" if code in low else None
        if hatch:  # overlay so the hatch can be gray while district borders stay white
            ax.add_patch(PathPatch(shape, facecolor="none", edgecolor=HATCH_GRAY, lw=0, hatch=hatch))
        area = abs(_area(outer.vertices))
        if area > parts.get(code, (0, None))[0]:
            parts[code] = (area, outer.vertices.mean(axis=0))
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(MAP_LAT)))
    ax.axis("off")
    ranked = d.sort_values("rate", ascending=False)
    for r in list(ranked.head(3).itertuples()) + [ranked.iloc[-1]]:
        x, y = parts[r.boro_cd][1]
        ax.text(x, y, f"{r.borough} {r.cd_number}\n{r.rate:,.0f}", ha="center", va="center", fontsize=8.5,
                fontfamily=nyt.SANS, fontweight="bold", color="#111111",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    key = ax.inset_axes([0.02, 0.90, 0.46, 0.03])
    for i, c in enumerate(colors):
        key.add_patch(Rectangle((i, 0), 1, 1, color=c))
    key.set_xlim(0, len(colors))
    key.set_ylim(0, 1)
    key.set_xticks(range(len(edges)))
    key.set_xticklabels([f"{e:,.0f}" for e in edges], fontsize=8.5)
    key.set_yticks([])
    for s in key.spines.values():
        s.set_visible(False)
    key.grid(False)
    key.set_title("Complaints per 1,000 residents a year (district fifths)", loc="left", fontsize=9.5,
                  fontfamily=nyt.SANS)
    ax.add_patch(Rectangle((0.80, 0.06), 0.035, 0.03, transform=ax.transAxes, facecolor="white",
                           edgecolor=HATCH_GRAY, lw=0.5, hatch="...."))
    ax.text(0.845, 0.075, "Parks & airports\n(not rated)", transform=ax.transAxes, va="center", fontsize=8.5,
            color=nyt.NOTE_GRAY)
    tx.axis("off")
    tx.text(0, 1, "Three insights for the Mayor", fontfamily=nyt.SANS, fontweight="bold", fontsize=13, va="top")
    y = 0.93
    for i, t in enumerate(texts, 1):
        tx.text(0, y, str(i), fontfamily=nyt.SANS, fontweight="bold", fontsize=15, color=nyt.CORAL.line, va="top")
        body = textwrap.fill(_fig_text(t), 52)
        tx.text(0.07, y, body, fontsize=10.5, va="top", linespacing=1.35)
        y -= 0.04 * (body.count("\n") + 1) + 0.035
    nyt.title(fig, f"311 complaints per resident vary {d.rate.max() / d.rate.min():.1f}-fold across "
                   f"{len(d)} community districts")
    nyt.note(fig, textwrap.fill(note, 175))
    return nyt.save(fig, path)


def fig_district_rates(d, city_rate, path, period):
    nyt.use()
    order = d.groupby("borough").rate.apply(lambda s: 1000 * d.loc[s.index, "n"].sum() /
                                            d.loc[s.index, "person_years"].sum()).sort_values(ascending=False)
    sizes = d.borough.value_counts().reindex(order.index)
    fig, axes = plt.subplots(len(order), 1, figsize=(10, 0.24 * len(d) + 2.2), sharex=True,
                             gridspec_kw={"height_ratios": sizes.to_numpy() + 1.2})
    xmax = d.rate.max() * 1.02
    for ax, (boro, brate) in zip(np.atleast_1d(axes), order.items()):
        s = d[d.borough == boro].sort_values("rate")
        y = np.arange(len(s))
        ax.barh(y, s.housing_rate, color=nyt.CORAL.line, height=0.72)
        ax.barh(y, s.other_rate, left=s.housing_rate, color=nyt.SAGE.band, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.cd_number} {textwrap.shorten(r.cd_name, 30, placeholder='...')}"
                            for r in s.itertuples()], fontsize=9)
        ax.axvline(city_rate, color=nyt.NOTE_GRAY, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(s) - 0.4)
        nyt.panel_title(ax, f"{boro}: {brate:,.0f} per 1,000", fontsize=10.5)
        nyt.format_x(ax)
    first = np.atleast_1d(axes)[0]
    first.text(0.99, 0.30, "Housing (heat, hot water, conditions)", transform=first.transAxes, ha="right",
               color=nyt.CORAL.line, fontfamily=nyt.SANS, fontweight="bold", fontsize=10)
    first.text(0.99, 0.12, "All other complaints", transform=first.transAxes, ha="right", color=nyt.SAGE.line,
               fontfamily=nyt.SANS, fontweight="bold", fontsize=10)
    first.text(city_rate, 1.0, f" citywide {city_rate:,.0f}", transform=first.get_xaxis_transform(), va="bottom",
               fontsize=8.5, color=nyt.NOTE_GRAY)
    top = d.loc[d.rate.idxmax()]
    nyt.title(fig, f"Complaints per 1,000 residents a year: {top.borough} {top.cd_number} leads at {top.rate:,.0f}, "
                   f"{100 * top.n_housing / top.n:.0f}% of it housing")
    nyt.note(fig, f"Note: {period}; boroughs ordered by rate, districts within each by rate. Dashed line: citywide "
                  "rate for the 59 districts.")
    return nyt.save(fig, path)


def fig_income(p, f, path, period):
    nyt.use()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    x = p.median_hh_income / 1000
    for ax, col, h, label, rho in [(axes[0], "housing_rate", nyt.CORAL, "Housing complaints", f["rho_housing"]),
                                   (axes[1], "other_rate", nyt.SAGE, "All other complaints", f["rho_other"])]:
        ax.scatter(x, p[col], color=h.line, s=22, alpha=0.85, linewidths=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.03)
        nyt.format_y(ax, nbins=4)
        nyt.panel_title(ax, f"{label} per 1,000 residents a year (rank correlation with income {rho:+.2f})",
                        fontsize=10.5)
    nyt.format_x(axes[1], prefix="$", suffix="k", nbins=6)
    nyt.title(fig, f"Poorest fifth of areas: {f['inc_low_housing']:,.0f} housing complaints per 1,000 vs "
                   f"{f['inc_high_housing']:,.0f} in the richest")
    nyt.note(fig, textwrap.fill(
        f"Note: each dot is one of {len(p)} PUMAs (59 districts; 4 pairs share a PUMA and are pooled). Median "
        f"household income: ACS 2010-2014, 2014 dollars. Complaints: {period}.", 105))
    return nyt.save(fig, path)


# ------------------------------------------------------------------ run
def run(cfg):
    con = cfg.connect()
    src = clean_glob(cfg)
    demo = load_demographics()
    days = analysis_days(con, src)
    n_2020 = int((days.dt.year >= POP_2020_FROM_YEAR).sum())
    n_2010 = len(days) - n_2020
    py = person_years(demo, n_2010, n_2020)
    counts, ex, no_district = district_counts(con, src, days)
    d = district_table(counts, demo, py)
    boroughs = borough_table(d, no_district, demo)
    pumas = puma_units(d)
    facts = insight_facts(d, pumas, counts=counts)
    texts = insights(facts)
    unknown = unknown_by_agency(con, src)
    period = f"{days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}"
    pops = ", ".join(x for x in [f"2010 Census for {n_2010:,} days in 2010-{POP_2020_FROM_YEAR - 1}" if n_2010 else "",
                                 f"2020 Census for {n_2020:,} days from {POP_2020_FROM_YEAR}" if n_2020 else ""] if x)
    exi = ex.set_index("step")
    note = (f"Note: {int(d.n.sum()):,} complaints with a community district, {period} ({len(days):,} complete days); "
            f"population: {pops}. Not rated: parks and airports "
            f"{exi.loc['park or airport (special_area_cd)', 'pct']:.1f}% and no district "
            f"{exi.loc['no community district', 'pct']:.1f}% of rows; information requests excluded. Sources: NYC "
            "311; NYC DCP districts and Census counts; ACS 2010-2014 income.")
    city_rate = 1000 * d.n.sum() / d.person_years.sum()

    agg, figs = cfg.out_dir / "agg", cfg.out_dir / "figs"
    agg.mkdir(parents=True, exist_ok=True)
    d.sort_values("rate", ascending=False).to_csv(agg / "q07_districts.csv", index=False)
    boroughs.to_csv(agg / "q07_boroughs.csv", index=False)
    ex.to_csv(agg / "q07_exclusions.csv", index=False)
    pumas.to_csv(agg / "q07_pumas.csv", index=False)
    unknown.to_csv(agg / "q07_no_district_by_agency.csv", index=False)
    fig_map(d, texts, figs / "q07_district_map.png", note)
    fig_district_rates(d, city_rate, figs / "q07_district_rates.png", period)
    fig_income(pumas, facts, figs / "q07_income.png", period)

    v = dict(d=d, boroughs=boroughs, ex=ex, pumas=pumas, facts=facts, texts=texts, unknown=unknown, period=period,
             pops=pops, days=days, city_rate=city_rate)
    _report(cfg.out_dir, v)
    _write_answer(cfg.out_dir, v)
    return {"districts": d, "boroughs": boroughs, "exclusions": ex, "pumas": pumas, "facts": facts,
            "insights": texts, "no_district_by_agency": unknown}


def _report(out_dir, v):
    d, f = v["d"], v["facts"]
    rep = Report(STAGE, out_dir)
    rep.add(f"Q7. Complaints per 1,000 residents a year by community district, {v['period']} "
            f"({len(v['days']):,} complete days)", f"population: {v['pops']}",
            f"citywide rate (59 districts): {v['city_rate']:,.1f}")
    rep.table(v["ex"].round(2), title="Rows (default set -> mapped to a district)")
    rep.table(v["unknown"], max_rows=6, title="No community district, by agency")
    rep.table(v["boroughs"].drop(columns=["person_years"]).round(1), title="Boroughs")
    cols = ["boro_cd", "borough", "cd_name", "n", "rate", "housing_share_pct", "top_category", "top_category_pct",
            "median_hh_income"]
    ranked = d.sort_values("rate", ascending=False)[cols].round(1)
    rep.table(ranked.head(TOP_K), title=f"Top {TOP_K} districts by rate")
    rep.table(ranked.tail(5), title="Bottom 5 districts by rate")
    if d.low_n.any():
        rep.add(f"districts with fewer than {MIN_N} complaints (hatched): " + ", ".join(map(str, d.boro_cd[d.low_n])))
    rep.add("", "Three insights (in the map figure and answers/q07.md):")
    for i, t in enumerate(v["texts"], 1):
        rep.add(*textwrap.wrap(f"{i}. {t}", 110, subsequent_indent="   "))
    rep.add("", f"gap contributions (top {TOP_K} vs bottom {TOP_K}): " + "; ".join(
        f"{c} {x:.1f}%" for c, x in f["gap_cats"]))
    rep.emit()


def _pipe_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return rows


def _md(text):
    """Column and flag names (snake_case) as `code`, so markdown renderers don't read underscores as italics."""
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: f"`{m.group(0)}`", text)


def _write_answer(out_dir, v):
    d, f, b, exi = v["d"], v["facts"], v["boroughs"], v["ex"].set_index("step")
    ranked = d.sort_values("rate", ascending=False)
    bt = pd.DataFrame({
        "Borough": b.borough, "Districts": b.districts, "Complaints": [f"{x:,}" for x in b.n],
        "Per 1,000 a year": [f"{x:,.0f}" for x in b.rate], "Housing share": [f"{x:.0f}%" for x in b.housing_share_pct],
        "Borough known, no district": [f"{x:,}" for x in b.n_with_borough_no_district],
        "Highest district": b.highest_district})
    unk = v["unknown"]
    commuter = ranked.head(TOP_K)[ranked.head(TOP_K).housing_share_pct < 50]
    lines = [
        "# Q7. Complaints by community district",
        "",
        f"*Drafted by `nyc311/q07_geography.py` for {v['period']}. Every number is computed at run time; write the "
        "prose from these bullets.*",
        "",
        "## Data and method",
        "",
        f"- **Rate** = complaints per 1,000 residents a year: complaints divided by resident-years, using the "
        f"{v['pops']} (rule: 2010 Census for days before {POP_2020_FROM_YEAR}, 2020 Census from "
        f"{POP_2020_FROM_YEAR}). Only complete days count ({len(v['days']):,} days), so partial days don't dilute rates. "
        f"Citywide: {v['city_rate']:,.0f} per 1,000 a year.",
        f"- **Rows.** Of the default set's {int(exi.rows.iloc[0]):,} rows, {int(d.n.sum()):,} complaints "
        f"({exi.loc['mapped to a community district', 'pct']:.1f}%) map to one of the 59 districts. Left out: " +
        "; ".join(f"{_md(k)} {int(r.rows):,} ({r.pct:.1f}%)" for k, r in exi.iloc[1:-1].iterrows()) + ".",
        f"- **No district** is mostly " + ", ".join(f"{r.agency} {r.n:,}" for r in unk.head(3).itertuples()) +
        " (requests with no usable address or coordinates).",
        "",
        "![Complaints per 1,000 residents by community district, with three insights](figs/q07_district_map.png)",
        "",
        "## Three insights for the Mayor",
        "",
        *[f"{i}. {t}" for i, t in enumerate(v["texts"], 1)],
        "",
        "## Disparities by borough and district",
        "",
        *_pipe_table(bt),
        "",
        f"- **Highest rates:** " + "; ".join(
            f"{r.borough} {r.cd_number} ({r.cd_name}) {r.rate:,.0f}, top category {r.top_category} "
            f"{r.top_category_pct:.0f}%" for r in ranked.head(5).itertuples()) + ".",
        f"- **Lowest rates:** " + "; ".join(
            f"{r.borough} {r.cd_number} ({r.cd_name}) {r.rate:,.0f}, top category {r.top_category} "
            f"{r.top_category_pct:.0f}%" for r in ranked.tail(3).itertuples()) + ".",
        "",
        "![District rates split into housing and other complaints](figs/q07_district_rates.png)",
        "",
        f"- **Income** (55 PUMAs; 4 district pairs share one ACS value and are pooled): rank correlation of median "
        f"household income with the housing-complaint rate {f['rho_housing']:+.2f}, with all other complaints "
        f"{f['rho_other']:+.2f}.",
        "",
        "![Complaint rates against median household income](figs/q07_income.png)",
        "",
        "## Limits",
        "",
        "- **311 measures reporting, not need.** Rates mix how often problems occur with how often residents call. "
        "Housing complaints are tenants reporting landlords to HPD, so districts with more renters generate more "
        "of them; a low rate is not proof of good conditions.",
        "- **Denominators.** Census counts (2020 counts carry small differential-privacy noise); ACS income is from "
        "PUMAs that only approximate districts, in 2014 dollars. Boundaries are current, not historical.",
        *([f"- **Workers and visitors.** Per-resident rates overstate districts where many complaints come from people "
            f"who don't live there. Among the top {TOP_K}, housing is under half of complaints in " + "; ".join(
                f"{r.borough} {r.cd_number} ({r.cd_name}, {r.housing_share_pct:.0f}% housing, top category "
                f"{r.top_category})" for r in commuter.itertuples()) + "."] if len(commuter) else []),
        f"- **Coverage.** Districts with fewer than {MIN_N} complaints would be hatched on the map "
        f"({int(d.low_n.sum())} here). Correlations across areas say nothing about individual households.",
        "",
    ]
    path = out_dir / "answers" / "q07.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
