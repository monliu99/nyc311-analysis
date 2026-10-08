"""Q9: equity in city services, by community district.

Complaints are the same set as Q7 and Q8: rules.COMPLAINTS_SQL rows (default analysis set minus information requests
and "Invalid entries") opened on complete days (q07_geography.analysis_days), mapped to one of the 59 standard
community districts. The exclusion waterfall is Q7's, so the totals are identical. Four parts:
  1. Unresolved complaints: count, per 1,000 residents a year and share, each compared with the citywide share for
     the same year (age) and for the same complaint type and year (age and mix), by indirect standardization.
  2. Concentration: location quotients (district share of a category or type / citywide share) and a Hoover
     index per category (% of its complaints that would have to move to match the overall map of complaints).
  3. Origin: the city / people (private) / natural / other mix by district and by income fifth (Q3's taxonomy).
  4. Inequity: rank correlations (n, p, 95% CI) of median household income with the mix-adjusted time to close
     (Q4's method, imported from q04_response_times) and the mix-adjusted unresolved share, by district and by
     PUMA (districts sharing one income value are pooled), plus income fifths and straight-line fits on
     log2(income) without and with borough effects. The figure title comes from the evidence: verdict() grades the
     PUMA-level rank correlation; when none is significant but a citywide straight-line fit is (income_tilt), the
     title names that tilt and whether it holds within boroughs; otherwise it says there is no clear link.
Population: Q7's rule (2010 Census for days before 2015, 2020 Census from 2015), as resident-years, so rates are
per 1,000 residents a year. As in Q7, commuter districts are left out of per-resident rankings and of every
comparison with income (income_districts); their PUMA partner stands alone. Unresolved complaints are also shown by
year opened, with the year whose unresolved share stands out (peak_year). Every number in text is computed here; the
same code runs on one month or twelve years.
"""
from __future__ import annotations

import re
import textwrap

import numpy as np
import pandas as pd

from nyc311 import q04_response_times as q4
from nyc311 import q07_geography as q7
from nyc311 import rules
from nyc311.clean import clean_glob
from nyc311.report import Report
from nyc311.taxonomy import ORIGINS

STAGE = "q09_equity"
DAYS_PER_YEAR = q7.DAYS_PER_YEAR
MIN_N = 100                 # complaints a district needs to be ranked and correlated (fewer: low_n, reported)
MIN_N_CLOSED = 100          # closed complaints for the mix-adjusted time to close (as in Q4)
MIN_CELL = 30               # complaints of a category/type in a district before the district is named for it
LQ_HIGH = 2.0               # "concentrated": at least twice the citywide share
MAIN_CATEGORY_SHARE = 2.0   # % of complaints: categories at least this big are the "main" ones
TOP_TYPES = 15              # canonical types (by volume) checked for concentration
FINAL_DAYS = 90             # "final months": complaints opened in the last 90 complete days of the data
TOP_K = 5
TOP_BARS = 10               # districts in the unresolved panel
INCOME_GROUPS = 5           # income fifths of PUMAs
ALPHA = 0.05
Z95 = 1.96
WEAK, MODERATE = 0.3, 0.5   # a significant |rho| below WEAK is "weak", below MODERATE "moderate", else "strong"
TILT_SIZE = {"weak": "Small", "moderate": "Moderate", "strong": "Clear"}   # title word for a tilt, by |rho| at its unit
PEAK_RATIO = 2.0            # a year opened "stands out" when its unresolved share is at least this x the median year's
MIN_PEAK_YEARS = 3          # years needed before one can be called an outlier
TITLE_GAP_IN = 0.18         # inches kept free between a panel title and the next panel's title
SPLIT = "case when special_area_cd is null then boro_cd end"   # district, NULL for parks/airports (as Q7)
ORIGIN_LABEL = {"city": "city (infrastructure and services)", "private": "people (residents and businesses)",
                "natural": "natural (weather, trees, animals)", "other": "other or mixed"}
ORIGIN_SHORT = {"private": "People", "city": "City", "natural": "Natural", "other": "Other"}
ORIGIN_ORDER = ["private", "city", "natural", "other"]
METRICS = {
    "adj_pct": "time to close vs typical for the same type and year (%)",
    "excess_pts_mix": "unresolved share minus expected for type and year (points)",
    "pct_unresolved": "unresolved share, raw (%)",
    "unresolved_per_1k_yr": "unresolved per 1,000 residents a year",
    "rate_per_1k_yr": "complaints per 1,000 residents a year",
    "pct_city": "city-origin share (%)",
    "pct_private": "people-origin share (%)",
}
PER_RESIDENT = ("unresolved_per_1k_yr", "rate_per_1k_yr")
EQUITY_METRICS = ["adj_pct", "excess_pts_mix"]
COUNT_COLS = ["n", "n_unresolved", "expected_year", "expected_mix", "var_mix", "n_unresolved_final", "n_closed",
              "person_years", "n_res", "n_res_unresolved", "person_years_res",
              *[f"n_{o}" for o in ORIGINS], *[f"nu_{o}" for o in ORIGINS]]


# ---------- pure functions ----------

def load_demographics(path=q7.DEMO_CSV):
    return q7.load_demographics(path)


def location_quotients(counts):
    """counts: boro_cd, key, n. Full district x key grid: share of the district's complaints, citywide share, LQ."""
    g = counts.groupby(["boro_cd", "key"]).n.sum().unstack(fill_value=0)
    c = g.stack().rename("n").reset_index()
    c["n_district"] = c.boro_cd.map(g.sum(axis=1))
    tot = float(c.n.sum())
    c["share_pct"] = 100 * c.n / c.n_district
    c["city_share_pct"] = 100 * c.key.map(g.sum(axis=0)) / tot
    c["lq"] = c.share_pct / c.city_share_pct
    c["n"] = c.n.astype(int)
    return c


def concentration(lq, min_cell=MIN_CELL, min_share=0.0, lq_high=LQ_HIGH):
    """Per key: Hoover index (% of its complaints that would have to move district to match all complaints), the
    districts with LQ >= lq_high and at least min_cell complaints of it, and the top such district by LQ."""
    tot = lq.drop_duplicates("boro_cd").set_index("boro_cd").n_district
    rows = []
    for key, g in lq.groupby("key", sort=False):
        nk = g.n.sum()
        if nk == 0 or g.city_share_pct.iloc[0] < min_share:
            continue
        s_k = g.set_index("boro_cd").n / nk
        hoover = 50 * float((s_k - tot.reindex(s_k.index) / tot.sum()).abs().sum())
        ok = g[g.n >= min_cell].sort_values("lq", ascending=False)
        high = ok[ok.lq >= lq_high]
        top = ok.iloc[0] if len(ok) else None
        rows.append({"key": key, "n": int(nk), "city_share_pct": float(g.city_share_pct.iloc[0]),
                     "hoover_pct": hoover, "n_districts_high": len(high),
                     "pct_of_key_in_high": 100 * high.n.sum() / nk,
                     "pct_of_all_in_high": 100 * high.n_district.sum() / tot.sum(),
                     "top_boro_cd": int(top.boro_cd) if top is not None else np.nan,
                     "top_lq": float(top.lq) if top is not None else np.nan,
                     "top_n": int(top.n) if top is not None else np.nan,
                     "top_share_pct": float(top.share_pct) if top is not None else np.nan,
                     "high_districts": list(high.boro_cd.astype(int))})
    cols = ["key", "n", "city_share_pct", "hoover_pct", "n_districts_high", "pct_of_key_in_high",
            "pct_of_all_in_high", "top_boro_cd", "top_lq", "top_n", "top_share_pct", "high_districts"]
    return pd.DataFrame(rows, columns=cols).sort_values("hoover_pct", ascending=False).reset_index(drop=True)


def spearman(x, y):
    """Rank correlation with n, a two-sided p-value (t-test on ranks, as scipy's) and a 95% CI (Fisher z with the
    Fieller-Hartley-Pearson variance 1.06 / (n - 3)). NaN when fewer than 4 complete pairs or no variation."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    out = {"n": n, "rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n < 4:
        return out
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return out
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1 - 1e-12:
        return {**out, "rho": rho, "p": 0.0, "ci_low": rho, "ci_high": rho}
    import statsmodels.api as sm
    p = float(sm.OLS(ry, sm.add_constant(rx)).fit().pvalues[1])
    z, se = np.arctanh(rho), np.sqrt(1.06 / (n - 3)) if n > 3 else np.inf
    return {"n": n, "rho": rho, "p": p, "ci_low": float(np.tanh(z - Z95 * se)), "ci_high": float(np.tanh(z + Z95 * se))}


def _rates(t):
    """Derived rates from pooled counts (districts or PUMAs). Per-resident rates use the *_res columns when present
    (pooled units leave commuter districts out), else all complaints and resident-years."""
    t["pct_unresolved"] = 100 * t.n_unresolved / t.n
    t["excess_pts_mix"] = 100 * (t.n_unresolved - t.expected_mix) / t.n
    if "person_years_res" in t:
        py = t.person_years_res.where(t.person_years_res > 0)
        t["unresolved_per_1k_yr"] = 1000 * t.n_res_unresolved / py
        t["rate_per_1k_yr"] = 1000 * t.n_res / py
    elif "person_years" in t:
        t["unresolved_per_1k_yr"] = 1000 * t.n_unresolved / t.person_years
        t["rate_per_1k_yr"] = 1000 * t.n / t.person_years
    for o in ORIGINS:
        if f"n_{o}" in t:
            t[f"pct_{o}"] = 100 * t[f"n_{o}"] / t.n
    return t


def puma_units(d):
    """Pool districts that share a PUMA (and so one income value): counts add up; the mix-adjusted time to close is
    pooled on the log scale, weighted by closed complaints; per-resident rates leave commuter districts out."""
    d = d.copy()
    res = ~d.commuter.astype(bool) if "commuter" in d else pd.Series(True, index=d.index)
    if "person_years" in d:
        d["n_res"] = d.n.where(res, 0)
        d["n_res_unresolved"] = d.n_unresolved.where(res, 0)
        d["person_years_res"] = d.person_years.where(res, 0.0)
    d["_lr"] = np.log1p(d.adj_pct / 100) * d.n_closed
    d["_w"] = d.n_closed.where(d.adj_pct.notna(), 0)
    agg = {c: (c, "sum") for c in COUNT_COLS if c in d}
    if "borough" in d:
        agg["borough"] = ("borough", "first")
    p = d.groupby("puma_2010").agg(median_hh_income=("median_hh_income", "first"), _lr=("_lr", "sum"),
                                   _w=("_w", "sum"), n_districts=("boro_cd", "size"),
                                   districts=("boro_cd", lambda s: ", ".join(str(int(x)) for x in sorted(s))), **agg)
    p["adj_pct"] = 100 * np.expm1(p._lr / p._w.where(p._w > 0))
    return _rates(p.drop(columns=["_lr", "_w"]).reset_index())


def income_districts(d):
    """Districts compared with income (Q7's rule): commuter districts are left out entirely, so a PUMA they share
    is represented by its residential partner alone."""
    return d[~d.commuter.astype(bool)] if "commuter" in d else d


def income_fifths(p, k=INCOME_GROUPS):
    """PUMAs in k income groups (ties broken by order); pooled metrics and origin shares per group."""
    p = p.dropna(subset=["median_hh_income"]).sort_values("median_hh_income").reset_index(drop=True)
    k = min(k, len(p))
    p["fifth"] = pd.qcut(p.median_hh_income.rank(method="first"), k, labels=False) + 1
    rows = []
    for f, g in p.groupby("fifth"):
        w = g.n_closed.where(g.adj_pct.notna(), 0)
        lr = (np.log1p(g.adj_pct.fillna(0) / 100) * w).sum() / w.sum() if w.sum() else np.nan
        row = {"fifth": int(f), "income_min": g.median_hh_income.min(), "income_max": g.median_hh_income.max(),
               "pumas": len(g), "districts": int(g.n_districts.sum()), "n": int(g.n.sum()),
               "adj_pct": 100 * np.expm1(lr),
               "excess_pts_mix": 100 * (g.n_unresolved.sum() - g.expected_mix.sum()) / g.n.sum(),
               "pct_unresolved": 100 * g.n_unresolved.sum() / g.n.sum(),
               "unresolved_per_1k_yr": (1000 * g.n_res_unresolved.sum() / g.person_years_res.sum()
                                        if "person_years_res" in g and g.person_years_res.sum() > 0 else np.nan)}
        for o in ORIGINS:
            row[f"pct_{o}"] = 100 * g[f"n_{o}"].sum() / g.n.sum() if f"n_{o}" in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def income_slopes(t, metrics=EQUITY_METRICS, unit="district"):
    """OLS of each metric on log2(income), without and with borough fixed effects (HC1 SEs): change per doubling."""
    import statsmodels.api as sm
    rows = []
    for m in metrics:
        s = t.dropna(subset=[m, "median_hh_income"])
        if len(s) < 8:
            continue
        x = np.log2(s.median_hh_income.to_numpy(float))
        for label, X in [("all", pd.DataFrame({"log2_income": x}, index=s.index)),
                         ("within borough", pd.concat([pd.DataFrame({"log2_income": x}, index=s.index),
                                                       pd.get_dummies(s.borough, drop_first=True, dtype=float)],
                                                      axis=1))]:
            fit = sm.OLS(s[m].to_numpy(float), sm.add_constant(X, has_constant="add")).fit(cov_type="HC1")
            ci = fit.conf_int(ALPHA).loc["log2_income"]
            rows.append({"unit": unit, "metric": m, "model": label, "n": len(s),
                         "per_doubling": float(fit.params["log2_income"]), "ci_low": float(ci.iloc[0]),
                         "ci_high": float(ci.iloc[1]), "p": float(fit.pvalues["log2_income"])})
    return pd.DataFrame(rows, columns=["unit", "metric", "model", "n", "per_doubling", "ci_low", "ci_high", "p"])


def verdict(s, within=None):
    """Plain-language grade of a rank correlation with income. strength: 'no clear' unless p < ALPHA, then 'weak'
    (|rho| < WEAK), 'moderate' (< MODERATE) or 'strong'. fragile: the within-borough slope's CI (lo, hi) spans 0."""
    rho, p = s.get("rho", np.nan), s.get("p", np.nan)
    finite = bool(np.isfinite(rho) and np.isfinite(p))
    sign = int(np.sign(rho)) if finite else 0
    if not finite or p >= ALPHA:
        return {"sig": False, "strength": "no clear", "sign": sign, "fragile": False}
    a = abs(rho)
    strength = "weak" if a < WEAK else "moderate" if a < MODERATE else "strong"
    fragile = within is not None and np.isfinite(within[0]) and within[0] <= 0 <= within[1]
    return {"sig": True, "strength": strength, "sign": sign, "fragile": bool(fragile)}


def income_tilt(slopes, metric, corr=None, units=("PUMA", "district")):
    """The first citywide straight-line fit ('all', income areas before districts) of metric on log2(income) with
    p < ALPHA, with the same unit's within-borough fit and, from corr, its rank correlation; None if none is."""
    for unit in units:
        s = slopes[(slopes.unit == unit) & (slopes.metric == metric)].set_index("model")
        if not {"all", "within borough"} <= set(s.index) or not s.loc["all", "p"] < ALPHA:
            continue
        a, w = s.loc["all"], s.loc["within borough"]
        rho = np.nan
        if corr is not None:
            c = corr[(corr.unit == unit) & (corr.metric == metric)]
            rho = float(c.rho.iloc[0]) if len(c) else np.nan
        return {"unit": unit, "n": int(a.n), "per_doubling": float(a.per_doubling), "p": float(a.p),
                "ci_low": float(a.ci_low), "ci_high": float(a.ci_high), "within": float(w.per_doubling),
                "within_low": float(w.ci_low), "within_high": float(w.ci_high), "rho": rho}
    return None


def tilt_within(tilt):
    """What the within-borough fit does to a citywide tilt: 'holds' or 'reverses' (its CI excludes 0),
    'disappears' (CI spans 0 and the estimate is under half the citywide one) or 'is unclear'."""
    lo, hi, w, b = tilt["within_low"], tilt["within_high"], tilt["within"], tilt["per_doubling"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "is unclear"
    if not lo <= 0 <= hi:
        return "holds" if np.sign(w) == np.sign(b) else "reverses"
    return "disappears" if abs(w) < abs(b) / 2 else "is unclear"


def _unit_noun(unit):
    return "districts" if unit == "district" else "areas"


def _tilt_phrase(metric, tilt):
    b = tilt["per_doubling"]
    per = "halving" if b < 0 else "doubling"
    if metric == "adj_pct":
        return f"wait about {abs(b):.1f}% longer per {per} of income for the same complaints"
    return f"have about {abs(b):.2f} points more of the same complaints left open per {per} of income"


def _tilt_title(tilts):
    """'Small citywide tilt: lower-income districts wait about 2.5% longer per halving of income for the same
    complaints, a gap that disappears within boroughs' (size from the tilt's own rank correlation)."""
    rhos = [abs(tl["rho"]) for _, tl in tilts if np.isfinite(tl.get("rho", np.nan))]
    a = max(rhos) if rhos else np.nan
    size = (TILT_SIZE["weak"] if a < WEAK else TILT_SIZE["moderate"] if a < MODERATE else TILT_SIZE["strong"]
            ) if np.isfinite(a) else "A"
    groups = {}
    for m, tl in tilts:
        who = f"{'lower' if tl['per_doubling'] < 0 else 'higher'}-income {_unit_noun(tl['unit'])}"
        groups.setdefault(who, []).append(_tilt_phrase(m, tl))
    body = "; ".join(f"{who} {' and '.join(ps)}" for who, ps in groups.items())
    status = {tilt_within(tl) for _, tl in tilts}
    tail = (f", a gap that {status.pop()} within boroughs" if len(status) == 1 else
            " (not all of it holds within boroughs)")
    return f"{size} citywide tilt: {body}{tail}"


def equity_title(t, u, t_within=None, u_within=None, t_tilt=None, u_tilt=None):
    """Figure title from the evidence: t / u are rank correlations of income with the mix-adjusted time to close /
    share left open (higher = worse service) across income areas, t_within / u_within the within-borough slope CIs.
    When neither correlation is significant, t_tilt / u_tilt (income_tilt) name a citywide straight-line tilt."""
    parts = []
    for s, w, worse in [(t, t_within, "wait longer for the same complaints"),
                        (u, u_within, "have more of the same complaints left open")]:
        v = verdict(s, w)
        if v["sig"]:
            who = "lower-income" if v["sign"] < 0 else "higher-income"
            parts.append((v, f"{who} areas {worse}"))
    if not parts:
        tilts = [(m, tl) for m, tl in [("adj_pct", t_tilt), ("excess_pts_mix", u_tilt)] if tl]
        if tilts:
            return _tilt_title(tilts)
        return ("No clear link between area income and how long the same complaints take to close or how many "
                "stay open")
    rank = {"weak": 1, "moderate": 2, "strong": 3}
    parts.sort(key=lambda x: -rank[x[0]["strength"]])
    if len(parts) == 2 and parts[0][0]["strength"] != parts[1][0]["strength"]:
        text = "; ".join(f"{v['strength']} link: {p}" for v, p in parts)
        text = text[0].upper() + text[1:]
    else:
        text = f"{parts[0][0]['strength'].capitalize()} link with income: " + " and ".join(p for _, p in parts)
    fragile = [v["fragile"] for v, _ in parts]
    if all(fragile):
        text += ", a gap that fades within boroughs"
    elif any(fragile):
        text += " (part of it fades within boroughs)"
    return text


def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def _minus(s):
    return s.replace("-", "\N{MINUS SIGN}")


def _rho_short(s):
    if not np.isfinite(s.get("rho", np.nan)):
        return f"not estimable (n {s.get('n', 0)})"
    tag = "" if s["p"] < ALPHA else ", not significant"
    return _minus(f"\N{GREEK SMALL LETTER RHO} {s['rho']:+.2f}") + f" (p {_p(s['p'])}{tag})"


def _fit_short(metric, tl):
    if metric == "adj_pct":
        name, f = "time to close", (lambda v: f"{v:+.1f}%")
    else:
        name, f = "share left open", (lambda v: f"{v:+.2f} points")
    return _minus(f"{name} {f(tl['per_doubling'])} per doubling of income (p {_p(tl['p'])}); within boroughs "
                  f"{f(tl['within'])} (95% CI {tl['within_low']:+.1f} to {tl['within_high']:+.1f})")


def equity_subtitle(t, u, t_tilt=None, u_tilt=None):
    text = (f"Rank correlation with median household income across {t.get('n', 0)} income areas: time to close "
            f"{_rho_short(t)}; share left open {_rho_short(u)}. Both compare like with like (same complaint type "
            f"and year).")
    for m, tl in [("adj_pct", t_tilt), ("excess_pts_mix", u_tilt)]:
        if tl:
            text += f" Straight-line fit across {tl['n']} {_unit_noun(tl['unit'])}: {_fit_short(m, tl)}."
    return text


def peak_year(yr, min_years=MIN_PEAK_YEARS):
    """The year opened with the highest unresolved share: its count, share of all unresolved, rate vs the median
    year's, the largest category among its unresolved, and whether it stands out (>= PEAK_RATIO x the median).
    None with fewer than min_years years."""
    y = yr[(yr.n > 0) & (yr.year >= 0)]
    if len(y) < min_years:
        return None
    r = y.loc[y.pct_unresolved.idxmax()]
    med = float(y.pct_unresolved.median())
    tot = int(y.n_unresolved.sum())
    ratio = float(r.pct_unresolved) / med if med > 0 else np.inf
    cat = r.get("top_unresolved_category")
    return {"year": int(r.year), "n_unresolved": int(r.n_unresolved), "total_unresolved": tot,
            "share_pct": 100 * float(r.n_unresolved) / max(tot, 1), "pct": float(r.pct_unresolved),
            "median_pct": med, "ratio": ratio, "n_years": len(y), "stands_out": bool(ratio >= PEAK_RATIO),
            "category": cat if isinstance(cat, str) else None,
            "category_pct": float(r.get("top_category_pct", np.nan))}


def peak_year_sentence(pk):
    """Plain-language sentence on peak_year(): an outlier year with its count, share and rate vs the median year,
    and a hedge (the data record status, not why a record was never closed)."""
    if pk is None:
        return ""
    if not pk["stands_out"]:
        return (f"No year stands out: the highest unresolved share ({pk['year']}, {pk['pct']:.1f}%) is under "
                f"{PEAK_RATIO:g} times the median year's {pk['median_pct']:.1f}% (of {pk['n_years']} years).")
    cat = (f" The largest category among them is {pk['category']} ({pk['category_pct']:.0f}%)."
           if pk["category"] and np.isfinite(pk["category_pct"]) else "")
    return (f"**{pk['year']} stands out:** {pk['n_unresolved']:,} of the {pk['total_unresolved']:,} unresolved "
            f"complaints ({pk['share_pct']:.0f}%) were opened that year, {pk['pct']:.1f}% of its complaints against "
            f"{pk['median_pct']:.1f}% in the median year (of {pk['n_years']}), {pk['ratio']:.1f} times as high.{cat} "
            "That more likely reflects records left open and never closed than a year of worse service, but the data "
            "cannot show why. Comparing each district with the citywide share for the same year removes that year's "
            "overall level, though not where its open records sit.")


# ---------- data ----------

def _load(con, src, days_rel):
    """Temp table q9all: complaints on the analysis days (district may be NULL; they set the citywide benchmarks),
    view q9 (one of the 59 districts) and view q9r (closed with a duration, instant closures left out)."""
    con.execute(f"""
        create or replace temp table q9all as
        select created, coalesce(year(created), -1) as yr, boro_cd, special_area_cd, canonical_type,
               coalesce(canonical_type, '(none)') as ct, coalesce(category, '(none)') as category,
               coalesce(origin, 'other') as origin, coalesce(unresolved, false) as unresolved, duration_h,
               closed_at_creation, date_only_created
        from {src} where {rules.COMPLAINTS_SQL} and created::date in (select d from {days_rel})""")
    con.execute("create or replace temp view q9 as select * from q9all "
                "where boro_cd is not null and special_area_cd is null")
    con.execute(f"create or replace temp view q9r as select * from q9all where duration_h is not null "
                f"and not ({rules.INSTANT_SQL})")
    return int(con.sql("select count(*) from q9all").fetchone()[0])


def unresolved_by(con, rel, group_col, d_final):
    """Observed vs expected unresolved per group (NULL group kept). expected_year = sum over rows of the citywide
    unresolved share for the row's year; expected_mix = the same for its canonical type and year."""
    return con.sql(f"""
        with b as (select {group_col} as grp, ct, yr, unresolved::int as u,
                          created::date >= '{d_final:%Y-%m-%d}' as late from {rel}),
        cy as (select yr, avg(u) p_y from b group by 1),
        cty as (select ct, yr, avg(u) p_ty from b group by all)
        select grp, count(*)::bigint as n, sum(u)::bigint as n_unresolved, sum(p_y) as expected_year,
               sum(p_ty) as expected_mix, sum(p_ty * (1 - p_ty)) as var_mix,
               count_if(u = 1 and late)::bigint as n_unresolved_final
        from b join cy using (yr) join cty using (ct, yr) group by grp""").df()


def by_year(con):
    """Complaints and unresolved by year opened (59 districts), with the largest category among each year's
    unresolved (ties broken by name)."""
    return con.sql("""
        with y as (select yr as year, count(*)::bigint n, count_if(unresolved)::bigint n_unresolved,
                          100 * avg(unresolved::int) pct_unresolved from q9 group by 1),
        c as (select yr as year, category, count(*)::bigint k,
                     row_number() over (partition by yr order by count(*) desc, category) as rk
              from q9 where unresolved group by yr, category)
        select y.year, y.n, y.n_unresolved, y.pct_unresolved, c.category as top_unresolved_category,
               100.0 * c.k / y.n_unresolved as top_category_pct
        from y left join c on c.year = y.year and c.rk = 1 order by y.year""").df()


def _wide(df, value, prefix):
    w = df.pivot_table(index="boro_cd", columns="origin", values=value, aggfunc="sum", fill_value=0)
    return w.reindex(columns=list(ORIGINS), fill_value=0).add_prefix(prefix).astype(int)


def district_table(con, demo, py, years, d_final, names):
    u = unresolved_by(con, "q9all", SPLIT, d_final)
    u_all = u.copy()
    u = u.dropna(subset=["grp"]).rename(columns={"grp": "boro_cd"})
    u["boro_cd"] = u.boro_cd.astype(int)
    o = con.sql("select boro_cd, origin, count(*)::bigint n, count_if(unresolved)::bigint nu from q9 group by all").df()
    o["boro_cd"] = o.boro_cd.astype(int)
    o = pd.concat([_wide(o, "n", "n_"), _wide(o, "nu", "nu_")], axis=1).reset_index()
    t = q4.mix_adjusted(con, "q9r", SPLIT, min_n=MIN_N_CLOSED).rename(
        columns={"grp": "boro_cd", "n": "n_closed"})[["boro_cd", "n_closed", "median_h", "adj_pct"]]
    t["boro_cd"] = t.boro_cd.astype(int)
    closed = con.sql(f"select {SPLIT} as boro_cd, count(*)::bigint n_closed_all from q9r where {SPLIT} is not null "
                     "group by 1").df().astype(int)
    keep = ["boro_cd", "borough", "cd_number", "cd_name", "puma_2010", "puma_shared_with", "median_hh_income"]
    pop = pd.DataFrame({"boro_cd": py.index.astype(int), "person_years": py.to_numpy(float)})
    d = (demo[keep].merge(pop, on="boro_cd").merge(u, on="boro_cd").merge(o, on="boro_cd", how="left")
         .merge(closed, on="boro_cd", how="left").merge(t.drop(columns="n_closed"), on="boro_cd", how="left"))
    d["n_closed"] = d.pop("n_closed_all").fillna(0).astype(int)
    d["residents"] = d.person_years / years
    d["commuter"] = d.boro_cd.isin(q7.COMMUTER_CDS)
    d = _rates(d)
    d["expected_pct_year"] = 100 * d.expected_year / d.n
    d["excess_pts_year"] = d.pct_unresolved - d.expected_pct_year
    d["expected_pct_mix"] = 100 * d.expected_mix / d.n
    d["z_mix"] = (d.n_unresolved - d.expected_mix) / np.sqrt(d.var_mix.where(d.var_mix > 0))
    d["dominant_origin"] = d[[f"pct_{o}" for o in ORIGINS]].idxmax(axis=1).str.replace("pct_", "", regex=False)
    d["low_n"] = d.n < MIN_N
    d["label"] = [q4.district_label(c, names) for c in d.boro_cd]
    d["short"] = [_short(r) for r in d.itertuples()]
    d["code"] = [_code(c) for c in d.boro_cd]
    return d.sort_values("n_unresolved", ascending=False).reset_index(drop=True), u_all


def _code(cd):
    return f"{q4.BORO_ABBR.get(int(cd) // 100, '?')} {int(cd) % 100}"


def _short(r):
    first = re.split(r",| and ", str(r.cd_name))[0].strip()
    return f"{first} ({_code(r.boro_cd)})"


def _dominant_origin(con, col):
    o = con.sql(f"select {col} as key, origin, count(*)::bigint n from q9 group by all").df()
    return o.sort_values("n", ascending=False).drop_duplicates("key").set_index("key").origin


def lq_tables(con, d_ok):
    cats = con.sql("select boro_cd, category as key, count(*)::bigint n from q9 group by all").df()
    cats = cats[cats.boro_cd.isin(d_ok)]
    lq_c = location_quotients(cats)
    types = con.sql("select boro_cd, ct as key, count(*)::bigint n from q9 group by all").df()
    types = types[types.boro_cd.isin(d_ok)]
    top = types.groupby("key").n.sum().sort_values(ascending=False).head(TOP_TYPES).index
    lq_all_types = location_quotients(types)
    lq_t = lq_all_types[lq_all_types.key.isin(top)].reset_index(drop=True)
    conc_c = concentration(lq_c, min_share=MAIN_CATEGORY_SHARE)
    conc_c["origin"] = conc_c.key.map(_dominant_origin(con, "category"))
    conc_t = concentration(lq_t)
    conc_t["origin"] = conc_t.key.map(_dominant_origin(con, "ct"))
    return lq_c, lq_t, conc_c, conc_t


def origin_city(con):
    o = con.sql("""select origin, count(*)::bigint n, count_if(unresolved)::bigint n_unresolved
                   from q9 group by 1""").df()
    o = o.set_index("origin").reindex(list(ORIGINS), fill_value=0).reset_index()
    o["pct"] = 100 * o.n / o.n.sum()
    o["pct_unresolved"] = 100 * o.n_unresolved / o.n.where(o.n > 0)
    o["pct_of_unresolved"] = 100 * o.n_unresolved / max(int(o.n_unresolved.sum()), 1)
    top = con.sql("""select origin, category, count(*)::bigint n from q9 group by all""").df()
    o["main_categories"] = [", ".join(f"{r.category} {100 * r.n / max(int(o.n[o.origin == k].iloc[0]), 1):.0f}%"
                                      for r in top[top.origin == k].nlargest(3, "n").itertuples())
                            for k in o.origin]
    return o


def correlations(d, p):
    """Spearman with income for every metric, by district (d) and by PUMA (p). The caller passes income_districts;
    per-resident metrics also drop any commuter district left in d."""
    rows = []
    for unit, t in [("district", d), ("PUMA", p)]:
        for m in METRICS:
            if m in t:
                y = t[m].where(~t.commuter) if (m in PER_RESIDENT and "commuter" in t) else t[m]
                rows.append({"unit": unit, "metric": m, "label": METRICS[m], **spearman(t.median_hh_income, y)})
    return pd.DataFrame(rows)


# ---------- chart ----------

def _colors():
    from nyc311 import nytplot as nyt
    return {"private": nyt.CORAL.line, "city": nyt.HUES[2].line, "natural": nyt.SAGE.line, "other": nyt.HUES[5].band}


def _hbar(ax):
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)


def _num(v):
    """Adaptive decimals for rates: 12, 3.4, 0.27."""
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:,.1f}" if abs(v) >= 1 else f"{v:,.2f}"


def _money(v):
    return f"\\${v / 1000:,.0f}k"


def fifth_names(k):
    if k == 1:
        return ["All areas"]
    mid = {2: "2nd", 3: "3rd", 4: "4th"}
    names = ["Poorest"] + [("Middle" if (k % 2 and i == k // 2) else mid.get(i + 1, f"{i + 1}th"))
                           for i in range(1, k - 1)] + ["Richest"]
    unit = "fifth" if k == 5 else "group"
    return [f"{n} {unit}" for n in names]


def _panel_unresolved(ax, d, nyt, col):
    s = d[~d.low_n].nlargest(TOP_BARS, "n_unresolved").iloc[::-1]
    y = np.arange(len(s))
    left = np.zeros(len(s))
    for o in ORIGIN_ORDER:
        v = s[f"nu_{o}"].to_numpy(float)
        ax.barh(y, v, left=left, color=col[o], height=0.72)
        left = left + v
    xmax = max(float(left.max()), 1.0)
    for yi, r, tot in zip(y, s.itertuples(), left):
        rate = "commuter district" if r.commuter else _num(r.unresolved_per_1k_yr)
        ax.text(tot + 0.015 * xmax, yi, f"{int(r.n_unresolved):,} \N{MIDDLE DOT} {rate}", va="center", fontsize=9.5,
                fontfamily=nyt.SANS, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(list(s.short), fontsize=9.5)
    ax.set_xlim(0, xmax * 1.42)
    ax.set_ylim(-0.6, len(s) - 0.4)
    _hbar(ax)
    nyt.format_x(ax, nbins=4)
    ax.set_xlabel(f"Unresolved complaints \N{MIDDLE DOT} per 1,000 residents a year", fontsize=9.5,
                  color=nyt.NOTE_GRAY, loc="left")
    if len(s):
        top = s.iloc[-1]
        return nyt.panel_title(ax, f"A. Most unresolved complaints: {top.short}, {int(top.n_unresolved):,}",
                               fontsize=11)
    return None


def _panel_concentration(ax, conc, nyt, col):
    c = conc.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(c))
    ax.barh(y, c.hoover_pct, color=[col.get(o, col["other"]) for o in c.origin], height=0.72)
    xmax = max(float(c.hoover_pct.max()) if len(c) else 1.0, 1.0)
    for yi, r in zip(y, c.itertuples()):
        where = (f" \N{MIDDLE DOT} {_code(r.top_boro_cd)} {r.top_lq:.1f}\N{MULTIPLICATION SIGN}"
                 if np.isfinite(r.top_boro_cd) else "")
        ax.text(r.hoover_pct + 0.015 * xmax, yi, f"{r.hoover_pct:.0f}%{where}", va="center", fontsize=9.5,
                fontfamily=nyt.SANS, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(list(c.key), fontsize=9.5)
    ax.set_xlim(0, xmax * 1.5)
    ax.set_ylim(-0.6, len(c) - 0.4)
    _hbar(ax)
    nyt.format_x(ax, suffix="%", nbins=3)
    ax.set_xlabel(textwrap.fill("Hoover index: share of the category that would have to move to match all "
                                "complaints \N{MIDDLE DOT} district where it is most over-represented, "
                                "\N{MULTIPLICATION SIGN} its citywide share", 54), fontsize=9.5,
                  color=nyt.NOTE_GRAY, loc="left")
    if len(c):
        top = c.iloc[-1]
        return nyt.panel_title(ax, f"B. Most concentrated: {top.key}, {top.hoover_pct:.0f}% would have to move",
                               fontsize=11)
    return None


def _panel_origin(ax, f, nyt, col, labels):
    y = np.arange(len(f))[::-1]
    left = np.zeros(len(f))
    for o in ORIGIN_ORDER:
        v = f[f"pct_{o}"].fillna(0).to_numpy(float)
        ax.barh(y, v, left=left, color=col[o], height=0.68)
        for yi, vi, li in zip(y, v, left):
            if vi >= 7:
                ax.text(li + vi / 2, yi, f"{vi:.0f}%", ha="center", va="center", fontsize=9.5, fontweight="bold",
                        fontfamily=nyt.SANS, color="#333333" if o == "other" else "white")
        left = left + v
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(f) - 0.4)
    _hbar(ax)
    nyt.format_x(ax, suffix="%", step=25)
    lo, hi = f.pct_city.iloc[0], f.pct_city.iloc[-1]
    first, last = (s.split("\n")[0].lower() for s in (labels[0], labels[-1]))
    return nyt.panel_title(ax, f"C. City-origin share: {lo:.0f}% in the {first}, {hi:.0f}% in the {last}",
                           fontsize=11)


def _panel_measure(ax, f, m, title, fmt, suffix, xlabel, nyt):
    y = np.arange(len(f))[::-1]
    v = f[m].to_numpy(float)
    ok = np.isfinite(v)
    color = nyt.HUES[5].line
    ax.barh(y[ok], v[ok], color=color, height=0.55)
    ax.axvline(0, color="#333333", lw=1.0, zorder=3)
    vv = v[ok] if ok.any() else np.array([0.0])
    lo, hi = min(0.0, float(vv.min())), max(0.0, float(vv.max()))
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - (0.55 * span if lo < 0 else 0), hi + (0.55 * span if hi > 0 else 0.05 * span))
    for yi, vi in zip(y, v):
        if not np.isfinite(vi):
            ax.text(0, yi, " n/a", va="center", fontsize=9, color=nyt.NOTE_GRAY)
            continue
        txt = _minus(fmt(vi))
        ax.text(vi + (0.03 if vi >= 0 else -0.03) * span, yi, txt, va="center", ha="left" if vi >= 0 else "right",
                fontsize=9.5, fontfamily=nyt.SANS, color="#333333")
    ax.set_ylim(-0.6, len(f) - 0.4)
    ax.tick_params(labelleft=False)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False)
    nyt.format_x(ax, suffix=suffix, nbins=3)
    ax.set_xlabel(xlabel, fontsize=9, color=nyt.NOTE_GRAY, loc="left")
    return nyt.panel_title(ax, title, fontsize=10.5)


def fit_width(t, max_px, renderer):
    """Keep Text t as it is if every line fits in max_px pixels; otherwise re-wrap its words greedily so that each
    line does (a single word longer than max_px stays on its own line)."""
    prop = t.get_fontproperties()

    def width(s):
        return renderer.get_text_width_height_descent(s, prop, ismath=False)[0]

    text = t.get_text()
    if all(width(line) <= max_px for line in text.split("\n")):
        return t
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if cur and width(cand) > max_px:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    t.set_text("\n".join(lines + [cur]))
    return t


def chart(res, path):
    from nyc311 import nytplot as nyt
    fig, _ = figure(res)
    return nyt.save(fig, path)


def figure(res):
    """The Q9 figure and its text objects by name (title, subtitle, note, origin_key, panel_a ... panel_d3). Every
    title, the subtitle and the note are wrapped to the space they have once the layout is known, so none runs into
    a neighbor or past the figure's edge, whatever the computed text."""
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    from nyc311 import nytplot as nyt

    nyt.use()
    f, facts, d = res["fifths"], res["facts"], res["districts"]
    col = _colors()
    fig = Figure(figsize=(11, 12.2))
    FigureCanvasAgg(fig)
    rend = fig.canvas.get_renderer()
    pad = mpl.rcParams["figure.constrained_layout.w_pad"] * fig.dpi
    full = fig.bbox.width - 2 * pad
    outer = fig.add_gridspec(3, 1, height_ratios=[0.5, 5.0, 3.5])
    key = fig.add_subplot(outer[0])
    top = outer[1].subgridspec(1, 2, width_ratios=[1.05, 1])
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    bot = outer[2].subgridspec(1, 4, width_ratios=[2.25, 1, 1, 1])
    ax_c = fig.add_subplot(bot[0])
    ax_d = [fig.add_subplot(bot[i]) for i in (1, 2, 3)]

    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    sub = fit_width(key.text(0, 0.98, facts["subtitle"], va="top", fontsize=10, color="#333333",
                             fontfamily=nyt.SANS), full, rend)
    outer.set_height_ratios([0.25 + 0.14 * (sub.get_text().count("\n") + 1), 5.0, 3.5])   # room for the key row
    oc = res["origin_city"].set_index("origin")
    parts = [TextArea(f"Complaint origin (colors in A\N{EN DASH}C), share of complaints across the {len(d)} "
                      "districts:", textprops={"fontsize": 10, "color": nyt.NOTE_GRAY, "fontfamily": nyt.SANS})]
    for o in ORIGIN_ORDER:
        c = "#8c8c8c" if o == "other" else col[o]
        parts.append(TextArea(f"{ORIGIN_SHORT[o]} {oc.loc[o, 'pct']:.0f}%", textprops={
            "fontsize": 10.5, "fontweight": "bold", "color": c, "fontfamily": nyt.SANS}))
    origin_key = AnchoredOffsetbox(loc="lower left", child=HPacker(children=parts, align="baseline", pad=0, sep=12),
                                   pad=0, borderpad=0, frameon=False, bbox_to_anchor=(0, 0.0),
                                   bbox_transform=key.transAxes)
    key.add_artist(origin_key)

    texts = {"subtitle": sub, "origin_key": origin_key,
             "panel_a": _panel_unresolved(ax_a, d, nyt, col),
             "panel_b": _panel_concentration(ax_b, res["concentration_categories"], nyt, col)}
    if len(f):
        labels = [f"{n}\n{_money(r.income_min)}\N{EN DASH}{_money(r.income_max)}"
                  for n, r in zip(fifth_names(len(f)), f.itertuples())]
        texts["panel_c"] = _panel_origin(ax_c, f, nyt, col, labels)
        t, u = facts["t"], facts["u"]
        r_unres = _corr(res, "PUMA", "unresolved_per_1k_yr")
        for i, (ax, m, title, fmt, suffix, xl) in enumerate([
                (ax_d[0], "adj_pct", "D. Time to close\nvs typical", lambda v: f"{v:+.1f}%", "%", _rho_short(t)),
                (ax_d[1], "excess_pts_mix", "Left open vs\nexpected (points)", lambda v: f"{v:+.2f}", "",
                 _rho_short(u)),
                (ax_d[2], "unresolved_per_1k_yr", "Unresolved per\n1,000 a year", _num, "", _rho_short(r_unres))],
                start=1):
            texts[f"panel_d{i}"] = _panel_measure(ax, f, m, title, fmt, suffix, textwrap.fill(xl, 24), nyt)
    else:
        for ax in [ax_c, *ax_d]:
            ax.axis("off")
        ax_c.text(0, 0.5, "Too few districts with enough complaints to compare income groups", fontsize=10,
                  color=nyt.NOTE_GRAY, transform=ax_c.transAxes)
    texts["title"] = fit_width(nyt.title(fig, facts["title"]), full, rend)
    texts["note"] = fit_width(nyt.note(fig, facts["note"]), full, rend)

    # panel titles and x-axis labels (both start at their axes' left edge): once constrained layout has placed the
    # axes, each may run up to the next panel in its row or to the figure's right margin. Axis labels need it too:
    # the tight bounding box leaves their width out, so a label past the edge would be cropped, not shown.
    fig.canvas.draw()
    gap = TITLE_GAP_IN * fig.dpi
    right = fig.bbox.width - pad
    for row in [[("a", ax_a), ("b", ax_b)], [("c", ax_c), *[(f"d{i}", ax) for i, ax in enumerate(ax_d, start=1)]]]:
        for (name, ax), nxt in zip(row, [*row[1:], None]):
            limit = (nxt[1].bbox.x0 - gap if nxt else right) - ax.bbox.x0
            if texts.get(f"panel_{name}") is not None:
                fit_width(texts[f"panel_{name}"], limit, rend)
            if ax.get_xlabel():
                texts[f"xlabel_{name}"] = fit_width(ax.xaxis.label, limit, rend)
    return fig, {k: v for k, v in texts.items() if v is not None}


# ---------- run ----------

def run(cfg):
    from nyc311.q05_top_types import period_label, period_long

    con = cfg.connect()
    src = clean_glob(cfg)
    demo = load_demographics()
    names = dict(zip(demo.boro_cd.astype(int), demo.cd_name))
    days = q7.analysis_days(con, src)
    days_rel = q7.days_table(con, days, "q09_days")
    n_2020 = int((days.dt.year >= q7.POP_2020_FROM_YEAR).sum())
    n_2010 = len(days) - n_2020
    py = q7.person_years(demo, n_2010, n_2020)
    _, ex, _ = q7.district_counts(con, src)          # Q7's waterfall: the same complaints, row for row
    n_benchmark = _load(con, src, days_rel)
    d0, d1 = days.min(), days.max()
    d_final = d1 - pd.Timedelta(days=FINAL_DAYS - 1)
    d, u_all = district_table(con, demo, py, len(days) / DAYS_PER_YEAR, d_final, names)
    ok = d[~d.low_n]
    inc = income_districts(ok)                       # every comparison with income leaves commuter districts out
    lq_c, lq_t, conc_c, conc_t = lq_tables(con, set(ok.boro_cd))
    p = puma_units(inc)
    yr = by_year(con)
    res = {
        "exclusions": ex, "n_benchmark": n_benchmark, "districts": d, "unresolved_all_groups": u_all,
        "by_year": yr, "peak_year": peak_year(yr), "lq_categories": lq_c, "lq_types": lq_t,
        "concentration_categories": conc_c,
        "concentration_types": conc_t, "origin": d[["boro_cd", "label", *[f"pct_{o}" for o in ORIGINS],
                                                     "dominant_origin"]],
        "origin_city": origin_city(con), "pumas": p, "correlations": correlations(inc, p),
        "fifths": income_fifths(p), "n_income_districts": len(inc),
        "slopes": pd.concat([income_slopes(inc, unit="district"), income_slopes(p, unit="PUMA")], ignore_index=True),
        "period": period_label(d0.date(), d1.date()), "period_long": period_long(d0.date(), d1.date()),
        "days": days, "n_days": len(days), "n_2010": n_2010, "n_2020": n_2020, "d_final": d_final,
        "n_shared_pairs": int((demo.puma_shared_with.notna() & (demo.boro_cd < pd.to_numeric(
            demo.puma_shared_with, errors="coerce"))).sum()),
    }
    res["facts"] = _facts(res)

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["short"]).to_csv(agg / "q09_districts.csv", index=False)
    lq_c.to_csv(agg / "q09_lq_categories.csv", index=False)
    lq_t.to_csv(agg / "q09_lq_types.csv", index=False)
    pd.concat([conc_c.assign(level="category"), conc_t.assign(level="canonical type")]).to_csv(
        agg / "q09_concentration.csv", index=False)
    res["origin_city"].to_csv(agg / "q09_origin.csv", index=False)
    res["correlations"].to_csv(agg / "q09_correlations.csv", index=False)
    p.to_csv(agg / "q09_pumas.csv", index=False)
    res["fifths"].to_csv(agg / "q09_income_fifths.csv", index=False)
    res["slopes"].to_csv(agg / "q09_income_slopes.csv", index=False)
    ex.to_csv(agg / "q09_exclusions.csv", index=False)

    figs = cfg.out_dir / "figs"
    for old in figs.glob("q09_*.png"):
        if old.name != "q09_equity.png":
            old.unlink()
    chart(res, figs / "q09_equity.png")
    _report(cfg, res)
    _write_answer(cfg, res)
    return res


# ---------- text ----------

def _pts(x):
    return _minus(f"{x:+.1f}") + " points"


def _rho(s):
    if not np.isfinite(s["rho"]):
        return f"not estimable (n {s['n']})"
    return _minus(f"\N{GREEK SMALL LETTER RHO} {s['rho']:+.2f} (95% CI {s['ci_low']:+.2f} to {s['ci_high']:+.2f}, "
                  f"n {s['n']}, p {_p(s['p'])})")


def _signed(v):
    return _minus(f"{v:+.2f}") if np.isfinite(v) else "n/a"


def _corr(res, unit, metric):
    c = res["correlations"]
    r = c[(c.unit == unit) & (c.metric == metric)]
    return r.iloc[0].to_dict() if len(r) else {"n": 0, "rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}


def _within(res, metric, unit="PUMA"):
    s = res["slopes"]
    r = s[(s.unit == unit) & (s.metric == metric) & (s.model == "within borough")]
    return (float(r.ci_low.iloc[0]), float(r.ci_high.iloc[0])) if len(r) else None


def _facts(res):
    d = res["districts"]
    t, u = _corr(res, "PUMA", "adj_pct"), _corr(res, "PUMA", "excess_pts_mix")
    t_tilt = income_tilt(res["slopes"], "adj_pct", res["correlations"])
    u_tilt = income_tilt(res["slopes"], "excess_pts_mix", res["correlations"])
    title = equity_title(t, u, _within(res, "adj_pct"), _within(res, "excess_pts_mix"), t_tilt=t_tilt, u_tilt=u_tilt)
    n = int(d.n.sum())
    comm = ", ".join(_code(c) for c in d.boro_cd[d.commuter])
    note = (f"Note: {n:,} complaints on {res['n_days']:,} complete days, {res['period']}, {len(d)} community "
            f"districts. Income: ACS 2010{chr(8211)}2014 median household income of {len(res['pumas'])} PUMAs, 2014 "
            f"dollars. D compares with the citywide figure for the same complaint type and year. Residents: 2010 "
            f"Census before {q7.POP_2020_FROM_YEAR}, 2020 Census from {q7.POP_2020_FROM_YEAR}"
            + (f". {comm} ({q7.COMMUTER_LABEL}) is left out of C, D and every comparison with income." if comm
               else "."))
    return {"title": title, "subtitle": equity_subtitle(t, u, t_tilt, u_tilt), "note": note, "t": t, "u": u,
            "t_tilt": t_tilt, "u_tilt": u_tilt,
            "verdict_t": verdict(t, _within(res, "adj_pct")), "verdict_u": verdict(u, _within(res, "excess_pts_mix"))}


def _peak_report(pk):
    if pk is None:
        return f"not assessed (fewer than {MIN_PEAK_YEARS} years)"
    return (f"{pk['year']}, {pk['n_unresolved']:,} unresolved ({pk['share_pct']:.1f}% of all), {pk['pct']:.2f}% of "
            f"its complaints vs {pk['median_pct']:.2f}% in the median year ({pk['ratio']:.1f}x; "
            f"{'stands out' if pk['stands_out'] else 'does not stand out'})")


def _report(cfg, res):
    rep = Report(STAGE, cfg.out_dir)
    d, ex, f = res["districts"], res["exclusions"], res["facts"]
    ok = d[~d.low_n]
    rep.add(f"Q9. Equity in city services by community district, {res['period_long']} ({res['n_days']:,} complete "
            "days)",
            f"population (Q7 rule): 2010 Census for {res['n_2010']:,} days before {q7.POP_2020_FROM_YEAR}, 2020 Census "
            f"for {res['n_2020']:,} days from {q7.POP_2020_FROM_YEAR}; per-resident rates are per 1,000 a year",
            f"districts ranked (>= {MIN_N} complaints): {len(ok)} of {len(d)}; {q7.COMMUTER_LABEL}s (left out of "
            f"per-resident numbers and every comparison with income, {res['n_income_districts']} districts left): "
            f"{', '.join(d.short[d.commuter]) or 'none'}",
            f"unresolved opened in the final {FINAL_DAYS} complete days (from {res['d_final']:%Y-%m-%d}): "
            f"{int(d.n_unresolved_final.sum()):,} of {int(d.n_unresolved.sum()):,}",
            "peak year opened: " + _peak_report(res["peak_year"]),
            f"title: {f['title']}", f"subtitle: {f['subtitle']}",
            "verdicts (PUMA rank correlation): " + "; ".join(
                f"{name} {v['strength']}" + (", fades within boroughs" if v["fragile"] else "")
                for name, v in [("time to close", f["verdict_t"]), ("left open", f["verdict_u"])]),
            f"citywide tilt (first straight-line fit with p < {ALPHA}, income areas then districts): " + "; ".join(
                f"{name} across {tl['n']} {_unit_noun(tl['unit'])} {tl['per_doubling']:+.2f} per doubling "
                f"(p {tl['p']:.3f}), within boroughs {tl['within']:+.2f} ({tl['within_low']:+.2f} to "
                f"{tl['within_high']:+.2f}): {tilt_within(tl)}" if tl else f"{name} none"
                for name, tl in [("time to close", f["t_tilt"]), ("left open", f["u_tilt"])]))
    rep.table(ex.round(2), title="Rows (default set -> analyzed; identical to Q7)")
    rep.table(res["by_year"].round(2), max_rows=14, title=f"Unresolved by year opened ({len(d)} districts)")
    cols = ["boro_cd", "short", "n", "n_unresolved", "pct_unresolved", "unresolved_per_1k_yr", "expected_pct_year",
            "expected_pct_mix", "excess_pts_mix", "z_mix", "adj_pct", "median_hh_income"]
    rep.table(ok.sort_values("n_unresolved", ascending=False)[cols].round(2), max_rows=10,
              title="Most unresolved complaints (expected = citywide share for same year / same type and year)")
    rep.table(ok.sort_values("excess_pts_mix", ascending=False)[cols].round(2), max_rows=TOP_K,
              title="Most still open beyond their complaint mix (excess_pts_mix)")
    cc = res["concentration_categories"].copy()
    cc["top"] = [_code(b) if np.isfinite(b) else "" for b in cc.top_boro_cd]
    rep.table(cc[["key", "origin", "n", "city_share_pct", "hoover_pct", "n_districts_high", "pct_of_key_in_high",
                  "top", "top_lq", "top_n"]].round(2), title=f"Main categories (>= {MAIN_CATEGORY_SHARE}% of "
              f"complaints): Hoover index; districts with LQ >= {LQ_HIGH} and >= {MIN_CELL} complaints")
    ct = res["concentration_types"].copy()
    ct["top"] = [_code(b) if np.isfinite(b) else "" for b in ct.top_boro_cd]
    rep.table(ct[["key", "n", "hoover_pct", "n_districts_high", "top", "top_lq", "top_n"]].round(2), max_rows=TOP_TYPES,
              title=f"Top {TOP_TYPES} canonical types: concentration")
    rep.table(res["origin_city"].round(2), title=f"Origin, across the {len(d)} districts")
    oc = [f"pct_{o}" for o in ORIGINS]
    rep.table(ok[oc].describe().loc[["min", "50%", "max"]].round(1).reset_index(), title="Origin shares across districts")
    rep.table(res["correlations"].drop(columns="label").round(3), title="Spearman with median household income")
    rep.table(res["fifths"].round(2), title="Income fifths of PUMAs (pooled; commuter districts left out, as in Q7)")
    rep.table(res["slopes"].round(3), title="OLS on log2(income), HC1: change per doubling of income")
    rep.emit()


def _md(text):
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: f"`{m.group(0)}`", text)


def _dl(rows, fmt):
    return "; ".join(fmt(r) for r in rows.itertuples())


def _series(f, m, fmt, extra=""):
    return ", ".join(fmt(v) for v in f[m]) + f" (poorest to richest{extra})"


def _write_answer(cfg, res):
    d, ex, f = res["districts"], res["exclusions"].set_index("step"), res["facts"]
    ok = d[~d.low_n]
    rated = ok[~ok.commuter]
    n_an = int(d.n.sum())
    tot_unres = int(d.n_unresolved.sum())
    city_pct = 100 * tot_unres / n_an
    by_count = ok.sort_values("n_unresolved", ascending=False).head(TOP_K)
    by_rate = rated.sort_values("unresolved_per_1k_yr", ascending=False).head(TOP_K)
    by_share = ok.sort_values("pct_unresolved", ascending=False).head(3)
    by_excess = ok.sort_values("excess_pts_mix", ascending=False)
    sig_over = by_excess[by_excess.z_mix > Z95]
    sig_under = by_excess[by_excess.z_mix < -Z95]
    yr = res["by_year"]
    n_final = int(d.n_unresolved_final.sum())
    conc_c, conc_t, oc = res["concentration_categories"], res["concentration_types"], res["origin_city"].set_index(
        "origin")
    t, u = f["t"], f["u"]
    td, ud = _corr(res, "district", "adj_pct"), _corr(res, "district", "excess_pts_mix")
    raw = _corr(res, "PUMA", "pct_unresolved")
    city_c, priv_c = _corr(res, "PUMA", "pct_city"), _corr(res, "PUMA", "pct_private")
    rate_c, unres_c = _corr(res, "PUMA", "rate_per_1k_yr"), _corr(res, "PUMA", "unresolved_per_1k_yr")
    both = spearman(ok.adj_pct, ok.excess_pts_mix)
    changed = (np.isfinite(raw["rho"]) and np.isfinite(u["rho"])
               and (np.sign(raw["rho"]) != np.sign(u["rho"]) or abs(raw["rho"] - u["rho"]) >= 0.2))
    like_line = (f"- **Like with like matters.** Across income areas, the raw unresolved share correlates with income "
                 f"at {_rho(raw)}; against the expected share for the same complaint types and years it is "
                 f"{_signed(u['rho'])}. "
                 + ("Much of the raw pattern comes from which kinds of complaints (and so which agencies) an area "
                    "files, not from how its complaints are handled." if changed else
                    "Adjusting for complaint type and year changes the picture little."))
    both_line = "- **The two service measures together:** across districts, time to close vs the excess still open " \
                f"{_rho(both)}. " + (
        "Districts with more complaints still open than expected tend to close the rest faster: closed-only times can "
        "flatter districts that leave slow complaints open, so read the two measures together."
        if np.isfinite(both["rho"]) and both["p"] < ALPHA and both["rho"] < 0 else
        "Districts that close slowly also leave more complaints open, so both measures point the same way."
        if np.isfinite(both["rho"]) and both["p"] < ALPHA else
        "The two measures are not clearly related, so each adds information.")
    fif = res["fifths"]
    sl = res["slopes"]
    excl = "; ".join(f"{_md(k)} {int(r.rows):,} ({r.pct:.1f}%)" for k, r in ex.iloc[1:-1].iterrows())
    short = res["n_days"] < 365
    names = dict(zip(d.boro_cd, d.cd_name))
    lab = dict(zip(d.boro_cd, d.label))
    comm = d[d.commuter]
    comm_note = f"; {', '.join(comm.label)} left out" if len(comm) else ""
    inc_set = set(int(b) for s in res["pumas"].districts for b in str(s).split(", "))
    partners = d[d.puma_2010.isin(set(comm.puma_2010)) & ~d.commuter & d.boro_cd.isin(inc_set)]
    p = res["pumas"]
    n_pooled = int((p.n_districts > 1).sum())

    def dname(b):
        return lab.get(int(b), str(b))

    def conc_line(r):
        hi = ", ".join(q4.district_short(b, names) for b in r.high_districts[:4])
        more = f" and {len(r.high_districts) - 4} more" if len(r.high_districts) > 4 else ""
        top = (f"; strongest in {dname(r.top_boro_cd)} ({r.top_share_pct:.1f}% of its complaints, LQ {r.top_lq:.1f}, "
               f"n {int(r.top_n):,})" if np.isfinite(r.top_boro_cd) else "")
        return (f"{r.key} (Hoover {r.hoover_pct:.0f}%; {r.n_districts_high} districts at LQ {LQ_HIGH:g}+"
                + (f": {hi}{more}" if hi else "") + f"{top})")

    def slope_line(m, level):
        s = sl[(sl.metric == m) & (sl.unit == level)].set_index("model")
        if s.empty or not {"all", "within borough"} <= set(s.index):
            return "not estimable"
        a, w = s.loc["all"], s.loc["within borough"]
        k = 1 if m == "adj_pct" else 2

        def f_(v):   # no "-0.0": round first, then add 0.0 to turn -0.0 into 0.0
            return f"{round(float(v), k) + 0.0:+.{k}f}"
        unit = "%" if m == "adj_pct" else " points"
        return _minus(f"{f_(a.per_doubling)}{unit} per doubling of income (95% CI {f_(a.ci_low)} to "
                      f"{f_(a.ci_high)}, p {_p(a.p)}); within boroughs {f_(w.per_doubling)} ({f_(w.ci_low)} to "
                      f"{f_(w.ci_high)})")

    # every citywide income test of the two service measures, so one pass is read against how many were run
    tests = []
    for m, what in [("adj_pct", "time to close"), ("excess_pts_mix", "share left open")]:
        for level, where in [("PUMA", "income areas"), ("district", "districts")]:
            tests.append((f"{what}, rank correlation across {where}", _corr(res, level, m)["p"]))
            s = sl[(sl.metric == m) & (sl.unit == level) & (sl.model == "all")]
            if len(s):
                tests.append((f"{what}, straight-line fit across {where}", float(s.p.iloc[0])))
    tests = [(name, pv) for name, pv in tests if np.isfinite(pv)]
    passed = [(name, pv) for name, pv in tests if pv < ALPHA]
    tests_line = (f"- **How much weight one test can bear.** {len(passed)} of the {len(tests)} citywide income tests of "
                  "the two service measures (rank correlation and straight-line fit, across income areas and across "
                  f"districts) {'has' if len(passed) == 1 else 'have'} p < {ALPHA:g}"
                  + (": " + "; ".join(f"{name} (p {_p(pv)})" for name, pv in passed) if passed else "")
                  + f". If income and service were unrelated, about {len(tests) * ALPHA:.1f} of {len(tests)} would "
                  "pass by chance; the tests share the same data, so passes do not add up as independent evidence.")

    if len(fif) >= 2:
        q_lo, q_hi = fif.iloc[0], fif.iloc[-1]
        fifth_line = (f"- **Income fifths** ({len(res['pumas'])} PUMAs in {len(fif)} groups, poorest "
                      f"${q_lo.income_min / 1000:,.0f}k-${q_lo.income_max / 1000:,.0f}k, richest "
                      f"${q_hi.income_min / 1000:,.0f}k-${q_hi.income_max / 1000:,.0f}k{comm_note}). Time to close vs "
                      "typical for "
                      f"the same complaints: {_series(fif, 'adj_pct', lambda v: _minus(f'{v:+.1f}%') if np.isfinite(v) else 'n/a')}"
                      f". Left open beyond the complaint mix: "
                      f"{_series(fif, 'excess_pts_mix', lambda v: _minus(f'{v:+.2f}'))} points. Unresolved per 1,000 "
                      f"residents a year: {_series(fif, 'unresolved_per_1k_yr', _num)}. "
                      + ("The time-to-close series rises and falls rather than stepping steadily with income."
                         if not (fif.adj_pct.is_monotonic_increasing or fif.adj_pct.is_monotonic_decreasing)
                         else "The time-to-close series moves steadily with income."))
    else:
        fifth_line = "- **Income fifths** could not be compared (too few districts with enough complaints)."

    years_line = ("- **By year opened** (share of complaints still unresolved): "
                  + ", ".join(f"{int(r.year)} {r.pct_unresolved:.1f}%" for r in yr.itertuples()) + "."
                  + (f" {peak_year_sentence(res['peak_year'])}" if res.get("peak_year") else
                     f" Too few years ({len(yr)}) to say whether one stands out."))
    final_line = (f"- **Recent complaints.** {n_final:,} of the {tot_unres:,} unresolved complaints "
                  f"({100 * n_final / max(tot_unres, 1):.0f}%) were opened in the final {FINAL_DAYS} complete days "
                  f"(from {res['d_final']:%b %d, %Y}). Those had the least time to close, so every comparison below is "
                  "against the citywide share for the same year.")
    if short:
        final_line += (f" **The data cover only {res['n_days']} days, so all of it counts as 'final months' and "
                       "the age adjustment cannot separate old from new backlog here.** Status is as of the data "
                       "extract, which the data do not record; rerun on the full data.")
    oc_text = "; ".join(
        f"{ORIGIN_LABEL[k]} {oc.loc[k, 'pct']:.1f}% ({int(oc.loc[k, 'n']):,}; {oc.loc[k, 'pct_unresolved']:.1f}% "
        f"unresolved, {oc.loc[k, 'pct_of_unresolved']:.0f}% of all unresolved; mainly {oc.loc[k, 'main_categories']})"
        for k in ORIGINS if oc.loc[k, "n"] > 0)

    lines = [
        "# Q9. Equity in city services",
        "",
        f"*Computed by `nyc311/q09_equity.py` from {n_an:,} complaints, {res['period_long']}.*",
        "",
        "## What is counted",
        "",
        f"- **Unit:** the {len(d)} standard community districts (`boro_cd`). The complaints are the same as Q7's: of "
        f"{int(res['exclusions'].rows.iloc[0]):,} rows in the default analysis set (not `is_test`, not `dup_key`), "
        f"{n_an:,} ({ex.pct.iloc[-1]:.1f}%) are analyzed. Left out: {excl}. Complaints on complete days without a "
        f"district still count in the citywide benchmarks ({res['n_benchmark']:,} complaints).",
        "- **Unresolved** = an open status (Open, Pending, Assigned, ...) and no close date (`unresolved`; an open "
        "status with a close date is a `status_conflict`, counted neither way). **Expected** = the citywide unresolved "
        "share for the same year (age), or for the same complaint type and year (age and mix); excess = observed "
        "minus expected, in percentage points.",
        f"- **Per 1,000 residents a year** divides by resident-years, Q7's rule: 2010 Census for "
        f"{res['n_2010']:,} days before {q7.POP_2020_FROM_YEAR}, 2020 Census for {res['n_2020']:,} days from "
        f"{q7.POP_2020_FROM_YEAR}."
        + (f" {', '.join(comm.label)} is a {q7.COMMUTER_LABEL} (few residents, many workers), so, as in Q7, it is "
           "left out of per-resident rankings and of every comparison with income (correlations, straight-line fits, "
           "income fifths)"
           + (f"; {', '.join(partners.label)}, which shares its income area (PUMA), stands alone there"
              if len(partners) else "") + "." if len(comm) else ""),
        f"- **Time to close** is Q4's mix-adjusted measure: each closed complaint vs the citywide geometric mean for "
        f"the same complaint type and year, averaged per district (instant closures left out; districts need "
        f"{MIN_N_CLOSED} closures).",
        "",
        "## Where the unresolved complaints are",
        "",
        f"- **Across the {len(d)} districts** {tot_unres:,} of {n_an:,} complaints ({city_pct:.1f}%) are unresolved.",
        f"- **Most unresolved complaints:** " + _dl(by_count, lambda r: f"{r.label} {int(r.n_unresolved):,} "
                                                     f"({r.pct_unresolved:.1f}% of {int(r.n):,})") + ".",
        f"- **Per 1,000 residents a year:** " + _dl(by_rate, lambda r: f"{r.label} {_num(r.unresolved_per_1k_yr)}")
        + "." + "".join(f" {r.label} ({q7.COMMUTER_LABEL}, not ranked): {_num(r.unresolved_per_1k_yr)}."
                        for r in comm.itertuples()),
        f"- **Highest share still open:** " + _dl(by_share, lambda r: f"{r.label} {r.pct_unresolved:.1f}% (expected "
                                                  f"{r.expected_pct_year:.1f}% for the year, {r.expected_pct_mix:.1f}% "
                                                  f"for its complaint mix)") + ".",
        f"- **Beyond their complaint mix** (observed minus expected for the same type and year), the most open: "
        + _dl(by_excess.head(3), lambda r: f"{r.label} {_pts(r.excess_pts_mix)} ({int(r.n_unresolved):,} open vs "
                                         f"{r.expected_mix:,.0f} expected)")
        + f"; the fewest: " + _dl(by_excess.tail(2).iloc[::-1], lambda r: f"{r.label} {_pts(r.excess_pts_mix)}")
        + f". {len(sig_over)} districts are above expectation and {len(sig_under)} below by more than 1.96 binomial "
        f"standard errors (which ignore clustering, so treat these as screens, not tests).",
        final_line,
        years_line,
        "",
        "## Are complaint types concentrated in specific areas?",
        "",
        f"- **Method.** Location quotient (LQ) = a district's share of a category / the citywide share; LQ {LQ_HIGH:g} "
        f"means twice as common there. A district is named only with {MIN_CELL}+ complaints of that kind. The Hoover "
        f"index is the % of a category's complaints that would have to move to another district to match the map of "
        f"all complaints (0 = spread like everything else).",
        f"- **Most concentrated main categories** ({MAIN_CATEGORY_SHARE:g}%+ of complaints): "
        + "; ".join(conc_line(r) for r in conc_c.head(4).itertuples()) + ".",
        f"- **Least concentrated:** " + "; ".join(f"{r.key} (Hoover {r.hoover_pct:.0f}%)"
                                              for r in conc_c.tail(2).iloc[::-1].itertuples()) + ".",
        f"- **Among the {len(conc_t)} biggest complaint types,** the most concentrated are "
        + "; ".join(conc_line(r) for r in conc_t.head(3).itertuples()) + ".",
        "",
        "## Where complaints originate",
        "",
        f"- **Mix across the {len(d)} districts** (Q3's taxonomy): {oc_text}.",
        f"- **By district,** people-origin is the largest share in {int((ok.dominant_origin == 'private').sum())} of "
        f"{len(ok)} districts (range {ok.pct_private.min():.0f}%-{ok.pct_private.max():.0f}%); city origin ranges "
        f"{ok.pct_city.min():.0f}%-{ok.pct_city.max():.0f}% (highest: "
        + _dl(ok.nlargest(3, "pct_city"), lambda r: f"{r.label} {r.pct_city:.0f}%") + "; lowest: "
        + _dl(ok.nsmallest(2, "pct_city"), lambda r: f"{r.label} {r.pct_city:.0f}%") + ").",
        (f"- **By income fifth:** city-origin share {_series(fif, 'pct_city', lambda v: f'{v:.0f}%')}; people-origin "
         f"{_series(fif, 'pct_private', lambda v: f'{v:.0f}%')}. Across income areas: city-origin share "
         f"{_rho(city_c)}; people-origin share {_rho(priv_c)}." if len(fif) else
         "- **By income fifth:** not computed (too few districts)."),
        "",
        "## Patterns that suggest inequitable service",
        "",
        f"- **Headline (from the numbers below):** {f['title']}.",
        f"- **Time to close vs income:** {_rho(t)} across {len(p)} income areas (PUMAs; {n_pooled} pairs of districts "
        f"that share one income value pooled); {_rho(td)} across {res['n_income_districts']} districts, which counts "
        f"those pairs twice. Straight-line fit across income areas: {slope_line('adj_pct', 'PUMA')}; across "
        f"districts: {slope_line('adj_pct', 'district')}.",
        f"- **Still open beyond the complaint mix vs income:** {_rho(u)} across income areas; {_rho(ud)} across "
        f"districts. Straight-line fit across income areas: {slope_line('excess_pts_mix', 'PUMA')}; across "
        f"districts: {slope_line('excess_pts_mix', 'district')}.",
        tests_line,
        like_line,
        both_line,
        fifth_line,
        f"- **Volume per resident:** complaints per 1,000 residents a year vs income {_rho(rate_c)}; unresolved per "
        f"1,000 a year {_rho(unres_c)} (income areas).",
        "",
        f"![{f['title']}](figs/q09_equity.png)",
        "",
        f"- **Reading the chart.** A: the {TOP_BARS} districts with the most unresolved complaints, bars split by "
        "origin, with the count and the rate per 1,000 residents a year. B: how concentrated each main category is "
        "(Hoover index), colored by its main origin, with the district where it is most over-represented. C: the "
        "origin mix by income fifth. D: by the same fifths, time to close and the share left open, both against the "
        "citywide figure for the same complaint type and year, and unresolved complaints per 1,000 residents a year."
        + (f" C and D leave out {', '.join(comm.label)}." if len(comm) else ""),
        "",
        "## Limits and what it would take to test causes",
        "",
        "- **Correlation, not causation.** These are associations across areas. Income moves with housing stock, "
        "renters, language, density and borough operations; an area-level link says nothing about individual "
        "households (ecological fallacy).",
        "- **Measurement.** Closing a ticket is not fixing the problem; agencies close differently (referrals, "
        "date-only HPD closures). 311 measures reporting, not need: lower-reporting areas can look well served. "
        f"ACS income is for PUMAs ({res['n_shared_pairs']} pairs share a value), in 2014 dollars; boundaries are "
        "current. Status is as of the data extract.",
        "- **To test causes:** ticket-level models with descriptor, building and agency-unit controls; independent "
        "measures of need (HPD violations, inspections, street-condition ratings) to separate reporting from "
        "conditions; crew and inspector deployment by district; and changes over time, such as a new staffing rule or "
        "service-level target rolled out in some districts first (difference-in-differences), or resident follow-up "
        "on whether the problem was fixed.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q09.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
