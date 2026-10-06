"""Q3: consistency issues in complaint_type (Q3a) and the cleaning strategy (Q3b).

Applies nyc311.taxonomy to every raw complaint_type in the raw Parquet, writes the applied map and the review
queue, prints the evidence for each kind of issue (including names that replaced each other over time), and drafts
outputs/answers/q03.md. Every number is computed from the data it runs on.
"""
from __future__ import annotations

import difflib
import re
from itertools import combinations

import pandas as pd

from nyc311 import ingest, rules, taxonomy
from nyc311.report import Report

STAGE = "q03_complaint_types"
APPLIED_COLUMNS = ["complaint_type", "n", "canonical_type", "category", "origin", "is_info_request", "needs_review"]
RARE_BELOW = 10            # a type with fewer rows than this is "rare"
LOOKALIKE_RATIO = 0.85     # names this similar (difflib, after normalize) count as look-alikes...
WORD_RATIO = 0.8           # ...if the words that differ are also this similar
GENERIC_WORDS = {"other", "miscellaneous", "misc", "general"}
CAPS_WORD_MIN = 6          # an ALL-CAPS name with a word this long is a legacy code, not an acronym
EXAMPLES = 4
# Renames over time
TRIM = 0.0005              # data window: 0.05th to 99.95th percentile of created dates, so a few stray rows
                           # (2015 rows inside a 2010 file) do not stretch it
MIN_SPAN_DAYS = 365        # renames are looked for only when the window covers at least this many days
GAP_DAYS = 30              # no rows in the window's last (first) GAP_DAYS: the spelling has ended (started)
REORG_WINDOW_DAYS = 7      # an agency's types that end or start within this many days of each other form one event...
REORG_MIN_TYPES = 3        # ...if at least this many types end or start, with at least one of each
HANDOFF_MIN_ROWS = 100     # both spellings in a rename handoff need at least this many rows to be dated
LEGACY_MIN_ROWS = 1000     # ended types at least this large get a descriptor crosswalk
CR = "try_cast(created_date as timestamp)"
COMBO_COLUMNS = ["complaint_type", "agency", "descriptor", "n", "first", "last"]
_IRREGULAR_SPACING = re.compile(r"\S-\s|\s-\S|\s/|/\s")
_TOKENS = re.compile(r"[a-z0-9]+")
_PLAIN = re.compile(r"[A-Za-z0-9 ./:_-]+")  # examples of invalid entries are quoted only if made of these characters


def _none_if_null(s):
    return s.astype(object).where(s.notna(), None)


def data_window(con, src):
    """First and last created date after trimming TRIM of rows at each end (None, None if no dates parse)."""
    lo, hi = con.sql(f"select quantile_disc(cr, {TRIM}), quantile_disc(cr, {1 - TRIM}) "
                     f"from (select {CR} cr from {src}) where cr is not null").fetchone()
    return (None, None) if lo is None else (pd.Timestamp(lo), pd.Timestamp(hi))


def type_combos(con, src, lo, hi):
    """One row per (complaint_type, agency, descriptor): rows, and first/last created date inside the window."""
    inside = "true" if lo is None else f"cr between timestamp '{lo}' and timestamp '{hi}'"
    combos = con.sql(f"""select complaint_type, agency, descriptor, count(*)::bigint n,
                                min(cr) filter (where {inside}) first_seen, max(cr) filter (where {inside}) last_seen
                         from (select complaint_type, agency, descriptor, {CR} cr from {src}) group by all""").df()
    combos = combos.rename(columns={"first_seen": "first", "last_seen": "last"})
    for c in ["complaint_type", "agency", "descriptor"]:
        combos[c] = _none_if_null(combos[c])
    for c in ["first", "last"]:
        combos[c] = pd.to_datetime(combos[c])
    return combos[COMBO_COLUMNS]


def type_profile(combos):
    """One row per raw complaint_type: rows, agencies, descriptor set, top descriptor, first and last date."""
    prof = {}
    for r in combos.itertuples(index=False):
        p = prof.setdefault(r.complaint_type, {"n": 0, "agencies": {}, "descriptors": {}, "first": [], "last": []})
        p["n"] += int(r.n)
        p["agencies"][r.agency] = p["agencies"].get(r.agency, 0) + int(r.n)
        p["descriptors"][r.descriptor] = p["descriptors"].get(r.descriptor, 0) + int(r.n)
        p["first"].append(r.first)
        p["last"].append(r.last)
    rows = []
    for ct, p in prof.items():
        ag = sorted(p["agencies"].items(), key=lambda kv: -kv[1])
        de = sorted(p["descriptors"].items(), key=lambda kv: -kv[1])
        rows.append({"complaint_type": ct, "n": p["n"], "main_agency": ag[0][0], "n_agencies": len(ag),
                     "agencies": " / ".join(f"{a} {v:,}" for a, v in ag),
                     "descriptors": frozenset(d for d, _ in de),
                     "top_descriptor": f"{de[0][0]} ({de[0][1] / p['n']:.0%})",
                     "first": pd.Series(p["first"], dtype="datetime64[ns]").min(),
                     "last": pd.Series(p["last"], dtype="datetime64[ns]").max()})
    return pd.DataFrame(rows)


def _label(r):
    return f"{r.complaint_type} ({r.n:,} {r.main_agency})"


def _shared(a, b):
    return f"{len(a & b)} of {len(a | b)}"


def spelling_variants(t):
    """Raw spellings that normalize() reads as one name, and whether review merged them."""
    rows = []
    for key, d in t.groupby(t.complaint_type.map(taxonomy.normalize), sort=False):
        if len(d) < 2:
            continue
        d = d.sort_values("n", ascending=False)
        if d.canonical_type.nunique() == 1:
            decision = f"merged -> {d.canonical_type.iloc[0]}"
        else:
            decision = f"kept apart: {_shared(*d.descriptors.iloc[:2])} descriptors shared"
        rows.append({"key": key, "n": int(d.n.sum()), "spellings": " | ".join(_label(r) for r in d.itertuples()),
                     "decision": decision})
    return pd.DataFrame(rows, columns=["key", "n", "spellings", "decision"]).sort_values("n", ascending=False)


def _is_all_caps(s):
    """ALL-CAPS words, not an acronym: upper case and at least one word of CAPS_WORD_MIN letters (SCRIE is not)."""
    return isinstance(s, str) and s == s.upper() and any(len(w) >= CAPS_WORD_MIN for w in re.findall(r"[A-Z]+", s))


def all_caps_by_agency(t):
    caps = t.complaint_type.map(_is_all_caps)
    rows = []
    for agency, d in t[caps].groupby("main_agency"):
        rows.append({"agency": agency, "caps_types": len(d), "other_types": int((t.main_agency == agency).sum() - len(d)),
                     "caps_rows": int(d.n.sum()), "examples": ", ".join(d.sort_values("n", ascending=False).complaint_type.head(EXAMPLES))})
    return pd.DataFrame(rows, columns=["agency", "caps_types", "other_types", "caps_rows", "examples"]).sort_values(
        "caps_rows", ascending=False), caps


def _contains(short, long):
    k = len(short)
    return 0 < k < len(long) and any(long[i:i + k] == short for i in range(len(long) - k + 1))


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def lookalike_pairs(t):
    """Pairs of raw names that look alike but were reviewed as different types: same normalized name; similar
    spelling (difflib >= LOOKALIKE_RATIO on the whole name and >= WORD_RATIO on the words that differ, so
    'Taxi Complaint'/'Taxi Compliment' count but family siblings like 'Street Sign - Damaged'/'- Missing' don't);
    or one name inside the other across two agencies."""
    recs = [r for r in t.sort_values("n", ascending=False).itertuples(index=False) if isinstance(r.complaint_type, str)]
    rows = []
    for a, b in combinations(recs, 2):
        if a.canonical_type == b.canonical_type:
            continue
        ka, kb = taxonomy.normalize(a.complaint_type), taxonomy.normalize(b.complaint_type)
        ratio = _ratio(ka, kb)
        ta, tb = _TOKENS.findall(ka), _TOKENS.findall(kb)
        words = _ratio(" ".join(w for w in ta if w not in tb), " ".join(w for w in tb if w not in ta))
        inside = a.main_agency != b.main_agency and (_contains(ta, tb) or _contains(tb, ta))
        if ka == kb or (ratio >= LOOKALIKE_RATIO and words >= WORD_RATIO) or inside:
            rows.append({"type_a": _label(a), "type_b": _label(b), "similarity": round(ratio, 2),
                         "descriptors_shared": _shared(a.descriptors, b.descriptors), "n": a.n + b.n})
    out = pd.DataFrame(rows, columns=["type_a", "type_b", "similarity", "descriptors_shared", "n"])
    out = out.sort_values(["similarity", "n"], ascending=False)
    seen, first = set(), []
    for i, r in out.iterrows():  # most similar first, but each raw type once before any repeats
        if r.type_a not in seen and r.type_b not in seen:
            first.append(i)
            seen.update([r.type_a, r.type_b])
    return pd.concat([out.loc[first], out.drop(index=first)]).drop(columns="n")


def vague_names(t, caps, merged):
    """One-word or generic names whose meaning sits in the descriptor; review renamed them."""
    key = t.complaint_type.map(taxonomy.normalize)
    renamed = key != t.canonical_type.map(taxonomy.normalize)
    words = key.map(lambda k: _TOKENS.findall(k))
    vague = words.map(lambda w: len(w) == 1 or bool(GENERIC_WORDS & set(w)))
    d = t[renamed & vague & ~caps & ~merged].sort_values("n", ascending=False)
    return d[["complaint_type", "n", "main_agency", "top_descriptor", "canonical_type"]], renamed


# ---------------------------------------------------------------- renames over time

def _day(ts):
    return "" if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def _period(first, last):
    return f"{_day(first)}..{_day(last)}"


def _dates(start, end):
    return start if start == end else f"{start}..{end}"


def _n(k, word):
    return f"{k:,} {word}" + ("" if k == 1 else "s")


HANDOFF_COLUMNS = ["canonical_type", "old_type", "old_period", "old_n", "new_type", "new_period", "new_n",
                   "overlap_days"]
REORG_COLUMNS = ["agency", "start", "end", "n_ended", "rows_ended", "n_started", "rows_started", "ended", "started"]
LEGACY_COLUMNS = ["complaint_type", "agency", "n", "last_seen", "canonical_type", "bridged_to", "top_successor",
                  "top_share", "matched_share", "n_successors", "flow"]


def rename_handoffs(t, hi):
    """Spellings of one canonical type that replaced each other: A stopped more than GAP_DAYS before the window
    ends; B has the same main agency, started at least GAP_DAYS after A started and ran on at least GAP_DAYS after
    A's last row (the largest such B); both have HANDOFF_MIN_ROWS rows. Siblings grouped under one canonical type
    (different agencies, or ending together) are not handoffs. overlap_days < 0 is a gap between them."""
    gap = pd.Timedelta(days=GAP_DAYS)
    cutoff = hi - gap
    rows = []
    for canon, d in t[t.n >= HANDOFF_MIN_ROWS].groupby("canonical_type"):
        if len(d) < 2:
            continue
        for a in d.itertuples(index=False):
            if pd.isna(a.last) or a.last >= cutoff:
                continue
            later = d[(d.main_agency == a.main_agency) & (d["first"] > a.first + gap) & (d["last"] > a.last + gap)]
            if later.empty:
                continue
            b = later.sort_values("n", ascending=False).iloc[0]
            rows.append({"canonical_type": canon, "old_type": a.complaint_type, "old_period": _period(a.first, a.last),
                         "old_n": int(a.n), "new_type": b.complaint_type, "new_period": _period(b["first"], b["last"]),
                         "new_n": int(b.n), "overlap_days": int((a.last.normalize() - b["first"].normalize()).days)})
    out = pd.DataFrame(rows, columns=HANDOFF_COLUMNS)
    return out.assign(_n=out.old_n + out.new_n).sort_values("_n", ascending=False).drop(columns="_n")


def reorganizations(t, lo, hi):
    """Dates on which one agency retired some types and introduced others: ends and starts of its types within
    REORG_WINDOW_DAYS of each other, at least REORG_MIN_TYPES of them and at least one of each kind. Types with
    fewer than RARE_BELOW rows are too small to date and are ignored."""
    end_cut, start_cut = hi - pd.Timedelta(days=GAP_DAYS), lo + pd.Timedelta(days=GAP_DAYS)
    rows = []
    for agency, d in t[t.n >= RARE_BELOW].groupby("main_agency"):
        recs = list(d.itertuples(index=False))
        events = [(r.last.normalize(), "ended", r) for r in recs if pd.notna(r.last) and r.last < end_cut]
        events += [(r.first.normalize(), "started", r) for r in recs if pd.notna(r.first) and r.first > start_cut]
        events.sort(key=lambda e: e[0])
        groups = []
        for e in events:
            if groups and (e[0] - groups[-1][-1][0]).days <= REORG_WINDOW_DAYS:
                groups[-1].append(e)
            else:
                groups.append([e])
        for g in groups:
            ended = sorted((r for _, k, r in g if k == "ended"), key=lambda r: -r.n)
            started = sorted((r for _, k, r in g if k == "started"), key=lambda r: -r.n)
            if ended and started and len(ended) + len(started) >= REORG_MIN_TYPES:
                rows.append({"agency": agency, "start": _day(g[0][0]), "end": _day(g[-1][0]),
                             "n_ended": len(ended), "rows_ended": sum(int(r.n) for r in ended),
                             "n_started": len(started), "rows_started": sum(int(r.n) for r in started),
                             "ended": "; ".join(f"{r.complaint_type} ({r.n:,})" for r in ended),
                             "started": "; ".join(f"{r.complaint_type} ({r.n:,})" for r in started)})
    return pd.DataFrame(rows, columns=REORG_COLUMNS).sort_values("rows_ended", ascending=False)


def _dkey(s):
    return re.sub(r"[-\s]+", " ", taxonomy.normalize(s)).strip()


def descriptor_successors(combos, raw, canon=None):
    """Where an ended type's rows went. Each descriptor goes to the type of the same (main) agency that carries a
    descriptor of the same name (after normalize(), '-' read as a space), started at least GAP_DAYS after the old
    type and is still in use GAP_DAYS after the old type's last row: one with the same canonical type (canon: raw ->
    canonical) if any, else the largest. Rows with no such match count under None. Returns {successor or None: rows}."""
    gap = pd.Timedelta(days=GAP_DAYS)
    old = combos[combos.complaint_type == raw]
    start, end = old["first"].min(), old["last"].max()
    if old.empty or pd.isna(end):
        return {}
    agency = old.groupby("agency").n.sum().idxmax()
    type_start = combos.groupby("complaint_type", dropna=False)["first"].min()
    newer = set(type_start[type_start > start + gap].index)
    later = combos[(combos.agency == agency) & (combos.complaint_type != raw) & combos.complaint_type.isin(newer)
                   & (combos["last"] > end + gap)]
    later = later.assign(key=later.descriptor.map(_dkey))
    cand = later.groupby(["key", "complaint_type"]).n.sum().reset_index()
    own = canon.get(raw) if canon else None
    cand["same"] = cand.complaint_type.map(lambda c: canon is not None and own is not None and canon.get(c) == own)
    best = (cand.sort_values(["same", "n"], ascending=False).drop_duplicates("key").set_index("key").complaint_type)
    flows = {}
    for r in old.itertuples(index=False):
        key = _dkey(r.descriptor)
        succ = best.get(key) if r.agency == agency and key else None
        flows[succ] = flows.get(succ, 0) + int(r.n)
    return flows


def legacy_types(t, combos, handoffs, hi):
    """Types that ended before the window's end (at least LEGACY_MIN_ROWS rows) and where their descriptors went."""
    cutoff = hi - pd.Timedelta(days=GAP_DAYS)
    bridged = dict(zip(handoffs.old_type, handoffs.new_type))
    canon = dict(zip(t.complaint_type, t.canonical_type))
    rows = []
    for r in t[(t["last"] < cutoff) & (t.n >= LEGACY_MIN_ROWS)].sort_values("n", ascending=False).itertuples(index=False):
        flows = descriptor_successors(combos, r.complaint_type, canon)
        total = sum(flows.values()) or 1
        named = sorted(((k, v) for k, v in flows.items() if k is not None), key=lambda kv: -kv[1])
        unmatched = flows.get(None, 0)
        flow = "; ".join(f"{k} {v / total:.0%}" for k, v in named)
        flow = f"{flow}; no same-name descriptor later {unmatched / total:.0%}" if named else "no same-name descriptor later"
        rows.append({"complaint_type": r.complaint_type, "agency": r.main_agency, "n": int(r.n),
                     "last_seen": _day(r.last), "canonical_type": r.canonical_type,
                     "bridged_to": bridged.get(r.complaint_type, ""),
                     "top_successor": named[0][0] if named else "", "top_share": named[0][1] / total if named else 0.0,
                     "matched_share": 1 - unmatched / total, "n_successors": len(named), "flow": flow})
    return pd.DataFrame(rows, columns=LEGACY_COLUMNS)


def renames_over_time(t, combos, lo, hi):
    """Handoffs, reorganizations and legacy types, or empty tables if the window is shorter than MIN_SPAN_DAYS."""
    span = 0 if lo is None else (hi - lo).days
    if span < MIN_SPAN_DAYS:
        return {"too_short": True, "span_days": span, "handoffs": pd.DataFrame(columns=HANDOFF_COLUMNS),
                "reorgs": pd.DataFrame(columns=REORG_COLUMNS), "legacy": pd.DataFrame(columns=LEGACY_COLUMNS)}
    handoffs = rename_handoffs(t, hi)
    return {"too_short": False, "span_days": span, "handoffs": handoffs, "reorgs": reorganizations(t, lo, hi),
            "legacy": legacy_types(t, combos, handoffs, hi)}


# ---------------------------------------------------------------- unusual entries and summaries

def _missing(d):
    """A descriptor that says nothing: null, blank, or a value the cleaning rules treat as missing ('N/A')."""
    return d is None or (not isinstance(d, str) and pd.isna(d)) or str(d).strip() in ["", *rules.NULL_IF["descriptor"]]


def unusual_entries(t):
    """Invalid entries (map category) by canonical type, and valid types whose rows never carry a descriptor."""
    inv = t[t.category == taxonomy.INVALID]
    by = inv.groupby("canonical_type").agg(types=("complaint_type", "size"), rows=("n", "sum"),
                                           first=("first", "min"), last=("last", "max")).reset_index()
    by = by.sort_values("rows", ascending=False)
    by["period"] = [_period(a, b) for a, b in zip(by["first"], by["last"])]
    plain = inv[inv.complaint_type.map(lambda s: isinstance(s, str) and bool(_PLAIN.fullmatch(s))).astype(bool)]
    examples = plain.sort_values("n", ascending=False).complaint_type.head(EXAMPLES).tolist()
    valid = t[t.category != taxonomy.INVALID]
    blank = valid[valid.descriptors.map(lambda ds: all(_missing(d) for d in ds)).astype(bool)]
    blank = blank.sort_values("n", ascending=False)[["complaint_type", "n", "main_agency", "canonical_type"]]
    return by[["canonical_type", "types", "rows", "period"]], examples, blank


def _pct(x, total):
    return f"{100 * x / total:.1f}%"


def summary_tables(t, total):
    cat = t.groupby("category").agg(canonical_types=("canonical_type", "nunique"), raw_types=("complaint_type", "size"),
                                    rows=("n", "sum")).sort_values("rows", ascending=False).reset_index()
    cat["pct_rows"] = [_pct(v, total) for v in cat.rows]
    complaints = t[~t.is_info_request]
    org = t.groupby("origin").agg(canonical_types=("canonical_type", "nunique"), rows=("n", "sum")).reset_index()
    org = org.merge(complaints.groupby("origin").n.sum().rename("complaint_rows").reset_index(), how="left").fillna(0)
    org["pct_rows"] = [_pct(v, total) for v in org.rows]
    org["pct_of_complaints"] = [_pct(v, complaints.n.sum()) for v in org.complaint_rows]
    org = org.sort_values("rows", ascending=False).drop(columns="complaint_rows")
    return cat, org


def run(cfg):
    con = cfg.connect()
    src = ingest.raw_source(cfg)
    lo, hi = data_window(con, src)
    return analyze(cfg, type_combos(con, src, lo, hi), lo, hi)


def analyze(cfg, combos, lo, hi):
    """Everything after the one SQL pass: classify, write tables, the report block and the answer draft."""
    prof = type_profile(combos)
    n_descriptors = combos.descriptor.nunique()
    t = prof.merge(taxonomy.classify(prof.complaint_type), on="complaint_type", how="left")
    t = t.sort_values("n", ascending=False).reset_index(drop=True)
    valid = t[t.category != taxonomy.INVALID].reset_index(drop=True)  # name analyses leave invalid entries out

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    t[APPLIED_COLUMNS].to_csv(agg / "q03_type_map_applied.csv", index=False)
    review = t[t.needs_review]
    review[APPLIED_COLUMNS].to_csv(agg / "q03_unreviewed.csv", index=False)

    total, n_raw = int(t.n.sum()), len(t)
    n_canon, n_cat = t.canonical_type.nunique(), t.category.nunique()
    variants = spelling_variants(valid)
    caps_tab, caps = all_caps_by_agency(valid)
    merged = valid.canonical_type.duplicated(keep=False)
    pairs = lookalike_pairs(valid)
    vague, renamed = vague_names(valid, caps, merged)
    req = t[t.is_info_request]
    info_cat = req[req.category == "Information requests"]
    multi = valid[valid.n_agencies > 1][["complaint_type", "agencies"]]
    rare = valid[valid.n < RARE_BELOW].sort_values("n")
    once = rare[rare.n == 1]
    spacing = valid[valid.complaint_type.map(lambda s: isinstance(s, str) and bool(_IRREGULAR_SPACING.search(s)))]
    cat, org = summary_tables(t, total)
    n_merged_raw, n_merged_canon = int(merged.sum()), valid.canonical_type[merged].nunique()
    n_renamed = int((renamed & ~merged).sum())
    invalid, invalid_examples, blank = unusual_entries(t)
    rt = renames_over_time(valid, combos[combos.complaint_type.isin(set(valid.complaint_type))], lo, hi)
    tmap = taxonomy.load_map()
    map_families = int(tmap.canonical_type.value_counts().gt(1).sum())
    rt["handoffs"].to_csv(agg / "q03_renames.csv", index=False)
    rt["reorgs"].to_csv(agg / "q03_reorganizations.csv", index=False)
    rt["legacy"].to_csv(agg / "q03_legacy_types.csv", index=False)

    rep = Report(STAGE, cfg.out_dir)
    rep.add(f"Q3a. complaint_type consistency on {total:,} rows (window {_period(lo, hi)}, trimmed {TRIM:.2%} each end)",
            f"raw complaint_type values: {n_raw} -> canonical types: {n_canon} -> categories: {n_cat}",
            f"types needing review: {len(review)} ({int(review.n.sum()):,} rows) | descriptor values kept as the "
            f"finest level: {n_descriptors:,}")
    rep.table(variants, max_rows=EXAMPLES, title="1. One name, several spellings (case / spacing / plural)")
    rep.table(caps_tab, max_rows=EXAMPLES, title=f"2. Two naming systems: ALL-CAPS codes hold {int(valid.n[caps].sum()):,} "
              f"rows ({_pct(valid.n[caps].sum(), total)}); every other type is Title Case")
    rep.table(pairs, max_rows=EXAMPLES + 2, title="3. Look-alike names reviewed as different types (agency or descriptors differ)")
    rep.table(vague, max_rows=EXAMPLES + 2, title="4. Vague or generic names; the meaning is only in the descriptor")
    rep.add("", f"## 5. Not complaints: {len(req)} types, {int(req.n.sum()):,} rows ({_pct(req.n.sum(), total)}) are "
            f"requests or compliments",
            f"pure information requests: {len(info_cat)} types, {int(info_cat.n.sum()):,} rows | service/program "
            f"requests and compliments in other categories: {len(req) - len(info_cat)} types, "
            f"{int(req.n.sum() - info_cat.n.sum()):,} rows")
    rep.table(req[["complaint_type", "n", "main_agency", "category"]], max_rows=EXAMPLES)
    rep.table(multi, max_rows=EXAMPLES, title="6. One type, several agencies")
    rep.add("", f"## 7. Rare entries: {len(rare)} types have < {RARE_BELOW} rows ({int(rare.n.sum()):,} rows); "
            f"{len(once)} appear once: {_names(once.complaint_type)}")
    rep.table(spacing[["complaint_type", "canonical_type"]], max_rows=EXAMPLES, title="8. Irregular spacing around - and /")
    rep.add("", f"## 9. Unusual entries: {int(invalid.types.sum())} raw types ({int(invalid.rows.sum()):,} rows) are not "
            f"service requests (category '{taxonomy.INVALID}'); {len(blank)} valid types never carry a descriptor")
    rep.table(invalid, max_rows=EXAMPLES)
    if invalid_examples:
        rep.add(f"examples: {' | '.join(invalid_examples)}")
    rep.table(blank, max_rows=EXAMPLES)
    if rt["too_short"]:
        rep.add("", f"## 10. Renamed over time: window is {rt['span_days']} days, shorter than {MIN_SPAN_DAYS}; "
                "renames not looked for")
    else:
        h, g, lg = rt["handoffs"], rt["reorgs"], rt["legacy"]
        rep.table(h.drop(columns="canonical_type"), max_rows=10,
                  title=f"10. Renamed over time: {len(h)} handoffs where one spelling replaced another (same canonical type)")
        rep.table(g[["agency", "start", "end", "n_ended", "rows_ended", "n_started", "ended", "started"]], max_rows=5,
                  title=f"11. Agency reorganizations: types of one agency ending and starting within "
                        f"{REORG_WINDOW_DAYS} days")
        rep.table(lg[["complaint_type", "n", "last_seen", "bridged_to", "flow"]], max_rows=EXAMPLES + 2,
                  title=f"12. Types that ended (>= {LEGACY_MIN_ROWS:,} rows): where their descriptors went "
                        "(same-name descriptor, same agency)")
    rep.table(cat, max_rows=20, title="Categories after cleaning")
    rep.table(org, max_rows=5, title="Origin (for Q9); pct_of_complaints excludes requests")
    rep.add("", "Q3b. Strategy",
            "- Keep raw complaint_type; add canonical_type/category/origin/is_info_request from a reviewed lookup",
            "  (nyc311/complaint_type_map.csv, one row per raw spelling, notes for every merge/rename/judgment).",
            f"- Merge true duplicates and renames ({n_merged_raw} spellings -> {n_merged_canon} types here); rename "
            f"vague/ALL-CAPS names ({n_renamed}); keep look-alikes apart; split types stay legacy types.",
            f"- Hierarchy keeps granularity: {n_cat} categories > {n_canon} canonical types > {n_descriptors:,} descriptors.",
            "- New data: exact -> normalized (case/space/dash/plural) -> fuzzy (flagged) -> unreviewed (flagged);",
            "  flagged types go to agg/q03_unreviewed.csv with row counts; a reviewer adds one CSV row, no code change.")
    rep.emit()

    _write_answer(cfg, locals())
    return t[APPLIED_COLUMNS]


def _names(s, k=2 * EXAMPLES):
    """Up to k names joined, then how many more."""
    names = [str(x) for x in s]
    more = f" and {len(names) - k} more" if len(names) > k else ""
    return ", ".join(names[:k]) + more


def _bullets_from(df, fmt, k=EXAMPLES):
    return "; ".join(fmt(r) for r in df.head(k).itertuples(index=False))


def _renames_bullets(v):
    rt, lo, hi = v["rt"], v["lo"], v["hi"]
    if rt["too_short"]:
        return [f"- **Renamed over time.** This data covers {_period(lo, hi)} ({rt['span_days']} days), too short to see "
                f"one name replacing another (renames are looked for when the data span at least {MIN_SPAN_DAYS} days). "
                f"The reviewed map already bridges {v['map_families']} canonical types that have more than one raw "
                "spelling; a run on the full multi-year data lists the handoffs."]
    h, g, lg = rt["handoffs"], rt["reorgs"], rt["legacy"]
    out = [f"- **Renamed over time (similar categories with different names).** {len(h)} times one spelling replaced "
           "another for the same work; the map gives both one canonical type so the series is continuous. "
           "Largest: " + _bullets_from(h, lambda r: f"`{r.old_type}` ({r.old_n:,} rows, {r.old_period}) -> "
                                                 f"`{r.new_type}` ({r.new_n:,}, {r.new_period})", k=6) + "."]
    if not g.empty:
        out.append("- **Agency reorganizations** (an agency retiring and introducing types within "
                   f"{REORG_WINDOW_DAYS} days). " + _bullets_from(
                       g, lambda r: f"{r.agency} {_dates(r.start, r.end)}: {_n(r.n_ended, 'type')} ended "
                                    f"({r.rows_ended:,} rows), {r.n_started} started", k=EXAMPLES) + ".")
    unbridged = lg[lg.bridged_to == ""]
    split = unbridged[unbridged.n_successors >= 2]
    if not split.empty:
        out.append(f"- **Split, not renamed.** {_n(len(split), 'retired type')} with at least {LEGACY_MIN_ROWS:,} rows had their "
                   "descriptors taken over by two or more new types, so no single successor exists and they keep their "
                   "own canonical type. Matching descriptor names within the agency: " +
                   _bullets_from(split, lambda r: f"`{r.complaint_type}` ({r.n:,} rows, last {r.last_seen}): "
                                                  f"{r.flow}", k=3) + ".")
    if len(unbridged) > len(split):
        out.append(f"- **Retired without a bridged successor.** {_n(len(unbridged) - len(split), 'other type')} with at "
                   f"least {LEGACY_MIN_ROWS:,} rows stop before the data end; at most one later type of the same agency "
                   "reuses any of their descriptor names (descriptors were renamed too, or the service left 311). "
                   "Largest: " +
                   _bullets_from(unbridged[unbridged.n_successors < 2],
                                 lambda r: f"`{r.complaint_type}` ({r.n:,}, {r.agency}, last {r.last_seen})") + ".")
    return out


def _write_answer(cfg, v):
    t, total, req, info_cat, rare, once, org = v["t"], v["total"], v["req"], v["info_cat"], v["rare"], v["once"], v["org"]
    caps, valid, invalid, blank = v["caps"], v["valid"], v["invalid"], v["blank"]
    n_disjoint = int(v["pairs"].descriptors_shared.str.startswith("0 of").sum())
    by_origin = "; ".join(f"{r.origin} {r.rows:,} rows ({r.pct_rows} of all, {r.pct_of_complaints} of complaints)"
                          for r in org.itertuples(index=False))
    if invalid.empty:
        unusual = "- **Unusual entries.** No test records or web-form probes in this data"
    else:
        unusual = (f"- **Unusual entries.** {int(invalid.types.sum())} raw types ({int(invalid.rows.sum()):,} rows) are "
                   "not service requests: " + _bullets_from(
                       invalid, lambda r: f"{r.canonical_type} {r.types} types, {r.rows:,} rows ({r.period})") +
                   (" (e.g. " + ", ".join(f"`{x}`" for x in v["invalid_examples"]) + ")" if v["invalid_examples"] else ""))
    unusual += (f". {len(blank)} other types never carry a descriptor" + (
        ", e.g. " + _bullets_from(blank, lambda r: f"`{r.complaint_type}` ({r.n:,}, {r.main_agency})") if len(blank) else "")
        + ".")
    lg = v["rt"]["legacy"]
    n_legacy = int(((lg.bridged_to == "") & (lg.n_successors >= 2)).sum())
    lines = [
        "# Q3. Complaint categories",
        "",
        f"*Drafted by `nyc311/q03_complaint_types.py` from {total:,} rows. Numbers are computed; write the prose from "
        "these bullets.*",
        "",
        "## Q3a. Consistency issues in complaint_type",
        "",
        f"- **Scale.** {v['n_raw']} raw complaint_type values; after review they map to {v['n_canon']} canonical types "
        f"in {v['n_cat']} categories. The descriptor field ({v['n_descriptors']:,} values) carries the finer detail.",
        "- **One name, several spellings.** " + _bullets_from(
            v["variants"], lambda r: f"{r.spellings}: {r.decision}") + ".",
        *_renames_bullets(v),
        f"- **Two naming systems.** {int(caps.sum())} types are ALL-CAPS codes ({int(valid.n[caps].sum()):,} rows, "
        f"{_pct(valid.n[caps].sum(), total)} of all rows): " + _bullets_from(
            v["caps_tab"], lambda r: f"{r.agency} {r.caps_types}, e.g. {r.examples}; {r.agency} also has "
                                     f"{r.other_types} Title Case type(s)") +
        ". Every other type is Title Case, so HPD's `PLUMBING`/`ELECTRIC` sit next to DOB's `Plumbing`/`Electrical`.",
        f"- **Look-alike names that mean different things.** {len(v['pairs'])} pairs of names look alike (same word, "
        f"similar spelling, or one name inside another across agencies); {n_disjoint} of them share no descriptor at "
        "all, so they are different work. Examples (similarity after normalizing, descriptors shared): " + _bullets_from(
            v["pairs"], lambda r: f"{r.type_a} vs {r.type_b}: {r.similarity:.2f}, {r.descriptors_shared}", k=6) + ".",
        "- **Vague or generic names** whose meaning is only in the descriptor: " + _bullets_from(
            v["vague"], lambda r: f"`{r.complaint_type}` ({r.n:,}, {r.main_agency}; top descriptor {r.top_descriptor}) "
                                  f"-> {r.canonical_type}", k=6) + ".",
        unusual,
        f"- **Not complaints.** {len(req)} types, {int(req.n.sum()):,} rows ({_pct(req.n.sum(), total)}) are requests "
        f"for information, materials, documents or services, or compliments; {len(info_cat)} of them "
        f"({int(info_cat.n.sum()):,} rows) are pure information requests. Largest: " + _bullets_from(
            req, lambda r: f"{r.complaint_type} ({r.n:,}, {r.main_agency})") + ".",
        "- **One type, several agencies:** " + _bullets_from(v["multi"], lambda r: f"{r.complaint_type} ({r.agencies})") + ".",
        f"- **Rare entries.** {len(rare)} types have fewer than {RARE_BELOW} rows ({int(rare.n.sum()):,} rows in total); "
        f"{len(once)} appear once ({_names(once.complaint_type)}).",
        "- **Irregular spacing/punctuation:** " + _bullets_from(
            v["spacing"], lambda r: f"`{r.complaint_type}` -> {r.canonical_type}") + ".",
        "",
        "## Q3b. Cleaning strategy",
        "",
        "- **Never overwrite the raw value.** Add `canonical_type`, `category`, `origin` and `is_info_request` from a "
        "reviewed lookup, `nyc311/complaint_type_map.csv`: one row per raw spelling, with a note for every merge, rename "
        "and judgment call. The lookup is data, so analysts can review it without reading code.",
        f"- **Three levels keep the granularity operations need:** {v['n_cat']} categories > {v['n_canon']} canonical "
        f"types > {v['n_descriptors']:,} descriptors (unchanged). Categories by rows: " + "; ".join(
            f"{r.category} {r.rows:,} ({r.pct_rows})" for r in v["cat"].itertuples(index=False)) + ".",
        f"- **Merge only true duplicates and true renames** ({v['n_merged_raw']} spellings -> {v['n_merged_canon']} "
        "types here). Check agency and descriptors before merging: look-alikes handled by different agencies with no "
        f"shared descriptors stay apart. Rename only to clarify ({v['n_renamed']} types: HPD's ALL-CAPS codes become a "
        "`Housing - ...` family; vague names take their meaning from the descriptor).",
        "- **Make time series continuous without hiding reorganizations.** When one name replaced another for the same "
        "work (same agency, descriptors carry on), both spellings share one canonical type. When an agency split a type "
        "across several new ones, the old type keeps its own canonical type (continuous at the category level) and "
        "`outputs/agg/q03_legacy_types.csv` shows where each of its descriptors went"
        + (f" ({n_legacy} split types here)." if not v["rt"]["too_short"] else "."),
        "- **Adapts to new data without code changes.** `taxonomy.classify()` resolves each distinct raw value by exact "
        "match, then by normalized match (case, spacing, dashes/slashes, plural), then by a close fuzzy match (flagged "
        "for review), else marks it `Unreviewed` (flagged). Flagged types land in `outputs/agg/q03_unreviewed.csv` with "
        "row counts, largest first; a reviewer adds one CSV row each, pointing a renamed type at its existing canonical "
        f"type, and the next run lists the handoff in `outputs/agg/q03_renames.csv`. On this data {len(v['review'])} of "
        f"{v['n_raw']} types need review.",
        f"- **Separate complaints from requests and junk** with `is_info_request` and the `{taxonomy.INVALID}` category, "
        f"so complaint volumes and response-time KPIs exclude the {_pct(req.n.sum(), total)} of rows that are requests "
        f"and the {int(invalid.rows.sum()):,} rows that are not service requests at all.",
        f"- **Origin, for Q9:** {by_origin}. Judgment calls (sidewalks, rodents, snow, water meters, homeless "
        "encampments, DHS payments) are recorded in the map's notes.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q03.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
