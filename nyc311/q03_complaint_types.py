"""Q3: consistency issues in complaint_type (Q3a) and the cleaning strategy (Q3b).

Applies nyc311.taxonomy to every raw complaint_type in the raw Parquet, writes the applied map and the review
queue, prints the evidence for each kind of issue, and drafts outputs/answers/q03.md. Every number is computed.
"""
from __future__ import annotations

import difflib
import re
from itertools import combinations

import pandas as pd

from nyc311 import ingest, taxonomy
from nyc311.report import Report

STAGE = "q03_complaint_types"
APPLIED_COLUMNS = ["complaint_type", "n", "canonical_type", "category", "origin", "is_info_request", "needs_review"]
RARE_BELOW = 10            # a type with fewer rows than this is "rare"
LOOKALIKE_RATIO = 0.85     # names this similar (difflib, after normalize) count as look-alikes...
WORD_RATIO = 0.8           # ...if the words that differ are also this similar
GENERIC_WORDS = {"other", "miscellaneous", "misc", "general"}
CAPS_WORD_MIN = 6          # an ALL-CAPS name with a word this long is a legacy code, not an acronym
EXAMPLES = 4
_IRREGULAR_SPACING = re.compile(r"\S-\s|\s-\S|\s/|/\s")
_TOKENS = re.compile(r"[a-z0-9]+")


def _none_if_null(s):
    return s.astype(object).where(s.notna(), None)


def type_profile(con, src):
    """One row per raw complaint_type: rows, agencies, descriptor set and top descriptor."""
    combos = con.sql(f"select complaint_type, agency, descriptor, count(*)::bigint n from {src} group by all").df()
    for c in ["complaint_type", "agency", "descriptor"]:
        combos[c] = _none_if_null(combos[c])
    prof = {}
    for r in combos.itertuples(index=False):
        p = prof.setdefault(r.complaint_type, {"n": 0, "agencies": {}, "descriptors": {}})
        p["n"] += int(r.n)
        p["agencies"][r.agency] = p["agencies"].get(r.agency, 0) + int(r.n)
        p["descriptors"][r.descriptor] = p["descriptors"].get(r.descriptor, 0) + int(r.n)
    rows = []
    for ct, p in prof.items():
        ag = sorted(p["agencies"].items(), key=lambda kv: -kv[1])
        de = sorted(p["descriptors"].items(), key=lambda kv: -kv[1])
        rows.append({"complaint_type": ct, "n": p["n"], "main_agency": ag[0][0], "n_agencies": len(ag),
                     "agencies": " / ".join(f"{a} {v:,}" for a, v in ag),
                     "descriptors": frozenset(d for d, _ in de),
                     "top_descriptor": f"{de[0][0]} ({de[0][1] / p['n']:.0%})"})
    return pd.DataFrame(rows), combos.descriptor.nunique()


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
    src = f"read_parquet('{ingest.raw_glob(cfg)}')"
    prof, n_descriptors = type_profile(con, src)
    t = prof.merge(taxonomy.classify(prof.complaint_type), on="complaint_type", how="left")
    t = t.sort_values("n", ascending=False).reset_index(drop=True)

    agg = cfg.out_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    t[APPLIED_COLUMNS].to_csv(agg / "q03_type_map_applied.csv", index=False)
    review = t[t.needs_review]
    review[APPLIED_COLUMNS].to_csv(agg / "q03_unreviewed.csv", index=False)

    total, n_raw = int(t.n.sum()), len(t)
    n_canon, n_cat = t.canonical_type.nunique(), t.category.nunique()
    variants = spelling_variants(t)
    caps_tab, caps = all_caps_by_agency(t)
    merged = t.canonical_type.duplicated(keep=False)
    pairs = lookalike_pairs(t)
    vague, renamed = vague_names(t, caps, merged)
    req = t[t.is_info_request]
    info_cat = req[req.category == "Information requests"]
    multi = t[t.n_agencies > 1][["complaint_type", "agencies"]]
    rare = t[t.n < RARE_BELOW].sort_values("n")
    once = rare[rare.n == 1]
    spacing = t[t.complaint_type.map(lambda s: isinstance(s, str) and bool(_IRREGULAR_SPACING.search(s)))]
    cat, org = summary_tables(t, total)
    n_merged_raw, n_merged_canon = int(merged.sum()), t.canonical_type[merged].nunique()
    n_renamed = int((renamed & ~merged).sum())

    rep = Report(STAGE, cfg.out_dir)
    rep.add(f"Q3a. complaint_type consistency on {total:,} rows",
            f"raw complaint_type values: {n_raw} -> canonical types: {n_canon} -> categories: {n_cat}",
            f"types needing review: {len(review)} ({int(review.n.sum()):,} rows) | descriptor values kept as the "
            f"finest level: {n_descriptors:,}")
    rep.table(variants, max_rows=EXAMPLES, title="1. One name, several spellings (case / spacing / plural)")
    rep.table(caps_tab, max_rows=EXAMPLES, title=f"2. Two naming systems: ALL-CAPS codes hold {int(t.n[caps].sum()):,} "
              f"rows ({_pct(t.n[caps].sum(), total)}); every other type is Title Case")
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
            f"{len(once)} appear once: {', '.join(once.complaint_type.astype(str))}")
    rep.table(spacing[["complaint_type", "canonical_type"]], max_rows=EXAMPLES, title="8. Irregular spacing around - and /")
    rep.table(cat, max_rows=20, title="Categories after cleaning")
    rep.table(org, max_rows=5, title="Origin (for Q9); pct_of_complaints excludes requests")
    rep.add("", "Q3b. Strategy",
            "- Keep raw complaint_type; add canonical_type/category/origin/is_info_request from a reviewed lookup",
            "  (nyc311/complaint_type_map.csv, one row per raw spelling, notes for every merge/rename/judgment).",
            f"- Merge only true duplicates ({n_merged_raw} spellings -> {n_merged_canon}); rename vague/ALL-CAPS names "
            f"({n_renamed}); keep look-alikes apart.",
            f"- Hierarchy keeps granularity: {n_cat} categories > {n_canon} canonical types > {n_descriptors:,} descriptors.",
            "- New data: exact -> normalized (case/space/dash/plural) -> fuzzy (flagged) -> unreviewed (flagged);",
            "  flagged types go to agg/q03_unreviewed.csv with row counts; a reviewer adds one CSV row, no code change.")
    rep.emit()

    _write_answer(cfg, locals())
    return t[APPLIED_COLUMNS]


def _bullets_from(df, fmt, k=EXAMPLES):
    return "; ".join(fmt(r) for r in df.head(k).itertuples(index=False))


def _write_answer(cfg, v):
    t, total, req, info_cat, rare, once, org = v["t"], v["total"], v["req"], v["info_cat"], v["rare"], v["once"], v["org"]
    caps = v["caps"]
    n_disjoint = int(v["pairs"].descriptors_shared.str.startswith("0 of").sum())
    by_origin = "; ".join(f"{r.origin} {r.rows:,} rows ({r.pct_rows} of all, {r.pct_of_complaints} of complaints)"
                          for r in org.itertuples(index=False))
    lines = [
        "# Q3. Complaint categories",
        "",
        f"_Drafted by `nyc311/q03_complaint_types.py` from {total:,} rows. Numbers are computed; write the prose from "
        "these bullets._",
        "",
        "## Q3a. Consistency issues in complaint_type",
        "",
        f"- **Scale.** {v['n_raw']} raw complaint_type values; after review they map to {v['n_canon']} canonical types "
        f"in {v['n_cat']} categories. The descriptor field ({v['n_descriptors']:,} values) carries the finer detail.",
        "- **One name, several spellings.** " + _bullets_from(
            v["variants"], lambda r: f"{r.spellings}: {r.decision}") + ".",
        f"- **Two naming systems.** {int(caps.sum())} types are ALL-CAPS codes ({int(t.n[caps].sum()):,} rows, "
        f"{_pct(t.n[caps].sum(), total)} of all rows): " + _bullets_from(
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
        f"- **Not complaints.** {len(req)} types, {int(req.n.sum()):,} rows ({_pct(req.n.sum(), total)}) are requests "
        f"for information, materials, documents or services, or compliments; {len(info_cat)} of them "
        f"({int(info_cat.n.sum()):,} rows) are pure information requests. Largest: " + _bullets_from(
            req, lambda r: f"{r.complaint_type} ({r.n:,}, {r.main_agency})") + ".",
        "- **One type, several agencies:** " + _bullets_from(v["multi"], lambda r: f"{r.complaint_type} ({r.agencies})") + ".",
        f"- **Rare entries.** {len(rare)} types have fewer than {RARE_BELOW} rows ({int(rare.n.sum()):,} rows in total); "
        f"{len(once)} appear once ({', '.join(once.complaint_type.astype(str))}).",
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
        f"- **Merge only true duplicates** ({v['n_merged_raw']} spellings -> {v['n_merged_canon']} type). Check agency "
        "and descriptors before merging: look-alikes handled by different agencies with no shared descriptors stay "
        f"apart. Rename only to clarify ({v['n_renamed']} types: HPD's ALL-CAPS codes become a `Housing - ...` family; "
        "vague names take their meaning from the descriptor).",
        "- **Adapts to new data without code changes.** `taxonomy.classify()` resolves each distinct raw value by exact "
        "match, then by normalized match (case, spacing, dashes/slashes, plural), then by a close fuzzy match (flagged "
        "for review), else marks it `Unreviewed` (flagged). Flagged types land in `outputs/agg/q03_unreviewed.csv` with "
        f"row counts, largest first; a reviewer adds one CSV row each. On this data {len(v['review'])} of "
        f"{v['n_raw']} types need review.",
        f"- **Separate complaints from requests** with `is_info_request`, so complaint volumes and response-time KPIs "
        f"exclude the {_pct(req.n.sum(), total)} of rows that are not complaints.",
        f"- **Origin, for Q9:** {by_origin}. Judgment calls (sidewalks, rodents, water meters, homeless encampments, "
        "DHS payments) are recorded in the map's notes.",
        "",
    ]
    path = cfg.out_dir / "answers" / "q03.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
