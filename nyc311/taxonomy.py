"""Complaint-type taxonomy (Q3): a reviewed lookup from raw complaint_type to canonical type, category and origin.

Every raw spelling seen so far has one reviewed row in complaint_type_map.csv. classify() resolves any raw
string in four steps, so new data needs no code change:
  1. exact match on a reviewed spelling                      -> needs_review False
  2. match after normalize() (case, spacing, dashes, plural) -> needs_review False
  3. close fuzzy match (difflib, cutoff 0.85)                -> copies that row, needs_review True
  4. nothing close                                           -> category 'Unreviewed', needs_review True
A reviewer resolves flagged types by adding one row to the CSV.
"""
import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

MAP_PATH = Path(__file__).with_name("complaint_type_map.csv")
MAP_COLUMNS = ["complaint_type", "canonical_type", "category", "origin", "is_info_request", "notes"]
OUT_COLUMNS = ["complaint_type", "canonical_type", "category", "origin", "is_info_request", "needs_review"]
ORIGINS = ("city", "private", "natural", "other")
UNREVIEWED = "Unreviewed"
FUZZY_CUTOFF = 0.85

# Unicode hyphens, en/em dashes and the minus sign all become "-".
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_SEPARATOR = re.compile(r"\s*(?:[-/]\s*)+")
_LAST_WORD = re.compile(r"[a-z]+$")
# Endings of words that only look plural: Status, Bus, Analysis, Gas, Asbestos, Glass.
_NOT_PLURAL = ("ss", "us", "is", "as", "os")


def _singular(word):
    if len(word) <= 3 or not word.endswith("s") or word.endswith(_NOT_PLURAL):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes", "zes")):
        return word[:-2]
    return word[:-1]


def normalize(s):
    """Matching key: lowercase, trimmed, single spaces, '-' and '/' as one unspaced separator, last word singular."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().translate(_DASHES)
    s = " ".join(s.split())
    s = _SEPARATOR.sub("-", s)
    s = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", s))
    return _LAST_WORD.sub(lambda m: _singular(m.group(0)), s)


def load_map(path=MAP_PATH):
    """The reviewed lookup, validated: one row per raw spelling, origin in ORIGINS, is_info_request as bool."""
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in MAP_COLUMNS if c not in m.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for c in MAP_COLUMNS:
        m[c] = m[c].astype(str).str.strip()
    dup = m.complaint_type[m.complaint_type.duplicated()].tolist()
    if dup:
        raise ValueError(f"{path}: complaint_type listed twice: {dup}")
    bad = m.loc[~m.origin.isin(ORIGINS), "complaint_type"].tolist()
    if bad:
        raise ValueError(f"{path}: origin must be one of {ORIGINS}; check {bad}")
    flag = m.is_info_request.str.lower()
    bad = m.loc[~flag.isin(["true", "false"]), "complaint_type"].tolist()
    if bad:
        raise ValueError(f"{path}: is_info_request must be true/false; check {bad}")
    m["is_info_request"] = (flag == "true").astype(bool)
    return m


def _distinct(values):
    """Distinct raw values in first-seen order; every kind of null (None, NaN, NA) counts as one value, None."""
    seen, out = set(), []
    for v in values:
        if not isinstance(v, str) and (v is None or pd.isna(v)):
            v = None
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _title(s):
    """Readable name for an unreviewed type: single spaces, 'a - b' and 'a/b' spacing, Title Case."""
    s = " ".join(str(s).translate(_DASHES).split())
    s = re.sub(r"\s+-\s*|\s*-\s+", " - ", re.sub(r"\s*/\s*", "/", s))
    return re.sub(r"(^|[\s/(-])([a-z])", lambda m: m.group(1) + m.group(2).upper(), s.lower())


def classify(raw_types):
    """One row per distinct raw complaint type: canonical_type, category, origin, is_info_request, needs_review."""
    m = load_map()
    values = list(m[["canonical_type", "category", "origin", "is_info_request"]].itertuples(index=False, name=None))
    exact = dict(zip(m.complaint_type, values))
    by_key, ambiguous = {}, set()
    for raw, vals in zip(m.complaint_type, values):
        key = normalize(raw)
        if key in by_key and by_key[key][0] != vals[0]:
            ambiguous.add(key)
        by_key.setdefault(key, vals)  # first (higher-volume) row wins; ambiguous keys are flagged below
    keys = list(by_key)

    rows = []
    for raw in _distinct(raw_types):
        text = "" if raw is None else str(raw)
        key = normalize(text)
        if not key:
            rows.append((raw, "Unknown", UNREVIEWED, "other", False, True))
        elif text in exact:
            rows.append((raw, *exact[text], False))
        elif key in by_key:
            rows.append((raw, *by_key[key], key in ambiguous))
        else:
            close = difflib.get_close_matches(key, keys, n=1, cutoff=FUZZY_CUTOFF)
            if close:
                rows.append((raw, *by_key[close[0]], True))
            else:
                rows.append((raw, _title(text), UNREVIEWED, "other", False, True))
    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    return out.astype({"is_info_request": bool, "needs_review": bool})
