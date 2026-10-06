"""Complaint-type taxonomy (Q3): normalize(), the reviewed map, classify(), and the q03 stage."""
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from nyc311 import taxonomy

REPO = Path(__file__).resolve().parents[1]
COUNTS = REPO / "outputs" / "agg" / "complaint_type_counts.csv"
RAW_COMBOS = REPO / "outputs" / "agg" / "complaint_types_raw.csv"
RAW_PARQUET = REPO / "work" / "raw"
ORIGINS = {"city", "private", "natural", "other"}


@pytest.fixture(scope="module")
def raw_types():
    return pd.read_csv(COUNTS, keep_default_na=False).complaint_type.tolist()


@pytest.fixture(scope="module")
def tmap():
    return taxonomy.load_map()


def _one(raw):
    out = taxonomy.classify([raw])
    assert len(out) == 1
    return out.iloc[0]


# ---------------------------------------------------------------- normalize

@pytest.mark.parametrize("raw, expected", [
    ("PLUMBING", "plumbing"),
    ("  Plumbing  ", "plumbing"),
    ("PLUMBINGS", "plumbing"),
    ("Derelict Vehicles", "derelict vehicle"),
    ("Derelict   Vehicle", "derelict vehicle"),
    ("Noise\t-\nResidential", "noise-residential"),
    ("DHS Advantage -Landlord/Broker", "dhs advantage-landlord-broker"),
    ("DHS Advantage - Landlord / Broker", "dhs advantage-landlord-broker"),
    ("PAINT - PLASTER", "paint-plaster"),
    ("PAINT/PLASTER", "paint-plaster"),
    ("Paint – Plaster", "paint-plaster"),
    ("Overgrown Tree/Branches", "overgrown tree-branch"),
    ("Miscellaneous Categories", "miscellaneous category"),
    ("Missed Collection (All Materials)", "missed collection (all materials)"),
    ("Alzheimer's Care", "alzheimer's care"),
    ("Parking Tickets Requests", "parking tickets request"),
])
def test_normalize(raw, expected):
    assert taxonomy.normalize(raw) == expected


@pytest.mark.parametrize("word", ["Status", "Gas", "Glass", "Bus", "Asbestos", "Analysis"])
def test_normalize_keeps_words_that_only_look_plural(word):
    assert taxonomy.normalize(word) == word.lower()
    assert taxonomy.normalize(f"DOF Parking - Request {word}") == f"dof parking-request {word.lower()}"


def test_normalize_is_deterministic_and_idempotent(raw_types):
    for t in raw_types:
        once = taxonomy.normalize(t)
        assert once == taxonomy.normalize(t)
        assert taxonomy.normalize(once) == once


def test_normalize_only_joins_the_known_spelling_variants(raw_types):
    """Over-eager normalization would silently merge different types; on the 174 raw types it must only
    join the two groups that really are spelling variants of one name."""
    keys = pd.Series(raw_types).groupby(pd.Series(raw_types).map(taxonomy.normalize)).agg(set)
    shared = {k: v for k, v in keys.items() if len(v) > 1}
    assert shared == {"plumbing": {"PLUMBING", "Plumbing"},
                      "derelict vehicle": {"Derelict Vehicle", "Derelict Vehicles"}}


def test_normalize_handles_none():
    assert taxonomy.normalize(None) == ""


# ---------------------------------------------------------------- the reviewed map

def test_map_has_one_row_per_raw_type_seen(tmap, raw_types):
    assert len(raw_types) == 174
    assert tmap.complaint_type.is_unique
    assert set(tmap.complaint_type) == set(raw_types)


def test_map_values_are_valid(tmap):
    assert list(tmap.columns[:6]) == ["complaint_type", "canonical_type", "category", "origin",
                                      "is_info_request", "notes"]
    assert set(tmap.origin) <= ORIGINS
    assert tmap.is_info_request.dtype == bool
    for col in ["canonical_type", "category"]:
        assert (tmap[col].str.strip() != "").all()
        assert (tmap[col] == tmap[col].str.strip()).all()
    assert 10 <= tmap.category.nunique() <= 16


def test_each_canonical_type_sits_in_one_category_with_one_origin(tmap):
    per = tmap.groupby("canonical_type")[["category", "origin", "is_info_request"]].nunique()
    assert (per == 1).all().all(), per[(per > 1).any(axis=1)]


def test_requests_are_not_attributed_to_an_origin(tmap):
    assert (tmap.loc[tmap.is_info_request, "origin"] == "other").all()
    assert tmap.loc[tmap.category == "Information requests", "is_info_request"].all()


def test_every_merge_and_rename_is_explained(tmap):
    merged = tmap.canonical_type.duplicated(keep=False)
    renamed = tmap.complaint_type.map(taxonomy.normalize) != tmap.canonical_type.map(taxonomy.normalize)
    unexplained = tmap[(merged | renamed) & (tmap.notes.str.strip() == "")]
    assert unexplained.empty, unexplained.complaint_type.tolist()


def test_load_map_rejects_a_bad_origin(tmp_path):
    bad = taxonomy.load_map()
    bad.loc[0, "origin"] = "City"
    p = tmp_path / "map.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="origin"):
        taxonomy.load_map(p)


# ---------------------------------------------------------------- classify

def test_classify_output_shape_and_types():
    out = taxonomy.classify(["Snow", "Snow", "Rodent", "Snow"])
    assert list(out.columns) == ["complaint_type", "canonical_type", "category", "origin",
                                 "is_info_request", "needs_review"]
    assert out.complaint_type.tolist() == ["Snow", "Rodent"]
    assert out.is_info_request.dtype == bool and out.needs_review.dtype == bool


def test_classify_returns_one_row_per_distinct_input(raw_types):
    """The cleaning stage joins on complaint_type, so a duplicated key would duplicate data rows."""
    messy = pd.Series(raw_types * 3 + ["plumbing ", "plumbing ", "zzqx", "zzqx", None, float("nan"), ""])
    out = taxonomy.classify(messy)
    assert len(out) == 174 + 4  # + 'plumbing ', 'zzqx', '' and one null (None and NaN are the same null)
    assert not out.complaint_type.dropna().duplicated().any()
    assert out.complaint_type.isna().sum() == 1


def test_classify_accepts_any_iterable():
    gen = (t for t in ["Snow", "Rodent"])
    assert len(taxonomy.classify(gen)) == 2
    assert len(taxonomy.classify(pd.Series(["Snow", "Rodent"]))) == 2


def test_every_seen_raw_type_classifies_without_review(raw_types, tmap):
    out = taxonomy.classify(raw_types)
    assert len(out) == 174
    assert not out.needs_review.any(), out[out.needs_review].complaint_type.tolist()
    merged = out.merge(tmap, on="complaint_type", suffixes=("", "_map"))
    for col in ["canonical_type", "category", "origin", "is_info_request"]:
        assert (merged[col] == merged[f"{col}_map"]).all(), col
    assert set(out.origin) <= ORIGINS


def test_plumbing_spellings_stay_apart_because_descriptors_differ():
    """PLUMBING (HPD) is apartment maintenance: leaks, sinks, radiators. Plumbing (DOB) is building-code
    enforcement: illegal work, no permit, sprinklers, gas piping. Different agency, no shared descriptor,
    so they are different types that happen to share a word; merging them would mix two workloads."""
    raw = pd.read_csv(RAW_COMBOS, keep_default_na=False)
    hpd = raw[raw.complaint_type == "PLUMBING"]
    dob = raw[raw.complaint_type == "Plumbing"]
    assert set(hpd.agency) == {"HPD"} and set(dob.agency) == {"DOB"}
    assert not set(hpd.descriptor) & set(dob.descriptor)
    out = taxonomy.classify(["PLUMBING", "Plumbing"]).set_index("complaint_type")
    assert out.loc["PLUMBING", "canonical_type"] != out.loc["Plumbing", "canonical_type"]
    assert out.loc["PLUMBING", "category"] == "Housing conditions"
    assert out.loc["Plumbing", "category"] == "Buildings & construction"


def test_derelict_vehicle_spellings_merge():
    out = taxonomy.classify(["Derelict Vehicle", "Derelict Vehicles"])
    assert out.canonical_type.nunique() == 1
    assert out.canonical_type.iloc[0] == "Derelict Vehicle"


@pytest.mark.parametrize("variant", ["plumbing ", "PLUMBINGS", " Plumbing\t"])
def test_ambiguous_new_spellings_get_the_dominant_meaning_but_are_flagged(variant):
    """'plumbing' normalizes to both PLUMBING (HPD, 11k rows) and Plumbing (DOB, 306 rows), which mean different
    things; a new spelling gets the higher-volume meaning but is flagged so a person checks the agency."""
    row = _one(variant)
    assert row.needs_review
    assert row.canonical_type == _one("PLUMBING").canonical_type


@pytest.mark.parametrize("variant, seen", [
    ("derelict  vehicles ", "Derelict Vehicle"),
    ("PAINT/PLASTER", "PAINT - PLASTER"),
    ("DHS Advantage - Landlord/Broker", "DHS Advantage -Landlord/Broker"),
    ("noise - residential", "Noise - Residential"),
])
def test_unseen_spellings_resolve_without_review(variant, seen):
    row = _one(variant)
    assert not row.needs_review
    assert row.canonical_type == _one(seen).canonical_type


def test_close_misspelling_is_matched_but_flagged():
    row = _one("Street Light Conditon")
    assert row.needs_review
    assert row.canonical_type == "Street Light Condition"
    assert row.category == _one("Street Light Condition").category


def test_misspelling_between_complaint_and_compliment_is_flagged():
    """'Taxi Complaint' and 'Taxi Compliment' are one letter apart with opposite meanings, which is why a
    fuzzy match is never trusted without review."""
    row = _one("Taxi Complient")
    assert row.needs_review
    assert row.canonical_type in {"Taxi Complaint", "Taxi Compliment"}


def test_nonsense_goes_to_review():
    row = _one("zzqx  blorp-widget")
    assert row.needs_review
    assert row.category == "Unreviewed"
    assert row.origin == "other"
    assert row.canonical_type == "Zzqx Blorp-Widget"
    assert not row.is_info_request


def test_null_and_empty_are_unknown():
    out = taxonomy.classify([None, float("nan"), "", "   ", None])
    assert len(out) == 3
    assert (out.canonical_type == "Unknown").all()
    assert out.needs_review.all()
    assert set(out.origin) == {"other"}


def test_information_requests_are_flagged():
    out = taxonomy.classify(["Literature Request", "SCRIE", "HPD Literature Request", "Taxi Compliment",
                             "Noise - Residential", "HEATING"]).set_index("complaint_type")
    assert out.loc[["Literature Request", "SCRIE", "HPD Literature Request", "Taxi Compliment"],
                   "is_info_request"].all()
    assert not out.loc[["Noise - Residential", "HEATING"], "is_info_request"].any()


# ---------------------------------------------------------------- q03 stage

@pytest.mark.skipif(not any(RAW_PARQUET.glob("*.parquet")), reason="run the pipeline first (work/raw)")
def test_q03_run_writes_its_outputs(tmp_path, capsys):
    from nyc311 import q03_complaint_types
    from nyc311.config import Config

    cfg = Config.from_args(["--out", str(tmp_path), "--work", str(REPO / "work")])
    q03_complaint_types.run(cfg)

    applied = pd.read_csv(tmp_path / "agg" / "q03_type_map_applied.csv", keep_default_na=False)
    assert list(applied.columns) == ["complaint_type", "n", "canonical_type", "category", "origin",
                                     "is_info_request", "needs_review"]
    n_rows = duckdb.sql(f"select count(*) from read_parquet('{(RAW_PARQUET / '*.parquet').as_posix()}')").fetchone()[0]
    assert applied.n.sum() == n_rows
    assert len(applied) == 174

    unreviewed = pd.read_csv(tmp_path / "agg" / "q03_unreviewed.csv")
    assert unreviewed.empty

    md = (tmp_path / "answers" / "q03.md").read_text()
    assert "Q3a" in md and "Q3b" in md
    assert "174" in md and f"{n_rows:,}" in md

    printed = capsys.readouterr().out
    assert "BEGIN q03_complaint_types" in printed
    report = (tmp_path / "reports" / "q03_complaint_types.txt").read_text().splitlines()
    assert len(report) <= 152
