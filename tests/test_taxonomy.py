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
# Aggregates from the full cluster run (2,824 files, Jan 2010 - Nov 2021); the map was built against them.
CLUSTER_AGG = REPO / "cluster_outputs" / "full" / "outputs" / "agg"
CLUSTER_COUNTS = CLUSTER_AGG / "complaint_type_counts.csv"
CLUSTER_COMBOS = CLUSTER_AGG / "complaint_types_raw.csv"
NO_CLUSTER = (f"cluster aggregates not found ({CLUSTER_COUNTS.relative_to(REPO)}); unzip the full run's "
              "outputs_bundle.zip into cluster_outputs/full/ to run the full-data taxonomy tests")


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
    """Every sample spelling is reviewed; the map also holds the spellings seen only in the full data."""
    assert len(raw_types) == 174
    assert tmap.complaint_type.is_unique
    assert set(raw_types) <= set(tmap.complaint_type)


def test_map_values_are_valid(tmap):
    assert list(tmap.columns[:6]) == ["complaint_type", "canonical_type", "category", "origin",
                                      "is_info_request", "notes"]
    assert set(tmap.origin) <= ORIGINS
    assert tmap.is_info_request.dtype == bool
    for col in ["canonical_type", "category"]:
        assert (tmap[col].str.strip() != "").all()
        assert (tmap[col] == tmap[col].str.strip()).all()
    assert 10 <= tmap.category.nunique() <= 17  # 16 subject categories + 'Invalid entries'


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


# ---------------------------------------------------------------- the full cluster data (478 raw types)

@pytest.fixture(scope="module")
def cluster_counts():
    if not CLUSTER_COUNTS.exists():
        pytest.skip(NO_CLUSTER)
    return pd.read_csv(CLUSTER_COUNTS, keep_default_na=False)


@pytest.fixture(scope="module")
def cluster_combos():
    if not CLUSTER_COMBOS.exists():
        pytest.skip(NO_CLUSTER)
    c = pd.read_csv(CLUSTER_COMBOS, keep_default_na=False)
    c["first"] = c.first_seen.str[:10]
    c["last"] = c.last_seen.str[:10]
    return c


def _span(combos, raw):
    d = combos[combos.complaint_type == raw]
    assert not d.empty, raw
    return d["first"].min(), d["last"].max(), int(d.n.sum())


def _descriptors(combos, raw):
    return set(combos.loc[combos.complaint_type == raw, "descriptor"]) - {""}


def test_map_has_one_row_per_cluster_type_sorted_by_volume(tmap, cluster_counts):
    """478 raw spellings in 28.24M rows; the map holds exactly these, largest first, because classify()
    resolves a new spelling that normalizes to two reviewed names in favour of the first (higher-volume) row."""
    assert len(cluster_counts) == 478
    assert set(tmap.complaint_type) == set(cluster_counts.complaint_type)
    n = tmap.complaint_type.map(cluster_counts.set_index("complaint_type").n)
    assert n.is_monotonic_decreasing


def test_every_cluster_type_classifies_without_review(tmap, cluster_counts):
    out = taxonomy.classify(cluster_counts.complaint_type)
    assert len(out) == 478 and out.complaint_type.is_unique
    assert not out.needs_review.any(), out[out.needs_review].complaint_type.tolist()
    assert set(out.origin) <= ORIGINS
    merged = out.merge(tmap, on="complaint_type", suffixes=("", "_map"))
    for col in ["canonical_type", "category", "origin", "is_info_request"]:
        assert (merged[col] == merged[f"{col}_map"]).all(), col


def test_classify_on_cluster_types_returns_one_row_each(cluster_counts):
    """clean.py joins on complaint_type over 28M rows; one duplicated key would duplicate data rows."""
    types = cluster_counts.complaint_type.tolist()
    out = taxonomy.classify(types[::-1] + types)
    assert len(out) == 478
    assert out.complaint_type.is_unique
    assert set(out.complaint_type) == set(types)


# HPD's 2014-03-11 reorganization. Evidence (complaint_types_raw.csv): HEATING (descriptors HEAT, HEAT-PLANT)
# last seen 2014-03-10, HEAT/HOT WATER first seen 2014-03-11; PAINT - PLASTER (CEILING, WALLS) last seen
# 2014-03-04, its rows continue as PAINT/PLASTER (CEILING, WALL); NONCONST (VERMIN, RUBBISH, smoke/CO
# detectors) last seen 2014-03-10, and its vermin and rubbish rows (most of the type) continue as
# UNSANITARY CONDITION (PESTS, GARBAGE/RECYCLING STORAGE) from 2014-03-11. The Title Case spellings
# (Unsanitary Condition, General, Outside Building) are HPD rows from 2018-19 with blank descriptors.
HPD_FAMILIES = [
    ("HEATING", "HEAT/HOT WATER"),
    ("PAINT - PLASTER", "PAINT/PLASTER"),
    ("NONCONST", "UNSANITARY CONDITION", "Unsanitary Condition"),
    ("GENERAL", "General"),
    ("OUTSIDE BUILDING", "Outside Building"),
    ("STRUCTURAL", "WATER LEAK"),
    ("APPLIANCE", "Appliance"),
    ("ELECTRIC", "Electric"),
]


@pytest.mark.parametrize("family", HPD_FAMILIES, ids=lambda f: f[0])
def test_each_renamed_hpd_family_maps_to_one_canonical_type(family, cluster_combos):
    for raw in family:
        assert "HPD" in set(cluster_combos.loc[cluster_combos.complaint_type == raw, "agency"]), raw
    out = taxonomy.classify(list(family))
    assert out.canonical_type.nunique() == 1, out[["complaint_type", "canonical_type"]].to_string()
    assert out.category.iloc[0] in {"Housing conditions", "Heat & hot water"}
    assert not out.needs_review.any()


@pytest.mark.parametrize("old, new", [("HEATING", "HEAT/HOT WATER"), ("NONCONST", "UNSANITARY CONDITION"),
                                      ("PAINT - PLASTER", "PAINT/PLASTER")])
def test_hpd_renames_hand_over_in_march_2014(old, new, cluster_combos):
    old_first, old_last, _ = _span(cluster_combos, old)
    new_first, new_last, _ = _span(cluster_combos, new)
    assert old_last.startswith("2014-03") and new_last > old_last and old_first < new_first


def test_general_construction_split_so_it_stays_a_legacy_type(cluster_combos):
    """GENERAL CONSTRUCTION (last seen 2014-03-06) was not renamed but split by descriptor: MOLD went to
    UNSANITARY CONDITION, FLOOR and STAIRS to FLOORING/STAIRS, DOORS to DOOR/WINDOW, CABINETS and VENTILATION
    SYSTEM to GENERAL, FIRE-ESCAPE to SAFETY. No successor takes most of its rows, so it keeps its own canonical
    type; the series is continuous at the category level (Housing conditions)."""
    gc = cluster_combos[cluster_combos.complaint_type == "GENERAL CONSTRUCTION"]
    assert gc.n.max() / gc.n.sum() < 0.5
    successors = {"MOLD": "UNSANITARY CONDITION", "FLOOR": "FLOORING/STAIRS", "DOOR": "DOOR/WINDOW",
                  "CABINET": "GENERAL", "FIRE ESCAPE": "SAFETY"}
    after = cluster_combos[(cluster_combos.agency == "HPD") & (cluster_combos["last"] > "2014-03-11")]  # still in use
    for desc, new in successors.items():
        assert desc in set(after.loc[after.complaint_type == new, "descriptor"]), (desc, new)
    out = taxonomy.classify(["GENERAL CONSTRUCTION", *successors.values()]).set_index("complaint_type")
    assert out.canonical_type.is_unique
    assert set(out.category) == {"Housing conditions"}


@pytest.mark.parametrize("old, new", [
    ("Dead Tree", "Dead/Dying Tree"),                         # DPR: descriptor Dead/Dying Tree -> 2016-02-12
    ("Derelict Vehicle", "Abandoned Vehicle"),                # NYPD: both 'With License Plate'
    ("Traffic/Illegal Parking", "Illegal Parking"),           # NYPD split 2010-06-03; parking descriptors
    ("Animal Abuse", "Animal-Abuse"),                         # NYPD: same six descriptors
    ("Homeless Encampment", "Encampment"),                    # NYPD -> NYPD/DHS, 2020
    ("Missed Collection (All Materials)", "Missed Collection"),  # DSNY 2021-09-21 re-coding
    ("Sweeping/Missed-Inadequate", "Street Sweeping Complaint"),
    ("Litter Basket / Request", "Litter Basket Request"),
    ("Overflowing Litter Baskets", "Litter Basket Complaint"),
    ("Vacant Lot", "Lot Condition"),
    ("Derelict Bicycle", "Abandoned Bike"),
    ("Snow", "Snow or Ice"),
])
def test_other_renames_are_bridged(old, new, cluster_combos):
    old_first, old_last, _ = _span(cluster_combos, old)
    new_first, new_last, _ = _span(cluster_combos, new)
    assert old_first < new_first and old_last < new_last  # one name took over from the other
    out = taxonomy.classify([old, new])
    assert out.canonical_type.nunique() == 1 and out.category.nunique() == 1


@pytest.mark.parametrize("a, b", [
    ("PLUMBING", "Plumbing"),          # HPD apartment repairs vs DOB code enforcement
    ("ELEVATOR", "Elevator"),          # HPD 'MAINTENANCE' vs DOB defective/uninspected devices
    ("CONSTRUCTION", "Construction"),  # HPD roof/pavement vs DOB inspection appointments (Initial - CO, ...)
    ("ELECTRIC", "Electrical"),        # HPD lighting/outlets vs DOB defective wiring
    ("WATER LEAK", "Water Leak"),      # HPD apartment leaks vs one DOT 'General Maintenance' row
    ("Lead", "LEAD"),                  # DEP lead test-kit requests vs one DOHMH lead row
    ("GENERAL CONSTRUCTION", "General Construction/Plumbing"),  # HPD repairs vs DOB permits/structure
])
def test_lookalikes_from_different_agencies_stay_distinct(a, b, cluster_combos):
    """Same or similar name, different agency, no shared descriptor: different work, so different types."""
    ag_a = cluster_combos[cluster_combos.complaint_type == a].groupby("agency").n.sum().idxmax()
    ag_b = cluster_combos[cluster_combos.complaint_type == b].groupby("agency").n.sum().idxmax()
    assert ag_a != ag_b
    assert not _descriptors(cluster_combos, a) & _descriptors(cluster_combos, b)
    out = taxonomy.classify([a, b]).set_index("complaint_type")
    assert out.loc[a, "canonical_type"] != out.loc[b, "canonical_type"]


@pytest.mark.parametrize("a, b", [("Taxi Complaint", "Taxi Compliment"),
                                  ("Dispatched Taxi Complaint", "Dispatched Taxi Compliment"),
                                  ("Lost Property", "Found Property"),
                                  ("Taxi Report", "Taxi Complaint")])
def test_same_agency_lookalikes_with_different_meaning_stay_distinct(a, b):
    out = taxonomy.classify([a, b]).set_index("complaint_type")
    assert out.loc[a, "canonical_type"] != out.loc[b, "canonical_type"]


def test_dfta_housing_type_is_not_in_the_hpd_housing_family(cluster_combos):
    """'Housing - Low Income Senior' reads like the HPD 'Housing - ...' family, but it is DFTA information."""
    assert set(cluster_combos.loc[cluster_combos.complaint_type == "Housing - Low Income Senior", "agency"]) == {"DFTA"}
    row = _one("Housing - Low Income Senior")
    assert not row.canonical_type.startswith("Housing - ")
    assert row.category != "Housing conditions" and row.is_info_request


def test_normalized_collisions_in_the_full_data_are_merged_or_split_by_agency(tmap, cluster_combos):
    """Spellings that normalize() joins either share a canonical type or belong to different agencies."""
    main = cluster_combos.groupby(["complaint_type", "agency"]).n.sum().reset_index()
    main = main.sort_values("n").groupby("complaint_type").agency.last()
    m = tmap.assign(key=tmap.complaint_type.map(taxonomy.normalize), agency=tmap.complaint_type.map(main))
    for key, d in m.groupby("key"):
        if len(d) > 1 and d.canonical_type.nunique() > 1:
            assert d.agency.is_unique, (key, d[["complaint_type", "agency", "canonical_type"]].to_string())
            assert (d.notes != "").all(), key


def test_web_form_probes_are_invalid_entries(cluster_combos):
    """On 2019-06-07 the DVS web form received path-traversal, SQL and command-injection strings as complaint
    types (all with descriptor 'Trauma Counseling'). They are not complaints."""
    probes = cluster_combos[(cluster_combos.agency == "DVS") & (cluster_combos.descriptor == "Trauma Counseling")
                            & (cluster_combos["first"] == "2019-06-07")]
    raw = sorted(set(probes.complaint_type) - {"Forms"})  # 'Forms' is a real multi-agency type
    assert len(raw) >= 100
    out = taxonomy.classify(raw)
    assert set(out.category) == {"Invalid entries"} and set(out.origin) == {"other"}
    assert not out.is_info_request.any()
    assert _one("ZTESTINT").category == "Invalid entries"
    assert _one("Forms").category == "Information requests"


# ---------------------------------------------------------------- q03: renames over time (synthetic data)

LO, HI = pd.Timestamp("2010-01-01"), pd.Timestamp("2021-11-20")


def _periods(rows):
    t = pd.DataFrame(rows, columns=["complaint_type", "main_agency", "canonical_type", "n", "first", "last"])
    for c in ["first", "last"]:
        t[c] = pd.to_datetime(t[c], format="ISO8601")
    return t


def test_rename_handoffs_find_one_name_replacing_another():
    from nyc311 import q03_complaint_types as q
    t = _periods([
        ("HEATING", "HPD", "Heat", 900, "2010-01-01", "2014-03-10"),
        ("HEAT/HOT WATER", "HPD", "Heat", 1500, "2014-03-11", "2021-11-19"),
        ("Derelict Vehicle", "NYPD", "Derelict", 200, "2010-01-01 00:08", "2019-06-21"),
        ("Derelict Vehicles", "DSNY", "Derelict", 300, "2010-01-01 00:16", "2021-11-20"),  # alongside: no handoff
        ("Abandoned Vehicle", "NYPD", "Derelict", 100, "2018-06-15", "2021-11-20"),
        ("Noise", "DEP", "Noise", 50, "2010-01-01", "2021-11-20"),
        # siblings grouped under one canonical type are not renames: different agency ...
        ("Forms", "ACS", "Contact", 1261, "2011-09-26", "2019-06-27"),
        ("Question", "DCAS", "Contact", 2162, "2017-08-18", "2021-06-27"),
        # ... the later one stops right after the earlier one ...
        ("Advocate-Personal Exemptions", "DOF", "Advocate", 752, "2015-10-26", "2019-06-05"),
        ("Advocate-Prop Refunds/Credits", "DOF", "Advocate", 485, "2016-01-05", "2019-06-07"),
        # ... or one side is too small to date
        ("Employee Behavior", "DSNY", "Worker", 7, "2019-06-11", "2020-09-17"),
        ("Sanitation Worker or Vehicle Complaint", "DSNY", "Worker", 298, "2021-09-21", "2021-11-19"),
    ])
    h = q.rename_handoffs(t, HI).set_index("old_type")
    assert set(zip(h.index, h.new_type)) == {("HEATING", "HEAT/HOT WATER"), ("Derelict Vehicle", "Abandoned Vehicle")}
    assert h.loc["HEATING", "overlap_days"] == -1  # one day apart, no overlap
    assert h.loc["Derelict Vehicle", "overlap_days"] == 371
    assert h.loc["HEATING", "old_n"] == 900 and h.loc["HEATING", "new_n"] == 1500


def test_reorganizations_group_types_that_end_and_start_together():
    from nyc311 import q03_complaint_types as q
    t = _periods([
        ("HEATING", "HPD", "Heat", 900, "2010-01-01", "2014-03-10"),
        ("NONCONST", "HPD", "Unsan", 260, "2010-01-01", "2014-03-10"),
        ("GENERAL CONSTRUCTION", "HPD", "GC", 500, "2010-01-01", "2014-03-06"),
        ("HEAT/HOT WATER", "HPD", "Heat", 1500, "2014-03-11", "2021-11-19"),
        ("DOOR/WINDOW", "HPD", "DW", 250, "2014-03-11", "2021-11-19"),
        ("PLUMBING", "HPD", "Plumb", 800, "2010-01-01", "2021-11-19"),  # runs throughout
        ("Blip", "HPD", "Blip", 1, "2014-03-09", "2014-03-09"),          # one row: too small to date
        ("Dead Tree", "DPR", "Dead", 57, "2010-01-01", "2016-02-11"),   # one-for-one rename: too few types
        ("Dead/Dying Tree", "DPR", "Dead", 60, "2016-02-12", "2021-11-19"),
        ("Old A", "DOF", "A", 5, "2010-01-01", "2019-06-21"),           # types that stop with nothing new
        ("Old B", "DOF", "B", 5, "2010-01-01", "2019-06-20"),
        ("Old C", "DOF", "C", 5, "2010-01-01", "2019-06-19"),
    ])
    r = q.reorganizations(t, LO, HI)
    assert len(r) == 1
    row = r.iloc[0]
    assert row.agency == "HPD" and row.n_ended == 3 and row.n_started == 2
    assert (row.start, row.end) == ("2014-03-06", "2014-03-11")
    assert row.rows_ended == 1660
    assert "GENERAL CONSTRUCTION" in row.ended and "DOOR/WINDOW" in row.started and "PLUMBING" not in row.started


def test_descriptor_successors_show_where_a_split_type_went():
    from nyc311 import q03_complaint_types as q
    combos = pd.DataFrame([
        ("GENERAL CONSTRUCTION", "HPD", "MOLD", 40, "2010-01-01", "2014-03-03"),
        ("GENERAL CONSTRUCTION", "HPD", "DOORS", 30, "2010-01-01", "2014-03-02"),
        ("GENERAL CONSTRUCTION", "HPD", "WINDOWS", 20, "2010-01-01", "2014-03-06"),
        ("GENERAL CONSTRUCTION", "HPD", "FIRE-ESCAPE", 10, "2010-01-01", "2014-02-27"),
        ("UNSANITARY CONDITION", "HPD", "MOLD", 180, "2014-03-11", "2021-11-19"),
        ("DOOR/WINDOW", "HPD", "DOOR", 120, "2014-03-11", "2021-11-19"),
        ("SAFETY", "HPD", "FIRE ESCAPE", 13, "2014-03-11", "2021-11-19"),
        ("GENERAL", "HPD", "MOLD", 5, "2010-10-21", "2014-03-10"),   # ended too: not a successor
        ("SIBLING", "HPD", "DOORS", 500, "2010-01-01", "2021-11-19"),  # ran alongside from the start: not a successor
        ("Mold", "DOHMH", "MOLD", 99, "2010-01-01", "2021-11-19"),  # another agency: not a successor
    ], columns=["complaint_type", "agency", "descriptor", "n", "first", "last"])
    for c in ["first", "last"]:
        combos[c] = pd.to_datetime(combos[c])
    flows = q.descriptor_successors(combos, "GENERAL CONSTRUCTION")
    assert flows == {"UNSANITARY CONDITION": 40, "DOOR/WINDOW": 30, "SAFETY": 10, None: 20}


def test_descriptor_successors_prefer_the_same_canonical_type():
    """PAINT - PLASTER's WINDOW/FRAME rows continue under PAINT/PLASTER (its renamed self), not under the bigger
    DOOR/WINDOW type that also has a WINDOW FRAME descriptor."""
    from nyc311 import q03_complaint_types as q
    combos = pd.DataFrame([
        ("PAINT - PLASTER", "HPD", "WINDOW/FRAME", 5, "2010-01-04", "2014-02-27"),
        ("PAINT/PLASTER", "HPD", "WINDOW/FRAME", 6, "2013-09-24", "2021-11-19"),
        ("DOOR/WINDOW", "HPD", "WINDOW FRAME", 93, "2014-03-11", "2021-11-19"),
    ], columns=["complaint_type", "agency", "descriptor", "n", "first", "last"])
    for c in ["first", "last"]:
        combos[c] = pd.to_datetime(combos[c])
    assert q.descriptor_successors(combos, "PAINT - PLASTER") == {"DOOR/WINDOW": 5}
    canon = {"PAINT - PLASTER": "Paint", "PAINT/PLASTER": "Paint", "DOOR/WINDOW": "Door"}
    assert q.descriptor_successors(combos, "PAINT - PLASTER", canon) == {"PAINT/PLASTER": 5}


def test_renames_are_not_looked_for_in_a_short_window():
    from nyc311 import q03_complaint_types as q
    t = _periods([("A", "X", "A", 5, "2010-01-01", "2010-01-02"), ("B", "X", "A", 5, "2010-01-20", "2010-02-03")])
    combos = pd.DataFrame(columns=["complaint_type", "agency", "descriptor", "n", "first", "last"])
    out = q.renames_over_time(t, combos, pd.Timestamp("2010-01-01"), pd.Timestamp("2010-02-03"))
    assert out["too_short"]
    assert out["handoffs"].empty and out["reorgs"].empty and out["legacy"].empty


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
    # the report stage renders a markdown subset: one '# Q3' heading first, no raw HTML, '*' for italics
    assert md.splitlines()[0].startswith("# Q3") and sum(line.startswith("# ") for line in md.splitlines()) == 1
    assert "<" not in md and not any(line.startswith("_") for line in md.splitlines())
    # the sample covers about a month: renames are not looked for, and the findings say so
    assert "Renamed over time" in md and "too short" in md
    for name in ["q03_renames.csv", "q03_reorganizations.csv", "q03_legacy_types.csv"]:
        assert (tmp_path / "agg" / name).exists(), name

    printed = capsys.readouterr().out
    assert "BEGIN q03_complaint_types" in printed
    report = (tmp_path / "reports" / "q03_complaint_types.txt").read_text().splitlines()
    assert len(report) <= 152
