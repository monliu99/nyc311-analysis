"""Sanity checks on the external datasets written by nyc311/fetch_external.py."""

import json
from pathlib import Path

import pandas as pd
import pytest

EXT = Path(__file__).resolve().parents[1] / "data" / "external"

STANDARD_CDS = sorted(
    [100 + n for n in range(1, 13)]
    + [200 + n for n in range(1, 13)]
    + [300 + n for n in range(1, 19)]
    + [400 + n for n in range(1, 15)]
    + [500 + n for n in range(1, 4)]
)


@pytest.fixture(scope="module")
def weather():
    return pd.read_csv(EXT / "weather_central_park_daily.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def cd_geo():
    return json.loads((EXT / "community_districts.geojson").read_text())


@pytest.fixture(scope="module")
def demo():
    return pd.read_csv(EXT / "cd_demographics.csv", dtype={"puma_2010": str})


# ------------------------------------------------------------------ weather
def test_weather_columns(weather):
    assert list(weather.columns) == ["date", "tmax_f", "tmin_f", "prcp_in", "snow_in", "snwd_in"]


def test_weather_dates_unique_continuous_from_2010(weather):
    d = weather["date"]
    assert d.is_unique
    assert d.iloc[0] == pd.Timestamp("2010-01-01")
    assert d.is_monotonic_increasing
    assert (d.diff().dropna() == pd.Timedelta(days=1)).all()
    assert d.iloc[-1] >= pd.Timestamp("2025-12-31")


def test_weather_plausible_ranges(weather):
    assert weather["tmax_f"].dropna().between(-10, 110).all()
    assert weather["tmin_f"].dropna().between(-20, 100).all()
    assert (weather["tmin_f"] <= weather["tmax_f"]).dropna().all()
    for col in ["prcp_in", "snow_in", "snwd_in"]:
        assert (weather[col].dropna() >= 0).all()
    assert weather["prcp_in"].max() < 15


def test_weather_mostly_complete(weather):
    # Gaps are left as NaN, but core fields should be nearly complete.
    assert weather[["tmax_f", "tmin_f", "prcp_in"]].notna().mean().min() > 0.99


# ------------------------------------------------------------------ CD polygons
def test_geojson_has_all_standard_cds(cd_geo):
    ids = [f["properties"]["boro_cd"] for f in cd_geo["features"]]
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) == len(set(ids))
    assert set(STANDARD_CDS) <= set(ids)


def test_geojson_is_jia_flag(cd_geo):
    for f in cd_geo["features"]:
        p = f["properties"]
        assert isinstance(p["is_jia"], bool)
        assert p["is_jia"] == (p["boro_cd"] % 100 > 18)
    standard = [f for f in cd_geo["features"] if not f["properties"]["is_jia"]]
    assert sorted(f["properties"]["boro_cd"] for f in standard) == STANDARD_CDS


def test_geojson_wgs84_and_size(cd_geo):
    for f in cd_geo["features"]:
        g = f["geometry"]
        assert g["type"] in ("Polygon", "MultiPolygon")
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for ring in poly:
                assert ring[0] == ring[-1] and len(ring) >= 4
                for lon, lat in ring:
                    assert -74.3 < lon < -73.6 and 40.4 < lat < 41.0
    assert (EXT / "community_districts.geojson").stat().st_size < 5_000_000


# ------------------------------------------------------------------ demographics
def test_demographics_exact_cds(demo):
    assert len(demo) == 59
    assert sorted(demo["boro_cd"].tolist()) == STANDARD_CDS
    assert (demo["boro_cd"] // 100 * 100 + demo["cd_number"] == demo["boro_cd"]).all()


def test_demographics_population(demo):
    total_2010 = demo["population_2010"].sum()
    assert abs(total_2010 - 8_170_000) / 8_170_000 < 0.02
    assert (demo["population_2010"] > 0).all()
    total_2020 = demo["population_2020"].sum()
    assert abs(total_2020 - 8_800_000) / 8_800_000 < 0.02
    assert (demo["population_2020"] > 0).all()


def test_demographics_income(demo):
    assert (demo["median_hh_income"] > 0).all()
    assert demo["income_vintage"].notna().all()
    assert demo["puma_2010"].nunique() == 55
    shared = demo[demo["puma_shared_with"].notna()]
    assert sorted(shared["boro_cd"]) == [101, 102, 104, 105, 201, 202, 203, 206]
    # CDs sharing a PUMA carry the same income value.
    assert (shared.groupby("puma_2010")["median_hh_income"].nunique() == 1).all()
