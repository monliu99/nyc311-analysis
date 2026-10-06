"""Q8 neighborhood dashboard (folium web map), on the 20-file sample (fixture `cleaned_cfg`) plus pure functions."""
import html
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from nyc311 import q08_dashboard as q8
from nyc311.clean import clean_glob

COMPLAINTS = "not is_test and not dup_key and not coalesce(is_info_request, false)"


@pytest.fixture(scope="module")
def res(cleaned_cfg):
    return q8.run(cleaned_cfg)


@pytest.fixture(scope="module")
def html_text(res, cleaned_cfg):
    return (cleaned_cfg.out_dir / q8.DASHBOARD).read_text()


def _con_src(cfg):
    return cfg.connect(), clean_glob(cfg)


def _one(cfg, sql):
    con = cfg.connect()  # keep a reference: duckdb 1.4 closes an unreferenced connection before the fetch
    return con.sql(sql.format(src=clean_glob(cfg))).fetchone()[0]


def _complete_days_sql():
    return """(select created::date from {src} where not is_test and not dup_key and created is not null
               group by 1 having bool_or(not out_of_file_order) and not bool_or(possibly_incomplete_day))"""


# ------------------------------------------------------------------ pure functions
def test_simplify_ring_drops_vertices_but_keeps_shape_and_closure():
    t = np.linspace(0, 2 * np.pi, 2001)
    ring = np.column_stack([-73.9 + 0.05 * np.cos(t), 40.7 + 0.05 * np.sin(t)])
    ring[-1] = ring[0]
    out = np.asarray(q8.simplify_ring(ring.tolist(), 0.0001))
    assert 4 <= len(out) < len(ring) / 5
    assert out[0].tolist() == out[-1].tolist()
    # every original vertex stays within the tolerance of the simplified outline (checked at the vertices kept)
    assert np.abs(np.hypot(out[:, 0] + 73.9, out[:, 1] - 40.7) - 0.05).max() < 0.0001 + 1e-9
    tri = [[0, 0], [1, 0], [0, 1], [0, 0]]
    assert q8.simplify_ring(tri, 0.5) == tri


def test_simplify_geojson_counts_vertices_and_keeps_every_feature():
    gj = q8.load_districts()
    simple, before, after = q8.simplify_geojson(gj, q8.SIMPLIFY_DEG)
    assert len(simple["features"]) == len(gj["features"])
    assert after < before / 3
    assert {f["properties"]["boro_cd"] for f in simple["features"]} == {f["properties"]["boro_cd"] for f in gj["features"]}


def test_quantile_classes_are_monotone_and_fill_every_class():
    rates = pd.Series([5.0, 50, 10, 40, 20, 30, 25, 15, 35, 45])
    cls, edges = q8.classify(rates, 5)
    assert sorted(cls.unique()) == list(range(5))
    order = rates.sort_values().index
    assert cls[order].is_monotonic_increasing
    assert edges[0] == rates.min() and edges[-1] == rates.max() and edges == sorted(edges)


def test_person_years_use_2010_population_before_the_switch_year():
    demo = pd.DataFrame({"boro_cd": [101, 102], "population_2010": [1000, 500], "population_2020": [2000, 500]})
    days = pd.Series({q8.POP_2020_FROM_YEAR - 1: 365.25, q8.POP_2020_FROM_YEAR: 730.5})
    py = q8.person_years(demo, days)
    assert py.loc[(101, q8.POP_2020_FROM_YEAR - 1)] == pytest.approx(1000)
    assert py.loc[(101, q8.POP_2020_FROM_YEAR)] == pytest.approx(4000)
    assert py.groupby(level=0).sum().loc[102] == pytest.approx(1500)


def test_sparkline_has_one_bar_per_year_and_none_for_a_single_year():
    years = pd.DataFrame({"year": [2010, 2011, 2012], "n": [10, 30, 20], "rate": [1.0, 3.0, 2.0]})
    svg = q8.sparkline_svg(years)
    assert svg.startswith("<svg") and svg.count("<rect") == 3 and "2010" in svg and "2012" in svg
    assert q8.sparkline_svg(years.head(1)) == ""


def test_popup_escapes_text_and_suppresses_thin_medians():
    row = pd.Series({"label": "Manhattan 3: <Lower East Side>", "n": 50, "rate": 12.34, "rank": 1, "n_districts": 59,
                     "n_closed": 10, "median_days": 2.5, "adj_pct": 40.0, "pct_unresolved": 1.5, "low_n": True,
                     "population": 100000, "pop_label": "2010 Census"})
    top = pd.DataFrame({"canonical_type": ["Noise & stuff"], "n": [30]})
    city = {"rate": 10.0, "median_days": 3.0, "pct_unresolved": 2.0}
    html = q8.district_popup(row, top, pd.DataFrame(columns=["year", "n", "rate"]), city)
    assert "&lt;Lower East Side&gt;" in html and "Noise &amp; stuff" in html
    assert "50" in html and "12.3" in html and "fewer than" in html
    assert "40%" not in html  # adjusted gap hidden below MIN_N closed complaints


# ------------------------------------------------------------------ the sample run
def test_waterfall_reconciles_with_the_raw_row_count(res, cleaned_cfg):
    w = res["waterfall"]
    assert (w.rows >= 0).all()
    steps = w.step.tolist()
    k = steps.index("complaints on complete days")
    assert w.rows.iloc[0] == _one(cleaned_cfg, "select count(*) from {src}")
    assert w.rows.iloc[k] == w.rows.iloc[0] - w.rows.iloc[1:k].sum()
    assert w.rows.iloc[-1] == w.rows.iloc[k] - w.rows.iloc[k + 1:-1].sum()
    assert steps[-1] == "mapped to a community district" and w.rows.iloc[-1] == res["districts"].n.sum()
    named = dict(zip(w.step, w.rows))
    assert named["- is_test"] == _one(cleaned_cfg, "select count_if(is_test) from {src}")
    assert named["- possibly_incomplete_day"] == _one(
        cleaned_cfg, f"select count(*) from {{src}} where {COMPLAINTS} and created is not null and possibly_incomplete_day")


def test_district_counts_match_an_independent_query(res, cleaned_cfg):
    d = res["districts"].set_index("boro_cd")
    assert len(d) == 59
    sql = (f"select boro_cd, count(*) n from {{src}} where {COMPLAINTS} and boro_cd is not null "
           f"and created::date in {_complete_days_sql()} group by 1")
    con, src = _con_src(cleaned_cfg)
    ind = con.sql(sql.format(src=src)).df().set_index("boro_cd").n
    assert (d.n.reindex(ind.index) == ind).all()
    assert d.n.sum() == ind.sum()


def test_rates_are_complaints_over_resident_years(res):
    d = res["districts"]
    assert np.allclose(d.rate, 1000 * d.n / d.person_years)
    assert (d.rate >= 0).all() and (d.person_years > 0).all()
    assert d["rank"].tolist() == d.rate.rank(ascending=False, method="first").astype(int).tolist()
    assert res["period"][1].year < q8.POP_2020_FROM_YEAR  # the sample is all before the population switch
    assert np.allclose(d.person_years, d.population_2010 * res["n_days"] / q8.DAYS_PER_YEAR)
    assert 0 <= d.pct_unresolved.min() and d.pct_unresolved.max() <= 100
    city = res["city"]
    assert city["rate"] == pytest.approx(1000 * d.n.sum() / d.person_years.sum())


def test_mix_adjusted_gap_matches_an_independent_pandas_computation(res, cleaned_cfg):
    con, src = _con_src(cleaned_cfg)
    r = con.sql(f"""select boro_cd, coalesce(canonical_type, '') ct, year(created) yr, duration_h from {src}
                    where {COMPLAINTS} and duration_h is not null and not (closed_at_creation and not date_only_created)
                    and created::date in {_complete_days_sql().format(src=src)}""").df()
    r["y"] = np.log(r.duration_h + 1)
    r["e"] = r.y - r.groupby(["ct", "yr"]).y.transform("mean")  # benchmark cells include rows with no district
    g = r.dropna(subset=["boro_cd"]).groupby("boro_cd")
    adj = 100 * (np.exp(g.e.mean()) - 1)
    med = g.duration_h.median() / 24
    d = res["districts"].set_index("boro_cd")
    assert np.allclose(d.adj_pct.reindex(adj.index), adj, atol=1e-6)
    assert np.allclose(d.median_days.reindex(med.index), med, atol=1e-9)
    assert (d.n_closed.reindex(adj.index) == g.size()).all()


def test_themes_are_the_largest_real_categories(res, cleaned_cfg):
    con, src = _con_src(cleaned_cfg)
    c = con.sql(f"""select category, count(*) n from {src} where {COMPLAINTS} and boro_cd is not null
                    and category is not null and created::date in {_complete_days_sql().format(src=src)}
                    group by 1 order by n desc, category""").df()
    c = c[~c.category.isin(q8.NOT_THEMES)]
    assert res["themes"] == c.category.head(q8.N_THEMES).tolist()
    d = res["districts"]
    for i, t in enumerate(res["themes"]):
        col = q8.theme_col(i)
        assert np.allclose(d[f"{col}_rate"], 1000 * d[f"{col}_n"] / d.person_years)
        assert d[f"{col}_n"].sum() == c.set_index("category").n[t]


def test_point_layer_is_a_capped_sample_of_the_latest_complete_days(res, cleaned_cfg):
    p, info = res["points"], res["point_info"]
    assert len(p) == min(q8.POINT_CAP, info["eligible"]) and len(p) <= q8.POINT_CAP
    assert p.unique_key.is_unique
    days = set(info["window_days"])
    assert len(days) == min(q8.WINDOW_DAYS, res["n_days"])
    assert set(pd.to_datetime(p.created).dt.date) <= days
    assert p.latitude.notna().all() and p.longitude.notna().all()
    eligible = _one(cleaned_cfg, f"""select count(*) from {{src}} where {COMPLAINTS} and latitude is not null
                    and longitude is not null and created::date in ({", ".join(f"'{d}'" for d in sorted(days))})""")
    assert info["eligible"] == eligible
    # incomplete days are never in the window, and the window ends on the last complete day
    inc = _one(cleaned_cfg, "select count(*) from {src} where possibly_incomplete_day and created::date in ("
               + ", ".join(f"'{d}'" for d in sorted(days)) + ")")
    assert inc == 0 and max(days) == res["period"][1]


def test_point_sample_is_repeatable_and_respects_a_small_cap(res, cleaned_cfg):
    con, src = _con_src(cleaned_cfg)
    a = q8.sample_points(con, src, res["point_info"]["window_days"], cap=500)
    con.execute("set threads = 1")  # the sample must not depend on the thread count
    b = q8.sample_points(con, src, res["point_info"]["window_days"], cap=500)
    assert len(a) == 500 and a.unique_key.tolist() == b.unique_key.tolist()
    c = q8.sample_points(con, src, res["point_info"]["window_days"], cap=500, seed=q8.SEED + 1)
    assert len(set(a.unique_key) & set(c.unique_key)) < 100  # another seed gives a different sample


class _Probe(HTMLParser):
    def __init__(self):
        super().__init__()
        self.map_divs, self.scripts, self.panel, self._in_script = 0, [], False, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div" and "folium-map" in (a.get("class") or ""):
            self.map_divs += 1
        if tag == "div" and a.get("id") == "q08-panel":
            self.panel = True
        self._in_script = tag == "script" and "src" not in a
        if self._in_script:
            self.scripts.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.scripts[-1] += data


def test_html_is_a_leaflet_map_with_layers_panel_and_every_district(res, html_text):
    p = _Probe()
    p.feed(html_text)
    assert p.map_divs == 1 and p.panel
    js = "\n".join(p.scripts)
    assert "L.map(" in js and "L.control.layers(" in js and "markerClusterGroup" in js
    for name in [res["layer_names"]["all"], *res["layer_names"]["themes"], res["layer_names"]["points"]]:
        assert q8.js_escape(name) in js
    d = res["districts"]
    for r in d[d.n > 0].itertuples():  # every district present in the data is named on the map
        assert q8.js_escape(html.escape(r.cd_name)) in html_text, r.cd_name
    feats = re.findall(r'"boro_cd": (\d+)', html_text)
    assert {int(x) for x in feats} >= set(d.boro_cd)


def test_popups_carry_the_computed_fields(res, html_text, cleaned_cfg):
    import html as h
    d = res["districts"].set_index("boro_cd")
    top = d.sort_values("rank").iloc[0]
    pop = res["popups"][int(top.name)]
    assert h.escape(top.label) in pop
    assert f"{int(top.n):,}" in pop and q8.fmt_rate(top.rate) in pop and f"{top.pct_unresolved:.1f}%" in pop
    assert q8.fmt_days(top.median_days) in pop and q8.fmt_pct(top.adj_pct) in pop
    con, src = _con_src(cleaned_cfg)
    ind = con.sql(f"""select canonical_type, count(*) n from {src} where {COMPLAINTS} and boro_cd = {int(top.name)}
                      and canonical_type is not null and created::date in {_complete_days_sql().format(src=src)}
                      group by 1 order by n desc, canonical_type limit {q8.TOP_TYPES}""").df()
    tt = res["top_types"]
    mine = tt[tt.boro_cd == top.name]
    assert mine.canonical_type.tolist() == ind.canonical_type.tolist() and mine.n.tolist() == ind.n.tolist()
    for r in ind.itertuples():
        assert h.escape(r.canonical_type) in pop and f"{r.n:,}" in pop
    years = res["years"][res["years"].boro_cd == top.name]
    assert all(str(y) in pop for y in years.year)
    # the popup text reaches the file (JSON-escaped inside the GeoJSON layer)
    assert q8.js_escape(pop) in html_text


def test_point_data_in_the_file_matches_the_sample(res, html_text):
    m = re.search(r"var data = (\[.*?\]);\n", html_text, re.S)
    rows = json.loads(m.group(1))
    assert len(rows) == len(res["points"])
    keys = {r[q8.POINT_FIELDS.index("unique_key")] for r in rows}
    assert keys == set(res["points"].unique_key)
    lat = [r[0] for r in rows]
    assert 40.4 < min(lat) and max(lat) < 41.0


def test_file_size_is_under_the_limit(res, cleaned_cfg):
    size = (cleaned_cfg.out_dir / q8.DASHBOARD).stat().st_size
    assert size == res["html_bytes"]
    assert size < q8.MAX_HTML_MB * 1e6


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_inline_javascript_parses(html_text, tmp_path):
    p = _Probe()
    p.feed(html_text)
    for i, s in enumerate(p.scripts):
        f = tmp_path / f"s{i}.js"
        f.write_text(s)
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:500]


def test_running_twice_gives_identical_files(res, cleaned_cfg):
    out = cleaned_cfg.out_dir
    files = [q8.DASHBOARD, "agg/q08_district_summary.csv", "agg/q08_district_years.csv", "answers/q08.md"]
    first = {f: (out / f).read_bytes() for f in files}
    q8.run(cleaned_cfg)
    for f in files:
        assert (out / f).read_bytes() == first[f], f


def test_report_block_and_findings(res, cleaned_cfg):
    out = cleaned_cfg.out_dir
    block = (out / "reports" / "q08_dashboard.txt").read_text()
    assert block.startswith("===== BEGIN q08_dashboard =====") and len(block.splitlines()) <= 152
    summary = pd.read_csv(out / "agg" / "q08_district_summary.csv")
    assert len(summary) == 59 and summary.n.sum() == res["districts"].n.sum()
    text = (out / "answers" / "q08.md").read_text()
    assert text.splitlines()[0].startswith("# Q8")
    assert f"{int(res['districts'].n.sum()):,}" in text and f"{len(res['points']):,}" in text
    top = res["districts"].sort_values("rank").iloc[0]
    assert top.cd_name in text and q8.fmt_rate(top.rate) in text
    assert "possibly_incomplete_day" in text and "is_info_request" in text
    assert not re.search(r"<[a-zA-Z/][^>]*>", text)
    assert not re.search(r"\bnan\b|\bNone\b", text)
    assert not re.search(r"\{[A-Za-z_]", text)  # no f-string placeholder left unformatted
    panel = re.search(r'<div id="q08-panel">.*?</details></div>', (out / q8.DASHBOARD).read_text(), re.S).group(0)
    assert not re.search(r"\{[A-Za-z_]", panel) and "nan" not in panel.split()


def test_short_period_is_said_in_the_findings(res, cleaned_cfg):
    text = (cleaned_cfg.out_dir / "answers" / "q08.md").read_text()
    if res["n_days"] < 365:
        assert "one year" in text or "seasonal" in text
