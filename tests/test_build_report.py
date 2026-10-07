import base64

import pytest

from nyc311 import build_report

PNG = base64.b64decode(  # 1x1 transparent PNG
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


def test_report_is_the_last_stage():
    from nyc311.config import STAGE_MODULES, STAGES
    assert STAGES[-1] == "report"
    assert STAGE_MODULES["report"] == "nyc311.build_report"


def test_markdown_blocks_become_html(tmp_path):
    out = build_report.md_to_html(
        "## Findings\n\n- one **bold**\n  - nested *it*\n- two `code`\n\n1. first\n2. second\n\n"
        "A [link](https://example.com) here.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", tmp_path)
    assert "<h4>Findings</h4>" in out
    assert out.count("<ul>") == 2 and "<strong>bold</strong>" in out and "<em>it</em>" in out
    assert "<code>code</code>" in out and "<ol>" in out
    assert '<a href="https://example.com">link</a>' in out
    assert "<table>" in out and '<th class="num">a</th>' in out and '<td class="num">2</td>' in out


def test_markdown_escapes_html_and_keeps_snake_case(tmp_path):
    out = build_report.md_to_html("- <script>x</script> and out_of_file_order and 5 * 3 * 2\n", tmp_path)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "out_of_file_order" in out
    assert "<em>" not in out


def test_figures_are_embedded_and_missing_ones_noted(tmp_path):
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "q05_top.png").write_bytes(PNG)
    out = build_report.md_to_html("![Top ten](figs/q05_top.png)\n\n![Gone](figs/q05_missing.png)\n", tmp_path)
    assert 'src="data:image/png;base64,' in out and "<figcaption>Top ten</figcaption>" in out
    assert "Figure not found" in out


@pytest.fixture()
def built(tmp_path):
    out, prose = tmp_path / "outputs", tmp_path / "prose"
    for d in (out / "answers", out / "figs", out / "dashboard", out / "reports", prose):
        d.mkdir(parents=True)
    (out / "answers" / "q11.md").write_text("# Q11 Executive summary\n\n- Finding one SUMMARYMARK\n")
    (out / "answers" / "q05.md").write_text("# Q5 Top complaint types\n\n- Heat leads. FINDINGMARK\n\n"
                                            "![Top ten types](figs/q05_top10.png)\n")
    (out / "figs" / "q05_top10.png").write_bytes(PNG)
    (out / "figs" / "q05_extra.png").write_bytes(PNG)
    (out / "dashboard" / "q08_dashboard.html").write_text("<html><body><div id='map'>MAPMARK</div></body></html>")
    (out / "reports" / "03_clean.txt").write_text("===== BEGIN 03_clean =====\nrows: 10\n===== END 03_clean =====")
    (prose / "q05.md").write_text("My own words PROSEMARK.\n")
    path = build_report.build(out, prose, meta={"title_note": "test data"})
    return path.read_text()


def test_summary_comes_first_then_questions_in_order(built):
    assert built.index("SUMMARYMARK") < built.index('id="q01"') < built.index('id="q05"') < built.index('id="q10"')


def test_prose_comes_before_computed_findings(built):
    assert built.index("PROSEMARK") < built.index("FINDINGMARK")


def test_report_is_self_contained(built):
    assert "<img" in built
    assert all(chunk.startswith('"data:image/png;base64,') for chunk in built.split("<img src=")[1:])
    assert built.count("data:image/png;base64,") == 2  # referenced figure plus the unreferenced q05_extra.png


def test_dashboard_is_embedded_as_a_live_map(built):
    assert "<iframe" in built and "srcdoc=" in built and "MAPMARK" in built


def test_missing_stages_say_so_and_brief_questions_appear(built):
    q01 = built[built.index('id="q01"'):built.index('id="q02"')]
    assert "did not produce results" in q01
    assert "top 10 complaint types" in built.lower()
    assert "rows: 10" in built


def test_emphasis_can_wrap_inline_code(tmp_path):
    out = build_report.md_to_html("*Drafted by `q08_dashboard.py`*\n", tmp_path)
    assert "<em>Drafted by <code>q08_dashboard.py</code></em>" in out
    assert "*" not in out


def test_numeric_table_columns_are_right_aligned(tmp_path):
    out = build_report.md_to_html(
        "| District | Complaints | Share | vs. typical | Median | Applies to |\n|---|---|---|---|---|---|\n"
        "| Bronx 5 | 550,716 | 9.8% | \u22128% | 47.6 hours | 2 of 478 types |\n"
        "| Bronx 7 | **601,992** | \u2014 | +4% | 3.0 days | all types |\n", tmp_path)
    assert '<th>District</th>' in out and '<td>Bronx 5</td>' in out
    for header in ("Complaints", "Share", "vs. typical", "Median"):
        assert f'<th class="num">{header}</th>' in out
    assert '<td class="num"><strong>601,992</strong></td>' in out and '<th>Applies to</th>' in out
    assert build_report.numeric_columns([["2010-01-01"], ["1,234"]]) == [False]
    assert build_report.numeric_columns([["n/a"], [""]]) == [False]


def test_figures_are_numbered_in_reading_order(built):
    assert built.count("<figure") == 3  # two q05 charts and the q08 dashboard
    first, second, third = (built.index(f'id="figure-{n}"') for n in (1, 2, 3))
    assert first < second < third
    assert '<span class="fig-num">Figure 1.</span> Top ten types</figcaption>' in built
    assert '<span class="fig-num">Figure 2.</span> Extra</figcaption>' in built
    assert '<figure id="figure-3" class="dashboard">' in built
    assert "Figure 4." not in built


def test_number_figures_counts_and_leaves_other_markup_alone():
    body, n = build_report.number_figures("<p>x</p><figure><img src=a alt=b><figcaption>One</figcaption></figure>"
                                          '<figure class="dashboard"><iframe></iframe><figcaption>Two</figcaption>'
                                          "</figure>")
    assert n == 2 and body.startswith("<p>x</p>")
    assert '<figure id="figure-2" class="dashboard">' in body
    assert '<figcaption><span class="fig-num">Figure 2.</span> Two</figcaption>' in body


def test_contents_lists_summary_parts_and_questions_in_order(built):
    nav = built[built.index('<nav class="toc"'):built.index("</nav>")]
    anchors = ["#summary", "#part-1", "#q01", "#q04", "#part-2", "#q05", "#q08", "#part-3", "#q09", "#q10",
               "#appendix"]
    positions = [nav.index(f'href="{a}"') for a in anchors]
    assert positions == sorted(positions)
    assert "Q11" in nav and "Executive summary" in nav and "Part 2: Data visualization and insights" in nav
    for n in (1, 2, 3):
        assert f'<h2 class="part" id="part-{n}">' in built
    assert built.index('<nav class="toc"') < built.index('<main id="report">')


def test_prose_and_computed_findings_are_separate_blocks(built):
    q05 = built[built.index('id="q05"'):built.index('id="q06"')]
    assert q05.index('<div class="prose">') < q05.index("PROSEMARK") < q05.index('<div class="findings">')
    assert q05.index('<div class="findings">') < q05.index("What the data shows") < q05.index("FINDINGMARK")
    q01 = built[built.index('id="q01"'):built.index('id="q02"')]
    assert '<div class="prose">' not in q01 and "What the data shows" in q01


def test_provenance_line_becomes_a_source_note(tmp_path):
    out, prose = tmp_path / "outputs", tmp_path / "prose"
    (out / "answers").mkdir(parents=True)
    prose.mkdir()
    (out / "answers" / "q05.md").write_text("# Q5\n\n*Drafted by `q05.py` from 10 rows.*\n\n- *a* and *b*\n")
    built = build_report.build(out, prose).read_text()
    assert '<p class="source"><em>Drafted by <code>q05.py</code> from 10 rows.</em></p>' in built
    assert "<li><em>a</em> and <em>b</em></li>" in built


def test_page_needs_no_network_and_has_print_rules(built):
    page = built.split('srcdoc="')[0] + built.split('srcdoc="')[1].split('"', 1)[1]  # drop the embedded dashboard
    for remote in ("http://", "https://", "<link", "@import", "url(", "<script src"):
        assert remote not in page
    assert "@media print" in built and ".toc,.skip{display:none}" in built and ".part{break-before:page" in built


def test_png_display_size_uses_saved_dpi(tmp_path):
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", 600, 300, 8, 0, 0, 0, 0)
    phys = struct.pack(">IIB", 11811, 11811, 1)  # 300 dpi
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"pHYs", phys) + chunk(b"IEND", b"")
    assert build_report._display_size(png) == (240, 120)
    assert build_report._display_size(PNG) is None  # no dpi recorded: let the browser use the pixel size
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "c.png").write_bytes(png)
    out = build_report.md_to_html("![C](figs/c.png)\n", tmp_path)
    assert 'width="240" height="120"' in out


def test_wrapped_list_lines_stay_in_their_item(tmp_path):
    out = build_report.md_to_html("- first item wraps\n  onto a second line\n- second **bold\nacross lines**\n\n"
                                  "After the list.\n", tmp_path)
    assert "<li>first item wraps onto a second line</li>" in out
    assert "<strong>bold across lines</strong>" in out and "<p>After the list.</p>" in out
    out = build_report.md_to_html("- item\n| a | b |\n|---|---|\n| 1 | 2 |\n", tmp_path)
    assert "<li>item</li>" in out and "<table>" in out
